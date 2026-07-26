import torch, json, os, random
from .losses import total_loss

# Routes exercised by this dataset:
#   route 0 = "reflexo"  (repeticao steps)
#   route 1 = "memoria"  (everything else)
# "cortex" (2) and "abster" (3) receive NO supervision here — the dataset
# does not contain steps that exercise those routes. Task-9 evaluation must
# report route_acc only on classes {0, 1}; all four are defined in ROUTES but
# only two carry gradients during this training phase.
SUPERVISED_ROUTES = frozenset({0, 1})  # exposed so Task-9 can query this


def _batch(model, sessions, rng, B=16):
    """Amostra um PASSO de sessão preservando o dt real. O estado vem da sessão (continuidade).

    NOTE: only routes 0 ("reflexo") and 1 ("memoria") are supervised.
    Routes 2 ("cortex") and 3 ("abster") get no training signal from this
    dataset. See SUPERVISED_ROUTES.
    """
    qs, gold_f, gold_r, dts = [], [], [], []
    key2idx = {f.key: i for i, f in enumerate(model.memory.graph.facts)} if model.use_water else {}
    missing_key_count = 0
    for _ in range(B):
        s = rng.choice(sessions)
        st = rng.choice(s.steps)
        qs.append(st.query)
        dts.append([st.dt_seconds])
        # Guard: if use_water is active and the gold key is missing from the
        # graph, training toward index 0 would be silently wrong — skip it.
        if model.use_water and st.gold_fact_key and st.gold_fact_key not in key2idx:
            missing_key_count += 1
            gold_f.append(-1)   # sentinel; caller should handle or mask
        else:
            gold_f.append(key2idx.get(st.gold_fact_key, 0))
        # Only reflexo/memoria labels are valid for this dataset (see module docstring).
        gold_r.append(1 if st.kind != "repeticao" else 0)
    dev = next(model.parameters()).device
    return {
        "queries": qs,
        "dt": torch.tensor(dts, dtype=torch.float32, device=dev),
        "gold_fact_idx": torch.tensor(gold_f, device=dev),
        "gold_route": torch.tensor(gold_r, device=dev),
        "acertou": torch.ones(B, 1, device=dev),
        "_missing_key_count": missing_key_count,
    }


def train(model, sessions, steps=2000, lr=3e-4, ckpt_path="iara_core.pt",
          log_path="iara_core_log.json", seed=0, B=16, log_every=100):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    start = 0
    hist = []

    # Build the optimizer BEFORE loading checkpoint so we can restore its state.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        start = ck["step"]
        hist = ck.get("hist", [])
        # Restore optimizer state so Adam momentum/variance survive reconnects.
        # Fall back gracefully if an older checkpoint has no "opt" key.
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        print(f"resume do passo {start}")

    state = None
    total_missing_keys = 0
    for s in range(start, steps):
        b = _batch(model, sessions, rng, B)
        total_missing_keys += b.get("_missing_key_count", 0)
        out = model.step(b["queries"], b["dt"], state)
        state = out.state.detach()                             # continuidade entre passos
        loss, parts = total_loss(out, b, model)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.neuro.update(float(loss), float(b["dt"].mean()))
        if model.neuro.precisa_dormir():
            model.neuro.dormir()
            state = None                                       # sono zera o estado
        hist.append(float(loss))
        if s % log_every == 0 or s == steps - 1:
            print(f"[{s}/{steps}] loss {sum(hist[-50:])/min(50,len(hist)):.4f} {parts} dop {model.neuro.dop:.2f}")
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": s + 1,
                "hist": hist,
            }, ckpt_path)
            with open(log_path, "w") as fh:
                json.dump({"hist": hist}, fh)

    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": steps,
        "hist": hist,
    }, ckpt_path)

    if total_missing_keys:
        print(f"[train] WARNING: {total_missing_keys} samples had missing gold_fact_key (sentinel -1 used).")

    # loss_inicial: mean of first 5 entries (more stable than a single noisy step)
    loss_inicial = sum(hist[:5]) / min(5, len(hist))
    loss_final = sum(hist[-50:]) / min(50, len(hist))
    return {
        "loss_inicial": loss_inicial,
        "loss_final": loss_final,
        "hist": hist,
        "missing_key_count": total_missing_keys,
    }
