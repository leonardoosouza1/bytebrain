import torch, json, os, random
from .losses import total_loss


def _batch(model, sessions, rng, B=16):
    """Amostra um PASSO de sessão preservando o dt real. O estado vem da sessão (continuidade)."""
    qs, gold_f, gold_r, dts = [], [], [], []
    key2idx = {f.key: i for i, f in enumerate(model.memory.graph.facts)} if model.use_water else {}
    for _ in range(B):
        s = rng.choice(sessions)
        st = rng.choice(s.steps)
        qs.append(st.query)
        dts.append([st.dt_seconds])
        gold_f.append(key2idx.get(st.gold_fact_key, 0))
        gold_r.append(1 if st.kind != "repeticao" else 0)     # repetição => reflexo; senão memória
    dev = next(model.parameters()).device
    return {
        "queries": qs,
        "dt": torch.tensor(dts, dtype=torch.float32, device=dev),
        "gold_fact_idx": torch.tensor(gold_f, device=dev),
        "gold_route": torch.tensor(gold_r, device=dev),
        "acertou": torch.ones(B, 1, device=dev),
    }


def train(model, sessions, steps=2000, lr=3e-4, ckpt_path="iara_core.pt",
          log_path="iara_core_log.json", seed=0, B=16, log_every=100):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    start = 0
    hist = []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        start = ck["step"]
        hist = ck.get("hist", [])
        print(f"resume do passo {start}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    state = None
    l0 = None
    for s in range(start, steps):
        b = _batch(model, sessions, rng, B)
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
        if l0 is None:
            l0 = float(loss)
        hist.append(float(loss))
        if s % log_every == 0 or s == steps - 1:
            print(f"[{s}/{steps}] loss {sum(hist[-50:])/min(50,len(hist)):.4f} {parts} dop {model.neuro.dop:.2f}")
            torch.save({"model": model.state_dict(), "step": s + 1, "hist": hist}, ckpt_path)
            json.dump({"hist": hist}, open(log_path, "w"))
    torch.save({"model": model.state_dict(), "step": steps, "hist": hist}, ckpt_path)
    return {"loss_inicial": l0, "loss_final": sum(hist[-50:]) / min(50, len(hist)), "hist": hist}
