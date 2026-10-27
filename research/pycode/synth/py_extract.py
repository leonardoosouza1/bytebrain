"""py_extract (ledger #51): map where Python and pt-BR requests live in a Qwen2.5-Coder, extract only that MLP part.

His words (10-27): "se a gente mapear python nos pesos e tokens podemos pegar um modelo de 3B, extrair apenas o python
e chat, ele ficaria menor, provavelmente menos inteligente, mas é mais forte do que um modelo de 0.5B".

Subcommands (all GPU work is one process at a time, gate 12):
  map      hooks on every layer: per MLP neuron mean |act| and mean act^2 (act = silu(gate)*up = input of down_proj),
           per attention head mean ||head output|| (input of o_proj, per head), token counts over the full vocab,
           per domain: py (pycode clean_train functions), ptreq (belt TRAIN-side requests in the understander chat
           format + gold JSON), ctl (English licence prose), ptdialog (pt-BR dialogue corpus). -> npz
  extract  uniform per-layer keep of the top-F MLP neurons by importance on the chosen domains (default py,ptreq);
           physically slices gate/up/down; --random keeps F random neurons per layer (fixed seed). -> HF model dir
  check    max |logit diff| between two models on a few texts (positive control: extract F=1.0 == full model)
  eval     held-out Python bits/byte, pt-BR request task bits/byte (gold JSON given the chat prompt), request text
           bits/byte, pt dialogue and English control bits/byte; optional equal LoRA heal (--heal-steps) on Python
           train text for EVERY arm (native models too: equal training budget, rule 8). One json per arm.

Data provenance (gate 4):
  py map   = clean_train.txt functions, shuffled with seed 0, first --n-py; heal = the next --heal-docs of the same
             shuffle (disjoint from map); eval = clean_holdout.txt (4110 held-out functions, never mapped or healed on).
  ptreq    map = e2e/para_train/train_para.json + para_train_disp/disp_main.json (text + orig_text, all split=train);
           eval = e2e/requests.json (set A) + e2e/para/requests.json (set B). v2pool/para are never mapped or used to
           pick anything. Set A was used to pick earlier variants (it is a dev set); here it only scores.
  ctl      /usr/share/common-licenses/* paragraphs, even index -> map, odd index -> eval. (No English prose corpus on
           /media/llm-data; licence text is a narrow genre, so "ctl" = "not Python, not Portuguese", nothing more.)
  ptdialog dialogo-pt.txt bytes [0, 1 MB) -> map, [1 MB, 2 MB) -> eval.

Param arithmetic (configs on disk): Coder-3B 3.086B = MLP 36*3*2048*11008 = 2.435B + attention 0.343B + tied embedding
151936*2048 = 0.311B (+norms). Native Coder-1.5B = 1.544B. Uniform keep F gives 0.651B + F*2.435B, so F = 0.366 matches
the 1.5B (`extract --match Qwen/Qwen2.5-Coder-1.5B-Instruct` computes it exactly). Grid: F = 0.75, 0.5, match(~0.366).
Embedding/lm_head (tied) is 10% of the 3B and ~20% of an extracted 1.5B-size model: vocab pruning is reported, not done.

VRAM budget (RX 6750 XT 12 GB, cap 80% = 9.6 GB via set_per_process_memory_fraction(0.8)):
  map 3B: 6.2 GB weights + batch 8x512 activations (~0.2 GB, no lm_head) + fp32 accumulators (36x11008x2x4 B) ~ 6.8 GB.
  eval 3B: 6.2 GB + lm_head in chunks of 1024 positions (1024x151936 bf16+fp32 = 0.9 GB) ~ 7.5 GB.
  heal 3B: 6.2 GB + LoRA r16 on gate/up/down (~30M params + AdamW fp32 ~0.4 GB) + gradient checkpointing hidden states
           (micro 2x512x2048x36 bf16 = 0.15 GB) + one layer recompute + logits 1024x151936 (0.9 GB) ~ 8.5 GB (71%).

=====================================================================================================================
PREDICTION — written 2026-10-27 09:51 (from `date`), before any run of this file on any model.
Gate 1 baseline row: unigram over the map token counts (add-one) — predicted ~1.5 bits/byte on Python; every arm must
beat it. Same tokenizer for all arms (Qwen2.5 family), so bits/byte orders arms exactly as nats/token does.
Python held-out bits/byte (ctx 512, bf16), before heal:
  full 3B 0.22 | native 1.5B 0.25 | native 0.5B 0.31
  extracted-3B F=0.75 0.25 | F=0.5 0.34 | F=0.366 0.55   ; random F=0.75 0.45 | F=0.5 1.0 | F=0.366 >= 1.5
After equal heal (--heal-steps 300, LoRA r16 on MLP, same Python text): extracted F=0.366 0.30, random F=0.366 0.45,
native 1.5B 0.24 (heal on in-domain text also helps the natives a little), full 3B 0.21.
Mechanism: the criterion works (in banco_pesos top-50% by |act| kept 96.9% of tokens vs 23.5% for random), so
extracted >> random at every F. But SwiGLU never zeroes (0.00% exact zeros, 90% of |act| mass needs 52% of the
neurons): at F=0.366 each layer loses roughly a quarter of its MLP output mass, the error compounds over 36 layers,
and 300 LoRA steps re-learn only part of it, while the native 1.5B was pre-trained at that size on trillions of
tokens. Carve V1/V2 (Qwen3-4B, 07-02) needed 65-80% keep to stay usable without heal; 36.6% is far below that.
So the predicted verdict is: extracted > random (criterion is real), extracted-3B@1.5B-size < native 1.5B (extraction
does not beat pre-training at equal size). F=0.75 (2.48B params) should sit between full 3B and native 1.5B.
pt-BR task bits/byte (gold JSON | chat prompt, zero-shot): full 3B ~0.9, 1.5B ~1.0, 0.5B ~1.2, extracted F=0.366
~1.3 (format is unknown to all of them; the 0.5B student's 97.8% belt came from LoRA training, not zero-shot loss).
Control bits/byte: extracted loses MORE on ctl/ptdialog than on py (that is the point of "less general"):
  extracted F=0.366 ctl/full-3B ratio >= 1.5x the py ratio.
SUCCESS CRITERION (stated before the run): extracted-3B at ~1.5B params has lower Python bits/byte than the
random-pruned 3B at the same F, AND <= native 1.5B on Python bits/byte after the same heal. If the first holds and
the second fails, the importance map is real but extraction is not the cause of any gain over simply using the native
1.5B; the next arm would then be the belt understander (s5 recipe) on the F=0.75 trunk vs native 1.5B (#49), not a
smaller F.
=====================================================================================================================

Usage: see `py_extract.py <cmd> -h`. Python: /home/leonardo/projects/LLM/.venv-rocm/bin/python only.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = "/media/llm-data/models/hf/hub"
RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
OUT_MAP = os.path.join(RUN, "py_extract")
OUT_MODELS = "/media/llm-data/models/extract"
PY_TRAIN = "/media/llm-data/datasets/pycode/clean_train.txt"
PY_HOLD = "/media/llm-data/datasets/pycode/clean_holdout.txt"
PT_TRAIN = [os.path.join(RUN, "e2e/para_train/train_para.json"), os.path.join(RUN, "e2e/para_train_disp/disp_main.json")]
PT_EVAL = {"A": os.path.join(RUN, "e2e/requests.json"), "B": os.path.join(RUN, "e2e/para/requests.json")}
DIALOG = "/media/llm-data/datasets/dialogo-pt.txt"
LICENSES = "/usr/share/common-licenses"
S_SYS = "Turn the Portuguese request into the JSON spec for the code generator. Answer with the JSON only."  # = student
DOMAINS = ("py", "ptreq", "ctl", "ptdialog")
VRAM_CAP = 0.8
LN2 = math.log(2.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chat(system, user):   # identical to e2e_pipeline.chat(system, [], user); copied so the running queue's module
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


# ----------------------------------------------------------------------------------------------------------- data
def py_docs(path, max_bytes=None):
    s = open(path, encoding="utf-8", errors="ignore").read(max_bytes) if max_bytes else open(path).read()
    docs = [p.strip("\n") for p in re.split(r"\n\n+(?=\S)", s) if p.strip()]
    return [d for d in docs if len(d) >= 64]


def py_train_split(n_map, n_heal, seed=0):
    docs = py_docs(PY_TRAIN)
    perm = np.random.default_rng(seed).permutation(len(docs))
    return [docs[i] for i in perm[:n_map]], [docs[i] for i in perm[n_map:n_map + n_heal]]


def pt_train_items():
    """(prompt, answer) pairs, train side only: rewritten text and its templated original, both with gold."""
    seen, out = set(), []
    for f in PT_TRAIN:
        for x in json.load(open(f)):
            assert x["split"] == "train", (f, x["id"])
            for t in (x["text"], x.get("orig_text")):
                if t and t not in seen:
                    seen.add(t)
                    out.append((chat(S_SYS, t), json.dumps(x["gold"]) + "<|im_end|>"))
    return out


def pt_eval_items():
    out = []
    for tag, f in PT_EVAL.items():
        for x in json.load(open(f)):
            out.append((tag, x["text"], json.dumps(x["gold"])))
    return out


def license_paras():
    paras = []
    for f in sorted(glob.glob(os.path.join(LICENSES, "*"))):
        if os.path.islink(f):
            continue
        txt = open(f, encoding="utf-8", errors="ignore").read()
        paras += [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", txt)]
    paras = [p for p in paras if len(p) >= 200 and sum(c.isalpha() for c in p) / len(p) > 0.7]
    return paras[0::2], paras[1::2]          # map, eval


def dialog_chunks(lo, hi, size=1500):
    with open(DIALOG, "rb") as f:
        f.seek(lo)
        raw = f.read(hi - lo).decode("utf-8", errors="ignore")
    lines, out, cur = raw.split("\n")[1:-1], [], ""   # drop the partial first/last line
    for ln in lines:
        if len(cur) + len(ln) > size and cur:
            out.append(cur)
            cur = ""
        cur += ln + "\n"
    return out


# ---------------------------------------------------------------------------------------------------------- model
def setup_device(dev):
    if dev.startswith("cuda"):
        assert torch.cuda.is_available(), "cuda asked but no GPU visible"
        torch.cuda.set_per_process_memory_fraction(VRAM_CAP)      # hard cap: 80% of the card (root CLAUDE.md)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        log(f"GPU {torch.cuda.get_device_name(0)} total {total:.1f} GiB, cap {VRAM_CAP:.0%} = {VRAM_CAP*total:.1f} GiB")
    else:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    return torch.device(dev)


def peak(dev):
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None


def is_hub_id(m):
    return not os.path.isdir(m)


def load(m, dev, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(cache_dir=CACHE, local_files_only=True) if is_hub_id(m) else dict(local_files_only=True)
    tok = AutoTokenizer.from_pretrained(m, **kw)
    model = AutoModelForCausalLM.from_pretrained(m, dtype=dtype, **kw).to(dev).eval()
    return tok, model


def n_params(model):
    return sum(p.numel() for p in model.parameters())     # tied embedding counted once


def config_params(m):
    """Param count of a model id/dir without loading weights (meta device)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    kw = dict(cache_dir=CACHE, local_files_only=True) if is_hub_id(m) else dict(local_files_only=True)
    cfg = AutoConfig.from_pretrained(m, **kw)
    with torch.device("meta"):
        mm = AutoModelForCausalLM.from_config(cfg)
    return n_params(mm), cfg


def batches(tok, items, ctx, bs):
    """items: list of (text_or_ids, score_from_char_or_None). Yields padded id/mask/score tensors, sorted by length.
    Each item is either a plain string (score every token after the first) or (prompt, answer): score answer only."""
    enc = []
    for it in items:
        if isinstance(it, tuple):
            p = tok(it[0], add_special_tokens=False).input_ids
            a = tok(it[1], add_special_tokens=False).input_ids
            ids, s0 = (p + a)[:ctx], len(p)
        else:
            ids, s0 = tok(it, add_special_tokens=False).input_ids[:ctx], 1
        if len(ids) >= 2 and s0 < len(ids):
            enc.append((ids, s0))
    enc.sort(key=lambda e: len(e[0]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(enc), bs):
        chunk = enc[i:i + bs]
        T = max(len(e[0]) for e in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        am = torch.zeros((len(chunk), T), dtype=torch.long)
        sc = torch.zeros((len(chunk), T), dtype=torch.bool)    # sc[b, t]: token t is a scored target
        nbytes = 0
        for b, (x, s0) in enumerate(chunk):
            ids[b, :len(x)] = torch.tensor(x)
            am[b, :len(x)] = 1
            sc[b, s0:len(x)] = True
            nbytes += len(tok.decode(x[s0:]).encode("utf-8"))
        yield ids, am, sc, nbytes


# ------------------------------------------------------------------------------------------------------------ map
class Recorder:
    """Accumulates per-layer sums inside hooks with whole-tensor ops (no per-token Python loops; root CLAUDE.md)."""

    def __init__(self, model, dev):
        cfg = model.config
        self.L, self.I, self.nH = cfg.num_hidden_layers, cfg.intermediate_size, cfg.num_attention_heads
        self.hd = cfg.hidden_size // self.nH
        self.mask = None
        self.reset(dev)
        self.hs = []
        for i, layer in enumerate(model.model.layers):
            self.hs.append(layer.mlp.down_proj.register_forward_pre_hook(self._mlp(i)))
            self.hs.append(layer.self_attn.o_proj.register_forward_pre_hook(self._attn(i)))

    def reset(self, dev):
        self.abs = torch.zeros(self.L, self.I, device=dev)
        self.sq = torch.zeros(self.L, self.I, device=dev)
        self.head = torch.zeros(self.L, self.nH, device=dev)
        self.shapes = {}

    def _mlp(self, i):
        def h(mod, inp):                       # must not return anything (returned value would replace the input)
            x = inp[0].float()                 # [B, T, I] = silu(gate) * up
            self.shapes.setdefault("mlp_in", tuple(x.shape))
            m = self.mask
            self.abs[i] += torch.einsum("bti,bt->i", x.abs(), m)
            self.sq[i] += torch.einsum("bti,bt->i", x * x, m)
        return h

    def _attn(self, i):
        def h(mod, inp):
            x = inp[0]
            B, T, _ = x.shape
            self.shapes.setdefault("o_in", tuple(x.shape))
            n = x.view(B, T, self.nH, self.hd).float().norm(dim=-1)        # [B, T, nH]
            self.head[i] += torch.einsum("bth,bt->h", n, self.mask)
        return h

    def remove(self):
        for h in self.hs:
            h.remove()


def cmd_map(a):
    dev = setup_device(a.device)
    t0 = time.time()
    tok, model = load(a.model, dev)
    V = model.config.vocab_size
    log(f"loaded {a.model} params {n_params(model)/1e9:.3f}B ({time.time()-t0:.0f}s)")
    py_map, _ = py_train_split(a.n_py, 0)
    ctl_map, _ = license_paras()
    data = {"py": py_map, "ptreq": pt_train_items()[:a.n_pt], "ctl": ctl_map[:a.n_ctl],
            "ptdialog": dialog_chunks(0, 1_000_000)[:a.n_dialog]}
    rec = Recorder(model, dev)
    out, meta = {}, {"model": a.model, "ctx": a.ctx, "bs": a.bs, "n_py_arg": a.n_py, "date": time.strftime("%Y-%m-%d %H:%M"),
                     "sources": {"py": PY_TRAIN + " (seed-0 shuffle, first n_py)", "ptreq": PT_TRAIN,
                                 "ctl": LICENSES + " even paragraphs", "ptdialog": DIALOG + " [0,1MB)"},
                     "n_texts": {}, "n_tok": {}, "shapes": {}}
    with torch.no_grad():
        for d in DOMAINS:
            if not data[d]:
                continue
            rec.reset(dev)
            counts = torch.zeros(V, dtype=torch.long)
            ntok, nused, td = 0, 0, time.time()
            for ids, am, sc, _ in batches(tok, data[d], a.ctx, a.bs):
                nused += len(ids)
                ids, am = ids.to(dev), am.to(dev)
                rec.mask = am.float()
                model.model(input_ids=ids, attention_mask=am)           # trunk only: no 151k-wide logits
                ntok += int(am.sum())
                counts += torch.bincount(ids[am.bool()].cpu(), minlength=V)
            out[f"mlp_abs_{d}"] = (rec.abs / ntok).cpu().numpy()
            out[f"mlp_sq_{d}"] = (rec.sq / ntok).cpu().numpy()
            out[f"head_{d}"] = (rec.head / ntok).cpu().numpy()
            out[f"tok_{d}"] = counts.numpy()
            meta["n_texts"][d], meta["n_tok"][d] = nused, ntok      # texts whose answer fits in ctx
            meta["shapes"][d] = {k: list(v) for k, v in rec.shapes.items()}
            log(f"map {d}: {nused}/{len(data[d])} texts {ntok} tokens {time.time()-td:.1f}s peak {peak(dev)} GiB "
                f"hook shapes {rec.shapes}")
    rec.remove()
    if a.vocab_mb > 0:     # token usage over a bigger Python slice (tokenizer only, no model)
        big = tok(open(PY_TRAIN, encoding="utf-8", errors="ignore").read(int(a.vocab_mb * 2**20)),
                  add_special_tokens=False).input_ids
        out["tok_py_big"] = np.bincount(np.asarray(big), minlength=V)
        meta["n_tok"]["py_big"] = len(big)
    out["down_norm"] = torch.stack([l.mlp.down_proj.weight.detach().float().norm(dim=0)
                                    for l in model.model.layers]).cpu().numpy()          # [L, I]: ||W_down[:, j]||
    out["meta"] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **out)
    summarize_map(a.out, model.config)
    log(f"saved {a.out} ({time.time()-t0:.0f}s total, peak {peak(dev)} GiB)")


def summarize_map(path, cfg):
    z = np.load(path)
    doms = [d for d in DOMAINS if f"mlp_abs_{d}" in z]
    L, I = z[f"mlp_abs_{doms[0]}"].shape
    E = cfg.vocab_size * cfg.hidden_size
    s = {"domains": doms}
    for F_ in (0.75, 0.5, 0.366):
        k = int(round(F_ * I))
        tops = {d: np.argsort(-z[f"mlp_abs_{d}"], axis=1)[:, :k] for d in doms}
        jac = {}
        for i, d1 in enumerate(doms):
            for d2 in doms[i + 1:]:
                j = [len(set(tops[d1][l]) & set(tops[d2][l])) / len(set(tops[d1][l]) | set(tops[d2][l]))
                     for l in range(L)]
                jac[f"{d1}~{d2}"] = round(float(np.mean(j)), 3)
        s[f"jaccard_top{F_}"] = jac
    if "ctl" in doms:
        for d in ("py", "ptreq"):
            if d in doms:
                r = (z[f"mlp_abs_{d}"] / z[f"mlp_abs_{d}"].mean(1, keepdims=True)) / \
                    (z["mlp_abs_ctl"] / z["mlp_abs_ctl"].mean(1, keepdims=True) + 1e-9)
                s[f"{d}_selective_frac(ratio>2 vs ctl)"] = round(float((r > 2).mean()), 4)
                s[f"{d}_selective_by_layer"] = [round(float(x), 3) for x in (r > 2).mean(1)]
    used = {d: z[f"tok_{d}"] > 0 for d in doms}
    if "tok_py_big" in z:
        used["py_big"] = z["tok_py_big"] > 0
    union = np.zeros(cfg.vocab_size, bool)
    for d in ("py", "py_big", "ptreq"):
        if d in used:
            union |= used[d]
    s["vocab_rows_used"] = {d: int(u.sum()) for d, u in used.items()}
    s["vocab_rows_used_py_union_ptreq"] = int(union.sum())
    s["vocab_rows_unused_frac"] = round(1 - union.sum() / cfg.vocab_size, 4)
    s["embedding_params(tied)"] = E
    log("map summary " + json.dumps(s))
    json.dump(s, open(path.replace(".npz", "") + "_summary.json", "w"), indent=1)


# -------------------------------------------------------------------------------------------------------- extract
def importance(z, domains, score):
    """[L, I] importance: per domain mean |act| (x ||W_down col|| for 'contrib'), normalised by its layer mean so
    a domain with more tokens or larger activations does not dominate; then averaged over the chosen domains."""
    tot = 0
    for d in domains:
        s = z[f"mlp_abs_{d}"].astype(np.float64)
        if score == "contrib":
            s = s * z["down_norm"]
        tot = tot + s / s.mean(1, keepdims=True)
    return tot / len(domains)


def slice_mlp(model, keep_idx):
    """keep_idx: list of sorted LongTensors, one per layer, all the same length (HF Qwen2 has one intermediate_size)."""
    k = len(keep_idx[0])
    for layer, idx in zip(model.model.layers, keep_idx):
        assert len(idx) == k
        mlp = layer.mlp
        dev, dt = mlp.gate_proj.weight.device, mlp.gate_proj.weight.dtype
        idx = idx.to(dev)
        g = nn.Linear(mlp.gate_proj.in_features, k, bias=False, device=dev, dtype=dt)
        u = nn.Linear(mlp.up_proj.in_features, k, bias=False, device=dev, dtype=dt)
        d = nn.Linear(k, mlp.down_proj.out_features, bias=False, device=dev, dtype=dt)
        with torch.no_grad():
            g.weight.copy_(mlp.gate_proj.weight[idx])
            u.weight.copy_(mlp.up_proj.weight[idx])
            d.weight.copy_(mlp.down_proj.weight[:, idx])
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = g, u, d
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = k
    model.config.intermediate_size = k


def match_keep(src, ref):
    p_src, cfg = config_params(src)
    p_ref, _ = config_params(ref)
    mlp = cfg.num_hidden_layers * 3 * cfg.hidden_size * cfg.intermediate_size
    return (p_ref - (p_src - mlp)) / mlp, p_src, p_ref


def cmd_extract(a):
    dev = setup_device(a.device)
    keeps = []
    for f in a.keep.split(","):
        if f.startswith("match:"):
            F_, p_src, p_ref = match_keep(a.model, f[6:])
            log(f"match {f[6:]}: src {p_src/1e9:.4f}B ref {p_ref/1e9:.4f}B -> F = {F_:.4f}")
            keeps.append(F_)
        else:
            keeps.append(float(f))
    z = np.load(a.map)
    meta = json.loads(str(z["meta"]))
    assert meta["model"] == a.model, f"map was made on {meta['model']}, not {a.model}"
    doms = a.domains.split(",")
    imp = importance(z, doms, a.score)
    L, I = imp.shape
    map_sha = hashlib.sha256(open(a.map, "rb").read()).hexdigest()[:16]
    tag = os.path.basename(a.model.rstrip("/")).replace("Qwen2.5-Coder-", "c").replace("-Instruct", "")
    for F_ in keeps:
        k = int(round(F_ * I / a.multiple)) * a.multiple
        k = max(a.multiple, min(I, k))
        tok, model = load(a.model, dev)
        p0 = n_params(model)
        if a.random:
            rng = np.random.default_rng(a.seed)
            idx = [torch.as_tensor(np.sort(rng.permutation(I)[:k])) for _ in range(L)]
            name = f"{tag}-random-s{a.seed}-F{k/I:.3f}"
        else:
            idx = [torch.as_tensor(np.sort(np.argsort(-imp[l], kind="stable")[:k])) for l in range(L)]
            name = f"{tag}-{a.score}-{'+'.join(doms)}-F{k/I:.3f}"
        kept_mass = float(np.mean([imp[l][idx[l].numpy()].sum() / imp[l].sum() for l in range(L)]))
        slice_mlp(model, idx)
        p1 = n_params(model)
        odir = os.path.join(a.out_dir, name)
        model.save_pretrained(odir, safe_serialization=True)
        tok.save_pretrained(odir)
        z_used = np.zeros(model.config.vocab_size, bool)
        for d in ("py", "py_big", "ptreq"):
            if f"tok_{d}" in z:
                z_used |= z[f"tok_{d}"] > 0
        emb = model.config.vocab_size * model.config.hidden_size
        info = {"source": a.model, "map": a.map, "map_sha256_16": map_sha, "domains": doms, "score": a.score,
                "random": a.random, "seed": a.seed if a.random else None, "F_asked": F_, "k": k, "I": I,
                "F_real": k / I, "params_before": p0, "params_after": p1,
                "importance_mass_kept_mean": round(kept_mass, 4), "kept_idx_sha16":
                hashlib.sha256(np.concatenate([i.numpy() for i in idx]).tobytes()).hexdigest()[:16],
                "vocab_rows_unused_by_py_ptreq": int((~z_used).sum()),
                "embedding_share_after": round(emb / p1, 4),
                "params_if_unused_rows_dropped": int(p1 - (~z_used).sum() * model.config.hidden_size),
                "date": time.strftime("%Y-%m-%d %H:%M")}
        json.dump(info, open(os.path.join(odir, "extract_info.json"), "w"), indent=1)
        log(f"extract {name}: {p0/1e9:.4f}B -> {p1/1e9:.4f}B (k={k}/{I}) mass kept {kept_mass:.3f} -> {odir}")
        log("  " + json.dumps({x: info[x] for x in ("vocab_rows_unused_by_py_ptreq", "embedding_share_after",
                                                    "params_if_unused_rows_dropped")}))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------- check
def check_texts():
    py = py_docs(PY_HOLD)[:3]
    pt = [chat(S_SYS, t) + json.dumps(g) for _, t, g in pt_eval_items()[:3]]
    return py + pt


def cmd_check(a):
    dev = setup_device(a.device)
    texts = check_texts()[:a.n]
    res = []
    for m in (a.a, a.b):
        tok, model = load(m, dev)
        with torch.no_grad():
            res.append([model(**{k: v.to(dev) for k, v in tok(t, return_tensors="pt", truncation=True,
                                                                  max_length=a.max_len).items()}).logits.float().cpu()
                        for t in texts])
        log(f"check {m}: params {n_params(model)/1e9:.4f}B")
        del model
    diffs = [float((x - y).abs().max()) for x, y in zip(*res)]
    agree = [float((x.argmax(-1) == y.argmax(-1)).float().mean()) for x, y in zip(*res)]
    log("check " + json.dumps({"texts": len(texts), "max_abs_logit_diff": max(diffs), "per_text": diffs,
                               "argmax_agree": agree}))


# ----------------------------------------------------------------------------------------------------------- eval
def eval_sets(a):
    hold = py_docs(PY_HOLD)
    step = max(1, len(hold) // a.n_py)
    py = hold[::step][:a.n_py]                 # spread over the whole held-out file, fixed
    pte = pt_eval_items()[:a.n_pt] if a.n_pt else pt_eval_items()
    _, ctl = license_paras()
    sets = {"py": py,
            "pt_task": [(chat(S_SYS, t), g + "<|im_end|>") for _, t, g in pte],   # gold JSON | chat prompt
            "pt_text": [t for _, t, _ in pte],
            "ctl": ctl[:a.n_ctl],
            "ptdialog": dialog_chunks(1_000_000, 2_000_000)[:a.n_dialog]}
    return {k: v for k, v in sets.items() if v}


@torch.no_grad()
def bits_per_byte(model, tok, items, ctx, bs, dev, chunk=1024):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    nats, nbytes, ntok = 0.0, 0, 0
    for ids, am, sc, nb in batches(tok, items, ctx, bs):
        ids, am, sc = ids.to(dev), am.to(dev), sc.to(dev)
        h = base.model(input_ids=ids, attention_mask=am).last_hidden_state
        m = sc[:, 1:]                                  # position t predicts token t+1
        hs, ys = h[:, :-1][m], ids[:, 1:][m]
        for i in range(0, len(ys), chunk):             # never the full [B, T, 151k] logits
            lg = base.lm_head(hs[i:i + chunk]).float()
            nats += float(F.cross_entropy(lg, ys[i:i + chunk], reduction="sum"))
        nbytes += nb
        ntok += int(m.sum())
    return {"bpb": round(nats / LN2 / nbytes, 5), "nats_per_tok": round(nats / ntok, 5), "tokens": ntok,
            "bytes": nbytes}


def unigram_row(map_path, tok, sets, ctx):
    z = np.load(map_path)
    c = (z["tok_py_big"] if "tok_py_big" in z else z["tok_py"]).astype(np.float64) + 1.0
    lp = np.log(c / c.sum())
    row = {}
    for name, items in sets.items():
        nats, nbytes = 0.0, 0
        for ids, am, sc, nb in batches(tok, items, ctx, 64):
            y = ids[:, 1:][sc[:, 1:]].numpy()
            nats -= float(lp[y].sum())
            nbytes += nb
        row[name] = {"bpb": round(nats / LN2 / nbytes, 5)}
    return row


def heal(model, tok, docs, steps, a, dev):
    """Equal LoRA heal for every arm: gate/up/down, same docs, order, lr, steps; rank fixed (--lora-r) or matched
    to a trainable-param target (--lora-target, review fix 2026-10-27 10:20)."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(a.seed)
    r = a.lora_r
    if a.lora_target:        # review 10-27: equal TRAINABLE params across arms of different shape
        per_r = sum(m.in_features + m.out_features for n, m in model.named_modules()
                    if isinstance(m, nn.Linear) and n.split(".")[-1] in ("gate_proj", "up_proj", "down_proj"))
        r = max(1, int(round(a.lora_target / per_r)))
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                             target_modules=["gate_proj", "up_proj", "down_proj"],
                                             task_type="CAUSAL_LM"))
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dev.type == "cuda":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    base = model.get_base_model()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(docs))
    accum = max(1, a.heal_bs // a.micro)
    losses, pos = [], 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / max(1, steps // 10))
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            pick = [docs[order[(pos + j) % len(docs)]] for j in range(a.micro)]
            pos += a.micro
            ids, am, sc, _ = next(batches(tok, pick, a.ctx, a.micro))
            ids, am, sc = ids.to(dev), am.to(dev), sc.to(dev)
            h = base.model(input_ids=ids, attention_mask=am).last_hidden_state
            m = sc[:, 1:]
            loss = F.cross_entropy(base.lm_head(h[:, :-1][m]).float(), ids[:, 1:][m]) / accum
            loss.backward()
            tot += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(round(tot, 4))
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log(f"  heal step {step+1}/{steps} loss {tot:.4f} peak {peak(dev)} GiB")
    model.eval()
    return model, {"lora_params": ntr, "lora_r": r, "lora_target": a.lora_target, "lr": a.lr, "heal_bs": a.heal_bs,
                   "micro": a.micro, "heal_docs": len(docs), "seed": a.seed, "steps": steps, "loss_first": losses[0], "loss_last": losses[-1],
                   "loss_log": losses[::max(1, steps // 50)]}


def cmd_eval(a):
    dev = setup_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    sets = None
    vocab_ref = None
    if a.map:                # review 10-27: heal docs must start after the mapped docs
        mm = json.loads(str(np.load(a.map)["meta"]))
        assert mm.get("n_py_arg", a.n_map_py) == a.n_map_py, f"--n-map-py {a.n_map_py} != map n_py {mm.get('n_py_arg')}"
    _, heal_docs = py_train_split(a.n_map_py, a.heal_docs) if a.heal_steps else (None, None)
    for spec in a.models.split(","):
        label, m = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/")), spec)
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tok, model = load(m, dev)
        v = hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()[:16]
        vocab_ref = vocab_ref or v
        assert v == vocab_ref, f"{label}: different tokenizer, bits/byte rows would still compare but check it"
        if sets is None:
            sets = eval_sets(a)
            log("eval sets " + json.dumps({k: len(x) for k, x in sets.items()}))
            if a.map:
                row = unigram_row(a.map, tok, sets, a.ctx)
                json.dump({"label": "unigram_py_counts", "before": row}, open(os.path.join(a.out, "eval_unigram.json"),
                                                                                "w"), indent=1)
                log("eval unigram baseline " + json.dumps(row))
        res = {"label": label, "seed": a.seed, "heal_steps": a.heal_steps, "model": m, "params": n_params(model), "ctx": a.ctx, "dtype": "bf16",
               "tokenizer_sha16": v, "date": time.strftime("%Y-%m-%d %H:%M")}
        res["before"] = {k: bits_per_byte(model, tok, x, a.ctx, a.bs, dev) for k, x in sets.items()}
        log(f"eval {label} ({res['params']/1e9:.4f}B) before: " +
            json.dumps({k: r["bpb"] for k, r in res["before"].items()}))
        if a.heal_steps:
            th = time.time()
            model, hinfo = heal(model, tok, heal_docs, a.heal_steps, a, dev)
            hinfo["heal_s"] = round(time.time() - th, 1)
            res["heal"] = hinfo
            res["after"] = {k: bits_per_byte(model, tok, x, a.ctx, a.bs, dev) for k, x in sets.items()}
            log(f"eval {label} after {a.heal_steps} heal steps: " +
                json.dumps({k: r["bpb"] for k, r in res["after"].items()}))
        res["wall_s"] = round(time.time() - t0, 1)
        res["peak_vram_gib"] = peak(dev)
        json.dump(res, open(os.path.join(a.out, f"eval_{label}_s{a.seed}.json"), "w"), indent=1)    # per-arm, per-seed file
        log(f"eval {label} done {res['wall_s']}s peak {res['peak_vram_gib']} GiB")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    table = {}
    for f in sorted(glob.glob(os.path.join(a.out, "eval_*.json"))):
        r = json.load(open(f))
        if "label" not in r or "before" not in r:
            continue
        table[f'{r["label"]}_s{r.get("seed", "?")}'] = {"lora_params": r.get("heal", {}).get("lora_params"), "params_B": round(r.get("params", 0) / 1e9, 4),
                             **{f"{k}_before": v["bpb"] for k, v in r["before"].items()},
                             **{f"{k}_after": v["bpb"] for k, v in r.get("after", {}).items()}}
    json.dump(table, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    log("summary " + json.dumps(table))


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("map")
    m.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    m.add_argument("--out", default=os.path.join(OUT_MAP, "map_c3b.npz"))
    m.add_argument("--device", default="cuda")
    m.add_argument("--ctx", type=int, default=512)
    m.add_argument("--bs", type=int, default=8)
    m.add_argument("--n-py", type=int, default=2048)
    m.add_argument("--n-pt", type=int, default=100000, help="cap on train-side requests (all ~1.8k by default)")
    m.add_argument("--n-ctl", type=int, default=100000)
    m.add_argument("--n-dialog", type=int, default=600)
    m.add_argument("--vocab-mb", type=float, default=8.0, help="MB of clean_train tokenised for vocab usage")

    e = sub.add_parser("extract")
    e.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    e.add_argument("--map", default=os.path.join(OUT_MAP, "map_c3b.npz"))
    e.add_argument("--keep", required=True, help="comma list of fractions and/or match:<model id>")
    e.add_argument("--domains", default="py,ptreq")
    e.add_argument("--score", choices=["contrib", "act"], default="contrib",
                   help="contrib = mean|act| x ||W_down col|| (output magnitude); act = mean|act| (carve v1/v2)")
    e.add_argument("--random", action="store_true")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--multiple", type=int, default=16, help="round k to this multiple")
    e.add_argument("--out-dir", default=OUT_MODELS)
    e.add_argument("--device", default="cuda")

    c = sub.add_parser("check")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--max-len", type=int, default=128)
    c.add_argument("--device", default="cuda")

    v = sub.add_parser("eval")
    v.add_argument("--models", required=True, help="comma list of label=model_id_or_dir")
    v.add_argument("--out", default=os.path.join(OUT_MAP, "eval"))
    v.add_argument("--map", default=None, help="map npz: adds the unigram baseline row (gate 1)")
    v.add_argument("--device", default="cuda")
    v.add_argument("--ctx", type=int, default=512)
    v.add_argument("--bs", type=int, default=8)
    v.add_argument("--n-py", type=int, default=600)
    v.add_argument("--n-pt", type=int, default=0, help="0 = all of set A + set B")
    v.add_argument("--n-ctl", type=int, default=100000)
    v.add_argument("--n-dialog", type=int, default=300)
    v.add_argument("--heal-steps", type=int, default=0)
    v.add_argument("--heal-docs", type=int, default=20000)
    v.add_argument("--n-map-py", type=int, default=2048, help="must equal map --n-py (heal docs come after them)")
    v.add_argument("--heal-bs", type=int, default=8)
    v.add_argument("--micro", type=int, default=2)
    v.add_argument("--lora-r", type=int, default=16)
    v.add_argument("--lora-target", type=int, default=0,
                   help="if >0, LoRA rank per model = round(target / params-per-rank): equal trainable params")
    v.add_argument("--lr", type=float, default=2e-4)
    v.add_argument("--seed", type=int, default=0)

    a = ap.parse_args()
    {"map": cmd_map, "extract": cmd_extract, "check": cmd_check, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
