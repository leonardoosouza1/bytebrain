#!/usr/bin/env python3
"""M150 — espectograma CROSS-EMPRESA: computa a energia cumulativa do delta (base→instruct) por rank p/
qualquer par, RAM-safe. Uso: python marco150_crosscompany.py <base> <donor> <label>. Compara com o Qwen
(o delta alto-rank é universal entre empresas?). Append em marco150_metrics.json."""
import sys, json, os, time
import numpy as np, torch
from safetensors import safe_open
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE, DONOR, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco150_metrics.json"
DEV = "cuda"; TOPK = 64
NL = json.load(open(f"{DONOR}/config.json"))["num_hidden_layers"]
PROJS = {"q": "self_attn.q_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "down": "mlp.down_proj"}
def wmap(d):
    idx = f"{d}/model.safetensors.index.json"
    if os.path.exists(idx): return json.load(open(idx))["weight_map"]
    return {k: "model.safetensors" for k in safe_open(f"{d}/model.safetensors", framework="pt").keys()}
bmap, dmap = wmap(BASE), wmap(DONOR)
def get1(d, mp, key):
    with safe_open(f"{d}/{mp[key]}", framework="pt", device="cpu") as f: return f.get_tensor(key)
log(f"{LABEL}: {NL} camadas")
cum_by_proj = {}
for pn, ps in PROJS.items():
    cums = []
    for li in range(NL):
        k = f"model.layers.{li}.{ps}.weight"
        if k not in dmap or k not in bmap: continue
        bt = get1(BASE, bmap, k).to(DEV).float(); dt = get1(DONOR, dmap, k).to(DEV).float()
        d = dt - bt; e = float(d.pow(2).sum())
        _, S, _ = torch.svd_lowrank(d, q=TOPK, niter=2)
        cs = np.cumsum(S.cpu().numpy() ** 2) / (e + 1e-9)
        cums.append([round(float(cs[min(r-1, len(cs)-1)]), 3) for r in [8, 16, 32, 64]])
        del bt, dt, d, S; torch.cuda.empty_cache()
    cum_by_proj[pn] = [round(float(x), 3) for x in np.mean(cums, axis=0)]  # média por camada
    log(f"  {pn:5}: cum energia @ ranks[8,16,32,64] = {cum_by_proj[pn]}")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res[LABEL] = {"NL": NL, "cum_ranks": [8, 16, 32, 64], "cum_by_proj": cum_by_proj}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"DONE M150 {LABEL} ({time.time()-t0:.0f}s)")
