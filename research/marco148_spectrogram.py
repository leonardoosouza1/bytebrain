#!/usr/bin/env python3
"""M148 — ESPECTOGRAMA do transplante: análise espectral (SVD) do delta base→Instruct por camada, tipo de
projeção. RAM-safe (per-tensor, `with safe_open`, SVD na GPU). Produz: (1) espectograma [camada × rank]
dos valores singulares do delta; (2) energia do delta por camada; (3) rank efetivo por camada/tipo;
(4) curva de energia cumulativa vs rank (compressibilidade). Dump marco148_metrics.json p/ visualizações."""
import sys, json, os, time
import numpy as np, torch
from safetensors import safe_open
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco148_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE = f"{MODELS}/Qwen2.5-7B"; DONOR = f"{MODELS}/Qwen2.5-7B-Instruct"; DEV = "cuda"
NL = 28; TOPK = 64
PROJS = {"q": "self_attn.q_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "down": "mlp.down_proj"}

def wmap(d): return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
bmap, dmap = wmap(BASE), wmap(DONOR)
def get1(d, mp, key):
    with safe_open(f"{d}/{mp[key]}", framework="pt", device="cpu") as f: return f.get_tensor(key)

res = {"NL": NL, "topk": TOPK, "projs": list(PROJS), "spectro": {}, "energia": {}, "rank_efetivo": {}, "cum_energia": {}}
for pn, ps in PROJS.items():
    spectro = []; energia = []; effr = []; cum = []
    for li in range(NL):
        k = f"model.layers.{li}.{ps}.weight"
        if k not in dmap or k not in bmap: continue
        bt = get1(BASE, bmap, k).to(DEV).float(); dt = get1(DONOR, dmap, k).to(DEV).float()
        delta = dt - bt
        e = float(delta.pow(2).sum())  # energia total (Frobenius²) — barato e exato
        _, S, _ = torch.svd_lowrank(delta, q=TOPK, niter=2)  # top-TOPK valores singulares (rápido)
        s = S.cpu().numpy()
        topn = s / (s[0] + 1e-9)  # espectro normalizado
        cs = np.cumsum(s ** 2) / (e + 1e-9)  # energia cumulativa (do topo)
        # rank pra 50% da energia (aprox pelo espectro top-K)
        r50 = int(np.searchsorted(cs, 0.5) + 1) if cs[-1] >= 0.5 else TOPK
        spectro.append([round(float(x), 4) for x in topn])
        energia.append(round(e, 2)); effr.append(r50)
        cum.append([round(float(cs[min(r-1, len(cs)-1)]), 3) for r in [1, 2, 4, 8, 16, 32, 48, 64]])
        del bt, dt, delta, S; torch.cuda.empty_cache()
    res["spectro"][pn] = spectro; res["energia"][pn] = energia
    res["rank_efetivo"][pn] = effr; res["cum_energia"][pn] = cum
    log(f"  {pn:5}: {len(energia)} camadas | rank efetivo médio {np.mean(effr):.0f} | "
        f"energia cum@rank32 média {np.mean([c[5] for c in cum]):.2f}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"DONE M148 ({time.time()-t0:.0f}s)")
