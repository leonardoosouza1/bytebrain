#!/usr/bin/env python3
"""BATERIA 11 — ROTEAMENTO com GATE MAIS RICO (lote 3). O lote 2 mostrou: gate por incerteza do BIGRAMA fecha
só ~32% do gap do oráculo. Hipótese honesta a testar: um gate CHEAP porém de ORDEM MAIOR (trigrama) sabe
melhor ONDE o bigrama falha → fecha mais do gap. Também registro a verdade honesta sobre 'especulativo':
draft+verify roda o modelo grande TODO passo (custo=sempre-caro no frame de energia; ele economiza LATÊNCIA,
não COMPUTE) — logo NÃO fecha o gap de energia; o lever real é o gate cheap mais rico. GPU."""
import sys, math
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
import numpy as np, torch, torch.nn.functional as F
from wisdom_bridge import load_byte_model, DEV

m = load_byte_model(trained=True)
path = "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt"
with open(path, "rb") as f:
    f.seek(5_050_000); data = f.read(40000)
N = len(data); W = 256

# órgão CARO (ByteBrain): bits/byte
exp = np.full(N, np.nan)
with torch.no_grad():
    for s in range(0, N-1, W):
        e = min(s+W, N); L = e-s
        if L < 2: break
        ids = torch.tensor([list(data[s:e])], device=DEV)
        lp = F.log_softmax(m(inputs_embeds=m.tok(ids))[0][:-1].float(), -1)
        b = (-(lp[torch.arange(L-1, device=DEV), ids[0,1:]])/0.69314718).cpu().numpy()
        exp[s+1:s+1+len(b)] = b

def online_ngram(order):
    """bits/byte e ENTROPIA preditiva (pre-outcome) de um modelo online de contexto de `order` bytes."""
    from collections import defaultdict
    ones = defaultdict(lambda: np.ones(8)); tot = defaultdict(lambda: 2.0)
    bits = np.full(N, np.nan); ent = np.full(N, np.nan)
    for i in range(order, N):
        ctx = bytes(data[i-order:i]); b = data[i]
        p = np.clip(ones[ctx]/tot[ctx], 1e-6, 1-1e-6)
        a = np.array([(b>>k)&1 for k in range(8)], float)
        bits[i] = float(np.sum(-(a*np.log2(p)+(1-a)*np.log2(1-p))))
        ent[i]  = float(np.sum(-(p*np.log2(p)+(1-p)*np.log2(1-p))))
        ones[ctx] += a; tot[ctx] += 1
    return bits, ent

che1, ent1 = online_ngram(1)   # bigrama (order-1) — o gate do lote 2
che2, ent2 = online_ngram(2)   # trigrama (order-2) — gate cheap mais rico

idx = np.array([i for i in range(2, N) if not np.isnan(exp[i]) and not np.isnan(ent2[i])])
EXP, CHE = exp[idx], che1[idx]                     # órgão barato de PREDIÇÃO segue sendo o bigrama
E1, E2 = ent1[idx], ent2[idx]
half = len(idx)//2; trS, teS = slice(0,half), slice(half,None)
print(f"held-out {N}B · {len(idx)} pos · barato(bigrama) {CHE.mean():.3f} · caro {EXP.mean():.3f} bpb")

def net(route, sel, c):
    r = route[sel]; return -(np.where(r, EXP[sel], CHE[sel]) + c*r).mean(), r.mean()

def best_threshold_gate(sig, c):
    ts = np.quantile(sig[trS], np.linspace(0,1,60))
    best_e, best_t = -1e9, None
    for t in ts:
        e,_ = net(sig > t, trS, c)
        if e > best_e: best_e, best_t = e, t
    return best_t

print(f"\n{'c':>4} {'melhor-fixo':>11} {'gate BIGRAMA':>13} {'gate TRIGRAMA':>14} {'oráculo':>9}  gap-fechado(bi→tri)")
for c in [1.0, 2.0, 3.0]:
    e_cheap,_ = net(np.zeros(len(idx),bool), teS, c)
    e_exp,_   = net(np.ones(len(idx),bool), teS, c)
    fixed = max(e_cheap, e_exp)
    t1 = best_threshold_gate(E1, c); e1,u1 = net(E1 > t1, teS, c)
    t2 = best_threshold_gate(E2, c); e2,u2 = net(E2 > t2, teS, c)
    e_or,_ = net((CHE - EXP) > c, teS, c)
    g1 = (e1-fixed)/(e_or-fixed)*100 if e_or-fixed>1e-9 else 0
    g2 = (e2-fixed)/(e_or-fixed)*100 if e_or-fixed>1e-9 else 0
    print(f"{c:>4} {fixed:>11.3f} {e1:>10.3f}({u1*100:>3.0f}%) {e2:>10.3f}({u2*100:>3.0f}%) {e_or:>9.3f}   {g1:>4.0f}% → {g2:>4.0f}%")

print("\nNOTA HONESTA sobre 'especulativo' (draft+verify): verificar roda o modelo grande TODO passo,")
print("então no frame de ENERGIA (custo por chamada cara) especulativo == sempre-caro — economiza LATÊNCIA,")
print("não compute; NÃO fecha o gap do oráculo. O lever real de energia é o gate cheap de ordem maior (acima).")
