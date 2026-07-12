#!/usr/bin/env python3
"""BATERIA 9 — ROTEADOR EVOLUÍDO (survival-as-routing, tese IARA-Router). Segue a bat.8: o limiar na incerteza
barata era fraco. Aqui a SOBREVIVÊNCIA (GA) evolui uma política LINEAR sobre várias features BARATAS pre-outcome
(entropia do bigrama, classe do byte anterior, incerteza local...) — route pro órgão caro se score>0. Fitness =
energia (−bits−c·uso) no TRAIN; mede no TEST. Quanto do gap do ORÁCULO uma política evoluída fecha? Honesto. GPU."""
import sys, math
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
import numpy as np, torch, torch.nn.functional as F
from wisdom_bridge import load_byte_model, DEV

m = load_byte_model(trained=True)
path = "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt"
with open(path, "rb") as f:
    f.seek(5_050_000); data = f.read(40000)
N = len(data); W = 256

exp = np.full(N, np.nan)
with torch.no_grad():
    for s in range(0, N-1, W):
        e = min(s+W, N); L = e-s
        if L < 2: break
        ids = torch.tensor([list(data[s:e])], device=DEV)
        lp = F.log_softmax(m(inputs_embeds=m.tok(ids))[0][:-1].float(), -1)
        b = (-(lp[torch.arange(L-1, device=DEV), ids[0,1:]])/0.69314718).cpu().numpy()
        exp[s+1:s+1+len(b)] = b

ones = np.ones((256,8)); tot = np.full(256,2.0)
che = np.full(N, np.nan); ent = np.full(N, np.nan)
for i in range(1, N):
    prev = data[i-1]; b = data[i]
    p = np.clip(ones[prev]/tot[prev], 1e-6, 1-1e-6)
    a = np.array([(b>>k)&1 for k in range(8)], float)
    che[i] = float(np.sum(-(a*np.log2(p)+(1-a)*np.log2(1-p))))
    ent[i] = float(np.sum(-(p*np.log2(p)+(1-p)*np.log2(1-p))))
    ones[prev] += a; tot[prev] += 1

def cls(byte):  # classe barata do byte anterior
    c = chr(byte) if byte < 128 else ""
    return (byte==0x20, c.isalpha(), c.isdigit())

idx = np.array([i for i in range(2, N) if not np.isnan(exp[i])])
# features pre-outcome (só dependem do passado / da predição barata, nunca do byte atual)
def feats(i):
    sp, al, di = cls(data[i-1])
    loc = np.nanmean(ent[max(1,i-8):i])                     # incerteza local recente
    return [ (ent[i]-3.0)/1.0, float(sp), float(al), float(di), (loc-3.0)/1.0, (ent[i-1]-3.0)/1.0, 1.0 ]
Xall = np.array([feats(i) for i in idx]); Fdim = Xall.shape[1]
EXP = exp[idx]; CHE = che[idx]
half = len(idx)//2
trS, teS = slice(0,half), slice(half,None)

def net_vec(route, sel, c):
    r = route[sel]; bits = np.where(r, EXP[sel], CHE[sel]) + c*r
    return -bits.mean(), r.mean()

rng = np.random.default_rng(7)
def evolve(c, pop=80, gens=80):
    G = rng.normal(0,0.6,(pop,Fdim))
    best=None; bestf=-1e9
    for _ in range(gens):
        fits=[]
        for w in G:
            r = (Xall @ w) > 0
            e,_ = net_vec(r, trS, c); fits.append(e)
        fits=np.array(fits); order=np.argsort(-fits)
        if fits[order[0]]>bestf: bestf=fits[order[0]]; best=G[order[0]].copy()
        el=G[order[:16]]; nxt=[el[k] for k in range(16)]
        while len(nxt)<pop:
            a,b=el[rng.integers(16)],el[rng.integers(16)]
            ch=np.where(rng.random(Fdim)<0.5,a,b)+rng.normal(0,0.15,Fdim); nxt.append(ch)
        G=np.array(nxt)
    return best

print(f"held-out {N}B · {len(idx)} posições · barato {CHE.mean():.3f} · caro {EXP.mean():.3f} bpb\n")
print(f"{'c':>4} {'sempre-caro':>12} {'roteador-EVO':>13} {'uso%':>6} {'ORÁCULO':>10} {'gap fechado':>11}")
for c in [0.5, 1.0, 2.0, 3.0]:
    w = evolve(c)
    r_evo = (Xall @ w) > 0
    e_evo, u_evo = net_vec(r_evo, teS, c)
    e_cheap,_ = net_vec(np.zeros(len(idx),bool), teS, c)
    e_exp,_   = net_vec(np.ones(len(idx),bool), teS, c)
    r_or = (CHE - EXP) > c; e_or,_ = net_vec(r_or, teS, c)
    fixed_best = max(e_cheap, e_exp)
    closed = (e_evo - fixed_best)/(e_or - fixed_best) if (e_or-fixed_best)>1e-9 else 0.0
    print(f"{c:>4} {e_exp:>12.3f} {e_evo:>13.3f} {u_evo*100:>5.1f}% {e_or:>10.3f} {closed*100:>10.1f}%")
print("\n→ 'gap fechado' = quanto o roteador EVOLUÍDO cobre da distância (melhor-fixo → oráculo).")
print("  >0 e alto = a sobrevivência descobre roteamento útil (ponte IARA-Router); ~0 = features baratas insuficientes.")
