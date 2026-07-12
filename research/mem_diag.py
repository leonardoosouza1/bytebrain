#!/usr/bin/env python3
"""Diagnóstico: o ruído do recall do 8M é falta de FRONTEIRA de fim-de-resposta? Planto com terminador \n
e paro o recall no \n (byte 10). Se limpar -> a cura é treinar um modelo com fronteiras (não é bug de código)."""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F, math
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from wisdom_bridge import load_byte_model, enc, dec, quant
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FMT = "P: {q}\nR:"; STOP = 10  # newline

m = load_byte_model(trained=True)
EL = m.tok
def emb(ids): return EL(torch.tensor([ids], device=DEV)).detach()[0]

def plant(prompt, target, K=8, steps=400, lr=0.1):
    pe = emb(enc(prompt)); te = emb(enc(target)); tids = enc(target)
    L = K + pe.shape[0] + te.shape[0]; E = torch.zeros(1, L, EL.weight.shape[1], device=DEV)
    E[0, K:K+pe.shape[0]] = pe; E[0, K+pe.shape[0]:L] = te
    Pp = torch.tensor([K+pe.shape[0]+i-1 for i in range(len(tids))], device=DEV); T = torch.tensor(tids, device=DEV)
    seed = nn.Parameter(torch.randn(K, EL.weight.shape[1], device=DEV)*0.1); opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr*0.5*(1+math.cos(math.pi*s/steps))
        X = E.clone(); X[0,:K] = seed
        loss = F.cross_entropy(m(inputs_embeds=X)[0][Pp], T)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed],1.0); opt.step()
    return seed.detach()

@torch.no_grad()
def recall(seed, prompt, n=24, stop_at=None):
    X = torch.cat([seed, emb(enc(prompt))], 0)[None]; out = []
    for _ in range(n):
        nx = int(m(inputs_embeds=X)[0,-1].argmax())
        if stop_at is not None and nx == stop_at: break
        out.append(nx); X = torch.cat([X, EL(torch.tensor([[nx]], device=DEV))], 1)
    return dec(out)

facts = [("Qual o codigo do cofre da IARA?","7492"),("Qual o planeta natal do Zephyr?","Krylon"),
         ("Quem e o guardiao da torre de Vaelis?","Orin"),("Qual a senha do laboratorio 7?","girassol")]
log("== SEM terminador (atual) vs COM terminador \\n + stop ==")
for q,a in facts:
    s1 = plant(FMT.format(q=q), " "+a)                 # sem terminador
    r1 = recall(quant(s1,4), FMT.format(q=q))
    s2 = plant(FMT.format(q=q), " "+a+"\n")             # COM terminador \n
    r2 = recall(quant(s2,4), FMT.format(q=q), stop_at=STOP)
    log(f"  {a:10} | sem-term: {r1[:24]!r:28} | com-term+stop: {r2[:24]!r}")
log(f"DONE ({time.time()-t0:.0f}s)")
