#!/usr/bin/env python3
"""Valida o MemByte-real (treinado em Wikipedia PT, priors AMPLOS). Testa recall LIMPO (terminador \\n +
stop) p/ fatos com palavras COMUNS e RARAS/fora-do-vocab-sintético (carmesim, químera, etc.) — se recuperar
limpo até as raras, os priors amplos curaram o gap. Custo de semente. GPU."""
import sys, time, math, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from src.model import ByteGPT
from wisdom_bridge import enc, dec, quant
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FMT = "P: {q}\nR:"; NL = 10

import os
CK = "/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt"
if not os.path.exists(CK): CK = "/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt.pt"
ck = torch.load(CK, map_location=DEV, weights_only=False); c = ck["config"]
m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV)
m.load_state_dict(ck["model"]); m.eval()
for p in m.parameters(): p.requires_grad_(False)
EL = m.tok; DIM = EL.weight.shape[1]
log(f"MemByte-real carregado ({m.n_params/1e6:.1f}M, dim={DIM}, step {ck['step']}, val {ck.get('best_val',0):.3f} bpb)")

def emb(ids): return EL(torch.tensor([ids], device=DEV)).detach()[0]
def plant(prompt, target, K=8, steps=350, lr=0.1):
    pe = emb(enc(prompt)); te = emb(enc(target)); tids = enc(target)
    L = K+pe.shape[0]+te.shape[0]; E = torch.zeros(1, L, DIM, device=DEV)
    E[0, K:K+pe.shape[0]] = pe; E[0, K+pe.shape[0]:L] = te
    Pp = torch.tensor([K+pe.shape[0]+i-1 for i in range(len(tids))], device=DEV); T = torch.tensor(tids, device=DEV)
    seed = nn.Parameter(torch.randn(K, DIM, device=DEV)*0.1); opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr*0.5*(1+math.cos(math.pi*s/steps))
        X = E.clone(); X[0,:K] = seed
        loss = F.cross_entropy(m(inputs_embeds=X)[0][Pp], T)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed],1.0); opt.step()
    return seed.detach()
@torch.no_grad()
def recall(seed, prompt, n=30):
    X = torch.cat([seed, emb(enc(prompt))], 0)[None]; out=[]
    for _ in range(n):
        nx = int(m(inputs_embeds=X)[0,-1].argmax())
        if nx == NL: break
        out.append(nx); X = torch.cat([X, EL(torch.tensor([[nx]], device=DEV))], 1)
    return dec(out).strip()

# fatos com palavras COMUNS e RARAS (as raras eram o gap do corpus sintético)
facts = [("Qual o codigo do cofre?","7492"),("Qual o planeta natal do Zephyr?","Krylon"),
         ("Qual a cor do dragao de Miro?","carmesim"),("Qual a criatura da caverna?","quimera"),
         ("Qual o mineral do templo?","malaquita"),("Qual o nome do navegador?","Vasco da Gama"),
         ("Qual a substancia rara?","paládio"),("Qual o codinome do agente?","Peregrino")]
log("== recall LIMPO (terminador \\n + stop) — inclui palavras RARAS/fora-do-sintético ==")
ok=oki4=0
for q,a in facts:
    s = plant(FMT.format(q=q), " "+a+"\n")
    r = recall(s, FMT.format(q=q)); h = a.lower() in r.lower()
    ri4 = recall(quant(s,4), FMT.format(q=q)); hi4 = a.lower() in ri4.lower()
    ok+=h; oki4+=hi4
    log(f"  {a:16} recall={'HIT' if h else 'no '} int4={'HIT' if hi4 else 'no '} | {r[:28]!r}")
kb = 8*DIM*2/1024
log(f"== MemByte-real: fp16 {ok}/{len(facts)} | int4 {oki4}/{len(facts)} | semente {kb:.1f}KB / {kb/4:.1f}KB int4 ==")
log(f"DONE ({time.time()-t0:.0f}s)")
