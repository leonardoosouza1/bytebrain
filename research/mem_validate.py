#!/usr/bin/env python3
"""Valida o MemByte especializado: (1) gera no formato e PARA nativamente na fronteira \\n\\n; (2) planta
sementes SEM o truque do terminador e vê se o recall para sozinho (a fronteira é NATIVA agora); (3) custo
de semente (int4) e recall limpo; (4) armazena fato PRIVADO e recupera. GPU."""
import sys, time, math, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from src.model import ByteGPT
from wisdom_bridge import enc, dec, quant
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FMT = "P: {q}\nR:"

ck = torch.load("/home/leonardo/projects/LLM/bytebrain/research/membyte.pt", map_location=DEV, weights_only=False)
cfg = ck["cfg"]; m = ByteGPT(**cfg).to(DEV); m.load_state_dict(ck["model"]); m.eval()
for p in m.parameters(): p.requires_grad_(False)
EL = m.tok; DIM = EL.weight.shape[1]
log(f"MemByte carregado ({m.n_params/1e6:.1f}M, dim={DIM})")

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
def recall(seed, prompt, n=30, stop_nl2=True):
    X = torch.cat([seed, emb(enc(prompt))], 0)[None]; out=[]
    for _ in range(n):
        nx = int(m(inputs_embeds=X)[0,-1].argmax()); out.append(nx)
        X = torch.cat([X, EL(torch.tensor([[nx]], device=DEV))], 1)
        if stop_nl2 and len(out)>=2 and out[-1]==10 and out[-2]==10: break  # fronteira NATIVA \n\n
    return dec(out).strip()

@torch.no_grad()
def gen(prompt, n=40):
    x = torch.tensor([list(prompt.encode())], device=DEV)
    for _ in range(n):
        nx = int(m(x[:, -cfg["context"]:])[0,-1].argmax()); x = torch.cat([x, torch.tensor([[nx]],device=DEV)],1)
        if x.shape[1]>=2 and x[0,-2:].tolist()==[10,10]: break
    return bytes(x[0].tolist()).decode("utf-8","ignore")

log("== (1) formato nativo: gera e PARA na fronteira? ==")
for p in ["P: Qual o planeta natal de Orin?\nR:", "P: Qual o codigo de neônio?\nR:"]:
    log(f"  {gen(p)!r}")

# fatos PRIVADOS (fora do corpus de treino) — a memória guarda via semente
priv = [("Qual o codigo do cofre da IARA?","7492"),("Qual o planeta natal do Zephyr?","Krylon"),
        ("Quem e o guardiao da torre de Vaelis?","Orin"),("Qual a senha do laboratorio 7?","girassol"),
        ("Qual a cor do dragao de Miro?","carmesim")]
log("== (2) semente SEM terminador + recall com parada NATIVA \\n\\n (a fronteira é do modelo agora) ==")
ok=oki4=0
for q,a in priv:
    s = plant(FMT.format(q=q), " "+a)          # SEM \n no alvo — a fronteira vem do modelo treinado
    r = recall(s, FMT.format(q=q)); h = a.lower() in r.lower()
    ri4 = recall(quant(s,4), FMT.format(q=q)); hi4 = a.lower() in ri4.lower()
    ok+=h; oki4+=hi4
    log(f"  {a:10} recall={'HIT' if h else 'no '} int4={'HIT' if hi4 else 'no '} | {r[:26]!r}")
kb = 8*DIM*2/1024
log(f"== RESULTADO MemByte: fp16 {ok}/{len(priv)} | int4 {oki4}/{len(priv)} | semente {kb:.1f}KB fp16 / {kb/4:.1f}KB int4 ==")
log(f"DONE ({time.time()-t0:.0f}s)")
