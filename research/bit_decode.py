#!/usr/bin/env python3
"""BIT-DECODE — explora o insight (modelo CRAVA estrutura, CHUTA conteúdo) na GERAÇÃO, sem re-treinar.
Decodifica ESTRUTURA (top-3 bits: caso/letra) com temperatura BAIXA (trava a gramática) e CONTEÚDO (low-5
bits) com temperatura mais alta (variedade). Compara com nucleus padrão em coesão (repetição/palavras).
Usa o MemByte na CPU."""
import sys, re, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,"/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV="cpu"
ck=torch.load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt",map_location=DEV,weights_only=False)
c=ck["config"]; M=ByteGPT(dim=c["dim"],n_layers=c["layers"],n_heads=c["heads"],context=c["ctx"]).to(DEV).eval(); M.load_state_dict(ck["model"])
CTX=c["ctx"]
struct_of=np.array([b>>5 for b in range(256)])       # top-3 bits (0-7)
S2B=[np.where(struct_of==s)[0] for s in range(8)]     # bytes de cada estrutura

@torch.no_grad()
def gen(prompt,mode,k=240,t_struct=0.35,t_cont=0.9,t_std=0.7):
    ids=list(prompt.encode()); x=torch.tensor([ids],device=DEV)
    for _ in range(k):
        lg=M(x[:,-CTX:])[0,-1].clone()
        for b in set(x[0,-40:].tolist()): lg[b]/=1.35
        if mode=="padrao":
            p=F.softmax(lg.float()/t_std,-1); nx=int(torch.multinomial(p,1))
        else:  # ciente-de-bit: estrutura fria, conteúdo quente
            p=F.softmax(lg.float(),-1).numpy()
            ps=np.array([p[S2B[s]].sum() for s in range(8)])            # P(estrutura)
            ps=ps**(1/t_struct); ps/=ps.sum(); s=int(np.random.choice(8,p=ps))   # amostra estrutura fria
            cb=S2B[s]; pc=p[cb]**(1/t_cont); pc/=pc.sum(); nx=int(cb[np.random.choice(len(cb),p=pc)])
        ids.append(nx); x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    return bytes(ids).decode("utf-8","ignore")
def rep4(t): g=[t[i:i+4] for i in range(len(t)-4)]; return round(1-len(set(g))/max(len(g),1),3)
def wordok(t):
    ws=re.findall(r"[a-zA-Zà-ü]+",t); ok=sum(1 for w in ws if len(w)>=3 and re.search(r"[aeiouáéíóú]",w)); return round(ok/max(len(ws),1),3)

np.random.seed(0)
for prompt in ["O Brasil é ","A ciência "]:
    for mode in ["padrao","ciente-de-bit"]:
        np.random.seed(7)
        t=gen(prompt,mode)
        print(f"[{mode:14}] rep4={rep4(t)} palavras-ok={wordok(t)}  {t[:100]!r}")
    print()
