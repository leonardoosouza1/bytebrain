#!/usr/bin/env python3
"""A/B — as descobertas do Qwen melhoram a ARQUITETURA do ByteBrain? 2026-07-14.
Reusa a ByteGPT do PROJETO (bytebrain_scale.py, agora com flags emb='bits' e lowrank).
Mesmo corpus, mesmo orçamento, muda só a arquitetura. Mede bpb de validação (coerência=compressão)."""
import sys, os, math, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytebrain_scale import ByteGPT                 # <-- a arquitetura DO PROJETO (upgradada)
DEV="cuda"; torch.manual_seed(0); np.random.seed(0)
CORPUS="/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt"
raw=open(CORPUS,"rb").read(40_000_000); arr=np.frombuffer(raw,np.uint8); n=len(arr)
CTX,B=192,48; tr=arr[:int(n*0.95)]; va=arr[int(n*0.95):]
print(f"corpus {n/1e6:.0f}MB PT · ctx={CTX} B={B} · aleatorio=8.00 bpb  (arquitetura = ByteGPT do projeto)")
def batch(data):
    i=np.random.randint(0,len(data)-CTX-1,B)
    return torch.from_numpy(np.stack([data[j:j+CTX+1] for j in i]).astype(np.int64)).to(DEV)
@torch.no_grad()
def bpb(m,iters=60):
    m.eval(); t=0.0
    for _ in range(iters):
        x=batch(va); lo=m(x[:,:-1]); t+=F.cross_entropy(lo.reshape(-1,256),x[:,1:].reshape(-1)).item()
    m.train(); return t/iters/math.log(2)            # vocab=byte -> nats/ln2 = bpb
def run(nome,emb,lowrank,steps=1600):
    m=ByteGPT(4,256,4,emb=emb,lowrank=lowrank).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0.01)
    npar=sum(p.numel() for p in m.parameters())/1e6; t0=time.time(); m.train()
    print(f"\n### {nome}  ({npar:.1f}M params)")
    for st in range(steps+1):
        x=batch(tr); lo=m(x[:,:-1]); loss=F.cross_entropy(lo.reshape(-1,256),x[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st%400==0: print(f"  passo {st:>4}  bpb_val {bpb(m):.3f}   ({time.time()-t0:.0f}s)")
    b=bpb(m,140); print(f"  -> FINAL bpb_val {b:.3f}"); return b,npar
print("\n"+"="*64+"\n# A/B: descobertas do Qwen na arquitetura do ByteBrain\n"+"="*64)
R={}
R["baseline (emb livre, attn cheia)"]=run("BASELINE","free",False)
R["augment (livre + prior de bit)"]=run("AUGMENT (livre + bits somados)","augment",False)
R["+bits (substitui por bits)"]=run("+BITS (substitui)","bits",False)
R["+bits +lowrank-attn"]=run("+BITS +LOWRANK","bits",True)
base=R["baseline (emb livre, attn cheia)"][0]
print("\n"+"="*64+"\nPLACAR bpb (menor = mais coerente/comprimido):")
for k,(b,p) in R.items():
    print(f"  {k:34} {b:.3f}  {p:4.1f}M  {'(baseline)' if b==base else f'{100*(base-b)/base:+.1f}%'}")
print("="*64)
