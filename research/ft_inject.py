#!/usr/bin/env python3
"""FT-INJECT — injeta conhecimento no gerador FLUENTE (ckpt_gen) via fine-tune num corpus factual destilado
do Qwen (limpo, chat) MISTURADO com Wikipedia (mantém fluência) + denoise 5% (robustez). Salva e re-proba.
Prova: o substrato byte ABSORVE conhecimento (0/7→N/7)? E herda os erros do professor?"""
import sys, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,"/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV="cuda"; t0=__import__("time").time()
def log(m): print(f"[{__import__('time').time()-t0:6.0f}s] {m}",flush=True)
CK="/home/leonardo/projects/LLM/bytebrain/research/ckpt_gen/ckpt_best.pt"
ck=torch.load(CK,map_location=DEV,weights_only=False); c=ck["config"]; CTX=c["ctx"]
m=ByteGPT(dim=c["dim"],n_layers=c["layers"],n_heads=c["heads"],context=CTX).to(DEV)
m.load_state_dict(ck["model"]); log(f"gerador {m.n_params/1e6:.0f}M carregado (val {ck['best_val']:.3f})")
# corpus de fatos (pequeno) tiled num buffer grande; Wikipedia via memmap
facts=open("/home/leonardo/projects/LLM/bytebrain/data/qwen_facts.txt","rb").read()
fbuf=np.frombuffer((facts+b"\n")*400,dtype=np.uint8); log(f"fatos: {len(facts)}B tiled->{len(fbuf)}B")
wiki=np.memmap("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt",dtype=np.uint8,mode="r"); nw=len(wiki)
BS,STEPS,PF,PFLIP=16,2200,0.45,0.05; CTX=min(CTX,384)   # janela menor: fatos são curtos, cabe em 12GB
def corrupt(x):
    mask=(torch.rand_like(x,dtype=torch.float)<PFLIP); bit=torch.randint(0,5,x.shape,device=DEV)
    return torch.where(mask, x^(1<<bit), x)
def batch():
    rows=[]
    for _ in range(BS):
        if np.random.rand()<PF: i=np.random.randint(0,len(fbuf)-CTX-1); rows.append(np.asarray(fbuf[i:i+CTX+1]))
        else: i=np.random.randint(0,nw-CTX-1); rows.append(np.asarray(wiki[i:i+CTX+1]))
    x=torch.from_numpy(np.stack(rows).astype(np.int64)).to(DEV); return x[:,:-1],x[:,1:]
opt=torch.optim.AdamW(m.parameters(),lr=5e-4,weight_decay=0.01); sc=torch.amp.GradScaler("cuda")
m.train()
for s in range(STEPS):
    xb,yb=batch(); xin=corrupt(xb)
    with torch.amp.autocast("cuda"): loss=F.cross_entropy(m(xin).reshape(-1,256),yb.reshape(-1))
    opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    if s%400==0: log(f"  step {s}  loss {loss.item()/math.log(2):.3f} bpb")
OUT="/home/leonardo/projects/LLM/bytebrain/research/ckpt_gen_facts"
import os; os.makedirs(OUT,exist_ok=True)
torch.save({"model":m.state_dict(),"config":c,"step":ck["step"]+STEPS,"best_val":0.0},f"{OUT}/ckpt_best.pt")
log(f"salvo -> {OUT}/ckpt_best.pt  |  agora rode: python3 intel_probe.py {OUT}/ckpt_best.pt")
log("DONE")
