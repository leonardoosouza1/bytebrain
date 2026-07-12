#!/usr/bin/env python3
"""BIT-VALIDATE — o ganho de coesão (bitembed_add/auxstruct) é REAL ou ruído? Treina baseline vs auxstruct
vs bitembed_add com MÚLTIPLAS sementes, gera VÁRIAS amostras cada, e reporta média ± desvio da coesão
(repetição/palavras-ok). Se o efeito for consistente entre sementes → real. GPU."""
import time, math, json, re
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda"; t0=time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}",flush=True)
DIM,LAYERS,HEADS,CTX,BS,STEPS=256,4,4,128,64,3500
SEEDS=[0,1,2]; GENS=6
data=np.memmap("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt",dtype=np.uint8,mode="r"); n=len(data); cut=int(n*0.999)
def get(seed=None):
    if seed is not None: np.random.seed(seed)
    ix=np.random.randint(0,cut-CTX-1,BS); x=np.stack([np.asarray(data[i:i+CTX+1]) for i in ix]).astype(np.int64)
    t=torch.from_numpy(x).to(DEV); return t[:,:-1],t[:,1:]
class Block(nn.Module):
    def __init__(s,d,h): super().__init__(); s.h=h; s.l1=nn.LayerNorm(d); s.l2=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.pr=nn.Linear(d,d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x):
        B,L,D=x.shape; hh=s.l1(x); q,k,v=s.qkv(hh).view(B,L,3,s.h,D//s.h).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True); x=x+s.pr(a.transpose(1,2).reshape(B,L,D)); return x+s.mlp(s.l2(x))
class M(nn.Module):
    def __init__(s,mode):
        super().__init__(); s.mode=mode; s.tok=nn.Embedding(256,DIM); s.pos=nn.Embedding(CTX,DIM)
        if mode=="bitembed_add": s.bitvec=nn.Parameter(torch.randn(8,DIM)*0.02)
        s.blocks=nn.ModuleList([Block(DIM,HEADS) for _ in range(LAYERS)]); s.lnf=nn.LayerNorm(DIM); s.head=nn.Linear(DIM,256)
        if mode=="auxstruct": s.aux=nn.Linear(DIM,3)
    def forward(s,x):
        pos=torch.arange(x.size(1),device=DEV); h=s.tok(x)+s.pos(pos)[None]
        if s.mode=="bitembed_add": h=h+(((x.unsqueeze(-1)>>torch.arange(8,device=DEV))&1).float())@s.bitvec
        for b in s.blocks: h=b(h)
        h=s.lnf(h); return (s.head(h),s.aux(h)) if s.mode=="auxstruct" else s.head(h)
def rep4(t): g=[t[i:i+4] for i in range(len(t)-4)]; return 1-len(set(g))/max(len(g),1)
def wok(t):
    ws=re.findall(r"[a-zA-Zà-ü]+",t); ok=sum(1 for w in ws if len(w)>=3 and re.search(r"[aeiouáéíóú]",w)); return ok/max(len(ws),1)
@torch.no_grad()
def gen(m,gseed,prompt="O Brasil é ",k=200):
    torch.manual_seed(gseed); m.eval(); ids=list(prompt.encode()); x=torch.tensor([ids],device=DEV)
    for _ in range(k):
        lg=(m(x[:,-CTX:])[0] if m.mode=="auxstruct" else m(x[:,-CTX:]))[0,-1].clone()
        for b in set(x[0,-40:].tolist()): lg[b]/=1.3
        p=F.softmax(lg.float()/0.7,-1); sp,si=torch.sort(p,descending=True); keep=torch.cumsum(sp,0)<=0.9; keep[0]=True
        sp=sp*keep; nx=int(si[torch.multinomial(sp/sp.sum(),1)]); ids.append(nx); x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    m.train(); return bytes(ids).decode("utf-8","ignore")
def run(mode,seed):
    torch.manual_seed(seed); m=M(mode).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.01); sc=torch.amp.GradScaler("cuda")
    np.random.seed(seed)
    for s in range(STEPS):
        xb,yb=get()
        with torch.amp.autocast("cuda"):
            if mode=="auxstruct":
                lg,aux=m(xb); ce=F.cross_entropy(lg.reshape(-1,256),yb.reshape(-1)); tb=torch.stack([((yb>>k)&1).float() for k in (7,6,5)],-1)
                loss=ce+0.3*F.binary_cross_entropy_with_logits(aux.reshape(-1,3),tb.reshape(-1,3))
            else: loss=F.cross_entropy(m(xb).reshape(-1,256),yb.reshape(-1))
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    reps=[]; woks=[]
    for g in range(GENS):
        t=gen(m,100+g); reps.append(rep4(t)); woks.append(wok(t))
    return np.mean(reps),np.mean(woks)
res={}
for mode in ["baseline","bitembed_add","auxstruct"]:
    r=[run(mode,sd) for sd in SEEDS]; reps=[x[0] for x in r]; woks=[x[1] for x in r]
    res[mode]={"rep4":[round(x,3) for x in reps],"word_ok":[round(x,3) for x in woks],
               "rep4_media":round(np.mean(reps),3),"word_media":round(np.mean(woks),3),"word_std":round(np.std(woks),3)}
    log(f"  {mode:13} word-ok {np.mean(woks):.3f}±{np.std(woks):.3f} {[round(x,2) for x in woks]} | rep4 {np.mean(reps):.3f} {[round(x,2) for x in reps]}")
json.dump(res,open("/home/leonardo/projects/LLM/bytebrain/research/bit_validate.json","w"),indent=1)
b=res["baseline"]["word_media"]
log(f"=== word-ok: baseline {b} | bitembed_add {res['bitembed_add']['word_media']} (+{res['bitembed_add']['word_media']-b:.3f}) | auxstruct {res['auxstruct']['word_media']} (+{res['auxstruct']['word_media']-b:.3f}) ===")
log("DONE")
