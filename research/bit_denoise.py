#!/usr/bin/env python3
"""BIT-DENOISE — melhora REAL no eixo do byte/bit: ROBUSTEZ. Treina um byte-model com AUMENTO POR CORRUPÇÃO
(bit-flips no contexto, tipo typos) vs baseline limpo, e mede bpb em contexto LIMPO e CORROMPIDO. Se o
denoise fica muito mais robusto (bpb-corrompido bem menor) a baixo custo no limpo → melhoria genuína.
2 sementes. GPU."""
import time, math, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda"; t0=time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}",flush=True)
DIM,LAYERS,HEADS,CTX,BS,STEPS=256,4,4,128,64,3500; SEEDS=[0,1]; PFLIP=0.10
data=np.memmap("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt",dtype=np.uint8,mode="r"); n=len(data); cut=int(n*0.999)
def get(split,seed=None):
    lo,hi=(0,cut) if split=="tr" else (cut,n-CTX-1)
    if seed is not None: np.random.seed(seed)
    ix=np.random.randint(lo,hi-CTX-1,BS); x=np.stack([np.asarray(data[i:i+CTX+1]) for i in ix]).astype(np.int64)
    t=torch.from_numpy(x).to(DEV); return t[:,:-1],t[:,1:]
def corrupt(x, p=PFLIP):   # flipa bit de CONTEÚDO (0-4) em p das posições = typo
    mask=(torch.rand_like(x,dtype=torch.float)<p)
    bit=torch.randint(0,5,x.shape,device=DEV)                 # bits de conteúdo
    return torch.where(mask, x ^ (1<<bit), x)
class Block(nn.Module):
    def __init__(s,d,h): super().__init__(); s.h=h; s.l1=nn.LayerNorm(d); s.l2=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.pr=nn.Linear(d,d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x):
        B,L,D=x.shape; hh=s.l1(x); q,k,v=s.qkv(hh).view(B,L,3,s.h,D//s.h).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True); x=x+s.pr(a.transpose(1,2).reshape(B,L,D)); return x+s.mlp(s.l2(x))
class M(nn.Module):
    def __init__(s): super().__init__(); s.tok=nn.Embedding(256,DIM); s.pos=nn.Embedding(CTX,DIM); s.blocks=nn.ModuleList([Block(DIM,HEADS) for _ in range(LAYERS)]); s.lnf=nn.LayerNorm(DIM); s.head=nn.Linear(DIM,256)
    def forward(s,x):
        pos=torch.arange(x.size(1),device=DEV); h=s.tok(x)+s.pos(pos)[None]
        for b in s.blocks: h=b(h)
        return s.head(s.lnf(h))
def train(denoise,seed):
    torch.manual_seed(seed); m=M().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.01); sc=torch.amp.GradScaler("cuda")
    np.random.seed(seed)
    for s in range(STEPS):
        xb,yb=get("tr"); xin=corrupt(xb) if denoise else xb
        with torch.amp.autocast("cuda"): loss=F.cross_entropy(m(xin).reshape(-1,256),yb.reshape(-1))
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    m.eval(); clean=corr=0.0
    with torch.no_grad():
        for i in range(20):
            xb,yb=get("va",seed=2000+i)
            clean+=F.cross_entropy(m(xb).reshape(-1,256),yb.reshape(-1)).item()
            torch.manual_seed(9+i); xc=corrupt(xb,0.10)
            corr+=F.cross_entropy(m(xc).reshape(-1,256),yb.reshape(-1)).item()
    return clean/20/math.log(2), corr/20/math.log(2)
res={}
for name,dn in [("baseline",False),("denoise",True)]:
    cs=[train(dn,sd) for sd in SEEDS]; cl=np.mean([c[0] for c in cs]); co=np.mean([c[1] for c in cs])
    res[name]={"bpb_limpo":round(cl,3),"bpb_corrompido":round(co,3),"seeds":[[round(a,2),round(b,2)] for a,b in cs]}
    log(f"  {name:9} limpo {cl:.3f} | corrompido(10% typo) {co:.3f}  (penalidade +{co-cl:.2f})")
json.dump(res,open("/home/leonardo/projects/LLM/bytebrain/research/bit_denoise.json","w"),indent=1)
dc=res["baseline"]["bpb_corrompido"]-res["denoise"]["bpb_corrompido"]
log(f"=== ROBUSTEZ: sob typo, baseline {res['baseline']['bpb_corrompido']} vs denoise {res['denoise']['bpb_corrompido']} bpb (denoise {dc:+.2f} melhor); custo no limpo: {res['denoise']['bpb_limpo']-res['baseline']['bpb_limpo']:+.3f} ===")
log("DONE")
