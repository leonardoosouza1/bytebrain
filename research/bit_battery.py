#!/usr/bin/env python3
"""BIT-BATTERY — testa se a ESTRUTURA DE BITS melhora o byte-model (a visão do Leonardo). A/B controlado,
mesmo orçamento (mesmo tamanho/dados/passos), compara val bits/byte:
  baseline  : byte->Embedding(256), softmax 256
  bitembed  : byte = base + soma dos vetores dos bits LIGADOS (estrutura explícita na entrada)
  auxstruct : baseline + cabeça auxiliar prevendo os bits 7,6,5 (força a coesão estrutural)
  factored  : prevê ESTRUTURA (top3 bits, 8-way) + CONTEÚDO (low5, 32-way) em cabeças separadas
GPU. Dump bit_battery.json."""
import sys, time, math, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
DIM, LAYERS, HEADS, CTX, BS, STEPS = 256, 4, 4, 128, 64, 3000
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt"
data = np.memmap(CORPUS, dtype=np.uint8, mode="r"); n = len(data); cut = int(n*0.999)
def get(split, seed=None):
    lo, hi = (0, cut) if split=="tr" else (cut, n-CTX-1)
    if seed is not None: np.random.seed(seed)
    ix = np.random.randint(lo, hi-CTX-1, BS)
    x = np.stack([np.asarray(data[i:i+CTX+1]) for i in ix]).astype(np.int64)
    t = torch.from_numpy(x).to(DEV); return t[:,:-1], t[:,1:]

class Block(nn.Module):
    def __init__(s,d,h): super().__init__(); s.h=h; s.l1=nn.LayerNorm(d); s.l2=nn.LayerNorm(d); s.qkv=nn.Linear(d,3*d); s.pr=nn.Linear(d,d); s.mlp=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(s,x):
        B,L,D=x.shape; hh=s.l1(x); q,k,v=s.qkv(hh).view(B,L,3,s.h,D//s.h).permute(2,0,3,1,4)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True); x=x+s.pr(a.transpose(1,2).reshape(B,L,D)); return x+s.mlp(s.l2(x))

class M(nn.Module):
    def __init__(s, mode):
        super().__init__(); s.mode=mode
        if mode=="bitembed":
            s.base=nn.Parameter(torch.zeros(DIM)); s.bitvec=nn.Parameter(torch.randn(8,DIM)*0.02)
        else: s.tok=nn.Embedding(256,DIM)
        s.pos=nn.Embedding(CTX,DIM); s.blocks=nn.ModuleList([Block(DIM,HEADS) for _ in range(LAYERS)]); s.lnf=nn.LayerNorm(DIM)
        if mode=="factored": s.hs=nn.Linear(DIM,8); s.hc=nn.Linear(DIM,32)
        else: s.head=nn.Linear(DIM,256)
        if mode=="auxstruct": s.aux=nn.Linear(DIM,3)
    def emb(s,x):
        if s.mode=="bitembed":
            bits=((x.unsqueeze(-1)>>torch.arange(8,device=DEV))&1).float()   # [B,L,8]
            return s.base + bits@s.bitvec
        return s.tok(x)
    def forward(s,x):
        pos=torch.arange(x.size(1),device=DEV); h=s.emb(x)+s.pos(pos)[None]
        for b in s.blocks: h=b(h)
        h=s.lnf(h)
        if s.mode=="factored": return s.hs(h), s.hc(h)
        return (s.head(h), s.aux(h)) if s.mode=="auxstruct" else s.head(h)

def loss_of(model, xb, yb):
    if model.mode=="factored":
        ls,lc=model(xb); ys=(yb>>5); yc=(yb&31)
        return F.cross_entropy(ls.reshape(-1,8),ys.reshape(-1))+F.cross_entropy(lc.reshape(-1,32),yc.reshape(-1))
    if model.mode=="auxstruct":
        lg,aux=model(xb); ce=F.cross_entropy(lg.reshape(-1,256),yb.reshape(-1))
        tb=torch.stack([((yb>>k)&1).float() for k in (7,6,5)],-1)   # bits estruturais
        return ce + 0.3*F.binary_cross_entropy_with_logits(aux.reshape(-1,3), tb.reshape(-1,3))
    return F.cross_entropy(model(xb).reshape(-1,256), yb.reshape(-1))

@torch.no_grad()
def val_bpb(model):
    model.eval(); tot=0.0
    for i in range(15):
        xb,yb=get("va", seed=1000+i)
        if model.mode=="factored":
            ls,lc=model(xb); ys=(yb>>5); yc=(yb&31)
            nll=F.cross_entropy(ls.reshape(-1,8),ys.reshape(-1))+F.cross_entropy(lc.reshape(-1,32),yc.reshape(-1))
        elif model.mode=="auxstruct":
            nll=F.cross_entropy(model(xb)[0].reshape(-1,256),yb.reshape(-1))
        else:
            nll=F.cross_entropy(model(xb).reshape(-1,256),yb.reshape(-1))
        tot+=nll.item()
    model.train(); return tot/15/math.log(2)

res={}
for mode in ["baseline","bitembed","auxstruct","factored"]:
    torch.manual_seed(42); np.random.seed(42)
    m=M(mode).to(DEV); opt=torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01); sc=torch.amp.GradScaler("cuda")
    np.random.seed(0)  # mesma ordem de dados p/ todos
    for s in range(STEPS):
        xb,yb=get("tr")
        with torch.amp.autocast("cuda"): loss=loss_of(m,xb,yb)
        opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
    vb=val_bpb(m); res[mode]={"val_bpb":round(vb,4),"params":sum(p.numel() for p in m.parameters())}
    log(f"  {mode:10} val {vb:.4f} bpb  ({res[mode]['params']/1e6:.1f}M)")
base=res["baseline"]["val_bpb"]
for k in res: res[k]["delta_vs_base"]=round(res[k]["val_bpb"]-base,4)
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/bit_battery.json","w"), indent=1)
best=min(res, key=lambda k:res[k]["val_bpb"])
log(f"=== MELHOR: {best} ({res[best]['val_bpb']} bpb, {res[best]['delta_vs_base']:+.4f} vs baseline) ===")
log("DONE")
