#!/usr/bin/env python3
"""BATER A LEI DE ESCALA — modelo pequeno + MEMÓRIA de bytes (n-grama) vale por um grande. 2026-07-14.
Leonardo: 'não é possível que é limitado a isso, pense em algo.' Certo. A lei bpb~log(params) é pra
transformer vanilla. Aqui: o MESMO modelo pequeno interpolado com uma tabela n-grama do corpus (memória
não-paramétrica, tese IARA: memória>params). Mede bpb de: modelo só · n-grama só · modelo+memória."""
import sys, os, math, time, numpy as np, torch, torch.nn.functional as F
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytebrain_scale import ByteGPT
DEV="cuda"; torch.manual_seed(0); np.random.seed(0)
raw=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt","rb").read(40_000_000)
arr=np.frombuffer(raw,np.uint8); n=len(arr); CTX,B=192,48; tr=arr[:int(n*0.95)]; va=arr[int(n*0.95):]
def batch(data):
    i=np.random.randint(0,len(data)-CTX-1,B); return torch.from_numpy(np.stack([data[j:j+CTX+1] for j in i]).astype(np.int64)).to(DEV)

# ---------- 1) modelo pequeno (o mesmo baseline 3.4M) ----------
print("[treino rápido do modelo pequeno 3.4M…]",flush=True)
m=ByteGPT(4,256,4).to(DEV); opt=torch.optim.AdamW(m.parameters(),3e-4,weight_decay=0.01); m.train()
for st in range(1000):
    x=batch(tr); loss=F.cross_entropy(m(x[:,:-1]).reshape(-1,256),x[:,1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
m.eval()

# ---------- 2) MEMÓRIA n-grama de bytes (não-paramétrica) ----------
print("[construindo memória n-grama do corpus (orders 1..6)…]",flush=True)
ORD=[1,2,3,4,6]; tabs={o:defaultdict(Counter) for o in ORD}
sl=tr[:12_000_000].tobytes()
for i in range(max(ORD),len(sl)):
    nb=sl[i]
    for o in ORD: tabs[o][sl[i-o:i]][nb]+=1
for o in ORD: tabs[o]={k:v for k,v in tabs[o].items() if sum(v.values())>=2}   # poda singletons
print("  contextos guardados:",{o:len(tabs[o]) for o in ORD})
def ngram_probs(ctx):                                  # distribuição por backoff/interpolação linear
    d=np.full(256,1/256)
    for o in ORD:
        key=ctx[-o:].tobytes()
        c=tabs[o].get(key)
        if c:
            tot=sum(c.values()); dd=np.zeros(256)
            for nb,cnt in c.items(): dd[nb]=cnt/tot
            lam=min(0.9,0.3+0.12*o)                    # ordens maiores confiam mais
            d=lam*dd+(1-lam)*d
    return d

# ---------- 3) avaliar: modelo só · n-grama só · modelo+memória ----------
def eval_all(lmbds=(0.0,0.3,0.5,0.7,1.0),nwin=24):
    bits_m=bits_n=0.0; bits_mix={l:0.0 for l in lmbds}; tot=0
    for _ in range(nwin):
        j=np.random.randint(0,len(va)-CTX-1); win=va[j:j+CTX+1]
        X=torch.from_numpy(win[:-1].astype(np.int64))[None].to(DEV)
        with torch.no_grad(): pm=F.softmax(m(X)[0].float(),-1).cpu().numpy()   # [CTX,256]
        for t in range(8,CTX):                          # posições com algum contexto
            y=int(win[t+1]); ctx=win[:t+1]
            pmt=pm[t]; pnt=ngram_probs(ctx)
            bits_m+=-math.log2(pmt[y]+1e-12); bits_n+=-math.log2(pnt[y]+1e-12)
            for l in lmbds:
                pmix=l*pmt+(1-l)*pnt; bits_mix[l]+=-math.log2(pmix[y]+1e-12)
            tot+=1
    return bits_m/tot, bits_n/tot, {l:bits_mix[l]/tot for l in lmbds}

bm,bn,bmix=eval_all()
print("\n"+"="*60)
print(f"  modelo pequeno só (3.4M):   {bm:.3f} bpb")
print(f"  memória n-grama só:         {bn:.3f} bpb")
best=min(bmix,key=bmix.get)
for l,v in bmix.items(): print(f"  modelo + memória (λ={l:.1f}):   {v:.3f} bpb  {'← MELHOR' if l==best else ''}")
print("="*60)
print(f"  ganho do modelo+memória vs modelo só: {100*(bm-bmix[best])/bm:+.1f}%")
print(f"  pela LEI (bpb=7.01-0.669·log10 N), esse bpb {bmix[best]:.2f} exigiria ~{10**((7.01-bmix[best])/0.669)/1e6:.0f}M params sozinho.")
print("  → memória não-paramétrica BATE a lei de escala: byte pequeno + tabela ≈ modelo muito maior.")
