#!/usr/bin/env python3
"""GERAÇÃO = GRAFO + SEMENTES + B-SPLINE — pensar longe. 2026-07-14.
Leonardo: junta geração e grafos, usa sementes e B-splines. O grafo do corpus dá fluência LOCAL mas
deriva no global. Fix: as SEMENTES (temas-âncora) viram PONTOS DE CONTROLE de uma B-SPLINE; a geração
caminha grafos temáticos MISTURADOS pela curva → o texto flui suave de um tema ao próximo, fluente E
com rumo. Cada semente = um grafo local do corpus (janelas ao redor dela). Sem LLM, sem gradiente."""
import numpy as np, math, random
from collections import defaultdict, Counter
random.seed(0); np.random.seed(0)
raw=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt","rb").read(40_000_000); C=raw
ORD=[2,4,6,9]

def build(data,orders=ORD,prune=2):
    t={o:defaultdict(Counter) for o in orders}
    for i in range(max(orders),len(data)):
        nb=data[i]
        for o in orders: t[o][data[i-o:i]][nb]+=1
    return {o:{k:v for k,v in t[o].items() if sum(v.values())>=prune} for o in orders}
def pdist(tab,ctx):                                   # distribuição PPM (backoff) de um grafo
    d=np.full(256,1/256)
    for o in ORD:
        c=tab.get(o,{}).get(bytes(ctx[-o:]))
        if c:
            tot=sum(c.values()); dd=np.zeros(256)
            for b,ct in c.items(): dd[b]=ct/tot
            lam=min(0.95,0.4+0.09*o); d=lam*dd+(1-lam)*d
    return d

print("[grafo GLOBAL do corpus (20MB)…]",flush=True); G=build(C[:20_000_000])

# --- SEMENTES = pontos de controle; cada uma vira um grafo TEMÁTICO (janelas do corpus ao redor dela) ---
SEMENTES=["Amazônia","floresta","rios","biodiversidade","clima"]
print(f"[grafos temáticos por semente {SEMENTES}…]",flush=True)
themes=[]
for s in SEMENTES:
    key=s.encode(); pos=[]; i=C.find(key)
    while i!=-1 and len(pos)<600: pos.append(i); i=C.find(key,i+1)
    win=b"".join(C[max(0,p-160):p+160] for p in pos)          # concatena vizinhança da semente
    themes.append(build(win,orders=[2,4,6],prune=2));
    print(f"  '{s}': {len(pos)} ocorrências, tema com {sum(len(v) for v in themes[-1].values())} nós")

def bspline_w(t,k):                                   # pesos de B-spline cúbica uniforme (4 pontos de controle ativos)
    if k==1: return {0:1.0}
    u=t*(k-1); i=min(int(u),k-2); f=u-i
    b=[(-f**3+3*f**2-3*f+1)/6,(3*f**3-6*f**2+4)/6,(-3*f**3+3*f**2+3*f+1)/6,f**3/6]
    w=defaultdict(float)
    for j,bw in zip([i-1,i,i+1,i+2],b): w[min(max(j,0),k-1)]+=bw
    return w

def gera(N=420,beta=2.2,seed_txt="A ",guiado=True):
    out=bytearray(seed_txt.encode())
    for step in range(N):
        ctx=out[-max(ORD):]
        pg=pdist(G,ctx)                               # fluência local (grafo global)
        if guiado:
            w=bspline_w(step/max(1,N-1),len(themes))  # onde estamos na curva das sementes
            pt=np.zeros(256)
            for i,wi in w.items():
                if wi>1e-4: pt+=wi*pdist(themes[i],ctx)   # mistura dos grafos temáticos pela B-spline
            p=pg*(pt+1e-3)**beta                       # grafo PUXADO pela curva temática
        else:
            p=pg
        p=p/p.sum(); nb=int(np.random.choice(256,p=p)); out.append(nb)
    return out.decode("utf-8","replace")

print("\n"+"="*70)
print("### SEM guia (grafo puro) — fluente mas DERIVA de tema:")
print("  "+gera(guiado=False))
print("\n### COM sementes+B-spline (Amazônia→floresta→rios→biodiversidade→clima):")
print("  "+gera(guiado=True))
print("="*70)
print("→ as sementes ancoram o TEMA; a B-spline transiciona SUAVE entre elas; o grafo mantém fluente.")
print("  geração + grafos + sementes + B-spline, 100% byte, sem LLM. (a IARA volta quando isto amadurecer)")
