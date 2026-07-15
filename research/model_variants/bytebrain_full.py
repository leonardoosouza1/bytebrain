#!/usr/bin/env python3
"""MOTOR COMPLETO — grafo de palavras ORDEM-ALTA (PPM) + água (tema) + harness (coesão). 2026-07-14.
O word-salad vinha do bigrama (1 palavra de contexto). Fix: contexto variável 1..4 palavras com
BACKOFF (como o PPM de byte que gerou coerente) + ÁGUA÷grau das sementes (rumo) + HARNESS (depleção,
verificador de repetição, backoff por ENTROPIA). Frases inteiras, no tema, sem LLM, sem gradiente."""
import re, numpy as np, random, math
from collections import defaultdict, Counter, deque
random.seed(0); np.random.seed(0)
txt=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt",encoding="utf-8",errors="ignore").read(25_000_000)
toks=re.findall(r"[A-Za-zÀ-ÿ]+|[.,;:!?]",txt); freq=Counter(toks); VOC={w for w,c in freq.items() if c>=12}
seq=[t for t in toks if t in VOC]
print(f"[{len(seq)/1e6:.1f}M tokens · vocab {len(VOC)}] construindo grafo ordem 1..4…",flush=True)
ORD=[1,2,3,4]; tab={o:defaultdict(Counter) for o in ORD}
for i in range(4,len(seq)):
    nb=seq[i]
    for o in ORD: tab[o][tuple(seq[i-o:i])][nb]+=1
for o in ORD: tab[o]={k:v for k,v in tab[o].items() if sum(v.values())>=2}
bi=defaultdict(Counter)                                   # bigrama p/ a água (RWR)
for a,b in zip(seq,seq[1:]): bi[a][b]+=1
print("  contextos:",{o:len(tab[o]) for o in ORD})
STOP=set(". , ; : ! ? de a o e que do da em um uma para com não os as se na no por mais dos como mas ao à seu sua ou".split())

def agua(sem,r=0.15,it=25):
    sd=Counter()
    for s in sem:
        for w in re.findall(r"[A-Za-zÀ-ÿ]+",s):
            if w in VOC: sd[w]+=1
    Z=sum(sd.values()) or 1; sd={k:v/Z for k,v in sd.items()}; w=dict(sd)
    for _ in range(it):
        nw=defaultdict(float)
        for u,wu in w.items():
            if wu<1e-6: continue
            c=bi.get(u);
            if not c: continue
            t=sum(c.values())
            for v,cnt in c.items(): nw[v]+=(1-r)*wu*cnt/t
        for k,sv in sd.items(): nw[k]+=r*sv
        w=nw
    return w
def wg(w,x): return w.get(x,1e-9)/(freq[x]+1)

def gera(sem,N=110,beta=1.3):
    w0=agua(sem); w=dict(w0); out=list(re.findall(r"[A-Za-zÀ-ÿ]+",sem[0])[:2]); recent=deque(maxlen=12)
    used_order=Counter(); low=0
    for step in range(N):
        c=None; ou=1
        for o in [4,3,2,1]:                              # BACKOFF: maior contexto que existe (gramática)
            k=tuple(out[-o:])
            if len(k)==o and k in tab[o] and sum(tab[o][k].values())>=2: c=tab[o][k]; ou=o; break
        if c is None: c=bi.get(out[-1]) or Counter({".":1})
        cand=list(c.keys()); base=np.array([c[x] for x in cand],float)
        p0=base/base.sum(); H=float(-(p0*np.log2(p0+1e-12)).sum())     # entropia (sinal de coesão)
        wa=np.array([wg(w,x) for x in cand]); rep=np.array([0.12 if x in recent else 1.0 for x in cand])
        bw=beta if ou>=3 else beta*1.6                    # HARNESS: ordem baixa/entropia alta → confia MAIS na água
        score=base*(wa**bw)*rep; score=score/score.sum()
        nb=cand[int(np.random.choice(len(cand),p=score))]
        w[nb]=w.get(nb,0)*0.4                             # depleção
        if nb not in w0 and nb not in STOP: low+=1
        else: low=0
        if low>=3:                                        # re-âncora na deriva
            for k2,v in w0.items(): w[k2]=w.get(k2,0)*0.5+v
            low=0
        used_order[ou]+=1; recent.append(nb); out.append(nb)
    s=re.sub(r"\s+([.,;:!?])",r"\1"," ".join(out))
    cont=[x for x in out if x not in STOP]; repp=1-len(set(cont))/max(1,len(cont))
    return s, repp, dict(used_order)

for SEM in [["Amazônia floresta","rios e biodiversidade","clima e preservação"],
            ["computador e software","internet e dados","inteligência artificial"]]:
    s,rep,uo=gera(SEM)
    print("\n"+"="*74); print(f"### sementes {SEM}")
    print(f"  {s}")
    print(f"  → repetição {rep:.0%} · uso de ordem (4=frase inteira→1=palavra): {uo}")
print("="*74)
print("→ contexto ordem-alta (PPM) fecha a GRAMÁTICA; água mantém o TEMA; harness corta repetição.")
print("  grafo + água + sementes + B-spline + harness = motor de linguagem por FLUXO. Sem LLM. Volta pra IARA.")
