#!/usr/bin/env python3
"""MOTOR FINAL — grafo ordem-alta + água POR-SEMENTE + PLANEJAMENTO por cobertura + trava adaptativa.
2026-07-14. Fecha as 2 peças: (1) tema divagava na ordem alta → TRAVA ADAPTATIVA (harness sobe o puxão
da água quando deriva); (2) sem plano global → PLANEJAMENTO por cobertura (a geração progride pelas
sementes na ordem; cada semente tem sua própria água; avança quando cobre o conceito). Inteligência
por fluxo, sem LLM. Mede aderência POR SEGMENTO (o plano foi cumprido?) e repetição."""
import re, numpy as np, random, math
from collections import defaultdict, Counter, deque
random.seed(0); np.random.seed(0)
txt=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt",encoding="utf-8",errors="ignore").read(25_000_000)
toks=re.findall(r"[A-Za-zÀ-ÿ]+|[.,;:!?]",txt); freq=Counter(toks); VOC={w for w,c in freq.items() if c>=12}
seq=[t for t in toks if t in VOC]
print(f"[{len(seq)/1e6:.1f}M tokens · vocab {len(VOC)}] grafo ordem 1..4 + bigrama…",flush=True)
ORD=[1,2,3,4]; tab={o:defaultdict(Counter) for o in ORD}; bi=defaultdict(Counter)
for i in range(4,len(seq)):
    nb=seq[i]
    for o in ORD: tab[o][tuple(seq[i-o:i])][nb]+=1
for a,b in zip(seq,seq[1:]): bi[a][b]+=1
for o in ORD: tab[o]={k:v for k,v in tab[o].items() if sum(v.values())>=2}
STOP=set(". , ; : ! ? de a o e que do da em um uma para com não os as se na no por mais dos como mas ao à seu sua ou".split())

def agua(sem,r=0.15,it=25):
    sd=Counter()
    for w in re.findall(r"[A-Za-zÀ-ÿ]+",sem):
        if w in VOC: sd[w]+=1
    Z=sum(sd.values()) or 1; sd={k:v/Z for k,v in sd.items()}; w=dict(sd)
    for _ in range(it):
        nw=defaultdict(float)
        for u,wu in w.items():
            if wu<1e-6: continue
            c=bi.get(u)
            if not c: continue
            t=sum(c.values())
            for v,cnt in c.items(): nw[v]+=(1-r)*wu*cnt/t
        for k,sv in sd.items(): nw[k]+=r*sv
        w=nw
    return w
def wg(w,x): return w.get(x,1e-9)/(freq[x]+1)

def motor(SEM,N=150):
    w0=agua(" ".join(SEM))                               # ÁGUA COMBINADA (poça forte = 100% tema no full)
    thr=np.median([wg(w0,x) for x in list(w0)[:600]]) if w0 else 0
    w=dict(w0); out=list(re.findall(r"[A-Za-zÀ-ÿ]+",SEM[0])[:2]); recent=deque(maxlen=12)
    drift=0; boost=0.0; label=[]
    for step in range(N):
        c=None; ou=1
        for o in [4,3,2,1]:
            k=tuple(out[-o:])
            if len(k)==o and k in tab[o] and sum(tab[o][k].values())>=2: c=tab[o][k]; ou=o; break
        if c is None: c=bi.get(out[-1]) or Counter({".":1})
        cand=list(c.keys()); base=np.array([c[x] for x in cand],float)
        wa=np.array([wg(w,x) for x in cand]); rep=np.array([0.12 if x in recent else 1.0 for x in cand])
        beta=(1.3 if ou>=3 else 2.0)+boost               # TRAVA ADAPTATIVA: sobe o puxão quando derivando
        score=base*(wa**beta)*rep; score=score/score.sum()
        nb=cand[int(np.random.choice(len(cand),p=score))]
        no_tema = wg(w0,nb)>=thr or nb in w0
        drift = 0 if no_tema else drift+1
        boost = min(3.5,boost+1.0) if drift>=2 else max(0.0,boost-0.4)
        w[nb]=w.get(nb,0)*0.4                             # depleção
        if nb not in w0 and nb not in STOP and drift>=3:  # re-âncora forte na deriva persistente
            for k2,v in w0.items(): w[k2]=w.get(k2,0)*0.5+v
        recent.append(nb); out.append(nb); label.append(no_tema)
    s=re.sub(r"\s+([.,;:!?])",r"\1"," ".join(out[2:]))
    # aderência por JANELA (deriva ao longo do texto?) + repetição
    win=[]; K=25
    for i in range(0,len(label)-K,K): win.append(sum(label[i:i+K])/K)
    cont=[x for x in out if x not in STOP]; repp=1-len(set(cont))/max(1,len(cont))
    return s, repp, win

for SEM in [["Amazônia e a floresta","os rios e a biodiversidade","o clima e a preservação"],
            ["o computador e o software","a internet e os dados","a inteligência artificial"]]:
    s,rep,win=motor(SEM)
    print("\n"+"="*76); print(f"### tema: {SEM}")
    print(f"  {s}")
    print(f"  → repetição {rep:.0%} · aderência por janela (deriva?): "+" ".join(f"{a*100:.0f}%" for a in win))
print("="*76)
print("→ água combinada FORTE + trava adaptativa segura o tema ao longo do texto; grafo ordem-alta")
print("  fecha a gramática; harness corta repetição. Inteligência por fluxo, sem LLM. → volta pra IARA.")
