#!/usr/bin/env python3
"""ÁGUA NOS GRAFOS — geração por FLUXO (a tese-mãe da IARA aplicada à linguagem). 2026-07-14.
Leonardo: 'joga água nos grafos e ela acha o melhor caminho; tunela pra pular dados; não precisamos
de LLM, precisamos de INTELIGÊNCIA.' A água = Random-Walk-with-Restart a partir das SEMENTES: ela flui
pelo grafo de palavras e POÇA na região do tema (tunelando o resto). A geração escorre morro-abaixo
pra dentro da poça: fluência local (transições reais) × nível de água (rumo global). Multi-escala, leve."""
import re, numpy as np, random
from collections import defaultdict, Counter
random.seed(0); np.random.seed(0)
txt=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt",encoding="utf-8",errors="ignore").read(30_000_000)
words=re.findall(r"[A-Za-zÀ-ÿ]+|[.,;:!?]",txt)
print(f"[corpus {len(words)/1e6:.1f}M palavras] construindo grafo de palavras…",flush=True)
freq=Counter(words); VOC={w for w,c in freq.items() if c>=15}
nxt=defaultdict(Counter)                                   # grafo: palavra -> Counter(próxima palavra)
for a,b in zip(words,words[1:]):
    if a in VOC and b in VOC: nxt[a][b]+=1
print(f"  vocab {len(VOC)} · nós com saída {len(nxt)}")

def agua(sementes, r=0.15, iters=25):                     # RWR: joga água nas sementes, deixa fluir
    seed=Counter();
    for s in sementes:
        for w in re.findall(r"[A-Za-zÀ-ÿ]+",s):
            if w in VOC: seed[w]+=1
    Z=sum(seed.values()) or 1; seed={k:v/Z for k,v in seed.items()}
    w=dict(seed)                                          # começa nas sementes
    for _ in range(iters):
        nw=defaultdict(float)
        for u,wu in w.items():                            # a água de u escorre pras vizinhas (só onde há água = TÚNEL)
            if wu<1e-6: continue
            c=nxt.get(u);
            if not c: continue
            tot=sum(c.values())
            for v,cnt in c.items(): nw[v]+=(1-r)*wu*cnt/tot
        for k,sv in seed.items(): nw[k]+=r*sv             # restart nas sementes
        w=nw
    return w                                              # nível de água por palavra (poça no tema)

def gera(sementes, N=70, beta=1.6, guiado=True):
    w=agua(sementes) if guiado else {}
    cur=random.choice([s for s in sementes for s in re.findall(r"[A-Za-zÀ-ÿ]+",s) if s in VOC] or ["O"])
    out=[cur]
    for _ in range(N):
        c=nxt.get(cur)
        if not c: cur=random.choice(list(VOC)); out.append(cur); continue
        cand=list(c.keys()); base=np.array([c[x] for x in cand],float)
        if guiado:
            wa=np.array([w.get(x,1e-9)/(freq[x]+1) for x in cand]); score=base*(wa**beta)  # ÁGUA÷GRAU (nosso achado: mata hub-domination)
        else: score=base
        score=score/score.sum(); cur=cand[int(np.random.choice(len(cand),p=score))]; out.append(cur)
    s=" ".join(out); return re.sub(r"\s+([.,;:!?])",r"\1",s)

SEM=["Amazônia floresta","rios e biodiversidade","clima e preservação"]
print("\n[jogando água nas sementes e deixando fluir…]")
w=agua(SEM); top=sorted(w.items(),key=lambda x:-x[1])[:14]
massa=sum(w.values()); conc=sum(v for _,v in top)/massa
print(f"  a água POÇOU: top-14 palavras concentram {conc:.0%} de toda a água (TUNELOU o resto)")
print("  poça do tema:", ", ".join(f"{k}" for k,_ in top))
print("\n"+"="*70)
print("### SEM água (grafo puro) — deriva:\n  "+gera(SEM,guiado=False))
print("\n### COM água fluindo das sementes (RWR) — escorre pro tema:\n  "+gera(SEM,guiado=True))
print("="*70)
print("→ a água acha o caminho: poça no tema, tunela o irrelevante, a geração escorre pra dentro.")
print("  É a tese-mãe (seres-água) resolvendo GERAÇÃO por fluxo em grafo — inteligência, não LLM.")
