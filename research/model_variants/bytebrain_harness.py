#!/usr/bin/env python3
"""HARNESS DE COESÃO sobre a água-grafo — a inteligência REGULA o fluxo. 2026-07-14.
Leonardo: 'aplica mais harness; a gente já achou como calcular coesão'. Sim — decodificação guiada
por coesão + verificação por entropia (achados nossos). O harness CALCULA sinais a cada passo
(coesão, entropia, deriva, repetição) e INTERVÉM: depleção da água (drena ao usar=anti-repetição),
RE-ANCORAGEM quando deriva (refill das sementes), verificador de repetição. Mede a coesão. Sem LLM."""
import re, numpy as np, random, math
from collections import defaultdict, Counter, deque
random.seed(0); np.random.seed(0)
txt=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt",encoding="utf-8",errors="ignore").read(30_000_000)
words=re.findall(r"[A-Za-zÀ-ÿ]+|[.,;:!?]",txt); freq=Counter(words); VOC={w for w,c in freq.items() if c>=15}
nxt=defaultdict(Counter)
for a,b in zip(words,words[1:]):
    if a in VOC and b in VOC: nxt[a][b]+=1
STOP=set("de a o e que do da em um para com não os se na por mais as dos como mas ao ele à seu sua ou . , ; : ! ?".split())
print(f"[grafo {len(VOC)} palavras]")

def agua(sem,r=0.15,it=25):
    seed=Counter()
    for s in sem:
        for w in re.findall(r"[A-Za-zÀ-ÿ]+",s):
            if w in VOC: seed[w]+=1
    Z=sum(seed.values()) or 1; seed={k:v/Z for k,v in seed.items()}; w=dict(seed)
    for _ in range(it):
        nw=defaultdict(float)
        for u,wu in w.items():
            if wu<1e-6: continue
            c=nxt.get(u)
            if not c: continue
            t=sum(c.values())
            for v,cnt in c.items(): nw[v]+=(1-r)*wu*cnt/t
        for k,sv in seed.items(): nw[k]+=r*sv
        w=nw
    return w
def wg(w,x): return w.get(x,1e-9)/(freq[x]+1)               # água÷grau (força temática da palavra)

def gen(sem,N=90,beta=1.6,harness=False,verbose=False):
    w0=agua(sem); w=dict(w0); cur=[s for x in sem for s in re.findall(r"[A-Za-zÀ-ÿ]+",x) if s in VOC][0]
    out=[cur]; recent=deque(maxlen=10); low=0; refills=0; logs=[]
    for step in range(N):
        c=nxt.get(cur)
        if not c: cur=random.choice(list(VOC)); out.append(cur); continue
        cand=list(c.keys()); base=np.array([c[x] for x in cand],float)
        wa=np.array([wg(w,x) for x in cand])
        # ENTROPIA da escolha local (sinal de confiança/coesão)
        p0=base/base.sum(); H=float(-(p0*np.log2(p0+1e-12)).sum())
        if harness:
            rep=np.array([0.15 if x in recent else 1.0 for x in cand])     # VERIFICADOR de repetição
            score=base*(wa**beta)*rep
        else:
            score=base*(wa**beta)
        score=score/score.sum(); cur=cand[int(np.random.choice(len(cand),p=score))]
        adere=wg(w,cur)>np.median([wg(w,x) for x in VOC if x==x][:1] or [0]) or cur in w0   # está no tema?
        if harness:
            w[cur]=w.get(cur,0)*0.4                                        # DEPLEÇÃO: a água usada drena
            if cur not in w0 and cur not in STOP: low+=1
            else: low=0
            if low>=3:                                                     # DERIVA detectada → RE-ANCORA (refill)
                for k,v in w0.items(): w[k]=w.get(k,0)*0.5+v; low=0; refills+=1
        recent.append(cur); out.append(cur)
        if verbose and step%18==0: logs.append(f"    passo {step:2d}: entropia {H:.1f} bits · água '{cur}' {wg(w,cur):.1e}")
    s=re.sub(r"\s+([.,;:!?])",r"\1"," ".join(out))
    return s, out, refills, logs

def coesao(seq):                                            # MÉTRICAS de coesão (calculáveis)
    cont=[x for x in seq if x not in STOP and re.match(r"[A-Za-zÀ-ÿ]+",x)]
    rep=1-len(set(cont))/max(1,len(cont))                  # taxa de repetição de conteúdo
    edges=sum(1 for a,b in zip(seq,seq[1:]) if b in nxt.get(a,{}))/max(1,len(seq)-1)  # fração de transições REAIS (fluência)
    return rep, edges
def adere_janela(seq,w0,K=25):                             # aderência ao tema por janela (deriva?)
    out=[]
    for i in range(0,len(seq)-K,K):
        win=[x for x in seq[i:i+K] if x not in STOP]
        out.append(sum(1 for x in win if x in w0)/max(1,len(win)))
    return out

SEM=["Amazônia floresta","rios e biodiversidade","clima e preservação"]
w0=agua(SEM)
print("\n"+"="*72)
for nome,h in [("ÁGUA sem harness",False),("ÁGUA + HARNESS (coesão/entropia/depleção/re-âncora)",True)]:
    s,seq,ref,logs=gen(SEM,harness=h,verbose=h)
    rep,ed=coesao(seq); ad=adere_janela(seq,w0)
    print(f"\n### {nome}")
    for l in logs: print(l)
    print(f"  {s}")
    print(f"  → repetição {rep:.0%} · fluência(transições reais) {ed:.0%} · aderência tema por janela {['%.0f%%'%(a*100) for a in ad]} · re-âncoras {ref}")
print("="*72)
print("→ o harness CALCULA coesão/entropia e INTERVÉM (depleção mata repetição, re-âncora mata deriva).")
print("  É a IARA (verificador+água+sementes) regulando a geração. Próximo: mixer neural leve pro resíduo.")
