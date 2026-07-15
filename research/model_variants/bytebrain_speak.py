#!/usr/bin/env python3
"""FALA(Qwen) + FATOS(grafo) = gerador de bytes fluente E grounded. 2026-07-14.
Sacada do Leonardo: o Qwen JÁ sabe a FALA (oi/tudo bem/gramática) — só não sabe os fatos específicos;
o grafo/água tem os fatos. Junta: o MOTOR-FLUXO propõe candidatos grounded (no tema), a FALA do Qwen
(1 forward/passo) escolhe o mais fluente. Grafo dá o conteúdo, Qwen dá a forma. Um gerador de bytes."""
import re, numpy as np, random, math, torch
from collections import defaultdict, Counter, deque
from transformers import AutoTokenizer, AutoModelForImageTextToText
random.seed(0); np.random.seed(0); ID="Qwen/Qwen2.5-VL-3B-Instruct"
# ---- motor-fluxo (grafo de palavras ordem-alta + água) ----
txt=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt",encoding="utf-8",errors="ignore").read(12_000_000)
toks=re.findall(r"[A-Za-zÀ-ÿ]+|[.,;:!?]",txt); freq=Counter(toks); VOC={w for w,c in freq.items() if c>=10}
seq=[t for t in toks if t in VOC]; ORD=[1,2,3]; tab={o:defaultdict(Counter) for o in ORD}; bi=defaultdict(Counter)
for i in range(3,len(seq)):
    for o in ORD: tab[o][tuple(seq[i-o:i])][seq[i]]+=1
for a,b in zip(seq,seq[1:]): bi[a][b]+=1
for o in ORD: tab[o]={k:v for k,v in tab[o].items() if sum(v.values())>=2}
def agua(sem,r=0.15,it=20):
    sd=Counter()
    for w in re.findall(r"[A-Za-zÀ-ÿ]+",sem):
        if w in VOC: sd[w]+=1
    Z=sum(sd.values()) or 1; sd={k:v/Z for k,v in sd.items()}; w=dict(sd)
    for _ in range(it):
        nw=defaultdict(float)
        for u,wu in w.items():
            c=bi.get(u)
            if wu<1e-6 or not c: continue
            t=sum(c.values())
            for v,cnt in c.items(): nw[v]+=(1-r)*wu*cnt/t
        for k,sv in sd.items(): nw[k]+=r*sv
        w=nw
    return w
def wg(w,x): return w.get(x,1e-9)/(freq[x]+1)
print("[motor-fluxo pronto] carregando o Qwen (a FALA)…",flush=True)
tk=AutoTokenizer.from_pretrained(ID); qm=AutoModelForImageTextToText.from_pretrained(ID,dtype=torch.float16).to("cuda").eval()

@torch.no_grad()
def qwen_next_logprobs(prefix):                          # a "fala": distribuição do próximo token do Qwen
    ids=tk(prefix,return_tensors="pt").input_ids.to("cuda")
    return torch.log_softmax(qm(input_ids=ids).logits[0,-1].float(),-1).cpu()

def gera(sem,N=40,beta=1.6,alpha=1.2,usa_fala=True):
    w=agua(sem); out=list(re.findall(r"[A-Za-zÀ-ÿ]+",sem)[:2]); recent=deque(maxlen=10)
    for step in range(N):
        c=None
        for o in [3,2,1]:
            k=tuple(out[-o:])
            if len(k)==o and k in tab[o]: c=tab[o][k]; break
        if c is None: c=bi.get(out[-1]) or Counter({".":1})
        cand=list(c.keys())[:40]; base=np.array([c[x] for x in cand],float)
        wa=np.array([wg(w,x) for x in cand]); rep=np.array([0.2 if x in recent else 1.0 for x in cand])
        gscore=np.log(base/base.sum()+1e-9)+beta*np.log(wa+1e-12)+np.log(rep)   # FLUXO: grounded+tema
        if usa_fala:
            lp=qwen_next_logprobs(" ".join(out))          # FALA: 1 forward do Qwen
            qs=np.array([float(lp[tk(" "+x,add_special_tokens=False).input_ids[0]]) if tk(" "+x,add_special_tokens=False).input_ids else -20 for x in cand])
            score=gscore+alpha*qs                         # grafo PROPÕE, fala do Qwen DISPÕE
        else: score=gscore
        p=np.exp(score-score.max()); p/=p.sum()
        nb=cand[int(np.random.choice(len(cand),p=p))]; w[nb]=w.get(nb,0)*0.4; recent.append(nb); out.append(nb)
    return re.sub(r"\s+([.,;:!?])",r"\1"," ".join(out))

SEM="Amazônia floresta rios biodiversidade clima preservação"
print("\n"+"="*74)
print("### só FLUXO (grafo+água) — grounded/no-tema mas associativo:")
print("  "+gera(SEM,usa_fala=False))
print("\n### FLUXO + FALA do Qwen (grafo propõe candidatos, Qwen escolhe o fluente):")
print("  "+gera(SEM,usa_fala=True))
print("="*74)
print("→ o grafo dá os FATOS/tema; a fala do Qwen dá a FORMA fluente. Um gerador de bytes: fala+fatos.")
print(f"VRAM {round(torch.cuda.max_memory_allocated()/1e9,1)}GB")
