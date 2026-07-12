#!/usr/bin/env python3
"""INTEL-PROBE — mede a INTELIGÊNCIA atual do byte-model, sem maquiar. Bateria: (1) factual-completion
(sabe o fato?), (2) Q&A format (segue instrução?), (3) coerência livre. Auto-métrica + texto pra leitura
humana. Objetivo: saber ONDE estamos no eixo inteligência antes de treinar mais."""
import sys, re, torch
sys.path.insert(0,"/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
from src.sample import generate
DEV="cuda" if torch.cuda.is_available() else "cpu"
CK=sys.argv[1] if len(sys.argv)>1 else "/home/leonardo/projects/LLM/bytebrain/research/ckpt_gen/ckpt_best.pt"
ck=torch.load(CK,map_location=DEV,weights_only=False); c=ck["config"]
m=ByteGPT(dim=c["dim"],n_layers=c["layers"],n_heads=c["heads"],context=c["ctx"]).to(DEV).eval()
m.load_state_dict(ck["model"])
print(f"### {CK.split('/')[-2]} | {m.n_params/1e6:.0f}M | step {ck['step']} | val {ck.get('best_val',0):.3f}\n",flush=True)
def wok(t): ws=re.findall(r"[a-zA-Zà-ü]+",t); return sum(1 for w in ws if len(w)>=3 and re.search(r"[aeiouáéíóú]",w))/max(len(ws),1)
def g(p,n=120,t=0.6): return generate(m,prompt=p,n=n,temperature=t,top_p=0.9,rep_penalty=1.3,device=DEV)
# (1) FACTUAL — o fato certo aparece na continuação?
FACT=[("A capital do Brasil é","brasília"),("A capital da França é","paris"),("A capital do Japão é","tóqui"),
      ("A velocidade da luz é de aproximadamente","300"),("O maior planeta do sistema solar é","júpiter"),
      ("A fórmula da água é","h2o"),("O autor de Dom Casmurro foi","machado")]
print("=== (1) FACTUAL (o certo aparece nas primeiras palavras?) ===",flush=True)
hits=0
for p,gold in FACT:
    out=g(p,n=40,t=0.4); cont=out[len(p):].lower(); ok=gold in cont[:60]; hits+=ok
    print(f"[{'✓' if ok else '✗'}] {p!r} → {cont[:55].strip()!r}",flush=True)
print(f"  ACERTOS: {hits}/{len(FACT)}\n",flush=True)
# (2) Q&A / instrução
print("=== (2) Q&A (segue o formato P:/R:?) ===",flush=True)
for q in ["P: O que é uma célula?\nR:","P: Quem descobriu o Brasil?\nR:","P: O que é fotossíntese?\nR:"]:
    out=g(q,n=90,t=0.6); print(f"{q} {out[len(q):].strip()[:90]!r}",flush=True)
# (3) coerência livre
print("\n=== (3) COERÊNCIA LIVRE (word-ok) ===",flush=True)
for p in ["O Brasil é ","A inteligência artificial ","A história da humanidade "]:
    out=g(p,n=140,t=0.7); print(f"[wok {wok(out):.2f}] {out.strip()[:150]!r}",flush=True)
print("\nDONE",flush=True)
