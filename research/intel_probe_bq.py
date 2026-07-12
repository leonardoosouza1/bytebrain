#!/usr/bin/env python3
"""INTEL-PROBE ByteQwen — o transplante Qwen→byte CARREGA os fatos do Qwen? Mesmo battery do intel_probe,
no ckpt_byteqwen. Se souber fatos que o gerador Wikipedia (0/7) não sabe → o transplante é o portador de
inteligência. Se 0/7 também → trocar o embedding apagou o conhecimento (honesto)."""
import re, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda"
from transformers import AutoModelForCausalLM
QWEN="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-1.5B-Instruct"
CTX=512
m=AutoModelForCausalLM.from_pretrained(QWEN,dtype=torch.float16).to(DEV)
H=m.config.hidden_size
m.model.embed_tokens=nn.Embedding(256,H).to(DEV).to(torch.float32)
m.lm_head=nn.Linear(H,256,bias=False).to(DEV).to(torch.float32)
sd=torch.load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_byteqwen/ck.pt",map_location=DEV)
m.load_state_dict(sd["model"]); m.eval()
print(f"### ByteQwen | step {sd['step']} | {sum(p.numel() for p in m.parameters())/1e9:.2f}B\n",flush=True)
def wok(t): ws=re.findall(r"[a-zA-Zà-ü]+",t); return sum(1 for w in ws if len(w)>=3 and re.search(r"[aeiouáéíóú]",w))/max(len(ws),1)
@torch.no_grad()
def g(prompt,k=110,temp=0.5):
    ids=list(prompt.encode()); x=torch.tensor([ids],device=DEV)
    for _ in range(k):
        with torch.amp.autocast("cuda"): lg=m(x[:,-CTX:]).logits[0,-1]
        p=F.softmax(lg.float()/temp,-1); nx=int(torch.multinomial(p,1)); ids.append(nx); x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    return bytes(ids).decode("utf-8","ignore")
FACT=[("A capital do Brasil é","brasília"),("A capital da França é","paris"),("A capital do Japão é","tóqui"),
      ("A velocidade da luz é de aproximadamente","300"),("O maior planeta do sistema solar é","júpiter"),
      ("A fórmula da água é","h2o"),("O autor de Dom Casmurro foi","machado")]
print("=== (1) FACTUAL ===",flush=True); hits=0
for p,gold in FACT:
    out=g(p,k=40,temp=0.35); cont=out[len(p):].lower(); ok=gold in cont[:60]; hits+=ok
    print(f"[{'✓' if ok else '✗'}] {p!r} → {cont[:55].strip()!r}",flush=True)
print(f"  ACERTOS: {hits}/{len(FACT)}\n",flush=True)
print("=== (2) Q&A ===",flush=True)
for q in ["P: O que é uma célula?\nR:","P: Quem descobriu o Brasil?\nR:","P: O que é fotossíntese?\nR:"]:
    print(f"{q} {g(q,k=90,temp=0.5)[len(q):].strip()[:90]!r}",flush=True)
print("\n=== (3) COERÊNCIA LIVRE ===",flush=True)
for p in ["O Brasil é ","A inteligência artificial ","A história da humanidade "]:
    out=g(p,k=140,temp=0.6); print(f"[wok {wok(out):.2f}] {out.strip()[:150]!r}",flush=True)
print("\nDONE",flush=True)
