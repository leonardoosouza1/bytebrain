#!/usr/bin/env python3
"""M111 — quebrar o teto int4 (~55): planta 1 seed K=8 em 100 fatos (batched) e testa 4 quantizações
(int4/int8 × per-tensor/per-token) pra ver qual segura MAIS fatos. GPU. Dump marco111_metrics.json."""
import json, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
M="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"; DEV="cuda"; t0=time.time()
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H=model.config.hidden_size; EL=model.get_input_embeddings(); random.seed(1)
facts=[]
for i in range(60): facts.append((f"O código secreto {i} é", f" {random.randint(1000,9999)}"))
for a in range(2,9):
  for b in range(2,8): facts.append((f"{a} vezes {b} é", f" {a*b}"))
facts=facts[:100]; N=len(facts)
P=[(tok(p).input_ids, tok(tg,add_special_tokens=False).input_ids) for p,tg in facts]
print(f"{N} fatos ({time.time()-t0:.0f}s)",flush=True)
def build(K):
  ml=K+max(len(pi)+len(ti) for pi,ti in P); E=torch.zeros(N,ml,H,device=DEV,dtype=torch.float16)
  am=torch.zeros(N,ml,device=DEV,dtype=torch.long); tp=[]; td=[]
  for i,(pi,ti) in enumerate(P):
    pe=EL(torch.tensor([pi],device=DEV)).detach()[0]; te=EL(torch.tensor([ti],device=DEV)).detach()[0]
    L=K+len(pi)+len(ti); E[i,K:K+len(pi)]=pe; E[i,K+len(pi):L]=te; am[i,:L]=1; tp.append((i,K+len(pi),len(ti))); td.append(ti)
  return E,am,tp,td
def run(seed,E,am): E=E.clone(); E[:,:seed.shape[0]]=seed.to(torch.float16); return model(inputs_embeds=E,attention_mask=am).logits
def recall(seed,E,am,tp,td):
  lg=run(seed,E,am); ok=0
  for (i,st,lt),ti in zip(tp,td): ok+=bool((lg[i,st-1:st-1+lt].argmax(-1)==torch.tensor(ti,device=DEV)).all())
  return ok
K=8; E,am,tp,td=build(K); seed=nn.Parameter(torch.randn(K,H,device=DEV,dtype=torch.float32)*0.1)
opt=torch.optim.AdamW([seed],lr=0.3)
for s in range(300):
  lg=run(seed,E,am); loss=0
  for (i,st,lt),ti in zip(tp,td): loss=loss+F.cross_entropy(lg[i,st-1:st-1+lt].float(),torch.tensor(ti,device=DEV))
  (loss/N).backward(); opt.step(); opt.zero_grad()
seed=seed.detach().to(torch.float16)
def q(t,bits,pertoken):
  qm=2**(bits-1)-1; t=t.float()
  if pertoken: s=(t.abs().amax(1,keepdim=True)/qm).clamp_min(1e-8)
  else: s=(t.abs().max()/qm).clamp_min(1e-8)
  return ((t/s).round().clamp(-qm,qm)*s).to(torch.float16)
res={"N":N,"fp16":recall(seed,E,am,tp,td)}
for name,bits,pt in [("int8_pt",8,False),("int8_ptok",8,True),("int4_pt",4,False),("int4_ptok",4,True)]:
  res[name]=recall(q(seed,bits,pt),E,am,tp,td)
  print(f"  {name}: {res[name]}/{N} ({time.time()-t0:.0f}s)",flush=True)
print(f"  fp16: {res['fp16']}/{N}",flush=True)
json.dump(res,open("research/marco111_metrics.json","w"))
print(f"DONE M111 ({time.time()-t0:.0f}s)",flush=True)
