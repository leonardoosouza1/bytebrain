#!/usr/bin/env python3
"""WS5 — ROTEADOR AUTOMÁTICO: o cérebro escolhe o especialista sozinho.

HIPÓTESE: a ESCRITA profunda dos neurônios do generalista, lida por logit-lens contra
vetores-conceito de domínio (matemática/código/fatos), roteia a query pro especialista
certo — sem regex, sem eu apontar. (Roteador por território já deu 90-100% no 4B; aqui
a versão barata por escrita-de-MLP no 1.5B, plugável no cérebro acoplado.)

Mede: (1) acurácia de roteamento em 60 queries; (2) ganho END-TO-END: NLL da resposta
certa no modelo ROTEADO vs sempre-Instruct (a paleta do WS-G provou Math/Coder melhores
nos seus domínios; aqui fecha o ciclo: rotear → ganhar)."""
import torch, os, re, time, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
BASE=os.path.join(HERE,"../../llm-lab/models")
PATHS={"Instruct":f"{BASE}/Qwen2.5-1.5B-Instruct","Coder":f"{BASE}/Qwen2.5-Coder-1.5B","Math":f"{BASE}/Qwen2.5-Math-1.5B"}
DEV="cuda"; DEEP=list(range(16,28))
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

QUERIES={
 "math":[("What is 34 + 58?"," 92"),("Compute 17 * 6."," 102"),("What is 144 divided by 12?"," 12"),
   ("If x + 7 = 20, what is x?"," 13"),("What is the square root of 81?"," 9"),("Calculate 25% of 200."," 50"),
   ("What is 7 squared?"," 49"),("Solve: 3x = 27. x ="," 9"),("What is 1000 - 377?"," 623"),
   ("Add 45 and 67."," 112"),("What is 12 times 11?"," 132"),("Half of 86 is"," 43"),
   ("What is 9 + 9 + 9?"," 27"),("Compute 64 / 8."," 8"),("What is 15 * 4?"," 60"),
   ("2 to the power of 6 is"," 64"),("What is 100 minus 64?"," 36"),("Triple of 21 is"," 63"),
   ("What is 55 + 45?"," 100"),("Divide 90 by 6."," 15")],
 "code":[("How do I print in Python? Use the function"," print"),("To import numpy write: import numpy as"," np"),
   ("In Python, to get the length of a list use"," len"),("To read a CSV with pandas: df = pd."," read_csv"),
   ("A Python function is defined with the keyword"," def"),("To return a value from a function use"," return"),
   ("To create a loop over 10 numbers: for i in"," range"),("Exceptions in Python are caught with try and"," except"),
   ("To open a file in Python call"," open"),("Lists are appended with the method"," append"),
   ("String formatting uses f-strings: f\"{"," name"),("To install a package use pip"," install"),
   ("A dictionary maps keys to"," values"),("Comments in Python start with"," #"),
   ("To sort a list in place call"," sort"),("The boolean values in Python are True and"," False"),
   ("To convert to integer use the function"," int"),("Classes are defined with the keyword"," class"),
   ("To exit a loop early use"," break"),("Import matplotlib.pyplot as"," plt")],
 "facts":[("What is the capital of France? It is"," Paris"),("The capital of Japan is"," Tokyo"),
   ("The capital of Germany is"," Berlin"),("What is the capital of Italy? It is"," Rome"),
   ("The capital of Spain is"," Madrid"),("The capital of Egypt is"," Cairo"),
   ("The capital of Canada is"," Ottawa"),("The capital of Greece is"," Athens"),
   ("The capital of Portugal is"," Lisbon"),("The capital of Austria is"," Vienna"),
   ("The capital of Norway is"," Oslo"),("The capital of Poland is"," Warsaw"),
   ("The capital of China is"," Beijing"),("The capital of Thailand is"," Bangkok"),
   ("The capital of Ireland is"," Dublin"),("The capital of Kenya is"," Nairobi"),
   ("The capital of Peru is"," Lima"),("The capital of Cuba is"," Havana"),
   ("The capital of Sweden is"," Stockholm"),("The capital of Hungary is"," Budapest")],
}
SPECIALIST={"math":"Math","code":"Coder","facts":"Instruct"}

log(f"\n{'='*72}\n# WS5 — ROTEADOR AUTOMÁTICO — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(PATHS["Instruct"])
model=AutoModelForCausalLM.from_pretrained(PATHS["Instruct"],dtype=torch.float16).to(DEV).eval()
E=model.get_output_embeddings().weight.detach(); norm_w=model.model.norm.weight.detach()
writes={}
for L in DEEP:
    model.model.layers[L].mlp.down_proj.register_forward_hook(
        (lambda L:(lambda m,i,o: writes.__setitem__(L,o.detach()[0,-1])))(L))

def concept(words):
    ids=sorted({i for w in words for i in tok.encode(w,add_special_tokens=False)})
    v=E[ids].mean(0); return v/v.norm()
CV={
 "math":concept([" equation"," math"," sum"," number"," calculate"," plus"," minus"," multiply"," divide"," algebra"," =", " 7"," 12"]),
 "code":concept([" def"," import"," return"," function"," python"," print"," class"," code"," variable"," list"]),
 "facts":concept([" country"," capital"," city"," France"," Japan"," Germany"," geography"," nation"," Paris"," currency"]),
}
@torch.no_grad()
def route(q):
    model(tok(q,return_tensors="pt").input_ids.to(DEV))
    w=(sum(writes[L] for L in DEEP)*norm_w); w=w/w.norm()
    scores={k:float(w@v) for k,v in CV.items()}
    return max(scores,key=scores.get), scores

hits=0; total=0; routed={}
for dom,qs in QUERIES.items():
    ok=0
    for q,_ in qs:
        pred,_=route(q); routed[q]=pred; ok+= (pred==dom); hits+=(pred==dom); total+=1
    log(f"  roteamento {dom:>6}: {ok}/{len(qs)}")
log(f"  ROTEAMENTO GERAL: {hits}/{total} = {hits/total:.0%}  (meta ≥85%)")

# end-to-end: NLL da resposta no modelo escolhido pelo roteador vs sempre-Instruct
@torch.no_grad()
def nll(m,p,a):
    pi=tok(p,return_tensors="pt").input_ids.to(DEV); ai=tok(a,add_special_tokens=False).input_ids
    full=torch.cat([pi,torch.tensor([ai],device=DEV)],1); lg=m(full).logits[0]; b=pi.shape[1]-1
    return sum(-torch.log_softmax(lg[b+k].float(),-1)[t].item() for k,t in enumerate(ai))/len(ai)/0.6931
nll_inst={}
for dom,qs in QUERIES.items():
    nll_inst[dom]=sum(nll(model,q,a) for q,a in qs)/len(qs)
del model,E; gc.collect(); torch.cuda.empty_cache()
nll_routed={d:0.0 for d in QUERIES}
for name in ["Math","Coder"]:
    m=AutoModelForCausalLM.from_pretrained(PATHS[name],dtype=torch.float16).to(DEV).eval()
    for dom,qs in QUERIES.items():
        sel=[(q,a) for q,a in qs if SPECIALIST[routed[q]]==name]
        for q,a in sel: nll_routed[dom]+=nll(m,q,a)
    del m; gc.collect(); torch.cuda.empty_cache()
mi=AutoModelForCausalLM.from_pretrained(PATHS["Instruct"],dtype=torch.float16).to(DEV).eval()
for dom,qs in QUERIES.items():
    sel=[(q,a) for q,a in qs if SPECIALIST[routed[q]]=="Instruct"]
    for q,a in sel: nll_routed[dom]+=nll(mi,q,a)
    nll_routed[dom]/=len(qs)
del mi; gc.collect(); torch.cuda.empty_cache()
log(f"\n  END-TO-END (bits da resposta certa, menor=melhor):")
for dom in QUERIES:
    d=nll_inst[dom]-nll_routed[dom]
    log(f"  {dom:>6}: sempre-Instruct {nll_inst[dom]:.3f} → roteado {nll_routed[dom]:.3f}  (Δ {d:+.3f} {'✓ ganha' if d>0.02 else '≈'})")
log(f"VEREDITO WS5: roteamento {hits/total:.0%} · wall {(time.time()-t0)/60:.1f} min")
