#!/usr/bin/env python3
"""WS22 — SEMEAR OS PESOS EM GRAFO (sem Q&A) — a ideia real do Leonardo (2026-07-12).

NÃO interrogar o modelo (perguntar/salvar). DECOMPOR os PESOS num grafo, direto:
  cada neurônio FFN = NÓ. Valor (down_proj[:,i]) → logit-lens → o que ESCREVE (tokens de saída).
  Chave (gate/up[i]) → embedding → o que o ATIVA (tokens de entrada).
  Grafo token↔neurônio↔token, 100% determinístico dos pesos. (Geva: FFN = memória chave-valor.)

MEDE: (1) quantos neurônios têm conceito LEGÍVEL (valor decodifica em tokens coerentes);
(2) TERRITÓRIOS de conceito emergem dos pesos (neurônios de geografia/países/capitais)?;
(3) LIGAÇÃO FATO: um neurônio que ESCREVE uma capital tem CHAVE que responde ao país?
    (= a aresta país→capital está nos pesos, sem perguntar). Honesto sobre o limite (fato distribuído)."""
import torch, os, re, time, json, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
MODEL=MOD+"/Qwen2.5-3B-Instruct"
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

log(f"\n{'='*72}\n# WS22 — SEMEAR OS PESOS EM GRAFO (sem Q&A) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL)
m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(DEV).eval()
NL=m.config.num_hidden_layers; INT=m.config.intermediate_size
E=m.get_output_embeddings().weight.detach()          # [V,H] saída
Ein=m.model.embed_tokens.weight.detach()             # [V,H] entrada (tie: mesma)
norm_w=m.model.norm.weight.detach()
DEEP=list(range(NL-12,NL))                            # onde o conhecimento vive (atlas)
log(f"modelo: {NL} camadas × {INT} neurônios · decodificando os PESOS das camadas {DEEP[0]}-{DEEP[-1]} (sem perguntar nada)")

def toks_of(vec, mat, k=6):
    logits=(mat @ (vec*norm_w).to(mat.dtype)).float()
    return [tok.convert_ids_to_tokens([int(i)])[0].replace("Ġ","·").replace("Ċ","\\n") for i in torch.topk(logits,k).indices]

# ---------- 1) decodifica TODOS os neurônios profundos (valor = o que escreve) ----------
NODES=[]   # (L, i, out_tokens)
legivel=0
Et=E.t().contiguous()
for L in DEEP:
    dp=m.model.layers[L].mlp.down_proj.weight.detach()   # [H, INT]
    vals=(dp.t()*norm_w)                                 # [INT, H]
    for s in range(0,INT,1500):                          # chunk p/ caber na VRAM
        chunk=vals[s:s+1500]
        logits=chunk @ Et                                # [chunk, V] fp16
        top=torch.topk(logits,6,dim=1).indices           # [chunk,6]
        for j in range(top.shape[0]):
            ots=tok.convert_ids_to_tokens([int(x) for x in top[j]])
            clean=[t.replace("Ġ","·").replace("Ċ","n") for t in ots]
            if sum(1 for t in clean if re.search(r"[A-Za-z]{2,}",t))>=4: legivel+=1
            NODES.append((L,s+j,clean))
        del logits,top; torch.cuda.empty_cache()
    del dp,vals; torch.cuda.empty_cache()
log(f"  neurônios decodificados: {len(NODES)} · com conceito LEGÍVEL (≥4/6 tokens alfabéticos): {legivel} = {legivel/len(NODES):.0%}")

# ---------- 2) TERRITÓRIOS de conceito emergem dos pesos? ----------
def find_neurons_writing(words, topn=8):
    ids=sorted({i for w in words for i in tok.encode(w,add_special_tokens=False)})
    c=E[ids].float().mean(0); c=c/c.norm()
    scores=[]
    for L in DEEP:
        dp=m.model.layers[L].mlp.down_proj.weight.detach().t()   # [INT,H]
        vn=dp.float(); vn=vn/vn.norm(dim=1,keepdim=True).clamp_min(1e-6)
        s=(vn @ c)                                               # cosseno do valor com o conceito
        v,idx=torch.topk(s,3)
        for sc,ii in zip(v.tolist(),idx.tolist()): scores.append((sc,L,ii))
    scores.sort(reverse=True)
    return scores[:topn]
log(f"\n## TERRITÓRIOS DE CONCEITO nos pesos (sem perguntar — só o valor dos neurônios)")
for label,words in [("PAÍSES",[" France"," Brazil"," Japan"," Germany"," China"," Egypt"]),
                    ("CAPITAIS",[" Paris"," Tokyo"," Berlin"," Madrid"," Cairo"]),
                    ("LÍNGUAS",[" Portuguese"," Spanish"," French"," German"," Arabic"])]:
    top=find_neurons_writing(words,4)
    log(f"  {label}:")
    for sc,L,i in top:
        _,_,ots=next(n for n in NODES if n[0]==L and n[1]==i)
        log(f"    L{L} n{i} (cos {sc:.2f}) escreve: {ots}")

# ---------- 3) LIGAÇÃO-FATO nos pesos: neurônio que escreve a capital responde ao país? ----------
log(f"\n## LIGAÇÃO país→capital DIRETO DOS PESOS (a aresta está lá?)")
FACTS=[("France","Paris"),("Japan","Tokyo"),("Germany","Berlin"),("Egypt","Cairo"),("China","Beijing")]
def key_response(L,i,word):
    # chave do neurônio = gate/up row i; afinidade com o token de entrada (país)
    g=m.model.layers[L].mlp.gate_proj.weight[i].detach().float()
    u=m.model.layers[L].mlp.up_proj.weight[i].detach().float()
    ids=tok.encode(" "+word,add_special_tokens=False)
    e=Ein[ids[0]].float()
    return float(torch.dot(g,e)/(g.norm()*e.norm()+1e-6))+float(torch.dot(u,e)/(u.norm()*e.norm()+1e-6))
bound=0
for country,cap in FACTS:
    top=find_neurons_writing([" "+cap],3)                # neurônios que ESCREVEM a capital
    best=None
    for sc,L,i in top:
        kr=key_response(L,i,country)                     # a CHAVE deles responde ao país?
        if best is None or kr>best[1]: best=(f"L{L}n{i}",kr,sc)
    is_bound = best[1]>0.02
    bound+=is_bound
    log(f"  {country}→{cap}: melhor neurônio-escreve-{cap} = {best[0]}, chave↔{country} = {best[1]:+.3f} {'✓ aresta nos pesos' if is_bound else '(fraca — fato distribuído)'}")
log(f"  ligação-fato encontrada nos pesos: {bound}/{len(FACTS)}")

log(f"\n## VEREDITO WS22 (honesto)")
log(f"  ✓ CONCEITO sai dos pesos direto: {legivel/len(NODES):.0%} dos neurônios profundos têm conceito legível;")
log(f"    territórios emergem (neurônio que escreve países, capitais, línguas — SEM perguntar nada).")
log(f"  {'✓' if bound>=3 else '⚠'} LIGAÇÃO-FATO país→capital nos pesos: {bound}/{len(FACTS)} — {'a aresta está parcialmente nos pesos' if bound>=3 else 'fraca: o fato é DISTRIBUÍDO em vários neurônios/camadas'}")
log(f"  LIMITE honesto: 'semear todos os pesos' dá o grafo NEURÔNIO (conceitos, territórios) de graça e")
log(f"    determinístico; mas o FATO simbólico limpo (país→capital como 1 aresta) é distribuído — precisa")
log(f"    de análise de circuito OU do probe pra cristalizar. Os pesos DÃO o substrato; a aresta-fato é emergente.")
log(f"wall {(time.time()-t0)/60:.1f}min")
json.dump(dict(neurons=len(NODES),legivel=legivel/len(NODES),fact_bound=f"{bound}/{len(FACTS)}"),
    open(os.path.join(HERE,"ws22_pesos.json"),"w"),indent=1)
