#!/usr/bin/env python3
"""WS25 — HEAD-TO-HEAD AO VIVO: robustez a TYPO, byte-IARA vs token (Qwen-3B) — honesto (2026-07-12).

Fecha a última lacuna de honestidade: eu citava '~45% do token' de memória. Aqui MEÇO agora, no
MESMO conjunto (as 61 capitais do grafo), limpo vs com typo no nome do país. Comparo a DEGRADAÇÃO:
  byte-IARA (edit-distance no substrato byte): 95% → 90% (medido na Fase 3b, -5pp)
  token 3B (lê o país tokenizado por BPE): mede-se aqui.
Se o 3B degrada MAIS sob typo, o substrato byte ganha em robustez (a tese). Se degrada menos/igual,
eu me corrijo e reporto honesto. GPU, threads-capados."""
import torch, os, re, time, json, unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def norm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()
def sub(a,b): na,nb=norm(a),norm(b); return bool(na and nb and (na in nb or nb in na))
def typo(w):                                            # mesmo typo determinístico da IARA final
    if len(w)<4: return w
    i=(sum(map(ord,w))%(len(w)-2))+1
    return w[:i]+w[i+1]+w[i]+w[i+2:]

# capitais reais do grafo (o que AMBOS deveriam saber)
g=json.load(open(os.path.join(HERE,"iara_graph_big.json")))["graph"]
caps=[(k.split("|")[0],v) for k,v in g.items() if k.endswith("|capital") and v]
log(f"\n{'='*72}\n# WS25 — HEAD-TO-HEAD TYPO: byte-IARA vs token-3B — {time.strftime('%H:%M')}\n{'='*72}")
log(f"conjunto: {len(caps)} capitais do grafo · pergunta limpa vs com typo no país")

tok=AutoTokenizer.from_pretrained(MOD+"/Qwen2.5-3B-Instruct")
m=AutoModelForCausalLM.from_pretrained(MOD+"/Qwen2.5-3B-Instruct",dtype=torch.float16).to(DEV).eval()
@torch.no_grad()
def ask3b(country):
    s=f"<|im_start|>user\nWhat is the capital of {country}? Answer with only the city name.<|im_end|>\n<|im_start|>assistant\n"
    ids=tok(s,return_tensors="pt").input_ids.to(DEV)
    o=m.generate(ids,max_new_tokens=8,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()

t0=time.time(); clean=dirty=0; ex_fail=[]
for c,cap in caps:
    ent=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",c)           # South Korea
    ac=ask3b(ent); ad=ask3b(typo(ent))
    hc=sub(cap,ac); hd=sub(cap,ad)
    clean+=hc; dirty+=hd
    if hc and not hd and len(ex_fail)<4: ex_fail.append(f"{ent}→typo '{typo(ent)}' : 3B disse {ad[:18]!r} (certo={cap})")
N=len(caps)
log(f"\n## RESULTADO token-3B (ao vivo)")
log(f"  LIMPO: {clean}/{N} = {clean/N:.0%}")
log(f"  COM TYPO: {dirty}/{N} = {dirty/N:.0%}  → degradação {(clean-dirty)/N*100:+.0f}pp")
for e in ex_fail: log(f"    ✗ {e}")
log(f"\n## COMPARATIVO honesto")
log(f"  byte-IARA : 95% → 90%  (degradação -5pp) — edit-distance conserta o país no substrato byte")
log(f"  token-3B  : {clean/N:.0%} → {dirty/N:.0%}  (degradação {(clean-dirty)/N*100:+.0f}pp) — BPE re-tokeniza a palavra com erro")
delta_byte=5; delta_tok=(clean-dirty)/N*100
verdict=("byte GANHA em robustez (degrada menos)" if delta_tok>delta_byte else
         "token igual/mais robusto — me corrijo (o 3B é grande e tolera typo)" if delta_tok<=delta_byte else "empate")
log(f"  VEREDITO: {verdict}")
json.dump(dict(N=N,tok_clean=f"{clean}/{N}",tok_typo=f"{dirty}/{N}",tok_drop_pp=round(delta_tok,1),
    byte_drop_pp=5,winner="byte" if delta_tok>delta_byte else "token"),
    open(os.path.join(HERE,"ws25_headhead.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f}min")
