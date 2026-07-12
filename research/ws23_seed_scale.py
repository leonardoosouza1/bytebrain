#!/usr/bin/env python3
"""FASE 1 — SEMEAR GRAFO GRANDE: cobertura real p/ a IARA final (2026-07-12).

Fecha o gap #1 (cobertura). Extrai um grafo GRANDE e correto do 3B (rápido, GPU, sem fault),
por AUTO-CONSISTÊNCIA (2 fraseados → mesmo 1º valor → cristaliza; sem precisar de gold predefinido,
escala p/ qualquer entidade). Checkpoint incremental. Score de acurácia numa amostra com gold."""
import torch, os, re, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md"); CKPT=os.path.join(HERE,"iara_graph_big.json")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

COUNTRIES=("Brazil Argentina Chile Peru Colombia Bolivia Uruguay Paraguay Ecuador Venezuela "
 "France Germany Spain Portugal Italy Poland Greece Netherlands Belgium Austria Sweden Norway Denmark Finland Ireland "
 "Switzerland Czechia Hungary Romania Bulgaria Croatia Serbia Ukraine Russia "
 "Japan China India Thailand Vietnam Indonesia Malaysia Philippines Pakistan Bangladesh SouthKorea Turkey Iran Iraq "
 "SaudiArabia Israel Jordan Lebanon Kazakhstan Mongolia Nepal "
 "Egypt Nigeria Kenya Morocco Ghana Ethiopia Tanzania Uganda Algeria Tunisia Senegal Angola "
 "Canada Mexico Cuba Guatemala Panama CostaRica "
 "Australia NewZealand").split()
RELS=["capital","continent","region","language","currency","hemisphere"]
PROMPT={"capital":["The capital of {} is","The capital city of {} is"],
        "continent":["The continent where {} is located is","{} is on the continent of"],
        "region":["Which world region is {} in? Answer:","{} is located in the region of"],
        "language":["The main language of {} is","People in {} mostly speak"],
        "currency":["The official currency of {} is the","In {} people pay with the"],
        "hemisphere":["Is {} in the Northern or Southern hemisphere? Answer:","{} lies in the hemisphere"]}
# gold só p/ AMOSTRA (medir acurácia)
GOLD={"Brazil":dict(capital="Bras",continent="America",language="Portuguese"),
 "France":dict(capital="Paris",continent="Europe",language="French"),
 "Japan":dict(capital="Tokyo",continent="Asia",language="Japanese"),
 "Egypt":dict(capital="Cairo",continent="Africa",language="Arabic"),
 "Peru":dict(capital="Lima",continent="America",language="Spanish"),
 "Germany":dict(capital="Berlin",continent="Europe",language="German"),
 "China":dict(capital="Beijing",continent="Asia",language="Chinese"),
 "Kenya":dict(capital="Nairobi",continent="Africa",language="Swahili"),
 "Canada":dict(capital="Ottawa",continent="America",language="English"),
 "Australia":dict(capital="Canberra",continent="Oceania",language="English")}
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def has(g,s): return norm(g) in norm(s)
def first_word(s):
    m=re.match(r"\s*([A-Za-z][A-Za-zÀ-ÿ]{1,18})",s); return m.group(1) if m else None

log(f"\n{'='*72}\n# FASE 1 — SEMEAR GRAFO GRANDE ({len(COUNTRIES)} países × {len(RELS)} relações) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(MOD+"/Qwen2.5-3B-Instruct")
m=AutoModelForCausalLM.from_pretrained(MOD+"/Qwen2.5-3B-Instruct",dtype=torch.float16).to(DEV).eval()
@torch.no_grad()
def gen(p,n=8):
    ids=tok(p,return_tensors="pt").input_ids.to(DEV)
    o=m.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()

GRAPH={}; SRC={}
if os.path.exists(CKPT):
    d=json.load(open(CKPT)); GRAPH={tuple(k.split("|")):v for k,v in d["graph"].items()}
    log(f"retomando checkpoint: {len(GRAPH)} arestas")
def save():
    json.dump({"graph":{f"{c}|{r}":v for (c,r),v in GRAPH.items()},"src":{f"{c}|{r}":'3B' for (c,r) in GRAPH}},open(CKPT,"w"))

todo=[(c,r) for c in COUNTRIES for r in RELS if (c,r) not in GRAPH]
log(f"extraindo {len(todo)} arestas por auto-consistência (2 fraseados concordam)...")
done=0
for i,(c,r) in enumerate(todo):
    ent=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",c)                      # SouthKorea -> South Korea
    try:
        a=first_word(gen(PROMPT[r][0].format(ent)))
        b=first_word(gen(PROMPT[r][1].format(ent)))
    except Exception as e:
        log(f"  ⚠ {c},{r}: {str(e)[:40]}"); continue
    if a and b and norm(a)==norm(b):                              # concordam = sólido
        GRAPH[(c,r)]=a; SRC[(c,r)]="3B"; done+=1
    if (i+1)%60==0:
        save(); el=time.time()-t0; log(f"  {i+1}/{len(todo)} · {el:.0f}s · ~{el/(i+1):.1f}s/aresta · grafo {len(GRAPH)}")
save()
log(f"GRAFO GRANDE: {len(GRAPH)} arestas de {len(COUNTRIES)} países (auto-consistência)")
# acurácia na amostra com gold
tot=cor=0
for c,g in GOLD.items():
    for r,gv in g.items():
        if (c,r) in GRAPH:
            tot+=1; cor+= has(gv,GRAPH[(c,r)]) or has(GRAPH[(c,r)],gv)
log(f"acurácia na amostra-gold: {cor}/{tot} = {cor/max(1,tot):.0%}")
cov=len(GRAPH)/(len(COUNTRIES)*len(RELS))
log(f"cobertura: {cov:.0%} das {len(COUNTRIES)*len(RELS)} arestas possíveis · ~{len(GRAPH)*6/1024:.1f}KB")
log(f"FASE 1 DONE em {(time.time()-t0)/60:.1f}min · checkpoint iara_graph_big.json")
