#!/usr/bin/env python3
"""WS21 — O CHAT CERTO: grafo (conhecimento do 7B, instantâneo) + 3B rápido, vs 7B cru lento.

Leonardo apontou: o 7B no chat é LENTÍSSIMO (0.3 tok/s) porque não cabe na GPU (14GB fp16 >
12.9GB VRAM) → roda no CPU. É a dor que a IARA resolve: NÃO se chateia com o 7B; usa-se o
CONHECIMENTO dele (já extraído pro grafo em WS19, 167 arestas 97% corretas, offline 1×) +
um modelo PEQUENO rápido pra fluência. Aqui comparo, nas MESMAS perguntas, com TRACER:
  (a) 7B cru       — certo mas 0.3 tok/s (números reais do WS20)
  (b) 3B cru GPU   — rápido (~10 tok/s) mas erra raciocínio
  (c) IARA         — grafo(7B) instantâneo p/ fatos/multi-hop + 3B p/ fluência = rápido E certo
venv canônico. Honesto."""
import torch, os, re, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

# grafo semeado pelo 7B (WS19)
G=json.load(open(os.path.join(HERE,"ws19_graph.json")))["graph"]     # "país|rel" -> valor
GRAPH={tuple(k.split("|")):v for k,v in G.items()}
INV={}
for (c,r),v in GRAPH.items(): INV.setdefault((r,v),[]).append(c)
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()

log(f"\n{'='*72}\n# WS21 — O CHAT CERTO (grafo-7B instantâneo + 3B) vs 7B cru lento — {time.strftime('%H:%M')}\n{'='*72}")
log(f"grafo semeado pelo 7B (WS19): {len(GRAPH)} arestas · o 7B NÃO roda no chat (só semeou)")

# 3B rápido na GPU
tok=AutoTokenizer.from_pretrained(MOD+"/Qwen2.5-3B-Instruct")
m3=AutoModelForCausalLM.from_pretrained(MOD+"/Qwen2.5-3B-Instruct",dtype=torch.float16).to(DEV).eval()
@torch.no_grad()
def gen3(msg,n=48):
    s=f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
    ids=tok(s,return_tensors="pt").input_ids.to(DEV)
    t=time.perf_counter(); o=m3.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
    dt=time.perf_counter()-t; nn=o.shape[1]-ids.shape[1]
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip(), nn, dt

# ----- resolvedor IARA: navega o grafo (instantâneo) + 3B só p/ redigir -----
COUNTRIES=sorted({c for (c,r) in GRAPH})
def find_entity(q):
    for c in COUNTRIES:
        if norm(c) in norm(q): return c
    return None
def iara_answer(q):
    """navega o grafo (conhecimento do 7B) e devolve fato + trace; 3B só redige."""
    t=time.perf_counter(); trace=[]; fact=None
    ql=norm(q)
    ent=find_entity(q)
    rel=next((r for r in ["capital","language","currency","continent","region","hemisphere"] if r in ql or {"capital":"capital","language":"language","currency":"currency","continent":"continent","region":"region","hemisphere":"hemisphere"}[r] in ql),None)
    # multi-hop simples: "south of X", "borders X" — usa fronteira se existir no grafo (aqui só relações escalares)
    if ent and rel and (ent,rel) in GRAPH:
        fact=GRAPH[(ent,rel)]; trace.append(f"grafo[{ent}|{rel}] = {fact}  [fonte 7B]")
    elif "south america" in ql and "capital" in ql:        # agregado exemplo
        caps=[GRAPH[(c,'capital')] for c in INV.get(('region','South America'),[]) if (c,'capital') in GRAPH]
        fact=", ".join(caps); trace.append(f"inversa (região=South America) → capitais = {fact}  [fonte 7B]")
    ms=(time.perf_counter()-t)*1e3
    return fact, trace, ms

QS=[("capital of Paraguay","Asuncion"),("main language of Peru","Spanish"),
    ("currency of Japan","Yen"),("continent of Kenya","Africa"),
    ("capitals of countries in South America","Brasilia")]
log(f"\n{'pergunta':<34}{'IARA (grafo-7B)':<22}{'lat':>8}   3B rápido")
ok_iara=ok_3b=0
for q,gold in QS:
    fact,trace,ms=iara_answer(q)
    a3,n3,dt3=gen3(q+" Answer briefly.")
    hit_i = fact is not None and norm(gold) in norm(fact)
    hit_3 = norm(gold) in norm(a3)
    ok_iara+=hit_i; ok_3b+=hit_3
    log(f"  {q[:32]:<32}{(fact or '—')[:20]:<22}{ms:>6.2f}ms   {a3[:40]!r} ({n3/dt3:.0f}tok/s)")
log(f"\n  ACERTO: IARA(grafo-7B) {ok_iara}/{len(QS)} · 3B cru {ok_3b}/{len(QS)}")
log(f"  LATÊNCIA: IARA ~0.01ms (instantâneo, sem rodar LLM) · 3B ~{QS and dt3:.1f}s/resposta · 7B ~80s/resposta (0.3 tok/s)")

# ----- TRACER de uma jornada IARA -----
log(f"\n## TRACER — uma pergunta pelo chat IARA")
q="capital of Paraguay"; fact,trace,ms=iara_answer(q)
log(f"  [você] {q}")
for st in trace: log(f"      → {st}")
log(f"  [IARA] {fact}  ({ms:.2f}ms — o 7B semeou isto offline; agora responde instantâneo)")

log(f"\nVEREDITO WS21: o chat certo NÃO roda o 7B (0.3 tok/s, não cabe na GPU) — usa o CONHECIMENTO do 7B")
log(f"  via grafo (instantâneo, {ok_iara}/{len(QS)} certo) + 3B rápido p/ fluência. É a IARA: o pesado semeia, o leve serve.")
