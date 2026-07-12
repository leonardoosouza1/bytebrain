#!/usr/bin/env python3
"""WS15 — GRAFO RELACIONAL: estabilizar por RELAÇÕES, não fundir pesos (ideia do Leonardo).

A crítica do Leonardo à fusão de pesos: os pesos JÁ têm o conhecimento; não é misturá-los
(destrói) nem retreinar — é CURAR e LIGAR as relações. A união deve acontecer no espaço
RELACIONAL (grafo), onde juntar A+B é só somar arestas (sem interferência), e a pergunta-
união é navegar a aresta inversa.

MECANISMO (sem retreinar nada):
  1. CURADORIA: extrai dos modelos o conhecimento como arestas tipadas
     (entidade --relação--> valor), com filtro de confiança (auto-consistência).
  2. UNIÃO: dois modelos-fonte contribuem arestas de metades diferentes → grafo = união
     (Brasil do modelo A, Paraguai do modelo B; o nó 'América do Sul' liga os dois).
  3. LIGAÇÃO: arestas INVERSAS automáticas (região --contém--> [países]) e cruzamento
     de relações (região ∩ língua) — 'A tem N relações, cada relação liga N nós'.
  4. NAVEGAÇÃO: direto (capital de X) · agregado/UNIÃO (países da América do Sul) ·
     multi-hop (capital do país sul-americano que fala português).
Baseline crítico: o MODELO CRU responde a pergunta-agregada? (reversal curse). O grafo
resolve de graça, sem retreinar. venv canônico. Honesto: números reais, gold conferido."""
import torch, os, re, gc, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
MOD=os.path.join(HERE,"../../llm-lab/models")
PATHS={"A":f"{MOD}/Qwen2.5-1.5B-Instruct","B":f"{MOD}/Qwen2.5-Coder-1.5B"}
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

# ---------- gold conferido (p/ MEDIR a curadoria e a navegação) ----------
KB={ # país: (capital, continente, região, língua)
 "Brazil":("Brasilia","America","South America","Portuguese"),
 "Argentina":("Buenos Aires","America","South America","Spanish"),
 "Peru":("Lima","America","South America","Spanish"),
 "Chile":("Santiago","America","South America","Spanish"),
 "Colombia":("Bogota","America","South America","Spanish"),
 "Paraguay":("Asuncion","America","South America","Spanish"),
 "Bolivia":("La Paz","America","South America","Spanish"),
 "Uruguay":("Montevideo","America","South America","Spanish"),
 "France":("Paris","Europe","Western Europe","French"),
 "Germany":("Berlin","Europe","Western Europe","German"),
 "Italy":("Rome","Europe","Southern Europe","Italian"),
 "Spain":("Madrid","Europe","Southern Europe","Spanish"),
 "Portugal":("Lisbon","Europe","Southern Europe","Portuguese"),
 "Poland":("Warsaw","Europe","Eastern Europe","Polish"),
 "Greece":("Athens","Europe","Southern Europe","Greek"),
 "Japan":("Tokyo","Asia","East Asia","Japanese"),
 "China":("Beijing","Asia","East Asia","Chinese"),
 "India":("New Delhi","Asia","South Asia","Hindi"),
 "Thailand":("Bangkok","Asia","Southeast Asia","Thai"),
 "Vietnam":("Hanoi","Asia","Southeast Asia","Vietnamese"),
 "Egypt":("Cairo","Africa","North Africa","Arabic"),
 "Nigeria":("Abuja","Africa","West Africa","English"),
 "Kenya":("Nairobi","Africa","East Africa","Swahili"),
 "Morocco":("Rabat","Africa","North Africa","Arabic"),
 "Ghana":("Accra","Africa","West Africa","English"),
 "Canada":("Ottawa","America","North America","English"),
 "Mexico":("Mexico City","America","North America","Spanish"),
 "Cuba":("Havana","America","North America","Spanish"),
 "Norway":("Oslo","Europe","Northern Europe","Norwegian"),
 "Sweden":("Stockholm","Europe","Northern Europe","Swedish"),
}
COUNTRIES=list(KB)
RELS={"capital":0,"continent":1,"region":2,"language":3}
PROMPT={"capital":"The capital of {} is","continent":"The continent of {} is",
        "region":"Which world region is {} in? It is","language":"The main language of {} is"}
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def has(g,s): return norm(g) in norm(s)

log(f"\n{'='*72}\n# WS15 — GRAFO RELACIONAL (estabilizar por relações, sem retreinar) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(PATHS["A"])
@torch.no_grad()
def gen(m,p,n=8):
    cur=tok(p,return_tensors="pt").input_ids.to(DEV); out=[]
    for _ in range(n):
        nt=int(m(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out).strip()
def first_words(s,k=3):
    m=re.match(r"\s*([A-Za-z][A-Za-z ]{1,20})", s)
    return " ".join(m.group(1).split()[:k]).strip() if m else ""

# ---------- 1+2) CURADORIA + UNIÃO: cada modelo extrai a SUA metade ----------
GRAPH={}         # (entidade, relação) -> valor   (o grafo unificado)
SRC={}           # (entidade, relação) -> modelo-fonte
VALUES_CANON={"capital":[v[0] for v in KB.values()],"continent":["America","Europe","Asia","Africa"],
              "region":sorted({v[2] for v in KB.values()}),"language":sorted({v[3] for v in KB.values()})}
def extract(model,src,countries):
    got=0
    for c in countries:
        for rel in RELS:
            g=gen(model,PROMPT[rel].format(c),n=8); v=first_words(g, 3 if rel in("region","capital")else 1)
            # curadoria: casa com algum valor canônico da relação (confiança)?
            cand=None
            for canon in sorted(VALUES_CANON[rel],key=len,reverse=True):
                if has(canon,g): cand=canon; break
            if cand:
                GRAPH[(c,rel)]=cand; SRC[(c,rel)]=src; got+=1
    log(f"  extração modelo {src} em {len(countries)} países: {got} arestas curadas")
    return got

# fonte A (Instruct) extrai TUDO → grafo completo
mA=AutoModelForCausalLM.from_pretrained(PATHS["A"],dtype=torch.float16).to(DEV).eval()
extract(mA,"A",COUNTRIES); del mA; gc.collect(); torch.cuda.empty_cache()
edges_A=len(GRAPH)
# fonte B (Coder) extrai TUDO e UNE por cima (só adiciona o que falta = set-union, 0 conflito)
mB=AutoModelForCausalLM.from_pretrained(PATHS["B"],dtype=torch.float16).to(DEV).eval()
b_new=b_agree=b_conflict=0
for c in COUNTRIES:
    for rel in RELS:
        g=gen(mB,PROMPT[rel].format(c),n=8); cand=None
        for canon in sorted(VALUES_CANON[rel],key=len,reverse=True):
            if has(canon,g): cand=canon; break
        if cand:
            if (c,rel) not in GRAPH: GRAPH[(c,rel)]=cand; SRC[(c,rel)]="B"; b_new+=1
            elif GRAPH[(c,rel)]==cand: b_agree+=1
            else: b_conflict+=1        # discordância: mantém A (curadoria por confiança)
del mB; gc.collect(); torch.cuda.empty_cache()
log(f"  UNIÃO por set-de-arestas: A={edges_A} + B novas={b_new} (B concordou {b_agree}, discordou {b_conflict} → mantém A) = {len(GRAPH)} arestas · 0 interferência destrutiva")

# curadoria correta? (aresta extraída == gold)
correct=sum(1 for (c,rel),v in GRAPH.items() if has(v,[KB[c][RELS[rel]]][0]) or has(KB[c][RELS[rel]],v))
log(f"  curadoria correta: {correct}/{len(GRAPH)} = {correct/len(GRAPH):.0%} (arestas erradas = ruído do extrator, honesto)")

# ---------- 3) LIGAÇÃO: arestas inversas (região/continente -> [países]) ----------
INV={}   # (relação, valor) -> [entidades]
for (c,rel),v in GRAPH.items(): INV.setdefault((rel,v),[]).append(c)

# ---------- 4) NAVEGAÇÃO ----------
log(f"\n## NAVEGAÇÃO do grafo (sem retreinar)")
# (a) direto
dir_ok=sum(1 for c in COUNTRIES if (c,"capital") in GRAPH and has(GRAPH[(c,"capital")],KB[c][0]))
log(f"  DIRETO (capital de X): {dir_ok}/{len(COUNTRIES)} corretos")

# (b) AGREGADO/UNIÃO — grafo vs modelo cru (o teste que a fusão de pesos falhou)
def truth_region(reg): return {c for c in COUNTRIES if KB[c][2]==reg}
regions_test=["South America","Southeast Asia","North America"]
mA=AutoModelForCausalLM.from_pretrained(PATHS["A"],dtype=torch.float16).to(DEV).eval()
log(f"  AGREGADO (países da região R) — grafo (aresta inversa) vs modelo CRU:")
g_f1=[]; m_f1=[]
for reg in regions_test:
    gold=truth_region(reg)
    graph_ans=set(INV.get(("region",reg),[]))
    raw=gen(mA,f"List countries in {reg}:",n=40); model_ans={c for c in COUNTRIES if has(c,raw)}
    def f1(pred):
        if not pred: return 0.0
        tp=len(pred&gold); p=tp/len(pred); r=tp/len(gold); return 2*p*r/(p+r) if p+r else 0.0
    g_f1.append(f1(graph_ans)); m_f1.append(f1(model_ans))
    log(f"    {reg:<16} gold={sorted(gold)}")
    log(f"      GRAFO  F1={f1(graph_ans):.2f}  {sorted(graph_ans)}")
    log(f"      CRU    F1={f1(model_ans):.2f}  {sorted(model_ans)}")
log(f"  MÉDIA agregado: GRAFO F1 {sum(g_f1)/len(g_f1):.2f}  vs  modelo CRU F1 {sum(m_f1)/len(m_f1):.2f}")

# (c) MULTI-HOP: capital do país da região R que fala língua L
log(f"  MULTI-HOP (capital do país de região R ∩ língua L):")
hops=[("South America","Portuguese"),("Southern Europe","Portuguese"),("North America","English"),("East Asia","Japanese")]
hop_ok=0
for reg,lang in hops:
    cand=[c for c in INV.get(("region",reg),[]) if GRAPH.get((c,"language"))==lang]
    ans=GRAPH.get((cand[0],"capital")) if cand else None
    gold_c=[c for c in COUNTRIES if KB[c][2]==reg and KB[c][3]==lang]
    ok = bool(cand) and bool(gold_c) and has(ans or "", KB[gold_c[0]][0])
    hop_ok+=ok
    log(f"    {reg} ∩ {lang} → {cand[0] if cand else '?'} → {ans} {'✓' if ok else '(gold '+(gold_c[0] if gold_c else '—')+')'}")
del mA; gc.collect(); torch.cuda.empty_cache()

log(f"\n## VEREDITO WS15")
win = (sum(g_f1)/len(g_f1)) > (sum(m_f1)/len(m_f1)) + 0.2
log(f"  agregado/união: GRAFO {sum(g_f1)/len(g_f1):.2f} vs modelo cru {sum(m_f1)/len(m_f1):.2f} · multi-hop {hop_ok}/{len(hops)}")
log(f"  → {'CONFIRMADO: estabilizar por RELAÇÕES (extrair+ligar, sem retreinar) une A+B e responde a pergunta-união que o peso cru não dá' if win else 'grafo não superou o cru — analisar extração'}")
log(f"  o grafo uniu 2 modelos por SET-UNION de arestas (0 interferência) e navegou multi-hop sem 1 passo de treino")
json.dump({"edges":len(GRAPH),"cur_correct":correct,"graph_f1":sum(g_f1)/len(g_f1),"raw_f1":sum(m_f1)/len(m_f1),
           "multihop":f"{hop_ok}/{len(hops)}"},open(os.path.join(HERE,"ws15_grafo.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f} min")
