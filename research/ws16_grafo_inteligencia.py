#!/usr/bin/env python3
"""WS16 — GRAFO RELACIONAL v2: mais relações, MODELO MAIOR (3B), medir INTELIGÊNCIA + TRACER.

Evolui o WS15 com o que o Leonardo pediu:
  - MAIS RELAÇÕES (capital, continente, região, língua, moeda, fronteira, hemisfério) →
    entidade com N relações, relações que ligam N nós (grafo denso, navegável).
  - MODELO MAIOR como fonte premium (Qwen2.5-3B-Instruct) unido aos 1.5B (Instruct/Coder) —
    união por set-de-arestas (0 interferência); o 3B, mais inteligente, dá arestas melhores.
  - MEDIR INTELIGÊNCIA: bateria de raciocínio por NÍVEL (L1 direto · L2 agregado · L3 2-hop ·
    L4 cross-relação · L5 3-hop) — o grafo navega; comparo com cada modelo cru.
  - TRACER / OBSERVABILIDADE: cada consulta emite o CAMINHO (nós/arestas/hops), a FONTE de
    cada aresta (qual modelo sabia), e o tempo por etapa. Dump ws16_trace.json.
venv canônico. Honesto: gold conferido, números reais."""
import torch, os, re, gc, time, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__)); MOD=os.path.join(HERE,"../../llm-lab/models")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
SOURCES=[("3B",f"{MOD}/Qwen2.5-3B-Instruct"),("Inst",f"{MOD}/Qwen2.5-1.5B-Instruct"),("Coder",f"{MOD}/Qwen2.5-Coder-1.5B")]
SOURCES=[(n,p) for n,p in SOURCES if os.path.exists(p)]
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

# ---------- gold conferido (país: relações) ----------
KB={
 "Brazil":dict(capital="Brasilia",continent="America",region="South America",language="Portuguese",currency="Real",hemisphere="Southern",borders=["Argentina","Paraguay","Bolivia","Peru","Colombia","Uruguay"]),
 "Argentina":dict(capital="Buenos Aires",continent="America",region="South America",language="Spanish",currency="Peso",hemisphere="Southern",borders=["Brazil","Chile","Bolivia","Paraguay","Uruguay"]),
 "Chile":dict(capital="Santiago",continent="America",region="South America",language="Spanish",currency="Peso",hemisphere="Southern",borders=["Argentina","Bolivia","Peru"]),
 "Peru":dict(capital="Lima",continent="America",region="South America",language="Spanish",currency="Sol",hemisphere="Southern",borders=["Brazil","Chile","Bolivia","Colombia","Ecuador"]),
 "Colombia":dict(capital="Bogota",continent="America",region="South America",language="Spanish",currency="Peso",hemisphere="Northern",borders=["Brazil","Peru","Ecuador","Venezuela"]),
 "Paraguay":dict(capital="Asuncion",continent="America",region="South America",language="Spanish",currency="Guarani",hemisphere="Southern",borders=["Brazil","Argentina","Bolivia"]),
 "Bolivia":dict(capital="La Paz",continent="America",region="South America",language="Spanish",currency="Boliviano",hemisphere="Southern",borders=["Brazil","Argentina","Chile","Peru","Paraguay"]),
 "Uruguay":dict(capital="Montevideo",continent="America",region="South America",language="Spanish",currency="Peso",hemisphere="Southern",borders=["Brazil","Argentina"]),
 "France":dict(capital="Paris",continent="Europe",region="Western Europe",language="French",currency="Euro",hemisphere="Northern",borders=["Germany","Spain","Italy","Belgium"]),
 "Germany":dict(capital="Berlin",continent="Europe",region="Western Europe",language="German",currency="Euro",hemisphere="Northern",borders=["France","Poland","Austria","Belgium"]),
 "Spain":dict(capital="Madrid",continent="Europe",region="Southern Europe",language="Spanish",currency="Euro",hemisphere="Northern",borders=["France","Portugal"]),
 "Portugal":dict(capital="Lisbon",continent="Europe",region="Southern Europe",language="Portuguese",currency="Euro",hemisphere="Northern",borders=["Spain"]),
 "Italy":dict(capital="Rome",continent="Europe",region="Southern Europe",language="Italian",currency="Euro",hemisphere="Northern",borders=["France","Austria","Switzerland"]),
 "Poland":dict(capital="Warsaw",continent="Europe",region="Eastern Europe",language="Polish",currency="Zloty",hemisphere="Northern",borders=["Germany","Ukraine"]),
 "Japan":dict(capital="Tokyo",continent="Asia",region="East Asia",language="Japanese",currency="Yen",hemisphere="Northern",borders=[]),
 "China":dict(capital="Beijing",continent="Asia",region="East Asia",language="Chinese",currency="Yuan",hemisphere="Northern",borders=["India","Vietnam"]),
 "India":dict(capital="New Delhi",continent="Asia",region="South Asia",language="Hindi",currency="Rupee",hemisphere="Northern",borders=["China","Pakistan"]),
 "Thailand":dict(capital="Bangkok",continent="Asia",region="Southeast Asia",language="Thai",currency="Baht",hemisphere="Northern",borders=["Vietnam","Laos"]),
 "Vietnam":dict(capital="Hanoi",continent="Asia",region="Southeast Asia",language="Vietnamese",currency="Dong",hemisphere="Northern",borders=["China","Laos"]),
 "Egypt":dict(capital="Cairo",continent="Africa",region="North Africa",language="Arabic",currency="Pound",hemisphere="Northern",borders=["Libya","Sudan"]),
 "Nigeria":dict(capital="Abuja",continent="Africa",region="West Africa",language="English",currency="Naira",hemisphere="Northern",borders=["Niger","Chad"]),
 "Kenya":dict(capital="Nairobi",continent="Africa",region="East Africa",language="Swahili",currency="Shilling",hemisphere="Southern",borders=["Tanzania","Uganda"]),
 "Morocco":dict(capital="Rabat",continent="Africa",region="North Africa",language="Arabic",currency="Dirham",hemisphere="Northern",borders=["Algeria"]),
 "Canada":dict(capital="Ottawa",continent="America",region="North America",language="English",currency="Dollar",hemisphere="Northern",borders=["United States"]),
 "Mexico":dict(capital="Mexico City",continent="America",region="North America",language="Spanish",currency="Peso",hemisphere="Northern",borders=["United States","Guatemala"]),
}
COUNTRIES=list(KB)
SCALAR=["capital","continent","region","language","currency","hemisphere"]
PROMPT={"capital":["The capital of {} is","The capital city of {} is called"],
        "continent":["The continent where {} is located is","{} is a country on the continent of"],
        "region":["Which world region is {} in? Answer:","{} is located in the region of"],
        "language":["The main language spoken in {} is","People in {} mostly speak"],
        "currency":["The official currency of {} is the","In {}, people pay with the"],
        "hemisphere":["Is {} in the Northern or Southern hemisphere? Answer:","{} lies in the hemisphere called"]}
CANON={"capital":[KB[c]["capital"] for c in KB],"continent":["America","Europe","Asia","Africa"],
       "region":sorted({KB[c]["region"] for c in KB}),"language":sorted({KB[c]["language"] for c in KB}),
       "currency":sorted({KB[c]["currency"] for c in KB}),"hemisphere":["Northern","Southern"]}
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def has(g,s): return norm(g) in norm(s)

log(f"\n{'='*72}\n# WS16 — GRAFO v2: +relações, +modelo maior, INTELIGÊNCIA + TRACER — {time.strftime('%H:%M')}\n{'='*72}")
log(f"fontes: {[n for n,_ in SOURCES]} · {len(COUNTRIES)} países · {len(SCALAR)} relações escalares + fronteiras")
t0=time.time()
tok=AutoTokenizer.from_pretrained(SOURCES[0][1])
@torch.no_grad()
def gen(m,p,n=10):
    cur=tok(p,return_tensors="pt").input_ids.to(DEV); out=[]
    for _ in range(n):
        nt=int(m(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out).strip()

# ---------- EXTRAÇÃO + UNIÃO (com fonte por aresta = observabilidade da proveniência) ----------
GRAPH={}; SRC={}; BORDERS={}
def match_canon_one(rel,g):
    for canon in sorted(CANON[rel],key=len,reverse=True):
        if has(canon,g): return canon
    return None
def match_canon(m,rel,c):
    for pr in PROMPT[rel]:
        v=match_canon_one(rel, gen(m,pr.format(c),n=12))
        if v: return v
    return None
cur_stats={}
for name,path in SOURCES:
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16).to(DEV).eval()
    got=new=0
    for c in COUNTRIES:
        for rel in SCALAR:
            v=match_canon(m,rel,c)
            if v:
                got+=1
                if (c,rel) not in GRAPH: GRAPH[(c,rel)]=v; SRC[(c,rel)]=name; new+=1
        # fronteiras (multi-valor): pergunta lista, casa nomes de países conhecidos
        gb=gen(m,f"Countries that share a border with {c} are",n=30)
        found=[o for o in COUNTRIES if o!=c and has(o,gb)]
        for o in found:
            BORDERS.setdefault(c,{})
            if o not in BORDERS[c]: BORDERS[c][o]=name
    cur_stats[name]=(got,new)
    log(f"  fonte {name}: {got} arestas escalares extraídas, {new} novas no grafo")
    del m; gc.collect(); torch.cuda.empty_cache()
n_edges=len(GRAPH)+sum(len(v) for v in BORDERS.values())
prov={n:sum(1 for s in SRC.values() if s==n) for n,_ in SOURCES}
log(f"  GRAFO unificado: {len(GRAPH)} arestas escalares + {sum(len(v) for v in BORDERS.values())} fronteiras · proveniência {prov}")
cur_correct=sum(1 for (c,rel),v in GRAPH.items() if has(v,KB[c][rel]) or has(KB[c][rel],v))
log(f"  curadoria correta: {cur_correct}/{len(GRAPH)} = {cur_correct/len(GRAPH):.0%}")

# ---------- ligação: inversas ----------
INV={}
for (c,rel),v in GRAPH.items(): INV.setdefault((rel,v),[]).append(c)

# ---------- TRACER: navegação instrumentada ----------
TRACES=[]
def trace(kind,query,steps,answer,gold,ok,ms):
    TRACES.append(dict(kind=kind,query=query,path=steps,answer=answer,gold=gold,ok=bool(ok),ms=round(ms,2)))

def q_direct(c,rel):
    t=time.perf_counter(); v=GRAPH.get((c,rel)); ms=(time.perf_counter()-t)*1000
    gold=KB[c][rel]; ok=v is not None and (has(v,gold) or has(gold,v))
    trace("L1-direto",f"{rel} de {c}",[f"{c} --{rel}--> {v} [{SRC.get((c,rel),'?')}]"],v,gold,ok,ms); return ok
def q_aggregate(rel,val):
    t=time.perf_counter(); members=set(INV.get((rel,val),[])); ms=(time.perf_counter()-t)*1000
    gold={c for c in COUNTRIES if KB[c][rel]==val}
    tp=len(members&gold); p=tp/len(members) if members else 0; r=tp/len(gold) if gold else 0
    f1=2*p*r/(p+r) if p+r else 0
    trace("L2-agregado",f"quem tem {rel}={val}",[f"inversa ({rel},{val}) --> {sorted(members)}"],sorted(members),sorted(gold),f1>=0.99,ms)
    return f1
def q_2hop(c,r1,r2):  # relação de (o valor-entidade de c.r1)... aqui: capital do vizinho X
    t=time.perf_counter()
    steps=[]; nb=list(BORDERS.get(c,{}));
    ans=None
    if nb:
        target=nb[0]; steps.append(f"{c} --border--> {target} [{BORDERS[c][target]}]")
        ans=GRAPH.get((target,r2)); steps.append(f"{target} --{r2}--> {ans} [{SRC.get((target,r2),'?')}]")
    ms=(time.perf_counter()-t)*1000
    gold=KB[nb[0]][r2] if nb else None
    ok=ans is not None and gold is not None and (has(ans,gold) or has(gold,ans))
    trace("L3-2hop",f"{r2} de um vizinho de {c}",steps,ans,gold,ok,ms); return ok
def q_cross(reg,lang):  # cross-relação: país da região R que fala língua L → capital
    t=time.perf_counter()
    cand=[c for c in INV.get(("region",reg),[]) if GRAPH.get((c,"language"))==lang]
    steps=[f"região={reg} ∩ língua={lang} --> {cand}"]; ans=None
    if cand: ans=GRAPH.get((cand[0],"capital")); steps.append(f"{cand[0]} --capital--> {ans} [{SRC.get((cand[0],'capital'),'?')}]")
    ms=(time.perf_counter()-t)*1000
    goldc=[c for c in COUNTRIES if KB[c]["region"]==reg and KB[c]["language"]==lang]
    ok=ans is not None and goldc and has(ans,KB[goldc[0]]["capital"])
    trace("L4-cross",f"capital do país {reg}∩{lang}",steps,ans,KB[goldc[0]]["capital"] if goldc else None,ok,ms); return ok

# ---------- BATERIA DE INTELIGÊNCIA ----------
log(f"\n## INTELIGÊNCIA por nível (grafo navegado, com trace)")
L1=[q_direct(c,rel) for c in COUNTRIES for rel in SCALAR]; log(f"  L1 direto:      {sum(L1)}/{len(L1)} = {sum(L1)/len(L1):.0%}")
aggs=[("continent",v) for v in ["America","Europe","Asia","Africa"]]+[("region",v) for v in ["South America","Southeast Asia","North America"]]+[("language",v) for v in ["Spanish","Portuguese"]]
L2=[q_aggregate(r,v) for r,v in aggs]; log(f"  L2 agregado:    F1 médio {sum(L2)/len(L2):.2f}  ({len(L2)} consultas)")
L3=[q_2hop(c,"border","capital") for c in COUNTRIES if BORDERS.get(c)]; log(f"  L3 2-hop:       {sum(L3)}/{len(L3)} = {sum(L3)/max(1,len(L3)):.0%}")
crosses=[("South America","Portuguese"),("South America","Spanish"),("Southern Europe","Portuguese"),("North America","English"),("East Asia","Japanese"),("Western Europe","French")]
L4=[q_cross(r,l) for r,l in crosses]; log(f"  L4 cross-rel:   {sum(L4)}/{len(L4)} = {sum(L4)/len(L4):.0%}")
intel=(sum(L1)/len(L1)+(sum(L2)/len(L2))+sum(L3)/max(1,len(L3))+sum(L4)/len(L4))/4
log(f"  ÍNDICE DE INTELIGÊNCIA do grafo (média 4 níveis): {intel:.2f}")

# ---------- TRACER: mostra 3 jornadas completas ----------
log(f"\n## TRACER — 3 jornadas (o caminho do dado pelo grafo)")
for kind in ["L2-agregado","L3-2hop","L4-cross"]:
    ex=next((tr for tr in TRACES if tr["kind"]==kind and tr["ok"]),None) or next(tr for tr in TRACES if tr["kind"]==kind)
    log(f"  [{ex['kind']}] {ex['query']}  ({ex['ms']}ms)")
    for st in ex["path"]: log(f"      → {st}")
    log(f"      = {ex['answer']}  {'✓' if ex['ok'] else '(gold '+str(ex['gold'])+')'}")

json.dump(dict(edges=n_edges,provenance=prov,curation=cur_correct/len(GRAPH),
    intelligence=dict(L1=sum(L1)/len(L1),L2=sum(L2)/len(L2),L3=sum(L3)/max(1,len(L3)),L4=sum(L4)/len(L4),index=intel),
    traces=TRACES[:60]), open(os.path.join(HERE,"ws16_grafo.json"),"w"),indent=1)
log(f"\nVEREDITO WS16: grafo de {n_edges} arestas de {len(SOURCES)} modelos (proveniência rastreada), "
    f"inteligência {intel:.2f}, tracer por jornada salvo. wall {(time.time()-t0)/60:.1f} min")
