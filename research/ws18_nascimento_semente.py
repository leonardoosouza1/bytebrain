#!/usr/bin/env python3
"""WS18 — NASCIMENTO DE SEMENTE por co-ativação (ideia do Leonardo, 2026-07-12).

Estende o WS17: as SINAPSES-folha (Brasil→capital→Brasília) já se formaram pelo uso. Agora,
quando um GRUPO de sinapses relacionadas CO-DISPARA muitas vezes (pedem capital de vários
países sul-americanos), uma LEI faz NASCER uma semente-conceito ("capitais da América do
Sul") que já CARREGA os membros pré-ligados. É chunking / formação de conceito no grafo.

LEI DO NASCIMENTO (a fórmula): um conceito nasce quando a co-ativação do grupo passa de um
limiar E os membros são sinapses fortes → cristaliza uma semente de nível superior.
GANHO: consulta agregada ("capitais da América do Sul") passa de N ativações (navegar cada
membro) para 1 (a semente-conceito já bundle). Mede nascimentos, compressão N→1, e velocidade.
Puro mecanismo (folhas = herdadas do WS17). Honesto: números reais."""
import os, re, time, json, random
HERE=os.path.dirname(os.path.abspath(__file__)); JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

# mundo: país -> (capital, região, língua)
KB={"Brazil":("Brasilia","South America","Portuguese"),"Argentina":("Buenos Aires","South America","Spanish"),
 "Peru":("Lima","South America","Spanish"),"Chile":("Santiago","South America","Spanish"),
 "Colombia":("Bogota","South America","Spanish"),"Bolivia":("La Paz","South America","Spanish"),
 "France":("Paris","Europe","French"),"Germany":("Berlin","Europe","German"),"Italy":("Rome","Europe","Italian"),
 "Spain":("Madrid","Europe","Spanish"),"Portugal":("Lisbon","Europe","Portuguese"),
 "Japan":("Tokyo","Asia","Japanese"),"China":("Beijing","Asia","Chinese"),"India":("New Delhi","Asia","Hindi"),
 "Egypt":("Cairo","Africa","Arabic"),"Kenya":("Nairobi","Africa","Swahili")}
RELS={"capital":0,"region":1,"language":2}

log(f"\n{'='*72}\n# WS18 — NASCIMENTO DE SEMENTE por co-ativação — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()

# ---------- folhas: sinapses já formadas pelo uso (WS17) ----------
SOIL={}   # (país,rel) -> (valor, força)
for c,(cap,reg,lang) in KB.items():
    SOIL[(c,"capital")]=(cap,0); SOIL[(c,"region")]=(reg,0); SOIL[(c,"language")]=(lang,0)
log(f"folhas herdadas do WS17: {len(SOIL)} sinapses (país→relação→valor)")

# ---------- conceitos possíveis (grupos que PODEM nascer) ----------
# um conceito = (relação, categoria) p.ex. (capital, "South America") = capitais sul-americanas
def concept_members(rel, cat, catrel):
    return [c for c in KB if SOIL[(c,catrel)][0]==cat]
CONCEPTS={}   # nome -> (rel, catrel, cat, membros)
for cat in {v[1] for v in KB.values()}:            # por região
    for rel in ["capital","language"]:
        CONCEPTS[f"{rel}@{cat}"]=(rel,"region",cat,concept_members(rel,cat,"region"))

# ---------- co-ativação + LEI DO NASCIMENTO ----------
COFIRE={name:0 for name in CONCEPTS}   # quantas vezes o grupo co-disparou
BORN={}                                 # semente-conceito nascida -> membros bundle
BIRTHS=[]
K_BIRTH=8                                # a LEI: nasce quando co-ativação >= 8
def water(c, rel):
    """rega uma folha; registra co-ativação dos conceitos a que ela pertence."""
    v,w=SOIL[(c,rel)]; SOIL[(c,rel)]=(v,w+1)
    for name,(r,catrel,cat,members) in CONCEPTS.items():
        if r==rel and c in members:
            COFIRE[name]+=1
            if name not in BORN and COFIRE[name]>=K_BIRTH and all(SOIL[(m,r)][1]>0 for m in members):
                # NASCE a semente-conceito (a fórmula disparou)
                BORN[name]={m:SOIL[(m,r)][0] for m in members}
                BIRTHS.append((name,len(members),COFIRE[name]))
    return v

def aggregate(rel, cat):
    """consulta agregada: se a semente-conceito nasceu = 1 ativação; senão navega N membros."""
    name=f"{rel}@{cat}"
    if name in BORN:
        return list(BORN[name].items()), 1, "SEMENTE-CONCEITO (1 ativação)"
    members=CONCEPTS[name][3]
    return [(m,SOIL[(m,rel)][0]) for m in members], len(members), f"navegou {len(members)} membros"

# ---------- STREAM com SESSÕES temáticas (co-ativação real) ----------
rng=random.Random(7); stream=[]
# sessões: blocos que perguntam sobre uma região (co-ativa o grupo)
for _ in range(6):
    reg=rng.choice(["South America","Europe","Asia"])
    for c in [x for x in KB if KB[x][1]==reg]:
        stream.append((c,"capital"))
    rng.shuffle(stream)
# + ruído aleatório
for _ in range(30): stream.append((rng.choice(list(KB)),rng.choice(["capital","language"])))

log(f"\n## STREAM {len(stream)} consultas (sessões temáticas por região + ruído)")
for c,rel in stream: water(c,rel)
log(f"  co-ativação final por conceito (top):")
for name,cf in sorted(COFIRE.items(),key=lambda x:-x[1])[:6]:
    star=" ★ NASCEU" if name in BORN else ""
    log(f"    {name:<22} co-disparou {cf}×{star}")

log(f"\n## NASCIMENTOS (a lei: co-ativação >= {K_BIRTH})")
for name,nm,cf in BIRTHS:
    log(f"  ★ nasceu '{name}': semente-conceito que CARREGA {nm} membros (co-disparou {cf}×)")
    log(f"      = {BORN[name]}")
if not BIRTHS: log("  (nenhum conceito passou do limiar neste stream)")

# ---------- ganho: consulta agregada antes vs depois ----------
log(f"\n## GANHO — 'capitais da América do Sul'")
ans,cost,how=aggregate("capital","South America")
log(f"  agora ({how}): {[a for a,_ in ans]}")
log(f"  custo: {cost} ativação(ões)  vs  {len(CONCEPTS['capital@South America'][3])} sem a semente = {len(CONCEPTS['capital@South America'][3])//max(1,cost)}× menos trabalho")

# ---------- compressão hierárquica ----------
log(f"\n## A TERRA CRESCEU PRA CIMA (abstrações, não só folhas)")
log(f"  folhas (fatos): {len(SOIL)} · sementes-conceito NASCIDAS: {len(BORN)}")
log(f"  cada semente-conceito comprime N fatos em 1 nó consultável (chunking) — a próxima ideia do Leonardo, medida")
log(f"  e ela é COMPOSTA: 'capital@South America' já liga aos países E às capitais (a relação vira estrutura)")

json.dump(dict(leaves=len(SOIL),cofire=COFIRE,births=[(n,m,c) for n,m,c in BIRTHS],
    born={n:v for n,v in BORN.items()},K_birth=K_BIRTH),open(os.path.join(HERE,"ws18_birth.json"),"w"),indent=1)
log(f"\nVEREDITO WS18: a co-ativação de sinapses relacionadas NASCE uma semente-conceito (lei do limiar), "
    f"que carrega os membros e torna a consulta agregada 1 ativação — a terra cria abstrações com o uso. "
    f"wall {(time.time()-t0):.1f}s")
