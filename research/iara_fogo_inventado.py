#!/usr/bin/env python3
"""TESTE DE FOGO A — domínio 100% INVENTADO (a Qwen NÃO tem isso nos pesos).
Prova: (1) "não-sei" PURO (sem o Qwen adivinhar), (2) aprende fato novo de verdade,
(3) COMBINA fatos (multi-hop), (4) o NOVO grafo semântico separa tópicos sem cross-link feio.
Roda contra o cérebro vivo (:3050). Mundo fictício de Kelbros/zorvane/Torians."""
import urllib.request,json,os
BASE="http://localhost:3050"; HERE=os.path.dirname(os.path.abspath(__file__))
LOG=open(os.path.join(HERE,"iara_fogo_inventado.log"),"w")
def out(s): print(s,flush=True); LOG.write(s+"\n")
def post(p,d):
    r=urllib.request.Request(BASE+p,json.dumps(d).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=90))
def get(p): return json.load(urllib.request.urlopen(BASE+p,timeout=30))
def trace(ind="      "):
    for t in get("/state")["trace"]:
        t=list(t)+[""]*(3-len(t)); out(f"{ind}[{t[0]:11}] {t[1]} {t[2]}")

out("="*74); out("# TESTE DE FOGO A — ENSINAR UM MUNDO INVENTADO (a Qwen não conhece)"); out("="*74)

PERG=["o que é zorvane?","quem são os Torians?","qual a capital de Kelbros?","de que é feita a nave dos Torians?"]
FATOS=["Zorvane é um metal líquido inventado no planeta Kelbros",
 "Kelbros é o terceiro planeta da estrela Vundo",
 "Zorvane derrete a 400 graus e conduz luz em vez de eletricidade",
 "Velanium é a liga de zorvane com prata",
 "As naves dos Torians são construídas de velanium",
 "Torians são a espécie que vive em Kelbros",
 "Aurthen é a capital de Kelbros"]

# FASE 1 — ANTES: ela NÃO pode saber (nada disso existe nos pesos)
out("\n### FASE 1 — ANTES (a Qwen não tem esse mundo; espero ABSTENÇÃO pura)")
n_abs=0
for q in PERG[:3]:
    r=post("/think",{"q":q}); say=r.get("say"); out(f"\n  você: {q}\n  IARA: {say}")
    if "não sei" in (say or "").lower(): n_abs+=1
    trace()
out(f"\n→ abstenções puras: {n_abs}/3  (se abstém, é o 'não-sei' REAL — o Qwen não chutou)")

# FASE 2 — ENSINO
out("\n### FASE 2 — EU ENSINO o mundo inventado")
for f in FATOS:
    r=post("/think",{"q":f}); out(f"\n  professor: {f}\n  IARA: {r.get('say')}"); trace()

# FASE 3 — DEPOIS + multi-hop
out("\n### FASE 3 — DEPOIS (agora sabe) + COMBINAÇÃO multi-hop")
for q in PERG:
    r=post("/think",{"q":q}); out(f"\n  você: {q}\n  IARA: {r.get('say')}"); trace()
# multi-hop de verdade: nave→velanium→(zorvane+prata). ninguém disse "a nave tem zorvane"
r=post("/think",{"q":"a nave dos Torians tem zorvane?"})
out(f"\n  você (multi-hop): a nave dos Torians tem zorvane?\n  IARA: {r.get('say')}"); trace()

# FASE 4 — semente no disco
out("\n### FASE 4 — COMO A SEMENTE NOVA É GRAVADA (disco)")
sf=os.path.join(HERE,"iara_cerebro_state.json")
if os.path.exists(sf):
    K=json.load(open(sf)).get("seeds",{})
    inv=[(k,v) for k,v in K.items() if any(x in (k+' '+v.get('stmt','')).lower() for x in ['zorvane','kelbros','torian','velanium','vundo','aurthen'])]
    out(f"  sementes do mundo inventado salvas: {len(inv)}")
    for k,v in inv[:8]: out(f"    • '{k}'  ({v['kind']})  →  {v['stmt']}")

# FASE 5 — o NOVO grafo semântico (tópicos + relações tipadas)
out("\n### FASE 5 — O GRAFO SEMÂNTICO (tópicos, relações, sem lixo de fragmento)")
g=get("/graph"); nod={n['id']:n for n in g['nodes']}
tops=[n for n in g['nodes'] if n['group']=='topic']
out(f"  nós: {len(g['nodes'])} · arestas: {len(g['edges'])} · TÓPICOS: {[n['label'] for n in tops]}")
from collections import defaultdict
under=defaultdict(list)
for e in g['edges']: under[e['from']].append((e.get('label','?'),e['to']))
for t in tops:
    out(f"\n  ◆ {t['label']}")
    for rel,to in under.get(t['id'],[]): out(f"       —[{rel}]→  {to}")
# checa cross-links entre tópicos (o que ele reclamou): NÃO deve existir
cross=[e for e in g['edges'] if nod.get(e['from'],{}).get('group')=='seed' and nod.get(e['to'],{}).get('group')=='seed']
out(f"\n  cross-links semente↔semente (deviam ser 0, agrupados por tópico): {len(cross)}")
out(f"\n{'='*74}\nFIM — log em iara_fogo_inventado.log"); LOG.close()
