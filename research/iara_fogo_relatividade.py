#!/usr/bin/env python3
"""TESTE DE FOGO — ensinar a RELATIVIDADE à IARA e traçar TUDO (2026-07-12).

Pedido do Leonardo: ela não sabe → eu ensino → ela sabe. Rastrear cada passo contra o cérebro vivo
(iara_cerebro.py, :3050): ANTES (memória sem relatividade) → ENSINO (o professor responde) → DEPOIS
(agora da memória) → COMO A SEMENTE É GRAVADA → a ÁRVORE de relações (galhos/folhas). Salva o log."""
import urllib.request, json, time, os
BASE="http://localhost:3050"; HERE=os.path.dirname(os.path.abspath(__file__))
LOG=open(os.path.join(HERE,"iara_fogo_relatividade.log"),"w")
def out(s): print(s,flush=True); LOG.write(s+"\n")
def post(p,d):
    r=urllib.request.Request(BASE+p,json.dumps(d).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=90))
def get(p): return json.load(urllib.request.urlopen(BASE+p,timeout=30))
def show_trace(indent="      "):
    for t in get("/state")["trace"]:
        t=list(t)+[""]*(3-len(t)); out(f"{indent}[{t[0]:11}] {t[1]} {t[2]}")

out("="*74); out("# TESTE DE FOGO — ENSINAR A TEORIA DA RELATIVIDADE À IARA (rastro completo)"); out("="*74)

PERGUNTAS=["o que é a teoria da relatividade?","quem criou a relatividade?",
           "qual a fórmula da relatividade?","o que é espaço-tempo?"]
FATOS=["Relatividade é a teoria de Albert Einstein sobre espaço, tempo e gravidade",
 "Einstein é o físico que criou a teoria da relatividade",
 "E=mc² é a fórmula que diz que energia é igual a massa vezes a velocidade da luz ao quadrado",
 "Espaço-tempo é a união do espaço e do tempo em quatro dimensões",
 "Dilatação do tempo é o tempo passar mais devagar perto de massas grandes",
 "Gravidade é a curvatura do espaço-tempo causada pela massa",
 "Relatividade especial é de 1905 e a relatividade geral é de 1915"]

# ---------- FASE 0 ----------
sem0=[x['c'] for x in get("/state")["sementes"]]
out(f"\n### FASE 0 — MEMÓRIA ANTES\nsementes na memória: {sem0}")
rel0=[c for c in sem0 if any(k in c for k in ['relativ','einstein','espaco','gravidade','tempo','mc'])]
out(f"→ sementes de RELATIVIDADE: {rel0 or 'NENHUMA (ela não tem isso na memória)'}")

# ---------- FASE 1 — ANTES ----------
out(f"\n### FASE 1 — ANTES DE ENSINAR (pergunto o que ela ainda não guardou)")
for q in PERGUNTAS:
    r=post("/think",{"q":q}); out(f"\n  você: {q}\n  IARA: {r.get('say')}"); show_trace()
out("\n→ ela respondeu pelo CÓRTEX (Qwen, lento) OU abstém; ainda não há SEMENTE-FATO explícita disso.")

# ---------- FASE 2 — ENSINO ----------
out(f"\n### FASE 2 — EU ENSINO (o professor responde o que ela não sabia)")
for f in FATOS:
    r=post("/think",{"q":f}); out(f"\n  professor: {f}\n  IARA: {r.get('say')}"); show_trace()
out("\n→ cada fato virou uma SEMENTE gravada no hipocampo (dopamina ↑ a cada aprendizado).")

# ---------- FASE 3 — DEPOIS ----------
out(f"\n### FASE 3 — DEPOIS (agora ela SABE — vem da memória)")
for q in PERGUNTAS:
    r=post("/think",{"q":q}); out(f"\n  você: {q}\n  IARA: {r.get('say')}"); show_trace()
# pergunta que COMBINA fatos
r=post("/think",{"q":"quem criou a fórmula E=mc²?"}); out(f"\n  você: quem criou a fórmula E=mc²?\n  IARA: {r.get('say')}"); show_trace()
out("\n→ agora responde da MEMÓRIA (reflexo rápido / usa as sementes) — não recalcula do zero.")

# ---------- FASE 4 — COMO É GRAVADO ----------
out(f"\n### FASE 4 — COMO A SEMENTE É GRAVADA (no disco)")
sf=os.path.join(HERE,"iara_cerebro_state.json")
if os.path.exists(sf):
    K=json.load(open(sf)).get("seeds",{})
    rel=[(k,v) for k,v in K.items() if any(x in (k+' '+v.get('stmt','')).lower() for x in ['relativ','einstein','espaco','gravidade','dilata','mc','1905','1915'])]
    out(f"  sementes de relatividade salvas no disco ({sf}): {len(rel)}")
    for k,v in rel[:6]: out(f"    • chave '{k}'  →  {json.dumps(v,ensure_ascii=False)}")

# ---------- FASE 5 — A ÁRVORE ----------
out(f"\n### FASE 5 — A ÁRVORE DE RELAÇÕES (galhos e folhas)")
g=get("/graph"); nod={n['id']:n for n in g['nodes']}
seedn=[n for n in g['nodes'] if n['group'] in ('seed','reason')]
out(f"  sementes (troncos) e a que se ligam:")
from collections import defaultdict
adj=defaultdict(list)
for e in g['edges']: adj[e['from']].append(e['to'])
for n in seedn:
    galhos=adj.get(n['id'],[])
    if any(k in n['id'] for k in ['relativ','einstein','espaco','gravidade','tempo','mc','dilata']) or galhos:
        out(f"    🌱 {n['id']}  →  galhos/folhas: {galhos[:6]}")
out(f"\n  total: {len(g['nodes'])} nós, {len(g['edges'])} ligações no grafo do cérebro.")
out(f"\n{'='*74}\nFIM — log salvo em iara_fogo_relatividade.log"); LOG.close()
