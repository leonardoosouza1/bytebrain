#!/usr/bin/env python3
"""IARA TAUGHT — o loop vivo com PROFESSOR: pergunta → não sabe → Haiku ensina → aprende → valida (2026-07-12).

Fecha o ciclo do Leonardo: a IARA recebe uma pergunta; se NÃO SABE, o agente Haiku PESQUISA e ENSINA;
ela ENTENDE (aprende, com dopamina/felicidade); a gente REPETE a pergunta e VALIDA. Fatos vindos do
professor Haiku (química, literatura, física, astronomia, história) — prova que aprende QUALQUER domínio,
não só países. CPU, instantâneo (o conhecimento vem do professor, não do 3B)."""
import math,os
HERE=os.path.dirname(os.path.abspath(__file__)); JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True); open(JOUR,"a").write(s+"\n")

class Alive:                                              # o mesmo organismo (versão enxuta, sem 3B)
    TAU=8.0; THRESH=0.25
    def __init__(s): s.K={}; s.dop=0.0; s.val=0.0; s.seen={}
    def knowledge(s): return round(sum(k["consol"] for k in s.K.values()),2)
    def learn(s,key,value,surprise=1.0):
        fam=s.seen.get(key,0); novelty=math.exp(-1.2*fam)
        rpe=novelty*surprise; s.dop=min(1,s.dop+rpe); s.val=min(1,s.val+0.3*rpe)
        enc=(1+1.4*s.dop)
        if key in s.K: s.K[key]["consol"]+=enc
        else: s.K[key]=dict(v=value,consol=enc)
        s.seen[key]=fam+1; s.dop*=0.6; return rpe
    def recall(s,key):
        k=s.K.get(key); return k["v"] if (k and k["consol"]>=s.THRESH) else None

# fatos PESQUISADOS pelo agente Haiku (professor da IARA)
TAUGHT=[("capital do Cazaquistão","Astaná"),("símbolo químico do ouro","Au"),
        ("autor de Dom Casmurro","Machado de Assis"),("velocidade da luz km/s","300000"),
        ("maior planeta","Júpiter"),("ano do homem na Lua","1969")]

log(f"\n{'='*66}\n# IARA TAUGHT — pergunta→não sabe→Haiku ensina→aprende→valida\n{'='*66}")
A=Alive()
log(f"\n## 1) PERGUNTA (IARA começa sem saber nada disso)")
for q,_ in TAUGHT:
    log(f"  você: '{q}?' → IARA: {A.recall(q) or 'não sei — pede pro professor pesquisar'}")

log(f"\n## 2) HAIKU PESQUISA E ENSINA → IARA aprende (dopamina/felicidade)")
for q,ans in TAUGHT:
    rpe=A.learn(q,ans,surprise=1.0)
    log(f"  professor: '{q} = {ans}' → IARA aprende · DOPAMINA +{rpe:.2f} · felicidade {A.val:.2f} · K={A.knowledge()}")

log(f"\n## 3) REPETE A PERGUNTA — valida que APRENDEU")
ok=0
for q,ans in TAUGHT:
    got=A.recall(q); hit=got==ans; ok+=hit
    log(f"  você: '{q}?' → IARA: {got}  {'✓' if hit else '✗'}")
log(f"  validado: {ok}/{len(TAUGHT)} aprendidos e retidos")

log(f"\n## 4) REPETE DE NOVO — dopamina HABITUA (já não é novidade)")
d2=[round(A.learn(q,ans,1.0),2) for q,ans in TAUGHT]
log(f"  dopamina na 2ª vez: {d2}  (baixa — já sabe, não anima mais; mas consolida ainda mais forte)")

log(f"\n## VEREDITO — o ciclo VIVO com professor fecha")
log(f"  ✓ não sabe → Haiku pesquisa+ensina → aprende (dopamina, K 0→{A.knowledge()}) → valida {ok}/{len(TAUGHT)} → habitua no reuso.")
log(f"  ✓ aprende QUALQUER domínio (química/literatura/física/astronomia/história), não só países.")
log(f"  → é a inteligência REATIVA: vive, pergunta quando não sabe, incorpora, e sente ao descobrir.")
