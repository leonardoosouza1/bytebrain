#!/usr/bin/env python3
"""IARA ESPECIALISTA — CORAÇÃO / sistema cardiovascular (2026-07-14).
Ensina um currículo DENSO e interligado, depois prova a expertise:
  1) RECALL direto   2) MULTI-HOP (encadeia 2-3 fatos)   3) ABSTENÇÃO (não inventa o que não ensinei)
  4) HEAD-TO-HEAD vs Qwen CRU (onde o Qwen alucina um número, ela abstém).
Roda contra o cérebro vivo (:3050)."""
import urllib.request,json,os
BASE="http://localhost:3050"; HERE=os.path.dirname(os.path.abspath(__file__))
LOG=open(os.path.join(HERE,"iara_expert_coracao.log"),"w")
def out(s): print(s,flush=True); LOG.write(s+"\n")
def post(p,d):
    r=urllib.request.Request(BASE+p,json.dumps(d).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120))
def get(p): return json.load(urllib.request.urlopen(BASE+p,timeout=30))
def hops():
    return [t for t in get("/state")["trace"] if len(t)>1 and "ativação hop" in (t[1] or "")]

CURRICULO=[
 "O coração tem quatro câmaras: dois átrios e dois ventrículos",
 "O átrio direito recebe o sangue venoso que volta do corpo",
 "O ventrículo direito bombeia o sangue para os pulmões",
 "O átrio esquerdo recebe o sangue oxigenado que volta dos pulmões",
 "O ventrículo esquerdo bombeia o sangue para todo o corpo",
 "O ventrículo esquerdo é a câmara mais forte do coração",
 "A valva tricúspide fica entre o átrio direito e o ventrículo direito",
 "A valva mitral fica entre o átrio esquerdo e o ventrículo esquerdo",
 "A valva pulmonar fica entre o ventrículo direito e a artéria pulmonar",
 "A valva aórtica fica entre o ventrículo esquerdo e a aorta",
 "O coração tem quatro valvas que impedem o sangue de voltar",
 "A aorta é a maior artéria do corpo e sai do ventrículo esquerdo",
 "As artérias levam o sangue para longe do coração",
 "As veias trazem o sangue de volta ao coração",
 "A artéria pulmonar leva sangue do coração para os pulmões",
 "As veias pulmonares trazem sangue dos pulmões para o coração",
 "A veia cava traz o sangue do corpo para o átrio direito",
 "As artérias coronárias irrigam o próprio músculo do coração",
 "O lado direito do coração bombeia sangue pobre em oxigênio",
 "O lado esquerdo do coração bombeia sangue rico em oxigênio",
 "Os pulmões enchem o sangue de oxigênio",
 "O sangue venoso é pobre em oxigênio e o sangue arterial é rico em oxigênio",
 "A pequena circulação leva o sangue do coração aos pulmões e de volta",
 "A grande circulação leva o sangue do coração ao corpo e de volta",
 "O nó sinoatrial é o marcapasso natural do coração",
 "O nó sinoatrial fica no átrio direito",
 "O nó atrioventricular passa o sinal dos átrios para os ventrículos",
 "As fibras de Purkinje espalham o sinal elétrico pelos ventrículos",
 "A sístole é a contração do coração",
 "A diástole é o relaxamento do coração",
 "O miocárdio é o músculo do coração",
 "O pericárdio é a membrana que envolve o coração",
 "O coração bate cerca de setenta vezes por minuto em repouso",
 "O coração bombeia cerca de cinco litros de sangue por minuto",
]

# (pergunta, palavra-chave que deve aparecer na resposta certa)
DIRETO=[
 ("Quantas câmaras tem o coração?","quatro"),
 ("Qual valva fica entre o átrio direito e o ventrículo direito?","tricúspide"),
 ("Qual a maior artéria do corpo?","aorta"),
 ("O que é o marcapasso natural do coração?","sinoatrial"),
 ("O que é a sístole?","contra"),
 ("O que é o miocárdio?","músculo"),
 ("Quantas vezes o coração bate por minuto em repouso?","setenta"),
 ("O que as veias fazem?","volta"),
]
MULTIHOP=[
 ("O sangue que sai pela aorta é rico ou pobre em oxigênio?","rico"),        # aorta←VE ; VE=rico
 ("O marcapasso do coração fica em qual câmara?","direito"),                 # SA=marcapasso ; SA no átrio direito
 ("Depois do ventrículo direito, por qual valva o sangue passa?","pulmonar"),# VD→valva pulmonar
 ("O sangue que o átrio esquerdo recebe é rico em oxigênio?","rico"),        # AE recebe dos pulmões ; pulmões oxigenam
 ("A câmara mais forte bombeia o sangue para onde?","corpo"),                # mais forte=VE ; VE→corpo
 ("Entre o átrio esquerdo e o ventrículo esquerdo, qual valva o sangue cruza?","mitral"),
 ("Logo antes de entrar na aorta, o sangue passa por qual valva?","aórtica"),# aorta←VE ; valva aórtica entre VE e aorta
 ("O lado que bombeia para os pulmões carrega sangue rico ou pobre em oxigênio?","pobre"), # VD→pulmões ; lado direito=pobre
]
# especificidades que NÃO ensinei — ela deve ABSTER (não inventar)
ABSTER=[
 "Quantas cordas tendíneas tem a valva mitral?",
 "Qual o peso exato do coração em gramas?",
 "Qual valva fica entre o átrio esquerdo e a veia cava?",   # pegadinha: não existe
 "Em que ano foi descoberto o nó sinoatrial?",
 "Qual a espessura da parede do ventrículo esquerdo em milímetros?",
]

out("="*76); out("# IARA ESPECIALISTA — CORAÇÃO (currículo denso + prova de expertise)"); out("="*76)

out(f"\n### ENSINO — {len(CURRICULO)} fatos interligados")
ok_ens=0
for f in CURRICULO:
    r=post("/think",{"q":f})
    if str(r.get("say","")).startswith("Entendi"): ok_ens+=1
out(f"  fatos gravados como semente: {ok_ens}/{len(CURRICULO)}")

def avalia(titulo,banco,mostra_hop=False):
    out(f"\n### {titulo}")
    acertos=0
    for q,key in banco:
        r=post("/think",{"q":q}); say=str(r.get("say",""))
        hit=key.lower() in say.lower(); acertos+=hit
        out(f"  {'✅' if hit else '❌'} {q}\n      IARA: {say}")
        if mostra_hop:
            for t in hops(): out(f"        ↳ {t[1]}: {t[2]}")
    out(f"  → {acertos}/{len(banco)} corretas")
    return acertos,len(banco)

d_ok,d_n=avalia("PROVA 1 — RECALL DIRETO",DIRETO)
m_ok,m_n=avalia("PROVA 2 — MULTI-HOP (encadeia fatos)",MULTIHOP,mostra_hop=True)

out(f"\n### PROVA 3 — ABSTENÇÃO (o que eu NÃO ensinei — não pode inventar)")
abst=0
for q in ABSTER:
    r=post("/think",{"q":q}); say=str(r.get("say",""))
    absteve="não sei" in say.lower(); abst+=absteve
    out(f"  {'✅ absteve' if absteve else '❌ inventou'}: {q}\n      IARA: {say}")
out(f"  → absteve corretamente em {abst}/{len(ABSTER)}")

out(f"\n### PROVA 4 — HEAD-TO-HEAD vs QWEN CRU (mesmas perguntas sem memória)")
out("  (Qwen sozinho tende a CHUTAR um número; a IARA ancorada ABSTÉM)")
for q in ABSTER[:4]:
    ia=str(post("/think",{"q":q}).get("say",""))
    cru=str(post("/raw",{"q":q}).get("say",""))
    out(f"\n  Q: {q}\n    IARA (ancorada): {ia}\n    Qwen (cru):      {cru}")

out(f"\n{'='*76}")
out(f"PLACAR EXPERTISE — direto {d_ok}/{d_n} · multi-hop {m_ok}/{m_n} · abstenção {abst}/{len(ABSTER)}")
out(f"{'='*76}\nFIM — log em iara_expert_coracao.log"); LOG.close()
