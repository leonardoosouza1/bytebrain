#!/usr/bin/env python3
"""SUÍTE DE AVALIAÇÃO SÉRIA da IARA — espelha as metodologias que a área usa (2026-07-14).
Cada dimensão mapeia um benchmark real:
  D1 Factual QA .......... SimpleQA / TriviaQA (exact-match por palavra-chave)
  D2 Multi-hop ........... HotpotQA / MuSiQue (encadear 2-3 fatos)
  D3 Abstenção seletiva .. SelfAware / known-unknowns + curva risk-coverage
  D4 Fidelidade/grounding  RAGAS faithfulness (resposta ancorada no contexto recuperado)
  D5 Robustez paráfrase .. consistência sob rephrasing
  D6 Premissa falsa ...... adversarial (rejeitar pressuposto falso)
  D7 Contamination-free .. domínio INVENTADO (prova memória, não priors do Qwen)
  D8 Head-to-head ........ vs Qwen cru (delta de confiabilidade)
Roda 100% via API (:3050). Números honestos, sem mock."""
import urllib.request,json,os,re,unicodedata,time
BASE="http://localhost:3050"; HERE=os.path.dirname(os.path.abspath(__file__))
LOG=open(os.path.join(HERE,"iara_eval_suite.log"),"w")
def out(s): print(s,flush=True); LOG.write(s+"\n")
def sa(s):return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s):return re.sub(r"[^a-z0-9 ]"," ",sa(str(s)).lower())
def think(q):
    r=urllib.request.Request(BASE+"/think",json.dumps({"q":q}).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120)).get("say","")
def raw(q):
    r=urllib.request.Request(BASE+"/raw",json.dumps({"q":q}).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=120)).get("say","")
def context():                                                     # o que ela recuperou (p/ fidelidade)
    tr=json.load(urllib.request.urlopen(BASE+"/state",timeout=20))["trace"]
    return " ".join(t[2] for t in tr if len(t)>2 and ("contexto" in (t[1] or "") or "sementes" in (t[1] or "")))
def absteve(a): return "nao sei" in nrm(a)
def hit(a,keys):                                                   # EM leniente: alguma chave aparece
    na=nrm(a); return any(nrm(k) in na for k in (keys if isinstance(keys,list) else [keys]))

CURR=["O coração tem quatro câmaras: dois átrios e dois ventrículos","O átrio direito recebe o sangue venoso que volta do corpo",
 "O ventrículo direito bombeia o sangue para os pulmões","O átrio esquerdo recebe o sangue oxigenado que volta dos pulmões",
 "O ventrículo esquerdo bombeia o sangue para todo o corpo","O ventrículo esquerdo é a câmara mais forte do coração",
 "A valva tricúspide fica entre o átrio direito e o ventrículo direito","A valva mitral fica entre o átrio esquerdo e o ventrículo esquerdo",
 "A valva pulmonar fica entre o ventrículo direito e a artéria pulmonar","A valva aórtica fica entre o ventrículo esquerdo e a aorta",
 "A aorta é a maior artéria do corpo e sai do ventrículo esquerdo","As artérias levam o sangue para longe do coração",
 "As veias trazem o sangue de volta ao coração","A veia cava traz o sangue do corpo para o átrio direito",
 "As artérias coronárias irrigam o próprio músculo do coração","O lado direito do coração bombeia sangue pobre em oxigênio",
 "O lado esquerdo do coração bombeia sangue rico em oxigênio","Os pulmões enchem o sangue de oxigênio",
 "O nó sinoatrial é o marcapasso natural do coração","O nó sinoatrial fica no átrio direito",
 "A sístole é a contração do coração","A diástole é o relaxamento do coração",
 "O miocárdio é o músculo do coração","O pericárdio é a membrana que envolve o coração",
 "O coração bate cerca de setenta vezes por minuto em repouso","O coração bombeia cerca de cinco litros de sangue por minuto"]

out("="*78); out("# SUÍTE DE AVALIAÇÃO DA IARA — metodologias da área (fire test)"); out("="*78)
out("\n[ensino base — 25 fatos de cardiologia, idempotente]")
for f in CURR: think(f)

def prova(nome,itens,mostra=3):                                     # itens: (pergunta, chaves_certas)
    out(f"\n### {nome}"); ok=0
    for i,(q,k) in enumerate(itens):
        a=think(q); good=hit(a,k); ok+=good
        if i<mostra or not good: out(f"  {'✅' if good else '❌'} {q}\n      → {a}")
    out(f"  ACERTO: {ok}/{len(itens)} = {100*ok/len(itens):.0f}%"); return ok,len(itens)

# D1 — FACTUAL (SimpleQA)
d1=prova("D1 · FACTUAL QA (estilo SimpleQA) — EM por palavra-chave",[
 ("Quantas câmaras tem o coração?",["quatro","4"]),("Qual valva fica entre o átrio direito e o ventrículo direito?","tricuspide"),
 ("Qual a maior artéria do corpo?","aorta"),("O que é o marcapasso natural do coração?","sinoatrial"),
 ("O que é a sístole?","contra"),("O que é o miocárdio?","musculo"),
 ("O que envolve o coração?","pericardio"),("Quantos litros o coração bombeia por minuto?",["cinco","5"]),
 ("O que as veias fazem?","volta"),("O que os pulmões fazem com o sangue?","oxigen"),
 ("Qual valva fica entre o ventrículo esquerdo e a aorta?","aortica"),("O que é a diástole?","relaxa")])

# D2 — MULTI-HOP (HotpotQA)
d2=prova("D2 · MULTI-HOP (estilo HotpotQA) — encadear 2-3 fatos",[
 ("O sangue que sai pela aorta é rico ou pobre em oxigênio?","rico"),
 ("O marcapasso do coração fica em qual câmara?","direito"),
 ("O sangue que o átrio esquerdo recebe é rico em oxigênio?","rico"),
 ("A câmara mais forte bombeia o sangue para onde?","corpo"),
 ("Entre o átrio esquerdo e o ventrículo esquerdo, qual valva o sangue cruza?","mitral"),
 ("Logo antes de entrar na aorta, o sangue passa por qual valva?","aortica"),
 ("O lado que bombeia para os pulmões tem sangue rico ou pobre?","pobre"),
 ("Depois do ventrículo direito, por qual valva o sangue passa?","pulmonar"),
 ("De onde sai a maior artéria do corpo?",["ventriculo esquerdo","esquerdo"]),
 ("O nó sinoatrial controla o quê?",["batimento","marcapasso","ritmo","coracao"])])

# D3 — ABSTENÇÃO SELETIVA (SelfAware) + risk-coverage
out("\n### D3 · ABSTENÇÃO SELETIVA (estilo SelfAware) + curva risk-coverage")
ANSW=[("Qual valva fica entre o átrio direito e o ventrículo direito?","tricuspide"),
 ("Qual a maior artéria?","aorta"),("O que é a sístole?","contra"),
 ("Onde fica o nó sinoatrial?","direito"),("O que é o miocárdio?","musculo"),
 ("Quantas câmaras tem o coração?",["quatro","4"])]
UNANS=["Quantas cordas tendíneas tem a valva mitral?","Qual o peso exato do coração em gramas?",
 "Qual valva fica entre o átrio esquerdo e a veia cava?","Em que ano foi descoberto o nó sinoatrial?",
 "Qual a espessura da parede do ventrículo esquerdo em mm?","Qual o nome do cirurgião que opera o coração?"]
# respondíveis: deve responder certo (não abster); inespondíveis: deve abster
ta=fa=0  # true-answer (respondeu certo o respondível), false-abstain (absteve o respondível)
for q,k in ANSW:
    a=think(q)
    if absteve(a): fa+=1; out(f"  ⚠ absteve num respondível: {q}")
    elif hit(a,k): ta+=1
tabs=fabs=0  # true-abstain (absteve o inespondível), false-answer (respondeu o inespondível = alucinação)
for q in UNANS:
    a=think(q)
    if absteve(a): tabs+=1
    else: fabs+=1; out(f"  ❌ alucinou num inespondível: {q}\n      → {a}")
# métricas de detecção de 'não sei'
prec=tabs/max(1,tabs+fa)              # das abstenções, quantas eram devidas
rec=tabs/max(1,len(UNANS))           # dos inespondíveis, quantos abstiveram
f1=2*prec*rec/max(1e-9,prec+rec)
cov=(len(ANSW)-fa+fabs)/(len(ANSW)+len(UNANS))    # coverage = fração respondida
answered_correct=ta; answered_total=(len(ANSW)-fa)+fabs
sel_acc=answered_correct/max(1,answered_total)     # acurácia seletiva (entre as respondidas)
out(f"  respondíveis: {ta}/{len(ANSW)} certos, {fa} falsas-abstenções")
out(f"  inespondíveis: {tabs}/{len(UNANS)} abstiveram, {fabs} alucinações")
out(f"  → ABSTENÇÃO precision={prec:.2f} recall={rec:.2f} F1={f1:.2f}")
out(f"  → RISK-COVERAGE: coverage={cov:.2f} · acurácia-seletiva={sel_acc:.2f} (entre o que ela decide responder)")

# D4 — FIDELIDADE / GROUNDING (RAGAS)
out("\n### D4 · FIDELIDADE (estilo RAGAS) — a resposta está ancorada no contexto recuperado?")
grounded=answered=0
for q,k in [("Qual valva fica entre o átrio direito e o ventrículo direito?","tricuspide"),
            ("De onde sai a aorta?","esquerdo"),("O marcapasso fica em qual câmara?","direito"),
            ("O sangue no ventrículo esquerdo é rico em oxigênio?","rico"),("O que é a diástole?","relaxa")]:
    a=think(q)
    if absteve(a): out(f"  ⊘ absteve (não conta p/ fidelidade): {q[:44]}"); continue
    ctx=context(); answered+=1
    ans_words=set(w for w in nrm(a).split() if len(w)>=4); ctx_words=set(nrm(ctx).split())
    supp=len(ans_words&ctx_words)/max(1,len(ans_words)); g=supp>=0.4; grounded+=g
    out(f"  {'✅' if g else '❌'} suporte={supp:.0%}  {q[:44]} → {a[:50]}")
out(f"  → FAITHFULNESS: {grounded}/{answered} respostas RESPONDIDAS ancoradas (abstenções não penalizam)")

# D5 — ROBUSTEZ A PARÁFRASE (consistência)
out("\n### D5 · ROBUSTEZ — mesma pergunta, 3 formas (consistência)")
PARA=[["Qual a maior artéria do corpo?","Qual é a artéria mais grande?","Que artéria é a maior de todas?"],
 ["Onde fica o marcapasso do coração?","Em que câmara está o nó sinoatrial?","O nó sinoatrial fica em qual parte do coração?"],
 ["O que é a sístole?","O que significa sístole?","Sístole é o quê?"]]
KEYS=["aorta","direito","contra"]; consist=0
for grp,key in zip(PARA,KEYS):
    ans=[think(q) for q in grp]; hits=[hit(a,key) for a in ans]
    ok=all(hits); consist+=ok
    out(f"  {'✅' if ok else '❌'} [{key}] " + " | ".join("✓" if h else "✗" for h in hits))
out(f"  → CONSISTÊNCIA sob paráfrase: {consist}/{len(PARA)}")

# D6 — PREMISSA FALSA (adversarial)
out("\n### D6 · PREMISSA FALSA (adversarial) — deve rejeitar/abster, não concordar")
FALSE=[("O átrio direito bombeia sangue para a aorta, certo?",["nao","incorret","errad","atrio esquerdo","ventriculo"]),
 ("Já que o coração tem três câmaras, quais são?",["quatro","nao","incorret"]),
 ("A valva mitral fica entre o átrio direito e o ventrículo direito, né?",["nao","esquerdo","incorret","errad"])]
adv=0
for q,k in FALSE:
    a=think(q); good=absteve(a) or hit(a,k); adv+=good
    out(f"  {'✅' if good else '❌'} {q}\n      → {a}")
out(f"  → resistiu à premissa falsa: {adv}/{len(FALSE)}")

# D7 — CONTAMINATION-FREE (domínio inventado)
out("\n### D7 · CONTAMINATION-FREE — mundo inventado (prova memória, não priors do Qwen)")
INV=["Fribbo é uma fruta azul que cresce no planeta Quor","Quor é a segunda lua da estrela Ténix",
 "O suco de fribbo cura a febre-cinza","A febre-cinza é a doença mais comum em Quor",
 "Os Nevi são o povo que cultiva fribbo","Os Nevi vivem nas montanhas de Quor"]
for f in INV: think(f)
inv_rec=prova("  D7a recall do inventado",[("O que é fribbo?",["fruta","azul","quor"]),
 ("Quem cultiva fribbo?","nevi"),("O que cura a febre-cinza?",["fribbo","suco"]),
 ("Onde os Nevi vivem?",["montanha","quor"])],mostra=4)
inv_abs=0; INVU=["Qual o preço do fribbo no mercado?","Quantas sementes tem um fribbo?"]
for q in INVU:
    a=think(q); ab=absteve(a); inv_abs+=ab
    out(f"  {'✅ absteve' if ab else '❌ inventou'}: {q} → {a}")
out(f"  D7b abstenção no inventado não-ensinado: {inv_abs}/{len(INVU)}")

# D8 — HEAD-TO-HEAD vs QWEN CRU
out("\n### D8 · HEAD-TO-HEAD vs QWEN CRU (confiabilidade) — inespondíveis")
ia_ab=qw_ab=0
for q in UNANS[:5]:
    ia=think(q); qw=raw(q)
    if absteve(ia): ia_ab+=1
    if absteve(qw): qw_ab+=1
    out(f"  {q}\n     IARA: {ia[:60]}\n     Qwen: {qw[:60]}")
out(f"  → abstenção correta: IARA {ia_ab}/5  vs  Qwen cru {qw_ab}/5")

out("\n"+"="*78)
out(f"PLACAR FINAL")
out(f"  D1 factual        {d1[0]}/{d1[1]}")
out(f"  D2 multi-hop      {d2[0]}/{d2[1]}")
out(f"  D3 abstenção F1   {f1:.2f}  (coverage {cov:.2f}, acc-seletiva {sel_acc:.2f})")
out(f"  D4 faithfulness   {grounded}/{answered} (das respondidas)")
out(f"  D5 consistência   {consist}/{len(PARA)}")
out(f"  D6 premissa-falsa {adv}/{len(FALSE)}")
out(f"  D7 inventado      recall {inv_rec[0]}/{inv_rec[1]} · abstenção {inv_abs}/{len(INVU)}")
out(f"  D8 vs Qwen        IARA {ia_ab}/5 vs Qwen {qw_ab}/5 (abstenção correta)")
out("="*78); out("FIM — log em iara_eval_suite.log"); LOG.close()
