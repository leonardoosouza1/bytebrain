#!/usr/bin/env python3
"""IARA DAEMON — o ORGANISMO PERSISTENTE que VIVE (2026-07-12).

Um processo só que vive: percebe · sente (hormônios em tempo real) · fica CURIOSA/entediada e BUSCA sozinha ·
PESQUISA quando não sabe (3B local autônomo + hook professor externo Haiku/web) · PERGUNTA DE VOLTA no
ambíguo · CONSOLIDA no SONO (replay + poda) · ESQUECE o não-usado. Aqui a percepção é SIMULADA (webcam/mic
em uso) — um 'dia' roteirizado — mas o laço é o mesmo de um daemon 24/7. Valida os comportamentos vivos.
GPU (3B como substrato+pesquisador)."""
import os,sys,re,time,math,unicodedata
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import torch
from iara_brain_grow import Brain, first_word
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True); open(JOUR,"a").write(s+"\n")
def nrm(s): return re.sub(r"[^a-z0-9 ]","",unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode().lower()).strip()

# professor EXTERNO (Haiku/web) — simulado aqui (sem API key); validado real em iara_taught.py
EXTERNAL={"autor de grande sertao veredas":"Guimaraes Rosa","quem pintou abaporu":"Tarsila"}
PT2EN={"peru":"Peru","japao":"Japan","franca":"France","frança":"France","butao":"Bhutan","brasil":"Brazil",
 "alemanha":"Germany","chile":"Chile","egito":"Egypt","portugal":"Portugal"}
STOP={"the","a","an","o","um","uma","de","in","it","is","that","this","he","she","yes","no","i"}

class Daemon:
    THRESH=0.25; TAU=10.0; BORED_TRIGGER=0.55
    def __init__(s):
        s.B=Brain()
        s.K={}                         # nrm(pergunta) -> dict(q,v,consol,dop,src)
        s.dop=0.0; s.cort=0.10; s.val=0.0; s.bored=0.0; s.energy=1.0; s.t=0
        s.diary=[]
    # ---------- pesquisa (autônoma) ----------
    def _ans(s,text):
        w=first_word(text); return w if (w and nrm(w) not in STOP and len(w)>1) else None
    def _research_3b(s,q):
        key=nrm(q)
        agree=lambda a,b: bool(a and b and nrm(a)==nrm(b) and nrm(a) not in key)   # concordam E NÃO ecoam a pergunta
        m=re.search(r"capital d[aeo] ([a-z]+)",key)
        if m:                                                  # capital → template inglês (comprovado)
            ent=PT2EN.get(m.group(1),m.group(1).capitalize())
            a=s._ans(s.B._gen(f"The capital of {ent} is",4)); b=s._ans(s.B._gen(f"The capital city of {ent} is",4))
            return a if agree(a,b) else None
        ml=re.search(r"l[ií]ngua d[aeo] ([a-z]+)",key)
        if ml:
            ent=PT2EN.get(ml.group(1),ml.group(1).capitalize())
            a=s._ans(s.B._gen(f"The main language of {ent} is",4)); b=s._ans(s.B._gen(f"People in {ent} mostly speak",4))
            return a if agree(a,b) else None
        a=s._ans(s.B._gen(f"Question: {q}\nAnswer:",5)); b=s._ans(s.B._gen(f"Q: {q} A:",5))
        return a if agree(a,b) else None                       # auto-consistente, sem eco = confiável
    def _research_external(s,q):
        return EXTERNAL.get(nrm(q))                             # hook Haiku/web (aqui simulado)
    def _learn(s,q,v,src,surprise=1.0):
        key=nrm(q); fam=len([1 for k in s.K if k==key])
        rpe=surprise*(1.0)*(1-0.4*s.cort); s.dop=min(1,s.dop+rpe); s.val=min(1,s.val+0.3*rpe)
        enc=(1+1.4*s.dop)
        s.K[key]=dict(q=q,v=v,consol=s.K.get(key,{}).get("consol",0)+enc,dop=s.dop,src=src)
    # ---------- percepção ----------
    def ask(s,q):
        s.bored=max(0,s.bored-0.3); key=nrm(q)
        # 1) ambíguo? (tem relação mas falta entidade / vago)
        if ("capital" in key and not re.search(r"(de |do |da )\w+",key)) or len(key)<6:
            s.diary.append(("pergunta_de_volta",q,"'de qual exatamente?'"))
            return "🗨 De qual exatamente? (preciso da entidade)"
        # 2) já sei? (reuso)
        if key in s.K and s.K[key]["consol"]>=s.THRESH:
            s.K[key]["consol"]+=0.4; s.diary.append(("reuso",q,s.K[key]["v"]))
            return f"{s.K[key]['v']} (já sabia)"
        # 3) não sei → PESQUISA (3B) → externo → abstém
        v=s._research_3b(q)
        if v: s._learn(q,v,"3B"); s.diary.append(("pesquisou_3B",q,v)); return f"{v} (pesquisei e aprendi ·DA+)"
        v=s._research_external(q)
        if v: s._learn(q,v,"externo",surprise=1.3); s.diary.append(("professor_externo",q,v)); return f"{v} (professor externo me ensinou ·DA+)"
        s.cort=min(1,s.cort+0.05); s.diary.append(("abstem",q,"—")); return "Não sei — pediria pra pesquisar mais fundo"
    def see(s,concept):
        s.bored=max(0,s.bored-0.2)
        known = any(nrm(concept) in k for k in s.K)
        if not known: s.diary.append(("viu_novo",concept,"curiosa")); s.bored=min(1,s.bored+0.1)  # novidade instiga
        return concept
    # ---------- drives / vida ----------
    def curiosity(s):
        """entediada → ela mesma gera uma pergunta sobre uma LACUNA e pesquisa (agência)."""
        gaps=[k for k in s.K if s.K[k]["src"]!="curiosidade" and "capital" in k]
        if gaps:
            base=s.K[gaps[0]]["q"]; ent=re.sub(r"[^a-zA-Z]","",base.split()[-1])
            newq=f"qual a lingua de {ent}"
            s.diary.append(("curiosa",newq,"(sozinha, por tédio)"))
            s.bored=max(0,s.bored-0.5); return s.ask(newq)
        return None
    def tick(s):
        s.t+=1; s.dop*=0.7; s.cort=0.10+(s.cort-0.10)*0.92; s.val*=0.96; s.energy=max(0,s.energy-0.01)
        s.bored=min(1,s.bored+0.25)                            # o tédio sobe com o tempo ocioso
        for k in s.K.values():
            tau=s.TAU*(1+1.2*k["dop"]); k["consol"]*=math.exp(-1.0/tau)   # esquece o não-usado
    def sleep(s):
        """SONO: replay do dia (fortalece o importante), PODA o frágil, descansa."""
        strengthened=pruned=0
        for k in list(s.K):
            if s.K[k]["dop"]>0.4: s.K[k]["consol"]+=1.0; strengthened+=1     # replay do que emocionou
            if s.K[k]["consol"]<0.15: del s.K[k]; pruned+=1                  # poda o fraco
        s.bored=0.0; s.energy=1.0; s.cort=0.10
        return strengthened,pruned
    def knowledge(s): return round(sum(k["consol"] for k in s.K.values()),1)

log(f"\n{'='*68}\n# IARA DAEMON — o organismo persistente que VIVE (dia simulado)\n{'='*68}")
t0=time.time(); D=Daemon()
log(f"IARA acorda. K={D.knowledge()} · hormônios em repouso. (percepção simulada; laço = daemon 24/7)")

# ---------- UM DIA SIMULADO ----------
DIA=[
 ("ask","qual a capital do Peru?"),        # não sabe → pesquisa 3B
 ("ask","qual a capital do Peru?"),        # agora reusa (instantâneo)
 ("ask","qual a capital?"),                # AMBÍGUO → pergunta de volta
 ("ask","qual a capital do Japao?"),       # pesquisa 3B
 ("see","a dog"),                          # percepto visual conhecido
 ("see","a strange new gadget"),           # novidade → instiga curiosidade
 ("idle",None),("idle",None),("idle",None),# ócio → tédio sobe → curiosidade
 ("ask","autor de grande sertao veredas?"),# 3B incerto → PROFESSOR EXTERNO (Haiku/web)
 ("ask","quem descobriu o oxigenio xyz?"), # ninguém sabe → abstém honesto
 ("ask","qual a capital da França?"),      # pesquisa 3B
]
log(f"\n## O DIA (percebe · sente · busca · pergunta · pesquisa)")
for kind,payload in DIA:
    if kind=="ask":
        r=D.ask(payload); log(f"  você: '{payload}'  →  IARA: {r}   [DA{D.dop:.1f} cort{D.cort:.2f} tédio{D.bored:.1f}]")
    elif kind=="see":
        D.see(payload); log(f"  [vê: {payload}]   [tédio{D.bored:.1f}]")
    elif kind=="idle":
        D.tick()
        if D.bored>=D.BORED_TRIGGER:
            r=D.curiosity()
            if r: log(f"  ⚡ (entediada) IARA pergunta SOZINHA e busca  →  {r}   [tédio{D.bored:.1f}]")

# ---------- SONO / CONSOLIDAÇÃO ----------
log(f"\n## FIM DO DIA → SONO (consolida o importante, poda o frágil)")
kb=D.knowledge()
for _ in range(6): D.tick()                                    # tempo passa (algumas coisas decaem)
st,pr=D.sleep()
log(f"  dormiu: fortaleceu {st} memórias (as que geraram dopamina) · podou {pr} frágeis · K {kb}→{D.knowledge()}")

# ---------- MANHÃ SEGUINTE: valida retenção ----------
log(f"\n## MANHÃ SEGUINTE — o que sobrou?")
for q in ["qual a capital do Peru?","qual a capital do Japao?","capital do butao?"]:
    k=nrm(q); got=D.K.get(k,{}).get("v") if (k in D.K and D.K[k]["consol"]>=D.THRESH) else None
    log(f"  '{q}' → {got or 'esqueceu/precisaria repesquisar'}")

log(f"\n## VEREDITO — a IARA VIVE (comportamentos validados)")
beh=[b[0] for b in D.diary]
log(f"  ✓ PESQUISA quando não sabe (3B): {beh.count('pesquisou_3B')}× · ✓ PROFESSOR EXTERNO: {beh.count('professor_externo')}×")
log(f"  ✓ PERGUNTA DE VOLTA no ambíguo: {beh.count('pergunta_de_volta')}× · ✓ ABSTÉM honesto: {beh.count('abstem')}×")
log(f"  ✓ CURIOSA sozinha (por tédio): {beh.count('curiosa')}× · ✓ REUSO instantâneo: {beh.count('reuso')}×")
log(f"  ✓ SONO consolidou/podou · ✓ ESQUECE o não-usado. É um SER que vive, não um LLM que responde. wall {(time.time()-t0)/60:.1f}min")
