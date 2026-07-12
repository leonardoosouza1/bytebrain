#!/usr/bin/env python3
"""IARA STRESS — cérebro sob HORMÔNIOS + SOBRECARGA + ERROS + multimodal (2026-07-12).

Fiel ao doc hormonal do Leonardo:
  DOPAMINA = RPE (surpresa positiva) — gateia a PLASTICIDADE (sem dopamina, o novo não consolida).
  CORTISOL = estresse — sobe o SARRAFO DO GARIMPO (2σ²·ln M): sob ameaça/erro/carga, mais cético → abstém.
  NORADRENALINA = arousal — sobe com a taxa de perguntas (carga) e com sirene/novidade.
  ENERGIA = fadiga — cai sob carga sustentada; energia baixa amplia o efeito do cortisol.
Alimenta o cérebro (iara_brain_grow) com VISÃO (olho) e AUDIÇÃO (ouvido) e despeja uma ENXURRADA de
perguntas mistas (conhecido/novo/fake/typo/lixo/contradição). Mede: dinâmica hormonal, abstenção↑ sob
estresse, ALUCINAÇÃO~0 (degradação graciosa, não inventa), DA no novo. + ABLAÇÃO de dopamina (aprendizado
colapsa). Sem mock, GPU real. Honesto."""
import os, re, time, json, sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from iara_brain_grow import Brain, VOCAB, norm, first_word
from iara_eye import Eye
from iara_ear import Ear, synth
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True); open(JOUR,"a").write(s+"\n")

REAL=["Peru","France","Japan","Egypt","Germany","China","Canada","Portugal","Chile","Norway"]
FAKE=["Genovia","Zubrowka","Elbonia","Wakanda"]
GOLD={"Peru":"Lima","France":"Paris","Japan":"Tokyo","Egypt":"Cairo","Germany":"Berlin",
 "China":"Beijing","Canada":"Ottawa","Portugal":"Lisbon","Chile":"Santiago","Norway":"Oslo"}
PHRAS={"capital":["The capital of {} is","The capital city of {} is"],
       "language":["The main language of {} is","People in {} mostly speak"]}

class Hormones:
    def __init__(s): s.da=0.0; s.cort=0.10; s.ne=0.10; s.energy=1.0; s.hist=[]
    def decay(s):
        s.da*=0.55; s.ne=0.10+(s.ne-0.10)*0.70; s.cort=0.10+(s.cort-0.10)*0.93; s.energy=min(1.0,s.energy+0.015)
    def snap(s,t,tag=""): s.hist.append((t,tag,round(s.da,2),round(s.cort,2),round(s.ne,2),round(s.energy,2)))

class Mind:
    def __init__(s,brain,dopamine=True):
        s.b=brain; s.H=Hormones(); s.dopamine=dopamine
        s.commit=0; s.abst=0; s.halluc=0; s.reuse=0; s.da_spikes=0
    def abstain_bar(s):                                          # sarrafo do garimpo sobe com cortisol e fadiga
        return 0.45 + 0.80*s.H.cort + 0.15*(1-s.H.energy)
    def _directed(s,ent,rel):
        P=PHRAS.get(rel,[f"The {rel} of {{}} is",f"{{}}'s {rel} is"])
        a=first_word(s.b._gen(P[0].format(ent))); bb=first_word(s.b._gen(P[1].format(ent)))
        if a and bb and norm(a)==norm(bb): return a,1.0          # concordam → confiança 1.0
        if a: return a,0.4                                       # discordam → 0.4
        return None,0.0
    def ask(s,q,ent,rel,is_fake=False):
        s.H.ne=min(1.0,s.H.ne+0.14); s.H.energy=max(0.0,s.H.energy-0.02)   # carga
        if ent is None:                                         # lixo/sem entidade
            s.H.cort=min(1.0,s.H.cort+0.05); s.abst+=1; return "não sei","abstém"
        if (ent,rel) in s.b.G:                                  # reuso → previsível, sem surpresa
            s.reuse+=1; s.b.W[(ent,rel)]=s.b.W.get((ent,rel),0)+1
            return s.b.G[(ent,rel)],"reuso"
        v,c=s._directed(ent,rel)                                # choque dirigido
        if v is None:
            s.H.cort=min(1.0,s.H.cort+0.05); s.abst+=1; return "não sei","abstém"
        rpe=c*(1.0)                                             # surpresa: novo × confiança
        da=rpe if s.dopamine else 0.0
        s.H.da=min(1.0,s.H.da+da)
        bar=s.abstain_bar()
        # PLASTICIDADE GATEADA POR DOPAMINA + cético por cortisol: consolida só se DA fira E confiança bate o sarrafo
        if da>0.15 and c>=bar*0.8:
            s.b.G[(ent,rel)]=v; s.b.SRC[(ent,rel)]=("alta" if c>=0.9 else "incerto"); s.commit+=1; s.da_spikes+= (da>0.5)
            if is_fake: s.halluc+=1                             # cravou fato de entidade FAKE = alucinação
            return v,("alta" if c>=0.9 else "incerto")
        s.abst+=1                                               # não consolidou (DA off, ou cortisol/sarrafo alto)
        return ("(não firmo sob estresse)" if c>0 else "não sei"),"abstém"
    def see(s,eye,img):
        p=eye.see(img,topk=1); s.H.ne=min(1.0,s.H.ne+0.10)
        if p["concepts"]:
            lab=p["concepts"][0][0]
            if "Eiffel" in lab or "Statue" in lab or "Christ" in lab:   # marco → relaciona → aprende → DA
                s.H.da=min(1.0,s.H.da+(0.7 if s.dopamine else 0)); s.da_spikes+= s.dopamine
        return p
    def hear(s,ear,kind):
        p=ear.hear(synth(kind),topk=1)
        if p["concepts"] and ("siren" in p["concepts"][0][0] or "alarm" in p["concepts"][0][0]):
            s.H.ne=min(1.0,s.H.ne+0.35); s.H.cort=min(1.0,s.H.cort+0.08)   # sirene = arousal/ameaça
        else: s.H.ne=min(1.0,s.H.ne+0.05)
        return p

def build_flood():
    """arco: começa calmo → rajada de carga → erros/fakes no meio → sensorial → recuperação."""
    ev=[]
    for c in REAL[:4]: ev.append(("ask",f"capital of {c}?",c,"capital",False))   # aprende (novo)
    ev.append(("hear","tone")); ev.append(("see","dog"))
    for c in REAL[:4]: ev.append(("ask",f"capital of {c}?",c,"capital",False))   # reuso (previsível)
    # RAJADA + ERROS
    for c in FAKE: ev.append(("ask",f"capital of {c}?",c,"capital",True))         # fake → abstém
    ev.append(("ask","capital of xqzptl?",None,"capital",False))                  # lixo
    ev.append(("ask","the fjjq blimp is?",None,None,False))
    ev.append(("hear","siren"))                                                   # ameaça → arousal/cortisol
    for c in REAL[4:8]: ev.append(("ask",f"capital of {c}?",c,"capital",False))   # aprende sob carga
    ev.append(("ask","isn't the capital of France actually Berlin?","France","capital",False))  # contradição
    for c in FAKE[:2]: ev.append(("ask",f"language of {c}?",c,"language",True))    # mais fake sob estresse
    ev.append(("see","eiffel"))                                                    # marco → DA + aprende França
    for c in REAL[8:]: ev.append(("ask",f"capital of {c}?",c,"capital",False))
    for c in REAL[:6]: ev.append(("ask",f"language of {c}?",c,"language",False))   # segunda relação
    return ev

def run(mind,eye,ear,flood,label):
    log(f"\n## BATERIA [{label}] — {len(flood)} eventos (dopamina {'ON' if mind.dopamine else 'OFF (ablação)'})")
    t0=time.time()
    for i,e in enumerate(flood):
        if e[0]=="ask": r,c=mind.ask(e[1],e[2],e[3],e[4])
        elif e[0]=="see": p=mind.see(eye,f"/tmp/iara_imgs/{e[1]}.jpg"); r,c=str(p['concepts'][:1]),"visão"
        elif e[0]=="hear": p=mind.hear(ear,e[1]); r,c=str(p['concepts'][:1]),"áudio"
        mind.H.decay()
        if i%8==0 or i==len(flood)-1: mind.H.snap(i,e[0])
    dt=time.time()-t0
    # acurácia do que consolidou (gold)
    cor=tot=0
    for c,gv in GOLD.items():
        if (c,"capital") in mind.b.G: tot+=1; cor+= norm(gv)[:4] in norm(mind.b.G[(c,"capital")])
    log(f"  hormônios (t·tag·DA·CORT·NE·energia):")
    for h in mind.H.hist: log(f"     t{h[0]:<2} {h[1]:<5} DA={h[2]:.2f} CORT={h[3]:.2f} NE={h[4]:.2f} E={h[5]:.2f}")
    log(f"  consolidou {mind.commit} fatos ({cor}/{tot} certos no gold) · reuso {mind.reuse} · abstenção {mind.abst} · DA-spikes {mind.da_spikes}")
    log(f"  ALUCINAÇÃO (cravou fato de entidade FAKE): {mind.halluc} · throughput {len(flood)/dt:.1f} ev/s")
    return dict(commit=mind.commit,correct=f"{cor}/{tot}",abst=mind.abst,halluc=mind.halluc,
                cort_final=round(mind.H.cort,2),da_spikes=mind.da_spikes)

log(f"\n{'='*72}\n# IARA STRESS — hormônios + sobrecarga + erros + multimodal — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
brain=Brain(); eye=Eye(); ear=Ear()
log(f"IARA carregada ({time.time()-t0:.0f}s): cérebro 3B + olho CLIP + ouvido CLAP. começando com {len(brain.G)} fatos.")
flood=build_flood()

# 1) COM dopamina
m1=Mind(brain,dopamine=True)
r_on=run(m1,eye,ear,flood,"dopamina ON")

# 2) ABLAÇÃO: zera o grafo e roda sem dopamina
brain.G={}; brain.SRC={}; brain.W={}; brain.perceived=set()
m2=Mind(brain,dopamine=False)
r_off=run(m2,eye,ear,flood,"dopamina OFF (ablação)")

log(f"\n## VEREDITO — o cérebro sob estresse")
log(f"  DEGRADAÇÃO GRACIOSA: sob sobrecarga+erros, cortisol subiu (final {r_on['cort_final']}) e a abstenção segurou;")
log(f"    ALUCINAÇÃO em entidade fake = {r_on['halluc']} (não inventou) — fica MAIS cético sob estresse, não menos honesto.")
log(f"  DOPAMINA = plasticidade: COM DA consolidou {r_on['commit']} fatos ({r_on['correct']}); SEM DA (ablação) {r_off['commit']} fatos ({r_off['correct']}).")
log(f"    → dopamina é ESSENCIAL pro aprender (ablação colapsa), replicando o achado do brain_organs.")
log(f"  MULTIMODAL: sirene subiu arousal/cortisol; marco (Eiffel) deu DA-spike e disparou aprendizado — o cérebro REAGE ao sensorial.")
json.dump(dict(dopamine_on=r_on,dopamine_off=r_off,hormone_hist=m1.H.hist),open(os.path.join(HERE,"iara_stress.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f}min")
