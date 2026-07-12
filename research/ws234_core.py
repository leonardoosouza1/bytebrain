#!/usr/bin/env python3
"""WS2+WS3+WS4 — o núcleo do programa (uma carga de modelo, três frentes).

WS2 RETRIEVAL v2 — HIPÓTESE: o grafo-PMI degradou (94%→54%) porque co-ocorrência embaça
  com fatos parecidos. Retrievers que usam o PRÓPRIO gerador devem vencer:
    (a) embedding denso (média do último hidden da frase-fato)
    (b) assinatura-de-neurônios (a ESCRITA profunda acumulada da frase — original nosso)
  vs (c) PMI-água (baseline de ontem). +stress com 60 fatos-distratores FALSOS no índice.

WS3 VERIFICADOR v2 — HIPÓTESE: a bandeira de ontem (precisão 41% @ recall 100%) melhora com
  sinais ricos: tipo-válido, acordo-território, MARGEM de candidato (1º vs 2º na escrita
  profunda), MARGEM de logit do gerador. Varre limiares → curva precisão/recall → ponto de
  operação (recall ≥95%, máxima precisão). Re-mede o pipeline-produto com a bandeira nova.

WS4 PARÁFRASE — HIPÓTESE: consistência 60% sobe com VOTO entre 3 fraseados; e a CONFIANÇA
  do verificador escolhe o fraseado certo (self-consistency barata, só 3 forwards).

205 fatos, geração aberta. Journal: PROGRAMA_JOURNAL.md."""
import torch, os, re, time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "../../llm-lab/models/Qwen2.5-1.5B-Instruct")
DEV = "cuda"; DEEP = list(range(16, 28))
JOUR = os.path.join(HERE, "PROGRAMA_JOURNAL.md")
def log(s):
    print(s, flush=True)
    with open(JOUR, "a") as f: f.write(s + "\n")

# ---------------- KB (205 fatos, mesma de ontem) ----------------
CAPITAL = {
 "Afghanistan":"Kabul","Albania":"Tirana","Algeria":"Algiers","Argentina":"Buenos Aires","Australia":"Canberra",
 "Austria":"Vienna","Bangladesh":"Dhaka","Belgium":"Brussels","Bolivia":"Sucre","Brazil":"Brasilia",
 "Bulgaria":"Sofia","Cambodia":"Phnom Penh","Cameroon":"Yaounde","Canada":"Ottawa","Chile":"Santiago",
 "China":"Beijing","Colombia":"Bogota","Croatia":"Zagreb","Cuba":"Havana","Czechia":"Prague",
 "Denmark":"Copenhagen","Ecuador":"Quito","Egypt":"Cairo","Ethiopia":"Addis Ababa","Finland":"Helsinki",
 "France":"Paris","Germany":"Berlin","Ghana":"Accra","Greece":"Athens","Hungary":"Budapest",
 "Iceland":"Reykjavik","India":"New Delhi","Indonesia":"Jakarta","Iran":"Tehran","Iraq":"Baghdad",
 "Ireland":"Dublin","Israel":"Jerusalem","Italy":"Rome","Japan":"Tokyo","Jordan":"Amman",
 "Kazakhstan":"Astana","Kenya":"Nairobi","Laos":"Vientiane","Lebanon":"Beirut","Libya":"Tripoli",
 "Malaysia":"Kuala Lumpur","Mexico":"Mexico City","Mongolia":"Ulaanbaatar","Morocco":"Rabat","Nepal":"Kathmandu",
 "Netherlands":"Amsterdam","Nigeria":"Abuja","Norway":"Oslo","Pakistan":"Islamabad","Paraguay":"Asuncion",
 "Peru":"Lima","Philippines":"Manila","Poland":"Warsaw","Portugal":"Lisbon","Qatar":"Doha",
 "Romania":"Bucharest","Senegal":"Dakar","Serbia":"Belgrade","Slovakia":"Bratislava","Spain":"Madrid",
 "Sweden":"Stockholm","Switzerland":"Bern","Syria":"Damascus","Thailand":"Bangkok","Tunisia":"Tunis",
 "Uganda":"Kampala","Ukraine":"Kyiv","Uruguay":"Montevideo","Uzbekistan":"Tashkent","Venezuela":"Caracas",
 "Vietnam":"Hanoi","Zimbabwe":"Harare","Angola":"Luanda","Armenia":"Yerevan","Belarus":"Minsk",
 "Estonia":"Tallinn","Latvia":"Riga","Lithuania":"Vilnius","Slovenia":"Ljubljana","Yemen":"Sanaa",
}
CONTINENT = {
 "Afghanistan":"Asia","Albania":"Europe","Algeria":"Africa","Argentina":"South America","Australia":"Oceania",
 "Austria":"Europe","Bangladesh":"Asia","Belgium":"Europe","Bolivia":"South America","Brazil":"South America",
 "Bulgaria":"Europe","Cambodia":"Asia","Cameroon":"Africa","Canada":"North America","Chile":"South America",
 "China":"Asia","Colombia":"South America","Croatia":"Europe","Cuba":"North America","Czechia":"Europe",
 "Denmark":"Europe","Ecuador":"South America","Egypt":"Africa","Ethiopia":"Africa","Finland":"Europe",
 "France":"Europe","Germany":"Europe","Ghana":"Africa","Greece":"Europe","Hungary":"Europe",
 "Iceland":"Europe","India":"Asia","Indonesia":"Asia","Iran":"Asia","Iraq":"Asia","Ireland":"Europe",
 "Italy":"Europe","Japan":"Asia","Jordan":"Asia","Kenya":"Africa","Laos":"Asia","Lebanon":"Asia",
 "Libya":"Africa","Malaysia":"Asia","Mexico":"North America","Mongolia":"Asia","Morocco":"Africa",
 "Nepal":"Asia","Netherlands":"Europe","Nigeria":"Africa","Norway":"Europe","Pakistan":"Asia",
 "Paraguay":"South America","Peru":"South America","Philippines":"Asia","Poland":"Europe","Portugal":"Europe",
 "Romania":"Europe","Senegal":"Africa","Serbia":"Europe","Slovakia":"Europe","Spain":"Europe",
 "Sweden":"Europe","Syria":"Asia","Thailand":"Asia","Tunisia":"Africa","Uganda":"Africa","Ukraine":"Europe",
 "Uruguay":"South America","Venezuela":"South America","Vietnam":"Asia","Zimbabwe":"Africa","Angola":"Africa",
}
CURRENCY = {
 "Japan":"Yen","China":"Yuan","India":"Rupee","Brazil":"Real","France":"Euro","Germany":"Euro","Italy":"Euro",
 "Spain":"Euro","Portugal":"Euro","Ireland":"Euro","Austria":"Euro","Belgium":"Euro","Netherlands":"Euro",
 "Finland":"Euro","Greece":"Euro","Slovakia":"Euro","Slovenia":"Euro","Latvia":"Euro","Lithuania":"Euro",
 "Estonia":"Euro","Switzerland":"Franc","Sweden":"Krona","Norway":"Krone","Denmark":"Krone","Poland":"Zloty",
 "Czechia":"Koruna","Hungary":"Forint","Egypt":"Pound","Nigeria":"Naira","Kenya":"Shilling","Thailand":"Baht",
 "Vietnam":"Dong","Indonesia":"Rupiah","Malaysia":"Ringgit","Philippines":"Peso","Mexico":"Peso",
 "Argentina":"Peso","Chile":"Peso","Colombia":"Peso","Peru":"Sol","Israel":"Shekel","Iran":"Rial",
 "Iraq":"Dinar","Pakistan":"Rupee","Bangladesh":"Taka","Ukraine":"Hryvnia","Russia":"Ruble",
}
REL = {"capital":(CAPITAL,"The capital of {} is"),
       "continent":(CONTINENT,"The country {} is located in the continent of"),
       "currency":(CURRENCY,"The currency of {} is the")}
PARA = {
 "capital":["The capital of {} is","What is the capital city of {}? It is","{}'s capital is"],
 "continent":["The country {} is located in the continent of","{} is a country located in","Which continent is {} in? It is in"],
 "currency":["The currency of {} is the","In {}, people pay with the","{}'s official currency is the"],
}
FACTS=[dict(rel=rk,country=c,ans=a) for rk,(d,_) in REL.items() for c,a in d.items()]
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(gold,gen):
    g=norm(gold); return g in norm(gen) or norm(gen).startswith(g[:max(3,len(g)//2)])

log(f"\n{'='*72}\n# WS2+3+4 — CORE (retrieval v2 · verificador v2 · paráfrase) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
E = model.get_output_embeddings().weight.detach(); norm_w = model.model.norm.weight.detach()
writes={}
for L in DEEP:
    model.model.layers[L].mlp.down_proj.register_forward_hook(
        (lambda L:(lambda m,i,o: writes.__setitem__(L,o.detach()[0,-1])))(L))
def fid(w): return tok.encode(" "+w, add_special_tokens=False)[0]
CANDS={rk:list({fid(a) for a in d.values()}) for rk,(d,_) in REL.items()}
CANDSET={rk:set(v) for rk,v in CANDS.items()}

@torch.no_grad()
def gen_open(prompt, n=8, capture=False):
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    logits=model(ids).logits[0,-1]
    snap={L:writes[L].clone() for L in DEEP} if capture else None
    lm=torch.topk(logits.float(),2).values; lmargin=float(lm[0]-lm[1])
    first=int(logits.argmax()); out=[first]; cur=torch.cat([ids,torch.tensor([[first]],device=DEV)],1)
    for _ in range(n-1):
        nt=int(model(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out).strip(), first, snap, lmargin

@torch.no_grad()
def embed_sentence(s):
    ids=tok(s,return_tensors="pt").input_ids.to(DEV)
    hs=model(ids,output_hidden_states=True).hidden_states[-1][0]   # [T,H]
    dense=hs.mean(0).float().cpu().numpy()
    sig=(sum(writes[L] for L in DEEP)).float().cpu().numpy()       # assinatura da escrita profunda
    return dense, sig

# ---------------- passo 1: roda tudo (3 fraseados, captura sinais no fraseado 1) ----------------
for r in FACTS:
    tpls=PARA[r["rel"]]
    g0,first,snap,lmargin=gen_open(tpls[0].format(r["country"]),capture=True)
    g1,_,_,_=gen_open(tpls[1].format(r["country"]))
    g2,_,_,_=gen_open(tpls[2].format(r["country"]))
    r["gens"]=[g0,g1,g2]; r["ok"]=[match(r["ans"],g) for g in (g0,g1,g2)]
    r["first"]=first; r["lmargin"]=lmargin
    w=(sum(snap[L] for L in DEEP)*norm_w)
    ll=(E[CANDS[r["rel"]]]@w).float()
    top2=torch.topk(ll,2)
    r["terr"]=CANDS[r["rel"]][int(top2.indices[0])]
    r["cmargin"]=float(top2.values[0]-top2.values[1])
    r["valid"]=r["first"] in CANDSET[r["rel"]]
    r["agree"]=(r["terr"]==r["first"])
log(f"forward 3×{len(FACTS)} fraseados + sinais: {time.time()-t0:.0f}s")
base_acc=sum(r["ok"][0] for r in FACTS)/len(FACTS)
log(f"baseline (fraseado 1): {base_acc:.0%}")

# ---------------- WS2: retrieval v2 ----------------
log(f"\n## WS2 — RETRIEVAL v2 (205 fatos + stress com 60 distratores falsos)")
LINES=[f"the {r['rel']} of {r['country'].lower()} is {r['ans'].lower()}" for r in FACTS]
FAKEC=["Genovia","Wakanda","Zubrowka","Freedonia","Latveria","Elbonia","Sokovia","Kamistan","Borduria","Syldavia"]
FAKES=[f"the capital of {f.lower()} is {x.lower()}" for f in FAKEC for x in ["Vasco","Porto Rey","Zenda"]]+\
      [f"the currency of {f.lower()} is the {x.lower()}" for f in FAKEC for x in ["Dollarine","Pesoto","Krunet"]]
t1=time.time()
FD=[]; FS=[]
for ln in LINES+FAKES:
    d,s=embed_sentence(ln); FD.append(d); FS.append(s)
FD=np.stack(FD); FS=np.stack(FS)
FD/= np.linalg.norm(FD,axis=1,keepdims=True)+1e-9; FS/=np.linalg.norm(FS,axis=1,keepdims=True)+1e-9
def q_of(r): return REL[r["rel"]][1].format(r["country"])
QD=[]; QS=[]
for r in FACTS:
    d,s=embed_sentence(q_of(r)); QD.append(d); QS.append(s)
QD=np.stack(QD); QS=np.stack(QS)
QD/=np.linalg.norm(QD,axis=1,keepdims=True)+1e-9; QS/=np.linalg.norm(QS,axis=1,keepdims=True)+1e-9
ALL_LINES=LINES+FAKES
def ret_acc(Q,F,n_index):
    sim=Q@F[:n_index].T; top=sim.argmax(1)
    return sum(1 for i,r in enumerate(FACTS) if norm(r["ans"]) in norm(ALL_LINES[top[i]]))/len(FACTS)
acc_d=ret_acc(QD,FD,len(LINES)); acc_s=ret_acc(QS,FS,len(LINES))
acc_d_stress=ret_acc(QD,FD,len(ALL_LINES)); acc_s_stress=ret_acc(QS,FS,len(ALL_LINES))
log(f"  denso (hidden do gerador):        {acc_d:.0%}   com distratores: {acc_d_stress:.0%}")
log(f"  assinatura-de-neurônios (escrita): {acc_s:.0%}   com distratores: {acc_s_stress:.0%}")
log(f"  PMI-água (ontem, baseline):        54%")
best_name,bestQ,bestF = ("denso",QD,FD) if acc_d>=acc_s else ("assinatura",QS,FS)
log(f"  → retriever do produto: {best_name} · embeddings em {time.time()-t1:.0f}s")

# ---------------- WS3: verificador v2 ----------------
log(f"\n## WS3 — VERIFICADOR v2 (varredura de sinais → ponto de operação)")
errs=[r for r in FACTS if not r["ok"][0]]
log(f"  erros do gerador (fraseado 1): {len(errs)}/{len(FACTS)}")
def pr(flags):   # precisão/recall da bandeira sobre os erros
    fl=[r for r,f in zip(FACTS,flags) if f]
    tp=sum(1 for r in fl if not r["ok"][0])
    return (tp/len(fl) if fl else 1.0, tp/len(errs) if errs else 1.0, len(fl))
fA=[not(r["valid"] and r["agree"]) for r in FACTS]
pA,rA,nA=pr(fA); log(f"  regra ONTEM (¬válido ∨ ¬acordo):        precisão {pA:.0%} recall {rA:.0%} ({nA} flags)")
cms=sorted(set(round(r["cmargin"],1) for r in FACTS))
best=None
for tau in cms:
    fB=[(not r["valid"]) or ((not r["agree"]) and r["cmargin"]< tau) for r in FACTS]
    p,rc,nB=pr(fB)
    if rc>=0.95 and (best is None or p>best[0]): best=(p,rc,tau,nB)
if best:
    log(f"  regra NOVA (¬válido ∨ (¬acordo ∧ margem<τ)): precisão {best[0]:.0%} recall {best[1]:.0%} @ τ={best[2]} ({best[3]} flags)")
for tau_l in [2.0,4.0,6.0]:
    fC=[(not r["valid"]) or ((not r["agree"]) and (r["cmargin"]<(best[2] if best else 1e9) or r["lmargin"]<tau_l)) for r in FACTS]
    p,rc,nC=pr(fC); log(f"    +margem-logit<{tau_l}: precisão {p:.0%} recall {rc:.0%} ({nC} flags)")
TAU=best[2] if best else None
def flag_v2(r): return (not r["valid"]) or ((not r["agree"]) and r["cmargin"]<TAU)
# pipeline-produto v2: confia ∨ (RAG com retriever v2 e re-gera)
t2=time.time(); fixed=0; calls=0; final=0
for i,r in enumerate(FACTS):
    if not flag_v2(r): final+=r["ok"][0]; continue
    calls+=1
    sim=bestQ[i]@bestF[:len(LINES)].T; doc=LINES[int(sim.argmax())]
    g,_,_,_=gen_open(f"{doc.capitalize()}. {q_of(r)}")
    hit=match(r["ans"],g); final+=hit
    if hit and not r["ok"][0]: fixed+=1
log(f"  PIPELINE v2: {base_acc:.0%} → {final/len(FACTS):.0%}  (RAG {calls}/{len(FACTS)}={calls/len(FACTS):.0%}, consertou {fixed}/{len(errs)}) · {time.time()-t2:.0f}s")
log(f"  ontem: 98% @ 28% de chamadas — comparar custo×acerto")

# ---------------- WS4: paráfrase ----------------
log(f"\n## WS4 — PARÁFRASE (voto entre 3 fraseados)")
per=[sum(r["ok"][k] for r in FACTS)/len(FACTS) for k in range(3)]
log(f"  acurácia por fraseado: {per[0]:.0%} / {per[1]:.0%} / {per[2]:.0%}")
cons=sum(1 for r in FACTS if len(set(r["ok"]))==1)/len(FACTS)
vote=0
for r in FACTS:
    ns=[norm(g) for g in r["gens"]]
    agree01 = match(r["gens"][0], r["gens"][1]) or match(r["gens"][1], r["gens"][0]) or ns[0]==ns[1]
    agree02 = match(r["gens"][0], r["gens"][2]) or match(r["gens"][2], r["gens"][0]) or ns[0]==ns[2]
    agree12 = match(r["gens"][1], r["gens"][2]) or match(r["gens"][2], r["gens"][1]) or ns[1]==ns[2]
    if agree01 or agree02: pick=0
    elif agree12: pick=1
    else: pick=0
    vote+=r["ok"][pick]
log(f"  consistência (mesmo veredito nos 3): {cons:.0%}  (ontem 60%)")
log(f"  VOTO (par que concorda vence):       {vote/len(FACTS):.0%}  vs melhor fraseado {max(per):.0%}")
log(f"\nWS2-4 wall total {(time.time()-t0)/60:.1f} min")
