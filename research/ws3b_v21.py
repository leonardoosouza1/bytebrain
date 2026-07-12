#!/usr/bin/env python3
"""WS2.1+3.1+4.1 — os consertos da iteração 1 (entender→resolver→prosseguir).

FALHAS DIAGNOSTICADAS na v2:
  WS2: embeddings de frase-template diluem a entidade (37% < água 54%). CONSERTOS:
       (a) embedding do ÚLTIMO token (estado "prestes a responder"), (b) subtração do
       template médio, (c) híbrido ESTRUTURADO: entidade casada por léxico byte (o órgão
       do WS1) + relação por palavra-chave → lookup exato (o que um produto faria).
  WS3: "válido" olhava só o 1º token → "the city of Paris" virava falso-flag. CONSERTO:
       validade a nível de STRING (a geração contém algum candidato do tipo?).
  WS4: voto simples perde pro melhor fraseado. CONSERTO: escolher por CONFIANÇA do
       verificador v2.1 medida EM CADA fraseado (não só no primeiro)."""
import torch, os, re, time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
MODEL=os.path.join(HERE,"../../llm-lab/models/Qwen2.5-1.5B-Instruct")
DEV="cuda"; DEEP=list(range(16,28))
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

CAPITAL={
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
CONTINENT={
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
CURRENCY={
 "Japan":"Yen","China":"Yuan","India":"Rupee","Brazil":"Real","France":"Euro","Germany":"Euro","Italy":"Euro",
 "Spain":"Euro","Portugal":"Euro","Ireland":"Euro","Austria":"Euro","Belgium":"Euro","Netherlands":"Euro",
 "Finland":"Euro","Greece":"Euro","Slovakia":"Euro","Slovenia":"Euro","Latvia":"Euro","Lithuania":"Euro",
 "Estonia":"Euro","Switzerland":"Franc","Sweden":"Krona","Norway":"Krone","Denmark":"Krone","Poland":"Zloty",
 "Czechia":"Koruna","Hungary":"Forint","Egypt":"Pound","Nigeria":"Naira","Kenya":"Shilling","Thailand":"Baht",
 "Vietnam":"Dong","Indonesia":"Rupiah","Malaysia":"Ringgit","Philippines":"Peso","Mexico":"Peso",
 "Argentina":"Peso","Chile":"Peso","Colombia":"Peso","Peru":"Sol","Israel":"Shekel","Iran":"Rial",
 "Iraq":"Dinar","Pakistan":"Rupee","Bangladesh":"Taka","Ukraine":"Hryvnia","Russia":"Ruble",
}
REL={"capital":(CAPITAL,"The capital of {} is"),
     "continent":(CONTINENT,"The country {} is located in the continent of"),
     "currency":(CURRENCY,"The currency of {} is the")}
PARA={
 "capital":["The capital of {} is","What is the capital city of {}? It is","{}'s capital is"],
 "continent":["The country {} is located in the continent of","{} is a country located in","Which continent is {} in? It is in"],
 "currency":["The currency of {} is the","In {}, people pay with the","{}'s official currency is the"],
}
FACTS=[dict(rel=rk,country=c,ans=a) for rk,(d,_) in REL.items() for c,a in d.items()]
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(g,s): return norm(g) in norm(s) or norm(s).startswith(norm(g)[:max(3,len(norm(g))//2)])
VALSET={rk:sorted({a for a in d.values()}) for rk,(d,_) in REL.items()}

log(f"\n{'='*72}\n# WS2.1+3.1+4.1 — CONSERTOS v2.1 — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(DEV).eval()
E=model.get_output_embeddings().weight.detach(); norm_w=model.model.norm.weight.detach()
writes={}
for L in DEEP:
    model.model.layers[L].mlp.down_proj.register_forward_hook(
        (lambda L:(lambda m,i,o: writes.__setitem__(L,o.detach()[0,-1])))(L))
def fid(w): return tok.encode(" "+w,add_special_tokens=False)[0]
CANDS={rk:list({fid(a) for a in d.values()}) for rk,(d,_) in REL.items()}

@torch.no_grad()
def gen_open(prompt,n=8,capture=False):
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    logits=model(ids).logits[0,-1]
    snap={L:writes[L].clone() for L in DEEP} if capture else None
    first=int(logits.argmax()); out=[first]; cur=torch.cat([ids,torch.tensor([[first]],device=DEV)],1)
    for _ in range(n-1):
        nt=int(model(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out).strip(), first, snap

def str_valid(gen_str, rel):        # CONSERTO WS3: validade a nível de string
    g=norm(gen_str)
    for v in VALSET[rel]:
        if norm(v) in g: return v
    return None
def terr_of(snap, rel):
    w=(sum(snap[L] for L in DEEP)*norm_w)
    return CANDS[rel][int((E[CANDS[rel]]@w).argmax())]

# ---- roda 3 fraseados COM sinais em todos (conserto WS4) ----
for r in FACTS:
    per=[]
    for tpl in PARA[r["rel"]]:
        g,first,snap=gen_open(tpl.format(r["country"]),capture=True)
        v=str_valid(g,r["rel"]); terr=terr_of(snap,r["rel"])
        trust = (v is not None) and (terr==fid(v))       # v2.1: string-válida E território concorda com ELA
        per.append(dict(g=g,ok=match(r["ans"],g),valid=v,trust=trust,terr=terr))
    r["per"]=per
log(f"3×{len(FACTS)} fraseados com sinais: {time.time()-t0:.0f}s")

# ---- WS3.1: bandeira v2.1 no fraseado 1 ----
errs=[r for r in FACTS if not r["per"][0]["ok"]]
flags=[r for r in FACTS if not r["per"][0]["trust"]]
tp=sum(1 for r in flags if not r["per"][0]["ok"])
prec=tp/len(flags) if flags else 1.0; rec=tp/len(errs) if errs else 1.0
acc_trust=sum(r["per"][0]["ok"] for r in FACTS if r["per"][0]["trust"])/max(1,sum(r["per"][0]["trust"] for r in FACTS))
log(f"\n## WS3.1 — verificador v2.1 (validade por STRING + acordo com a resposta)")
log(f"  confiados: {sum(r['per'][0]['trust'] for r in FACTS)}/{len(FACTS)} → acurácia {acc_trust:.0%}")
log(f"  bandeira: precisão {prec:.0%} (v2: 42%) · recall {rec:.0%} ({len(flags)} flags; erros {len(errs)})")

# ---- WS2.1: retrieval consertado ----
log(f"\n## WS2.1 — retrieval consertado")
LINES=[f"the {r['rel']} of {r['country'].lower()} is {r['ans'].lower()}" for r in FACTS]
@torch.no_grad()
def emb_last(s):
    ids=tok(s,return_tensors="pt").input_ids.to(DEV)
    hs=model(ids,output_hidden_states=True).hidden_states[-1][0]
    return hs[-1].float().cpu().numpy()                 # ÚLTIMO token (não média)
F=np.stack([emb_last(ln) for ln in LINES])
Q=np.stack([emb_last(REL[r["rel"]][1].format(r["country"])) for r in FACTS])
# subtração do template (conserto b): centra por relação
for rk in REL:
    idx=[i for i,r in enumerate(FACTS) if r["rel"]==rk]
    F[idx]-=F[idx].mean(0); Q[idx]-=Q[idx].mean(0)
Fn=F/ (np.linalg.norm(F,axis=1,keepdims=True)+1e-9); Qn=Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)
top=(Qn@Fn.T).argmax(1)
acc_last=sum(1 for i,r in enumerate(FACTS) if norm(r["ans"]) in norm(LINES[top[i]]))/len(FACTS)
log(f"  último-token + centrado por relação: {acc_last:.0%}  (v2 média: 37% · água: 54%)")
# híbrido estruturado (o que o produto faz): entidade por léxico byte + relação da pergunta
def lev(a,b,cap=3):
    if abs(len(a)-len(b))>cap: return cap+1
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(ca!=cb)))
        if min(cur)>cap: return cap+1
        prev=cur
    return prev[-1]
ENT=sorted({r["country"] for r in FACTS})
KB={(r["country"],r["rel"]):r["ans"] for r in FACTS}
def hybrid(query_country, rel):
    best,bd=None,99
    for e in ENT:
        d=lev(query_country.lower(),e.lower())
        if d<bd: bd,best=d,e
    return KB.get((best,rel)) if bd<=2 else None
acc_h=sum(1 for r in FACTS if hybrid(r["country"],r["rel"])==r["ans"])/len(FACTS)
log(f"  híbrido estruturado (léxico-byte + relação): {acc_h:.0%}  ← o retriever do PRODUTO")

# ---- WS4.1: escolha por confiança entre fraseados ----
log(f"\n## WS4.1 — paráfrase com escolha por CONFIANÇA (v2.1)")
per_acc=[sum(r["per"][k]["ok"] for r in FACTS)/len(FACTS) for k in range(3)]
pick=0; used_alt=0
for r in FACTS:
    k=next((k for k in range(3) if r["per"][k]["trust"]),0)
    if k>0: used_alt+=1
    pick+=r["per"][k]["ok"]
log(f"  por fraseado: {per_acc[0]:.0%} / {per_acc[1]:.0%} / {per_acc[2]:.0%}")
log(f"  ESCOLHA-POR-CONFIANÇA: {pick/len(FACTS):.0%}  (usou fraseado alternativo em {used_alt})")

# ---- PIPELINE v2.1 completo ----
log(f"\n## PIPELINE v2.1 (confiança v2.1 → senão híbrido-byte → re-gera)")
final=0; calls=0
for r in FACTS:
    p0=r["per"][0]
    if p0["trust"]: final+=p0["ok"]; continue
    k=next((k for k in range(1,3) if r["per"][k]["trust"]),None)
    if k is not None: final+=r["per"][k]["ok"]; continue
    calls+=1
    fact=hybrid(r["country"],r["rel"])
    if fact:
        doc=f"The {r['rel']} of {r['country']} is {fact}."
        g,_,_=gen_open(f"{doc} {REL[r['rel']][1].format(r['country'])}")
        final+=match(r["ans"],g)
log(f"  RESULTADO: 88% (base) → {final/len(FACTS):.0%}  (RAG {calls}/{len(FACTS)}={calls/len(FACTS):.0%})")
log(f"wall {(time.time()-t0)/60:.1f} min")
