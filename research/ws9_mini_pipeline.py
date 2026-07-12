#!/usr/bin/env python3
"""WS9 — O TESTE FINAL DA LINHAGEM: IARA-mini (1.26B) + órgãos ≥ base (1.54B) sozinha?

Se sim, a tese fecha o círculo: um modelo 18% MENOR, com os órgãos baratos em volta,
responde MELHOR que o modelo cheio pelado. "Menor e mais inteligente" com número.

E resolve a pendência da bandeira por outro ângulo: em vez de caçar precisão 70%, mede a
CURVA ACURÁCIA × ORÇAMENTO — ordena as bandeiras por margem de confiança e gasta RAG só
nas K piores. O produto escolhe o ponto da curva (o botão custo×qualidade)."""
import torch, os, re, time, sys, json
from transformers import AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from iara_mini_loader import load_iara_mini
MINI=os.path.join(HERE,"../../llm-lab/models/iara-mini-v01")
DEV="cuda"
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
VALSET={rk:sorted({a for a in d.values()}) for rk,(d,_) in REL.items()}
KB={(r["country"],r["rel"]):r["ans"] for r in FACTS}
ENT=sorted({r["country"] for r in FACTS})
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(g,s): return norm(g) in norm(s) or norm(s).startswith(norm(g)[:max(3,len(norm(g))//2)])
def lev(a,b,cap=3):
    if abs(len(a)-len(b))>cap: return cap+1
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(ca!=cb)))
        if min(cur)>cap: return cap+1
        prev=cur
    return prev[-1]
def hybrid(country,rel):
    best,bd=None,99
    for e in ENT:
        d=lev(country.lower(),e.lower())
        if d<bd: bd,best=d,e
    return KB.get((best,rel)) if bd<=2 else None

log(f"\n{'='*72}\n# WS9 — IARA-mini + ÓRGÃOS vs base pelada — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(MINI)
model=load_iara_mini(MINI,DEV)
NL=model.config.num_hidden_layers; DEEP=list(range(NL-12,NL))
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
def str_valid(g,rel):
    gn=norm(g)
    for v in VALSET[rel]:
        if norm(v) in gn: return v
    return None

# 3 fraseados com sinais + margem (pra curva de orçamento)
for r in FACTS:
    per=[]
    for tpl in PARA[r["rel"]]:
        g,first,snap=gen_open(tpl.format(r["country"]),capture=True)
        v=str_valid(g,r["rel"])
        w=(sum(snap[L] for L in DEEP)*norm_w)
        ll=(E[CANDS[r["rel"]]]@w).float()
        top2=torch.topk(ll,2)
        terr=CANDS[r["rel"]][int(top2.indices[0])]
        margin=float(top2.values[0]-top2.values[1])
        trust=(v is not None) and (terr==fid(v))
        per.append(dict(g=g,ok=match(r["ans"],g),valid=v,trust=trust,margin=margin))
    r["per"]=per
mini_alone=sum(r["per"][0]["ok"] for r in FACTS)/len(FACTS)
log(f"IARA-mini pelada (1.26B): {mini_alone:.0%}   [base 1.54B pelada: 88%]")

# pipeline v2.1 completo no mini
final=0; calls=0; via=dict(t0=0,alt=0,rag=0)
order=[]
for r in FACTS:
    p0=r["per"][0]
    if p0["trust"]: final+=p0["ok"]; via["t0"]+=1; continue
    k=next((k for k in range(1,3) if r["per"][k]["trust"]),None)
    if k is not None: final+=r["per"][k]["ok"]; via["alt"]+=1; continue
    calls+=1; via["rag"]+=1
    fact=hybrid(r["country"],r["rel"])
    if fact:
        g,_,_=gen_open(f"The {r['rel']} of {r['country']} is {fact}. {REL[r['rel']][1].format(r['country'])}")
        final+=match(r["ans"],g)
    order.append((p0["margin"],r))
pipe=final/len(FACTS)
log(f"IARA-mini + ÓRGÃOS (verificador+paráfrase+híbrido-byte): {pipe:.0%}  (RAG {calls}/{len(FACTS)}={calls/len(FACTS):.0%})")
log(f"  fluxo: confiou direto {via['t0']} · fraseado alternativo {via['alt']} · RAG {via['rag']}")
verd = pipe>=0.88
log(f"VEREDITO WS9: mini(1.26B)+órgãos = {pipe:.0%} {'≥' if verd else '<'} base(1.54B) pelada 88% → "
    f"{'TESE FECHADA: menor E mais inteligente COM órgãos' if verd else 'não fechou — registrar'}")

# CURVA ACURÁCIA × ORÇAMENTO (o botão do produto): flags ordenadas por margem (pior primeiro)
log(f"\nCURVA acurácia × orçamento de RAG (bandeiras ordenadas pela MARGEM, pior primeiro):")
flagged=[r for r in FACTS if not any(p["trust"] for p in r["per"])]
flagged.sort(key=lambda r: r["per"][0]["margin"])
trusted_ok=sum((r["per"][0]["ok"] if r["per"][0]["trust"] else r["per"][next(k for k in range(3) if r["per"][k]["trust"])]["ok"]) for r in FACTS if any(p["trust"] for p in r["per"]))
for frac in [0.0,0.25,0.5,0.75,1.0]:
    k=int(len(flagged)*frac); ok=trusted_ok
    for i,r in enumerate(flagged):
        if i<k:
            fact=hybrid(r["country"],r["rel"]); ok+= 1 if fact==r["ans"] else 0
        else: ok+=r["per"][0]["ok"]
    log(f"  orçamento {frac:>4.0%} das bandeiras ({k:>2} chamadas = {k/len(FACTS):.0%} do total) → {ok/len(FACTS):.0%}")
log(f"wall {(time.time()-t0)/60:.1f} min")
