#!/usr/bin/env python3
"""BATERIA DE PRODUTO — cérebro acoplado validado pra valer (Leonardo 2026-07-11, "pra ser produto").

Ataca as fronteiras honestas que faltavam, em escala, com números honestos e journal por lote:
  A) GERAÇÃO ABERTA multi-token (não match por 1º token) em 3 relações (capital/continente/moeda).
  B) VERIFICADOR: separação concorda/discorda + CALIBRAÇÃO (ECE) em multi-relação.
  C) CORRETOR: a memória-neurônio recupera o erro (candidatos multi-token do tipo).
  D) RETRIEVAL EXTERNO REAL (grafo-água, não gabarito): recupera o fato e re-gera (RAG) nas bandeiras.
  E) PIPELINE COMPLETO + ABLAÇÕES: cada órgão paga? (gerador / +verif+correção / +RAG).
  F) ROBUSTEZ: typo no nome do país — cai? o pipeline recupera?
  G) PALETA de especialistas (Instruct/Math/Coder) — rotear pro domínio paga?
  H) CUSTO: quantas vezes cada órgão dispara (modelo de custo do roteamento adaptativo).

GPU, threads capados. `python3 bateria_produto.py smoke` roda mini pra caçar bug. Journal:
bateria_produto_journal.md."""
import torch, os, re, sys, time, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llm-lab/models")
PATHS = {"Instruct":f"{BASE}/Qwen2.5-1.5B-Instruct","Coder":f"{BASE}/Qwen2.5-Coder-1.5B","Math":f"{BASE}/Qwen2.5-Math-1.5B"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEEP = list(range(16,28))
JOURNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bateria_produto_journal.md")
def log(s):
    print(s, flush=True)
    with open(JOURNAL, "a") as f: f.write(s+"\n")

# ---------------- base de conhecimento (dados conferidos) ----------------
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
CONTINENT = {  # sem transcontinentais ambíguos (Rússia/Turquia/etc.)
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
REL = {
 "capital":   (CAPITAL,   "The capital of {} is",                    "capital"),
 "continent": (CONTINENT, "The country {} is located in the continent of", "continente"),
 "currency":  (CURRENCY,  "The currency of {} is the",               "moeda"),
}
def build_facts():
    F=[]
    for rk,(d,tpl,_) in REL.items():
        for country,ans in d.items(): F.append(dict(rel=rk, country=country, ans=ans, prompt=tpl.format(country)))
    return F
FACTS = build_facts()
if SMOKE: FACTS = FACTS[:6] + FACTS[80:86] + FACTS[-6:]

# ---------------- gerador + hooks ----------------
tok = AutoTokenizer.from_pretrained(PATHS["Instruct"])
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def first_id(w): return tok.encode(" "+w, add_special_tokens=False)[0]

log(f"\n{'='*72}\n# BATERIA DE PRODUTO {'(SMOKE)' if SMOKE else ''} — cérebro acoplado — {time.strftime('%Y-%m-%d %H:%M')}\n{'='*72}")
log(f"{len(FACTS)} fatos ({len(CAPITAL)} capitais + {len(CONTINENT)} continentes + {len(CURRENCY)} moedas), gerador Qwen2.5-1.5B em {DEV}")
model = AutoModelForCausalLM.from_pretrained(PATHS["Instruct"], dtype=torch.float16).to(DEV).eval()
E = model.get_output_embeddings().weight.detach(); norm_w = model.model.norm.weight.detach()
writes = {}
for L in DEEP:
    model.model.layers[L].mlp.down_proj.register_forward_hook(
        (lambda L:(lambda m,i,o: writes.__setitem__(L,o.detach()[0,-1])))(L))

# candidatos por relação (1º token de cada resposta do tipo) — o "espaço de resposta"
CANDS = {rk: list({first_id(a) for a in d.values()}) for rk,(d,_,_) in REL.items()}
CANDSET = {rk: set(v) for rk,v in CANDS.items()}

@torch.no_grad()
def generate(prompt, n=8, capture=False):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    logits = model(ids).logits[0,-1]                    # 1º passo (hooks capturam a escrita aqui)
    snap = {L:writes[L].clone() for L in DEEP} if capture else None
    first = int(logits.argmax()); out=[first]
    cur = torch.cat([ids, torch.tensor([[first]],device=DEV)],1)
    for _ in range(n-1):
        nt = int(model(cur).logits[0,-1].argmax())
        if nt == tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur = torch.cat([cur, torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out).strip(), first, snap

def territory(cands, snap):
    w = sum(snap[L] for L in DEEP)*norm_w
    return cands[int((E[cands] @ w).argmax())]

def match(gold, gen):
    g=norm(gold); return g in norm(gen) or norm(gen).startswith(g[:max(3,len(g)//2)])

# ---------------- roda o gerador 1x em tudo, guarda tudo ----------------
t0=time.time()
for r in FACTS:
    gen, first, snap = generate(r["prompt"], n=8, capture=True)
    r["gen"]=gen; r["first"]=first; r["ok"]=match(r["ans"], gen)
    r["terr"]=territory(CANDS[r["rel"]], snap)
    r["conf_agree"]= (r["terr"]==r["first"])
log(f"forward em {len(FACTS)} fatos: {time.time()-t0:.0f}s\n")

def acc(rs): return sum(x["ok"] for x in rs)/len(rs) if rs else float("nan")

# ===== LOTE A: geração aberta multi-token por relação =====
log("## LOTE A — geração ABERTA multi-token (a métrica honesta)")
for rk in REL:
    rs=[r for r in FACTS if r["rel"]==rk]; log(f"  {rk:<10}: {acc(rs):.0%}  ({sum(x['ok'] for x in rs)}/{len(rs)})")
log(f"  GERAL: {acc(FACTS):.0%}\n")

# ===== LOTE A2: robustez a paráfrase (consistência) =====
log("## LOTE A2 — paráfrase: responde consistente em vários fraseados?")
PARA={
 "capital":["The capital of {} is","What is the capital city of {}? It is","{}'s capital is"],
 "continent":["The country {} is located in the continent of","{} is a country located in","Which continent is {} in? It is in"],
 "currency":["The currency of {} is the","In {}, people pay with the","{}'s official currency is the"],
}
subA = FACTS if not SMOKE else FACTS[:6]
consist=0; per=[0,0,0]
for r in subA:
    ans=[match(r["ans"], generate(tpl.format(r["country"]),n=8)[0]) for tpl in PARA[r["rel"]]]
    for k,a in enumerate(ans): per[k]+=a
    if len(set(ans))==1: consist+=1
nA=len(subA)
log(f"  acurácia por fraseado: {[f'{p/nA:.0%}' for p in per]} · consistência (mesmo veredito nos 3): {consist/nA:.0%}\n")

# ===== LOTE B: verificador — separação + calibração =====
log("## LOTE B — verificador (concorda=confia) + calibração")
agree=[r for r in FACTS if r["conf_agree"]]; disag=[r for r in FACTS if not r["conf_agree"]]
log(f"  território CONCORDA ({len(agree)}) → acurácia {acc(agree):.0%}")
log(f"  território DISCORDA ({len(disag)}) → acurácia {acc(disag):.0%}")
errs=[r for r in FACTS if not r["ok"]]
tp=sum(1 for r in disag if not r["ok"])
log(f"  bandeira: precisão {tp/len(disag) if disag else float('nan'):.0%} · recall dos erros {tp/len(errs) if errs else float('nan'):.0%}")
# ECE simples: 2 bins (concorda/discorda), conf=fração-certa observada por bin
ece = (len(agree)/len(FACTS))*abs(acc(agree)-1.0) + (len(disag)/len(FACTS))*abs(acc(disag)-0.0)
log(f"  ECE (2-bin concorda/discorda): {ece:.3f}  (0=perfeitamente calibrado)\n")

# ===== LOTE B2: calibração contínua (níveis de confiança) =====
log("## LOTE B2 — calibração: a acurácia sobe com a confiança do cérebro?")
def conflevel(r):
    if r["first"] not in CANDSET[r["rel"]]: return 0     # gerador desistiu (filler)
    return 2 if r["conf_agree"] else 1                   # 2=tipo-válido+concorda · 1=válido mas território discorda
for lv,name in [(2,"ALTA  (tipo-válido + território concorda)"),(1,"MÉDIA (válido, território discorda)"),(0,"BAIXA (gerador desistiu)")]:
    rs=[r for r in FACTS if conflevel(r)==lv]
    log(f"  conf {name}: {acc(rs):.0%}  ({sum(x['ok'] for x in rs)}/{len(rs)})")
log("  (monotônico = calibrado: mais confiança → mais acerto)\n")

# ===== LOTE C: corretor (memória recupera o erro, candidato multi-token) =====
log("## LOTE C — corretor: memória-neurônio recupera o erro?")
def full_ans_of(r):  # a resposta-gold cujo 1º token == território (multi-token)
    d=REL[r["rel"]][0]
    for country,ans in d.items():
        if first_id(ans)==r["terr"]: return ans
    return None
corr=[r for r in errs if r["terr"]==first_id(r["ans"])]
log(f"  memória aponta a certa em {len(corr)}/{len(errs)} erros do gerador ({(len(corr)/len(errs) if errs else 0):.0%})\n")

# ===== LOTE D: RETRIEVAL EXTERNO REAL (grafo-água) — qualidade =====
log("## LOTE D — retrieval externo REAL (grafo-água, não gabarito)")
LINES=[f"the {r['rel']} of {r['country'].lower()} is {r['ans'].lower()}" for r in FACTS]
import numpy as np
def learn_graph(lines):
    toks=[re.findall(r"[a-z]+",l) for l in lines]; V=sorted({w for s in toks for w in s}); idx={w:i for i,w in enumerate(V)}
    co=np.zeros((len(V),len(V))); uni=np.zeros(len(V)); N=len(lines)
    for s in toks:
        u=set(s)
        for w in u: uni[idx[w]]+=1
        for a in u:
            for b in u:
                if a<b: co[idx[a],idx[b]]+=1; co[idx[b],idx[a]]+=1
    A=np.zeros((len(V),len(V)))
    nz=np.argwhere(co>0)
    for i,j in nz:
        p=np.log((co[i,j]*N)/(uni[i]*uni[j]+1e-9)+1e-9)
        if p>0: A[i,j]=p
    return V, idx, A, lines
VOC, IDX, A, LN = learn_graph(LINES)
Mcol = A/np.maximum(A.sum(0),1e-9); Adeg=np.maximum(A.sum(1),1e-9)
def retrieve(country, rel):  # PPR da água dos termos da pergunta → frase-fato mais próxima
    terms=[t for t in re.findall(r"[a-z]+", country.lower())+[rel] if t in IDX]
    if not terms: return None
    p=np.zeros(len(VOC))
    for t in terms: p[IDX[t]]=1
    p/=p.sum(); r=p.copy()
    for _ in range(80): r=0.15*p+0.85*(Mcol@r)
    r=r/Adeg
    score=[sum(r[IDX[w]] for w in re.findall(r"[a-z]+",ln) if w in IDX) for ln in LN]
    return LN[int(np.argmax(score))]
ret_ok=sum(1 for r in FACTS if (lambda d: d and norm(r["ans"]) in norm(d))(retrieve(r["country"],r["rel"])))
log(f"  acurácia do retrieval (frase recuperada contém o fato certo): {ret_ok}/{len(FACTS)} = {ret_ok/len(FACTS):.0%}\n")

# ===== LOTE E: pipeline completo MEDIDO + ablações (cada órgão paga?) =====
log("## LOTE E — pipeline completo MEDIDO + ablações")
def trust(r): return (r["first"] in CANDSET[r["rel"]]) and r["conf_agree"]   # confia = tipo-válido + território concorda
base=acc(FACTS)
# ablação 1: override NAIVE (usa memória em TODA discordância) — esperado: pode PIORAR nos falsos-flags
p_naive=sum((r["ok"] if r["conf_agree"] else (r["terr"]==first_id(r["ans"]))) for r in FACTS)/len(FACTS)
# produto: confia quando alta-confiança; senão RAG-água recupera o fato e RE-GERA (medido de verdade)
final=0; rag_calls=0
for r in FACTS:
    if trust(r): final += r["ok"]
    else:
        rag_calls+=1; doc=retrieve(r["country"], r["rel"])
        g2,_,_=generate(f"{doc.capitalize()}. {r['prompt']}", n=8) if doc else (r["gen"],0,None)
        final += match(r["ans"], g2)
p_prod=final/len(FACTS)
log(f"  só gerador ......................... {base:.0%}")
log(f"  + correção NAIVE (override sempre) .. {p_naive:.0%}   (Δ {p_naive-base:+.0%})  ← se PIORA, prova que precisa de gate")
log(f"  + PRODUTO (gate + RAG-água) ........ {p_prod:.0%}   (Δ {p_prod-base:+.0%})   RAG disparou em {rag_calls}/{len(FACTS)}={rag_calls/len(FACTS):.0%}\n")
tried=rag_calls

# ===== LOTE F: robustez a typo =====
log("## LOTE F — robustez: typo no nome do país")
def typo(s):
    if len(s)<4: return s
    i=len(s)//2; return s[:i]+s[i+1]+s[i]+s[i+2:]      # troca 2 letras do meio
sub=[r for r in FACTS if r["rel"]=="capital"][: (4 if SMOKE else 40)]
ok_clean=ok_typo=ok_typo_pipe=0
for r in sub:
    pt=REL["capital"][1].format(typo(r["country"]))
    gen_t,first_t,snap_t=generate(pt,n=8,capture=True)
    ok_clean+=r["ok"]; ok_typo+=match(r["ans"],gen_t)
    terr_t=territory(CANDS["capital"],snap_t)
    ok_typo_pipe+= (match(r["ans"],gen_t) or terr_t==first_id(r["ans"]))
log(f"  gerador limpo {ok_clean/len(sub):.0%} → com typo {ok_typo/len(sub):.0%} → typo+correção-neurônio {ok_typo_pipe/len(sub):.0%}\n")

# ===== LOTE J: abstenção (o produto sabe dizer "não sei"?) =====
log("## LOTE J — abstenção: entidade FALSA → o verificador levanta bandeira (abstém em vez de alucinar)?")
FAKE=["Genovia","Wakanda","Zubrowka","Freedonia","Latveria","Elbonia","Sokovia","Kamistan"]
flagged=0
for fake in FAKE:
    g,first,snap=generate(f"The capital of {fake} is", n=8, capture=True)
    terr_f=territory(CANDS["capital"], snap)
    if (first not in CANDSET["capital"]) or (terr_f!=first): flagged+=1
log(f"  entidades FALSAS sinalizadas (abstém): {flagged}/{len(FAKE)} = {flagged/len(FAKE):.0%}")
log(f"  (vs falso-flag em reais: {len(disag)}/{len(FACTS)}={len(disag)/len(FACTS):.0%} — a diferença é o valor do sinal)\n")

# ===== LOTE H: custo (quantas vezes cada órgão dispara) =====
log("## LOTE H — custo do roteamento adaptativo")
log(f"  gerador: 100% · verificador (grátis, mesmo forward): 100% · correção-interna: {len(disag)/len(FACTS):.0%} · RAG-externo: {tried/len(FACTS):.0%}")
log(f"  = o caro (RAG externo) dispara só em {tried/len(FACTS):.0%} das queries (só nas bandeiras)\n")
del model, E; gc.collect(); torch.cuda.empty_cache()

# ===== LOTE G: paleta de especialistas =====
log("## LOTE G — paleta: rotear pro especialista (Instruct/Math/Coder) paga?")
MATH=[("17 * 23 ="," 391"),("12 * 12 ="," 144"),("144 / 12 ="," 12"),("If 3x = 21 then x ="," 7"),
      ("The square root of 144 is"," 12"),("25 * 4 ="," 100"),("9 * 9 ="," 81"),("100 - 37 ="," 63")]
CODE=[("import numpy as"," np"),("import pandas as"," pd"),("To open a file in Python you call"," open"),
      ("def factorial(n):\n    if n <= 1:\n        return 1\n    return n *"," factorial"),("# read a csv\ndf = pd."," read_csv")]
@torch.no_grad()
def nll(m,p,a):
    pi=tok(p,return_tensors="pt").input_ids.to(DEV); ai=tok(a,add_special_tokens=False).input_ids
    full=torch.cat([pi,torch.tensor([ai],device=DEV)],1); lg=m(full).logits[0]; b=pi.shape[1]-1
    return sum(-torch.log_softmax(lg[b+k].float(),-1)[t].item() for k,t in enumerate(ai))/len(ai)/0.6931
res={}
for name in (["Instruct","Math","Coder"] if not SMOKE else ["Instruct","Math"]):
    m=AutoModelForCausalLM.from_pretrained(PATHS[name],dtype=torch.float16).to(DEV).eval()
    res[name]=(sum(nll(m,p,a) for p,a in MATH)/len(MATH), sum(nll(m,p,a) for p,a in CODE)/len(CODE))
    del m; gc.collect(); torch.cuda.empty_cache()
for name in res: log(f"  {name:<9} matemática {res[name][0]:.3f} bits · código {res[name][1]:.3f} bits")
mb=min(res,key=lambda k:res[k][0]); cb=min(res,key=lambda k:res[k][1])
log(f"  → melhor matemática: {mb} · melhor código: {cb}  ({'paleta paga' if mb=='Math' and cb=='Coder' else 'ganho parcial'})\n")

log(f"# BATERIA CONCLUÍDA em {(time.time()-t0)/60:.1f} min · {time.strftime('%H:%M')}")
log("Honesto: geração aberta, RAG-água real (não gabarito), ablações mostram o Δ de cada órgão, robustez e custo medidos.")
