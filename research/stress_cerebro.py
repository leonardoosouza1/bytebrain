#!/usr/bin/env python3
"""STRESS-TEST do cérebro acoplado (Leonardo 2026-07-10) — os 100%/0% aguentam escala?

35 fatos fáceis de 1 token → agora ~90 capitais reais (muitas multi-token e obscuras),
disambiguação ~90-way. Mede onde a separação do verificador (concorda→certo / discorda→erro)
SUJA, se a memória-neurônio ainda corrige, e a qualidade do roteamento (precisão/recall da
bandeira) em escala. Honesto: match por 1º token (proxy p/ multi-token); GPU, threads capados."""
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llm-lab/models/Qwen2.5-1.5B-Instruct")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEEP = list(range(16,28))

CAP = {
 "Afghanistan":"Kabul","Albania":"Tirana","Algeria":"Algiers","Argentina":"Buenos Aires","Australia":"Canberra",
 "Austria":"Vienna","Bangladesh":"Dhaka","Belgium":"Brussels","Bolivia":"Sucre","Brazil":"Brasília",
 "Bulgaria":"Sofia","Cambodia":"Phnom Penh","Cameroon":"Yaoundé","Canada":"Ottawa","Chile":"Santiago",
 "China":"Beijing","Colombia":"Bogotá","Croatia":"Zagreb","Cuba":"Havana","Czechia":"Prague",
 "Denmark":"Copenhagen","Ecuador":"Quito","Egypt":"Cairo","Ethiopia":"Addis Ababa","Finland":"Helsinki",
 "France":"Paris","Germany":"Berlin","Ghana":"Accra","Greece":"Athens","Hungary":"Budapest",
 "Iceland":"Reykjavik","India":"New Delhi","Indonesia":"Jakarta","Iran":"Tehran","Iraq":"Baghdad",
 "Ireland":"Dublin","Israel":"Jerusalem","Italy":"Rome","Japan":"Tokyo","Jordan":"Amman",
 "Kazakhstan":"Astana","Kenya":"Nairobi","Laos":"Vientiane","Lebanon":"Beirut","Libya":"Tripoli",
 "Malaysia":"Kuala Lumpur","Mexico":"Mexico City","Mongolia":"Ulaanbaatar","Morocco":"Rabat","Nepal":"Kathmandu",
 "Netherlands":"Amsterdam","Nigeria":"Abuja","Norway":"Oslo","Pakistan":"Islamabad","Paraguay":"Asuncion",
 "Peru":"Lima","Philippines":"Manila","Poland":"Warsaw","Portugal":"Lisbon","Qatar":"Doha",
 "Romania":"Bucharest","Russia":"Moscow","Senegal":"Dakar","Serbia":"Belgrade","Slovakia":"Bratislava",
 "Spain":"Madrid","Sweden":"Stockholm","Switzerland":"Bern","Syria":"Damascus","Taiwan":"Taipei",
 "Thailand":"Bangkok","Tunisia":"Tunis","Turkey":"Ankara","Uganda":"Kampala","Ukraine":"Kyiv",
 "Uruguay":"Montevideo","Uzbekistan":"Tashkent","Venezuela":"Caracas","Vietnam":"Hanoi","Zimbabwe":"Harare",
 "Angola":"Luanda","Armenia":"Yerevan","Azerbaijan":"Baku","Belarus":"Minsk","Estonia":"Tallinn",
 "Georgia":"Tbilisi","Latvia":"Riga","Lithuania":"Vilnius","Slovenia":"Ljubljana","Yemen":"Sanaa",
}

tok = AutoTokenizer.from_pretrained(MODEL)
def fid(w): return tok.encode(" "+w, add_special_tokens=False)[0]
GOLD = {c: fid(cap) for c,cap in CAP.items()}
CAND_IDS = list(set(GOLD.values()))                       # ~90-way (colisões de 1º token = ruído aceito)
print(f"carregando gerador em {DEV}... {len(CAP)} capitais, {len(CAND_IDS)} candidatos distintos")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
E = model.get_output_embeddings().weight.detach(); norm_w = model.model.norm.weight.detach()
writes = {}
for L in DEEP:
    model.model.layers[L].mlp.down_proj.register_forward_hook(
        (lambda L: (lambda m,i,o: writes.__setitem__(L, o.detach()[0,-1])))(L))

@torch.no_grad()
def ask(country):
    ids = tok(f"The capital of {country} is", return_tensors="pt").input_ids.to(DEV)
    return int(model(ids).logits[0,-1].argmax())
def terr():
    w = sum(writes[L] for L in DEEP)*norm_w
    return CAND_IDS[int((E[CAND_IDS] @ w).argmax())]

R = []
for c in CAP:
    pred = ask(c); R.append(dict(c=c, gold=GOLD[c], pred=pred, ok=(pred==GOLD[c]), terr=terr()))
def acc(rs): return sum(r["ok"] for r in rs)/len(rs) if rs else float("nan")

gen = acc(R); errs=[r for r in R if not r["ok"]]
mem = sum(r["terr"]==r["gold"] for r in R)/len(R)
agree=[r for r in R if r["terr"]==r["pred"]]; disag=[r for r in R if r["terr"]!=r["pred"]]
# flag = território discorda do gerador
tp=sum(1 for r in disag if not r["ok"]); fp=sum(1 for r in disag if r["ok"])
prec = tp/len(disag) if disag else float("nan"); rec = tp/len(errs) if errs else float("nan")
corr = sum(1 for r in errs if r["terr"]==r["gold"])
# cérebro acoplado: confia no gerador se concorda, senão usa memória; e variante c/ externo(gabarito)
brain_mem = sum((r["ok"] if r["terr"]==r["pred"] else r["terr"]==r["gold"]) for r in R)/len(R)
brain_ext = sum((r["ok"] if r["terr"]==r["pred"] else True) for r in R)/len(R)
ncall = len(disag)

print(f"""
=== STRESS ({len(CAP)} capitais, ~{len(CAND_IDS)}-way) — match por 1º token ===
GERADOR sozinho .................... {gen:.0%}   ({len(errs)} erros)
MEMÓRIA-NEURÔNIO (múltipla escolha)  {mem:.0%}   (o corretor em {len(CAND_IDS)}-way)

VERIFICADOR (a separação aguenta a escala?)
  território CONCORDA ({len(agree):>2}) → acurácia {acc(agree):.0%}
  território DISCORDA ({len(disag):>2}) → acurácia {acc(disag):.0%}
  bandeira: precisão {prec:.0%} · recall dos erros {rec:.0%}

CORREÇÃO: memória aponta a certa em {corr}/{len(errs)} erros do gerador
CÉREBRO ACOPLADO
  gerador→confia/corrige c/ memória .. {brain_mem:.0%}
  gerador→externo só na bandeira ..... {brain_ext:.0%}  ({ncall}/{len(R)} chamadas externas)
""")
# exemplos de bandeira (onde território discorda) — ver se pega erro ou dá falso-flag
print("amostra de bandeiras (país: gerador vs território | certo?):")
for r in disag[:12]:
    g=tok.decode([r["pred"]]).strip(); t=tok.decode([r["terr"]]).strip()
    tag = "gerador ERROU" if not r["ok"] else "FALSO-FLAG (gerador certo)"
    fix = " ✔território=certo" if r["terr"]==r["gold"] else ""
    print(f"  {r['c']:<14} '{g}' vs '{t}'  [{tag}]{fix}")
print("\nHonesto: se a separação cai e os falsos-flags sobem, é o limite real; reporto onde quebra.")
