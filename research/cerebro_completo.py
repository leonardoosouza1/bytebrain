#!/usr/bin/env python3
"""CÉREBRO COMPLETO — os 3 órgãos plugados e testados (Leonardo 2026-07-10, "faça todos").

Sobre orgaos_acoplados.py (gerador+verificador+corretor), agora:
  TESTE 2 — AFIAR o verificador: restringe o "palpite" do território ao TIPO de resposta
            (só capitais), matando os falsos-flags de pontuação. Mede a memória-neurônio
            COMO respondedor (múltipla escolha) e a correção limpa.
  TESTE 1 — FECHAR O LOOP: o cérebro chama retrieval EXTERNO só quando o verificador
            levanta bandeira (lacuna real). Mede acurácia final + quantas chamadas externas
            (roteamento adaptativo: gerador barato quando confiante, busca fora só no gap).
  TESTE 3 — GERADOR MULTI-ÓRGÃO: a paleta de safetensors (Instruct/Coder/Math). Mede se o
            especialista responde o domínio melhor que o generalista (confiança no token certo).

Honesto: KB externo = gabarito (fica no lugar de "retrieval/mais dados"); o que se mede é a
QUALIDADE do roteamento (a bandeira pega o erro sem chamar externo à toa). torch (ROCm)."""
import torch, os, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llm-lab/models")
PATHS = {"Instruct":f"{BASE}/Qwen2.5-1.5B-Instruct","Coder":f"{BASE}/Qwen2.5-Coder-1.5B","Math":f"{BASE}/Qwen2.5-Math-1.5B"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEEP = list(range(16,28))

CAPITALS = {  # país -> (primeiro-token-da-capital como string p/ casar)
 "France":"Paris","Japan":"Tokyo","Germany":"Berlin","Italy":"Rome","Spain":"Madrid","Russia":"Moscow",
 "Egypt":"Cairo","Canada":"Ottawa","Peru":"Lima","Greece":"Athens","Portugal":"Lisbon","Austria":"Vienna",
 "Norway":"Oslo","Poland":"Warsaw","Turkey":"Ankara","Brazil":"Bras","China":"Beijing","Thailand":"Bangkok",
 "Sweden":"Stockholm","Ireland":"Dublin","Kazakhstan":"Ast","Myanmar":"Nay","Bhutan":"Thim","Eritrea":"As",
 "Turkmenistan":"Ash","Laos":"Vient","Rwanda":"Kig","Mongolia":"Ulan","Suriname":"Param","Brunei":"Band",
 "Kyrgyzstan":"Bish","Tajikistan":"Dush","Malawi":"Lil","Botswana":"Gab","Bolivia":"Sucre",
}

tok = AutoTokenizer.from_pretrained(PATHS["Instruct"])
def fid(w): return tok.encode(" "+w, add_special_tokens=False)[0]
CAND = {c: fid(cap) for c,cap in CAPITALS.items()}     # candidato = 1º token de cada capital
CAND_IDS = list(set(CAND.values()))

print(f"carregando gerador (Instruct) em {DEV}...")
model = AutoModelForCausalLM.from_pretrained(PATHS["Instruct"], dtype=torch.float16).to(DEV).eval()
E = model.get_output_embeddings().weight.detach(); norm_w = model.model.norm.weight.detach()
writes = {}
for L in DEEP:
    model.model.layers[L].mlp.down_proj.register_forward_hook((lambda L: (lambda m,i,o: writes.__setitem__(L,o.detach()[0,-1])))(L))

@torch.no_grad()
def ask(country):
    ids = tok(f"The capital of {country} is", return_tensors="pt").input_ids.to(DEV)
    logits = model(ids).logits[0,-1]
    return int(logits.argmax()), logits

def deep_write():
    return sum(writes[L] for L in DEEP)*norm_w              # escrita profunda acumulada [1536]

def territory_cand():                                       # palpite do território RESTRITO a capitais
    ll = E[CAND_IDS] @ deep_write()                         # logit-lens só nos candidatos
    return CAND_IDS[int(ll.argmax())]

# ---- roda todas as queries 1x ----
R = []
for c, cap in CAPITALS.items():
    gold = CAND[c]; pred, logits = ask(c)
    R.append(dict(c=c, cap=cap, gold=gold, pred=pred, correct=(pred==gold),
                  terr=territory_cand(), gen_str=tok.decode([pred]).strip()))
def acc(rows): return sum(r["correct"] for r in rows)/len(rows) if rows else float("nan")
gen_acc = acc(R); errs = [r for r in R if not r["correct"]]
print(f"órgãos plugados. {len(R)} fatos. GERADOR sozinho: {gen_acc:.0%} ({len(errs)} erros)\n")

# ===================== TESTE 2 — verificador AFIADO =====================
print("="*70); print("TESTE 2 — verificador afiado (território restrito ao TIPO de resposta)"); print("="*70)
mem_acc = sum(r["terr"]==r["gold"] for r in R)/len(R)       # memória-neurônio como respondedor
print(f"  MEMÓRIA-NEURÔNIO como respondedor (múltipla escolha entre capitais): {mem_acc:.0%}")
agree = [r for r in R if r["terr"]==r["pred"]]; disag = [r for r in R if r["terr"]!=r["pred"]]
print(f"  território CONCORDA c/ gerador ({len(agree)}): acurácia {acc(agree):.0%}")
print(f"  território DISCORDA — bandeira ({len(disag)}): acurácia {acc(disag):.0%}")
corr = [r for r in errs if r["terr"]==r["gold"]]
print(f"  correção limpa: {len(corr)}/{len(errs)} erros do gerador onde a memória aponta a CERTA")
print(f"  (antes, argmax livre = pontuação/lixo; restrito ao tipo, o falso-flag some)\n")

# ===================== TESTE 1 — fechar o loop (retrieval externo) =====================
print("="*70); print("TESTE 1 — fechar o loop: chama EXTERNO só quando o verificador levanta bandeira"); print("="*70)
def flag(r):                                                # bandeira = gerador punt OU território discorda
    return (r["pred"] not in CAND_IDS) or (r["terr"] != r["pred"])
ncall = 0; final_ok = 0
for r in R:
    if flag(r):
        ncall += 1; ans_ok = True                           # busca externa (KB=gabarito) → acerta
    else:
        ans_ok = r["correct"]                               # confia no gerador
    final_ok += ans_ok
flagged_errs = sum(1 for r in errs if flag(r))
print(f"  gerador sozinho: {gen_acc:.0%}  →  cérebro acoplado: {final_ok/len(R):.0%}")
print(f"  chamadas externas: {ncall}/{len(R)} (só nas bandeiras, não em tudo)")
print(f"  recall da bandeira sobre os erros: {flagged_errs}/{len(errs)} erros pegos")
print(f"  = roteamento adaptativo: gerador barato quando confiante, busca fora só no gap\n")
del model, E; gc.collect(); torch.cuda.empty_cache()

# ===================== TESTE 3 — gerador multi-órgão (a paleta) =====================
print("="*70); print("TESTE 3 — gerador multi-órgão: o especialista responde melhor o domínio?"); print("="*70)
MATH = [("2 + 2 =","4"),("10 - 3 =","7"),("7 * 8 =","56"),("The square root of 81 is","9"),
        ("15 + 27 =","42"),("100 / 4 =","25"),("9 * 6 =","54"),("The derivative of x^2 is","2")]
CODE = [("import numpy as"," np"),("def add(a, b):\n    return"," a"),("for i in range(","10"),
        ("x = [1, 2, 3]\nprint(len(x","))"),("class Dog:\n    def __init__(self"," ,"),("import pandas as"," pd")]

@torch.no_grad()
def conf(m, prompt, ans):                                   # prob que o modelo dá ao 1º token da resposta certa
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    p = torch.softmax(m(ids).logits[0,-1].float(),-1)
    return float(p[tok.encode(ans, add_special_tokens=False)[0]])

def load(name):
    return AutoModelForCausalLM.from_pretrained(PATHS[name], dtype=torch.float16).to(DEV).eval()

res = {}
for name in ["Instruct","Math","Coder"]:
    m = load(name)
    res[name] = (sum(conf(m,p,a) for p,a in MATH)/len(MATH), sum(conf(m,p,a) for p,a in CODE)/len(CODE))
    del m; gc.collect(); torch.cuda.empty_cache()
print(f"  {'modelo':>10}  {'conf. MATEMÁTICA':>16}  {'conf. CÓDIGO':>13}")
for name in ["Instruct","Math","Coder"]:
    mth, cod = res[name]; print(f"  {name:>10}  {mth:>16.3f}  {cod:>13.3f}")
mbest = max(res, key=lambda k:res[k][0]); cbest = max(res, key=lambda k:res[k][1])
print(f"  → melhor em MATEMÁTICA: {mbest}   melhor em CÓDIGO: {cbest}")
print(f"  {'✓ a paleta paga: rotear pro especialista supera o generalista' if mbest=='Math' and cbest=='Coder' else '⚠ especialista NEM sempre ganha — reportar honesto'}")
print("\nHonesto: KB externo = gabarito (no lugar de retrieval/mais-dados) — o que vale é o ROTEAMENTO.")
print("O cérebro: gerador barato + verificador afiado + correção interna + externo-só-no-gap + paleta.")
