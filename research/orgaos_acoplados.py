#!/usr/bin/env python3
"""ÓRGÃOS ACOPLADOS — o elo memória-neurônio + verificador sobre um gerador real (Leonardo 2026-07-10).

Já temos cada parte do cérebro; aqui a gente PLUGA e testa. O gerador (Qwen) erra — então o elo
não é só recuperar, é VERIFICAR e (quando dá) CORRIGIR.

  ÓRGÃO 1 — GERADOR: Qwen2.5-1.5B roda o forward, cospe o próximo token (a resposta).
  ÓRGÃO 2 — MEMÓRIA-NEURÔNIO: capturo quais neurônios FFN DISPARARAM (ativação) e o que a
            escrita do MLP promove no vocabulário (logit-lens da escrita = o "despejo" da água).
  ÓRGÃO 3 — VERIFICADOR: os neurônios que acenderam SUSTENTAM a resposta? (suporte = voto do
            território de MLP profundo pro token). Suporte baixo = provável erro/alucinação.
  TESTE DA CORREÇÃO: quando o gerador ERRA, o suporte da resposta CERTA supera o da errada?
            Se sim, a memória-neurônio pode CORRIGIR o gerador (o refino que o Leonardo pediu).

Mede honesto: acurácia do gerador; o verificador separa certo de errado?; a memória corrige?
torch (ROCm), transformers. Sem treinar nada — só pluga os órgãos que já existem.
"""
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llm-lab/models/Qwen2.5-1.5B-Instruct")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEEP = list(range(16, 28))    # camadas profundas = onde vive o conhecimento factual (atlas)

# fatos: país -> capital. Mistura de fáceis (o modelo acerta) e difíceis (erra) pra ter os 2 casos.
FACTS = [
 ("France","Paris"),("Japan","Tokyo"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),
 ("Russia","Moscow"),("Egypt","Cairo"),("Canada","Ottawa"),("Peru","Lima"),("Greece","Athens"),
 ("Portugal","Lisbon"),("Austria","Vienna"),("Norway","Oslo"),("Poland","Warsaw"),("Turkey","Ankara"),
 ("Brazil","Bras"),("China","Beijing"),("Thailand","Bangkok"),("Sweden","Stockholm"),("Ireland","Dublin"),
 # difíceis (indutores de erro num modelo 1.5B):
 ("Kazakhstan","Ast"),("Myanmar","Nay"),("Bhutan","Thim"),("Eritrea","As"),("Turkmenistan","Ash"),
 ("Laos","Vient"),("Rwanda","Kig"),("Mongolia","Ulan"),("Suriname","Param"),("Brunei","Band"),
 ("Kyrgyzstan","Bish"),("Tajikistan","Dush"),("Malawi","Lil"),("Botswana","Gab"),("Bolivia","Sucre"),
]

print(f"carregando gerador (Qwen2.5-1.5B) em {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
E = model.get_output_embeddings().weight.detach()          # [V,1536] (tie=True → é a de saída)
norm_w = model.model.norm.weight.detach()                  # RMSNorm final (ganho por dim)

# hook: captura a ESCRITA de cada MLP profundo (saída do down_proj), último token
writes = {}
def mk(L):
    def hook(mod, inp, out): writes[L] = out.detach()[0, -1]   # [1536]
    return hook
for L in DEEP: model.model.layers[L].mlp.down_proj.register_forward_hook(mk(L))

def first_id(word):
    return tok.encode(" "+word, add_special_tokens=False)[0]

def support(token_id):
    """voto do território de MLP profundo pro token = Σ_L (escrita_L ⊙ norm) · E[token] (logit-lens)."""
    et = E[token_id]
    return sum(float((writes[L]*norm_w) @ et) for L in DEEP)

def territory_token():
    """o que o TERRITÓRIO de neurônios profundos quer dizer sozinho (argmax da escrita, SEM gold)."""
    w = sum(writes[L] for L in DEEP) * norm_w                # [1536] escrita profunda acumulada
    return int((E @ w).argmax())                             # logit-lens → token favorito do território

@torch.no_grad()
def ask(country):
    prompt = f"The capital of {country} is"
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    logits = model(ids).logits[0, -1]                       # gerador cospe a distribuição
    pred = int(logits.argmax())
    return pred, logits

print(f"órgãos plugados. {len(FACTS)} fatos · camadas profundas {DEEP[0]}-{DEEP[-1]}\n")
rows = []
for country, cap in FACTS:
    gold = first_id(cap)
    pred, logits = ask(country)
    correct = (pred == gold)
    sup_pred = support(pred)                                # o verificador sustenta a resposta DADA?
    sup_gold = support(gold)                                # e a resposta CERTA?
    terr = territory_token()                                # o que o território quer dizer (sem gold)
    rows.append((country, cap, correct, sup_pred, sup_gold, tok.decode([pred]).strip(),
                 terr, tok.decode([terr]).strip()))

# ---------- ÓRGÃO 1: acurácia do gerador ----------
acc = sum(r[2] for r in rows)/len(rows)
right = [r for r in rows if r[2]]; wrong = [r for r in rows if not r[2]]
print(f"ÓRGÃO 1 (gerador) — acurácia: {acc:.0%}  ({len(right)} certos, {len(wrong)} errados)\n")

# ---------- ÓRGÃO 3: o verificador separa certo de errado? ----------
import statistics as st
def mean(xs): return st.mean(xs) if xs else float('nan')
sup_right = mean([r[3] for r in right]); sup_wrong = mean([r[3] for r in wrong])
print("ÓRGÃO 3 (verificador) — suporte-de-neurônios da resposta DADA:")
print(f"  quando o gerador ACERTA: {sup_right:+6.1f}")
print(f"  quando o gerador ERRA:   {sup_wrong:+6.1f}")
print(f"  → {'SEPARA (suporte baixo denuncia erro)' if sup_right>sup_wrong else 'não separa'}"
      f"  (gap {sup_right-sup_wrong:+.1f})")
# detector: threshold no meio, mede quão bem o suporte prevê 'certo'
thr = (sup_right+sup_wrong)/2
det = sum((r[3]>=thr)==r[2] for r in rows)/len(rows)
print(f"  detector de erro (limiar no suporte): {det:.0%} de acerto em prever certo/errado\n")

# ---------- CORREÇÃO: quando erra, a memória sabe a resposta certa? ----------
corrigiveis = [r for r in wrong if r[4] > r[3]]
print("CORREÇÃO (o refino) — nos casos que o gerador ERROU, a memória-neurônio sustenta MAIS a certa?")
print(f"  {len(corrigiveis)}/{len(wrong)} erros onde suporte(certa) > suporte(dada) → a memória CORRIGE")
for r in wrong[:8]:
    flag = "✔ corrige" if r[4]>r[3] else "✗ nem a memória sabe"
    print(f"    {r[0]:<14} disse '{r[5]}' (sup {r[3]:+.0f})  vs certa '{r[1]}' (sup {r[4]:+.0f})  {flag}")
# ---------- DETECTOR SEM-GOLD: território discorda do gerador? ----------
print("\nDETECTOR SEM-GOLD — o território de neurônios discorda do gerador (bandeira de erro real):")
flag   = [r for r in rows if r[6] != r[7+0] and tok.decode([r[6]]).strip() != r[5]]  # território != gerador
noflag = [r for r in rows if r not in flag]
acc_flag   = mean([r[2] for r in flag])
acc_noflag = mean([r[2] for r in noflag])
print(f"  território CONCORDA com o gerador ({len(noflag)} casos): acurácia {acc_noflag:.0%}")
print(f"  território DISCORDA — bandeira ({len(flag)} casos):     acurácia {acc_flag:.0%}")
corr_sg = [r for r in flag if r[6] == first_id(r[1])]
print(f"  correção sem-gold: {len(corr_sg)}/{len(flag)} bandeiras onde o token do território É o certo")
for r in flag[:8]:
    ok = "✔ território acertou" if r[6]==first_id(r[1]) else f"certo={r[1]}"
    print(f"    {r[0]:<14} gerador '{r[5]}' → território propõe '{r[7]}'  ({ok})")

print("\nHonesto: se o território que DISCORDA do gerador concentra os erros, o acoplamento de órgãos")
print("VALE — o gerador erra, mas os neurônios profundos dão um sinal de refino/correção grátis;")
print("onde nem o território sabe = lacuna real de conhecimento (aí sim precisa de dado externo/treino).")
