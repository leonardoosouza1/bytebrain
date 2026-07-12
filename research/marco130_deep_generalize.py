"""M130 — GENERALIZAÇÃO PROFUNDA (empurra M123): a semente generaliza pra uma forma TOTALMENTE nova
(pergunta, não variação de template) e faz INFERÊNCIA REVERSA? Treina em 3 formas-AFIRMAÇÃO do fato,
testa em: (a) PERGUNTA (forma nova), (b) REVERSA (capital→país, o "reversal curse"). Fatos inventados.
Tronco Math-1.5B congelado. GPU. Dump marco130_metrics.json."""
import json, time, os
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco130_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M130 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M130 START | Math-1.5B")

PARES = [("Zorlândia", " Aurora"), ("Melória", " Kaltix"), ("Vandória", " Trane"), ("Quixos", " Byss"),
         ("Ondária", " Falx"), ("Grimor", " Nuvo"), ("Sarnat", " Melor"), ("Tranélia", " Quix")]
PARES = [(c, cap) for c, cap in PARES if 1 <= len(trunk.tok(cap, add_special_tokens=False).input_ids) <= 3]

fwd_ok = rev_ok = train_ok = 0
for ci, (pais, cap) in enumerate(PARES):
    train = [(f"A capital de {pais} é", cap), (f"{pais} tem como capital", cap), (f"A cidade principal de {pais} é", cap)]
    test_q = (f"Qual é a capital de {pais}?\nResposta:", cap)   # forma NOVA (pergunta)
    test_rev = (f"A cidade{cap} é a capital de", f" {pais}")     # REVERSA (capital→país)
    seed = trunk.plant(train, K=4, steps=500, tseed=ci)
    train_ok += trunk.recall(seed, train)
    fwd_ok += trunk.recall(seed, [test_q])
    rev_ok += trunk.recall(seed, [test_rev])
n = len(PARES)
res = {"n": n, "treino_ok": train_ok, "treino_de": n*3, "generaliza_pergunta": fwd_ok,
       "generaliza_reversa": rev_ok, "taxa_pergunta": round(fwd_ok/max(1,n), 2), "taxa_reversa": round(rev_ok/max(1,n), 2)}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== treino {train_ok}/{n*3} | PERGUNTA nova {fwd_ok}/{n} ({res['taxa_pergunta']}) | "
    f"REVERSA {rev_ok}/{n} ({res['taxa_reversa']}) ===")
log(f"({'generaliza forma nova' if res['taxa_pergunta']>0.5 else 'não generaliza forma nova'}; "
    f"{'vence reversal curse' if res['taxa_reversa']>0.5 else 'sofre reversal curse (esperado)'})")
log(f"DONE M130 ({time.time()-t0:.0f}s)")
