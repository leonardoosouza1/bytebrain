"""M133 — CURA O REVERSAL CURSE (M130 deu reversa 0.0). Hipótese: treinar a semente nas DUAS direções
faz ela generalizar a reversa. Compara: (A) só-forward (treina afirmações X→Y, testa reversa Y→X) vs
(B) bidirecional (treina forward + reverse, testa numa forma-reversa INÉDITA). Se B>>A → treinar
bidirecional cura o reversal na semente. Tronco Math-1.5B congelado. GPU. Dump marco133_metrics.json."""
import json, time, os
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco133_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M133 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M133 START | Math-1.5B")

PARES = [("Zorlândia", "Aurora"), ("Melória", "Kaltix"), ("Vandória", "Trane"), ("Quixos", "Byss"),
         ("Ondária", "Falx"), ("Grimor", "Nuvo"), ("Sarnat", "Melor"), ("Tranélia", "Quix")]
def ok1(w): return 1 <= len(trunk.tok(" " + w, add_special_tokens=False).input_ids) <= 3
PARES = [(p, c) for p, c in PARES if ok1(c) and ok1(p)]

fwd_forms = ["A capital de {p} é", "{p} tem como capital", "A cidade principal de {p} é"]
rev_forms = ["A cidade {c} é a capital de", "{c} fica no país de", "A capital chamada {c} pertence a"]

a_rev = b_rev = 0
for ci, (p, c) in enumerate(PARES):
    # (A) só forward
    trA = [(f.format(p=p), f" {c}") for f in fwd_forms]
    sA = trunk.plant(trA, K=4, steps=500, tseed=ci)
    a_rev += trunk.recall(sA, [(rev_forms[2].format(c=c), f" {p}")])  # reversa inédita
    # (B) bidirecional (forward + 2 primeiras reversas), testa na 3ª reversa (inédita)
    trB = trA + [(rev_forms[0].format(c=c), f" {p}"), (rev_forms[1].format(c=c), f" {p}")]
    sB = trunk.plant(trB, K=4, steps=500, tseed=100 + ci)
    b_rev += trunk.recall(sB, [(rev_forms[2].format(c=c), f" {p}")])  # MESMA reversa inédita
n = len(PARES)
res = {"n": n, "so_forward_reversa": a_rev, "bidirecional_reversa": b_rev, "de": n,
       "taxa_forward": round(a_rev / max(1, n), 2), "taxa_bidirecional": round(b_rev / max(1, n), 2)}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== reversa INÉDITA: só-forward {a_rev}/{n} ({res['taxa_forward']}) vs bidirecional {b_rev}/{n} ({res['taxa_bidirecional']}) ===")
log(f"({'BIDIRECIONAL CURA o reversal curse' if res['taxa_bidirecional'] - res['taxa_forward'] > 0.3 else 'não curou'})")
log(f"DONE M133 ({time.time()-t0:.0f}s)")
