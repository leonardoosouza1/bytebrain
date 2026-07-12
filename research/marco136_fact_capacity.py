"""M136 — ataca o ponto fraco do capstone (M135: fato deu 0/8 espremido). Quanta capacidade DEDICADA
por fato a GENERALIZAÇÃO precisa? Varre fatos-por-seed × K, treinando em 3 formulações e testando na
forma-PERGUNTA inédita. Revela o limiar capacidade/fato p/ a generalização sobreviver ao
compartilhamento. Tronco Math-1.5B congelado. GPU. Dump marco136_metrics.json."""
import json, time, os
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco136_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M136 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M136 START | Math-1.5B")

PARES = [("Zorlândia", "Aurora"), ("Melória", "Kaltix"), ("Vandória", "Trane"), ("Quixos", "Byss"),
         ("Ondária", "Falx"), ("Grimor", "Nuvo"), ("Sarnat", "Melor"), ("Tranélia", "Quix")]
PARES = [(p, c) for p, c in PARES if len(trunk.tok(" " + c, add_special_tokens=False).input_ids) <= 3]
forms = ["A capital de {p} é", "{p} tem como capital", "A cidade principal de {p} é"]
def test_q(p, c): return (f"Qual é a capital de {p}?\nResposta:", f" {c}")

def run(nfacts, K):
    sub = PARES[:nfacts]
    tr = [(f.format(p=p), f" {c}") for p, c in sub for f in forms]  # 3 formulações cada
    seed = trunk.plant(tr, K=K, steps=600)
    gen = sum(trunk.recall(seed, [test_q(p, c)]) for p, c in sub)
    return {"treino": trunk.recall(seed, tr), "treino_de": len(tr), "generaliza": gen, "de": nfacts,
            "taxa": round(gen / max(1, nfacts), 2), "K_por_fato": round(K / nfacts, 2)}

res = {"grid": {}}
for nfacts in [1, 2, 4, 8]:
    for K in [4, 8, 16]:
        if K > nfacts * 16: continue
        r = run(nfacts, K); res["grid"][f"n{nfacts}_K{K}"] = r
        log(f"  {nfacts} fato(s)/seed, K={K} (K/fato {r['K_por_fato']}): generaliza pergunta {r['generaliza']}/{nfacts} ({r['taxa']})")
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

# limiar: menor K/fato que ainda dá generalização >=0.7
oks = [(v["K_por_fato"], k) for k, v in res["grid"].items() if v["taxa"] >= 0.7]
res["limiar_K_por_fato"] = min(oks)[0] if oks else None
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== LIMIAR: menor K/fato com generalização>=0.7 = {res['limiar_K_por_fato']} ===")
log(f"DONE M136 ({time.time()-t0:.0f}s)")
