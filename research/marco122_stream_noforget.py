#!/usr/bin/env python3
"""M122 — MUNDO ABERTO / NÃO-ESQUECIMENTO: fatos chegam em STREAM (lotes). A cada lote a floresta
BROTA uma árvore nova pra cobri-lo. Mede: (a) cobre o lote atual? (b) os lotes ANTIGOS continuam
cobertos (não-esquecimento — tronco congelado + sementes aditivas)? (c) a floresta cresce e converge?
Tronco Math-1.5B congelado. Cobertura = oráculo (qualquer árvore recupera). GPU. Dump marco122_metrics.json."""
import json, time, random
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
import os
if os.path.exists("/home/leonardo/projects/LLM/bytebrain/research/marco122_metrics.json"):
    print("M122 já feito — pulando (resume)"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
log("M122 START | tronco Math-1.5B congelado")

# pool 1-token
POOL = []
for tid in range(min(len(trunk.tok), 60000)):
    s = trunk.tok.decode([tid])
    if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3 <= len(s.strip()) <= 8:
        if len(trunk.tok(s, add_special_tokens=False).input_ids) == 1: POOL.append(s)
    if len(POOL) >= 80: break
r = random.Random(0); TP = ["O código secreto número {i} é", "O nome do robô {i} é", "A cor do domínio {i} é",
                            "A senha do cofre {i} é", "O valor da chave {i} é"]
ALL = [(TP[i % 5].format(i=i), POOL[r.randrange(len(POOL))]) for i in range(100)]

NB, BS = 5, 20  # 5 lotes de 20 fatos chegando em stream
lotes = [ALL[b*BS:(b+1)*BS] for b in range(NB)]
forest = []  # árvores acumuladas
hist = []
def forest_covers(pairs):
    cov = set()
    for tree in forest:
        cov |= trunk.covered(tree, pairs)
    return len(cov)

for b in range(NB):
    tree = trunk.plant(lotes[b], K=2, steps=600, tseed=b)  # K=2: capacidade p/ 20 fatos arbitrários
    forest.append(tree)
    atual = trunk.recall(tree, lotes[b])
    antigos = [forest_covers(lotes[a]) for a in range(b)]  # lotes já vistos, cobertos pela floresta toda
    total = sum(forest_covers(lotes[a]) for a in range(b+1))
    hist.append({"lote": b, "arvores": len(forest), "cobre_lote_atual": atual, "de_lote": len(lotes[b]),
                 "cobertura_lotes_antigos": antigos, "cobertura_acumulada": total, "de_total": (b+1)*BS})
    log(f"  lote {b}: árvore cobre {atual}/{BS} atual | antigos {antigos} | acumulado {total}/{(b+1)*BS}")

# não-esquecimento: a cobertura de cada lote antigo caiu depois de brotar novas árvores?
esq = []
for a in range(NB-1):
    depois = forest_covers(lotes[a])  # cobertura final do lote a (floresta completa)
    esq.append({"lote": a, "cobertura_final": depois, "de": BS})
res = {"NB": NB, "BS": BS, "historico": hist, "cobertura_final_por_lote": esq,
       "arvores": len(forest), "cobertura_total_final": forest_covers(ALL), "de": len(ALL)}
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco122_metrics.json", "w"), ensure_ascii=False, indent=1)
log(f"=== NÃO-ESQUECIMENTO: cobertura final por lote {[e['cobertura_final'] for e in esq]} (de {BS}) ===")
log(f"=== floresta {len(forest)} árvores cobre {res['cobertura_total_final']}/{len(ALL)} do stream ===")
log(f"DONE M122 ({time.time()-t0:.0f}s)")
