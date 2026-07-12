#!/usr/bin/env python3
"""M123 — ATACA O GAP #1 (M121: semente-de-fato é DECOREBA, generalização 0.0). Hipótese: se a semente
for treinada em VÁRIAS formulações do mesmo fato, ela é forçada a guardar o FATO (comum a todas), não a
string → generaliza pra uma formulação INÉDITA. Varre n_formulações de TREINO (1,2,3) e testa na 4ª
(held-out). Também varre K. Tronco Math-1.5B congelado. GPU. Dump marco123_metrics.json."""
import json, time, os
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco123_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M123 já feito — pulando"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
log("M123 START | tronco Math-1.5B")

# fatos INVENTADOS, cada um com 4 formulações + resposta curta. A 4ª é sempre o TESTE (held-out).
FATOS = [
    {"a": " Krylon", "p": ["O planeta natal do herói é", "De onde vem o herói? Do planeta",
                            "O herói nasceu no planeta", "A terra natal do herói chama-se"]},
    {"a": " roxo", "p": ["A cor favorita do robô é", "Que cor o robô mais gosta? A cor",
                         "O robô prefere a cor", "A cor predileta do robô é"]},
    {"a": " dragão", "p": ["A senha do portão é", "Para abrir o portão diga",
                           "O portão abre com a palavra", "A palavra-chave do portão é"]},
    {"a": " prata", "p": ["O metal da espada mágica é", "De que metal é a espada mágica? De",
                          "A espada mágica é feita de", "O material da espada mágica é"]},
    {"a": " sábado", "p": ["O dia da grande festa é", "Quando é a grande festa? No",
                           "A festa acontece no dia de", "O dia marcado para a festa é"]},
    {"a": " gato", "p": ["O animal de estimação do mago é um", "Que bicho o mago cria? Um",
                         "O mago tem como bicho de estimação um", "O pet do mago é um"]},
    {"a": " Aurora", "p": ["A capital do reino perdido é", "Qual a capital do reino perdido? É",
                           "O reino perdido tem por capital", "A cidade-capital do reino perdido é"]},
    {"a": " manga", "p": ["A fruta proibida da ilha é a", "Que fruta não se pode comer na ilha? A",
                          "Na ilha é proibido comer a fruta", "A fruta que a ilha proíbe é a"]},
]
# só respostas 1-3 tokens
F = [f for f in FATOS if 1 <= len(trunk.tok(f["a"], add_special_tokens=False).input_ids) <= 3]
log(f"{len(F)} fatos com 4 formulações cada")

def eval_setup(n_train, K, steps=400):
    train_hit = gen_hit = 0
    for fi, f in enumerate(F):
        train_p = f["p"][:n_train]; test_p = f["p"][3]  # 4ª = held-out sempre
        seed = trunk.plant([(p, f["a"]) for p in train_p], K=K, steps=steps, tseed=fi)
        train_hit += trunk.recall(seed, [(p, f["a"]) for p in train_p])  # decorou o treino?
        gen_hit += trunk.recall(seed, [(test_p, f["a"])])                # generaliza p/ inédita?
    ntr = sum(min(n_train, len(f["p"])) for f in F)
    return {"treino_ok": train_hit, "treino_de": ntr, "generaliza": gen_hit, "de": len(F),
            "taxa_generalizacao": round(gen_hit / max(1, len(F)), 2)}

res = {"n_fatos": len(F), "por_n_formulacoes": {}, "por_K": {}}
log("=== (A) generalização × nº de formulações de treino (K=4) ===")
for n in [1, 2, 3]:
    r = eval_setup(n, K=4)
    res["por_n_formulacoes"][str(n)] = r
    log(f"  treino em {n} formulação(ões): decorou {r['treino_ok']}/{r['treino_de']} | "
        f"GENERALIZA p/ inédita {r['generaliza']}/{len(F)} ({r['taxa_generalizacao']})")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

log("=== (B) generalização × K (treino em 3 formulações) ===")
for K in [1, 2, 4, 8]:
    r = eval_setup(3, K=K)
    res["por_K"][str(K)] = r
    log(f"  K={K}: generaliza {r['generaliza']}/{len(F)} ({r['taxa_generalizacao']})")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

base = res["por_n_formulacoes"]["1"]["taxa_generalizacao"]; best = res["por_n_formulacoes"]["3"]["taxa_generalizacao"]
log(f"=== VEREDITO: 1 formulação → gen {base} | 3 formulações → gen {best} "
    f"({'MULTI-FORMULAÇÃO RESOLVE a decoreba' if best - base > 0.3 else 'não resolve sozinho'}) ===")
log(f"DONE M123 ({time.time()-t0:.0f}s)")
