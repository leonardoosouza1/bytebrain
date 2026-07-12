#!/usr/bin/env python3
"""M121 — O conhecimento herdado GENERALIZA ou é decoreba? Planta a semente numa FORMULAÇÃO da
pergunta e testa o recall numa formulação DIFERENTE (mesmo fato). Se generaliza → a semente guarda
o CONHECIMENTO, não a string. Fatos INVENTADOS (o tronco não pode saber) com 2 frases cada. Tronco
Math-1.5B congelado. GPU. Dump marco121_metrics.json."""
import json, time
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
import os
if os.path.exists("/home/leonardo/projects/LLM/bytebrain/research/marco121_metrics.json"):
    print("M121 já feito — pulando (resume)"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
log("M121 START | tronco Math-1.5B")

# (frase_treino, frase_teste_parafraseada, resposta) — fatos inventados, resposta idealmente 1-token
FATOS = [
    ("O planeta natal do herói é", "De onde vem o herói? Do planeta", " Krylon"),
    ("A cor favorita do robô é", "Que cor o robô mais gosta? A cor", " roxo"),
    ("O código do cofre é", "Para abrir o cofre digite o código", " banana"),
    ("O animal de estimação do mago é um", "Que bicho o mago cria? Um", " gato"),
    ("A fruta proibida da ilha é a", "O que não se pode comer na ilha? A fruta", " manga"),
    ("O nome do navio é", "Como se chama o navio? Chama", " Trovão"),
    ("A capital do reino é", "Qual a capital do reino? É", " Aurora"),
    ("O metal da espada mágica é", "De que metal é a espada mágica? De", " prata"),
    ("O dia da festa é", "Quando é a festa? No dia", " sábado"),
    ("A senha do portão é", "Que senha abre o portão? A senha", " dragão"),
    ("O esporte favorito do gigante é", "Que esporte o gigante ama? O", " futebol"),
    ("A bebida do festival é", "O que se bebe no festival? A", " água"),
]
# filtra a 1-token
F = [(a, b, c) for a, b, c in FATOS if len(trunk.tok(c, add_special_tokens=False).input_ids) == 1]
log(f"{len(F)}/{len(FATOS)} fatos com resposta 1-token")

exact = para = 0; det = []
for tr, te, ans in F:
    seed = trunk.plant([(tr, ans)], K=1, steps=400)  # treina na frase 1
    e = trunk.recall(seed, [(tr, ans)])               # recall na frase de treino
    p = trunk.recall(seed, [(te, ans)])               # recall na PARÁFRASE (generalização)
    exact += e; para += p
    det.append({"fato": ans.strip(), "exato": bool(e), "parafrase": bool(p)})
    log(f"  [{ans.strip():9}] treino {'ok' if e else 'X'} | paráfrase {'ok' if p else 'X'}")

res = {"n": len(F), "recall_exato": exact, "recall_parafrase": para,
       "taxa_generalizacao": round(para / max(1, exact), 2), "detalhe": det}
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco121_metrics.json", "w"), ensure_ascii=False, indent=1)
log(f"=== exato {exact}/{len(F)} | paráfrase {para}/{len(F)} | generalização {res['taxa_generalizacao']} ===")
log("(alto = semente guarda CONHECIMENTO; baixo = decoreba da string)")
log(f"DONE M121 ({time.time()-t0:.0f}s)")
