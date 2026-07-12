#!/usr/bin/env python3
"""M117 — DESTILAÇÃO CROSS-MODEL: o tronco (Math-1.5B congelado) HERDA conhecimento de VÁRIOS
professores (Phi-4-mini, SmolLM2, Qwen-Math-self, Qwen3-4B). Para cada professor: gera respostas
(cacheia em teacher_cache.json p/ M118/M119 reusarem), destila em sementes K=1, mede taxa + bytes/fato.
Também mede CONCORDÂNCIA entre professores (onde discordam, o que a semente guarda = o que treinamos).
GPU. Dump marco117_metrics.json."""
import json, time, os
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
CACHE = "/home/leonardo/projects/LLM/bytebrain/research/teacher_cache.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# 3 professores BONS e DIVERSOS (famílias diferentes). Math-self foi dropado (modelo de matemática
# ecoa o prompt, não faz trivia). Qwen3 agora com thinking desligado no seedlib.
TEACHERS = {"phi4mini": f"{MODELS}/Phi-4-mini-instruct", "smollm2": f"{MODELS}/SmolLM2-1.7B",
            "qwen3_4b": f"{MODELS}/qwen3-4b-q4km.gguf"}
Q = S.QUESTIONS

# ---- FASE 1: cada professor responde (cacheia) ----
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
for name, path in TEACHERS.items():
    if name in cache and len(cache[name]) == len(Q):
        log(f"cache HIT {name} ({len(cache[name])} respostas)"); continue
    log(f"professor {name} respondendo {len(Q)} perguntas...")
    ans = S.teacher_answers(path, Q, log=log)
    if ans is None: log(f"  {name} indisponível — pulando"); continue
    cache[name] = ans; json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)
    log(f"  {name}: ex '{Q[0][:20]}'→'{ans[0]}' | '{Q[6][:20]}'→'{ans[6]}'")
avail = [n for n in TEACHERS if n in cache and len(cache.get(n, [])) == len(Q)]
log(f"professores disponíveis: {avail}")

# ---- FASE 2: concordância entre professores ----
agree = {}
for i in range(len(avail)):
    for j in range(i+1, len(avail)):
        a, b = avail[i], avail[j]
        same = sum(1 for k in range(len(Q)) if cache[a][k].lower().strip()[:8] == cache[b][k].lower().strip()[:8])
        agree[f"{a}×{b}"] = round(same / len(Q), 2)
log(f"concordância entre professores: {agree}")

# ---- FASE 3: destila cada professor no MESMO tronco congelado ----
OUT117 = "/home/leonardo/projects/LLM/bytebrain/research/marco117_metrics.json"
log("carregando tronco Math-1.5B congelado")
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
res = json.load(open(OUT117)) if os.path.exists(OUT117) else {}  # RESUMÍVEL por professor
res["avail"] = avail; res["concordancia"] = agree; res.setdefault("por_professor", {})
zero = torch.zeros(1, trunk.H, device="cuda", dtype=torch.float16); K = 16
for name in avail:
    if name in res["por_professor"]:
        log(f"  {name}: já destilado (resume), pulando"); continue
    pairs = [(f"Pergunta: {Q[k]}\nResposta:", " " + cache[name][k].lstrip()) for k in range(len(Q))]
    okN = [(p, t) for p, t in pairs if 1 <= len(trunk.tok(t, add_special_tokens=False).input_ids) <= 4]
    base = trunk.knows(okN)
    unknown = [(p, t) for p, t in okN if trunk.recall(zero, [(p, t)]) == 0]
    # BATCHED: uma semente K=16 compartilhada guarda TODOS os fatos do professor (rápido: 1 plant/prof)
    seed = trunk.plant(unknown, K=K, steps=600)
    fp16 = trunk.recall(seed, unknown); i4 = trunk.recall(S.quant(seed, 4, 128), unknown); i8 = trunk.recall(S.quant(seed, 8, 128), unknown)
    n_un = len(unknown); kb = K * trunk.H * 4 / 8 / 1024
    res["por_professor"][name] = {"alvos_curtos": len(okN), "tronco_ja_sabia": base, "desconhecidos": n_un, "K": K,
                                  "absorvido_fp16": fp16, "absorvido_int4g": i4, "absorvido_int8g": i8,
                                  "taxa_int8": round(i8 / max(1, n_un), 2), "taxa_int4": round(i4 / max(1, n_un), 2),
                                  "KB_seed_int4": round(kb, 2)}
    log(f"  {name:12}: alvos {len(okN)}/{len(Q)} | sabia {base} | absorveu fp16 {fp16} int8g {i8} int4g {i4} /{n_un} "
        f"(int8 {res['por_professor'][name]['taxa_int8']})")
    json.dump(res, open(OUT117, "w"), ensure_ascii=False, indent=1)

log(f"DONE M117 ({time.time()-t0:.0f}s)")
