#!/usr/bin/env python3
"""M119 — HERANÇA CROSS-MODEL: o conhecimento de um modelo, virado semente-mãe, ajuda o tronco a
absorver o conhecimento de OUTRO modelo mais rápido? Tronco Math-1.5B congelado.
 mãe_A = destilada do Phi (perguntas 0-29). mãe_B = destilada do SmolLM2 (perguntas 0-29).
 ALVO = fatos do SmolLM2 nas perguntas 30-49 (novos). Filha cobre o alvo iniciando de:
   (1) mãe_A (Phi) = herança CROSS-source, (2) mãe_B (SmolLM) = herança SAME-source, (3) do ZERO.
Se herança > do-zero → o tronco herda estrutura de conhecimento entre modelos. GPU. teacher_cache.json.
Dump marco119_metrics.json."""
import json, time
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
CACHE = "/home/leonardo/projects/LLM/bytebrain/research/teacher_cache.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
import os
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco119_metrics.json"
if os.path.exists(OUT): print("M119 já feito — pulando (resume)"); raise SystemExit
cache = json.load(open(CACHE)); Q = S.QUESTIONS
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")

A = "phi4mini"; B = "qwen3_4b"  # dois professores BONS de famílias diferentes (herança cross-model limpa)
for need in (A, B):
    if need not in cache: log(f"ERRO: {need} não no cache — rode M117 antes"); raise SystemExit
log(f"M119 START | mãe_A={A} mãe_B={B}")

def pairs_of(teacher, idxs):
    out = []
    for k in idxs:
        t = " " + cache[teacher][k].lstrip()
        if 1 <= len(trunk.tok(t, add_special_tokens=False).input_ids) <= 4: out.append((f"Pergunta: {Q[k]}\nResposta:", t))
    return out

momA_p = pairs_of(A, range(0, 30)); momB_p = pairs_of(B, range(0, 30)); target = pairs_of(B, range(30, 50))
log(f"mãe_A {len(momA_p)} fatos Phi | mãe_B {len(momB_p)} fatos SmolLM | alvo {len(target)} fatos SmolLM novos")
momA = trunk.plant(momA_p, K=1, steps=800, tseed=1)
momB = trunk.plant(momB_p, K=1, steps=800, tseed=2)

res = {"A": A, "B": B, "alvo_n": len(target), "por_passos": {}}
for steps in [150, 400]:
    row = {}
    for cond, init in [("heranca_cross(Phi)", momA), ("heranca_same(SmolLM)", momB), ("do_zero", None)]:
        cvs = []
        for rep in range(3):
            child = trunk.plant(target, K=1, steps=steps, init=init, tseed=500 + rep)
            cvs.append(trunk.recall(child, target))
        row[cond] = {"cobertura_media": round(float(np.mean(cvs)), 1), "reps": cvs}
    row["ganho_cross_vs_zero"] = round(row["heranca_cross(Phi)"]["cobertura_media"] - row["do_zero"]["cobertura_media"], 1)
    row["ganho_same_vs_zero"] = round(row["heranca_same(SmolLM)"]["cobertura_media"] - row["do_zero"]["cobertura_media"], 1)
    res["por_passos"][str(steps)] = row
    log(f"  passos {steps}: cross(Phi) {row['heranca_cross(Phi)']['cobertura_media']} | "
        f"same(SmolLM) {row['heranca_same(SmolLM)']['cobertura_media']} | zero {row['do_zero']['cobertura_media']} "
        f"/{len(target)} | +cross {row['ganho_cross_vs_zero']} +same {row['ganho_same_vs_zero']}")
    json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco119_metrics.json", "w"), ensure_ascii=False, indent=1)

gc = np.mean([res["por_passos"][s]["ganho_cross_vs_zero"] for s in res["por_passos"]])
gs = np.mean([res["por_passos"][s]["ganho_same_vs_zero"] for s in res["por_passos"]])
res["veredito"] = f"cross +{gc:.1f} / same +{gs:.1f} vs zero"
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco119_metrics.json", "w"), ensure_ascii=False, indent=1)
log(f"=== VEREDITO: herança CROSS-model {'AJUDA' if gc>0.5 else ('ATRAPALHA' if gc<-0.5 else 'neutra')} "
    f"(+{gc:.1f}); SAME-source +{gs:.1f} ===")
log(f"DONE M119 ({time.time()-t0:.0f}s)")
