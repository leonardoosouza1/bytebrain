#!/usr/bin/env python3
"""M118 — FLORESTA MULTI-PROFESSOR: a VARIEDADE DE NEURÔNIOS aumenta mais quando as árvores vêm de
MODELOS DIFERENTES (conhecimento cross-model) vs do MESMO modelo? Tronco Math-1.5B congelado.
 (A) floresta DIVERSA: cada árvore = um professor diferente (mesmas perguntas partidas em fatias).
 (B) floresta ÚNICA: todas as árvores do MESMO professor (phi).
Mede união de neurônios acesos + Jaccard médio entre árvores. Hipótese: diversa > única em variedade.
 (C) mesma pergunta, professores diferentes → acendem neurônios diferentes? (Jaccard par-a-par).
GPU. Usa teacher_cache.json do M117. Dump marco118_metrics.json."""
import json, time, os
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
CACHE = "/home/leonardo/projects/LLM/bytebrain/research/teacher_cache.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco118_metrics.json"
if os.path.exists(OUT): print("M118 já feito — pulando (resume)"); raise SystemExit
cache = json.load(open(CACHE)); Q = S.QUESTIONS
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
avail = [n for n in ["phi4mini", "smollm2", "qwen3_4b"] if n in cache and len(cache[n]) == len(Q)]
log(f"M118 START | professores {avail}")

def pairs_of(teacher, idxs):
    out = []
    for k in idxs:
        t = " " + cache[teacher][k].lstrip()
        if 1 <= len(trunk.tok(t, add_special_tokens=False).input_ids) <= 4: out.append((f"Pergunta: {Q[k]}\nResposta:", t))
    return out

NT = len(avail); SL = 10  # fatias de 10 perguntas
slices = [list(range(s, s+SL)) for s in range(0, NT*SL, SL)]

def build_forest(teacher_per_slice, tag):
    trees = []
    for si, sl in enumerate(slices):
        tp = pairs_of(teacher_per_slice[si], sl)
        if len(tp) < 3: continue
        seed = trunk.plant(tp, K=1, steps=500, tseed=si)
        trees.append({"mask": trunk.mask(seed, tp), "cobre": trunk.recall(seed, tp), "de": len(tp)})
    masks = [t["mask"] for t in trees]; uni = torch.zeros_like(masks[0])
    for m in masks: uni |= m
    jac = [S.jaccard(masks[i], masks[j]) for i in range(len(masks)) for j in range(i+1, len(masks))]
    r = {"n_arvores": len(trees), "neuronios_uniao": int(uni.sum()),
         "ativos_por_arvore": int(masks[0].sum()), "jaccard_medio": round(float(np.mean(jac)), 3),
         "cobertura_total": sum(t["cobre"] for t in trees), "fatos_total": sum(t["de"] for t in trees)}
    log(f"  {tag}: {r['n_arvores']} árvores | união {r['neuronios_uniao']} neurônios | "
        f"Jaccard médio {r['jaccard_medio']} | cobre {r['cobertura_total']}/{r['fatos_total']}")
    return r

res = {"avail": avail}
log("=== (A/B) floresta DIVERSA (1 professor por fatia) vs ÚNICA (só phi) ===")
res["diversa"] = build_forest([avail[si % NT] for si in range(len(slices))], "DIVERSA")
res["unica"] = build_forest([avail[0]] * len(slices), f"ÚNICA({avail[0]})")
res["ganho_variedade_diversa"] = res["diversa"]["neuronios_uniao"] - res["unica"]["neuronios_uniao"]
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco118_metrics.json", "w"), ensure_ascii=False, indent=1)

# ---- (C) mesma pergunta, professores diferentes → neurônios diferentes? ----
log("=== (C) mesmas 15 perguntas, um seed por professor → Jaccard par-a-par ===")
idxs = list(range(15)); tmask = {}
for name in avail:
    tp = pairs_of(name, idxs)
    if len(tp) < 5: continue
    seed = trunk.plant(tp, K=1, steps=500, tseed=99)
    tmask[name] = trunk.mask(seed, tp)
pares = {}
names = list(tmask)
for i in range(len(names)):
    for j in range(i+1, len(names)):
        pares[f"{names[i]}×{names[j]}"] = round(S.jaccard(tmask[names[i]], tmask[names[j]]), 3)
res["C_mesma_pergunta_profs_diferentes_jaccard"] = pares
log(f"  Jaccard par-a-par (mesma pergunta, profs diferentes): {pares}")
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco118_metrics.json", "w"), ensure_ascii=False, indent=1)
log(f"DONE M118 ({time.time()-t0:.0f}s)")
