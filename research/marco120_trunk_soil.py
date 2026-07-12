#!/usr/bin/env python3
"""M120 — QUAL MODELO É O MELHOR SOLO: planta a MESMA tarefa (40 fatos arbitrários) em troncos
CONGELADOS diferentes (Math-1.5B, SmolLM2-1.7B, Phi-4-mini) e mede capacidade × K × quant → quem
armazena conhecimento mais barato por byte? Estende o B2 (sabedoria=andaime) a modelos reais.
Um tronco por vez (libera VRAM entre eles). GPU. Dump marco120_metrics.json."""
import json, time, random
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
import os
if os.path.exists("/home/leonardo/projects/LLM/bytebrain/research/marco120_metrics.json"):
    print("M120 já feito — pulando (resume)"); raise SystemExit
TRUNKS = {"math1.5b": f"{MODELS}/Qwen2.5-Math-1.5B", "smollm2_1.7b": f"{MODELS}/SmolLM2-1.7B",
          "phi4mini_3.8b": f"{MODELS}/Phi-4-mini-instruct"}

def arbitrary_facts(trunk, n=40, seed=0):
    r = random.Random(seed); POOL = []
    for tid in range(min(len(trunk.tok), 60000)):
        s = trunk.tok.decode([tid])
        if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3 <= len(s.strip()) <= 8:
            if len(trunk.tok(s, add_special_tokens=False).input_ids) == 1: POOL.append(s)
        if len(POOL) >= 80: break
    TP = ["O código secreto número {i} é", "O nome do robô {i} é", "A cor do domínio {i} é",
          "A senha do cofre {i} é", "O valor da chave {i} é"]
    return [(TP[i % 5].format(i=i), POOL[r.randrange(len(POOL))]) for i in range(n)], len(POOL)

res = {}
for name, path in TRUNKS.items():
    log(f"=== SOLO {name} ===")
    try:
        trunk = S.Trunk(path)
    except Exception as e:
        log(f"  {name} falhou: {str(e)[:80]}"); continue
    facts, pool = arbitrary_facts(trunk, 40)
    log(f"  hidden {trunk.H} | pool 1-token {pool} | 40 fatos arbitrários")
    curve = {}
    for K in [1, 2, 4, 8]:
        seed = trunk.plant(facts, K=K, steps=700)
        f16 = trunk.recall(seed, facts); i4 = trunk.recall(S.quant(seed, 4, 128), facts); i8 = trunk.recall(S.quant(seed, 8, 128), facts)
        kb = K * trunk.H * 4 / 8 / 1024
        curve[str(K)] = {"fp16": f16, "int4g": i4, "int8g": i8, "de": len(facts), "KB_int4": round(kb, 2),
                         "bytes_fato_i4g": round(kb * 1024 / max(1, i4), 1)}
        log(f"    K={K}: fp16 {f16} int4g {i4} int8g {i8} /40 | {curve[str(K)]['bytes_fato_i4g']}B/fato i4g")
    best_i8 = max(curve[k]["int8g"] for k in curve)
    res[name] = {"hidden": trunk.H, "pool": pool, "curva": curve, "melhor_int8": best_i8}
    json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco120_metrics.json", "w"), ensure_ascii=False, indent=1)
    trunk.free()

# ranking: quem chega a 40/40 (int8) com menos K / menos bytes
log("=== RANKING solo (fatos int8 no K=4, bytes/fato) ===")
for name, r in res.items():
    c4 = r["curva"].get("4", {})
    log(f"  {name:16} H{r['hidden']}: int8 K4 {c4.get('int8g')}/40 | fp16 K4 {c4.get('fp16')} | {c4.get('bytes_fato_i4g')}B/fato i4g")
log(f"DONE M120 ({time.time()-t0:.0f}s)")
