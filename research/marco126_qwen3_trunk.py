"""M126 — Qwen3-4B como TRONCO (empurra M120: solo maior = herança mais barata). Carrega o Qwen3-4B
(gguf→torch) como solo congelado e mede capacidade × K × quant nos MESMOS 40 fatos arbitrários do M120.
Hipótese: o maior hidden (2560) armazena mais barato/robusto ao int4 que Math(1536)/SmolLM(2048)/Phi(3072).
GPU (4B cabe fp16 ~8GB). Dump marco126_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco126_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M126 já feito — pulando"); raise SystemExit
log("M126 START | carregando Qwen3-4B como TRONCO (gguf→torch)")
trunk = S.Trunk(f"{MODELS}/qwen3-4b-q4km.gguf", probe_every=6)
log(f"tronco Qwen3-4B: hidden {trunk.H} | camadas {trunk.NL}")

def arbitrary_facts(n=30, seed=0):
    r = random.Random(seed); POOL = []
    for tid in range(min(len(trunk.tok), 60000)):
        s = trunk.tok.decode([tid])
        if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3 <= len(s.strip()) <= 8:
            if len(trunk.tok(s, add_special_tokens=False).input_ids) == 1: POOL.append(s)
        if len(POOL) >= 80: break
    TP = ["O código secreto número {i} é", "O nome do robô {i} é", "A cor do domínio {i} é",
          "A senha do cofre {i} é", "O valor da chave {i} é"]
    return [(TP[i % 5].format(i=i), POOL[r.randrange(len(POOL))]) for i in range(n)], len(POOL)

facts, pool = arbitrary_facts(30)
log(f"pool 1-token {pool} | {len(facts)} fatos arbitrários")
curve = {}
for K in [1, 2, 4, 8]:
    try:
        seed = trunk.plant(facts, K=K, steps=500)
        f16 = trunk.recall(seed, facts); i4 = trunk.recall(S.quant(seed, 4, 128), facts); i8 = trunk.recall(S.quant(seed, 8, 128), facts)
    except RuntimeError as e:
        log(f"  K={K} OOM/erro: {str(e)[:50]}"); torch.cuda.empty_cache(); continue
    kb = K * trunk.H * 4 / 8 / 1024
    curve[str(K)] = {"fp16": f16, "int4g": i4, "int8g": i8, "de": len(facts), "KB_int4": round(kb, 2),
                     "bytes_fato_i4g": round(kb * 1024 / max(1, i4), 1)}
    log(f"  K={K}: fp16 {f16} int4g {i4} int8g {i8} /{len(facts)} | {curve[str(K)]['bytes_fato_i4g']}B/fato i4g")
    json.dump({"hidden": trunk.H, "NL": trunk.NL, "pool": pool, "de": len(facts), "curva": curve},
              open(OUT, "w"), ensure_ascii=False, indent=1)

# comparação direta com M120 (int4g K=1 = robustez do solo)
i4k1 = curve.get("1", {}).get("int4g", "?")
log(f"=== SOLO Qwen3-4B (hidden {trunk.H}): int4g K=1 = {i4k1}/{len(facts)} ===")
log("(M120: Phi-3072 K1=36/40, SmolLM-2048 5/40, Math-1536 1/40 — Qwen3 deve ficar entre/acima)")
log(f"DONE M126 ({time.time()-t0:.0f}s)")
