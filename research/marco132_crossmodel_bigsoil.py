"""M132 — ARMAZÉM CROSS-MODEL NO SOLO GRANDE (combina M117+M126): destila o conhecimento de OUTROS
modelos (Phi, SmolLM) dentro do Qwen3-4B congelado (o solo mais barato) e mede quanto absorve e a que
custo. Junta as duas descobertas: herança cross-model + solo grande = armazém barato. Usa teacher_cache.json.
GPU. Dump marco132_metrics.json."""
import json, time, os
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
CACHE = "/home/leonardo/projects/LLM/bytebrain/research/teacher_cache.json"
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco132_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M132 já feito"); raise SystemExit
cache = json.load(open(CACHE)); Q = S.QUESTIONS
log("M132 START | carregando Qwen3-4B como TRONCO (solo grande)")
trunk = S.Trunk(f"{MODELS}/qwen3-4b-q4km.gguf", probe_every=6)
S.CH = 12  # chunk pequeno: limita memória de ativação do backprop no 4B (12GB VRAM)
log(f"tronco Qwen3-4B hidden {trunk.H} | chunk {S.CH}")

# conhecimento EXTERNO (Phi + SmolLM), fatos curtos que o Qwen3-tronco NÃO já sabe
ext = []
for tea in ["phi4mini", "smollm2"]:
    for k in range(len(Q)):
        t = " " + cache[tea][k].lstrip()
        if 1 <= len(trunk.tok(t, add_special_tokens=False).input_ids) <= 4:
            ext.append((f"Pergunta: {Q[k]}\nResposta:", t))
zero = torch.zeros(1, trunk.H, device="cuda", dtype=torch.float16)
unknown = [(p, t) for p, t in ext if trunk.recall(zero, [(p, t)]) == 0]
# dedup por prompt
seen = set(); unknown = [(p, t) for p, t in unknown if not (p in seen or seen.add(p))]
log(f"{len(ext)} fatos externos | {len(unknown)} que o Qwen3 não sabia (dedup)")

res = {"hidden": trunk.H, "n_externos": len(ext), "n_desconhecidos": len(unknown), "curva": {}}
for K in [1, 4, 16]:
    try:
        seed = trunk.plant(unknown, K=K, steps=500)
        f16 = trunk.recall(seed, unknown); i4 = trunk.recall(S.quant(seed, 4, 128), unknown); i8 = trunk.recall(S.quant(seed, 8, 128), unknown)
    except RuntimeError as e:
        log(f"  K={K} OOM: {str(e)[:40]}"); torch.cuda.empty_cache(); continue
    kb = K * trunk.H * 4 / 8 / 1024
    res["curva"][str(K)] = {"fp16": f16, "int4g": i4, "int8g": i8, "de": len(unknown),
                            "bytes_fato_i4g": round(kb * 1024 / max(1, i4), 1)}
    log(f"  K={K}: absorveu fp16 {f16} int4g {i4} int8g {i8} /{len(unknown)} | {res['curva'][str(K)]['bytes_fato_i4g']}B/fato i4g")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== ARMAZÉM: Qwen3-4B herdou conhecimento de Phi+SmolLM; melhor int4g K=1 = {res['curva'].get('1',{}).get('int4g','?')}/{len(unknown)} ===")
log(f"DONE M132 ({time.time()-t0:.0f}s)")
