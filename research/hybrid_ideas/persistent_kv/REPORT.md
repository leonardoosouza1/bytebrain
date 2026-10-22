# Persistent KV cache as "context saved instead of re-read"

Design note + benchmark script. All timings here are CALCULATED from the model config with assumed
bandwidth/FLOPS; `kv_bench.py` measures them on the GPU (pending run).

## Sizes (Qwen2.5-Coder-0.5B: 24 layers, 2 KV heads, head dim 64, bf16)
KV per token = 24 x 2 x 64 x 2 (K,V) x 2 B = 12 KiB. Calculator file = 3,999 tokens.

| Context | KV | Reload PCIe 25 GB/s | Reload NVMe 3 GB/s | Recompute @10 TFLOPS |
|---|---|---|---|---|
| 2k | 24 MiB | 1.0 ms | 8.4 ms | 0.17 s |
| 4k (calc file) | 47 MiB | 2.0 ms | 16 ms | 0.36 s |
| 8k | 96 MiB | 4.0 ms | 34 ms | 0.88 s |
| 32k | 384 MiB | 16 ms | 134 ms | 7.0 s |

Simplest: keep the cache in VRAM (47 MiB = 0.4% of the card).

## Edits
Editing a function in the middle invalidates every cached token after it (attention + positions).
Prefill per iteration (39 functions, 1 edited, 16 samples x 256 tokens):

| Scheme | TFLOP | @10 TFLOPS | Exact |
|---|---|---|---|
| today: `num_return_sequences=16` (prompt expanded BEFORE prefill) | 59.3 | 5.9 s | yes |
| prefill once, copy cache to 16 | 3.71 | 0.37 s | yes |
| (a) keep prefix, recompute after edit | 2.09 | 0.21 s | yes |
| (b) recompute only edited block | 0.21 | 0.02 s | no |
| (c) edited part last, file only grows | 0.23 | 0.02 s | yes |
| (d) signatures only | 0.29 | 0.03 s | yes, no bodies |

Recommendation: (c), fallback (a). Generation is per function so the prefix only grows; re-editing
appends a new `def` (last definition wins), with occasional full re-prefill.

## The real bottleneck is decode
Measured from `candidates_s0.json`: ~178 ms per step, flat across batch 1 and 16. A 0.5B model should
need ~3 ms/step, so this is overhead, not compute. Suspects (not yet separated): Python/`generate`
overhead, `stop_strings` check, the CU mask / other GPU job at the time. `kv_bench.py` phase iii separates them.

## kv_bench.py
- i: prefill file, time clone / copy x16 / RAM->VRAM reload
- ii: 16x256 tokens with shared cache vs 16 re-prefills, logits equality check
- iii: decode batch 1/16, with/without `stop_strings`, bare forward loop
- iv: edit `sqrt` (107 -> 142 tokens); KL and top-1 of exact / stale / (a) / (b)
Peak VRAM ~2.7 GB, hard 4 GB cap. Output: `/media/llm-data/universe-runs/hybrid-ideas-20261021/persistent_kv/`.

Risks: (b) quality loss unknown (phase iv); slow attention kernel fallback on gfx1032; tied to transformers 5.12 cache API.

## MEASURED 2026-10-22 (`kv_bench.py --samples 4`, 40 CUs, -25 mV/2400 MHz, idle GPU; max 170 W / junction 84 C)
JSON: `/media/llm-data/universe-runs/hybrid-ideas-20261021/persistent_kv/kv_bench_20261022_000242.json`
(first run with 16 samples hit the 4 GB cap in phase ii: SDPA allocated 390 MiB chunks -> materialized attention)

| What | Measured |
|---|---|
| prefill 4035 tokens | 1.53 s = 2.34 TFLOPS effective (assumed 10) |
| KV size | 47.3 MiB (12,288 B/token, as calculated) |
| clone in VRAM / expand x16 / RAM->VRAM | 3.9 ms / 18 ms / 9.6 ms |
| decode batch 1, `generate` | 81 ms/step (12.3 tok/s) |
| decode batch 1, bare forward loop | 81 ms/step -> NOT `generate` overhead |
| decode batch 16 | 97 ms/step (165 tok/s aggregate) -> launch/latency bound, not compute |
| `stop_strings` | +10% per step |
| 4 samples x 256 tokens: reuse cache vs re-prefill | 28.8 s vs 35.0 s (1.21x; decode dominates) |

Edit probe (edit `sqrt`, 107 -> 142 tokens, 3532 tokens after it; 48 reference tokens):

| Variant | KL mean | top-1 | build time |
|---|---|---|---|
| exact full re-prefill | 0.0002 | 0.979 | 1.56 s |
| stale (no edit) | 0.154 | 0.979 | 0 |
| (a) recompute suffix | 0.0003 | 0.979 | 1.45 s |
| (b) block isolated, rope-shifted | 0.110 | 0.917 | 0.061 s |
| (b) block in context, rope-shifted | 0.0023 | 1.000 | 0.066 s |

Findings: (1) re-prefilling only the edited block in context and rope-shifting the rest is ~24x faster than a
full re-prefill and near exact (KL 0.002). (2) Decode at 81 ms/step for a 0.5B model is per-kernel latency in
PyTorch/ROCm eager, flat with batch: the fix is a different runtime (llama.cpp HIP / own engine / graphs),
not KV tricks. Prediction "bare forward loop < 30 ms/step" was WRONG (81 ms).
