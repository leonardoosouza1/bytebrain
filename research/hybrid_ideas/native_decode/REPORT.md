# Native decode: llama.cpp HIP vs PyTorch eager (Qwen2.5-Coder-0.5B-Instruct)

2026-10-22. GGUF converted from the local HF snapshot with `llama.cpp-turboquant-hip/convert_hf_to_gguf.py`:
`/media/llm-data/models/gguf/qwen2.5-coder-0.5b-instruct-{bf16,q8_0}.gguf`.
Bench binary: `llm-lab/vendor/llama-cpp/lib/llama-bench` (build 0893f50f2), `-ngl 99 -r 3`.
Memory clock was stuck at 456 of 1124 MHz during this run (`clk.log`), sclk 2400, 60-84 W.
Raw: `/media/llm-data/universe-runs/hybrid-ideas-20261021/native_decode/bench_tg.txt`.

| | PyTorch eager (persistent_kv) | llama.cpp bf16 | llama.cpp q8_0 |
|---|---|---|---|
| decode, batch 1 | 81 ms/token (12.3 tok/s) | 8.0 ms (124 tok/s) | 5.8 ms (171 tok/s) |
| prefill 4096 | 1.53 s | 1.22 s (3358 tok/s) | 0.78 s (5261 tok/s) |
| prefill 512 | - | 4695 tok/s | 9863 tok/s |

- Decode 10x (bf16) / 14x (q8_0) faster: the 81 ms was PyTorch/ROCm per-kernel latency, as inferred.
- Prefill only 1.25x (bf16): prediction ">=5x" FAILED; PyTorch prefill was already compute-bound.
- bf16 decode reads ~1 GB of weights per token: 124 tok/s = ~124 GB/s, ~70% of what the stuck 456 MHz
  memory clock allows (see "Memory clock lock found" below).

## Not measured yet
- Batched decode (k samples per step, the generator's real case). `batched_decode.py` is written, but the
  pip `llama_cpp` 0.3.32 HIP lib segfaults on the first decode, even through the high-level `Llama` API
  (not this script). Needs a HIP build for gfx1030 of `llama-batched-bench` / `llama-server`
  (`llama.cpp-turboquant-hip`, CPU-heavy build: run when the machine is free).

## Memory clock lock found (2026-10-22)
Memory stayed at 456 MHz under load in every mode (mask, pin level 3, COMPUTE profile, auto, OD reset to stock).
Measured read bandwidth 161-168 GB/s. Cause: monitor at 200 Hz (DisplayPort-2, 2560x1080). At 144 Hz:
mclk 1124 MHz, read bandwidth 418.9 GB/s. Bench at 144 Hz, stock OD (`bench_tg_144hz.txt`, max 166 W / junction 80 C):

| | mclk 456 (200 Hz) | mclk 1124 (144 Hz) |
|---|---|---|
| decode bf16 | 124 tok/s | 179 tok/s (5.6 ms) |
| decode q8_0 | 171 tok/s | 261 tok/s (3.8 ms) |
| prefill 4096 q8_0 | 5261 tok/s | 6205 tok/s |

Prediction ">=250 bf16 / >=300 q8" FAILED: 1.5x, not 2.5x. bf16 decode now uses 179 of 419 GB/s, so batch-1
decode is back to kernel latency; the next lever is batched decode (k samples per step).

## Batched decode (2026-10-22, monitor 120 Hz, -35 mV/2400 MHz, mclk 1124)
HIP build for gfx1030 of `llama.cpp-turboquant-hip` in `/media/llm-data/scratch/llama-hip-build/bin`.
`llama-batched-bench -ngl 99 -fa on -npp 4000 -ntg 64 -npl 1,4,8,16,32,64 -pps` (prompt prefilled once, shared).
Raw: `batched_bf16.txt`, `batched_q8_0.txt`; max 156 W / junction 74 C.

| k | bf16 ms/step | bf16 tok/s | q8_0 ms/step | q8_0 tok/s | PyTorch eager ms/step |
|---|---|---|---|---|---|
| 1 | 5.7 | 175 | 4.0 | 249 | 81 |
| 4 | 10.6 | 377 | 5.9 | 681 | - |
| 8 | 16.4 | 487 | 7.9 | 1014 | - |
| 16 | 78.3 | 204 | 12.5 | 1280 | 97 |
| 32 | 84.0 | 381 | 19.2 | 1664 | - |
| 64 | 95.3 | 671 | 32.8 | 1951 | - |

Shared 4000-token prefill: 0.43 s q8_0, 0.87 s bf16.
- bf16 falls off a cliff at k>=16 (78 ms/step): RDNA2 has no bf16 matrix path, the batched matmul leaves the fast kernel. Use q8_0.
- Predictions (k=16 <=10 ms, >=1600 tok/s; k=64 <=25 ms) FAILED: 12.5 ms / 1280 tok/s; 32.8 ms.
- q8_0 k=16 vs PyTorch k=16: 7.8x per step. One function, 8 samples x 150 tokens: ~1.2 s vs ~14.6 s (inferred).
- Not checked yet: whether q8_0 answers pass the judge as often as bf16.
