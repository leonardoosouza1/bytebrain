# Fast memory bank: ns-level symbol lookup for the generation loop

Bank over 7 files (calc_ref, calc_generated, 5 stdlib modules): 2123 symbols, 6266 uses,
145 KB total (68 B/symbol). Raw outputs in `bank/` (`rust_results.txt`, `py_results.txt`,
`token_cost.txt`, `fit_stats.txt`). Numbers below were cross-checked against those files.

## Result
| Query | Native Rust | Python (best) |
|---|---|---|
| name -> id | 7.1 ns (FxHashMap) | 27 ns (dict) |
| resolve(scope, name) -> sym | 2.8 ns | - |
| id -> signature | 1.1 ns | 23 ns (list) |
| uses(sym) | 0.8 ns slice / 3.5 ns decoded | 23 ns |
| live locals at line L | 32 ns scope index (1.2 ns per-line CSR, 112 KB) | 640 ns |
| (scope, 4-bit slot) -> sym | 1.2 ns | - |
| nibble stream | 0.74 ns/token scalar, 0.07 ns/token SWAR | 31-38 ns/token |
| name -> id, 1M names | 30 ns | - |

- Python call overhead dominates: numpy (~200 ns/call) and ctypes (126 ns/call) lose to dict/list
  for single lookups; numpy wins only in batches (41 ns/query at 4096).
- 4-bit pointers: 348/357 functions (97.5%) have <=15 symbols, so a nibble slot inside the current
  scope fits with 0xF as escape. They do not work as file/function ids (16 files cap).
- Token alternative: mean function = 143 Qwen tokens, signature = 6-13. Tokenizer alone 1.2-1.9 us/token.
  Re-reading one function as prompt ~1.4e11 FLOPs (~18 ms on GPU, inferred, not measured).

## Recommendations
1. Bank as plain Python dict/list now; flat binary layout as shared format; native code only in batches.
2. "What is live here" via scope index. Give the model signatures, not bodies, or a logit mask of allowed names.
3. Salience model outputs a 4-bit slot in the current scope (0xF = hash fallback); bank returns the exact symbol.

## Limits (inferred, not tested)
- Resolver ignores `global`/`nonlocal`, attributes (`self.x`, `math.sin`), comprehension scopes.
- 1M-name result uses synthetic names.
- Runtime values from test runs are not stored yet; only literal constants.

## Reproduce (in this folder, `.venv-rocm` python, `taskset -c 3 nice -n 10`)
- `build_bank.py bank <files>`
- `rustc --edition 2021 -O -C target-cpu=native bench.rs -o bench && ./bench bank/bank.bin`
- `rustc --edition 2021 -O --crate-type cdylib ffi.rs -o libbank_ffi.so && python bench_py.py bank ./libbank_ffi.so`
- `HF_HOME=/media/llm-data/models/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= python token_cost.py bank/token_cost.json <files>`
