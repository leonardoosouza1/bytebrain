# Edit by address: fix only the region that fails

Loop: failing tests -> sandbox worker returns traceback lines (or innermost `return` line on a
wrong value) -> vote over failing calls picks one line -> `locate` turns it into an address ->
the model writes only that region -> `apply_patch` splices -> judge re-runs the function + the other 38.

## Localisation (calc_generated.py, 4 failing functions)
- Vote hits the right function 4/4 (nth_root L125, factorial L267, gcd L523, evaluate L622; 28/30 failing calls agree).
- A correct fix fits inside the located region in 11/12 (scheme x function), checked by apply + re-judge.
- Miss: evaluate single line 622 (`span_deep`). `evaluate('3 4')` never executes that line; widen to the
  enclosing statement or to the end of the function.

## Address to store
`function name + statement range` (e.g. `gcd`, statements 1-2): survives edits elsewhere in the file.
(line, col) only as display coordinates (shift after each edit). AST paths: exact but shift on earlier inserts.

## Output tokens the model must write (file = 3999 tokens; `oracle_results.json`)
| function | whole file | function | statement span | deep span |
|---|---|---|---|---|
| nth_root | 4000 | 195 | 30 | 30 |
| factorial | 4017 | 169 | 44 | 37 |
| gcd | 4000 | 86 | 24 | 5 |
| evaluate | 4353 | 872 | 699 | miss |

Prompt: ~4150 tokens (whole file) -> 229-737 (function / span). Small bugs: 50-800x less output.
evaluate is a real rewrite; little saving below whole function.

## Latency (median of 300, `measure.json`)
Full re-parse with cached tree ~1.5 ms; re-parse only the enclosing function 153-214 us (1.1 ms for
evaluate); broken patch rejected in 145 us; same accept/reject as full in 300 random cases.
Dominant cost is the sandbox re-judge: 24 ms one function, 124 ms all 39.

## Verified (re-run by main session 2026-10-21)
- `oracle_results.json`: 19/19 known-good answers fix their function, 0 regressions (evaluate span_deep = the miss).
- `test_addressing.py`: 5/5 OK.

## Not known yet
- Whether Qwen2.5-Coder-0.5B writes a correct region; `run_patch_model.py` model path compiled, never run.
- LEAK (partly fixed 2026-10-21): prompts now show failing calls from seed 2, final judge uses seeds 0,1.
  Fixed edge cases are shared across seeds, so this is a "bug report" scenario (user reports failing
  calls), not blind repair. Only nth_root fails a visible example.

## Run the model (~2 GB VRAM)
```
HSA_OVERRIDE_GFX_VERSION=10.3.0 HF_HOME=/media/llm-data/models/hf HF_HUB_OFFLINE=1 nice -n 5 \
  /home/leonardo/projects/LLM/.venv-rocm/bin/python -B run_patch_model.py --k 8 \
  --out /media/llm-data/universe-runs/hybrid-ideas-20261021/edit_by_address
```

## MEASURED 2026-10-22: model in the loop
- Instruction prompt (`run_patch_model.py`): 0/144, the model echoes the block unchanged.
- FIM (`fim_repair.py`, prefix/suffix = code around the region, `cut_to_region`), greedy + 8 samples, fixed with 0 regressions:

| function | span_stmt hint / nohint | span_suffix hint / nohint | span_deep hint / nohint |
|---|---|---|---|
| nth_root | 0/9 / 0/9 | 0/9 / 0/9 | 0/9 / 0/9 |
| factorial | 0/9 / 0/9 | 1/9 / 1/9 | 0/9 / 0/9 |
| gcd | 5/9 / 0/9 | 2/9 / 0/9 | 0/9 / 0/9 |
| evaluate | 0/9 / 0/9 | 0/9 / 0/9 | 0/9 / 0/9 |

Greedy never fixes; sampling + re-judge does. The failing-call comment is what makes gcd work (5/9 vs 0/9).
Raw: `/media/llm-data/universe-runs/hybrid-ideas-20261021/edit_by_address/fim/fim_results_s0_k8.json`. Max 154 W, junction 95 C.

## MEASURED 2026-10-22: FIM on llama-server q8_0 (native HIP), k=32 (`fim_server.py`)
Fixed with 0 regressions out of 33 (greedy + 32 samples). Raw: `/media/llm-data/universe-runs/hybrid-ideas-20261021/edit_by_address/fim_server/fim_server_k32.json`.

| function | span_stmt hint / nohint | span_suffix hint / nohint | span_deep hint / nohint | gen per arm |
|---|---|---|---|---|
| nth_root | 0 / 0 | **2** / 0 | 0 / 0 | ~3 s |
| factorial | 0 / 1 | 3 / **5** | 0 / 0 | 3-9 s |
| gcd | 7 / 2 | **9** / 2 | 0 / 0 | ~2.5 s |
| evaluate | 0 / 0 | 0 / 0 | 0 / 0 | 40-66 s (long outputs) |

- 3 of 4 failing functions now have at least one fixing answer (nth_root never fixed before). evaluate needs a rewrite.
- Rate dropped: gcd span_stmt hint 7/33 (21%) vs 5/9 on bf16 PyTorch k=8. q8_0 or sampling path; not separated.
- CAVEAT: "fixed" is the hidden judge (seeds 0,1) = oracle pick. A real loop must pick by the shown failing
  calls (seed 2) + no regressions, then confirm hidden. That selection is not measured yet.

## Realistic pick (2026-10-22, `select_visible.py`, no GPU)
Pick = answer passes the target's seed-2 cases (the calls shown to the user) and no other function regresses
on seed 2; then confirmed on hidden seeds 0,1. Raw: `.../fim_server/select_visible.json`.
- Precision visible -> hidden: 31/31.
- One pick per function (consensus among eligible answers in the arm with most eligible), applied to the file:
  calculator hidden 35/39 -> **38/39** (nth_root, factorial, gcd fixed; evaluate still fails).
- evaluate is wrong at the root (a `split()`-based RPN stack): needs whole-function regeneration (`whole_function.py`).
