# pycode — own byte-LM for Python, guided by measured "truth" (mirror), not by answers

Started 2026-10-22. Runs: `/media/llm-data/universe-runs/bytebrain-pycode-20261022/`.
Data: `/media/llm-data/datasets/pycode/`. Report trail: `META/handbook-llm/reports/2026-10-21-22-hybrid-codegen.md`,
`META/CLAUDE-UNIVERSE-AMBIENTE-20260925.md` (2026-10-22 pm entry).

Read this file first in a new session. Section 7 says what is running/next.

## 1. Goal (owner's words, condensed)

- A model that is "as intelligent as possible while costing little", learns on demand, light and versatile —
  same idea as Universe: the architecture around the model and its inputs (body + mind) matter.
- Not "more training on gigabytes of random text": ByteBrain already learned to *write words* of PT-BR from
  Wikipedia but without coherence. The problem is not lack of training; the data must be clean and the system
  needs something that says **whether a sentence is true**.
- Step back and isolate: train on **one clean domain** — Python functions only (no comments/docstrings), a
  language with a formal grammar, before natural language.
- Grade the error the way a gradient does: tell the byte model *how much* and *where* it was wrong.
- The corpus (fits in VRAM) is used as **orientation and mirror** — distances, laws, error positions —
  **never as the answer** injected into generation. ("não precisamos dar a resposta… é orientação e espelho")
- Validators should live **on the GPU, inside the per-byte generation loop** (parse/lint ideas, but as GPU
  state machines), not as CPU tools run afterwards.
- Look for **laws** (measured regularities) that give direction, e.g. which token can follow which, what makes
  an indent start or end.

## 2. Principles that came out of this work

1. Clean data first: same model, clean functions vs raw (with docstrings/comments) = +16.4 pp parse.
2. The corpus is a mirror, not a generator: a pure corpus-index generator parses 0-3% at every context length.
3. Every claim gets a prediction before the run and a PASS/FAIL after, on held-out functions (files never trained on).
4. Grade at two levels. Per-sample grades alone (parses, uses its parameters) make self-training collapse onto
   the cheapest template; only population axes (bits/line, I(k), name entropy, distinct lines) reveal it.
5. Final target metric = complete a held-out **signature** (task given, body by the model) and **run** it in bwrap.

## 3. Pipeline and files

| Stage | File | What it does |
|---|---|---|
| data | `build_corpus.py` | stdlib + venv site-packages functions; raw vs clean (`ast.unparse`, no docstrings); holdout by file hash (~3%); `small_*` = short self-contained |
| model | `../../train.py` | ByteGPT 26M (dim 512, L8, H8, ctx 256); `--mem-frac`; resumes from `ckpt.pt` |
| judge | `eval_samples.py` | parses / names_ok (free names ⊆ builtins) / runs (bwrap, fuzz args) for free generation and held-out signature completion |
| corpus index | `corpus_index.py` | GPU suffix array (prefix doubling), exact counts, batched next-byte histograms; cached `.sa64.npy` |
| mirror | `mirror.py` | per-axis distance of model samples to unseen real code: match-length spectrum, I(k) over byte classes, bigram KL, names, syntax |
| GPU checker L1 | `gpu_syntax.py` | per-byte batched automaton: lexer DFA [19×256], bracket stack, indent-width stack, expect-indent after ':'; `allowed()` mask, `complete()` |
| GPU checker L2 | `token_pairs.py` | byte-level tokenizer on top of L1; legality table of adjacent token classes **counted on the corpus** |
| laws | `grid_laws.py`, `line_laws.py` | code as line×column matrix: what predicts indent change; x0×row; SVD (".dna two matrices"); line-type transition matrix T |
| comparison | `ab_table.py` | one table: mirror + line law + GPU checker on the same samples |
| self-training | `self_round.py` | generate → grade (sample + population caps, optional signature prompts and bwrap run check) → mix with clean replay → fine-tune |
| index-only probes | `igram_probe.py`, `mix_eval.py` | corpus index as generator / mixed into the model (superseded: that was "giving the answer") |

Dependencies outside this folder (untracked, older sessions): `research/fast_gen.py` (CachedGen),
`research/battery_knowing.py` (ByteLM).

Typical commands (always `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `nice -n 5`, python from `.venv-rocm`):
```
python build_corpus.py
python ../../train.py --corpus $D/clean_train.txt --ckpt-dir $R/clean --max-steps 6000 --decay-steps 6000 \
    --warmup 300 --val-every 500 --batch 16 --accum 3 --amp --mem-frac 0.18 --seed 0
python mirror.py --ckpt $R/clean/ckpt.pt --mem-frac 0.45 --out $R/mirror_clean.json   # --real-only = noise floor
python eval_samples.py $R/clean/ckpt.pt --n 128 --mem-frac 0.2 --out $R/eval_clean.json
python token_pairs.py --build --check
python self_round.py --ckpt CKPT --out-dir $R/selfN --cap-per-name 8 --cap-per-line 4 [--sig-prompts --run-check]
python ab_table.py $R real clean self2 self3
```

## 4. Results (all on held-out functions)

Training arms, 6000 steps, free generation (n=256):

| arm | parse | params used | bits/line | match_tv | len |
|---|---|---|---|---|---|
| real code | 100% | 0.91 | 3.61 | 0.04 (noise floor) | 250 |
| clean | 71.1% | 0.57 | 3.25 | 0.12 | 408 |
| raw | 54.7% | 0.68 | 5.71 | 0.24 | 379 |
| small (4× fewer params) | 48.0% | 0.62 | 4.70 | 0.25 | 175 |

Self-training rounds from clean (2048 samples graded per round, fine-tune 1000 steps on graded ×4 + 4 MB clean replay):

| round | grade | free parse | params | bits/line | % `__init__` | completion parse / names_ok / **runs** |
|---|---|---|---|---|---|---|
| clean | — | 71.1% | 0.57 | 3.25 | 32% | 78.1 / 32.0 / **23.4%** |
| self1 | parses + all params used + ends + ≤700 B + not a copy | 79.7% | 0.79 | 2.81 | 41% | — |
| self2 | + no repeated line | 85.5% | 0.79 | 2.85 | **43%** | 86.7 / 46.1 / 33.6% |
| self3 | + population caps (8 per name, 4 per line >12 chars) | 81.6% | 0.72 | 3.22 | 28% | 82.8 / 47.7 / **35.2%** |

Free-generation "runs": clean 8.6% → self2 14.8% → self3 34.4%.

GPU checker (1000 real held-out + 1000 with one random corruption):
- L1 automaton: 0/1000 false positives; catches 79.0% of what `ast.parse` rejects. 7.7 µs per seq-byte (launch-bound).
- L1 + L2 corpus token-pair law: 83.8%, still 0 FP on real. Misses: inserted ':' (pair legal, needs context),
  keyword typo `ef`/`lse` (`NAME NAME` occurs 16× in 3.9 M pairs, so it counts as legal), deleted ',' / '='.

Laws measured on 28.5k held-out lines:
- Next indent change has 1.65 bits; first word gives 1.43, last char 0.98, x0 0.74, first word + level 1.55 (94%).
- Line ending ':' → next indents (100%). Line starting return/raise/continue/break/pass → next dedents (100%).
- x0 (first char) informs bytes 1-4 of its line (1.1-1.9 bits), fades by column ~8, floor by ~24.
- Grid SVD: real vs bytes shuffled inside each line differ by only 5-7 pp of reconstruction → grammar is not
  low-rank in a line×column byte matrix. The useful small matrix is the **line-type transition matrix**
  (2353 types): real 3.84 bits/line, 0.8% unseen transitions; same lines shuffled 7.08 bits, 28.8% unseen.

Corpus index alone as generator (contexts capped at 8/16/32/48 bytes): parse 3.1/1.6/0/0%, novel 100%.

## 5. Errors and traps paid in this work

- `bytes(row)` on an int64 numpy row returns the raw buffer → all index-probe samples garbage. Cast to uint8.
- Splitting the corpus on `\n\n` cuts functions at blank lines inside strings (2087/4096 chunks broken). Split on `\n\n(?=def )`.
- Indent rule "multiple of 4" gave false positives; Python keeps a stack of open widths — so does L1 now.
- A 293-byte prompt exceeded ctx 256 → position-table out-of-range → `HSA_STATUS_ERROR_EXCEPTION`, core dump. Prompts keep their tail (`ctx//2`).
- Two GPU jobs at once → "Memory access fault" in one and a hung process spinning (99% "use" at 42 °C) in the other; 41 min lost waiting. Run GPU jobs sequentially, with `timeout`, and check progress, not just liveness.
- Logs piped through `grep` to a file are block-buffered: they look empty until the end. Write logs with `python -u > log`.
- `eval_samples.py` default `--mem-frac 0.06` is too small for the 26M model (OOM) — pass 0.2.
- Per-arm val_bpb comes from each corpus's own tail → not comparable across arms; compare with the same judges.
- Self-training val_bpb is measured on the mix tail (contains own samples) → not a quality signal.

## 6. Open questions

- Correctness vs diversity frontier: population caps cost ~4 pp parse but doubled "runs". Can the proposal be
  diversified (signature prompts) instead of filtered?
- Use the L1/L2 checker inside generation: `allowed()` mask, or the error position as a localized training
  signal (unlikelihood on the first illegal byte).
- Fuse the checker into one HIP kernel / graph (target: ns per seq-byte instead of µs).
- Line-type matrix T as a soft grade inside self-training (bits/line toward real 3.6).
- Token-pair law with a count threshold (e.g. `NAME NAME` < 20 → illegal) to catch keyword typos.

## 7. Status / next (update this section every session)

- Round 4 (from self3; proposal = training signatures ×16 attempts, bwrap run check, population caps):
  REGRESSED. Completion parse/names_ok/runs 78.9/38.3/26.6% (self3: 82.8/47.7/35.2); free runs 21.1% (self3 34.4%).
  Mirror axes best so far (bits/line 3.36, I(k32) 2.3×, bigram KL 0.17). 727/2048 rejected for unused params:
  real signatures carry many params the model cannot use yet. Prediction (runs ≥42%) FAILED.
- Why round 4 regressed (counted on accepted `good.txt`): only 128 signatures x16 attempts -> **98 distinct
  def names** (self3: 475), self-contained 39% (self3 50%). Task diversity, not body diversity, was lost.
- Control round 4b (round 4 recipe, free `def ` prompts): completion 79.7/45.3/32.0%, free runs 28.1%.
  Prediction (4b >= self3) FAILED; 4b vs self3 within noise (n=128, SE ~4 pp). Self-training **plateaued at
  round 3**; another filter-and-retrain round does not climb. Best checkpoint: self3.
- L1 automaton as byte mask inside generation (`eval_samples.py --mask`, self3, n=256): free parse 81.6 -> 83.6%,
  completion runs 34.0 -> 34.0%. Prediction (+8 pp parse) FAILED. Remaining failures are not syntax: of 256
  completions 94 use free names, 39 crash, 17 do not terminate, ~12 are syntax errors.
- **Metric correction:** 240/256 completion targets are not self-contained even in the original, so "runs" rewarded
  inventing a trivially executable body, not doing the task. New judge `equiv.py`: real self-contained function vs
  model body on 40 deterministic fuzz inputs in bwrap, graded score = share of equal outputs (+ in-place effects).
- Equivalence (28 held-out refs, k=8): clean/self3/self4b/self3+mask mean score 0.7-1.1%, pass@8 3.6%; floor
  `return arg0` 9.4% / pass 7.1%. On 200 TRAINING functions: self3 1.4%, pass@8 5% (floor 17.8% / 12%).
  => the 26M model learned FORM (grammar, indentation, style), zero MEANING; it does not even memorize.
- **synth/** (meaning in isolation): `synth/synth.py`. Names compose meaning (`double_then_keep_even_then_sum`),
  13 list primitives x 7 finals, truth = output equality on 30 fixed int lists in bwrap. Train = 1- and 80% of
  2-primitive names; test_a = unseen pairs; test_b = 3 primitives (unseen length); `self` pool = other 3-primitive names.
  3M model (dim 256, L4), runs in `$R/synth/`.

  | ckpt | test_a pass@1 | test_b pass@1 | test_b pass@8 |
  |---|---|---|---|
  | base (3000 steps) | 47.3 | 14.1 | 35.0 |
  | self1, fine-tune at LR floor | 42.9 | 16.0 | 35.3 |
  | self1, real LR (+3000 steps) | 69.3 | 25.2 | 38.0 |
  | ctrl: +3000 steps train.txt only | **76.5** | 24.7 | 28.3 |

  Lessons: (1) fine-tune at the LR floor does not even fix the trained names (pass@1 on kept names 40.6 -> 42.9%)
  — likely also limited the pycode rounds. (2) Self-training gain was just extra training (control).
  (3) Mechanism: 92.5% of kept bodies SKIP a primitive — the base writes 2 lines (never saw 3) and passes only where a
  primitive is redundant on the inputs (`abs_then_abs`), so execution truth rewarded the shortcut; "shortest exact"
  selection made it worse.
- 2026-10-22 ~19:45 running `run_self2.sh`: selection = exact AND structural law measured on train (one `xs =` line
  per primitive in the name), k=32 on 4000 names, from ctrl; vs ctrl2 (same steps, train only).
  Prediction: self2 test_b pass@1 >= ctrl2 + 10 pp.
