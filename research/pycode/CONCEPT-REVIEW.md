# Concept review — what works, why, and what to change (ledger #43, 2026-10-26)

His request (14:30): "faça uma análise mais profunda do que está dando certo, reveja os conceitos e ajuste".

All jobs were stopped at 14:32 at his request. Nothing in this file was produced by a new training or GPU run.

The numbers come from artefacts that were already written:
- `$R/e2e/v2pool/traces_student_s5.jsonl`: student s5 on 593 Sonnet rewrites, finished 14:31;
- the results tables in README §7, `PIPELINE-VALIDATION.md` and `BYTE-MATRIX.md`.

The analysis scripts are inline Python reads of those JSON files (seconds, CPU).

## 1. The one principle behind every result that worked

Every success of the line gives the model three things:

1. **an address** — a state that says *where* it is: the plan slot, the pointer, the call-tree position;
2. **a truth check at each step**, not only at the end;
3. **training that copies its own verified outputs.**

| result | address | per-step truth | copy what is right | number |
|---|---|---|---|---|
| address planner vs plain byte seq2seq (same world) | yes / no | — | — | len6 100% vs 2% |
| pointer (done name segments upper-cased) vs plain | yes / no | yes | yes | unseen len4 67.7 vs 33.0 |
| loc_ce: on-policy rounds, every line checked | — | yes | yes | test_b 25.7 → 99.8 |
| same, final-output truth only | — | no | yes | 88.75 (−11 pp) |
| same, one round instead of six | — | yes | once | 57.6 (−42 pp) |
| call trees (compose) trained on 1–3 calls | yes | yes | — | 4–6 calls 99–100%, 8 calls 93% |
| belt blame = earliest broken stage | stage address | yes | — | Sonnet judge = oracle 60/60 |

Every failure lacked one of the three:

| result | what was missing | number |
|---|---|---|
| free byte-LM on real Python (26M) | address and per-step truth | form without meaning, equivalence 0.7–1.1% |
| unlikelihood on the first wrong line (push away the error) | a positive target | −26 pp vs plain CE on verified bodies |
| repair from the error line | nothing new to copy (the model either knows the body or not) | = resampling at equal budget |
| derivative from a random start (gradsearch v1) | the guess (a start near the answer) | loses to random search, len2 13% vs 40% hidden |
| SVD / learned factorization of byte *values* | content is symbols, not magnitudes | never < 8 bits/byte at ≥ 90% |

**Consequence for his ideas:**
- "Error as teacher" (#7, #35) works as a **location** and as a **filter** that keeps what is right. It does not
  work as a push-away gradient.
- "The derivative points the direction" (#26) holds when the target is the correct thing: CE toward a verified or
  fixed answer, or derive from a good guess. It fails from nothing.
- This is why the judges' useful output is the **fix**, a positive label, and not the blame. Code already computes
  the blame from the trace.

## 2. New finding today: the understander fails on the *form* of the spec, not on the meaning

Setup:
- Student s5: Qwen2.5-Coder-0.5B + LoRA, trained only on templated texts with gold labels.
- Test set: the 593 Sonnet rewrites of its own training requests (v2pool).
- Belt pass: **441/593 = 74.4%**.

| world | pass |
|---|---|
| calculator (dispatch) | 138/139 = 99.3% |
| formula (expr) | 303/454 = 66.7% |
| formula by size: 2 / 3 / 4 / 5 ops | 42/48, 211/309, 21/45, 28/50 |

Planner and writer never failed. Root = understander for 152/152 failures.

The 151 formula failures, classified from the raw student output:

| shape of the wrong output | n | exactly the gold's set of operations |
|---|---|---|
| arity: siblings instead of nesting, `mul(pct(b, a), mod(pct(b, a), a), b)` | 42 | 30 |
| step list in reading order, `maxx(a, b), sub(b, a), mod(sub(b, a), a)` | 32 | 24 |
| answered as a calculator: the op list in reading order, `["div", "mul", "pct"]` | 18 | 11 |
| infix, `mod(a, b) + div(a, b)` | 17 | (operators written as symbols, not counted) |
| parseable but a different tree | 41 | 20 |
| other | 1 | 0 |
| **total** | 151 | **87 (58%)** |

**Mechanism.**
- People write a formula as a **sequence**: "pega b por cento de a, calcula o resto disso por a e multiplica por b".
- The spec language wants a **nested tree written outside-in**: `mul(mod(pct(b, a), a), b)`. The last thing said
  is the outermost call.
- The training texts never needed that reversal. `verbalize()` in `e2e_pipeline.py` writes the nesting with
  parentheses inside the Portuguese ("a média de (o maior entre b e a) e b"), so the student learned to copy
  parentheses.
- Natural requests have no parentheses. The student then writes the steps in reading order:
  - as a comma list;
  - as extra arguments;
  - or in the only sequential container the spec offers, the calculator list.
- In 58% of the formula failures it had exactly the right operations.

This is the address principle again. Building the tree needs a place to hold intermediate results, and the spec
language gives the understander none.

(s5p, trained with 593 rewrites, raised set-B formulas from 56/80 to 74/80. Data can teach the reversal, but it
costs Sonnet calls and it brought the "main ⇒ formula" shortcut.)

## 2b. Independent audit (Opus agent, read-only) — only the points I re-checked myself

| claim | my check | verdict |
|---|---|---|
| The belt writer is a learned lookup. `multifn.Writer` is conditioned on `nn.Embedding(len(SYM))`, so a GRU memorizes about 30 fixed strings. The address-guided byte writer of #17 (`bytewriter.py`, test_num 0.86) is a **different** writer and is not in the belt. | `multifn.py:208-213`; `e2e_pipeline.py` loads `MF.Writer` | TRUE. "The writer generates code" in the belt means "replays a memorized function body". |
| The spec is the program. The expr string becomes main's return line, and planner and writer expand it by fixed rules. So belt pass ≈ understander accuracy. | p_ok = w_ok = 1.0 in every summary; `plan_of` is a few lines | TRUE. The belt is a compiler with a learned front end. |
| **V2 as designed tests nothing new.** Fix acceptance (`verify_fix`) needs `score_after > score_before` (score computed against gold) and `stage == oracle root`. Triage uses `end_pass` (gold). V2 also runs on exactly the 593 texts that s5p trained on. | `git show HEAD:…/e2e_pipeline.py` lines 1076-1080; `replay_labels.py:46`; smoke labels = gold 161/161 | TRUE. V2 ≈ s5p with gold labels and some items dropped. Cancel it as designed (§4). |
| The Qwen-3B judge is below a constant answer. pass_agree 15/30, while "always fail" gets 20/30. It called a failure all-correct on 14/20. | `judge_nat.jsonl`, counted | TRUE. Today's two V2 rows repeat it. |
| The τ-bank (#8) is a counter: `a = sigmoid(gate).cumsum()`. Infinite τ, changes only on events. It holds the **address**, not data. | `blockaddr.py:74-75` | TRUE. #8 goes back to PARTIAL. |
| Set B is now a dev set. Its failures were read to design fixes, and s5p's augmentation came from the same Sonnet prompt (a style match). | `paraphrase_requests.py` GEN used for both | TRUE. We need a blind set C. |
| The student is slower than the 3B: 1.8 s vs 0.87 s. | HF bf16 eager vs llama.cpp q8. The runtimes differ, so this is not like-for-like (#41 GGUF). | TRUE as measured; the cause is the runtime. |

Not accepted as stated:
- "Steps only change the reading order, not who does the work." True for the division of labour.
- But §2 shows reading order is exactly where 58% of the formula failures are. The point of A1 is the error form,
  not the labour split.

## 3. Scoring the system against his goal

His goal: "o mais inteligente possível custando menos, aprende sob demanda, leve e versátil, entradas como corpo
e mente".

| goal | state | evidence |
|---|---|---|
| cheap | planner + writer ~162k params, 0.3 ms + 0.04 ms per request. The understander (494M + 8.8M LoRA) is 99.8% of the latency (1.85 s) and holds 100% of the failures. | `v2pool/summary_student_s5.json` lat_ms |
| intelligent | inside the closed world (15 ops, ints a/b, ≤ 8 calls) the belt reaches 97.8% templated and ~86–89% natural. Outside it there is nothing. | README §7 |
| learns on demand | **not measured.** Adding one operation means retraining the understander, planner and writer. No test counts how many examples or seconds that costs. | — |
| light and versatile | light yes. Versatile no: one closed vocabulary. | — |
| inputs as body and mind | the spec is the only "body" interface, and its shape caused 58% of the remaining errors (§2). | §2 |

**Honest summary.**
- What exists is a compiler with a learned front end: very strong inside its world.
- The intelligence that is still borrowed (reading Portuguese) is where all the errors and all the cost are.
- His "learn on demand" is the part of the goal with no measurement yet.

## 4. Adjustments, in order (cheapest test first)

| # | change | serves | cheapest test | prediction |
|---|---|---|---|---|
| A0 | **Grammar-constrained decoding of the formula, no retraining.** Run the existing s5 LoRA with a mask that allows only BIN ops, 2 args, leaves a/b. Optionally the bridge folds n-ary add/mul. | #4 (his "validation inside the loop"), #34 | Eval only (2 × ~18 min GPU): v2pool and set B with the s5 LoRA, mask on vs off | Parse failures 91 → 0 on v2pool. Pass 74.4 → ≥ 78. The mask fixes the form, not which nesting is chosen. |
| A1 | **Addressed step spec** (built, not trained). `{"main": "expr", "steps": ["s1 = pct(b, a)", "s2 = mod(s1, a)", "s3 = mul(s2, b)"]}`. `parse_steps` names the broken step, and the old expr string stays accepted. | #33 (his JSON hierarchy), #12 (address), #34 | `student_understander.py --arm s5 --seed 1 --steps 523 --spec-format steps`: same data and order, eval on v2pool + B | Form failures (§2 shapes) fall ≥ 60%. v2pool formulas 66.7 → ≥ 80%. Calculators ≥ 98%. |
| A2 | A1 + step-grammar mask (A0's mask on the steps language) | #4 | A1 checkpoint, mask on/off | Shape failures 0. ≥ A0 and ≥ A1. |
| A3 | **Blind set C**: 30–50 requests written by him, or by another model with a different prompt. B becomes a dev set. Report formulas and calculators separately. | #34, #39 (end-to-end validation) | One belt run per understander | s5 on C ≤ B − 5 pp (stated now). |
| A4 | **V2 redesigned gold-free**. Triage by flags + Sonnet verdict. Accept a fix if the code runs with no flags and Sonnet re-passes it. Gold only reports label precision. Never on s5p's texts. | #37 (his loop: judge → retrain) | Label precision vs gold on the v2pool fails, then retrain | Label precision 0.85–0.93. |
| A5 | **3B judge: adapt, not drop** (his design, rule 7). Give it the executed facts per case (outputs, flags) and ask only for the corrected spec. Option: Qwen2.5-Coder-7B Q4 GGUF (~4.7 GB, fits). Code computes pass/fail and blame. | #36 | Re-judge the 30 natural traces | pass_agree ≥ 20/30 (the constant baseline). Fix acceptance > 0/4. |
| A6 | **A writer and planner that can fail, and the learn-on-demand meter.** Condition the writer on intent bytes instead of the symbol id. Hold out 3 ops. Add one new op with N = 8 / 32 / 128 examples and measure examples, steps and seconds to pass it with no forgetting. | his goal "aprende sob demanda", #35, #17 | Writer-only, CPU ~100 s | Held-op w_ok < 0.5 at first: the first natural micro errors, where routing and guess-and-derive can be tested. |
| A7 | **Guess + derive** (#26): planner logits as the start, plus the fixes in GUESS-AND-DERIVE §6 | #24–#26 | `gradsearch.py` len2–4 | guess+derive ≥ guess +10 pp on len4 hidden |
| A8 | s5p2 (calculator rewrites with "main") | #34 | already queued | A ≥ 0.97, held frame ≥ 54/56 |

Not changed, on purpose:
- The judge roles stay Qwen first, Sonnet last, as he designed. A5 adapts the Qwen side.
- No new stage is added before A0/A1 show whether the form, and not the model size, is the bottleneck.
- Cancelled: V2 as designed. It is gold-gated and runs on s5p's own texts (§2b), so A4 replaces it.

## 5. Queue left for when runs are allowed

Nothing is running now. Order, with one GPU slot:
1. A0: s5 LoRA eval with the expr grammar mask, on v2pool and B. Code built (below); review pending.
2. A1 smoke (30 steps), then A1 train + eval. The converter already round-trips 946/946 golds.
   A2 is the same command as A0 on the A1 adapter with `--spec-format steps --grammar-mask`.
3. A8 s5p2.
4. A4 gold-free V2, after A3 exists.

Sonnet calls: only in A4/A5. Never two `claude -p` in parallel. The llama-server must be restarted for any Qwen
judge.

## 6. Built during the stop (code only, no model run) — 2026-10-26

**A1 review (Sonnet agent, findings re-checked by me).** No blocker. Fixed:
- `parse_steps` crashed on `"s1 = add(s², b)"`: the unicode `\w` matched `²`, and `int("²")` raised. A
  5000-digit reference also raised. `STEP_RE` is now ASCII with at most 3 digits, and references are capped at 4
  characters. Both inputs now return `shape:step1` / `ref:step2`.
- A malformed `steps` field (`[]`, `null`, a string) gave the send-back text "step ss is wrong". The step blame is
  now given only when the suffix is a number.
- `canon_spec` credited extra keys. It now converts only `{main, steps}`.
- `to_steps_spec` asserts on a leaf-only formula, which has no steps form.
- `result.json` records `spec_format`.
- The judges' evidence shows a steps spec as the nested expr that RUBRIC.md documents.
- Kept on purpose: the steps regex is strict about spaces, because the training target is exact `json.dumps`.

**A0 grammar mask.**
- `e2e_pipeline.spec_prefix_ok(text, complete, steps)` is a character-level parser for the exact `json.dumps`
  form of the 3 spec languages: dispatch, nested expr and addressed steps. It answers whether the text can still
  become a valid spec.
- `student_understander.py --lora <adapter> --eval-dir <set> --grammar-mask [--spec-format steps]` decodes greedy
  among the top 64 tokens that keep a valid prefix. A complete spec allows only `<|im_end|>`. Output goes to
  `<eval-dir>/<arm>_gm/`, so the plain eval is never overwritten. `result.json` adds `gm_requests_forced`,
  `gm_tokens_forced` and `gm_steps_unmasked`.

Checks (CPU, single thread, < 3 s each; no model loaded):

| check | result |
|---|---|
| 946 golds (A, para, v2pool), each in the expr and the steps format: complete spec accepted | 1892/1892 |
| every proper prefix of those strings accepted | 0 rejected |
| expr spec checked as steps and vice versa | 0 accepted |
| 10 crafted bad specs (arity, infix-like, unknown op, empty calls, unused / self / future / misnumbered step) | 10/10 rejected |
| 600 one-character corruptions: mask verdict = `check_spec` verdict | 600/600 |
| mask with the real Qwen tokenizer, gold token on top: tokens forced | 0, both formats |
| a wrong token (` +-*x")`) put above the gold one 30% of the time: output = gold | 40/40 per format |

**Mask review (second Sonnet agent; each point re-checked by me).** No blocker. Fixed:
- High: the mask tried the top-64 tokens in `topk` order. Among bf16 ties that order differs from `argmax`: 7 of
  200 ties on this CPU. So the masked run could leave plain greedy even where plain greedy was grammatical. The
  argmax is now tried first. Check: with 2 valid tokens tied, the output equals the argmax 100/100, with 0 forced.
- Tokens that decode to nothing (ids above 151665) or only to leading spaces could stall decoding. A token must now
  add visible text. Check: an empty token above the gold one 30% of the time gives the gold output 40/40.
- A miss now escalates from the top 64 to the top 1024. `gm_requests_unmasked` counts the requests the mask could
  not hold; any nonzero value means "not masked" for those requests.
- The steps grammar now caps each expanded tree at 99 tokens, as `parse_steps` does: 8 self-doubling steps are
  rejected from step 6, as in `check_spec`. A prefix with more unused steps than the remaining steps can absorb is
  a dead end and is rejected. The expr depth cap went from 12 to 30.
- The tag is `_gm_st` for the steps grammar. A masked eval asserts that the adapter's `result.json` has the same
  `spec_format`. No empty directory is created in the training dir.
- Declared, not fixed:
  - Masked `ms` includes the mask's CPU time, so it is not comparable with plain `ms`.
  - A forced token changes the context the model conditions on, so "failures fixed by the mask" bounds the form
    failures; it is not an exact count.
  - A lone space token is valid text (a non-canonical BPE split) and is accepted, as plain greedy would.
- The reviewer ran one 60 s single-thread CPU check. That is over the 10 s limit I gave it; no GPU or model was
  used.

Prediction for A0, written before any run: parse failures on v2pool 91 → 0. Belt pass 74.4% → at least 78%.
- Most forced parses become a wrong tree, not a right one: the mask fixes the form, not the choice of nesting.
- So the step-list-in-reading-order cases (32) and the wrong-tree cases (41) stay mostly wrong.
- That gap is what A1 is for.

## 7. A0 result: grammar-masked decoding with the existing s5 adapter (2026-10-26)

Run: `$E2E/a0/run_a0.sh` (`E2E=/media/llm-data/universe-runs/bytebrain-pycode-20261022/e2e`). The masked eval
covers only the requests whose plain s5 output is not a complete valid spec (`v2pool_ungram` 93, `para_ungram` 13);
the other requests keep their plain output, so the two arms differ only where plain greedy was ungrammatical.
Merged raws: `a0/raws_{v2pool,para}_s5_gm.json`; belt logs `a0/run_{v2pool,para}.log`.

Smoke (12 v2pool requests: 6 plain-valid, 6 parse errors):
- The 6 plain-valid requests come out byte-identical, with 0 forced tokens. The masked arm is like-for-like.
- The 6 parse errors all become valid specs, and all are wrong. In each one the model wanted a third argument (it
  read the request as a sequence: "a, then b, then c"), and the mask forced `)` there, which cut the tree short.

Full:

| set | plain s5 belt | masked belt | requests fixed | masked output exact |
|---|---|---|---|---|
| v2pool (593) | 441 (74.4%) | **449 (75.7%)** | 8 of 93 | 5 of 93 |
| B / para (174) | 149 (85.6%) | **150 (86.2%)** | 1 of 13 | 0 of 13 |

- v2pool: 92 of 93 requests had at least one forced token (142 tokens in all).
- 4 requests were left unmasked (`gm_requests_unmasked` 4): the valid token was not in the model's top 1024.
  - `)` is valid at those prefixes (checked with `spec_prefix_ok`).
  - The mask now ends with a whole-vocabulary tier. Rerun on those 4 (`v2pool_unm4`, `a0/eval_unm4.log`): unmasked
    0, all 4 parse, 0 of 4 pass the belt (`a0/run_unm4.log`). So the v2pool number stays 449.
  - Mechanism confirmed: the only change was the tier, and it removed every miss.
- After the mask, the understander is still the root of 144 v2pool failures (152 before).

Predictions (§6 and the second one written before the full run):

| prediction | result | verdict |
|---|---|---|
| v2pool parse failures 91 → 0 | 0, after the whole-vocabulary tier (4 were unmasked before it) | PASS |
| v2pool belt ≥ 78% | 75.7% | **FAIL** |
| at most 10 of 93 fixed, belt ≤ 76% (written after the smoke) | 8 fixed, 75.7% | PASS |

Reading:
- The mask fixes the **form** and leaves the **choice of tree**. A parse error here is usually a sequential reading
  ("take a, then add b, then multiply by c") that does not fit one nested call. Forcing a valid prefix closes the
  call early, and the meaning is lost.
- So a grammar mask cannot reach the form failures that matter. It is cheap and safe (it never changes a valid
  output), so it stays available as a guard; it is not the fix.
- The fix has to change what the model is asked to write. A sequential request should map to a sequential spec.
  That is A1: addressed steps `s1 = op(x, y)`, one operation per step in the order the request says it.
