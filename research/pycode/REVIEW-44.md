# Review #44 — everything done 10-21 → 10-26, not the code (2026-10-26, evening)

His request (ledger #44): "revise tudo novamente... não revisa código e sim tudo que foi feito nesses dias... gaps e
problemas... agents opus... reveja futuros possíveis erros, pq vc falou de coisas que aconteceram, mas já deveria
ser resolvido pq era previsto".

Method: 4 Opus agents, read-only, no GPU, each with one lens:
- A: claims vs artefacts;
- B: foreseeable errors + pre-mortem of the queue;
- C: alignment with his requests;
- D: τ-bank (#45) + saci-code (#46).

An agent report is not evidence. Every point below marked **re-checked** was recomputed by me (command or file
named); the rest is marked *agent, not re-checked*.

`R = /media/llm-data/universe-runs/bytebrain-pycode-20261022`

## 1. Findings, most severe first

### 1.1 The "planner + writer 100% on len3-6" headline used a hand-picked planner — re-checked
- The 7 planner seeds (gate bias −4) score on len6: .645 / 1.0 / .955 / .98 / 1.0 / .58 / .1325. Mean **.76**,
  median .955. Source: `R/blockaddr/blockaddr_addr_s{0..6}_gb-4.json`.
- Every byte-writer run used `planner_seed: 4`, one of the two 1.0 seeds. Source: `R/bytewriter18/*.json`.
- So "3/3 seeds" were writer seeds on top of the best planner. The plain byte-LM baseline used unselected seeds.
  Rule 8 broken.
- Honest statement: the addressed planner reaches len6 ≥ .95 on 4 of 7 seeds and collapses on 1 (.13). It is
  fragile to the seed. With the default gate bias, len6 was 1.0 / .02 / 1.0 / 0 / .99 / 0 / 0, so 4 of 7 seeds
  collapsed (`R/blockaddr/blockaddr_addr_s{0..6}.json`, re-checked).

### 1.2 Judges and fix acceptance are gold-gated, the same flaw that cancelled V2 — agent A, partly re-checked
- `verify_fix` scores a proposed fix against gold. So "Sonnet fixes 20/20 accepted, 20 repaired" measures a fix
  that was checked with the answer key. That cannot run without gold. Status → PARTIAL (ceiling, not a loop).
- In the natural sample every root is the understander, so a constant "understander" blamer also scores 60/60.
  The constant baseline was applied to the 3B judge, never to Sonnet. "Sonnet validated as the final judge" is not
  shown until it is tested on a sample with mixed roots.

### 1.3 Runs that were made and never written down — re-checked for bridge_cpu, agent A for the rest
| run | what happened | where |
|---|---|---|
| τ-bank CfC (#8/#45), 10-22 | cfc_bank test_a .67-.78, **test_num .30-.40** (GRU .89-.91), lr best at grid edge, control `cfc_tau1` never ran (0-byte log) | `R/bridge_cpu` |
| qwen_closed_form, 10-23 | kill rule fired, strict free-run 12.5 / 2.5 / 4.2% | `R/qwen_closed_form` |
| qwen_law, 10-23 | G0 failed | `R/qwen_law/g0.json` |
| Qwen-0.5B teacher on CPU | strict 25% len1, 1.7% len2, 5% len3 | `R/qwen_cpu/teacher_0.5b` |
| addr2d, addrseg QAT arms | never finished / never ran | `R/units_addr` |
| gradsearch len4 | no result; GUESS-AND-DERIVE still says "running" | — |
| Qwen-3B compare | died after the smoke | `R/qwen_compare/full3b.log` |
| ternary byte writers | d32 ternary len6 .49-.67, d16 ternary 0; omitted from #18 | `R/bytewriter18` |

### 1.4 Numbers measured before an erratum still quoted — agent A, not re-checked
- 26M pycode evals (`R/eval_*.json`, `R/equiv_*.json`) predate the print-safe runner fix (16f8dfa). The "form, no
  meaning" verdict on real Python rests on them. Likely undercount.
- The "92% of errors are the middle line" diagnosis predates the unpadded re-evaluation.
- README still says unlikelihood "−33 pp"; the erratum says −26.

### 1.5 CONCEPT-REVIEW §2 overstated the form finding — re-checked
- "Natural requests have no parentheses": wrong. 284 of 454 v2pool formula rewrites keep brackets beyond
  `main(a, b)`. They pass 76.4%; the 170 without brackets pass 50.6%. The direction of the mechanism holds (no
  brackets → twice the failure), the absolute claim does not.
- "87 (58%) have exactly the gold's operations": the table column sums to 85, and that is the op **set**. Counting
  repeated ops (the multiset, which is what the meaning needs), at most 50 of 151 (≤ 33%) match: 39 by my regex
  recount plus at most 11 calculator-shaped answers.

### 1.6 Wrong names and counts — agent A, writer part re-checked
- The belt writer is `multifn.Writer`, `nn.Embedding(len(SYM))` conditioning (`multifn.py:211`), ~45k params: a
  learned lookup, not the byte writer. The belt has two planners (~149k each). "planner + writer ~162k" mixed in
  the d32 byte writer, which is not in the belt.
- "3B few-shot 78.8 / 74.7" is the run with 2 send-back rounds (`summary_q3b_sb2.json` pass .7877); plain few-shot
  is .7263. The students have no send-back: not like-for-like.

### 1.7 Stale statuses — re-checked in the ledger
- #13 DONE although "liquid neuron predicting" was never built (rule 6) → PARTIAL.
- #7 OPEN although the multi-model chain and the locating judge exist → PARTIAL.
- #22, #15 RUNNING with nothing running; #34/#35 mention s5p2/V2 as running (V2 cancelled, s5p2 never ran).
- #10 DONE ("bridges CfC/bytes") while the CfC bridge failed and its control never ran.

### 1.8 Drift — agent C, ledger tally re-checked
- After "pode continuar", the first GPU slot went to my own idea (C1/A0, +1.3 pp, prediction failed), while his
  most repeated asks (#8 τ-bank, about six times; #24-26 guess-and-derive) have no running code. Rule 4 broken.
- What blocks "aprende sob demanda" today (re-checked): a new operation is an **error** (`oov_op`,
  `e2e_pipeline.py:472`); the vocabulary is fixed in `multifn.DEFS`, `compose.BIN`, the prompt lexicon and the
  planner head; the writer is a lookup; the test inputs are a fixed grid.

## 2. Foreseeable errors (his "já deveria ser resolvido pq era previsto") — agent B; examples re-checked

Agent B counted 26 incidents in 6 days; **21 were predicted by a lesson already on disk**, 5 were new (count not
re-checked). For the four examples below, the lesson is in the named memory file (re-checked by grep):
- τ-bank lr grid edge + missing control: the lesson "CfC é instável de treinar" existed (project_celula_liquida).
- 3B judge below the constant baseline: "controle positivo E negativo do juiz" existed (project_hybrid_codegen).
- main ⇒ formula shortcut: the 42% paraphrase-leak lesson (project_stress_learning) and the degenerate-policy check.
- A0 predicted ≥ 78% while the same section said the forced parses would be mostly wrong: 78% needed ~21 of 93
  fixed. "Check the arithmetic of every prediction" existed (feedback_revisao_obrigatoria #15).

Three root causes:
1. Lessons are filed by past incident, not by trigger ("when you build X, run Y").
2. The pre-run gates check the machine (VRAM, disk, threads), not the inference (baseline, where gold enters,
   did the control run).
3. Fixes are applied to one file, not to the bug class; ledger statuses are written from intent.

Fix: 12 inference gates now live in `bytebrain/CLAUDE.md` (section "Pre-run gates"). They are checked before
every GPU slot, not after.

## 3. τ-bank (#45) — what exists and what never ran

| where | variation | result |
|---|---|---|
| pycode bridge_cpu (10-22) | CfC, 8 fixed τ groups 1..128, leak `s += (c − s)/τ` | lost to GRU; under-tuned; control never ran |
| pycode blockaddr | τ = ∞ counter `sigmoid(gate).cumsum()`, moves only on boundary events | works (len6 ≥ .95 on 4/7 seeds); holds the ADDRESS, not data |
| EXP10 survival world | 8 EMA, τ evolved | having spread τ pays (40× spread vs 5.3× random) |
| same | τ switched by regime | does not pay (+0.11%) |
| saci BTC volatility | evolved bank [1, 17, 120, 2000, 20000] | +2.1 pp R² over the hand timeframes |
| saci cells by BPTT | LTC / GRU / leaky | tie 53.6%; CfC 47.3 ± 5.1, unstable |
| pycode bridge_cpu45 (10-26 rerun, 3 seeds, lr grid 1e-3..2e-2) | K=0 / K=48 wait steps: gru, cfc_bank, cfc_tau1, learned τ, stress-gated (opens on \|c − s\|) | K=0 test_a gru .993, bank .823, learned .774, stress .820. K=48: gru .942, every CfC arm ≤ .027 and fails to fit TRAIN. Stress on \|c − s\| opens during the wait (PAD input ≠ held state) and overwrites; optimizer shut the slow groups. KILL for 'fixed huge τ holds the data' |
| same, FAILED-ADAPT (registered, running 10-26) | cfc_gate: stress read from the INPUT (one gate per τ group); rival: bank at 4× steps | pending |

Why the pycode test could not show a benefit (agent D, lengths re-derivable from the data): sequences are ≤ 21
tokens in train, so the τ = 64 and 128 groups barely move in one sequence; and the leak is a fixed average, not a
holder gated by "stress". The GRU's update gate is itself an input-gated τ per unit, and it won.

**Never tested:** learned τ in pycode; a content bank that changes only under stress (ARCHITECTURE §3.2); a small
network reading an explicit multi-τ bank. Plan in §6.

## 4. saci-code (#46) — design that survives the past lessons

What saci-trader actually taught (agent D, from the saci study docs and memories):
- Volatility/regime is predictable (R² ≈ 48-50% OOS); direction was not. The edge was **selectivity**: confidence
  + a consensus gate turned ~55% accuracy into 82-87% net-win on the trades it took.
- The 85-89% "accuracy" was a lagging EMA label (already happened). Lesson: no feature may see the future.

Translated: saci-code is most likely worth more as a **"this one is agitated, escalate / ask again"** signal than
as a mid-function repair, because:
- "repair" already tied fresh resampling (51.75 = 51.75 on 10-22) even when it was told the true first wrong line;
- the always-on address already fixed the "loses its place in the middle" errors (pointer 67.7 vs 33.0).

Measured today (re-checked on `v2pool/traces_student_s5.jsonl`): understander flags fire on 93 of 152 failures and
on 0 of 441 passes. **40 failures carry no flag at all**: these silent failures are the target. Per-token scores
were never saved (`raws.json` = text + ms), so the first step is one GPU eval pass that logs entropy, margin and
chosen logprob per token.

Arms it must beat: final-only check; current-token entropy; single τ; position only; a monitor that reads only the
request (if equal, it is not monitoring); resampling at equal compute.

#47 (his idea, running): an Opus agent writes 24 examples (request → spec → steps → program; 4 with a new op).
Each is executed in bwrap against `multifn.DEFS` before it counts as good. Mutants with a known first-wrong
position (like his `return a - b` in `sum`) give the bad side for free, with exact labels.

## 5. Gates for the queue (pre-mortem)

- **A1 (running):** gold steps pass the belt 179/179 (`R/e2e/a1/goldsteps/run.log`, re-checked); the longest gold
  steps spec is 59 tokens vs the 120 cap (re-checked). Set A is the fair number; B and v2pool are dev sets now.
- **A8 s5p2:** one-rule classifier "contains `main(`" → world on the final mix must be < 90%; decide its format
  after A1.
- **A4:** blind set C from him or another model family, with provenance; equal-step control; grep the accept path
  for gold.
- **#41:** GGUF q8 parity with HF greedy ≥ 99% on set A before any latency number.

## 6. Order after A1 (his items first, rule 4)

1. **#45 τ-bank rerun (CPU, minutes):** lr grid {0.003, 0.006, 0.01, 0.02}; run `cfc_tau1`; add learned-τ and
   stress-gated arms; 3 seeds; longer sequences where memory is needed. Prediction: tuned bank ≥ 90% test_a, and
   the stress-gated arm ≥ ungated at unseen length. If it fails, the bank moves to the saci-code monitor role.
2. **#46/#47 saci-code:** verify the Opus examples; build mutants; one GPU pass that logs per-token signals;
   CPU monitor test with the arms above.
3. **#24-26 guess-and-derive as learn-on-demand:** `oov_op` becomes a "learn" branch with I/O examples.
4. Then s5p2 / A4.

Corrections applied in this review: ledger statuses (#7, #10, #13, #15, #22, #34, #35), README planner-seed and
unlikelihood errata, CONCEPT-REVIEW §2 errata, unrecorded runs table above.
