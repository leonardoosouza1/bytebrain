# Stress-gated learning on demand

Pre-registered 2026-10-13, before the full run.

## Question

Can a network decide by itself *when* and *what* to learn, using "stress"
(surprise against a fast fact bank), and reach the accuracy of always-training
with far fewer updates? Does the stress signal select better than chance?

## Setup

- Substrate: Qwen3-0.6B (fp32) + LoRA r=16 on all attention/MLP projections.
- Bank: 500 invented facts (4 relations: capital, inventor, river, language),
  entity and answer are random pseudo-words, so the base model cannot know them.
  5 Portuguese paraphrases per fact: 3 appear in the stream, 2 are held out.
- Stream: 5000 questions, Zipf(1.0) over facts (few frequent, many rare).
- Stress `S` = mean NLL of the bank answer tokens given the question.
- Arms (fresh LoRA each, same facts and stream):
  - `never`: always answers from the bank, never trains.
  - `always`: one training step (new item + 3 replay) on every question.
  - `stress`: if `S > 0.5`, answers from the bank and trains with
    `lr = 3e-4 * min(S/0.5, 4)` until `S < 0.2` (max 5 steps), with replay.
  - `random`: same event count, steps and learning rates as `stress`, applied
    to randomly chosen questions.
- Metric: closed-book teacher-forced exact match (== greedy exact match) on
  held-out paraphrases of every fact seen so far; frequency-weighted variant;
  number of optimizer steps; bank consultations; retention of facts first
  learned in the first half of the stream.
- Seeds 0, 1, 2.

## Predictions (written before the run)

1. `stress` final held-out accuracy >= 90% of `always`, using <= 30% of its steps.
2. `stress` > `random` on final held-out accuracy at the same number of steps,
   in all 3 seeds.
3. `never` stays at ~0% closed-book (sanity: the facts are unknown).

If (2) fails, the stress signal does not select anything: it is only a counter.

## Smoke test (20 facts, 120 questions, seed 0) — not evidence

| arm | steps | held-out acc |
|---|---|---|
| never | 0 | 0.00 |
| always | 120 | 1.00 |
| stress | 52 | 0.82 |
| random | 52 | 0.59 |

## Run

```
HSA_OVERRIDE_GFX_VERSION=10.3.0 python stress_learning.py --seed N \
  --out /media/llm-data/universe-runs/stress-learning-20261013/full
```

## Amendment 1 (2026-10-13 12:10, before any test-seed result)

The pre-registered stress learning rate (`3e-4 * min(S/0.5, 4)`, max 5 steps)
collapsed on seed 0 (held-out 0.000 at question 500): every new fact arrives
with S ~ 7, so it is learned at the maximum rate and overwrites the previous
ones; replay of 3 items cannot keep up. Not NaN: s_after ~ 0.2 per event.

Recalibrated on a separate calibration seed (99, different facts), 300 questions:

| variant | steps | held-out |
|---|---|---|
| always, lr 3e-4 | 300 | 0.744 |
| stress, lr 3e-4 fixed, max 5 | 464 | 0.659 |
| stress, lr 1e-4 * min(S/0.5, 3), max 3 | 419 | 0.752 |

Test seeds 0-2 now use: stress `lr = 1e-4 * min(S/0.5, 3)`, max 3 steps;
always keeps lr 3e-4. Predictions unchanged.

## Results (2026-10-13, 3 seeds, 5000 questions, 500 facts)

Artefacts: `/media/llm-data/universe-runs/stress-learning-20261013/full/seed{0,1,2}_{arm}.json`

| seed | never | always | stress | random | stress steps | stress/always | served always | served stress | consults stress | S<0.5 hit / S>=0.5 hit | retention early | stressed facts relearned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.927 | 0.810 | 0.483 | 1764 (35%) | 0.87 | 0.865 | 0.964 | 814 | 0.957 / 0.025 | 0.824 | 0.46 |
| 1 | 0.000 | 0.935 | 0.883 | 0.467 | 1747 (35%) | 0.94 | 0.860 | 0.965 | 810 | 0.958 / 0.030 | 0.887 | 0.43 |
| 2 | 0.000 | 0.912 | 0.784 | 0.510 | 1752 (35%) | 0.86 | 0.865 | 0.961 | 852 | 0.953 / 0.032 | 0.802 | 0.48 |

Held-out = closed-book exact match on the 2 never-trained paraphrases, mean over facts seen.

Verdicts against the predictions:

1. FAIL. Stress reached 0.86-0.94 of `always` (>= 0.90 in 1/3 seeds) and used
   35% of its steps in all seeds (predicted <= 30%).
2. PASS, 3/3 seeds. Stress beats random at identical steps and learning rates
   by +32.7, +41.6, +27.4 points.
3. PASS. `never` is 0.000 in all seeds.

Observed, not predicted:

- Stress is a calibrated "I don't know" alarm: when S < 0.5 the model answers
  correctly 95.3-95.8% of the time; when S >= 0.5 only 2.5-3.2%.
- Served accuracy (bank when stressed, model otherwise) is 0.96 vs 0.86 for
  `always`, which answers alone even when it does not know.
- Cost falls with knowledge: steps per 1000 questions ~900 -> ~100-140.
- Weakness is forgetting: 43-48% of stressed facts had to be relearned at
  least once (up to 8 times). `always` re-sees facts constantly and forgets less.
- Frequency-weighted accuracy of `random` is close to stress on seed 0
  (0.862 vs 0.867): random naturally trains popular facts; its loss is on rare facts.

Next lever: consolidation against forgetting (batched "sleep" replay, or
updating only the fact's neuron territory).

---

# Sleep consolidation (pre-registered 2026-10-13, not yet run)

Script: `sleep_learning.py`. Runner: `run_sleep.sh`. Live view: `watch_sleep.py`.

## Question

Stress picks *what* to learn well, but 43-48% of stressed facts were forgotten
and relearned. Does an offline "sleep" phase, rehearsing what was learned while
awake, cut that forgetting? Is any gain from sleep itself or just from more replay?

## Arms (seeds 0-2, same facts and stream as the stress battery, checked by assert)

Wake is identical in all arms: the calibrated stress rule
(`lr = 1e-4 * min(S/0.5, 3)`, max 3 steps, stop at S < 0.2, 3 replay items).

| arm | what it does |
|---|---|
| `wake` | wake only; re-run of `stress` with this code |
| `wake_replay` | wake only, 7 replay items per step instead of 3; spends about as many item-gradients as `sleep_uniform` |
| `sleep_uniform` | every 250 questions: 20 steps x 16 facts drawn uniformly from the learned buffer |
| `sleep_fading` | every 250 questions: score S of every learned fact on one train paraphrase; train only those with S > 0.3, weakest first, up to 20 x 16; pad the last batch uniformly; skip if none fade |

Sleep lr is 1e-4. Eval runs **before** each sleep, so every arm is measured at
the end of a wake period (no fresh-rehearsal bonus); there is no sleep after
question 5000. Cost is counted in item-gradients (items per training step,
wake + sleep), plus forward-only scoring for `sleep_fading`.

References from the stress battery (same seeds): `always` 0.912-0.935 held-out
with 20 000 item-gradients (5000 steps x 4); `stress` 0.784-0.883 with ~7000.

## Predictions

1. Held-out order `sleep_fading` > `sleep_uniform` > `wake_replay` ~ `wake`,
   in at least 2 of 3 seeds.
2. `sleep_fading` relearned fraction < 0.25 (stress battery: 0.43-0.48).
3. `sleep_fading` held-out >= 0.95 x `always` while spending <= 70% of its
   item-gradients (<= 14 000).
4. `sleep_fading` sleep cost falls over the stream (fewer fading facts per sleep
   in the second half than in the first).

If `wake_replay` ~ `sleep_uniform`, sleep is just more replay. If
`sleep_fading` ~ `sleep_uniform`, stress-guided rehearsal adds nothing.

## Budget

Per seed 4 arms; each arm costs about one `stress` arm (~25-35 min with two
seeds in parallel) plus up to 400 sleep steps. Estimate ~2-2.5 h wall time for
3 seeds. GPU memory: same as the stress battery (two processes in parallel fit).
Disk: a few MB of JSON.

---

# Erratum (2026-10-13 15:15) — held-out paraphrases leaked into the stream

`make_stream` drew the paraphrase as `randrange(3)` and the encoder used it
directly as the paraphrase index, instead of mapping it through the fact's
`train_para`. In the stress battery, seed 0, **2091 of 5000 stream questions
(42%) used a paraphrase listed as held out**, touching 330 of 462 seen facts.

Consequences for the results above:
- "held-out" accuracy was partly measured on paraphrases the model trained on;
  it is not a clean generalization measure.
- arm comparisons (stress vs random vs always vs never) share the same leak, so
  the ranking and prediction 2 remain valid as comparisons; absolute held-out
  numbers and the stress/always ratio must be re-measured.

Fix: `train_paraphrases(facts, stream)` in `stress_learning.py`, used by all
three scripts. The sleep battery and any re-run use the fixed stream.

# Fast engine (phase 1) — not yet run

`fast_learning.py` keeps the per-question semantics and changes execution only:
GPU token table, scoring window cut at the first training trigger, answer-only
logits, `always` served from its own training forward, batched eval. Optional
`--dtype fp16` (fp16 frozen base, fp32 LoRA, fp32 answer logits).

Gate before any battery: `check_equivalence.py` (old sequential loop vs fast,
same facts/stream, 300 questions). Pass = identical event list, max |dS| < 1e-3,
no theta flips. Then the fp16 gate: same script with `--dtype fp16`
(pass = max |dS| < 0.02, flips only near theta).

Fast battery defaults: 200 facts, 2000 questions, eval and sleep every 100,
sleep 8 steps x 16, arms wake, random, always, wake_replay, sleep_uniform,
sleep_fading. Predictions of the sleep section stand, with references re-run
in the same battery (wake, always, random) instead of the old numbers.

# Results — lane battery (2026-10-13 16:14-17:02, fixed held-out)

`lanes_learning.py --seeds 0,1,2 --random`, 200 facts, 2000 questions.
Artefacts: `/media/llm-data/universe-runs/sleep-fast-20261013/lanes/`.
Wall time: 2394 s (15 lanes) + 476 s (3 random lanes) = 48 min; lane gate
measured 2.75x cheaper per lane than one lane alone.

Final held-out (seed 0 / 1 / 2), item-gradients, relearned fraction:

| arm | held-out | item-grads | relearned |
|---|---|---|---|
| always | 0.933 / 0.974 / 0.978 | 8000 | - |
| wake | 0.870 / 0.905 / 0.822 | ~2900 | 0.40-0.43 |
| random | 0.466 / 0.521 / 0.503 | same as wake | 0.41-0.44 |
| wake_replay | 0.922 / 0.905 / 0.951 | ~4800 | 0.19-0.24 |
| sleep_uniform | 0.990 / 0.974 / 0.981 | ~4850 | 0.19-0.22 |
| sleep_fading | 0.956 / 0.955 / 0.876 | ~3060 | 0.23-0.25 |

Verdicts:
1. FAIL. Order was sleep_uniform > sleep_fading, not the reverse (3/3).
   sleep_uniform > wake_replay at equal item-gradients in 3/3: the gain is
   from offline rehearsal, not from more replay alone.
2. Borderline: sleep_fading relearned 0.23-0.25 (target < 0.25).
3. 2/3: sleep_fading >= 0.95 x always with ~38% of its item-gradients in
   seeds 0 and 1; seed 2 reached 0.90.
4. NOT MEASURED (per-sleep cost trend not analysed).
Stress vs random with clean held-out: +40 / +38 / +32 points (3/3).
