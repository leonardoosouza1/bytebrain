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
