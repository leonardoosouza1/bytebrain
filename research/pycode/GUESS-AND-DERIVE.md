# Guess and derive — the "mathematical model" line (ledger #23–#26)

Written 2026-10-23 at Leonardo's request ("anote bem isso que está dando certo e as últimas coisas que falei").
Read this before touching `synth/gradsearch.py` or planning the next step of the pycode line.

## 1. His idea, in his words

- #24: "isso pode virar um modelo 'matemático', algo que dependendo do problema ele resolve ou encontra o melhor
  caminho, porque acho que podemos usar derivada"
- #25: "ele pode gerar em python... isso já é uma junção de modules, mistura de experts, então dá para encaixar
  mais coisas"
- #26: "se o modelo deriva e se ajusta, e isso morar em um loop de nano ou micro segundos, vira um modelo
  extraordinário... se ele chuta e deriva, a derivada vai apontar a direção, basta o modelo seguir e derivar, até
  chegar em algum momento que a distância da derivação quase não muda ou fica a mesma"

## 2. The architecture this describes

```
problem (I/O examples, or a name / description)
   │
   ├─ GUESS: a learned net (today: blockaddr planner, 149k) proposes the plan = router logits per slot
   │
   ├─ DERIVE (inner loop, target: µs per step, on GPU):
   │     each slot = soft router over EXPERT modules (units, calc ops, verified functions)
   │     differentiable executor runs the soft program on the examples
   │     loss = distance to the expected outputs; gradient -> router logits; step
   │     stop when |Δloss| ≈ 0 (fixed point: "a distância quase não muda")
   │
   ├─ SNAP: argmax -> discrete Python program
   │
   └─ JUDGE: bwrap execution on hidden inputs (truth, not the loss)
         verified program -> becomes a NEW EXPERT in the library (modules grow)
```

- Guess is the "intuition" part; derivative is the "calculation" part; the judge is the truth.
- The loop is a mixture of experts whose routing is solved per problem by gradient, not only learned in training.
- Convergence criterion is his: iterate until the derivative distance stops changing.

## 3. What is proven to work (evidence, all CPU, strict bwrap grade)

| piece | result | artefact |
|---|---|---|
| planner (learned address, 149k) + byte writer d32 (13k) | 100% pass on test_a, len3–6, 3/3 seeds [erratum review #44: 3 writer seeds on the best planner seed (4); planner len6 over 7 seeds = mean .76, one at .13] | `bytewriter18/*.json` |
| address is what gives length | plain byte seq2seq 190k, no address: len6 2% | `plain_bytelm/*.json` |
| main calling dependent functions, full 15-op calculator | 100%, 3/3 seeds (interleaved plan) | `multifn_il/*.json` |
| main as call tree (compose) | calls8 93–100% (3 seeds) | `compose/*.json` |
| like-for-like vs Qwen2.5-Coder-0.5B (494M), same head, 20 covering shots, greedy, same judge | Qwen: test_a 46, test_num 53, len3 26, len4 8, len5 9 | `qwen_compare/` |

Caveat on the Qwen row: our system only knows this closed 47-unit world; Qwen knows general Python. The result
says "a tiny specialist with a learned address beats a 3000x bigger generalist inside its own domain", not more.

## 4. What does NOT work yet (honest)

`synth/gradsearch.py`, first version: derivative alone, random init, no guess, 8 examples, 30 problems per length.

| length | grad fits 8 examples | grad correct on 40 hidden | random search (same budget) fits / hidden |
|---|---|---|---|
| len2 | 67% | 13% | 100% / 40% |
| len3 | 43% | 3% | 100% / 33% |
| len4 | no result recorded (the run was never written down; review #44 §1.3) | | |

[Erratum 10-26, v2 harness `gradsearch_v2/repro_legacy`, legacy flags reproduce this table exactly on len2 (fit 0.667,
strict 0.133, mean apps 94548.33): the "hidden" column was graded STRICT (intermediate list state must match). The
problem only gives I/O; graded on output only, the same len2 run gives derive 53% and random 80%. Ambiguity, exact
enumeration on len2: on average 325 programs fit the 8 examples, and 312 of them are output-equivalent on the 40
hidden inputs. Of the programs that fit, 80% are right on hidden I/O, for both arms. Reading 1 below was mostly the
strict grade.]

Reading:
1. 8 examples under-determine the program: many programs fit 8 and fail 40 (random fits 100% but is right 33–40%).
2. Pure derivative from a random start gets stuck (soft mixtures of filters/masks form local minima) — it loses
   to random search where the space is small.
3. Bug classes to check before calling it a failure (rule 7): sigmoid sharpness T fixed (no annealing), loss scale
   (cumsum/times blow values up), identity slot dominating, lr 0.3 fixed, no noise between restarts.

**v2, 10-26, guess + derive (`synth/gradsearch.py`, `gradsearch_v2/main_l23/gradsearch.json`).** 40 problems per
length, 8 examples, graded on 40 hidden inputs by output (io). Guess = noisy oracle (true one-hot x4 + N(0, 1.5);
Monte Carlo: whole program right .44 / .29 / .19 for len2/3/4), one guess per problem, Gumbel noise on restarts.

| len | guess only | guess + derive (sharp start) | guess + sample (same budget) | guess + derive (annealed) | derive only | random |
|---|---|---|---|---|---|---|
| len2 io | .475 | .875 | **.975** | .85 | .525 | .775 |
| len3 io | .40 | .675 | **.825** | .65 | .40 | .425 |
| len3 median work (apps) | 0 | 3525 | 4 | | | |
| len4 io (n=40, 10-26 18:32) | .525 | .775 | **.925** | .825 | .50 | .475 |
| len4 mean work (apps) | 0 | 52875 | 973 | 56870 | 181538 | 81681 |

Reading: the guess carries the result; the derivative helps it (+27.5 pp at len3) but drawing samples from the
guess's own distribution helps more for ~1/1000 of the work. Over discrete op slots the gradient direction is not
what solves it. Ambiguity on len2 (exact enumeration): 253 programs fit, 240 output-equivalent. len4 (`main_l4`):
P2 PASS (fit8 +37.5 pp, io +25 pp over guess alone), KILL not triggered, P3 holds again (sampling .925 vs derive .775 at
1/54 of the work).
Adaptation: derivative on what is continuous (constants, thresholds, the calc ops), sampling on the discrete slots.

**v3, 10-26 night, the derivative where it is native (`synth/gradconst.py`, `gradconst/rep_len{1,2,3}/gradconst.json`).**
Straight-line programs `s_k = op_k(x, y)` over 8 calc ops, x/y from {a, b, earlier steps, a real constant c_k}; the
hidden program uses >= 1 constant. 8 examples, graded on 40 hidden inputs (relative error 1e-4). Same noisy-oracle
guess as v2 (argmax right .735 / .59 / .42). Discrete op and wiring: sampled near the guess. Constants: start at 1.0,
then fitted by Levenberg–Marquardt (derivative) / (1+1)-ES (no derivative) / random draws, at the same budget of
program executions. n = 200 per length; the first 40 rows are the first run, identical row by row.

| len | sample + derive | sample + ES | sample + random | all soft (derive) | guess only | constant answer |
|---|---|---|---|---|---|---|
| len1 io | **.955** | .895 | .02 | .59 | .005 | .20 |
| len2 io | **.905** | .735 | .035 | .385 | 0 | .11 |
| len3 io | **.795** | .465 | .03 | .22 | .005 | .055 |
| len3 median executions | 229.5 | 2851.5 | 6000 (cap) | 6011 | 1 | 0 |
| len3 median wall / problem | 26 ms | 218 ms | 507 ms | 400 ms | | |

Reading: this is his loop working. "Chuta" = sample the discrete program near the guess; "deriva até |Δ| quase não
muda" = LM on the constants until the residual is under tolerance. The derivative beats derivative-free search at the
same budget (+17 pp len2, +33 pp len3, 12x fewer executions at len3) and the gap grows with length. All-soft (derive
the op choice too) loses by 57 pp at len3: v2's lesson holds — discrete by sampling, continuous by derivative.
Where it fails: the derivative is zero on flat regions (`maxx(a, c)` with c below every input: LM cannot move c);
constants identifiable only as a sum fit the 8 examples and miss the hidden ones (len3 fit .85 vs io .795).
Limits: the guess is an oracle stand-in; synthetic formulas; CPU, milliseconds, not microseconds yet.

## 5. Next steps, in order (his idea first)

1. Fix/adapt the derive loop: anneal T, normalise loss, Gumbel noise per restart, simplicity prior; rerun len2–4.
2. **Guess + derive** (#26 core): planner logits as the starting point, derivative corrects only what the guess
   got wrong; compare guess alone / derive alone / guess+derive on the same problems.
3. More examples / prefer the simplest fitting program (fixes ambiguity).
4. Move the inner loop to GPU in batch (thousands of problems per kernel), measure µs per derive step.
5. Plug more experts: calc ops (continuous, derive natively), compose tree slots, verified programs as experts.
   Plan after v2 (10-26, #25 + #26 adaptation, not built yet): a soft straight-line program in the A1 steps
   language, `s_k = op_k(x, y)`: per step a softmax over the calc ops (multifn.DEFS) and, per argument, a softmax over
   {a, b, s_1..s_k-1, c_k}, where c_k is a real constant. Discrete parts (op, wiring) are searched by sampling
   near the guess (what won in v2); the constants are fitted by the derivative until |dloss| ~ 0 (his stop rule),
   which is where the gradient is native. Arms at equal executor calls: sample+derive constants / sample+random
   constants / derive everything (soft ops + constants). Problems: formulas with 1-2 constants (unit conversions,
   'a vezes 1.8 mais 32'), 8 examples, 40 hidden. Same executor also repairs WIRING (the A1 failure) whenever a
   request carries examples.
   **Built and run 10-26 night (v3 in §4): PASS on all four registered predictions.** Next: replace the oracle guess
   by the real understander/planner output (A1 steps spec) and the 8 examples by examples the request carries
   ("a vezes 1.8 mais 32" → constants 1.8, 32 fitted, not guessed); then batch the LM loop on GPU (step 4).

## 6. Review 2026-10-26 (gaps found, nothing rerun)

- gradsearch len4 lost: the process died with the session and results are written only at the end. Fix: dump per length.
- gradsearch hidden grade is STRICT (list state before the final must match the true program). The problem only
  gives I/O, so an output-equivalent program with another intermediate state is counted wrong. Must use
  strict=False for this test (it biases both arms down, more for finals len/sum/count_gt).
- Ambiguity is unmeasured: we do not know how many distinct programs fit the 8 examples; need that count per problem.
- Qwen comparison: 100 sampled tasks per split vs our full splits; no failure taxonomy; raw completion without chat
  template (may understate an instruct model); 3B arm not run.
