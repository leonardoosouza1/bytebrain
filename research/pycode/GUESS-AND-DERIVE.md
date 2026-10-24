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
| planner (learned address, 149k) + byte writer d32 (13k) | 100% pass on test_a, len3–6, 3/3 seeds | `bytewriter18/*.json` |
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
| len4 | running | | |

Reading:
1. 8 examples under-determine the program: many programs fit 8 and fail 40 (random fits 100% but is right 33–40%).
2. Pure derivative from a random start gets stuck (soft mixtures of filters/masks form local minima) — it loses
   to random search where the space is small.
3. Bug classes to check before calling it a failure (rule 7): sigmoid sharpness T fixed (no annealing), loss scale
   (cumsum/times blow values up), identity slot dominating, lr 0.3 fixed, no noise between restarts.

## 5. Next steps, in order (his idea first)

1. Fix/adapt the derive loop: anneal T, normalise loss, Gumbel noise per restart, simplicity prior; rerun len2–4.
2. **Guess + derive** (#26 core): planner logits as the starting point, derivative corrects only what the guess
   got wrong; compare guess alone / derive alone / guess+derive on the same problems.
3. More examples / prefer the simplest fitting program (fixes ambiguity).
4. Move the inner loop to GPU in batch (thousands of problems per kernel), measure µs per derive step.
5. Plug more experts: calc ops (continuous, derive natively), compose tree slots, verified programs as experts.
