# Pipeline validation — the belt, who is to blame, and the judges (ledger #36–#41)

Written 2026-10-26 at Leonardo's request (voice message, ledger #36–#40):
"o Sonnet vai ser o avaliador final do quanto que o Queen errou... seguindo uma verdade, que é a regra",
"é uma esteira... um erro de um vai refletir no outro... validadores nesse loop que façam voltar",
"montar uma pipeline que valide de ponta a ponta... primeiro a gente deixa solto, vê onde é que tá errando".
Code: `synth/e2e_pipeline.py`. The judges' rule: `RUBRIC.md`. Runs: `/media/llm-data/universe-runs/bytebrain-pycode-20261022/e2e/`.

## 1. The belt

```
pedido pt-BR ──► UNDERSTANDER ──► spec JSON ──► bridge ──► PLANNER ──► plan symbols ──► WRITER ──► texts ──► ASSEMBLER ──► code ──► bwrap
  "crie uma        Qwen 3B          {"main":"dispatch",   rule      blockaddr net     def:mul def:square   byte GRU      rule         exec
   calculadora     (frozen,          "calls":["hyp",      spec ->   149k params,      def:sqrt def:add     13k-60k       defs+main
   com hipotenusa  understands,      "avg"]}              name      address τ=∞       def:hyp disp:hyp …   params        dispatch
   e média"        no code)          or {"main":"expr",                                                                  or return
                                     "expr":"hyp(a, b)"}                                                                 expression
```

| stage | input | output | what it is today | learned? |
|---|---|---|---|---|
| understander | pt-BR text | spec JSON (dispatch op list, or one formula) | Qwen2.5-Coder-3B q8, chat template, op vocabulary + 4 shots, grammar-constrained JSON | frozen (fine-tune later) |
| bridge | spec | name bytes `calc_hyp_avg` / `calc_hyp_a_b` | rule | no — PARTIAL (planner reading JSON is #33) |
| planner | name bytes | block symbols in order, with the dependency closure it inferred | `blockaddr.Net` (learned address, τ=∞ counter) | yes |
| writer | one symbol | the text of that block, byte by byte | `multifn.Writer` byte GRU | yes |
| assembler | plan + texts | module | rule | no |
| exec | module | values on 20 cases per capability | bwrap, no network, 1 s per call | — |

## 2. The core problem: errors flow downstream

If each stage is right with probability p_k, the belt is right ≈ Π p_k, and a downstream stage that receives a
wrong input will faithfully produce a wrong output. Two consequences:

1. **Blame must be conditional.** A stage is wrong only if it broke its contract *for the input it received*.
   The planner that perfectly plans the wrong spec is right; the understander that produced that spec is wrong.
2. **Training data must be on-policy** (DAgger, Ross 2011): train each stage on the inputs it actually receives
   from the stage above, labelled with the gold for *that* input, not only on clean gold inputs.

Report both numbers per stage: isolated (gold input) and chained (predicted input). The gap is the compounding.

## 3. Truth, in order (the anti-false-signal ladder)

| rank | source | what it decides | can be wrong? |
|---|---|---|---|
| 1 | bwrap execution vs gold reference | is each requested capability correct | no (inside the closed world) |
| 2 | oracle swap | which stage broke its contract; would fixing it alone repair the result | no (closed world) |
| 3 | machine validators | flags without gold (JSON, schema, OOV op, missing dispatch, undefined helper, exec error) | yes — precision measured vs 2 |
| 4 | Sonnet (final judge) | checks Qwen's verdict item by item against RUBRIC.md | yes — measured vs 1+2 |
| 5 | Qwen (first judge) | verdict: capabilities, blame, fix | yes — measured vs 1+2 |

A fix proposed by any judge enters the training data **only if** rerunning the belt with that fix raises the score
with no regression (rank 1 confirms it). A judge's opinion alone never becomes a label.

## 4. What the harness measures (per request)

- `cap_status`: each requested capability → correct / wrong / missing (execution vs gold reference).
- `score` = correct / requested (his "o pedido foi 80%"), computed by code, never by a judge.
- Contracts (conditional):
  - understander: reference(spec) computes every requested capability (extras allowed);
  - planner: plan + oracle texts == reference(spec) on the spec's own calls;
  - writer: each written function, run alone among oracle helpers, equals the oracle function and has the right name.
- `root` = earliest broken stage when the result fails.
- Counterfactuals: CF_U (gold spec through the real planner/writer), CF_P (oracle plan, real writer),
  CF_W (real plan, oracle texts). "Would fixing only this stage have repaired it?"
- Flags from validators, each with `from → to` stage ("writer avisa planner": a written body calls `y` but the plan
  has no `def:y`). Each flag gets `true` = its target stage really broke its contract → precision per flag kind.
- Per-stage latency.

## 5. Fault injection (known location)

The trained planner/writer are ~100% inside this world, so natural errors are rare there. To test the blame logic
and to calibrate the judges, faults with a known stage are injected:

| stage | faults |
|---|---|
| understander | drop_op, swap_op, add_op (harmless: extra capability) |
| planner | drop_def, drop_disp, swap_disp, dup_def (harmless) |
| writer | sign_flip (his `def sum(a,b): return a-b`), const_change, drop_guard, call_swap |

`check` (all-oracle belt + faults) must give: 100% clean pass, nothing broken, and root = injected stage on every
visible fault. Result after the Opus review fixes (2026-10-26, 3.8 s): **CHECK PASS** — clean 179/179, 0 wrong
roots, 0 unexplained; flag precision `undefined_helper` 48/57 (the misses are writer faults that call a wrong name —
RUBRIC §6 pitfall), `missing_disp` 36/36, `extra_disp` 18/18, `tree_mismatch` 15/15.

## 6. The judges (his design, #36–#37)

```
trace ──► Qwen judge (RUBRIC §2) ──► verdict {capabilities, blame, fix, justification}
              │                         score = computed by code (RUBRIC §3)
              ▼
        escalate? ──no──► final = Qwen verdict        (cost: zero Sonnet)
              │yes
              ▼
        Sonnet final judge (RUBRIC §4): checks every item of Qwen's verdict against the rule
              qwen_ok = true  ──► final = Qwen verdict
              qwen_ok = false ──► final = Sonnet's corrected verdict
              ▼
        fix (from the final verdict) ──► rerun belt with the fix ──► accepted only if execution confirms
              ▼
        replay buffer per stage: understander (request → spec), planner (name → plan), writer (symbol → text)
```

- Test phase: `--sonnet-mode all` (Sonnet sees every sampled trace, so its value can be measured).
- Later: `--sonnet-mode escalate` — Sonnet only when Qwen's verdict is invalid, contradicts a machine fact
  (says correct where the code raised), ignores a validator flag, or a 20% deterministic audit sample.
- Judges see what the code returned on all 20 cases the truth uses (first version showed 3 and was graded on 20 —
  review bug 5), never the expected values: in serving there is no gold, so the judge must know what each operation
  should return (RUBRIC §5 gives the meanings).
- Judge quality is measured against ranks 1–2: pass agreement, capability agreement, root-stage agreement,
  writer-function agreement, false alarms on passing traces, fixes accepted.
- Judges are learners too: Sonnet's corrections of Qwen verdicts are training data for the Qwen judge (later).

## 7. Error routing instead of backprop through the belt

No gradient crosses bwrap or the Qwen JSON. The derivative lives inside each learned stage (its own loss); the
belt decides **which stage gets which example**:

1. run the belt on a batch of requests;
2. blame (oracle in the closed world; judges + validators outside it);
3. accepted fixes → that stage's replay buffer (on-policy inputs);
4. retrain only the most-blamed stage;
5. repeat until the blame histogram and the pass rate stop changing (his "até a distância da derivação quase não
   mudar": Δpass < ε for 2 rounds).

## 8. Send-back channels (his "se o planner errou, o writer tem que avisar")

Loose first: in V0 the flags are recorded, not acted on. Each channel is enabled later only if its measured
precision and counterfactual repair rate justify it, with bounded retries (≤2 per stage, ≤4 total). A retry must
change something (sampling, top-2, or the input), because a greedy rerun of the same input gives the same output.

| channel | detector | target | measured precision (check, injected faults) |
|---|---|---|---|
| spec not JSON / bad schema / OOV op / expr parse error | validator | understander | Qwen run: expr_parse 18/18, oov 10/10 |
| spec op without dispatch line | validator | planner | 36/36 |
| formula tree differs from spec | validator | planner | 15/15 |
| body calls undefined helper | writer → planner | planner | 48/57 |
| exception on the spec's own calls | exec | unknown | never a stage verdict (needs a judge); inherent formula exceptions go to `oracle_notes` |

First channel switched on: understander send-back (`run --sendback 2`). Gold-free: only validator flags aimed at the
understander trigger it; the request is re-asked with the validator's message appended. Result in §11.

## 9. Ladder

| step | what | status |
|---|---|---|
| V0a | `check`: oracle belt + faults | DONE — CHECK PASS |
| V0b | `train`: save planners + writer, reproduce known numbers | DONE — 133.6 s, dispatch 100% test_comb/len4/len8/full, compose test_same 1.0 calls4 1.0 calls8 0.933 |
| V0c | `run`: gold understander + trained planner/writer | DONE — 179/179 |
| V0d | `run`: Qwen 3B understander + trained planner/writer (natural errors) | DONE — 72.6%, every failure rooted in the understander |
| V1 | `judge`: Qwen 3B judge + Sonnet final judge on natural + injected traces; agreement vs oracle | smoke done; n=30+30 running |
| V2 | retrain the most-blamed stage (= understander) | running: #34 student (Qwen 0.5B + LoRA transpiler), §11 |
| V3 | enable send-back channels by measured value | understander send-back ON: 72.6% → 78.8% |
| V4 | serving path: `one` subcommand, understander ~0.5 s on GPU + planner/writer ms + bwrap; judges off the hot path | preview built |

## 10. Honest limits

- Closed world: 15 operations, binary formulas. Gold exists only because the world is small.
- Requests are templated pt-BR (frames + lexicon), not human-written. Some single-op frames read unnaturally
  ("escolha entre valor absoluto").
- The bridge spec → name is a rule (PARTIAL). The understander gets the op vocabulary in its prompt (interface contract).
- Open requests ("calculadora científica") use a required set chosen by me (add sub mul div sqrt pow); extras allowed.
- The understander is frozen; "retrain the understander" means few-shot memory now, fine-tune/LoRA later.
- Judges see the 20 truth cases; a bug that shows only on other inputs is invisible to both judges and oracle.
- Latency `exec` is amortised over a batch that also runs the counterfactual modules; `one` measures the real path.
- Formula requests always say "main(a, b)"; calculator requests never do. A rule on that string would fix the
  world choice for free, but it reads the template, not the meaning, so it is not used.

## 11. Results (2026-10-26)

All on the 179 templated requests (`requests.json`, 99 calculator + 80 formula), trained planner + writer
(`train`, seed 0), bwrap execution on 20 cases per capability. Commands: `e2e_pipeline.py run --out $R --tag <tag> ...`.

| run | understander | pass | root of failures | notes |
|---|---|---|---|---|
| gold_net | gold spec | 179/179 | — | planner 0.39 ms, writer 0.09 ms per request |
| inject_net | gold + 1 injected fault per request | 37.4% | understander 32 · planner 35 · writer 45 | every root = injected stage; CF repair u 0.29 p 0.31 w 0.40 |
| q3b_net | Qwen2.5-Coder-3B q8, 4 shots, JSON grammar | 72.6% | understander 49 | planner + writer 100% on the specs they received |
| q3b_sb2 | same + send-back ≤ 2 rounds | **78.8%** | understander 38 | 28 triggered, 12 repaired, 3 still flagged; 0.87 s per request |

By split with send-back: calculator 97/99 (one 15/15, test_comb 40/40, len4 24/25, len5 14/15, open 4/4); formula
44/80 (test_same 29/40, calls4 12/25, calls5 3/15).

The 38 remaining failures, read one by one (`traces_q3b_sb2.jsonl`):

| kind | n | example |
|---|---|---|
| formula read as a calculator (expr → dispatch) | 10 | x110 "(a mais a) menos (a hipotenusa de catetos b e b)" → `{"main":"dispatch","calls":["add","hyp"]}` |
| wrong tree (nesting, operand order, dropped/added call) | 23 | x127 "o produto de ((b dividido por a) vezes a) e a" → `mul(div(b, a), a)` (one `mul` dropped) |
| name outside the vocabulary that survived 2 send-backs | 3 | x152 `max(...)`, x177 `rem(a, b)` and `b * a` |
| calculator item read wrong | 2 | d075 hipotenusa → `sqrt`; d085 calculator → formula |

Conclusion: the belt below the understander is not the bottleneck (planner and writer 100% on whatever spec they
get). The 3B reads nested Portuguese formulas into trees badly, worst on 4–5 calls. Changing the send-back wording
did not change the world-switch (identical 78.8%, 175/179 identical answers) — the 3B does not use the message.
That is his #34: fine-tune a small model to do this transpilation. Running next (see the ledger).

### Judges (V1, his design #36–#37)

`judge --sonnet-mode all --n 30` on two trace sets: 30 Sonnet calls each, 0 hook blocks, 0 failed calls, Sonnet
6–20 s per call. Sample = 20 failures + 10 passes per set. Truth = execution + oracle blame (ranks 1–2).

| set | judge | pass agree | capability agree | root-stage agree | writer function agree | false alarm |
|---|---|---|---|---|---|---|
| natural (`traces_q3b_sb2`, all roots understander) | Qwen 3B | 0.50 | 0.31 | 0.05 | — | 0.10 |
| | **final (Sonnet over Qwen)** | **1.0** | **1.0** | **1.0** | — | **0.0** |
| injected (`traces_inject_net`, roots u 7 · p 7 · w 6) | Qwen 3B | 0.43 | 0.56 | 0.30 | 0.0 | 0.20 |
| | **final (Sonnet over Qwen)** | **1.0** | **1.0** | **1.0** | **1.0** | **0.0** |

Fixes (accepted only if a rerun of the belt with the fix raises the score, no regression, right stage):

| set | Qwen fixes proposed / accepted | final fixes proposed / accepted / repaired |
|---|---|---|
| natural | 4 / 0 | 20 / 20 / 20 (all understander: corrected spec) |
| injected | 9 / 1 | 20 / 19 / 19 (understander 7, planner 6, writer 6; d091 planner fix did not raise the score → rejected) |

Sonnet's `qwen_ok` vs truth: 0.93 (natural), 0.87 (injected). It never approved a wrong Qwen verdict; its misses are
the other way (called a correct Qwen verdict wrong, 2 + 4 cases), which costs a Sonnet correction, not a wrong label.

Reading:
- The 3B first judge does not work as a judge here: on natural failures it blames no stage in 17 of 20 (it reports
  the capabilities as correct even when execution shows the wrong value). So `escalate` cannot rely on the 3B
  noticing its own doubt; escalation has to come from machine facts (exec flags, validator flags) or from Sonnet
  on every failure.
- Sonnet as the final judge, holding the RUBRIC and the execution facts (no expected values), reproduced the oracle
  blame on all 60 traces, including which writer function was wrong, and its fixes repaired 39 of 40 failures.
  These 39 rows plus the one accepted Qwen fix (planner) are the 40 rows of `replay_nat.jsonl` / `replay_inj.jsonl`:
  stage input → corrected output, on-policy.
- Hardening after the run: `claude -p` is now called with `--tools ""` (the judge's cwd sits inside the run dir,
  next to `requests.json`; the 6–20 s single-turn latencies say no file was read in these runs, but the door is
  closed now).
- Limit: n = 30 per set, one seed, templated requests; Sonnet saw the 20 truth cases (judges could only miss a bug
  that shows on other inputs, as the oracle would).

### Student understander (#34): Qwen2.5-Coder-0.5B + LoRA, trained on generator gold

`synth/student_understander.py` (r 16, 8.8M LoRA params, 1 epoch, greedy, batch 1). s3 = trees of 1–3 calls
(as the planner); s5 = s3 + 3000 random 4–5-call trees, every test tree banned. Two sentence frames held out.
Set A = the 179 templated requests. Set B (`para/`) = 174 of them rewritten by Sonnet the way a person types in a
chat, checked by a second Sonnet pass against the gold (1 of 175 rejected; open split left out); gold spec on B:
174/174. Belt = real planner + writer + bwrap; pass = all 20 cases right.

| understander | A belt pass (179) | A on the 174 common ids | B belt pass (174) | lost A→B | ms / request |
|---|---|---|---|---|---|
| Qwen-3B few-shot + grammar + 2 send-back | 78.8% | 78.2% | 74.7% | −3.5 | 866 |
| student s3 | 93.9% | 95.4% | 81.0% | −14.4 | 1806 |
| student s5 | **97.8%** | 99.4% | **85.6%** | −13.8 | 1822 |

Predictions (written before the run): s3 test_same ≥ 0.9 exact PASS (40/40); s3 calls4/5 ≤ 0.5 FAIL (0.92 / 0.67:
the pretrained model generalises depth far better than predicted); s5 calls4/5 ≥ 0.9 PASS (25/25, 14/15); unseen
frame within 10 points PASS (56/56 vs 118/119); open 0–2/4 PASS (1/4); s5 drops ≥ 15 points on B: exact PASS
(0.99 → 0.82), belt −13.8.

Reading:
- In the template world the student beats the 3B by 19 points; all of the 3B's deep-tree failures go away.
- Free phrasing costs the student 14 points, the 3B only 3.5: the student learned the templates, the 3B reads
  language. The student is still 11 points above the 3B on B.
- s5 failures on B, by kind: `%` read as `mod` (training only says "por cento"); requests written as steps
  ("pegue X e divida o resultado por a") answered with comma-separated pieces or wrong arity; indirection
  ("X% de a, onde X é ...") answered with a variable `x`; vague requests ("calculadora científica", A open split,
  1/4) where the 3B uses world knowledge (4/4).
- Exact match is stricter than the belt: Sonnet's rewrite sometimes swaps operands of avg/maxx/mul, so a correct
  reading of the rewrite differs from the gold string but executes the same. Belt numbers are the fair ones.
- Latency: the student runs in transformers eager mode (1.8 s); llama.cpp/GGUF is needed for #41.
- Not like-for-like as models: the student trained on 14–17k pairs from the same generator; the 3B is prompted.
  Like-for-like as belt options on the same request sets.

#### s5p: augmenting with Sonnet rewrites of training texts (FAILED-ADAPT)

His idea: make the student read free language, not templates. s5p = s5 + 593 Sonnet rewrites of s5 seed-1
**training** texts (`paraphrase_requests.py --from-train 600 --seed 1`, second Sonnet pass checks each rewrite
against its gold; 7 rejected), each repeated 3×, same 523 optimizer steps as s5 (like-for-like budget). The guard
in `student_understander.py` drops any extra pair whose text or gold equals a set A/B request or whose frame is held
out (0 dropped).

| arm | A belt (179) | A common ids (174) | B belt (174) | B exact | A unseen frame | A seen frame |
|---|---|---|---|---|---|---|
| s5 | 97.8% | 99.4% | 85.6% | 0.82 | 56/56 | 119/119 |
| s5p | 89.4% | 90.2% | 89.1% | 0.862 | **38/56** | 119/119 |

By split (belt pass), formulas vs calculators:

| split | kind | s5 A | s5p A | s5 B | s5p B |
|---|---|---|---|---|---|
| test_same | formula | 40/40 | 40/40 | 25/40 | **36/40** |
| calls4 | formula | 25/25 | 25/25 | 19/24 | **24/24** |
| calls5 | formula | 14/15 | 15/15 | 12/15 | 14/15 |
| test_comb | calculator | 40/40 | **34/40** | 40/40 | 36/40 |
| len4 | calculator | 25/25 | **21/25** | 25/25 | 22/25 |
| len5 | calculator | 15/15 | **11/15** | 14/15 | 10/15 |
| one | calculator | 15/15 | **12/15** | 14/15 | 13/15 |

Predictions (written before the run): B belt ≥ 0.90 and ≥ +10 over s5 — FAIL (0.891, +3.5); A loses ≤ 2 points —
FAIL (−8.4).

Root cause (found by reading the A failures, then counted):
- In the 593 rewrites, **0/139 calculator rewrites mention "main"; 454/454 formula rewrites do** (the formula gold
  is "main receives a and b and returns ...", the GEN prompt says so). In the original s5 training set, 0/3595
  calculator texts mention "main" either: the only calculator frame that says "main" is the held-out one
  ("preciso de uma função main que escolha entre ...").
- The student learned the shortcut "the word main ⇒ formula". On the held calculator frame it now writes a formula
  (`dispatch(...)`-like expressions) instead of a calculator: unseen frame 56/56 → 38/56. Rewrites that mention
  "main" for a calculator in set A/B fail the same way.
- Not a code bug: the augmentation itself carries a spurious correlation. Formula reading got much better
  (B formula splits +18 passes), calculator reading got worse.

Adaptation (s5p2, rule 7): ~300 extra Sonnet rewrites of **calculator** training items, about half describing a
main that receives the operation and the numbers, never "escolha entre", never starting with "preciso de uma
função main" (`paraphrase_requests.py --world dispatch --gen-extra ... --name disp_main`): 300 / 300 kept by the
check pass, 132 mention main, 0 say "escolha entre", 20 Sonnet calls.
Review (Sonnet agent, verified by hand) before the GPU run:
- the held-frame guard was a case-sensitive prefix test. It now drops any extra text that contains a held frame
  head, "preciso de uma função main" or "escolha entre", in any case and position. This drops 1 of 593 para_train
  pairs (a formula rewrite starting "preciso de uma função main(a, b)"; s5p trained on it) and 1 of 300 disp_main
  pairs;
- `replay_labels.py` (V2) now labels a passing trace with the canonical bridged spec, only when it is in the
  request's world (d010: a 1-op calculator answered as a formula passes the belt); it prefers the final judge's fix
  over the Qwen judge's, records its real source, and requires the fix's input text to equal the trace text.
  Smoke on s5p's set-A traces + replay_nat: 160 passes = 159 labels + 1 wrong world, 2 fixes, 161/161 labels equal
  gold.
Known remaining bias: 132 calculator rewrites vs 454 formula rewrites mention main, so "main ⇒ formula" is weaker
but still present; the cue that separates them is `main(a, b)` (formula) vs "main recebe a operação" (calculator).
Caveat: this fix was found by looking at set-A failures, so the held frame is no longer a blind test for s5p2;
set B and the split numbers stay the fair ones.

s5p2 predictions (written before the run, 10-26): A unseen frame ≥ 54/56; A belt ≥ 0.97; B belt ≥ 0.90; B formula
splits keep s5p's gain (test_same ≥ 34/40).
