# RUBRIC — the rule the judges follow (ledger #36, #37)

Loaded verbatim into the Qwen and Sonnet judge prompts by `synth/e2e_pipeline.py`. Change it here only.
His words: "uma regra para o Sonnet seguir... ele tem que dizer isso seguindo uma verdade, que é o quê? É a regra.
O quanto que o modelo errou do quanto que foi pedido."

## 0. Truth order

1. Execution results in the evidence (bwrap) are facts. Never contradict them.
2. This rubric decides how facts become a verdict.
3. Your opinion only fills what 1 and 2 leave open (for example: which capabilities a vague request asks for).

## 1. The belt you are judging

```
request (pt-BR) -> UNDERSTANDER -> spec JSON -> PLANNER -> plan symbols -> WRITER -> function bodies -> ASSEMBLER -> code
```

Each stage has a contract. A stage is wrong only when its output breaks its contract FOR THE INPUT IT RECEIVED.
A stage that faithfully processes a wrong input is NOT to blame. Blame the earliest stage that broke its contract.

| stage | contract (what "correct" means) |
|---|---|
| understander | the spec asks for exactly what the request asks: every requested capability present, nothing changed, formula tree as the request says, test expectations right |
| planner | the plan implements the spec: one `disp:<op>` per spec call (dispatch) or the same call tree (expr), and a `def:<x>` for every helper any written function calls |
| writer | each function body does what its name says (`add` returns a + b; `hyp` returns sqrt(a² + b²); `div` returns None when b == 0) |
| assembler | layout only: defs, then `main`, dispatch lines / return expression in the right place |

Output formats (a fix must use them exactly):

- spec, calculator: `{"main": "dispatch", "calls": ["add", "sqrt"]}` → `main(op, a, b=0)` dispatches on the op id.
- spec, one formula: `{"main": "expr", "expr": "avg(mul(a, b), b)"}` → `main(a, b)` returns it. `expr` is a STRING of
  nested calls of add sub mul div mod maxx hyp avg pct, exactly two arguments each, leaves `a` and `b`.
- plan: symbols separated by spaces, ending with `<eos>`. Calculator: `def:<op>` (a helper, before its first use) and
  `disp:<op>` (one dispatch line). Formula: `def:<op>`, then the tree in prefix order as `call:<op>`, `leaf:a`, `leaf:b`.
- writer: one full Python `def`, its name is the op id, signature `(a, b=0)`.

## 2. What to output (first judge)

One JSON object:

```json
{"capabilities": [{"name": "<capability asked in the REQUEST, in its words>", "status": "correct|wrong|missing|extra"}],
 "blame": [{"stage": "understander|planner|writer|assembler", "function": "<function name or empty>", "evidence": "<the fact that shows it>"}],
 "fix": null,
 "justification": "<two sentences at most>"}
```

- `capabilities`: list what the REQUEST asks (read the request, not the spec). Name each one by its operation id
  (add sub mul div mod maxx sqrt neg abs square pow fact hyp avg pct) for a calculator request, or `formula` for a
  request that asks main(a, b) to return one formula. A vague request ("calculadora básica") asks for the operations
  such a calculator must have. `correct` = the execution facts show the right value for every case listed.
  `wrong` = present but some test fails. `missing` = not implemented. `extra` = implemented but not requested
  (allowed, not an error).
- `blame`: empty when every requested capability is correct. Otherwise the root-cause stage first, then any other
  stage that independently broke its contract.
- `fix`: one minimal fix for the root-cause stage, in that stage's own output format, or null. A fix must change
  the stage's output: a replacement equal to the current SPEC, PLAN or function is not a fix.
  - understander: `{"stage": "understander", "function": "", "replacement": "<the corrected spec JSON as a string>"}`
  - planner: `{"stage": "planner", "function": "", "replacement": "<corrected plan symbols separated by spaces>"}`
  - writer: `{"stage": "writer", "function": "<name>", "replacement": "<the corrected full def, Python>"}`

## 3. Score (computed by code, never by the judge)

`request_score = correct / (correct + wrong + missing)` over the capabilities you listed. `extra` does not count.
"o pedido foi 80%" = 4 of 5 requested capabilities correct.

## 4. Final judge (Sonnet)

You receive the same evidence plus the first judge's verdict. Check each item of it against sections 0–3:

```json
{"qwen_ok": true,
 "checks": [{"item": "capabilities[0] | blame[0] | fix | ...", "agree": true, "why": "<one sentence>"}],
 "corrected": null}
```

- `qwen_ok` = true only if every item agrees. If the first verdict is good, say so and stop: `corrected` = null.
- If any item disagrees, `corrected` = the full verdict in the section 2 format, fixed. Do not restate items you
  agree with differently.
- A verdict that contradicts an execution fact is always wrong.

## 5. Reference meanings (what each operation must return)

add a+b · sub a-b · mul a*b · div a/b, None if b == 0 · mod a%b, None if b == 0 · maxx the larger of a and b ·
sqrt a**0.5, None if a < 0 · neg -a · abs |a| · square a*a · pow a**b (b >= 0) · fact a! (1 for a <= 1) ·
hyp sqrt(a² + b²) · avg (a+b)/2 · pct a*b/100. Floats are rounded to 6 decimals in the facts.
Inside a formula, when div, mod or sqrt return None, the next operation raises `!TypeError`: that exception IS the
right result of the requested formula on that case, not a code error.

## 6. Pitfalls (seen or expected)

- A failing call with `NameError` can be a planner error (helper never defined) OR a writer error (body calls the
  wrong name). Look at the body: if the called name is the right helper for this function, blame the planner.
- An extra function is not an error. A dispatch line for an op that was not requested is `extra`.
- A request for ONE operation does not fix the interface: `main(op, a, b)` dispatch and `main(a, b)` returning that
  operation both satisfy it.
- Wrong test expectation in the spec is an understander error even when the code is right.
- Portuguese "módulo" can mean absolute value (`abs`) or remainder (`mod`); judge by the rest of the sentence.
