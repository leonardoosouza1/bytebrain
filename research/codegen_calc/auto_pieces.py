"""evaluate() with the decomposition, the helper code and the tests all written by the frozen model.

pieces_native.py used a hand-made split (parser_ref.ORDER) and reference-derived cases per piece. Here:
  1. plan: the model writes helper signatures + docstrings (module-level functions, no bodies);
  2. code: k samples per helper with the whole plan as context;
  3. tests: the model writes test lines per helper from its docstring; one test per input = majority answer,
     weighted by its vote count (codet_rerank's "votes" rule);
  4. pick: coordinate passes - each helper re-picked with the other helpers fixed at their current pick;
  5. plan choice: assembled evaluate scored on its 3 docstring examples + model-written evaluate tests.
No reference code or reference-derived case enters the loop. Hidden seeds 0,1 and the grammar test
(verify_leakfree) are computed only for the report.

Usage: auto_pieces.py --out DIR [--plans 6] [--k 32] [--t 6] [--url http://127.0.0.1:8140]
"""

import argparse
import ast
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import calc_native as cn
import codet_rerank as R
import evaluate as ev
import judge
import pieces_native as P
import spec
import verify_leakfree as V

ENTRY = spec.SPEC[-1]
assert ENTRY["name"] == "evaluate"
ASK_PLAN = ("Plan a recursive-descent implementation of this function as small module-level helper functions "
            "(no classes, no global mutable state). Write ONLY the helper signatures, each with a docstring that "
            "states its inputs, its output and one example call with its result, and body `...`. "
            "Then write evaluate itself with a real body that calls the helpers.\n\n```python\n{stub}```")
CONVENTION = (" Convention: a tokenize(expr) helper returns a list of string tokens; every parse helper takes "
              "(tokens, i) and returns (value, new_i).")
ASK_IMPL = ("Here is the plan of a calculator expression evaluator:\n\n```python\n{plan}```\n\n"
            "Implement the helper `{name}` exactly as its docstring says. Use only the standard library.")


def complete_many(pool, url, text, n, max_new, seed0, temperature=0.8):
    return list(pool.map(lambda s: P.complete(url, text, max_new, temperature if s else 0.0, seed0 + s), range(n)))


def parse_plan(text):
    """-> (plan source, [helper names], evaluate source) or None."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    helpers = [f for f in funcs if f.name != "evaluate" and ast.get_docstring(f)]
    main = [f for f in funcs if f.name == "evaluate"]
    if not (2 <= len(helpers) <= 10) or not main:
        return None
    lines = text.splitlines()
    seg = lambda f: "\n".join(lines[f.lineno - 1 - len(f.decorator_list):f.end_lineno])  # noqa: E731
    plan = "import math\n\n" + "\n\n".join(seg(f) for f in helpers)
    return plan, [f.name for f in helpers], seg(main[0])


def helper_stub(plan_src, name):
    tree = ast.parse(plan_src)
    f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = plan_src.splitlines()
    head = "\n".join(lines[f.lineno - 1:f.body[0].end_lineno])  # def line .. docstring
    return head + "\n"


def extract_function(code, name):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    lines = code.splitlines()
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return "\n".join(lines[n.lineno - 1:n.end_lineno])
    return None


def write_tests(pool, url, stub, name, t, seed0):
    text = P.CHAT.format(user=R.ASK_TESTS.format(stub=stub, name=name), prefix="")
    votes, seen = defaultdict(Counter), {}
    for o in complete_many(pool, url, text, t, 400, seed0):
        for tc in {repr(x): x for x in R.parse_tests(name, o, containers=True)}.values():
            seen[repr(tc)] = tc
            votes[repr(tc[0])][repr(tc)] += 1
    tests, weight = [], []
    for c in votes.values():
        rt, w = c.most_common(1)[0]
        tests.append(seen[rt])
        weight.append(w)
    return tests, weight


def score(module, name, tests, weight):
    if not tests:
        return 0.0, 0
    o = judge.run_jobs_isolated([{"code": module, "fname": name, "calls": [list(a) for a, _ in tests]}])[0]
    if o["load"] != "ok":
        return -1.0, 0
    ok = [judge.matches(e, g) for (_, e), g in zip(tests, o["out"])]
    return sum(w for w, m in zip(weight, ok) if m), sum(ok)


def score_many(modules, name, tests, weight):
    if not tests:
        return [(0.0, 0)] * len(modules)
    res = judge.run_jobs_isolated([{"code": m, "fname": name, "calls": [list(a) for a, _ in tests]} for m in modules])
    out = []
    for o in res:
        if o["load"] != "ok":
            out.append((-1.0, 0))
            continue
        ok = [judge.matches(e, g) for (_, e), g in zip(tests, o["out"])]
        out.append((sum(w for w, m in zip(weight, ok) if m), sum(ok)))
    return out


def assemble(helpers, picks, main_src):
    return "import math\n\n" + "\n\n".join(picks[h] for h in helpers) + "\n\n" + main_src + "\n"


def report_scores(module):
    hid = cn.hidden_cases(ENTRY)
    o = judge.run_jobs_isolated([{"code": module, "fname": "evaluate", "calls": [list(a) for a in hid]}])[0]
    hidden = o["load"] == "ok" and all(judge.matches(spec.expected(ENTRY, a), g) for a, g in zip(hid, o["out"]))
    ok, _ = V.check(module, ENTRY, V.grammar_cases(300))
    return hidden, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plans", type=int, default=6, help="parsable plans to try")
    ap.add_argument("--k", type=int, default=32, help="samples per helper")
    ap.add_argument("--t", type=int, default=6, help="test-writing samples per function")
    ap.add_argument("--passes", type=int, default=2, help="coordinate re-pick passes over the helpers")
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--convention", action="store_true",
                    help="fix the helper interface (token list + index -> (value, new_i)); the model still plans")
    ap.add_argument("--temp-helper", type=float, default=0.8,
                    help="helper sampling temperature (0.8 collapsed: 12/12 identical with the plan in context)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pool = ThreadPoolExecutor(32)
    t0 = time.perf_counter()
    stub = spec.stub(ENTRY)
    # evaluate's own tests: docstring examples (weight = t, they are given) + model-written
    ev_tests, ev_w = write_tests(pool, args.url, stub, "evaluate", args.t, 7000)
    for a in ENTRY["examples"]:
        ev_tests.append((a, spec.expected(ENTRY, a)))
        ev_w.append(args.t)
    print(f"evaluate tests: {len(ev_tests)}", flush=True)
    ask = ASK_PLAN + (CONVENTION if args.convention else "")
    plan_text = P.CHAT.format(user=ask.format(stub=stub), prefix="import math\n\n")
    raw = complete_many(pool, args.url, plan_text, 4 * args.plans, 1200, 8000)
    plans = [p for p in (parse_plan("import math\n\n" + r) for r in raw) if p][:args.plans]
    print(f"plans parsable: {len(plans)}/{len(raw)}", flush=True)
    results = []
    for pi, (plan_src, helpers, main_src) in enumerate(plans):
        cands, tests = {}, {}
        for h in helpers:
            hs = helper_stub(plan_src, h)
            text = P.CHAT.format(user=ASK_IMPL.format(plan=plan_src, name=h), prefix="import math\n\n" + hs)
            outs = complete_many(pool, args.url, text, args.k + 1, 700, 9000 + 100 * pi, args.temp_helper)
            codes = [extract_function(ev.clean("import math\n\n" + hs + o), h) for o in outs]
            cands[h] = [c for c in dict.fromkeys(codes) if c]
            tests[h] = write_tests(pool, args.url, hs, h, args.t, 10000 + 100 * pi)
        if not all(cands[h] for h in helpers):
            print(f"plan {pi}: a helper has no parsable sample", flush=True)
            continue
        picks = {h: cands[h][0] for h in helpers}  # greedy sample first
        for _ in range(args.passes):
            for h in helpers:
                mods = [assemble(helpers, dict(picks, **{h: c}), main_src) for c in cands[h]]
                sc = score_many(mods, h, *tests[h])
                picks[h] = cands[h][max(range(len(sc)), key=lambda i: sc[i])]
        module = assemble(helpers, picks, main_src)
        ev_score, ev_pass = score(module, "evaluate", ev_tests, ev_w)
        hidden, grammar = report_scores(module)
        r = {"plan": pi, "helpers": helpers, "n_cands": {h: len(cands[h]) for h in helpers},
             "n_tests": {h: len(tests[h][0]) for h in helpers}, "eval_test_score": ev_score,
             "eval_tests_passed": ev_pass, "hidden_ok": hidden, "grammar": grammar, "module": module,
             "plan_src": plan_src}
        results.append(r)
        print(f"plan {pi}: helpers {len(helpers)} {helpers}  eval tests {ev_pass}/{len(ev_tests)} "
              f"(score {ev_score})  hidden {hidden:d}  grammar {grammar}/300", flush=True)
        with open(os.path.join(args.out, "auto_pieces.json"), "w") as fh:
            json.dump({"results": results}, fh, indent=1)
    if results:
        best = max(results, key=lambda r: r["eval_test_score"])
        print(f"picked plan {best['plan']} by model tests: hidden {best['hidden_ok']:d} grammar {best['grammar']}/300"
              f"   (oracle best grammar {max(r['grammar'] for r in results)}/300)  wall "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        with open(os.path.join(args.out, "evaluate_auto.py"), "w") as fh:
            fh.write(best["module"])


if __name__ == "__main__":
    main()
