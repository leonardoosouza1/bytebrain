"""Leak-free check of a whole calculator module.

(a) every function except evaluate on random-only cases of seeds 10-14 (spec edges excluded: they repeat in every
    seed, so any selection that saw a seed has seen them);
(b) evaluate on grammar-random expressions not in spec.EXPRS (evaluate's random generator only draws from EXPRS).
Expected values always come from calc_ref.

Usage: verify_leakfree.py MODULE.py [--n-expr 300]
"""

import argparse
import json
import random

import judge
import spec

FUNCS = ["sqrt", "sin", "cos", "tan", "log", "ln", "abs"]


def rand_expr(r, depth=0):
    k = r.random()
    if depth > 3 or k < 0.3:
        return r.choice([str(r.randint(0, 20)), f"{r.uniform(0, 10):.{r.choice([1, 2])}f}", "pi", "e"])
    if k < 0.65:
        return rand_expr(r, depth + 1) + r.choice(["+", "-", "*", "/", "^"]) + rand_expr(r, depth + 1)
    if k < 0.75:
        return "(" + rand_expr(r, depth + 1) + ")"
    if k < 0.85:
        return "-" + rand_expr(r, depth + 1)
    return r.choice(FUNCS) + "(" + rand_expr(r, depth + 1) + ")"


def corrupt(r, s):
    """Make an invalid expression: drop/insert a char or an unknown name."""
    k = r.random()
    if k < 0.3 and len(s) > 1:
        j = r.randrange(len(s))
        return s[:j] + s[j + 1:]
    if k < 0.6:
        j = r.randrange(len(s) + 1)
        return s[:j] + r.choice(["(", ")", "+", "*", "..", " 3 ", "x"]) + s[j:]
    return "foo(" + s + ")"


def grammar_cases(n, seed=99):
    r = random.Random(f"grammar-{seed}")
    out, seen = [], set(spec.EXPRS)
    while len(out) < n:
        s = rand_expr(r)
        if r.random() < 0.35:
            s = corrupt(r, s)
        if s not in seen:
            seen.add(s)
            out.append((s,))
    return out


def random_only(entry, seeds=range(10, 15)):
    edge = {repr(c) for c in entry["edge"]}
    seen, out = set(), []
    for s in seeds:
        for c in spec.hidden_cases(entry, seed=s):
            if repr(c) not in edge and repr(c) not in seen:
                seen.add(repr(c))
                out.append(c)
    return out


def check(code, entry, cases):
    o = judge.run_jobs_isolated([{"code": code, "fname": entry["name"], "calls": [list(a) for a in cases]}])[0]
    if o["load"] != "ok":
        return 0, [("load", o["load"])]
    fails = [(a, spec.expected(entry, a), g) for a, g in zip(cases, o["out"])
             if not judge.matches(spec.expected(entry, a), g)]
    return len(cases) - len(fails), fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--n-expr", type=int, default=300)
    args = ap.parse_args()
    code = open(args.module).read()
    report = {"functions": {}}
    for entry in spec.SPEC:
        cases = grammar_cases(args.n_expr) if entry["name"] == "evaluate" else random_only(entry)
        ok, fails = check(code, entry, cases)
        report["functions"][entry["name"]] = {"ok": ok, "n": len(cases), "fails": [repr(x) for x in fails[:5]]}
        if fails:
            print(f"{entry['name']:14s} {ok}/{len(cases)}  e.g. {fails[0]!r}"[:220], flush=True)
    whole = sum(v["ok"] == v["n"] for v in report["functions"].values())
    ev = report["functions"]["evaluate"]
    print(f"functions fully right: {whole}/{len(spec.SPEC)}   evaluate grammar {ev['ok']}/{ev['n']}")
    print(json.dumps({"whole": whole, "evaluate": [ev["ok"], ev["n"]]}))


if __name__ == "__main__":
    main()
