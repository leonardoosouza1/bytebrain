"""#47 check: run each Opus program in bwrap on CASES; compare with my own evaluator of the gold spec
(op meanings written here from scratch, not imported from multifn.DEFS)."""
import ast, json, math, sys
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/pycode/synth")
import e2e_pipeline as E

P = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/saci_code/opus_examples.jsonl"
recs = [json.loads(l) for l in open(P)]

def none_guard(f):
    def g(a, b=0):
        if a is None or b is None:
            raise TypeError
        return f(a, b)
    return g

def _pow(a, b):
    r = 1
    for _ in range(b):          # b < 0 -> 1 (house rule, documented)
        r *= a
    return r

MEAN = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "div": lambda a, b: None if b == 0 else a / b, "mod": lambda a, b: None if b == 0 else a % b,
    "maxx": lambda a, b: a if a > b else b, "sqrt": lambda a, b=0: None if a < 0 else a ** 0.5,
    "neg": lambda a, b=0: -a, "abs": lambda a, b=0: -a if a < 0 else a, "square": lambda a, b=0: a * a,
    "pow": _pow, "fact": lambda a, b=0: math.prod(range(2, a + 1)) if a >= 2 else 1,
    "hyp": lambda a, b: (a * a + b * b) ** 0.5, "avg": lambda a, b: (a + b) / 2,
    "pct": lambda a, b: a * b / 100,
    # novel ops, meaning from the request text
    "minn": lambda a, b: a if a < b else b, "cube": lambda a, b=0: a * a * a,
    "absdiff": lambda a, b: abs(a - b), "idiv": lambda a, b: None if b == 0 else a // b,
}
MEAN = {k: none_guard(v) for k, v in MEAN.items()}

def ev(node, env):
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Call):
        return MEAN[node.func.id](*[ev(x, env) for x in node.args])
    raise ValueError(ast.dump(node))

def fmt(v):
    return repr(round(v, 6) if isinstance(v, float) else v)

def expect(rec, a, b, op=None):
    g = rec["gold"]
    try:
        if g["main"] == "dispatch":
            return fmt(MEAN[op](a, b))
        return fmt(ev(ast.parse(g["expr"], mode="eval").body, {"a": a, "b": b}))
    except TypeError:
        return "!TypeError"

items, want = [], []
for r in recs:
    g = r["gold"]
    if g["main"] == "dispatch":
        calls = [("main", [o, a, b]) for o in g["calls"] for a, b in E.CASES] + [("main", ["nope", 1, 2])]
        w = [expect(r, a, b, o) for o in g["calls"] for a, b in E.CASES] + ["None"]
    else:
        calls = [("main", [a, b]) for a, b in E.CASES]
        w = [expect(r, a, b) for a, b in E.CASES]
    items.append((r["program"], calls)); want.append(w)

out = E.run_many(items)
bad = 0
for r, o, w in zip(recs, out, want):
    if o is None:
        print(r["id"], "MODULE FAILED"); bad += 1; continue
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(o, w)) if x != y]
    n_err = sum(x.startswith("!") for x in o)
    print(r["id"], r["category"], "novel" if r["novel"] else "", "OK" if not diff else f"DIFF {len(diff)}/{len(w)} {diff[:3]}",
          f"errors_on_cases={n_err}")
    bad += bool(diff)
print("records", len(recs), "mismatch", bad)
