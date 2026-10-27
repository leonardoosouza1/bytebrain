"""Scientific ops for the multi-function calculator (ledger #5 / #22 gap, 10-27): sin cos tan exp log, written
without imports (Taylor / atanh series with range reduction), in the DEFS format of multifn.py:
op -> (body lines, direct deps). cos / tan / exp call other ops, so they also test 'functions that depend on
functions'.

Not wired into multifn yet: multifn.py is imported by the running #7 / #3 / #12 GPU queue, and adding ops changes
OPS and with it every checkpoint's symbol table. Wiring goes behind a flag once the queue is done.

usage: sciops.py check     (accuracy vs math on random inputs; CASES through the real bwrap runner of multifn)
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import multifn as MF  # noqa: E402

TWO_PI, HALF_PI, LN2 = "6.283185307179586", "1.5707963267948966", "0.6931471805599453"
SCI_DEFS = {
    "sin": ([f"x = a % {TWO_PI}",
             "if x > 3.141592653589793:",
             f"    x = x - {TWO_PI}",
             "t = x",
             "s = x",
             "for i in range(1, 14):",
             "    t = -t * x * x / ((2 * i) * (2 * i + 1))",
             "    s = s + t",
             "return s"], []),
    "cos": ([f"return sin(add(a, {HALF_PI}))"], ["sin", "add"]),
    "tan": (["c = cos(a)",
             "if c == 0:",
             "    return None",
             "return div(sin(a), c)"], ["sin", "cos", "div"]),
    "exp": (["x = a",
             "n = 0",
             "while x > 0.5 or x < -0.5:",
             "    x = x / 2",
             "    n = n + 1",
             "t = 1",
             "s = 1",
             "for i in range(1, 20):",
             "    t = t * x / i",
             "    s = s + t",
             "for _ in range(n):",
             "    s = mul(s, s)",
             "return s"], ["mul"]),
    "log": (["if a <= 0:",
             "    return None",
             "x = a",
             "n = 0",
             "while x > 2:",
             "    x = x / 2",
             "    n = n + 1",
             "while x < 0.5:",
             "    x = x * 2",
             "    n = n - 1",
             "z = (x - 1) / (x + 1)",
             "t = z",
             "s = 0",
             "for k in range(30):",
             "    s = s + t / (2 * k + 1)",
             "    t = t * z * z",
             f"return 2 * s + n * {LN2}"], []),
}
REF = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
       "log": lambda a: math.log(a) if a > 0 else None}


def all_defs():
    d = dict(MF.DEFS)
    assert not set(d) & set(SCI_DEFS)
    d.update(SCI_DEFS)
    return d


def program(spec, defs):
    """Same layout as multifn.assemble (dep closure in dependency order, then main), over the extended table."""
    order, seen = [], set()

    def visit(o):
        if o not in seen:
            seen.add(o)
            for x in defs[o][1]:
                visit(x)
            order.append(o)
    for o in spec:
        visit(o)
    src = "".join(f"def {o}(a, b=0):\n" + "".join(f"    {l}\n" for l in defs[o][0]) for o in order)
    src += "def main(op, a, b=0):\n" + "".join(f"    if op == \"{o}\":\n        return {o}(a, b)\n" for o in spec)
    return src + "    return None\n", order


def check():
    defs = all_defs()
    src, order = program(tuple(SCI_DEFS), defs)
    g = {}
    exec(src, g)                                   # hand-written table above, not model output
    r = random.Random(0)
    worst = {}
    for o, f in REF.items():
        xs = [r.uniform(-30, 30) for _ in range(3000)] + list(range(-50, 51))
        if o == "exp":
            xs = [x for x in xs if x < 700]
        if o == "tan":
            xs = [x for x in xs if abs(math.cos(x)) > 1e-3]
        if o == "log":
            xs = [abs(x) * 10 ** r.randint(-8, 8) for x in xs] + [0, -1]
        err = 0.0
        for x in xs:
            want, got = f(x), g["main"](o, x)
            if want is None or got is None:
                assert want is None and got is None, (o, x, want, got)
                continue
            err = max(err, abs(got - want) / max(1.0, abs(want)))
        worst[o] = err
        assert err < 1e-9, (o, err)
    print("max rel/abs error vs math:", {k: f"{v:.1e}" for k, v in worst.items()})
    print("dep order for all 5:", order)
    # the real grader path: bwrap runner, CASES, canon rounding to 6 decimals
    want = [MF.RUNNER and None]
    rows = MF.run_many([[src, list(SCI_DEFS)]])[0]
    exp_rows = []
    for o in SCI_DEFS:
        for a, b in MF.CASES:
            v = REF[o](a)
            exp_rows.append(None if v is None else round(v, 6))
    got = [None if v == "None" else float(v) for v in rows]
    assert len(got) == len(exp_rows) and all(
        (w is None and x is None) or (w is not None and x is not None and abs(w - x) < 1.5e-6)
        for w, x in zip(exp_rows, got)), list(zip(exp_rows, got))[:10]
    print("bwrap runner on CASES:", len(got), "values match math (6 decimals)")
    old = max(len("".join(f"    {l}\n" for l in v[0])) for v in MF.DEFS.values())
    new = {o: len("".join(f"    {l}\n" for l in v[0])) for o, v in SCI_DEFS.items()}
    print("body bytes: longest old op", old, "| new", new)
    full, _ = program(tuple(sorted(defs)), defs)
    print("full 20-op program bytes", len(full), "| old full 15-op", len(MF.assemble(MF.plan_of_grouped(tuple(MF.OPS)))))
    del want


if __name__ == "__main__":
    assert sys.argv[1:] == ["check"], __doc__
    check()
