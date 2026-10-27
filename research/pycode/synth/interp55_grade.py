"""bwrap grader + controls + functional leak check for the mini-interpreter family (ledger #55, I6, 10-27).

Grader: generated code runs ONLY inside bwrap (--unshare-all, --clearenv, read-only /usr, no network, no home), one
forked child per module (fresh globals; a module cannot poison the next), SIGALRM 2 s per module, RLIMIT_AS 1 GB,
results compared on the host as canonical strings (interp55_family.canon, same code inside the sandbox).

  unit mode    : module = gold prelude + GOLD siblings + the candidate body for one unit; run that unit's cases.
                 (this is a unit test: the siblings are gold, as in a real repo where the callee already exists)
  program mode : module = every unit from the candidate bodies; run main() cases. No gold code inside.
  accept vs grade: accept cases are the spec examples (selection only); grade cases are hidden and disjoint.

Controls (`controls`):
  positive : every reference unit passes its accept + grade cases in unit mode, and every reference program passes
             main grade in program mode -> must be 100 %.
  negative : mutants of the references (one edit: swap an operator/compare, delete a statement, change a constant,
             swap an error message) -> the fraction caught by the grade cases is the grader's sensitivity.
  floors   : constant body `return None` per unit.
Leak check (`leak`): for every func_eval unit (prompt never trained), run every distinct TRAIN body of the same unit
name in the held variant's context on the grade cases. A train body passing all of them = functional leak (a lookup
could pass it). Reported per unit; the primary per-function metric is reported on the non-leaked subset too.

Usage:
  interp55_grade.py controls --data DIR/data.json --out DIR/controls.json
  interp55_grade.py leak     --data DIR/data.json --out DIR/leak.json
"""

import argparse
import ast
import inspect
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp55_family as F  # noqa: E402

RUNNER = r'''
import json, os, sys, signal, resource
CANON
items = json.load(sys.stdin)
res = []
for src, calls in items:
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            sys.setrecursionlimit(400)
            signal.alarm(2)
            g = {"__name__": "cand"}
            try:
                exec(compile(src, "cand", "exec"), g)
                ok = True
            except BaseException:
                ok = False
            out = []
            if ok:
                for c in calls:
                    try:
                        out.append(canon(eval(c, g)))
                    except BaseException as e:
                        if type(e).__name__ == "InterpError":
                            out.append("!InterpError:" + str(e))
                        else:
                            out.append("!" + type(e).__name__)
            data = json.dumps(out if ok else None).encode()
        except BaseException:
            data = b"null"
        os.write(w, data)
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    os.waitpid(pid, 0)
    try:
        res.append(json.loads(b"".join(chunks).decode()))
    except Exception:
        res.append(None)
print(json.dumps(res))
'''.replace("CANON", inspect.getsource(F.canon))

BWRAP = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
         "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
         "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
         "/usr/bin/python3", "-I", "-S", "-c", RUNNER]


def _run_batch(items):
    try:
        p = subprocess.run(BWRAP, input=json.dumps(items), capture_output=True, text=True,
                           timeout=30 + 3 * len(items))
        out = json.loads(p.stdout)
        assert len(out) == len(items)
        return out
    except (subprocess.TimeoutExpired, json.JSONDecodeError, AssertionError):
        if len(items) == 1:
            return [None]
        h = len(items) // 2
        return _run_batch(items[:h]) + _run_batch(items[h:])


def run_many(items, batch=64, jobs=2):
    """items: [(src, [call strings])] -> [list of canon results | None]."""
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
    with ThreadPoolExecutor(jobs) as ex:
        outs = list(ex.map(_run_batch, chunks))
    return [x for o in outs for x in o]


# ------------------------------------------------------------------------------------------------------- modules
def module_src(row, bodies):
    """bodies: {unit name: body text}; missing units -> gold."""
    gold = {u["name"]: u["body"] for u in row["units"]}
    gold.update(bodies)
    return F.assemble(row["cfg"], gold) + F.HARNESS


def unit_item(row, name, body, which="grade"):
    u = next(x for x in row["units"] if x["name"] == name)
    cases = u[which] if which != "both" else u["accept"] + u["grade"]
    return (module_src(row, {name: body}), [c for c, _ in cases]), [e for _, e in cases]


def program_item(row, bodies, which="grade"):
    u = next(x for x in row["units"] if x["name"] == "main")
    cases = u[which]
    return (F.assemble(row["cfg"], bodies) + F.HARNESS, [c for c, _ in cases]), [e for _, e in cases]


def passed(got, exp):
    return got is not None and len(got) == len(exp) and all(g == e for g, e in zip(got, exp))


def frac(got, exp):
    if got is None:
        return 0.0
    return sum(g == e for g, e in zip(got, exp)) / max(1, len(exp))


def safe_body(text):
    """syntax check as a function body (host side, no execution)."""
    try:
        ast.parse("def f():\n" + "".join("    " + l + "\n" for l in text.rstrip("\n").split("\n")))
        return True
    except SyntaxError:
        return False


# ------------------------------------------------------------------------------------------------------- mutants
def mutants(body, r, k=3):
    """up to k one-edit mutants of a body (text level)."""
    lines = body.rstrip("\n").split("\n")
    swaps = [(" + ", " - "), (" - ", " + "), (" * ", " + "), (" // ", " / "), (" / ", " // "), ("==", "!="),
             ("!=", "=="), (">=", ">"), ("< ", "<= "), ("i + 1", "i"), ("i += 1", "i += 2"), ("'syntax error'",
             "'bad token'"), ("'unknown variable'", "'syntax error'"), ("'division by zero'", "'syntax error'"),
             ("return None", "return 0"), ("[0]", "[1]"), ("'+'", "'-'"), ("'-'", "'+'"), ("prec[op] + 1", "prec[op]"),
             ("if err:", "if not err:"), ("'end'", "'op'"), ("str(value)", "repr(value)")]
    out = []
    for _ in range(40):
        if len(out) >= k:
            break
        kind = r.random()
        ls = list(lines)
        j = r.randrange(len(ls))
        if kind < 0.7:
            cand = [(a, b) for a, b in swaps if a in ls[j]]
            if not cand:
                continue
            a, b = r.choice(cand)
            ls[j] = ls[j].replace(a, b, 1)
        else:
            if len(ls) < 3 or ls[j].rstrip().endswith(":"):
                continue
            ind = len(ls[j]) - len(ls[j].lstrip())
            ls[j] = " " * ind + "pass"
        t = "\n".join(ls) + "\n"
        if t != body and safe_body(t) and t not in out:
            out.append(t)
    return out


def controls(data, out, jobs):
    V = data["variants"]
    t0 = time.time()
    items, exps, tags = [], [], []
    for k, row in V.items():
        for u in row["units"]:
            it, ex = unit_item(row, u["name"], u["body"], "both")
            items.append(it)
            exps.append(ex)
            tags.append(("pos_unit", k, u["name"]))
        it, ex = program_item(row, {u["name"]: u["body"] for u in row["units"]})
        items.append(it)
        exps.append(ex)
        tags.append(("pos_prog", k, "main"))
    r = random.Random(0)
    for k in sorted(V)[:120]:
        row = V[k]
        for u in row["units"]:
            for m in mutants(u["body"], r, 2):
                it, ex = unit_item(row, u["name"], m, "grade")
                items.append(it)
                exps.append(ex)
                tags.append(("mut_unit", k, u["name"]))
                it, ex = program_item(row, {**{x["name"]: x["body"] for x in row["units"]}, u["name"]: m})
                items.append(it)
                exps.append(ex)
                tags.append(("mut_prog", k, u["name"]))
            it, ex = unit_item(row, u["name"], "return None\n", "grade")
            items.append(it)
            exps.append(ex)
            tags.append(("const_unit", k, u["name"]))
    got = run_many(items, jobs=jobs)
    res = {}
    fails = []
    for (kind, k, name), g, e in zip(tags, got, exps):
        d = res.setdefault(kind, {"n": 0, "pass": 0})
        d["n"] += 1
        p = passed(g, e)
        d["pass"] += p
        if kind.startswith("pos") and not p and len(fails) < 5:
            fails.append({"variant": k, "unit": name, "got": g and g[:3], "exp": e[:3]})
    for d in res.values():
        d["rate"] = round(d["pass"] / max(1, d["n"]), 4)
    res["mutant_caught_unit"] = round(1 - res["mut_unit"]["rate"], 4)
    res["mutant_caught_prog"] = round(1 - res["mut_prog"]["rate"], 4)
    res["pos_fail_examples"] = fails
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def train_bodies_by_name(data):
    by = {}
    for u in data["train_units"]:
        by.setdefault(u["name"], [])
        if u["body"] not in by[u["name"]]:
            by[u["name"]].append(u["body"])
    return by


def leak(data, out, jobs):
    V = data["variants"]
    by = train_bodies_by_name(data)
    t0 = time.time()
    rows = []
    for fe in data["func_eval"]:
        row = V[fe["variant"]]
        u = next(x for x in row["units"] if x["prompt"] == fe["prompt"])
        cands = by.get(u["name"], [])
        items, exps = [], []
        for b in cands:
            it, ex = unit_item(row, u["name"], b, "grade")
            items.append(it)
            exps.append(ex)
        got = run_many(items, jobs=jobs) if items else []
        ok = [passed(g, e) for g, e in zip(got, exps)]
        best = max((frac(g, e) for g, e in zip(got, exps)), default=0.0)
        rows.append({"prompt_sha": F.hashlib.sha1(fe["prompt"].encode()).hexdigest()[:12], "unit": u["name"],
                     "variant": fe["variant"], "n_train_bodies": len(cands), "n_equiv": sum(ok),
                     "best_case_frac": round(best, 4)})
    n_leak = sum(r["n_equiv"] > 0 for r in rows)
    by_unit = {}
    for r in rows:
        d = by_unit.setdefault(r["unit"], {"n": 0, "leaked": 0})
        d["n"] += 1
        d["leaked"] += r["n_equiv"] > 0
    res = {"func_eval": len(rows), "leaked": n_leak, "by_unit": by_unit,
           "text_equal_body": 0, "wall_s": round(time.time() - t0, 1), "rows": rows}
    trb = {F.body_key(u["body"]) for u in data["train_units"]}
    for fe in data["func_eval"]:
        u = next(x for x in V[fe["variant"]]["units"] if x["prompt"] == fe["prompt"])
        res["text_equal_body"] += F.body_key(u["body"]) in trb
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["controls", "leak"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    data = json.load(open(a.data))
    {"controls": controls, "leak": leak}[a.cmd](data, a.out, a.jobs)


if __name__ == "__main__":
    main()
