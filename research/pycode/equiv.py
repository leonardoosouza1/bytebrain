"""Functional equivalence on held-out self-contained functions: the truth signal "does the body do what the real
one does", graded as the share of fuzz inputs where the model's output equals the real function's output.

1. Reference (cached): each held-out self-contained function runs on N_IN deterministic fuzz inputs in bwrap, twice.
   Kept if >= 3 inputs return, >= 2 distinct outputs, and both runs agree (no randomness/time).
2. Model: K completions per signature (batched, one prompt), each run on the same inputs.
   score = matched / returned-by-real; pass@1 = mean(score == 1); pass@K = any(score == 1).
3. Floor: trivial bodies (return None / return first arg) on the same inputs.

Usage: equiv.py CKPT --out OUT.json [--k 8] [--mask]
"""

import argparse
import ast
import json
import os
import random
import subprocess
import time

import torch

from eval_samples import HOLDOUT, ByteLM, CachedGen, cut_function, generate_masked

REF = "/media/llm-data/datasets/pycode/equiv_ref.json"
REF_TRAIN = "/media/llm-data/datasets/pycode/equiv_ref_train.json"
N_IN = 40
RUNNER = r'''
import json, sys, random, signal
src, fname, nargs, n_in = json.loads(sys.stdin.read())
import io
real_out, sys.stdout = sys.stdout, io.StringIO()     # generated code may print: keep the result line clean
g = {}
try:
    exec(src, g)
except BaseException as e:
    real_out.write(json.dumps("load") + "\n"); sys.exit()
def alarm(*_):
    raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
r = random.Random(0)
pool = [0, 1, 3, -2, 7, 2.5, "ab", "", "x y", [1, 2], [], [3, 1, 2], (1, 2), {"a": 1}, None, True, False]
out = {}
for i in range(n_in):
    args = [r.choice(pool) for _ in range(nargs)]
    args = [list(a) if isinstance(a, list) else dict(a) if isinstance(a, dict) else a for a in args]
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        res = g[fname](*args)
        signal.setitimer(signal.ITIMER_REAL, 0)
        out[i] = repr(res)[:300] + "|" + repr(args)[:300]     # result and in-place effect on the arguments
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
real_out.write(json.dumps(out) + "\n")
'''


def run(src, fname, nargs):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    try:
        p = subprocess.run(cmd, input=json.dumps([src, fname, nargs, N_IN]), capture_output=True, text=True, timeout=15)
        o = json.loads(p.stdout) if p.returncode == 0 else None
        return o if isinstance(o, dict) else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def signature(src):
    """-> (name, n positional args) for a plain def, else None."""
    try:
        fn = ast.parse(src).body[0]
    except (SyntaxError, ValueError, IndexError, RecursionError):
        return None
    if not isinstance(fn, ast.FunctionDef):
        return None
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs or a.defaults or not a.args:
        return None
    return fn.name, len(a.args)


def build_ref(train=False):
    if train:       # memorization probe: same filter on a sample of training functions
        import re
        from build_corpus import free_names
        from eval_samples import BUILTINS
        fs = re.split(r"\n\n(?=def )", open("/media/llm-data/datasets/pycode/clean_train.txt").read())
        random.Random(3).shuffle(fs)
        hold = []
        for f in fs[:20000]:
            fr = free_names(f) if len(f) <= 600 else None
            if fr is not None and fr <= BUILTINS:
                hold.append({"clean": f.strip() + "\n"})
    else:
        hold = [h for h in json.load(open(HOLDOUT)) if h["self_contained"] and len(h["clean"]) <= 600]
    refs = []
    for h in hold:
        sig = signature(h["clean"])
        if sig is None:
            continue
        o1, o2 = run(h["clean"], *sig), run(h["clean"], *sig)
        if not o1 or o1 != o2 or len(o1) < 3 or len({v.split("|")[0] for v in o1.values()}) < 2:
            continue
        refs.append({"src": h["clean"], "name": sig[0], "nargs": sig[1], "out": o1})
    json.dump(refs, open(REF_TRAIN if train else REF, "w"))
    return refs


def score(src, ref):
    sig = signature(src)
    if sig is None or sig != (ref["name"], ref["nargs"]):
        return 0.0
    o = run(src, *sig) or {}
    return sum(o.get(k) == v for k, v in ref["out"].items()) / len(ref["out"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bytes", type=int, default=600)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.25)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--train", action="store_true", help="references from training functions (memorization probe)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    path = REF_TRAIN if a.train else REF
    refs = build_ref(a.train) if a.rebuild or not os.path.exists(path) else json.load(open(path))
    refs = refs[:200]
    print(f"references: {len(refs)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    floor = {}
    for name, body in (("return_none", "    return None\n"), ("return_arg0", None)):
        s = []
        for r in refs:
            head = r["src"].split("\n")[0] + "\n"
            b = body or f"    return {ast.parse(r['src']).body[0].args.args[0].arg}\n"
            s.append(score(head + b, r))
        floor[name] = {"mean": round(sum(s) / len(s), 4), "pass": round(sum(x == 1 for x in s) / len(s), 4)}
    print("floor", floor, flush=True)
    if not a.ckpt:
        return
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(0)
    g = CachedGen(ByteLM(a.ckpt))
    rows = []
    for i, r in enumerate(refs):
        head = r["src"].split("\n")[0] + "\n"
        prompt = (b"\n\n" + head.encode())[-(g.ctx // 2):]
        out = (generate_masked(g, prompt, a.k, a.bytes, a.temp, head.encode()) if a.mask
               else g.generate(prompt, a.k, a.bytes, temp=a.temp)).cpu().numpy()
        srcs = [cut_function(head + bytes(row).decode("utf-8", "replace"))[0] for row in out]
        sc = [score(s, r) for s in srcs]
        rows.append({"name": r["name"], "scores": sc, "best": srcs[max(range(len(sc)), key=sc.__getitem__)]})
        if i % 20 == 0:
            print(i, r["name"], [round(x, 2) for x in sc], flush=True)
    n = len(rows)
    res = {"ckpt": a.ckpt, "mask": a.mask, "k": a.k, "n_funcs": n, "floor": floor,
           "mean_score": round(sum(sum(x["scores"]) / a.k for x in rows) / n, 4),
           "pass@1": round(sum(sum(s == 1 for s in x["scores"]) / a.k for x in rows) / n, 4),
           f"pass@{a.k}": round(sum(any(s == 1 for s in x["scores"]) for x in rows) / n, 4),
           "best_mean": round(sum(max(x["scores"]) for x in rows) / n, 4),
           "wall_s": round(time.perf_counter() - t0, 1), "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()
