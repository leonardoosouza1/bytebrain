"""synth v2: meaning beyond copying. A body line is no longer a copy of a name segment.

`def times_3_then_cumsum_then_count_gt_4(xs):`
    xs = [x * 3 for x in xs]          <- word -> operator ("times" -> *), number in the name -> slot in the code
    acc, out = 0, []                  <- loop with accumulator: indentation is born from meaning
    for x in xs:
        acc += x
        out.append(acc)
    xs = out
    return sum(1 for x in xs if x > 4)
`def add_k_then_sort(xs, k):`         <- the second argument is inferred from the name

Units (each writes 1..5 lines and ends with a top-level `xs = ...`):
  PARAM  op_N  7 ops x N in 2..6 (add, times, keep_gt, keep_lt, clip, mod, take)
  KARG   op_k  3 ops that read the argument k
  LOOP         cumsum, running_max, dedup_adjacent (for + accumulator; dedup has an if inside the for)
  PLAIN        sort, reverse, negate, abs, unique, drop_first
Splits:
  train    every seen unit alone (all finals) + 60% of seen pairs (2 finals each)
  test_a   unseen pairs of seen units
  test_num names with a HELD (op, N): each op hides one N that is seen with other ops -> binding a number to a slot
  self     3-unit names (self-training pool, names only)
  test_b   other 3-unit names
The prompt is `def NAME(`: the model writes the signature too. Grade = bwrap execution on 40 (list, k) inputs (output AND list state before the final),
called with the reference's number of arguments (a wrong signature fails every input).

Usage:
  synth2.py build
  synth2.py check                                   (judge sanity: references 1.0, broken bodies < 1, print safe)
  synth2.py eval CKPT --split test_a --out F.json   [--n 400 --k 8]
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from synth import generate_rows, summary  # noqa: E402
from eval_samples import ByteLM, CachedGen, cut_function  # noqa: E402

D = "/media/llm-data/datasets/pycode/synth2"
NUMS = [2, 3, 4, 5, 6]
PARAM = {
    "add": "xs = [x + {n} for x in xs]",
    "times": "xs = [x * {n} for x in xs]",
    "keep_gt": "xs = [x for x in xs if x > {n}]",
    "keep_lt": "xs = [x for x in xs if x < {n}]",
    "clip": "xs = [min(x, {n}) for x in xs]",
    "mod": "xs = [x % {n} for x in xs]",
    "take": "xs = xs[:{n}]",
}
KARG = {
    "add_k": "xs = [x + k for x in xs]",
    "keep_gt_k": "xs = [x for x in xs if x > k]",
    "take_k": "xs = xs[:k]",
}
LOOP = {
    "cumsum": ["acc, out = 0, []", "for x in xs:", "    acc += x", "    out.append(acc)", "xs = out"],
    "running_max": ["out = []", "for x in xs:", "    out.append(x if not out or x > out[-1] else out[-1])", "xs = out"],
    "dedup_adjacent": ["out = []", "for x in xs:", "    if not out or out[-1] != x:", "        out.append(x)", "xs = out"],
}
PLAIN = {
    "sort": "xs = sorted(xs)",
    "reverse": "xs = xs[::-1]",
    "negate": "xs = [-x for x in xs]",
    "abs": "xs = [abs(x) for x in xs]",
    "unique": "xs = list(dict.fromkeys(xs))",
    "drop_first": "xs = xs[1:]",
}
UNITS = {u: [c] for u, c in PLAIN.items()}
for _op, _tpl in PARAM.items():
    for _n in NUMS:
        UNITS[f"{_op}_{_n}"] = [_tpl.format(n=_n)]
UNITS.update({u: [c] for u, c in KARG.items()})
UNITS.update(LOOP)
HELD = [f"{op}_{NUMS[i % len(NUMS)]}" for i, op in enumerate(PARAM)]      # never in train, in any composition
FINAL = {"": "return xs", "sum": "return sum(xs)", "max": "return max(xs, default=0)", "len": "return len(xs)",
         "first": "return xs[0] if xs else 0"}
FINAL.update({f"count_gt_{n}": f"return sum(1 for x in xs if x > {n})" for n in NUMS})
_r = random.Random(11)
# wide values and lengths: with -5..9 a count_gt_6 / len final hid a skipped unit in 21% of broken bodies
INPUTS = [[[_r.randint(-9, 19) for _ in range(_r.randint(0, 9))], _r.randint(0, 6)] for _ in range(40)]
RUNNER = r'''
import json, sys, signal, io
items, inputs = json.loads(sys.stdin.read())
real_out, sys.stdout = sys.stdout, io.StringIO()     # generated code may print: keep it out of the result line
def alarm(*_):
    raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
res = []
for src, name, nargs in items:
    g = {}
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        exec(src, g)
        f = g[name]
        signal.setitimer(signal.ITIMER_REAL, 0)
    except BaseException:
        signal.setitimer(signal.ITIMER_REAL, 0)
        res.append(None)
        continue
    out = []
    for xs, k in inputs:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.05)
            out.append(repr(f(list(xs), k) if nargs == 2 else f(list(xs))))
        except BaseException:
            out.append("!")
        signal.setitimer(signal.ITIMER_REAL, 0)
    res.append(out)
real_out.write(json.dumps(res) + "\n")
'''


def name_of(units, final):
    return "_then_".join(units + ([final] if final else []))


def nargs(units):
    return 2 if any(u.endswith("_k") for u in units) else 1


def head(units, final):
    return f"def {name_of(units, final)}({'xs, k' if nargs(units) == 2 else 'xs'}):\n"


def source(units, final):
    return head(units, final) + "".join(f"    {l}\n" for u in units for l in UNITS[u]) + f"    {FINAL[final]}\n"


def reference(units, final):
    g = {}
    # neutral name: a function named after a unit (`abs`) would shadow the builtin it calls and recurse
    exec(source(units, final).replace(f"def {name_of(units, final)}(", "def _ref(", 1), g)
    f = g["_ref"]
    return [repr(f(list(xs), k) if nargs(units) == 2 else f(list(xs))) for xs, k in INPUTS]


def build():
    os.makedirs(D, exist_ok=True)
    r = random.Random(0)
    seen = [u for u in UNITS if u not in HELD]
    fin = list(FINAL)
    pairs = [list(p) for p in itertools.product(seen, seen)]
    r.shuffle(pairs)
    cut = int(len(pairs) * 0.6)
    triples = [list(t) for t in itertools.product(seen, repeat=3)]
    r.shuffle(triples)
    split = {
        "train": [[[u], f] for u in seen for f in fin] + [[p, f] for p in pairs[:cut] for f in r.sample(fin, 2)],
        "test_a": [[p, r.choice(fin)] for p in pairs[cut:]],
        "test_num": [[[h], f] for h in HELD for f in fin]
                    + [[[h, s] if r.random() < 0.5 else [s, h], r.choice(fin)] for h in HELD for s in r.sample(seen, 40)],
        "self": [[t, r.choice(fin)] for t in triples[:6000]],
        "test_b": [[t, r.choice(fin)] for t in triples[6000:8000]],
    }
    funcs = [source(u, f) for u, f in split["train"]]
    r.shuffle(funcs)
    open(f"{D}/train.txt", "w").write("\n".join(funcs) + "\n")   # one blank line between functions
    json.dump(split, open(f"{D}/splits.json", "w"))
    longest = max(len(source(u, f)) for v in split.values() for u, f in v)
    print({k: len(v) for k, v in split.items()}, "train bytes", os.path.getsize(f"{D}/train.txt"),
          "units", len(UNITS), "held", HELD, "longest source", longest)


def run_many(items):
    """items [(src, name, nargs)] -> [outputs or None], one bwrap call."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",  # review #48 g4 S1
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    try:
        p = subprocess.run(cmd, input=json.dumps([items, INPUTS]), capture_output=True, text=True,
                           timeout=30 + 2 * len(items))
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        if len(items) == 1:
            return [None]
        h = len(items) // 2                             # one bad sample must not zero the whole chunk
        return run_many(items[:h]) + run_many(items[h:])


def state_variant(src):
    """same source with its last top-level return replaced by `return xs`: exposes the list state before the final."""
    lines = src.rstrip("\n").split("\n")
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].startswith("    return"):
            return "\n".join(lines[:i] + ["    return xs"] + lines[i + 1:]) + "\n"
    return src


def grade(tasks, srcs, strict=True):
    """-> per task list of scores: share of inputs where the output equals the reference AND (strict) the list
    state before the final equals the reference state (a final like len/count_gt can hide a skipped unit)."""
    items = [(s, name_of(u, f), nargs(u)) for (u, f), ss in zip(tasks, srcs) for s in ss]
    if strict:
        items += [(state_variant(s), n, a) for s, n, a in items]
    outs = []
    for i in range(0, len(items), 256):
        outs += run_many(items[i:i + 256])
    half = len(items) // 2 if strict else len(items)
    k = len(srcs[0])
    scores = []
    for t, (u, f) in enumerate(tasks):
        ref, ref_st = reference(u, f), reference(u, "") if strict else None
        row = []
        for j in range(t * k, (t + 1) * k):
            o, st = outs[j], outs[half + j] if strict else None
            if o is None or (strict and st is None):
                row.append(0.0)
                continue
            ok = [a == b for a, b in zip(o, ref)]
            if strict:
                ok = [x and a == b for x, a, b in zip(ok, st, ref_st)]
            row.append(sum(ok) / len(ref))
        scores.append(row)
    return scores


def propose(g, tasks, k, n_bytes, temp, batch=512):
    """tasks [(units, final)] -> per task k sources. Prompt `def NAME(`; batched only with prompts of the same
    length (left padding shifts absolute position, which fine-tuned models read as a line-count cue)."""
    heads = [f"def {name_of(u, f)}(" for u, f in tasks]
    out = [None] * len(tasks)
    by_len = {}
    for i, h in enumerate(heads):
        by_len.setdefault(len(h), []).append(i)
    per = max(1, batch // k)
    for group in by_len.values():
        for c in range(0, len(group), per):
            idx = group[c:c + per]
            ids = torch.tensor([list(b"\n\n" + heads[i].encode()) for i in idx for _ in range(k)], device=g.m.device)
            rows = generate_rows(g, ids, n_bytes, temp).cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = [cut_function(heads[i] + bytes(rw).decode("utf-8", "replace"))[0] for rw in rows[j * k:(j + 1) * k]]
    return out


def kind(units):
    if any(u in HELD for u in units):
        return "held"
    if any(u in LOOP for u in units):
        return "loop"
    if any(u.endswith("_k") for u in units):
        return "karg"
    return "line"


def check():
    """judge sanity on CPU: every reference must score 1.0; broken bodies must not."""
    split = json.load(open(f"{D}/splits.json"))
    r = random.Random(3)
    tasks = [t for v in split.values() for t in r.sample(v, 40)]
    refs = [[source(u, f)] for u, f in tasks]
    ok = sum(s[0] == 1 for s in grade(tasks, refs))
    broken = []
    for (u, f), [s] in zip(tasks, refs):
        lines = s.split("\n")
        broken.append([s.replace("(xs, k):", "(xs):") if nargs(u) == 2 else "\n".join(lines[:1] + lines[2:]),
                       s.replace("    return", "    print('noise')\n    return")])
    sc = grade(tasks, broken)
    print(json.dumps({"tasks": len(tasks), "references_1.0": ok,
                      "broken_still_1.0": sum(s[0] == 1 for s in sc),
                      "print_safe_1.0": sum(s[1] == 1 for s in sc)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check", "eval"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--split", default="test_a")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bytes", type=int, default=360)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--mem-frac", type=float, default=0.4)
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    if a.cmd == "check":
        return check()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(0)
    split = json.load(open(f"{D}/splits.json"))
    tasks = split[a.split][:]
    random.Random(1).shuffle(tasks)
    tasks = tasks[:a.n]
    t0 = time.perf_counter()
    g = CachedGen(ByteLM(a.ckpt))
    srcs = propose(g, tasks, a.k, a.bytes, a.temp)
    sc = grade(tasks, srcs)
    by = {}
    for (u, _), s in zip(tasks, sc):
        by.setdefault(kind(u), []).append(s)
    res = {"ckpt": a.ckpt, "split": a.split, **summary(sc),
           "by_kind": {k: {"n": len(v), **{x: y for x, y in summary(v).items() if x.startswith("pass")}} for k, v in by.items()},
           "wall_s": round(time.perf_counter() - t0, 1), "examples": [s[0] for s in srcs[:6]]}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))


if __name__ == "__main__":
    main()
