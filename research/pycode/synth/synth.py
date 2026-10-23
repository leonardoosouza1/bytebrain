"""Meaning in isolation: list functions whose NAME composes their meaning, truth = execution.

`def double_then_keep_even_then_sum(xs):` -> one body line per primitive, in name order, then the final.
13 list primitives x 7 finals. Splits (by composition, every primitive is seen in training):
  train   = all 1-primitive + 80% of 2-primitive compositions        (bodies given)
  test_a  = the other 20% of 2-primitive compositions                 (unseen combinations)
  self    = half of the 3-primitive compositions                       (self-training pool: names only, no bodies)
  test_b  = the other half of the 3-primitive compositions             (unseen length, never used to train)
Grade: model body vs reference on 30 fixed int lists, in bwrap. score = share of equal outputs.

Usage:
  synth.py build                                  -> $D/{train.txt, splits.json}
  synth.py eval CKPT --split test_a --out F.json  [--k 8]
  synth.py self CKPT --out-dir DIR                [--k 8] [--law-lines] [--max-tasks N]   -> DIR/good.txt, DIR/mix.txt, DIR/stats.json
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_samples import ByteLM, CachedGen, cut_function  # noqa: E402

D = "/media/llm-data/datasets/pycode/synth"
PRIM = {
    "sort": "xs = sorted(xs)",
    "reverse": "xs = xs[::-1]",
    "double": "xs = [x * 2 for x in xs]",
    "square": "xs = [x * x for x in xs]",
    "negate": "xs = [-x for x in xs]",
    "add_one": "xs = [x + 1 for x in xs]",
    "keep_even": "xs = [x for x in xs if x % 2 == 0]",
    "keep_odd": "xs = [x for x in xs if x % 2 == 1]",
    "keep_pos": "xs = [x for x in xs if x > 0]",
    "drop_first": "xs = xs[1:]",
    "drop_last": "xs = xs[:-1]",
    "unique": "xs = list(dict.fromkeys(xs))",
    "abs": "xs = [abs(x) for x in xs]",
}
FINAL = {
    "": "return xs",
    "sum": "return sum(xs)",
    "max": "return max(xs, default=0)",
    "min": "return min(xs, default=0)",
    "len": "return len(xs)",
    "first": "return xs[0] if xs else 0",
    "last": "return xs[-1] if xs else 0",
}
_r = random.Random(11)
INPUTS = [[_r.randint(-5, 9) for _ in range(_r.randint(0, 7))] for _ in range(30)]
RUNNER = r'''
import json, sys, signal
items, inputs = json.loads(sys.stdin.read())
import io
real_out, sys.stdout = sys.stdout, io.StringIO()     # generated code may print: keep it out of the result line
def alarm(*_):
    raise TimeoutError
signal.signal(signal.SIGALRM, alarm)
res = []
for src, name in items:
    out = []
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
    for xs in inputs:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.05)
            out.append(repr(f(list(xs))))
        except BaseException:
            out.append("!")
        signal.setitimer(signal.ITIMER_REAL, 0)
    res.append(out)
real_out.write(json.dumps(res) + "\n")
'''


def name_of(prims, final):
    return "_then_".join(prims + ([final] if final else []))


def source(prims, final):
    body = "".join(f"    {PRIM[p]}\n" for p in prims)
    return f"def {name_of(prims, final)}(xs):\n{body}    {FINAL[final]}\n"


def reference(prims, final):
    g = {}
    exec(source(prims, final), g)
    return [repr(g[name_of(prims, final)](list(xs))) for xs in INPUTS]


def build():
    os.makedirs(D, exist_ok=True)
    r = random.Random(0)
    pairs = list(itertools.product(PRIM, PRIM))
    r.shuffle(pairs)
    cut = int(len(pairs) * 0.8)
    triples = list(itertools.product(PRIM, PRIM, PRIM))
    r.shuffle(triples)
    split = {
        "train": [[p] for p in PRIM] + [list(p) for p in pairs[:cut]],
        "test_a": [list(p) for p in pairs[cut:]],
        "self": [list(t) for t in triples[:len(triples) // 2]],
        "test_b": [list(t) for t in triples[len(triples) // 2:]],
    }
    split = {k: [[p, f] for p in v for f in FINAL] for k, v in split.items()}
    funcs = [source(p, f) for p, f in split["train"]]
    r.shuffle(funcs)
    open(f"{D}/train.txt", "w").write("\n".join(funcs) + "\n")   # one blank line between functions
    json.dump(split, open(f"{D}/splits.json", "w"))
    print({k: len(v) for k, v in split.items()}, "train bytes", os.path.getsize(f"{D}/train.txt"))


def run_many(items):
    """items [(src, name)] -> [outputs or None], one bwrap call."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    try:
        p = subprocess.run(cmd, input=json.dumps([items, INPUTS]), capture_output=True, text=True,
                           timeout=30 + 2 * len(items))
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return [None] * len(items)


def propose(g, tasks, k, n_bytes, temp, batch=512, law=False):
    """tasks [(prims, final)] -> per task k sources. Batched only with prompts of the same length: left padding
    shifts absolute position, which fine-tuned models read as a line-count cue."""
    heads = [f"def {name_of(p, f)}(xs):\n" for p, f in tasks]
    out = [None] * len(tasks)
    by_len = {}
    for i, h in enumerate(heads):
        by_len.setdefault(len(h), []).append(i)
    per = max(1, batch // k)
    for group in by_len.values():
        for c in range(0, len(group), per):
            idx = group[c:c + per]
            ids = torch.tensor([list(b"\n\n" + heads[i].encode()) for i in idx for _ in range(k)], device=g.m.device)
            need = torch.tensor([len(tasks[i][0]) for i in idx for _ in range(k)], device=g.m.device) if law else None
            rows = generate_rows(g, ids, n_bytes, temp, need).cpu().numpy()
            for j, i in enumerate(idx):
                out[i] = [cut_function(heads[i] + bytes(rw).decode("utf-8", "replace"))[0] for rw in rows[j * k:(j + 1) * k]]
    return out


@torch.no_grad()
def generate_rows(g, ids, n_bytes, temp, need=None):
    """CachedGen.generate for a batch of different (same-length) prompts.
    need [B] (optional) = structural law in the loop: number of primitives in the name. At the first byte of a body
    line, 'r' (return) is masked while fewer lines were written, 'x' (another xs line) once enough were.
    Says HOW MANY lines, never WHICH line."""
    dev = ids.device
    if need is not None:
        last5 = ids[:, -5:].clone()                                             # last 5 bytes (prompt ends in '\n')
        done = torch.zeros(ids.shape[0], dtype=torch.long, device=dev)        # body lines started with non-'r'
        start = torch.tensor(list(b"\n    "), device=dev)
        R, X = ord("r"), ord("x")
    g.reset()
    g.hist = ids.clone()
    logits = g._forward(ids)
    out = torch.zeros(ids.shape[0], n_bytes, dtype=torch.uint8, device=ids.device)
    for i in range(n_bytes):
        p = torch.softmax(logits.float() / temp, -1)
        if need is not None:
            at = (last5 == start).all(1)
            # at the first byte of the first body line, tail = "):\n    " -> last 5 = "\n    " as well
            p[:, R] *= ~(at & (done < need))
            p[:, X] *= ~(at & (done >= need))
        nxt = torch.multinomial(p / p.sum(-1, keepdim=True), 1).squeeze(-1)
        if need is not None:
            done += (at & (nxt != R)).long()
            last5 = torch.cat([last5[:, 1:], nxt[:, None]], 1)
        out[:, i] = nxt.to(torch.uint8)
        g.hist = torch.cat([g.hist, nxt[:, None]], 1)
        if g.pos >= g.ctx:
            tail = g.hist[:, -g.keep:]
            g.reset()
            g.hist = tail
            logits = g._forward(tail)
        else:
            logits = g._forward(nxt[:, None])
    return out


def grade(tasks, srcs):
    """-> per task list of scores (share of the 30 inputs with the reference output)."""
    items = [(s, name_of(p, f)) for (p, f), ss in zip(tasks, srcs) for s in ss]
    outs = []
    for i in range(0, len(items), 256):
        outs += run_many(items[i:i + 256])
    k = len(srcs[0])
    scores = []
    for t, (p, f) in enumerate(tasks):
        ref = reference(p, f)
        scores.append([0.0 if o is None else sum(a == b for a, b in zip(o, ref)) / len(ref)
                       for o in outs[t * k:(t + 1) * k]])
    return scores


def summary(scores):
    n, k = len(scores), len(scores[0])
    return {"n": n, "k": k, "mean_score": round(sum(sum(s) for s in scores) / (n * k), 4),
            "pass@1": round(sum(sum(x == 1 for x in s) for s in scores) / (n * k), 4),
            f"pass@{k}": round(sum(any(x == 1 for x in s) for s in scores) / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval", "self"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--split", default="test_a")
    ap.add_argument("--out")
    ap.add_argument("--out-dir")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tasks", type=int, default=400)
    ap.add_argument("--bytes", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--law-mask", action="store_true", help="line-count law inside generation (see generate_rows)")
    ap.add_argument("--law-lines", action="store_true", help="keep only exact bodies with one line per primitive")
    ap.add_argument("--mem-frac", type=float, default=0.3)
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(0)
    split = json.load(open(f"{D}/splits.json"))
    g = CachedGen(ByteLM(a.ckpt))
    t0 = time.perf_counter()
    if a.cmd == "eval":
        tasks = split[a.split]
        random.Random(1).shuffle(tasks)
        tasks = tasks[:a.max_tasks]
        srcs = propose(g, tasks, a.k, a.bytes, a.temp, law=a.law_mask)
        scores = grade(tasks, srcs)
        res = {"ckpt": a.ckpt, "split": a.split, **summary(scores), "wall_s": round(time.perf_counter() - t0, 1)}
        res["examples"] = [{"name": name_of(*t), "score": max(s), "src": ss[max(range(len(s)), key=s.__getitem__)]}
                           for t, s, ss in list(zip(tasks, scores, srcs))[:20]]
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
    else:   # one self-training round on the 'self' pool: keep bodies whose OUTPUTS match; bodies are never shown
        os.makedirs(a.out_dir, exist_ok=True)
        tasks = split["self"][:a.max_tasks]
        srcs = propose(g, tasks, a.k, a.bytes, a.temp, law=a.law_mask)
        scores = grade(tasks, srcs)
        good = []
        for (p, f), ss, s in zip(tasks, srcs, scores):
            best = [x for x, v in zip(ss, s) if v == 1]
            if a.law_lines:     # law measured on the training corpus: one `xs =` line per primitive in the name
                best = [x for x in best if sum(l.strip().startswith("xs =") for l in x.split("\n")) == len(p)]
            if best:
                good.append(min(best, key=len))      # one per name: the shortest exact body
        train = open(f"{D}/train.txt").read().split("\n\n")
        mix = [x.strip() + "\n" for x in good] * a.repeat + [x.strip() + "\n" for x in train if x.strip()]
        random.Random(1).shuffle(mix)
        open(f"{a.out_dir}/good.txt", "w").write("\n".join(good))
        open(f"{a.out_dir}/mix.txt", "w").write("\n".join(mix))
        st = {**summary(scores), "kept": len(good), "wall_s": round(time.perf_counter() - t0, 1)}
        json.dump(st, open(f"{a.out_dir}/stats.json", "w"), indent=1)
        print(json.dumps(st))


if __name__ == "__main__":
    main()
