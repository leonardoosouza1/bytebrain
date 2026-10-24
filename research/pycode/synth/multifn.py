"""Multi-function programs: helpers that call helpers + a main dispatcher (ledger #22, registered 2026-10-23, CPU).

Request: functions that depend on other functions, a basic and an advanced calculator, a main that calls the others;
find the limits and what is missing.

Task: a spec name like `calc_hyp_fact_avg` (bytes). The program must contain
  - one def per requested op AND every helper it depends on, transitively (hyp -> sqrt, add, square -> mul),
  - `def main(op, a, b=0):` with one dispatch line per requested op.
Nothing in the name lists the dependencies: the planner must infer them from what it learned about each op.

Model = the same two parts as bytewriter.py:
  planner  blockaddr.Net("addr", gate_bias=-4): name bytes -> learned boundary gate -> address -> block symbols
           (def:<op> blocks in dependency order, then disp:<op> lines, then <eos>)
  writer   byte GRU conditioned on the block symbol, writes each block's text byte by byte
Grade (bwrap): main(op, a, b) on 20 (a, b) cases per requested op equals the reference module.
Splits: train = specs of 1-3 ops; test_comb = unseen 3-op sets; len4..len8 = 4..8 ops (never trained);
        basic = add sub mul div; full = all 15 ops.
Prediction: test_comb >= 90%; len4 >= 80%; len8 >= 50%; full calc 15 ops: likely fails (2x the longest train plan).
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import blockaddr as BA  # noqa: E402

DEFS = {   # op -> (body lines, direct deps)
    "add": (["return a + b"], []),
    "sub": (["return a - b"], []),
    "mul": (["return a * b"], []),
    "div": (["if b == 0:", "    return None", "return a / b"], []),
    "mod": (["if b == 0:", "    return None", "return a % b"], []),
    "maxx": (["return a if a > b else b"], []),
    "sqrt": (["if a < 0:", "    return None", "return a ** 0.5"], []),
    "neg": (["return sub(0, a)"], ["sub"]),
    "abs": (["return neg(a) if a < 0 else a"], ["neg"]),
    "square": (["return mul(a, a)"], ["mul"]),
    "pow": (["r = 1", "for _ in range(b):", "    r = mul(r, a)", "return r"], ["mul"]),
    "fact": (["r = 1", "for i in range(2, a + 1):", "    r = mul(r, i)", "return r"], ["mul"]),
    "hyp": (["return sqrt(add(square(a), square(b)))"], ["sqrt", "add", "square"]),
    "avg": (["return div(add(a, b), 2)"], ["div", "add"]),
    "pct": (["return div(mul(a, b), 100)"], ["div", "mul"]),
}
OPS = sorted(DEFS)
SYM = ["<pad>", "<sep>", "<eos>"] + [f"def:{o}" for o in OPS] + [f"disp:{o}" for o in OPS]
MAX_OUT = 34                                     # grouped: 15 defs + 15 dispatch + eos; interleaved: set in main
CASES = [(a, b) for a in (-2, 0, 1, 3, 5) for b in (-1, 0, 2, 4)]


PLAN_FORMAT = "grouped"     # grouped: all defs then all dispatch lines (two passes over the name)
                            # interleaved: per op in name order: its dep closure, def, dispatch (one monotone pass)


def plan_of(spec):
    if PLAN_FORMAT == "interleaved":
        out = []
        for o in spec:
            out += [s for s in plan_of_grouped((o,)) if s != "<eos>"]
        return out + ["<eos>"]
    return plan_of_grouped(spec)


def plan_of_grouped(spec):
    order, seen = [], set()

    def visit(o):
        if o in seen:
            return
        seen.add(o)
        for d in DEFS[o][1]:
            visit(d)
        order.append(o)
    for o in spec:
        visit(o)
    return [f"def:{o}" for o in order] + [f"disp:{o}" for o in spec] + ["<eos>"]


def block_text(sym):
    kind, o = sym.split(":")
    if kind == "def":
        return f"def {o}(a, b=0):\n" + "".join(f"    {l}\n" for l in DEFS[o][0])
    return f"    if op == \"{o}\":\n        return {o}(a, b)\n"


def assemble(syms):
    defs = list(dict.fromkeys(s for s in syms if s.startswith("def:")))   # a repeated def is written once
    disp = [s for s in syms if s.startswith("disp:")]
    return "".join(block_text(s) for s in defs) + "def main(op, a, b=0):\n" + \
        "".join(block_text(s) for s in disp) + "    return None\n"


def spec_name(spec):
    return "calc_" + "_".join(spec)


RUNNER = r"""
import json, sys
items, cases = json.load(sys.stdin)
res = []
for src, ops in items:
    g = {}
    try:
        exec(src, g)
        row = []
        for o in ops:
            for a, b in cases:
                try:
                    v = g["main"](o, a, b)
                    row.append(repr(round(v, 6) if isinstance(v, float) else v))
                except Exception as e:
                    row.append("!" + type(e).__name__)
        res.append(row)
    except Exception as e:
        res.append(None)
print(json.dumps(res))
"""


def run_many(items):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    try:
        p = subprocess.run(cmd, input=json.dumps([items, CASES]), capture_output=True, text=True,
                           timeout=30 + 2 * len(items))
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        if len(items) == 1:
            return [None]
        h = len(items) // 2
        return run_many(items[:h]) + run_many(items[h:])


class Data:
    def __init__(self, seed=0):
        r = random.Random(seed)
        sets3 = sorted({tuple(sorted(r.sample(OPS, 3))) for _ in range(4000)})
        r.shuffle(sets3)
        held3 = set(sets3[:150])
        train = set()
        while len(train) < 1800:
            k = r.choice([1, 2, 2, 3, 3, 3])
            s = tuple(r.sample(OPS, k))
            if tuple(sorted(s)) not in held3:
                train.add(s)
        self.sets = {"train": sorted(train),
                     "test_comb": [tuple(r.sample(s, 3)) for s in sorted(held3)],
                     "basic": [("add", "sub", "mul", "div")],
                     "full": [tuple(OPS)]}
        for k in range(4, 9):
            self.sets[f"len{k}"] = [tuple(r.sample(OPS, k)) for _ in range(150)]


def setup_planner_globals():
    BA.SYM, BA.START, BA.MAX_OUT = SYM, len(SYM), MAX_OUT


def name_bytes(specs):
    ns = [[BA.B2I[b] for b in spec_name(s).encode()] for s in specs]
    L = max(map(len, ns))
    return torch.tensor([n + [0] * (L - len(n)) for n in ns])


def targets(specs):
    ys = [[SYM.index(s) for s in plan_of(sp)] for sp in specs]
    T = max(map(len, ys))
    y = torch.tensor([v + [0] * (T - len(v)) for v in ys])
    return torch.cat([torch.full((len(ys), 1), len(SYM)), y[:, :-1]], 1), y


def cosine_lr(opt, base, s, steps):
    for gp in opt.param_groups:
        gp["lr"] = base * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))


def train_planner(data, seed, steps):
    torch.manual_seed(seed)
    net = BA.Net("addr", gate_bias=-4.0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    r, tr = random.Random(seed + 100), data.sets["train"]
    for s in range(steps):
        cosine_lr(opt, 3e-3, s, steps)
        sp = [tr[r.randrange(len(tr))] for _ in range(48)]
        y_in, y = targets(sp)
        loss = F.cross_entropy(net(name_bytes(sp), y_in).transpose(1, 2), y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


CH = sorted(set("".join(block_text(s) for s in SYM[3:])))
END = len(CH)


class Writer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.cond = nn.Embedding(len(SYM), d)
        self.ch = nn.Embedding(len(CH) + 2, d)
        self.cell = nn.GRUCell(2 * d, d)
        self.out = nn.Linear(d, len(CH) + 1)

    def forward(self, c, y_in):
        cond, h, outs = self.cond(c), torch.zeros(c.shape[0], self.cell.hidden_size), []
        for t in range(y_in.shape[1]):
            h = self.cell(torch.cat([cond, self.ch(y_in[:, t])], 1), h)
            outs.append(self.out(h))
        return torch.stack(outs, 1)

    @torch.no_grad()
    def write(self, c, max_len=160):
        cond, h = self.cond(c), torch.zeros(c.shape[0], self.cell.hidden_size)
        prev, out = torch.full((c.shape[0],), len(CH) + 1), []
        for _ in range(max_len):
            h = self.cell(torch.cat([cond, self.ch(prev)], 1), h)
            prev = self.out(h).argmax(-1)
            out.append(prev)
        res = []
        for row in torch.stack(out, 1).tolist():
            row = row[:row.index(END)] if END in row else row
            res.append("".join(CH[i] for i in row))
        return res


def train_writer(data, d, seed, steps=1500, batch=64):
    torch.manual_seed(seed)
    pool = sorted({s for sp in data.sets["train"] for s in plan_of(sp) if s != "<eos>"})
    net = Writer(d)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    r = random.Random(seed)
    for s in range(steps):
        cosine_lr(opt, 3e-3, s, steps)
        bs = [pool[r.randrange(len(pool))] for _ in range(batch)]
        ys = [[CH.index(ch) for ch in block_text(b)] + [END] for b in bs]
        T = max(map(len, ys))
        y = torch.tensor([v + [-100] * (T - len(v)) for v in ys])
        y_in = torch.cat([torch.full((batch, 1), len(CH) + 1), y[:, :-1].clamp(min=0)], 1)
        loss = F.cross_entropy(net(torch.tensor([SYM.index(b) for b in bs]), y_in).transpose(1, 2), y,
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net, pool


def evaluate(planner, writer, data, names):
    planner.eval()
    res = {}
    for k in names:
        specs = data.sets[k]
        plans = []
        for i in range(0, len(specs), 256):
            for row in planner.greedy(name_bytes(specs[i:i + 256])).tolist():
                s = [SYM[j] for j in row]
                plans.append(s[:s.index("<eos>") + 1] if "<eos>" in s else s)
        flat = [s for p in plans for s in p if s.startswith(("def:", "disp:"))]
        texts = dict(zip(flat, writer.write(torch.tensor([SYM.index(s) for s in flat])))) if flat else {}
        srcs = []
        for p in plans:
            body = [s for s in p if s.startswith(("def:", "disp:"))]
            defs = "".join(texts[s] for s in dict.fromkeys(s for s in body if s.startswith("def:")))
            disp = "".join(texts[s] for s in body if s.startswith("disp:"))
            srcs.append(defs + "def main(op, a, b=0):\n" + disp + "    return None\n")
        refs = [assemble(plan_of(sp)) for sp in specs]
        outs = run_many([(s, list(sp)) for s, sp in zip(srcs, specs)] + [(s, list(sp)) for s, sp in zip(refs, specs)])
        got, ref = outs[:len(specs)], outs[len(specs):]
        miss_dep = sum(set(plan_of(sp)) - set(p) != set() and all(f"disp:{o}" in p for o in sp)
                       for p, sp in zip(plans, specs))
        res[k] = {"n": len(specs),
                  "exec": round(sum(g is not None and not any(v.startswith("!") for v in g) for g in got) / len(specs), 4),
                  "pass1": round(sum(g == r for g, r in zip(got, ref)) / len(specs), 4),
                  "plan_exact": round(sum(p == plan_of(sp) for p, sp in zip(plans, specs)) / len(specs), 4),
                  "text_exact": round(sum(s == r for s, r in zip(srcs, refs)) / len(specs), 4),
                  "missing_dep_only": miss_dep,
                  "plan_len": round(sum(len(plan_of(sp)) for sp in specs) / len(specs), 1),
                  "example_spec": spec_name(specs[0]), "example_plan": plans[0], "example": srcs[0]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--writer-dim", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--plan-format", default="grouped", choices=["grouped", "interleaved"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    global PLAN_FORMAT, MAX_OUT
    PLAN_FORMAT = a.plan_format
    MAX_OUT = max(34, len(plan_of(tuple(OPS))) + 4)
    setup_planner_globals()
    data = Data()
    t0 = time.perf_counter()
    planner = train_planner(data, a.seed, a.steps)
    writer, pool = train_writer(data, a.writer_dim, a.seed)
    names = ["test_comb", "len4", "len5", "len6", "len7", "len8", "basic", "full"]
    res = {"seed": a.seed, "steps": a.steps, "plan_format": a.plan_format, "max_out": MAX_OUT, "n_train": len(data.sets["train"]),
           "max_train_plan": max(len(plan_of(s)) for s in data.sets["train"]),
           "planner_params": sum(p.numel() for p in planner.parameters()),
           "writer_params": sum(p.numel() for p in writer.parameters()),
           "wall_s": round(time.perf_counter() - t0, 1), "eval": evaluate(planner, writer, data, names)}
    print(json.dumps({k: (v if k != "eval" else {s: {m: x for m, x in r.items() if not m.startswith("example")}
                                                  for s, r in v.items()}) for k, v in res.items()}), flush=True)
    json.dump(res, open(f"{a.out}/multifn_{a.plan_format}_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
