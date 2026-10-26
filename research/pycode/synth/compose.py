"""main that COMPOSES helpers: a call tree, not a dispatch list (ledger #22 continuation, registered 2026-10-23, CPU).

Spec name = the main expression in prefix, e.g. `calc_hyp_avg_a_b_sub_a_b` -> main(a, b) = hyp(avg(a, b), sub(a, b)).
The program must define every called helper and its dependencies (hyp -> sqrt, add, square -> mul), then main.
Plan (one monotone pass, interleaved as in multifn): for each prefix token in name order:
  call op -> its dependency closure as def:<x> symbols, then call:<op>;  leaf -> leaf:a / leaf:b;  then <eos>.
Planner = blockaddr address net; writer = byte GRU for the def blocks (multifn.Writer).
main's return line is rendered from the call/leaf prefix by a fixed rule (not learned; stated in the report).
Splits: train = trees with 1-3 calls; test_same = unseen 1-3 call trees; calls4..calls8 = deeper/larger trees.
Grade (bwrap): main(a, b) on 20 (a, b) cases equals the reference module.
Prediction: test_same >= 95%; calls4 >= 80%; calls8 >= 50% (plan grows to ~60 symbols, 3x train).
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import blockaddr as BA  # noqa: E402
import multifn as MF  # noqa: E402

BIN = ["add", "sub", "mul", "div", "avg", "pct", "hyp", "maxx", "mod"]
SYM = ["<pad>", "<sep>", "<eos>"] + [f"def:{o}" for o in MF.OPS] + [f"call:{o}" for o in BIN] + ["leaf:a", "leaf:b"]
CASES = MF.CASES


def rand_tree(r, n_calls):
    """random binary call tree with n_calls internal nodes -> prefix token list."""
    if n_calls == 0:
        return [r.choice("ab")]
    left = r.randint(0, n_calls - 1)
    return [r.choice(BIN)] + rand_tree(r, left) + rand_tree(r, n_calls - 1 - left)


def plan_of(prefix):
    out = []
    for t in prefix:
        if t in ("a", "b"):
            out.append(f"leaf:{t}")
        else:
            out += [s for s in MF.plan_of_grouped((t,)) if s.startswith("def:")] + [f"call:{t}"]
    return out + ["<eos>"]


def render_expr(syms):
    """call/leaf prefix -> python expression; None when the prefix is not a complete tree."""
    toks = [s for s in syms if s.startswith(("call:", "leaf:"))]

    def go(i):
        if i >= len(toks):
            raise ValueError
        k, v = toks[i].split(":")
        if k == "leaf":
            return v, i + 1
        x, j = go(i + 1)
        y, j = go(j)
        return f"{v}({x}, {y})", j
    try:
        e, j = go(0)
        return e if j == len(toks) else None
    except (ValueError, RecursionError):
        return None


def assemble(syms, def_text):
    defs = list(dict.fromkeys(s for s in syms if s.startswith("def:")))
    e = render_expr(syms)
    return "".join(def_text(s) for s in defs) + "def main(a, b=0):\n" + f"    return {e if e else 'None'}\n"


def spec_name(prefix):
    return "calc_" + "_".join(prefix)


RUNNER = r"""
import json, sys
items, cases = json.load(sys.stdin)
res = []
for src in items:
    g = {}
    try:
        exec(src, g)
        row = []
        for a, b in cases:
            try:
                v = g["main"](a, b)
                row.append(repr(round(v, 6) if isinstance(v, float) else v))
            except Exception as e:
                row.append("!" + type(e).__name__)
        res.append(row)
    except Exception:
        res.append(None)
print(json.dumps(res))
"""


def run_many(items):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",  # review #48 g4 S1
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
        small = set()
        for _ in range(20000):
            small.add(tuple(rand_tree(r, r.choice([1, 2, 2, 3, 3, 3]))))
        small = sorted(small)
        r.shuffle(small)
        self.sets = {"test_same": small[:200], "train": small[200:]}
        for k in (4, 5, 6, 8):
            self.sets[f"calls{k}"] = [tuple(rand_tree(r, k)) for _ in range(150)]


def name_bytes(specs):
    ns = [[BA.B2I[b] for b in spec_name(s).encode()] for s in specs]
    L = max(map(len, ns))
    return torch.tensor([n + [0] * (L - len(n)) for n in ns])


def targets(specs):
    ys = [[SYM.index(s) for s in plan_of(sp)] for sp in specs]
    T = max(map(len, ys))
    y = torch.tensor([v + [0] * (T - len(v)) for v in ys])
    return torch.cat([torch.full((len(ys), 1), len(SYM)), y[:, :-1]], 1), y


def train_planner(data, seed, steps):
    torch.manual_seed(seed)
    net = BA.Net("addr", gate_bias=-4.0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    r, tr = random.Random(seed + 100), data.sets["train"]
    for s in range(steps):
        MF.cosine_lr(opt, 3e-3, s, steps)
        sp = [tr[r.randrange(len(tr))] for _ in range(48)]
        y_in, y = targets(sp)
        loss = F.cross_entropy(net(name_bytes(sp), y_in).transpose(1, 2), y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


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
        defs = sorted({s for p in plans for s in p if s.startswith("def:")})
        texts = dict(zip(defs, writer.write(torch.tensor([MF.SYM.index(s) for s in defs])))) if defs else {}
        srcs = [assemble(p, texts.get) for p in plans]
        refs = [assemble(plan_of(sp), MF.block_text) for sp in specs]
        outs = run_many(srcs + refs)
        got, ref = outs[:len(specs)], outs[len(specs):]
        res[k] = {"n": len(specs),
                  "exec": round(sum(g is not None for g in got) / len(specs), 4),      # module loads; refs raise too (None into +)
                  "pass1": round(sum(g == r for g, r in zip(got, ref)) / len(specs), 4),
                  "plan_exact": round(sum(p == plan_of(sp) for p, sp in zip(plans, specs)) / len(specs), 4),
                  "tree_ok": round(sum(render_expr(p) == render_expr(plan_of(sp)) for p, sp in zip(plans, specs)) / len(specs), 4),
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
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    data = Data()
    max_out = max(len(plan_of(s)) for k in data.sets for s in data.sets[k]) + 4
    BA.SYM, BA.START, BA.MAX_OUT = SYM, len(SYM), max_out
    t0 = time.perf_counter()
    planner = train_planner(data, a.seed, a.steps)
    MF.PLAN_FORMAT = "grouped"
    wdata = MF.Data()                                  # writer: same def blocks as multifn (all 15 ops)
    writer, _ = MF.train_writer(wdata, a.writer_dim, a.seed)
    names = ["test_same", "calls4", "calls5", "calls6", "calls8"]
    res = {"seed": a.seed, "steps": a.steps, "n_train": len(data.sets["train"]), "max_out": max_out,
           "max_train_plan": max(len(plan_of(s)) for s in data.sets["train"]),
           "planner_params": sum(p.numel() for p in planner.parameters()),
           "writer_params": sum(p.numel() for p in writer.parameters()),
           "wall_s": round(time.perf_counter() - t0, 1), "eval": evaluate(planner, writer, data, names)}
    print(json.dumps({k: (v if k != "eval" else {s: {m: x for m, x in r.items() if not m.startswith("example")}
                                                  for s, r in v.items()}) for k, v in res.items()}), flush=True)
    json.dump(res, open(f"{a.out}/compose_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
