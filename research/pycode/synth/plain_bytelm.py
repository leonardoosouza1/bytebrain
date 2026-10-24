"""Reference arm R2 for the writer battery (ledger #18, registered 2026-10-23, CPU).

Plain byte seq2seq of matched size, NO address: name bytes -> bidirectional GRU encoder -> byte GRU decoder with
ordinary dot attention -> the whole function body, byte by byte. Same train split (1-2 units), same splits,
greedy k=1, no selector, same bwrap grade as bytewriter.py. Head line copied from the task, as in bytewriter.
Params ~185k vs planner 149k + writer d64 45k = 194k.
Prediction: test_a high (>= 80%), len3 <= 30%, len4..6 near 0: it never saw bodies longer than 2 units.

Usage: plain_bytelm.py --out DIR [--seed 0 --steps 3000]
"""

import argparse
import ast
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import blockaddr as BA  # noqa: E402
from bridge_cpu import Data  # noqa: E402
from synth2 import FINAL, UNITS, grade, head, name_of, nargs, run_many  # noqa: E402

CH = sorted(set("".join(l + "\n" for v in UNITS.values() for l in v) + "".join(v + "\n" for v in FINAL.values())))
END = len(CH)


def body_of(task):
    units, final = task
    return "".join(l + "\n" for u in units for l in UNITS[u]) + FINAL[final] + "\n"


class Net(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.emb = nn.Embedding(len(BA.BYTES) + 1, d)
        self.enc = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.ch = nn.Embedding(len(CH) + 2, d)
        self.cell = nn.GRUCell(d + 2 * d, 2 * d)
        self.out = nn.Linear(4 * d, len(CH) + 1)

    def encode(self, x):
        e, _ = self.enc(self.emb(x))
        return e, x > 0

    def step(self, e, m, prev, h, ctx):
        h = self.cell(torch.cat([self.ch(prev), ctx], 1), h)
        w = torch.softmax(torch.einsum("bld,bd->bl", e, h).masked_fill(~m, -1e9), 1)
        ctx = torch.einsum("bl,bld->bd", w, e)
        return self.out(torch.cat([h, ctx], 1)), h, ctx

    def forward(self, x, y_in):
        e, m = self.encode(x)
        B = x.shape[0]
        h, ctx, outs = torch.zeros(B, e.shape[2]), torch.zeros(B, e.shape[2]), []
        for t in range(y_in.shape[1]):
            lo, h, ctx = self.step(e, m, y_in[:, t], h, ctx)
            outs.append(lo)
        return torch.stack(outs, 1)

    @torch.no_grad()
    def write(self, x, max_len):
        e, m = self.encode(x)
        B = x.shape[0]
        h, ctx = torch.zeros(B, e.shape[2]), torch.zeros(B, e.shape[2])
        prev, out = torch.full((B,), len(CH) + 1), []
        for _ in range(max_len):
            lo, h, ctx = self.step(e, m, prev, h, ctx)
            prev = lo.argmax(-1)
            out.append(prev)
        res = []
        for row in torch.stack(out, 1).tolist():
            row = row[:row.index(END)] if END in row else row
            res.append("".join(CH[i] for i in row))
        return res


def evaluate(net, data, names):
    net.eval()
    res = {}
    for k in names:
        tasks = data.sets[k]
        t0 = time.perf_counter()
        max_len = max(len(body_of(t)) for t in tasks) + 40
        bodies = []
        for i in range(0, len(tasks), 256):
            bodies += net.write(BA.name_bytes(tasks[i:i + 256]), max_len)
        gen_s = time.perf_counter() - t0
        srcs = [head(*t) + "".join(f"    {l}\n" for l in b.split("\n") if l) for t, b in zip(tasks, bodies)]
        parse_ok = 0
        for s in srcs:
            try:
                ast.parse(s)
                parse_ok += 1
            except SyntaxError:
                pass
        runs = run_many([(s, name_of(*t), nargs(t[0])) for s, t in zip(srcs, tasks)])
        sc = grade(tasks, [[s] for s in srcs])
        ok = sum(s[0] == 1.0 for s in sc)
        res[k] = {"n": len(tasks), "parse": round(parse_ok / len(tasks), 4),
                  "exec": round(sum(o is not None and "!" not in o for o in runs) / len(tasks), 4),
                  "pass1": round(ok / len(tasks), 4),
                  "text_exact": round(sum(b == body_of(t) for b, t in zip(bodies, tasks)) / len(tasks), 4),
                  "gen_s": round(gen_s, 3), "correct_per_s": round(ok / max(gen_s, 1e-9), 1), "example": srcs[0]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    data = Data()
    torch.manual_seed(a.seed)
    net = Net(a.dim)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    r, tr, t0 = random.Random(a.seed + 100), data.sets["train"], time.perf_counter()
    for s in range(a.steps):
        for gp in opt.param_groups:
            gp["lr"] = 3e-3 * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
        ts = [tr[r.randrange(len(tr))] for _ in range(48)]
        ys = [[CH.index(c) for c in body_of(t)] + [END] for t in ts]
        T = max(map(len, ys))
        y = torch.tensor([v + [-100] * (T - len(v)) for v in ys])
        y_in = torch.cat([torch.full((len(ts), 1), len(CH) + 1), y[:, :-1].clamp(min=0)], 1)
        loss = F.cross_entropy(net(BA.name_bytes(ts), y_in).transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (s + 1) % 500 == 0:
            print(json.dumps({"step": s + 1, "loss": round(loss.item(), 4)}), flush=True)
    res = {"arm": "plain_bytelm", "seed": a.seed, "steps": a.steps,
           "params": sum(p.numel() for p in net.parameters()), "train_wall_s": round(time.perf_counter() - t0, 1),
           "eval": evaluate(net, data, ["test_a", "test_num", "len3", "len4", "len5", "len6"])}
    print(json.dumps({k: (v if k != "eval" else {s: {m: x for m, x in q.items() if m != "example"}
                                                  for s, q in v.items()}) for k, v in res.items()}), flush=True)
    json.dump(res, open(f"{a.out}/plain_bytelm_d{a.dim}_s{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
