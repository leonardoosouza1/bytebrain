"""Byte writer guided by the learned address planner (ledger #17, registered 2026-10-23, CPU).

Pipeline: name bytes -> blockaddr planner (learned boundary gate + cumulative address) -> block symbols
-> byte WRITER (small GRU, conditioned on the block's op + digit) writes each block's text byte by byte
-> head + blocks -> bwrap: executable? strict grade on 40 inputs.

The writer replaces the lookup table. It is trained only on blocks of TRAIN tasks: held units (add_2, times_3, ...)
never appear as text, so test_num asks it to compose op + unseen digit pairs into text it never saw.
Arms: --writer-dim {32,64,128}, --ter (ternary GRU/out weights, straight-through).
Metrics per split: exec = function defines and runs on all inputs without exception; pass1 = strict grade;
plan_exact = planner chain exact; text_exact = all written blocks equal the reference text.
Prediction (d=64 fp): test_a/len3..len6 pass1 within 3 pp of the lookup planner; test_num >= 80%
(digit copy); exec >= 98% everywhere. Ternary within 10 pp.

Usage: bytewriter.py --out DIR [--writer-dim 64 --ter --seed 0 --planner-seed 4]
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
from bridge_cpu import SYM, Data, blocks_of, parse  # noqa: E402
from synth2 import FINAL, UNITS, grade, head, name_of, nargs, run_many  # noqa: E402

CH = sorted(set("".join(l + "\n" for v in UNITS.values() for l in v) + "".join(v + "\n" for v in FINAL.values())))
END = len(CH)                       # end-of-block byte
C2I = {c: i for i, c in enumerate(CH)}
OPS = sorted({u.rsplit("_", 1)[0] if u.rsplit("_", 1)[-1].isdigit() else u for u in UNITS} |
             {f"fin:{f.rsplit('_', 1)[0] if f.rsplit('_', 1)[-1].isdigit() else f}" for f in FINAL})
DIG = ["-"] + [str(n) for n in range(10)]


def block_key(unit_or_final, is_final):
    """-> (op id, digit id)."""
    x = unit_or_final
    a, _, b = x.rpartition("_")
    op, d = (a, b) if b.isdigit() else (x, "-")
    return OPS.index(f"fin:{op}" if is_final else op), DIG.index(d)


def block_text(unit_or_final, is_final):
    return (FINAL[unit_or_final] + "\n") if is_final else "".join(l + "\n" for l in UNITS[unit_or_final])


def tern(w):
    s = w.abs().mean().clamp(min=1e-8)
    return w + ((w / s).round().clamp(-1, 1) * s - w).detach()


class Writer(nn.Module):
    def __init__(self, d, ter):
        super().__init__()
        self.ter = ter
        self.op, self.dig = nn.Embedding(len(OPS), d), nn.Embedding(len(DIG), d)
        self.ch = nn.Embedding(len(CH) + 2, d)          # + END + START
        self.cell = nn.GRUCell(2 * d, d)
        self.out = nn.Linear(d, len(CH) + 1)

    def W(self, w):
        return tern(w) if self.ter else w

    def _cell(self, x, h):
        gi = F.linear(x, self.W(self.cell.weight_ih), self.cell.bias_ih)
        gh = F.linear(h, self.W(self.cell.weight_hh), self.cell.bias_hh)
        ir, iz, inn = gi.chunk(3, 1)
        hr, hz, hn = gh.chunk(3, 1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        return (1 - z) * torch.tanh(inn + r * hn) + z * h

    def logits(self, h):
        return F.linear(h, self.W(self.out.weight), self.out.bias)

    def forward(self, op, dg, y_in):
        cond = self.op(op) + self.dig(dg)
        h, outs = torch.zeros(op.shape[0], self.cell.hidden_size), []
        for t in range(y_in.shape[1]):
            h = self._cell(torch.cat([cond, self.ch(y_in[:, t])], 1), h)
            outs.append(self.logits(h))
        return torch.stack(outs, 1)

    @torch.no_grad()
    def write(self, op, dg, max_len=160):
        cond = self.op(op) + self.dig(dg)
        B = op.shape[0]
        h, prev = torch.zeros(B, self.cell.hidden_size), torch.full((B,), len(CH) + 1)
        out = []
        for _ in range(max_len):
            h = self._cell(torch.cat([cond, self.ch(prev)], 1), h)
            prev = self.logits(h).argmax(-1)
            out.append(prev)
        res = []
        for row in torch.stack(out, 1).tolist():
            s = row[:row.index(END)] if END in row else row
            res.append("".join(CH[i] for i in s))
        return res


def blocks_of_task(task):
    units, final = task
    return [(u, False) for u in units] + [(final, True)]


def train_writer(tasks, d, ter, seed, steps=1500, batch=64):
    torch.manual_seed(seed)
    pool = sorted({b for t in tasks for b in blocks_of_task(t)})
    net = Writer(d, ter)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    r = random.Random(seed)
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = 3e-3 * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        bs = [pool[r.randrange(len(pool))] for _ in range(batch)]
        keys = [block_key(*b) for b in bs]
        ys = [[C2I[c] for c in block_text(*b)] + [END] for b in bs]
        T = max(map(len, ys))
        y = torch.tensor([v + [-100] * (T - len(v)) for v in ys])
        y_in = torch.cat([torch.full((batch, 1), len(CH) + 1), y[:, :-1].clamp(min=0)], 1)
        lo = net(torch.tensor([k[0] for k in keys]), torch.tensor([k[1] for k in keys]), y_in)
        loss = F.cross_entropy(lo.transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net, pool, round(loss.item(), 5)


def train_planner(data, seed, steps=1500):
    torch.manual_seed(seed)
    net = BA.Net("addr", gate_bias=-4.0)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    r, tr = random.Random(seed + 100), data.sets["train"]
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = 3e-3 * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        ts = [tr[r.randrange(len(tr))] for _ in range(48)]
        y_in, y = BA.targets(ts)
        loss = F.cross_entropy(net(BA.name_bytes(ts), y_in).transpose(1, 2), y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net


def evaluate(planner, writer, data, names):
    planner.eval()
    writer.eval()
    res = {}
    for k in names:
        tasks = data.sets[k]
        t0 = time.perf_counter()
        plans = []
        for i in range(0, len(tasks), 256):
            for row in planner.greedy(BA.name_bytes(tasks[i:i + 256])).tolist():
                s = [SYM[j] for j in row]
                plans.append(parse(s[:s.index("<eos>") + 1] if "<eos>" in s else s))
        blocks = [blocks_of_task(p) if p is not None else [] for p in plans]
        flat = [b for bl in blocks for b in bl]
        texts = writer.write(torch.tensor([block_key(*b)[0] for b in flat]),
                             torch.tensor([block_key(*b)[1] for b in flat])) if flat else []
        gen_s = time.perf_counter() - t0
        srcs, text_ok, j = [], [], 0
        for t, bl in zip(tasks, blocks):
            written = texts[j:j + len(bl)]
            j += len(bl)
            body = "".join("".join(f"    {l}\n" for l in w.split("\n") if l) for w in written)
            srcs.append(head(*t) + (body if bl else "    return None\n"))
            text_ok.append(bool(bl) and all(w == block_text(*b) for w, b in zip(written, bl)))
        runs = run_many([(s, name_of(*t), nargs(t[0])) for s, t in zip(srcs, tasks)])
        sc = grade(tasks, [[s] for s in srcs])
        parse_ok = 0
        for s_ in srcs:
            try:
                ast.parse(s_)
                parse_ok += 1
            except SyntaxError:
                pass
        res[k] = {"n": len(tasks),
                  "parse": round(parse_ok / len(tasks), 4),
                  "gen_s": round(gen_s, 3),
                  "correct_per_s": round(sum(s[0] == 1.0 for s in sc) / max(gen_s, 1e-9), 1),
                  "exec": round(sum(o is not None and "!" not in o for o in runs) / len(tasks), 4),
                  "pass1": round(sum(s[0] == 1.0 for s in sc) / len(tasks), 4),
                  "plan_exact": round(sum(p is not None and blocks_of(*p) == blocks_of(*t)
                                          for p, t in zip(plans, tasks)) / len(tasks), 4),
                  "text_exact": round(sum(text_ok) / len(tasks), 4),
                  "example": srcs[0]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--writer-dim", type=int, default=64)
    ap.add_argument("--ter", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner-seed", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    data = Data()
    t0 = time.perf_counter()
    planner = train_planner(data, a.planner_seed)
    writer, pool, wloss = train_writer(data.sets["train"], a.writer_dim, a.ter, a.seed)
    seen_text = {block_text(*b) for b in pool}
    res = {"writer_dim": a.writer_dim, "ter": a.ter, "seed": a.seed, "planner_seed": a.planner_seed,
           "planner_params": sum(p.numel() for p in planner.parameters()),
           "writer_params": sum(p.numel() for p in writer.parameters()),
           "writer_train_blocks": len(pool), "writer_loss": wloss, "wall_s": round(time.perf_counter() - t0, 1),
           "eval": evaluate(planner, writer, data, ["test_a", "test_num", "len3", "len4", "len5", "len6"])}
    res["held_text_in_writer_train"] = sum(block_text(u, False) in seen_text for u in __import__("synth2").HELD)
    print(json.dumps({k: (v if k != "eval" else {s: {m: x for m, x in r.items() if m != "example"} for s, r in v.items()})
                      for k, v in res.items()}), flush=True)
    tag = f"d{a.writer_dim}{'_ter' if a.ter else ''}_s{a.seed}"
    json.dump(res, open(f"{a.out}/bytewriter_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
