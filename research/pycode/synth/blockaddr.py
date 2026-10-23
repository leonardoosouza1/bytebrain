"""Learned address table over byte groups (registered 2026-10-23, CPU).

User idea: do not put meaning in one byte; bytes form groups, an addressing table ties each group to its place, a
liquid integrator carries the address, small parts together write the program.

Model (name bytes -> chain of block symbols -> lookup writer -> executable Python, bwrap grade):
  local   byte embedding + 2 causal-free local convs (window 5): sees only neighbours, never absolute position
  gate    g_i = sigmoid(w . h_i): learned "a group ends here" (no `_then_` rule anywhere)
  address a_i = sum_{j<=i} g_j: integrator with infinite tau, changes only on a boundary event
  reader  decoder step t keeps its own counter c_t; reads r_t = sum_i softmax(-(a_i - c_t)^2 / s) h_i
  decoder GRU cell over [r_t, prev symbol] -> symbol logits and advance p_t; c_{t+1} = c_t + p_t
Arms: addr (above), abspos (reader on absolute byte index i / 8 instead of a_i: the known failing control),
      addr_ter (addr with ternary weights in convs, gate and GRU, straight-through).
Train: bridge_cpu train split (1-2 units); test test_a, test_num, len3..len6 (same sets as bridge_cpu).
Prediction: addr len3 >= 60% and len4 >= 40% (bridge best: 18 / 7); abspos len3 <= 20%; addr_ter within 10 pp of addr.
KILL: addr len3 < 25%.

Usage: blockaddr.py --arm addr --out DIR [--steps 1500 --seed 0 --threads 2]
"""

import argparse
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
from bridge_cpu import MAX_OUT, SYM, Data, blocks_of, parse, render  # noqa: E402
from synth2 import grade, name_of  # noqa: E402

BYTES = sorted(set(b"abcdefghijklmnopqrstuvwxyz_0123456789"))
B2I = {b: i + 1 for i, b in enumerate(BYTES)}          # 0 = pad
START = len(SYM)                                        # decoder start symbol id


def tern(w):
    s = w.abs().mean().clamp(min=1e-8)
    return w + ((w / s).round().clamp(-1, 1) * s - w).detach()


class Net(nn.Module):
    def __init__(self, arm, d=64, gate_bias=-2.0):
        super().__init__()
        self.arm, self.q = arm, arm.endswith("_ter")
        self.emb = nn.Embedding(len(BYTES) + 1, d)
        self.c1, self.c2 = nn.Conv1d(d, d, 5, padding=2), nn.Conv1d(d, d, 5, padding=2)
        self.gate = nn.Linear(d, 1)
        nn.init.constant_(self.gate.bias, gate_bias)
        self.sym = nn.Embedding(len(SYM) + 1, d)
        self.cell = nn.GRUCell(2 * d, 2 * d)
        self.out = nn.Linear(2 * d, len(SYM))
        self.adv = nn.Linear(2 * d, 1)
        self.log_s = nn.Parameter(torch.tensor(math.log(0.3)))

    def W(self, m):
        return tern(m.weight) if self.q else m.weight

    def encode(self, x):
        """x [B, L] byte ids -> h [B, L, d], address [B, L], mask [B, L]."""
        m = x > 0
        h = self.emb(x).transpose(1, 2)
        h = F.gelu(F.conv1d(h, self.W(self.c1), self.c1.bias, padding=2))
        h = h + F.gelu(F.conv1d(h, self.W(self.c2), self.c2.bias, padding=2))
        h = h.transpose(1, 2) * m[..., None]
        if self.arm.startswith("abspos"):
            a = torch.arange(x.shape[1], dtype=torch.float)[None].expand(x.shape[0], -1) / 8.0
        else:
            g = torch.sigmoid(F.linear(h, self.W(self.gate), self.gate.bias))[..., 0] * m
            a = g.cumsum(1)
        return h, a, m

    def step(self, h, a, m, c, prev, st):
        w = -(a - c[:, None]) ** 2 / self.log_s.exp() ** 2
        w = torch.softmax(w.masked_fill(~m, -1e9), 1)
        r = (w[..., None] * h).sum(1)
        inp = torch.cat([r, self.sym(prev)], 1)
        if self.q:                                   # GRU with ternary weights (same equations, quantized matrices)
            gi = F.linear(inp, tern(self.cell.weight_ih), self.cell.bias_ih)
            gh = F.linear(st, tern(self.cell.weight_hh), self.cell.bias_hh)
            ir, iz, in_ = gi.chunk(3, 1)
            hr, hz, hn = gh.chunk(3, 1)
            rg, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            st = (1 - z) * torch.tanh(in_ + rg * hn) + z * st
        else:
            st = self.cell(inp, st)
        return self.out(st), torch.sigmoid(self.adv(st))[:, 0], st

    def forward(self, x, y_in):
        """teacher forced: y_in [B, T] previous symbols (START first) -> logits [B, T, V]."""
        h, a, m = self.encode(x)
        B = x.shape[0]
        c, st, outs = torch.zeros(B), torch.zeros(B, self.cell.hidden_size), []
        for t in range(y_in.shape[1]):
            lo, p, st = self.step(h, a, m, c, y_in[:, t], st)
            outs.append(lo)
            c = c + p
        return torch.stack(outs, 1)

    @torch.no_grad()
    def greedy(self, x):
        h, a, m = self.encode(x)
        B = x.shape[0]
        c, st, prev, out = torch.zeros(B), torch.zeros(B, self.cell.hidden_size), torch.full((B,), START), []
        for _ in range(MAX_OUT):
            lo, p, st = self.step(h, a, m, c, prev, st)
            prev = lo.argmax(-1)
            out.append(prev)
            c = c + p
        return torch.stack(out, 1)


def name_bytes(tasks):
    ns = [[B2I[b] for b in name_of(*t).encode()] for t in tasks]
    L = max(map(len, ns))
    return torch.tensor([n + [0] * (L - len(n)) for n in ns])


def targets(tasks):
    ys = [[SYM.index(s) for s in blocks_of(*t)] for t in tasks]
    T = max(map(len, ys))
    y = torch.tensor([v + [0] * (T - len(v)) for v in ys])        # 0 = <pad>, ignored
    y_in = torch.cat([torch.full((len(ys), 1), START), y[:, :-1]], 1)
    return y_in, y


def evaluate(net, data, names):
    net.eval()
    res = {}
    for k in names:
        tasks = data.sets[k]
        gen = []
        for i in range(0, len(tasks), 256):
            for row in net.greedy(name_bytes(tasks[i:i + 256])).tolist():
                s = [SYM[j] for j in row]
                gen.append(s[:s.index("<eos>") + 1] if "<eos>" in s else s)
        sc = grade(tasks, [[render(t, parse(g))] for g, t in zip(gen, tasks)])
        res[k] = {"n": len(tasks), "pass1": round(sum(s[0] == 1.0 for s in sc) / len(tasks), 4),
                  "block_exact": round(sum(g == blocks_of(*t) for g, t in zip(gen, tasks)) / len(tasks), 4)}
    net.train()
    return res


def boundaries(net, data, task):
    """debug: where the learned gate fires on one name (bytes with g > 0.5)."""
    x = name_bytes([task])
    with torch.no_grad():
        h, a, m = net.encode(x)
        g = torch.diff(a[0], prepend=torch.zeros(1))
    n = name_of(*task)
    return "".join(ch + ("|" if v > 0.5 else "") for ch, v in zip(n, g.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["addr", "abspos", "addr_ter"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gate-bias", type=float, default=-2.0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    data = Data()
    torch.manual_seed(a.seed)
    net = Net(a.arm, gate_bias=a.gate_bias)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    r, tr, t0, losses = random.Random(a.seed + 100), data.sets["train"], time.perf_counter(), []
    for s in range(a.steps):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
        ts = [tr[r.randrange(len(tr))] for _ in range(48)]
        y_in, y = targets(ts)
        loss = F.cross_entropy(net(name_bytes(ts), y_in).transpose(1, 2), y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (s + 1) % 250 == 0:
            print(json.dumps({"step": s + 1, "loss": round(sum(losses[-100:]) / 100, 4)}), flush=True)
    res = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "params": sum(p.numel() for p in net.parameters()),
           "train_wall_s": round(time.perf_counter() - t0, 1),
           "eval": evaluate(net, data, ["val", "test_a", "test_num", "len3", "len4", "len5", "len6"]),
           "gate_examples": [boundaries(net, data, t) for t in data.sets["len4"][:3]]}
    print(json.dumps(res), flush=True)
    res["gate_bias"] = a.gate_bias
    json.dump(res, open(f"{a.out}/blockaddr_{a.arm}_s{a.seed}_gb{a.gate_bias:g}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
