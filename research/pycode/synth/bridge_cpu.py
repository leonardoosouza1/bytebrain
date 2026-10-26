"""Plan walker on CPU: does a tau-bank CfC carry the unit chain past the trained length, versus a GRU and
transformers at equal non-embedding parameters? (proposer 3, registered 2026-10-22 before any run)

Task: function NAME (Qwen-Coder BPE ids, tokenizer only, no Qwen weights) -> chain of block symbols
(op, digit slot, final) -> EOS. The line writer is a lookup (synth2.UNITS) and the head is copied from the task, so
this tests the PLAN only. Train: synth2 train (1-2 units) minus 150 pair tasks kept for the lr choice. Test: test_a,
test_num, len3 (first 500 of test_b), len4/len5/len6 (new names of seen units). Greedy decoding.
Metrics: strict synth2 grade of the rendered body, block-exact, body nats per function (teacher forced), name_info
(nats lost when the name is swapped for another of the same arity and token length).

Arms (non-embedding parameters asserted within 3% of each other, ~99k):
  tabs     2-layer transformer d=64, learned absolute positions
  trope    same, rotary positions
  gru      1-layer GRU h=128
  cfc_bank closed-form CfC h=128, 8 groups leaking with tau = 1,2,4,...,128 steps
  cfc_tau1 same cell, every tau = 1 (no slow memory)
Rule (pre-registered by the proposer): G0 = test_a and test_num >= 90% to count. C1 trope - tabs >= +20 pp at len4.
C2 min(gru, cfc_bank) - trope >= +25 pp at len4 and cfc_bank len3 >= 90%, len4 >= 70%. C3 cfc_bank -
max(gru, cfc_tau1) >= +10 pp at len4 or len5. KILL: gru and cfc_bank within 5 pp of the best transformer at len4.
Every margin must exceed 3 pp and 2x the largest seed SD.

Rerun #45 (2026-10-26, registered before any run; the 10-22 run was under-tuned: cfc_bank best lr at the grid
edge, cfc_tau1 never ran). New arms, same task, same ~99k non-embedding parameters:
  cfc_learn  same cell, tau per unit learned (init 1..128 as cfc_bank)
  cfc_stress same bank, but a group opens (rate -> 1) when its mismatch |candidate - held| is large: the state
             holds unless "stressed" (his #8 idea). rate_g = 1/tau_g + (1 - 1/tau_g) * sigmoid(a_g * mean|c - s|_g + b_g),
             init a=4, b=-8 (closed); tau params and stress gates: no weight decay, 10x lr, final values saved
--delay K puts K wait inputs (PAD id) between the name and SEP, so the plan must be read from memory K steps later.
K=0 is bit-identical to the 10-22 code (repro smoke: `arm --arm gru --lrs 3e-3 --no-extend --seeds 1` must give
test_a 0.9953). --lrs sets the grid; a best lr at the top edge, with val block_exact < 0.99, extends it x2 while <= 0.1.
Verdicts use the mean of 3 seeds; a margin counts only if > 3 pp and > 2x the largest seed SD.
Predictions: P1 tuned cfc_bank K=0 passes G0 (test_a, test_num >= 90%). P2 K=0 len4 < 15% for every recurrent arm
(no address counter; 10-22 max .0675). P3 K=48 cfc_bank test_a >= cfc_tau1 + 20 pp (cfc_tau1 has no slow leak but is
still recurrent through bb, so it can hold some memory). P4 K=48 cfc_stress >= cfc_bank + 5 pp on test_a or len3,
counted only on a metric where cfc_bank < 95%. P5 K=0 gru within 5 pp of the best tau arm on test_a. KILL for "fixed huge tau holds the
data" here: K=48 cfc_bank <= cfc_tau1 + 5 pp.

FAILED-ADAPT of #45 (2026-10-26, registered before any run). Result above: at K=48 every CfC arm fails to fit even
train (train pass1 <= .08, loss 1.0-1.7 vs .03 at K=0) while gru fits (train .999, test_a .942). Own-bug check: mask,
prefix and grader are shared with gru (which fits); K=0 is unchanged. Mechanism hypothesis H1 (write/hold coupling): a
fixed leak writes and forgets at the same rate 1/tau, so a slow group absorbs only ~n_name/tau of the name and the 48
wait steps then pull it toward c(PAD, s); the stress gate cannot help because |c - s| is LARGE during the wait (c comes
from the PAD input), so opening on mismatch overwrites the memory (learned b went to -9..-11 on the slow groups: the
optimizer kept them shut). Rival H2: same cell, only the 1200-step budget is short. New arm and test:
  cfc_gate  same bank; his #8 "holds unless stressed" with the stress read from what comes IN, not from |c - s|:
            rate_g = 1/tau_g + (1 - 1/tau_g) * sigmoid(w_g . b + c_g), one gate per group from the backbone b
            (+1032 params, within 3%), init w=0, c=-4 (closed); no weight decay, 10x lr, like the stress gates
  H2 test   cfc_bank K=48 with --steps 4800 (4x), lr 6e-3 (its K=48 best)
A1 K=48 cfc_gate test_a >= .80 (3 seeds, 1200 steps). A2 K=48 cfc_bank 4800 steps test_a <= .10 (budget is not the
cause). A3 K=0 cfc_gate test_a >= K=0 cfc_bank - 5 pp. A4 cfc_gate mean gate on name inputs > 3x mean gate on the
wait inputs (it learned to write on input and hold on PAD). KILL of H1: cfc_gate K=48 <= cfc_bank K=48 + 10 pp,
or cfc_bank 4800 steps >= .50. Positive control before: cfc_bank K=0 seed 0 lr 1e-2 --no-extend --seeds 1 must give
test_a 0.8328 (the gate code must not move the old arms).

FAILED-ADAPT 2 of #45 (2026-10-26 18:40, registered before any run). Adapt 1 result (3 seeds, mean test_a): K=48
cfc_gate .103 (train .28) vs cfc_bank .013: A1 FAIL, KILL of H1 TRUE (.103 <= .013 + .10). The gate writes on the name
and holds on the wait in the slow groups (tau 32-128: name rate 10-33x the wait rate), and the name reaches the plan
(name_info 14-21 nats, true name better in >= 99.5%), yet train stays < .34. Mechanism H3 (holder capacity): the
cfc_gate rate is 1/tau_g + (1 - 1/tau_g) * gate, so it can only RAISE the leak above the floor 1/tau_g; a group never
holds longer than its tau. Over 48 wait steps tau=128 keeps (1-1/128)^48 = .69, tau=64 .47, tau=32 .22, tau<=16 < .05:
2-3 of 8 groups (32-48 of 128 units) carry the name through the wait, while the GRU update gate can hold on every
unit (gru K=48 .942). cfc_learn K=48 (one static rate per unit, no floor) moved the slow rates UP (tau=128 group:
.0078 -> .022-.065): with one static rate, writing wins over holding. New arm:
  cfc_hold  same cell; rate_g = sigmoid(w_g . b + c_g) per group, NO floor; init w=0, c_g = logit(1/tau_g) (starts as
            cfc_bank, tau=1 at .99); a group can write (rate -> 1) on the name and hold (rate -> 0, tau -> infinity) on
            the wait: his "banco de T infinito... só muda se houver um estresse", stress read from the input. +1032
            params as cfc_gate; same lr grid, 1200 steps, 10x lr and no weight decay on the gate.
Prediction arithmetic: K=0 cfc_bank is .823 and the GRU pays 5 pp for the delay (.992 -> .942), so the reachable
K=48 ceiling is ~.77. B1 K=48 cfc_hold test_a >= .70 (3 seeds). B2 on the wait inputs >= 4 of 8 groups hold with
mean rate < 1/64 (more holders than the floor let cfc_gate have). B3 K=0 cfc_hold test_a >= K=0 cfc_bank - 5 pp
(.773). KILL of H3: K=48 cfc_hold test_a <= K=48 cfc_gate + 10 pp (.203). Positive control before: the cfc_bank K=0
repro above (0.8328) with this file.

Own-bug check, adapt 3 (2026-10-26 18:45, registered before any run). Adapt 1 at K=0: cfc_gate test_a .066 (seeds
.066/.000/.131) vs cfc_bank .823, so the gate arm fails WITHOUT any delay and its K=48 number says nothing about
holding (A3 FAIL). Adapt 1 H2: cfc_bank K=48 at 4x steps .123 (train .38/.44/.53, still rising): A2 FAIL by 2 pp,
budget is part of it, KILL-2 (>= .50) not triggered. Suspect (own bug): the 10x lr and no weight decay were copied
from the 16 scalar stress gates (sa, sb) onto wg, a 128x8 matrix read from b; the gates saturate (K=48 lr 1e-2: rate
1.0 on groups 6-7 for every input; K=0 seed 1 train .003). Fix: --gate-lr-mult 1 (wg at the base lr, still no decay;
the scalar raw/sa/sb keep 10x, so every older arm is unchanged). Arms: cfc_gate and cfc_hold at K=0 and K=48, mult 1.
C1 K=0 cfc_gate and cfc_hold test_a >= .773 (bank - 5 pp: both start at the bank's rates). C2 K=48 cfc_hold >= .70.
C3 K=48 cfc_gate <= cfc_hold - 10 pp (the 1/tau floor limits holders). KILL of the lr-mult explanation: K=0 cfc_gate
mult 1 < .50.

Usage: bridge_cpu.py smoke --out DIR
       bridge_cpu.py arm --arm NAME --out DIR [--steps 1200 --threads 2 --delay 0 --lrs 1e-3,3e-3]
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
import synth2  # noqa: E402
from synth2 import FINAL, KARG, LOOP, NUMS, PARAM, PLAIN, UNITS, grade, head, name_of, nargs  # noqa: E402

TOKENIZER = ("/media/llm-data/models/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/"
             "ea3f2471cf1b1f0db85067f1ef93848e38e88c25/tokenizer.json")
OPS = list(PARAM) + list(KARG) + list(LOOP) + list(PLAIN)
FINS = ["", "sum", "max", "len", "first", "count_gt"]
SYM = ["<pad>", "<sep>", "<eos>"] + [f"op:{o}" for o in OPS] + [f"d:{n}" for n in NUMS] + [f"fin:{f}" for f in FINS]
PAD, SEP, EOS = 0, 1, 2
ARMS = ["tabs", "trope", "gru", "cfc_bank", "cfc_tau1", "cfc_learn", "cfc_stress", "cfc_gate", "cfc_hold"]
GATE_LR_MULT = 10.0                            # lr multiplier of the input-gate matrix wg (--gate-lr-mult)
MAX_OUT = 20                                   # 6 units x 2 + final 2 + eos = 15


def blocks_of(units, final):
    out = []
    for u in units:
        if u in PARAM or u in KARG or u in LOOP or u in PLAIN:
            out.append(f"op:{u}")
        else:
            op, n = u.rsplit("_", 1)
            out += [f"op:{op}", f"d:{n}"]
    out += ["fin:count_gt", f"d:{final.rsplit('_', 1)[1]}"] if final.startswith("count_gt_") else [f"fin:{final}"]
    return out + ["<eos>"]


def parse(syms):
    """-> (units, final) or None for a malformed chain."""
    units, i = [], 0
    while i < len(syms):
        s = syms[i]
        if s.startswith("fin:"):
            f = s[4:]
            if f == "count_gt":
                if i + 1 < len(syms) and syms[i + 1].startswith("d:"):
                    f, i = f"count_gt_{syms[i + 1][2:]}", i + 1
                else:
                    return None
            return (units, f) if i + 1 < len(syms) and syms[i + 1] == "<eos>" else None
        if not s.startswith("op:"):
            return None
        o = s[3:]
        if o in PARAM:
            if not (i + 1 < len(syms) and syms[i + 1].startswith("d:")):
                return None
            u, i = f"{o}_{syms[i + 1][2:]}", i + 2
        else:
            u, i = o, i + 1
        if u not in UNITS:
            return None
        units.append(u)
    return None


def render(task, parsed):
    """the head is the task's (declared: plan only); body = lookup lines of the predicted units + final."""
    (tu, tf) = task
    if parsed is None:
        return f"def {name_of(tu, tf)}(xs):\n    return None\n"
    units, final = parsed
    return head(tu, tf) + "".join(f"    {l}\n" for u in units for l in UNITS[u]) + f"    {FINAL[final]}\n"


class Data:
    def __init__(self, delay=0):
        self.delay = delay
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(TOKENIZER)
        split = json.load(open(f"{synth2.D}/splits.json"))
        r = random.Random(5)
        train = [t for t in split["train"]]
        pairs = [i for i, (u, _) in enumerate(train) if len(u) == 2]
        val_idx = set(r.sample(pairs, 150))
        self.sets = {"train": [t for i, t in enumerate(train) if i not in val_idx],
                     "val": [t for i, t in enumerate(train) if i in val_idx],
                     "test_a": split["test_a"], "test_num": split["test_num"], "len3": split["test_b"][:500]}
        seen = [u for u in UNITS if u not in synth2.HELD]
        known = {name_of(*t) for v in split.values() for t in v}
        g = random.Random(7)
        for k in (4, 5, 6):
            names, out = set(), []
            while len(out) < 400:
                t = [[g.choice(seen) for _ in range(k)], g.choice(list(FINAL))]
                if name_of(*t) not in known and name_of(*t) not in names:
                    names.add(name_of(*t))
                    out.append(t)
            self.sets[f"len{k}"] = out
        toks = sorted({i for v in self.sets.values() for t in v for i in self.tok.encode(name_of(*t)).ids})
        self.to_c = {i: len(SYM) + c for c, i in enumerate(toks)}
        self.vocab = len(SYM) + len(toks)
        train_toks = {i for t in self.sets["train"] for i in self.tok.encode(name_of(*t)).ids}
        self.unseen_name_tokens = {k: sum(any(i not in train_toks for i in self.tok.encode(name_of(*t)).ids) for t in v)
                                   for k, v in self.sets.items()}

    def name_ids(self, task):
        return [self.to_c[i] for i in self.tok.encode(name_of(*task)).ids]

    def target_ids(self, task):
        return [SYM.index(s) for s in blocks_of(*task)]

    def prefix(self, task, name=None):
        """name ids, then delay wait inputs (PAD id: never a target, right padding is never read causally)."""
        return (name if name is not None else self.name_ids(task)) + [PAD] * self.delay

    def seq(self, task, name=None):
        return self.prefix(task, name) + [SEP] + self.target_ids(task)


# ---------------------------------------------------------------- models: x [B, L] -> logits [B, L, V]

def rope(x):
    """x [B, H, L, D] -> rotated (D even)."""
    L, D = x.shape[2], x.shape[3]
    f = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float) / D))
    a = torch.arange(L, dtype=torch.float)[:, None] * f[None]
    c, s = a.cos()[None, None], a.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, use_rope):
        super().__init__()
        self.h, self.use_rope = h, use_rope
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class Tf(nn.Module):
    def __init__(self, V, use_rope, d=64, layers=2, heads=4, ctx=96):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = None if use_rope else nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, use_rope) for _ in range(layers))
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V, bias=False)

    def forward(self, x):
        h = self.emb(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(x.shape[1]))[None]
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))


class Gru(nn.Module):
    def __init__(self, V, h=128):
        super().__init__()
        self.emb, self.rnn, self.head = nn.Embedding(V, h), nn.GRU(h, h, batch_first=True), nn.Linear(h, V, bias=False)

    def forward(self, x):
        return self.head(self.rnn(self.emb(x))[0])


class CfC(nn.Module):
    """closed-form CfC cell (Hasani et al. 2022, unit time step) with a per-group leak h += (cfc - h) / tau."""

    def __init__(self, V, taus, h=128, mode="fixed"):
        super().__init__()
        self.h, self.mode, self.g = h, mode, len(taus)
        self.emb, self.head = nn.Embedding(V, h), nn.Linear(h, V, bias=False)
        self.bb = nn.Linear(2 * h, h)
        self.f1, self.f2, self.ta, self.tb = (nn.Linear(h, h) for _ in range(4))
        self.register_buffer("inv_tau", torch.tensor(taus, dtype=torch.float).repeat_interleave(h // len(taus)).reciprocal())
        if mode == "learn":                     # rate = sigmoid(raw), init at 1/tau (tau=1 -> 0.99)
            self.raw = nn.Parameter(torch.logit(self.inv_tau.clamp(max=0.99)))
        if mode == "stress":                    # per group: opens when the mismatch is large; starts closed
            self.sa, self.sb = nn.Parameter(torch.full((self.g,), 4.0)), nn.Parameter(torch.full((self.g,), -8.0))
        if mode == "gate":                      # per group: opens on what the backbone reads; starts closed
            self.wg = nn.Linear(h, self.g)
            nn.init.zeros_(self.wg.weight)
            nn.init.constant_(self.wg.bias, -4.0)
        if mode == "hold":                      # per group, no 1/tau floor: can write (rate -> 1) or hold (rate -> 0)
            self.wg = nn.Linear(h, self.g)      # starts at the bank: rate = 1/tau_g (tau=1 -> 0.99)
            nn.init.zeros_(self.wg.weight)
            with torch.no_grad():
                self.wg.bias.copy_(torch.logit(torch.tensor(taus, dtype=torch.float).reciprocal().clamp(max=0.99)))
        self.rates = None                       # set to [] to record the leak rate per step (diagnostic)

    def forward(self, x):
        e = self.emb(x)
        s = torch.zeros(x.shape[0], self.h)
        out = []
        for t in range(x.shape[1]):
            b = F.silu(self.bb(torch.cat([e[:, t], s], -1)))
            gate = torch.sigmoid(self.ta(b) + self.tb(b))
            c = torch.tanh(self.f1(b)) * (1 - gate) + gate * torch.tanh(self.f2(b))
            if self.mode == "learn":
                rate = torch.sigmoid(self.raw)
            elif self.mode == "stress":
                mis = (c - s).abs().view(-1, self.g, self.h // self.g).mean(-1)
                rate = torch.sigmoid(self.sa * mis + self.sb).repeat_interleave(self.h // self.g, -1)
                rate = self.inv_tau + (1 - self.inv_tau) * rate
            elif self.mode == "gate":
                rate = torch.sigmoid(self.wg(b)).repeat_interleave(self.h // self.g, -1)
                rate = self.inv_tau + (1 - self.inv_tau) * rate
            elif self.mode == "hold":
                rate = torch.sigmoid(self.wg(b)).repeat_interleave(self.h // self.g, -1)
            else:
                rate = self.inv_tau
            if self.rates is not None:
                self.rates.append(rate.expand(x.shape[0], self.h).view(-1, self.g, self.h // self.g).mean(-1))
            s = s + (c - s) * rate
            out.append(s)
        return self.head(torch.stack(out, 1))


def make(arm, V, delay=0):
    if arm in ("tabs", "trope"):
        return Tf(V, arm == "trope", ctx=96 + delay)
    if arm == "gru":
        return Gru(V)
    if arm == "cfc_tau1":
        return CfC(V, [1.0] * 8)
    return CfC(V, [2.0 ** i for i in range(8)],
               mode={"cfc_bank": "fixed", "cfc_learn": "learn", "cfc_stress": "stress", "cfc_gate": "gate",
                     "cfc_hold": "hold"}[arm])


def nonemb(net):
    skip = {id(p) for m in net.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    skip |= {id(p) for p in net.head.parameters()}
    return sum(p.numel() for p in net.parameters() if id(p) not in skip)


# ---------------------------------------------------------------- train / eval

def batchify(seqs):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for r, s in enumerate(seqs):
        x[r, :len(s)] = torch.tensor(s)
    return x


def seq_nll(net, data, seqs_with_start):
    """[(seq, n_name)] -> summed NLL of the targets (positions after SEP) per sequence; right padding (causal)."""
    out = []
    for c in range(0, len(seqs_with_start), 256):
        chunk = seqs_with_start[c:c + 256]
        x = batchify([s for s, _ in chunk])
        logits = net(x[:, :-1])
        nll = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
        w = torch.zeros_like(nll)
        for r, (s, n) in enumerate(chunk):
            w[r, n:len(s) - 1] = 1.0                     # targets: blocks + eos (predicted from SEP onward)
        out += (nll * w).sum(1).tolist()
    return out


@torch.no_grad()
def greedy(net, data, tasks):
    """grouped by name length (no padding at all); -> list of symbol lists."""
    res = [None] * len(tasks)
    by = {}
    for i, t in enumerate(tasks):
        by.setdefault(len(data.name_ids(t)), []).append(i)
    for idx in by.values():
        x = torch.tensor([data.prefix(tasks[i]) + [SEP] for i in idx])
        for _ in range(MAX_OUT):
            nxt = net(x)[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], 1)
            if bool((x[:, -(MAX_OUT if x.shape[1] > MAX_OUT else 1):] == EOS).any(1).all()):
                break
        for j, i in enumerate(idx):
            gen = [SYM[k] if k < len(SYM) else "?" for k in x[j, len(data.prefix(tasks[i])) + 1:].tolist()]
            res[i] = gen[:gen.index("<eos>") + 1] if "<eos>" in gen else gen
    return res


def name_info(net, data, tasks, r):
    """same-arity, same-token-length swapped names; -> (mean nats, share where the true name is better, skipped)."""
    by = {}
    for t in data.sets["train"] + tasks:
        by.setdefault((nargs(t[0]), len(data.name_ids(t))), set()).add(name_of(*t))
    true, swap, skipped = [], [], 0
    for t in tasks:
        pool = sorted(by[(nargs(t[0]), len(data.name_ids(t)))] - {name_of(*t)})
        if not pool:
            skipped += 1
            continue
        other = [data.to_c[i] for i in data.tok.encode(r.choice(pool)).ids]
        n = len(data.prefix(t))
        true.append((data.seq(t), n))
        swap.append((data.seq(t, other), n))
    lt, ls = seq_nll(net, data, true), seq_nll(net, data, swap)
    return (round((sum(ls) - sum(lt)) / len(lt), 3), round(sum(a < b for a, b in zip(lt, ls)) / len(lt), 3), skipped)


def evaluate(net, data, names):
    net.eval()
    out = {}
    with torch.no_grad():
        for k in names:
            tasks = data.sets[k]
            gen = greedy(net, data, tasks)
            exact = [g == blocks_of(*t) for g, t in zip(gen, tasks)]
            sc = grade(tasks, [[render(t, parse(g))] for g, t in zip(gen, tasks)])
            nll = seq_nll(net, data, [(data.seq(t), len(data.prefix(t))) for t in tasks])
            out[k] = {"n": len(tasks), "pass1": round(sum(s[0] == 1.0 for s in sc) / len(tasks), 4),
                      "block_exact": round(sum(exact) / len(tasks), 4),
                      "judge_pass_not_exact": sum(s[0] == 1.0 and not e for s, e in zip(sc, exact)),
                      "body_nats": round(sum(nll) / len(nll), 3)}
        for k in ("train", "test_a", "len4"):
            if k in names or k == "train":
                out.setdefault(k, {})["name_info"] = name_info(net, data, data.sets[k][:400], random.Random(3))
    net.train()
    return out


def train(arm, data, seed, lr, steps, batch=48):
    torch.manual_seed(seed)
    net = make(arm, data.vocab, data.delay)
    nd = {"raw", "sa", "sb", "wg.weight", "wg.bias"}       # tau / stress / input gates: no decay toward rate 0.5
    groups = [{"params": [p for n, p in net.named_parameters() if n not in nd]}]
    for names, mult in (({"raw", "sa", "sb"}, 10.0), ({"wg.weight", "wg.bias"}, GATE_LR_MULT)):
        if any(n in names for n, _ in net.named_parameters()):
            groups.append({"params": [p for n, p in net.named_parameters() if n in names], "weight_decay": 0.0,
                           "lr_mult": mult})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01)
    r = random.Random(seed + 100)
    tr = data.sets["train"]
    t0, losses = time.perf_counter(), []
    for s in range(steps):
        lr_s = lr * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr_s * gp.get("lr_mult", 1.0)
        ts = [tr[r.randrange(len(tr))] for _ in range(batch)]
        x = batchify([data.seq(t) for t in ts])
        logits = net(x[:, :-1])
        ce = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
        w = torch.zeros_like(ce)
        for row, t in enumerate(ts):
            n = len(data.prefix(t))
            w[row, n:n + len(data.target_ids(t))] = 1.0
        loss = (ce * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    info = {"train_loss_last100": round(sum(losses[-100:]) / 100, 4), "train_wall_s": round(time.perf_counter() - t0, 1)}
    if hasattr(net, "raw"):                                 # learned leak rate per tau group (mean over its units)
        info["learned_rate_per_group"] = [round(v, 4) for v in torch.sigmoid(net.raw).view(net.g, -1).mean(1).tolist()]
    if hasattr(net, "sa"):
        info["stress_a"], info["stress_b"] = [round(v, 3) for v in net.sa.tolist()], [round(v, 3) for v in net.sb.tolist()]
    if isinstance(net, CfC):
        info["rate_by_input"] = rate_by_input(net, data, data.sets["val"])
    return net, info


@torch.no_grad()
def rate_by_input(net, data, tasks):
    """mean leak rate per tau group, split by what the cell reads: name tokens, wait (delay PAD) inputs, SEP + plan."""
    seqs = [data.seq(t) for t in tasks]
    x = batchify(seqs)[:, :-1]
    net.rates = []
    net(x)
    r, net.rates = torch.stack(net.rates, 1), None          # [B, L, g]
    kind = torch.full(x.shape, -1)
    for row, t in enumerate(tasks):
        n = len(data.name_ids(t))
        kind[row, :n], kind[row, n:n + data.delay], kind[row, n + data.delay:len(seqs[row]) - 1] = 0, 1, 2
    return {name: [round(v, 4) for v in r[kind == k].mean(0).tolist()] if bool((kind == k).any()) else None
            for k, name in enumerate(("name", "wait", "plan"))}


def cmd_smoke(a):
    data = Data(a.delay)
    res = {"vocab": data.vocab, "sizes": {k: len(v) for k, v in data.sets.items()},
           "unseen_name_tokens": data.unseen_name_tokens}
    names = {k: {name_of(*t) for t in v} for k, v in data.sets.items()}
    res["train_test_overlap"] = {k: len(names["train"] & v) for k, v in names.items() if k != "train"}
    res["held_outside_test_num"] = {k: sum(any(u in synth2.HELD for u in t[0]) for t in v)
                                    for k, v in data.sets.items() if k != "test_num"}
    res["round_trip"] = all(data.tok.decode(data.tok.encode(name_of(*t)).ids) == name_of(*t)
                            for v in data.sets.values() for t in v)
    res["parse_round_trip"] = all(parse(blocks_of(*t)) == (t[0], t[1]) for v in data.sets.values() for t in v)
    rule, mutants = {}, {}
    for k, v in data.sets.items():
        if k == "train":
            continue
        rule[k] = sum(s[0] == 1.0 for s in grade(v, [[render(t, parse(blocks_of(*t)))] for t in v])) / len(v)
        mut = {"drop_last": lambda u: u[:-1], "reverse": lambda u: u[::-1], "repeat_first": lambda u: u[:1] + u}
        mutants[k] = {m: round(sum(s[0] == 1.0 for s in grade(v, [[render(t, (fn(t[0]), t[1]))] for t in v])) / len(v), 3)
                      for m, fn in mut.items()}
    res["rule_walker_pass1"], res["mutant_pass1"] = rule, mutants
    params = {arm: nonemb(make(arm, data.vocab, data.delay)) for arm in ARMS}
    res["nonemb_params"] = params
    res["params_within_3pct"] = max(params.values()) / min(params.values()) <= 1.03
    eq = {}
    for arm in ARMS:                                        # batched greedy == one-by-one greedy (random init)
        torch.manual_seed(0)
        net = make(arm, data.vocab, data.delay).eval()
        ts = data.sets["len4"][:100] + data.sets["test_a"][:100]
        b = greedy(net, data, ts)
        eq[arm] = sum(greedy(net, data, [t])[0] == g for t, g in zip(ts, b)) / len(ts)
    res["batch_equals_single"] = eq
    print(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/smoke.json", "w"), indent=1)


def cmd_arm(a):
    data = Data(a.delay)
    os.makedirs(a.out, exist_ok=True)
    grid = {}
    def best():
        def key(k):
            nats = grid[k][2]["body_nats"]
            return (grid[k][2]["block_exact"], -(nats if math.isfinite(nats) else float("inf")))
        return max(grid, key=key)
    todo = [float(x) for x in a.lrs.split(",")]
    while todo:                                             # lr chosen on val pairs only, seed 0
        lr = todo.pop(0)
        net, info = train(a.arm, data, 0, lr, a.steps)
        grid[lr] = (net, info, evaluate(net, data, ["val"])["val"])
        print(json.dumps({"arm": a.arm, "delay": a.delay, "lr": lr, **info, "val": grid[lr][2]}), flush=True)
        if (not todo and not a.no_extend and best() == max(grid) and grid[best()][2]["block_exact"] < 0.99
                and max(grid) * 2 <= 0.1):
            todo.append(max(grid) * 2)                      # best at the top edge: extend the grid
    lr = best()
    runs = []
    for seed in range(a.seeds):
        net, info = (grid[lr][0], grid[lr][1]) if seed == 0 else train(a.arm, data, seed, lr, a.steps)
        ev = evaluate(net, data, ["train", "test_a", "test_num", "len3", "len4", "len5", "len6"])
        runs.append({"arm": a.arm, "delay": a.delay, "seed": seed, "lr": lr, "nonemb_params": nonemb(net), **info, "eval": ev})
        print(json.dumps({"arm": a.arm, "seed": seed, **{k: v.get("pass1") for k, v in ev.items()},
                          "nats_len4": ev["len4"]["body_nats"], "name_info_test_a": ev["test_a"]["name_info"]}), flush=True)
        json.dump({"grid": {str(k): v[2] for k, v in grid.items()}, "runs": runs}, open(f"{a.out}/arm_{a.arm}_d{a.delay}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "arm"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--lrs", default="1e-3,3e-3")
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gate-lr-mult", type=float, default=10.0, help="lr multiplier of wg (cfc_gate, cfc_hold)")
    a = ap.parse_args()
    global GATE_LR_MULT
    GATE_LR_MULT = a.gate_lr_mult
    torch.set_num_threads(a.threads)
    {"smoke": cmd_smoke, "arm": cmd_arm}[a.cmd](a)


if __name__ == "__main__":
    main()
