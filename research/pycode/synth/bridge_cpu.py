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

Usage: bridge_cpu.py smoke --out DIR
       bridge_cpu.py arm --arm NAME --out DIR [--steps 1200 --threads 2]
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
ARMS = ["tabs", "trope", "gru", "cfc_bank", "cfc_tau1"]
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
    def __init__(self):
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

    def seq(self, task, name=None):
        return (name if name is not None else self.name_ids(task)) + [SEP] + self.target_ids(task)


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

    def __init__(self, V, taus, h=128):
        super().__init__()
        self.h = h
        self.emb, self.head = nn.Embedding(V, h), nn.Linear(h, V, bias=False)
        self.bb = nn.Linear(2 * h, h)
        self.f1, self.f2, self.ta, self.tb = (nn.Linear(h, h) for _ in range(4))
        self.register_buffer("inv_tau", torch.tensor(taus, dtype=torch.float).repeat_interleave(h // len(taus)).reciprocal())

    def forward(self, x):
        e = self.emb(x)
        s = torch.zeros(x.shape[0], self.h)
        out = []
        for t in range(x.shape[1]):
            b = F.silu(self.bb(torch.cat([e[:, t], s], -1)))
            gate = torch.sigmoid(self.ta(b) + self.tb(b))
            c = torch.tanh(self.f1(b)) * (1 - gate) + gate * torch.tanh(self.f2(b))
            s = s + (c - s) * self.inv_tau
            out.append(s)
        return self.head(torch.stack(out, 1))


def make(arm, V):
    if arm in ("tabs", "trope"):
        return Tf(V, arm == "trope")
    if arm == "gru":
        return Gru(V)
    return CfC(V, [2.0 ** i for i in range(8)] if arm == "cfc_bank" else [1.0] * 8)


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
        x = torch.tensor([data.name_ids(tasks[i]) + [SEP] for i in idx])
        for _ in range(MAX_OUT):
            nxt = net(x)[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], 1)
            if bool((x[:, -(MAX_OUT if x.shape[1] > MAX_OUT else 1):] == EOS).any(1).all()):
                break
        for j, i in enumerate(idx):
            gen = [SYM[k] if k < len(SYM) else "?" for k in x[j, len(data.name_ids(tasks[i])) + 1:].tolist()]
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
        n = len(data.name_ids(t))
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
            nll = seq_nll(net, data, [(data.seq(t), len(data.name_ids(t))) for t in tasks])
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
    net = make(arm, data.vocab)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    r = random.Random(seed + 100)
    tr = data.sets["train"]
    t0, losses = time.perf_counter(), []
    for s in range(steps):
        lr_s = lr * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr_s
        ts = [tr[r.randrange(len(tr))] for _ in range(batch)]
        x = batchify([data.seq(t) for t in ts])
        logits = net(x[:, :-1])
        ce = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
        w = torch.zeros_like(ce)
        for row, t in enumerate(ts):
            n = len(data.name_ids(t))
            w[row, n:n + len(data.target_ids(t))] = 1.0
        loss = (ce * w).sum() / w.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return net, {"train_loss_last100": round(sum(losses[-100:]) / 100, 4), "train_wall_s": round(time.perf_counter() - t0, 1)}


def cmd_smoke(a):
    data = Data()
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
    params = {arm: nonemb(make(arm, data.vocab)) for arm in ARMS}
    res["nonemb_params"] = params
    res["params_within_3pct"] = max(params.values()) / min(params.values()) <= 1.03
    eq = {}
    for arm in ARMS:                                        # batched greedy == one-by-one greedy (random init)
        torch.manual_seed(0)
        net = make(arm, data.vocab).eval()
        ts = data.sets["len4"][:100] + data.sets["test_a"][:100]
        b = greedy(net, data, ts)
        eq[arm] = sum(greedy(net, data, [t])[0] == g for t, g in zip(ts, b)) / len(ts)
    res["batch_equals_single"] = eq
    print(json.dumps(res, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/smoke.json", "w"), indent=1)


def cmd_arm(a):
    data = Data()
    os.makedirs(a.out, exist_ok=True)
    grid = {}
    for lr in (1e-3, 3e-3):                                 # lr chosen on val pairs only, seed 0
        net, info = train(a.arm, data, 0, lr, a.steps)
        grid[lr] = (net, info, evaluate(net, data, ["val"])["val"])
        print(json.dumps({"arm": a.arm, "lr": lr, **info, "val": grid[lr][2]}), flush=True)
    lr = max(grid, key=lambda k: (grid[k][2]["block_exact"], -grid[k][2]["body_nats"]))
    runs = []
    for seed in (0, 1, 2):
        net, info = (grid[lr][0], grid[lr][1]) if seed == 0 else train(a.arm, data, seed, lr, a.steps)
        ev = evaluate(net, data, ["train", "test_a", "test_num", "len3", "len4", "len5", "len6"])
        runs.append({"arm": a.arm, "seed": seed, "lr": lr, "nonemb_params": nonemb(net), **info, "eval": ev})
        print(json.dumps({"arm": a.arm, "seed": seed, **{k: v.get("pass1") for k, v in ev.items()},
                          "nats_len4": ev["len4"]["body_nats"], "name_info_test_a": ev["test_a"]["name_info"]}), flush=True)
        json.dump({"grid": {str(k): v[2] for k, v in grid.items()}, "runs": runs}, open(f"{a.out}/arm_{a.arm}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "arm"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    {"smoke": cmd_smoke, "arm": cmd_arm}[a.cmd](a)


if __name__ == "__main__":
    main()
