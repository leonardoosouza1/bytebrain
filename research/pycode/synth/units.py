"""Representation unit at equal compute: does leaving bytes pay at 3M parameters?

Same synth2 data, same strict judge, same model shape (non-embedding parameters identical), same training loop and
schedule; only the unit differs:
  bytes : 256-symbol vocabulary (the current thesis)
  qwen  : Qwen2.5-Coder byte-level BPE tokens, restricted to the ids that occur in synth2 (train text + prompts)
Equal FLOPs: every step sees 24576 units (bytes 48 x 512, tokens 128 x 192) for the same number of steps
(6 * N * units, N equal by construction). The token arm therefore sees ~3x more TEXT for the same compute; the
milestone near 1/ratio of the run is the equal-TEXT point.
Evaluated at milestones on test_a (unseen pairs), test_num (held number->slot bindings), test_b (3 units, unseen
length): strict pass@1, decode steps and wall time.

Usage:
  units.py smoke --unit {bytes,qwen}                       (CPU+GPU light: round trip, oracle grade, 30 steps)
  units.py train --unit {bytes,qwen} --out-dir DIR [--steps 3000]
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import synth2  # noqa: E402
from synth2 import grade, name_of, source  # noqa: E402
from synth import summary  # noqa: E402
from eval_samples import cut_function  # noqa: E402
from src.model import ByteGPT  # noqa: E402

TOKENIZER = ("/media/llm-data/models/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/"
             "ea3f2471cf1b1f0db85067f1ef93848e38e88c25/tokenizer.json")
SHAPE = {"bytes": {"ctx": 512, "batch": 48, "gen": 380}, "qwen": {"ctx": 192, "batch": 128, "gen": 160}}
SPLITS = ["test_a", "test_num", "test_b"]
MILESTONES = [0.125, 0.25, 0.35, 0.5, 1.0]
# every prompt starts with the tail of a previous function: in the corpus 80% of `def` follow the merged BPE token
# `)\n\n`, so a bare "\n\n" prompt would be off-distribution for the token arm (identical for both arms)
PREFIX = "    return sum(xs)\n\n"


class Bytes:
    """bytes, restricted (like the token arm) to the symbols seen in the training text."""

    def __init__(self, texts):
        seen = sorted({b for t in texts for b in t.encode()})
        self.to_c = {b: c for c, b in enumerate(seen)}
        self.to_q = seen
        self.vocab = len(seen)

    def encode(self, text):
        return [self.to_c[b] for b in text.encode()]

    def decode(self, ids):
        return bytes(self.to_q[i] for i in ids).decode("utf-8", "replace")

    def count(self, text):
        return len(text.encode())


class Qwen:
    """BPE ids of the full Qwen tokenizer, remapped to a compact vocabulary of the ids seen in synth2."""

    def __init__(self, texts):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(TOKENIZER)
        seen = sorted({i for t in texts for i in self.tok.encode(t).ids})
        self.to_c = {i: c for c, i in enumerate(seen)}
        self.to_q = seen
        self.vocab = len(seen)

    def encode(self, text):
        return [self.to_c[i] for i in self.tok.encode(text).ids]     # KeyError = a token outside the vocabulary

    def decode(self, ids):
        return self.tok.decode([self.to_q[i] for i in ids])

    def count(self, text):                                   # raw tokenizer: never fails on unseen merges
        return len(self.tok.encode(text).ids)


def prompt_ids(unit, units, final):
    """the prompt is PREFIX + `def NAME(` (bytes) or the longest token prefix of PREFIX + reference source whose
    text is a prefix of it (tokens: the in-corpus tokenization of the name, never a split token at the boundary)."""
    head = f"{PREFIX}def {name_of(units, final)}("
    if isinstance(unit, Bytes):
        return unit.encode(head)
    ids = unit.encode(PREFIX + source(units, final))
    n = 0
    while n < len(ids) and head.startswith(unit.decode(ids[:n + 1])):
        n += 1
    return ids[:n]


class UnitGPT(ByteGPT):
    """ByteGPT with a vocabulary of any size (embedding and head replaced; blocks identical across units)."""

    def __init__(self, vocab, dim, layers, heads, ctx):
        super().__init__(dim=dim, n_layers=layers, n_heads=heads, context=ctx)
        self.tok = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)


def _block_cached(blk, x, kc, vc, causal):
    B, L, D = x.shape
    h = blk.ln1(x)
    q, k, v = blk.qkv(h).view(B, L, 3, blk.n_heads, D // blk.n_heads).permute(2, 0, 3, 1, 4)
    if kc is not None:
        k = torch.cat([kc, k], dim=2)
        v = torch.cat([vc, v], dim=2)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    x = x + blk.proj(a.transpose(1, 2).reshape(B, L, D))
    return x + blk.mlp(blk.ln2(x)), k, v


def stop_tables(unit):
    """per unit id, the cut_function rule (a line after the def line starting with a non-blank ends the function):
    does the unit contain a newline + non-blank, does it start with a non-blank, does it end with a newline."""
    texts = [unit.decode([i]) for i in range(unit.vocab)]
    inside = [any(t[j - 1] == "\n" and not t[j].isspace() for j in range(1, len(t))) for t in texts]
    starts = [bool(t) and not t[0].isspace() for t in texts]
    return torch.tensor(inside), torch.tensor(starts), torch.tensor([t.endswith("\n") for t in texts])


def stop_step(stop, x, prev, done):
    inside, starts, ends_nl = stop
    return done | inside[x] | (prev & starts[x]), ends_nl[x]


@torch.no_grad()
def sample(net, ids, n, temp, gen=None, stop=None):
    """KV-cached sampling of n units for a batch of same-length prompts; stops a row's growth at the context end and
    the whole batch once every row has ended its function (prompts end inside the def line)."""
    kc, vc = [None] * len(net.blocks), [None] * len(net.blocks)
    if stop is not None:
        stop = [t.to(ids.device) for t in stop]
        prev = done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    n = min(n, net.context - ids.shape[1] + 1)               # last sampled unit is never fed back
    pos, x, out = 0, ids, []
    for _ in range(n):
        L = x.shape[1]
        h = net.tok(x) + net.pos(torch.arange(pos, pos + L, device=x.device))[None]
        for i, blk in enumerate(net.blocks):
            h, kc[i], vc[i] = _block_cached(blk, h, kc[i], vc[i], L > 1)
        pos += L
        logits = net.head(net.lnf(h))[:, -1].float()
        x = logits.argmax(-1, keepdim=True) if temp == 0 else \
            torch.multinomial(torch.softmax(logits / temp, -1), 1, generator=gen)
        out.append(x)
        if stop is not None:
            done, prev = stop_step(stop, x[:, 0], prev, done)
            if len(out) % 8 == 0 and bool(done.all()):
                break
    return torch.cat(out, 1)


def evaluate(net, unit, split, shape, n=200, k=4, temp=0.8):
    tasks = split[:]
    random.Random(1).shuffle(tasks)
    tasks = tasks[:n]
    prompts = [prompt_ids(unit, u, f) for u, f in tasks]
    srcs = [[None] * k for _ in tasks]
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    t0 = time.perf_counter()
    rng = torch.Generator(device="cuda").manual_seed(1)       # eval never touches the training RNG
    stop = stop_tables(unit)
    lens = []
    net.eval()
    for idx in by_len.values():
        for c in range(0, len(idx), 256 // k):
            part = idx[c:c + 256 // k]
            ids = torch.tensor([prompts[i] for i in part for _ in range(k)], device="cuda")
            gen = sample(net, ids, shape["gen"], temp, rng, stop).cpu().tolist()
            for j, i in enumerate(part):
                for r in range(k):
                    text = unit.decode(prompts[i] + gen[j * k + r])
                    srcs[i][r] = cut_function(text[len(PREFIX):] if text.startswith(PREFIX) else text)[0]
                    lens.append(unit.count(srcs[i][r]))
    wall = time.perf_counter() - t0
    res = summary(grade(tasks, srcs))
    # units_per_function = length of the cut function in this unit (decode work a sampler that stops needs)
    res.update({"decode_wall_s": round(wall, 1), "units_per_function": round(sum(lens) / len(lens), 1),
                "example": srcs[0][0]})
    return res


def corpus(unit):
    text = open(f"{synth2.D}/train.txt").read()
    return torch.tensor(unit.encode(PREFIX + text), dtype=torch.long)


def make_unit(name, split):
    texts = [PREFIX + open(f"{synth2.D}/train.txt").read()]   # vocabulary from training text only
    return Bytes(texts) if name == "bytes" else Qwen(texts)


def train(a, unit, split):
    shape = SHAPE[a.unit]
    data = corpus(unit).cuda()
    torch.manual_seed(a.seed)
    net = UnitGPT(unit.vocab, 256, 4, 4, shape["ctx"]).cuda()
    n_nonemb = sum(p.numel() for nm, p in net.named_parameters() if not nm.startswith(("tok.", "head.", "pos.")))
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05)
    warm = 300

    def lr_at(s):                                            # warmup then cosine decay to 10% (as train.py)
        if s < warm:
            return 3e-4 * (s + 1) / warm
        return 3e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    marks = sorted({max(1, round(m * a.steps)) for m in MILESTONES})
    os.makedirs(a.out_dir, exist_ok=True)
    info = {"unit": a.unit, "vocab": unit.vocab, "corpus_units": len(data), "nonemb_params": n_nonemb,
            "params": sum(p.numel() for p in net.parameters()), **shape, "steps": a.steps,
            "units_per_step": shape["ctx"] * shape["batch"], "seed": a.seed}
    print(json.dumps(info), flush=True)
    log = []
    train_s = 0.0
    for s in range(1, a.steps + 1):
        t0 = time.perf_counter()
        net.train()
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(s - 1)
        st = torch.randint(0, len(data) - shape["ctx"] - 1, (shape["batch"],), device="cuda")
        w = data[st[:, None] + torch.arange(shape["ctx"] + 1, device="cuda")[None]]
        loss = F.cross_entropy(net(w[:, :-1]).reshape(-1, unit.vocab), w[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        train_s += time.perf_counter() - t0
        if s in marks:
            row = {"step": s, "units_seen": s * info["units_per_step"],
                   "flops": 6 * n_nonemb * s * info["units_per_step"],
                   # + attention scores/values: 3 (fwd+bwd) x 2 x layers x ctx x dim per unit (bytes ctx 512 > 192)
                   "flops_with_attn": (6 * n_nonemb + 6 * 4 * shape["ctx"] * 256) * s * info["units_per_step"],
                   "loss": round(loss.item(), 4),
                   "train_wall_s": round(train_s, 1)}
            for sp in SPLITS:
                row[sp] = evaluate(net, unit, split[sp], shape, a.n, a.k)
                row[sp]["greedy_pass@1"] = evaluate(net, unit, split[sp], shape, a.n, 1, temp=0)["pass@1"]
            log.append(row)
            print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "example"})
                              for k, v in row.items()}), flush=True)
            tag = f"{a.unit}_s{a.seed}"
            json.dump({"info": info, "log": log}, open(f"{a.out_dir}/units_{tag}.json", "w"), indent=1)
            torch.save({"model": net.state_dict(), "info": info, "step": s}, f"{a.out_dir}/ckpt_{tag}.pt")


def smoke(a, unit, split):
    """round trip; oracle sources through the same decode/cut/grade path = 1.0; prompts never leak the body."""
    text = open(f"{synth2.D}/train.txt").read()
    rt = unit.decode(unit.encode(text)) == text
    tasks = [t for sp in SPLITS for t in random.Random(2).sample(split[sp], 30)]
    srcs, leak = [], 0
    for u, f in tasks:
        p = prompt_ids(unit, u, f)
        full = unit.encode(PREFIX + source(u, f))
        leak += unit.decode(p) not in (f"{PREFIX}def {name_of(u, f)}(", f"{PREFIX}def {name_of(u, f)}")
        srcs.append([cut_function(unit.decode(p + full[len(p):])[len(PREFIX):])[0]])
    ok = sum(s[0] == 1 for s in grade(tasks, srcs))
    stop, bad = stop_tables(unit), 0
    for u, f in tasks:
        p = prompt_ids(unit, u, f)
        rest = unit.encode(PREFIX + source(u, f) + "\n" + source(*tasks[0]))[len(p):]   # next function follows
        prev = done = torch.zeros(1, dtype=torch.bool)
        for m, x in enumerate(rest):
            done, prev = stop_step(stop, torch.tensor([x]), prev, done)
            if done[0]:
                break
        cut = lambda ids: cut_function(unit.decode(p + ids)[len(PREFIX):])  # noqa: E731
        bad += not (done[0] and cut(rest[:m + 1]) == (source(u, f), True) and not cut(rest[:m])[1])
    data = corpus(unit)
    print(json.dumps({"unit": a.unit, "vocab": unit.vocab, "round_trip": rt, "oracle_1.0": f"{ok}/{len(tasks)}",
                      "prompts_not_name_only": leak, "stop_rule_mismatch": bad, "corpus_units": len(data),
                      "bytes_per_unit": round(len(text.encode()) / len(data), 3)}), flush=True)
    a.steps, a.n, a.k = 30, 8, 2
    a.out_dir = os.path.join(a.out_dir or "/media/llm-data/universe-runs/bytebrain-pycode-20261022/synth2", "smoke")
    train(a, unit, split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "train"])
    ap.add_argument("--unit", required=True, choices=list(SHAPE))
    ap.add_argument("--out-dir")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.5)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    split = json.load(open(f"{synth2.D}/splits.json"))
    unit = make_unit(a.unit, split)
    (smoke if a.cmd == "smoke" else train)(a, unit, split)


if __name__ == "__main__":
    main()
