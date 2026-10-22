"""Line-type transition matrix T[type_i, type_i+1] measured on the training corpus: the small matrix that carries
the block structure (grid_laws.py: first word + indent level hold 94% of the next indent change).

Line type = (first word, top-K else OTHER) x (last char class) x (indent change from previous line, -3..+1 levels).
Mirror use: bits per line of a text under T (add-0.5 smoothing) vs the same for real held-out code, and the
share of transitions never seen in the corpus. No byte is proposed; it only measures distance.

Usage: line_laws.py [--samples FILE.json ...]      (rows[...]['src'] as in eval_samples/mix_eval)
"""

import argparse
import json
import random
from collections import Counter

import torch

D = "/media/llm-data/datasets/pycode"
TOPK = 48
LAST = {":": 0, ")": 1, "]": 2, "}": 3, ",": 4, "(": 5}


def split_funcs(path):
    import re
    return re.split(r"\n\n(?=(?:async )?def )", open(path, "rb").read().decode())


def first_word(body):
    w = body.split()[0]
    for ch in "([.=:,'\"":
        w = w.split(ch)[0]
    return w or body[0]


def line_seq(src):
    """[(word, last_class, delta_levels)] for the non-blank lines of a function; None if indent is not a multiple of 4."""
    out, prev = [], 0
    for l in src.split("\n"):
        if not l.strip():
            continue
        ind = len(l) - len(l.lstrip(" "))
        body = l.strip()
        d = max(-3, min(1, (ind - prev) // 4)) if (ind - prev) % 4 == 0 else 2
        out.append((first_word(body), LAST.get(body[-1], 6 if body[-1].isalnum() or body[-1] == "_" else 7), d))
        prev = ind
    return out


class LineLaw:
    def __init__(self, funcs, dev="cuda"):
        words = Counter(w for f in funcs for w, _, _ in line_seq(f))
        self.vocab = {w: i for i, (w, _) in enumerate(words.most_common(TOPK))}
        self.n = (TOPK + 1) * 8 * 6 + 1          # +1: start-of-function state
        self.dev = dev
        ids = [self.ids(f) for f in funcs]
        a = torch.tensor([x for s in ids for x in s[:-1]], device=dev)
        b = torch.tensor([x for s in ids for x in s[1:]], device=dev)
        self.T = torch.zeros(self.n * self.n, device=dev).index_add_(0, a * self.n + b, torch.ones_like(a, dtype=torch.float)).view(self.n, self.n)
        self.P = (self.T + 0.5) / (self.T.sum(1, keepdim=True) + 0.5 * self.n)

    def ids(self, src):
        s = [self.n - 1]
        for w, c, d in line_seq(src):
            s.append((self.vocab.get(w, TOPK) * 8 + c) * 6 + (d + 3))
        return s

    def score(self, srcs):
        bits, lines, unseen = 0.0, 0, 0
        for src in srcs:
            s = torch.tensor(self.ids(src), device=self.dev)
            if len(s) < 2:
                continue
            bits += float(-torch.log2(self.P[s[:-1], s[1:]]).sum())
            unseen += int((self.T[s[:-1], s[1:]] == 0).sum())
            lines += len(s) - 1
        return {"bits_per_line": round(bits / max(1, lines), 3), "unseen_transitions": round(unseen / max(1, lines), 4),
                "lines": lines}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--mem-frac", type=float, default=0.05)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    law = LineLaw([f for f in split_funcs(f"{D}/clean_train.txt") if f.strip()])
    print(f"types {law.n}, transitions seen {int((law.T > 0).sum())}, total {int(law.T.sum())}")
    hold = json.load(open(f"{D}/holdout_functions.json"))
    random.Random(2).shuffle(hold)
    real = [h["clean"] for h in hold[:1000]]
    print("real held-out:", law.score(real))
    r = random.Random(0)
    shuf = []
    for t in real:                      # control: same lines, order shuffled inside the function (keeps indents)
        ls = t.split("\n")
        body = ls[1:]
        r.shuffle(body)
        shuf.append("\n".join(ls[:1] + body))
    print("lines shuffled:", law.score(shuf))
    for f in a.samples:
        rows = json.load(open(f))["rows"]
        srcs = [r["src"] for part in rows.values() for r in part]
        print(f, law.score(srcs))


if __name__ == "__main__":
    main()
