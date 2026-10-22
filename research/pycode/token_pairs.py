"""Layer 2 of the GPU checker: adjacent-token legality measured on the corpus, not written by hand.

A byte-level tokenizer runs on top of SyntaxAutomaton (same loop, same tensors): NAME / NUMBER / STRING / each
keyword / each operator character / NEWLINE (end of logical line). Counting mode runs it over the training
corpus and fills pairs[prev_token, token]; checking mode flags a sequence the first time it produces a pair the
corpus never showed (count < min_count). The law is the corpus' own: which token can follow which.

Usage: token_pairs.py --build    (count on clean_train, save table)   then
       token_pairs.py --check    (real held-out false positives; corrupted-real recall vs ast.parse, with layer 1)
"""

import argparse
import ast
import json
import keyword
import random
import re
import time

import numpy as np
import torch

from gpu_syntax import CODE, NL, SyntaxAutomaton

D = "/media/llm-data/datasets/pycode"
TABLE = f"{D}/token_pairs.npy"
KW = keyword.kwlist + [k for k in keyword.softkwlist if k != "_"]   # match/case/type as own classes, so NAME NAME stays rare
OPS = "()[]{}:,.;=+-*/%<>!&|^~@"
NEWLINE, NAME, NUMBER, STRING = 0, 1, 2, 3
KW0 = 4
OP0 = KW0 + len(KW)
N_CLS = OP0 + len(OPS)
P = 2147483647


def khash(word):
    h = 0
    for c in word.encode():
        h = (h * 131 + c) % P
    return h


def cls_name(i):
    return (["NEWLINE", "NAME", "NUMBER", "STRING"] + KW + list(OPS))[i]


class TokenPairs:
    def __init__(self, B, device="cuda", table=None, min_count=1):
        self.syn = SyntaxAutomaton(B, device)
        self.B, self.dev = B, device
        z = lambda: torch.zeros(B, dtype=torch.long, device=device)  # noqa: E731
        self.kind, self.h, self.idlen = z(), z(), z()     # kind: 0 none, NAME, NUMBER
        self.prev = z()                                   # previous token class (NEWLINE at start)
        self.err = torch.zeros(B, dtype=torch.bool, device=device)
        self.err_pos = torch.full((B,), -1, dtype=torch.long, device=device)
        self.counts = torch.zeros(N_CLS * N_CLS, dtype=torch.long, device=device) if table is None else None
        self.legal = None if table is None else (torch.as_tensor(table, device=device) >= min_count)
        self.kw_hash = torch.tensor([khash(k) for k in KW], device=device)
        lut = torch.full((256,), -1, dtype=torch.long)
        for i, ch in enumerate(OPS):
            lut[ord(ch)] = OP0 + i
        self.op_cls = lut.to(device)
        isid = torch.zeros(256, dtype=torch.bool)
        for ch in range(256):
            isid[ch] = chr(ch).isalnum() and ch < 128 or ch == 95
        self.isid = isid.to(device)

    def _emit(self, mask, cls):
        if not bool(mask.any()):
            return
        if self.counts is not None:
            self.counts.index_put_((self.prev[mask] * N_CLS + cls[mask],), torch.ones_like(cls[mask]), accumulate=True)
        else:
            bad = mask & ~self.legal[self.prev, cls] & ~self.err
            self.err_pos = torch.where(bad, torch.full_like(self.err_pos, self.syn.t), self.err_pos)
            self.err |= bad
        self.prev = torch.where(mask, cls, self.prev)

    def _close_ident(self, mask):
        kw = (self.h[:, None] == self.kw_hash[None]) & (self.kind == NAME)[:, None]
        cls = torch.where(kw.any(1), KW0 + kw.long().argmax(1), self.kind)
        self._emit(mask, cls)
        self.kind = torch.where(mask, torch.zeros_like(self.kind), self.kind)

    @torch.no_grad()
    def step(self, c):
        c = c.long()
        syn = self.syn
        s0 = syn.state
        code = syn.code[s0, c] & ~syn.err
        syn.step(c)
        s1 = syn.state
        # a string token ends when the lexer leaves a string state
        str_end = (s0 >= 3) & (s1 < 3) & ~syn.err
        self._emit(str_end, torch.full_like(c, STRING))
        isid = self.isid[c]
        in_id = self.kind > 0
        cont = code & in_id & (isid | ((self.kind == NUMBER) & (c == 46)))
        self.h = torch.where(cont, (self.h * 131 + c) % P, self.h)
        self.idlen = self.idlen + cont.long()
        quote = (c == 39) | (c == 34)
        prefix = code & quote & (self.kind == NAME) & (self.idlen <= 2)    # f'..', rb"..": part of the string
        close = code & in_id & ~cont & ~prefix
        self._close_ident(close)
        self.kind = torch.where(prefix, torch.zeros_like(self.kind), self.kind)
        start = code & ~cont & isid
        digit = (c >= 48) & (c <= 57)
        self.kind = torch.where(start, torch.where(digit, NUMBER, NAME), self.kind)
        self.h = torch.where(start, c, self.h)
        self.idlen = torch.where(start, torch.ones_like(self.idlen), self.idlen)
        op = code & (self.op_cls[c] >= 0) & ~((self.kind == NUMBER) & (c == 46))
        self._emit(op, self.op_cls[c].clamp(min=0))
        nl = code & (c == NL) & (syn.ptr == 0) & (self.prev != NEWLINE)
        self._emit(nl, torch.full_like(c, NEWLINE))

    @property
    def flagged(self):
        return self.syn.err | self.err | ~self.syn.complete()


def run(texts, tp_factory, device="cuda"):
    data = [t.encode() + b"\n" for t in texts]
    L = max(len(d) for d in data)
    buf = torch.full((len(data), L), NL, dtype=torch.long)
    for i, d in enumerate(data):
        buf[i, :len(d)] = torch.tensor(list(d))
    buf = buf.to(device)
    tp = tp_factory(len(data))
    for t in range(L):
        tp.step(buf[:, t])
    return tp


def build(n_chunks=4096):
    # split only at function starts: a blank line inside a string must not cut a function in two
    funcs = re.split(r"\n\n(?=(?:async )?def )", open(f"{D}/clean_train.txt", "rb").read().decode())
    funcs = [f for f in funcs if f.strip()]
    chunks = [[] for _ in range(n_chunks)]
    sizes = [0] * n_chunks
    for f in sorted(funcs, key=len, reverse=True):          # greedy balance so the batch has little padding
        i = int(np.argmin(sizes))
        chunks[i].append(f)
        sizes[i] += len(f) + 2
    texts = ["\n\n".join(c) for c in chunks]
    t0 = time.perf_counter()
    tp = run(texts, lambda B: TokenPairs(B))
    torch.cuda.synchronize()
    counts = tp.counts.view(N_CLS, N_CLS).cpu().numpy()
    np.save(TABLE, counts)
    nz = int((counts > 0).sum())
    print(f"counted {counts.sum()} token pairs over {len(funcs)} functions in {time.perf_counter() - t0:.0f}s; "
          f"{nz}/{N_CLS * N_CLS} pairs seen; layer-1 errors in training corpus: {int(tp.syn.err.sum())}/{n_chunks}")


def check(min_count):
    table = np.load(TABLE)
    hold = json.load(open(f"{D}/holdout_functions.json"))
    random.Random(3).shuffle(hold)
    real = [h["clean"] for h in hold[:1000]]
    tp = run(real, lambda B: TokenPairs(B, table=table, min_count=min_count))
    fp = tp.flagged.cpu()
    print(f"min_count {min_count}: real held-out flagged {int(fp.sum())}/{len(real)}")
    for i in [i for i in range(len(real)) if fp[i]][:3]:
        pos = int(tp.err_pos[i])
        print(f"   FP at {pos}: ...{real[i][max(0, pos - 40):pos + 1]!r}")
    r = random.Random(4)
    bad = []
    for t in real:
        j = r.randrange(len(t))
        bad.append(t[:j] + r.choice("()[]{}'\":") + t[j:] if r.random() < 0.5 else t[:j] + t[j + 1:])
    truth = []
    for t in bad:
        try:
            ast.parse(t)
            truth.append(True)
        except (SyntaxError, ValueError):
            truth.append(False)
    tp = run(bad, lambda B: TokenPairs(B, table=table, min_count=min_count))
    f1 = (tp.syn.err | ~tp.syn.complete()).cpu()
    f2 = tp.flagged.cpu()
    fails = [i for i, ok in enumerate(truth) if not ok]
    for name, f in (("layer 1", f1), ("layers 1+2", f2)):
        caught = sum(bool(f[i]) for i in fails)
        wrong = sum(bool(f[i]) for i, ok in enumerate(truth) if ok)
        print(f"   corrupted real, {name}: catches {caught}/{len(fails)} ({caught / len(fails):.1%}) ast failures, "
              f"flags {wrong} that parse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--min-count", type=int, nargs="*", default=[1, 2, 5])
    ap.add_argument("--mem-frac", type=float, default=0.12)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    if args.build:
        build()
    if args.check:
        for m in args.min_count:
            check(m)


if __name__ == "__main__":
    main()
