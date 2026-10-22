"""Code as a byte matrix (rows = lines, columns = byte positions): which measured quantities announce structure.

  1. indent birth/death: mutual information between features of line i (last char, first char x0, first word,
     length, previous indent change) and the indent change of line i+1 (-k..+k levels).
  2. x0 x row: I(x0; byte at column j of the same line, counted from the first non-space char) as j grows.
  3. two matrices (.dna idea): one-hot grid [lines, cols*C] factorised by truncated SVD (U S V^T); exact byte
     reconstruction accuracy vs rank, real grid vs the same grid with bytes shuffled inside each line.
All counting and the SVD run on the GPU; input is held-out real clean functions.

Usage: grid_laws.py [--n 4000] [--out FILE.json]
"""

import argparse
import json
import math
import random

import torch

D = "/media/llm-data/datasets/pycode"
COLS = 64


def mi(x, y):
    """I(X;Y) in bits from two int tensors of equal length (GPU joint histogram)."""
    _, xi = torch.unique(x, return_inverse=True)
    _, yi = torch.unique(y, return_inverse=True)
    nx, ny = int(xi.max()) + 1, int(yi.max()) + 1
    j = torch.zeros(nx * ny, device=x.device).index_add_(0, xi * ny + yi, torch.ones_like(xi, dtype=torch.float))
    p = (j / j.sum()).view(nx, ny)
    pxy = p.sum(1, keepdim=True) @ p.sum(0, keepdim=True)
    nz = p > 0
    return float((p[nz] * torch.log2(p[nz] / pxy[nz])).sum())


def entropy(y):
    _, c = torch.unique(y, return_counts=True)
    p = c.float() / c.sum()
    return float(-(p * torch.log2(p)).sum())


def lines_of(funcs):
    rows = []
    for f in funcs:
        ls = [l for l in f.split("\n") if l.strip()]
        ind = [len(l) - len(l.lstrip(" ")) for l in ls]
        for i in range(len(ls) - 1):
            body = ls[i].strip()
            rows.append((ls[i], ind[i], ind[i + 1] - ind[i], ind[i] - ind[i - 1] if i else 4, body))
    return rows


def indent_law(rows, dev):
    def ids(vals):
        table = {}
        return torch.tensor([table.setdefault(v, len(table)) for v in vals], device=dev)
    dlt = torch.tensor([max(-12, min(12, r[2])) // 4 if r[2] % 4 == 0 else 99 for r in rows], device=dev)
    feats = {
        "last char": ids([r[4][-1] for r in rows]),
        "x0 (first char)": ids([r[4][0] for r in rows]),
        "first word": ids([r[4].split()[0].split("(")[0] for r in rows]),
        "length bucket": ids([min(len(r[4]) // 10, 9) for r in rows]),
        "previous change": ids([max(-12, min(12, r[3])) for r in rows]),
        "indent level": ids([min(r[1] // 4, 8) for r in rows]),
    }
    h = entropy(dlt)
    out = {"H(next indent change)": round(h, 3)}
    for k, v in feats.items():
        out[k] = round(mi(v, dlt), 3)
    out["last char + first word"] = round(mi(feats["last char"] * 100000 + feats["first word"], dlt), 3)
    out["first word + indent level"] = round(mi(feats["first word"] * 100 + feats["indent level"], dlt), 3)
    colon = torch.tensor([r[4].endswith(":") for r in rows], device=dev)
    up = torch.tensor([r[2] > 0 for r in rows], device=dev)
    down = torch.tensor([r[2] < 0 for r in rows], device=dev)
    out["P(indent up | ends ':')"] = round(float(up[colon].float().mean()), 4)
    out["P(ends ':' | indent up)"] = round(float(colon[up].float().mean()), 4)
    first = [r[4].split()[0].split("(")[0] for r in rows]
    for w in ("return", "raise", "continue", "break", "pass"):
        m = torch.tensor([x == w for x in first], device=dev)
        if bool(m.any()):
            out[f"P(indent down | line starts '{w}')"] = round(float(down[m].float().mean()), 3)
    out["P(indent down) overall"] = round(float(down.float().mean()), 3)
    return out


def x0_row(rows, dev):
    body = [r[4].encode()[:COLS] for r in rows]
    x0 = torch.tensor([b[0] for b in body], device=dev)
    out = {}
    for j in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
        m = [i for i, b in enumerate(body) if len(b) > j]
        idx = torch.tensor(m, device=dev)
        yj = torch.tensor([body[i][j] for i in m], device=dev)
        perm = idx[torch.randperm(len(m), device=dev)]
        out[j] = [round(mi(x0[idx], yj), 3), round(mi(x0[perm], yj), 3)]   # real, shuffled-pairs floor
    return out


def grid_onehot(rows, dev, shuffle=False, C=32):
    """[lines, COLS*C] one-hot over a 32-symbol alphabet (byte classes fine enough to read code: a-z merged)."""
    g = torch.full((len(rows), COLS), 32, dtype=torch.long)  # space-padded
    r = random.Random(0)
    for i, row in enumerate(rows):
        b = list(row[0].encode()[:COLS])
        if shuffle:
            r.shuffle(b)
        g[i, :len(b)] = torch.tensor(b)
    g = g.to(dev)
    sym = torch.full((256,), 31, dtype=torch.long, device=dev)
    for k, ch in enumerate(" ()[]{}:,.=+-*/'\"_#<>%!\n"):
        sym[ord(ch)] = k
    sym[97:123] = 24
    sym[65:91] = 25
    sym[48:58] = 26
    s = sym[g]
    return torch.nn.functional.one_hot(s, C).float().view(len(rows), -1), s


def svd_law(rows, dev, ranks=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    out = {}
    for name, shuf in (("real", False), ("bytes shuffled in line", True)):
        X, s = grid_onehot(rows, dev, shuf)
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        nonpad = s != 0                       # columns that are not padding/space
        acc = {}
        for k in ranks:
            R = (U[:, :k] * S[:k]) @ Vh[:k]
            pred = R.view(len(rows), COLS, -1).argmax(2)
            acc[k] = [round(float((pred == s).float().mean()), 3), round(float((pred == s)[nonpad].float().mean()), 3)]
        e = (S ** 2).cumsum(0) / (S ** 2).sum()
        out[name] = {"acc[all, non-space] by rank": acc,
                     "rank for 90% energy": int((e < 0.9).sum()) + 1}
        del U, S, Vh, X
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="/media/llm-data/universe-runs/bytebrain-pycode-20261022/grid_laws.json")
    ap.add_argument("--mem-frac", type=float, default=0.1)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(0)
    dev = "cuda"
    hold = json.load(open(f"{D}/holdout_functions.json"))
    rows = lines_of([h["clean"] for h in hold[:a.n]])
    res = {"lines": len(rows), "indent": indent_law(rows, dev), "x0_row": x0_row(rows, dev)}
    print(json.dumps(res, indent=1), flush=True)
    res["svd"] = svd_law(rows[:12000], dev)
    print(json.dumps(res["svd"], indent=1))
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
