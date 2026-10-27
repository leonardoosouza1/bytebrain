"""Water lens (ledger #9): atoms eat the residual of a Python function seen as a 2D byte image.

Written 2026-10-27 09:47 (from `date`), BEFORE the run.

Idea (iara-agua): a population of atoms is born where the residual is largest, eats the part it covers,
and the sum of the living atoms IS the reconstruction. Here: greedy matching pursuit (birth + eat) over a
fixed separable dictionary on the 2D map row = line, col = column. No geometry refinement/mutation, no
death rule (binary map has no noise sigma) - this is the water's birth/eat loop only.

Maps:
  layout  = ink map (non-space byte = 1), 1 channel.
  content = one-hot byte class, 5 channels (letter/_ , digit, quote, bracket, other punct). Energy of both = ink count.
Dictionary (67 atoms): boxes h in {1,2,4,8,16} x w in {1,...,64}; Gabor along columns (lambda 4/8/16, 2 phases,
  1 or 4 rows); Gabor along rows (lambda 2/4/8, 2 phases, 4 or 16 cols); diagonals slope 1 and 4 cols/row,
  4 or 8 rows, both directions.
Controls: shuffled lines (row order permuted inside the function), random bytes (the function's own bytes
  permuted over the same line-length envelope).

PREDICTIONS (written before running):
  P1a layout clean: median atoms to 90% energy ~30 (n90/ink ~0.12); to 99% ~70.
  P1b content clean needs >= 2x the layout atoms at 90%.
  P1c shuffled lines: layout n90 within 1.15x of clean (lines are horizontal; vertical structure is weak).
  P1d random bytes: layout n90 >= 3x clean.
  P2a the first 10 atoms on clean layout: >= 60% are 1-row boxes of width >= 16 (a line or half line).
  P2b tall boxes (h >= 4) aligned to an indentation block are < 15% of the first 10 atoms.
  P3a Spearman(n90 layout, ink) >= 0.9; Spearman(n90, rows) >= 0.8.
  P3b after removing ink (residual of log n90 on log ink), |Spearman| with nesting depth < 0.2.
  P3c one-line indentation corruption (+-4 cols): mean paired |delta n90| < 1 atom; AUC(n90/ink separates
      unparsable from parsable) in [0.4, 0.6]  -> the water does not see syntax.

Run (CPU only):
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n 5 timeout 3600 /home/leonardo/projects/LLM/.venv-rocm/bin/python water_lens.py
"""
import argparse
import ast
import json
import math
import os
import random
import re
import time

import numpy as np
from scipy import signal, stats

CORPUS = "/media/llm-data/datasets/pycode/clean_train.txt"
OUT = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/water_lens"
HMAX, WMAX = 16, 64


# ---------------------------------------------------------------- data
def functions(path, n, seed, max_rows, max_cols):
    text = open(path, encoding="utf-8", errors="replace").read()
    fns = re.split(r"\n\n(?=@|def |async def )", text)
    random.Random(seed).shuffle(fns)
    kept = []
    for f in fns:
        rows = [line.encode("utf-8") for line in f.rstrip("\n").split("\n")]
        if len(rows) < 3 or len(rows) > max_rows or max(len(r) for r in rows) > max_cols or b"\t" in b"".join(rows):
            continue
        kept.append(rows)
        if len(kept) == n:
            break
    return kept


def byte_class(b):
    if b == 32:
        return 0
    ch = chr(b)
    if ch.isalpha() or ch == "_" or b >= 128:
        return 1
    if ch.isdigit():
        return 2
    if ch in "'\"":
        return 3
    if ch in "()[]{}":
        return 4
    return 5


CLS = np.array([byte_class(b) for b in range(256)], dtype=np.int64)


def grid(rows):
    h, w = len(rows), max(1, max(len(r) for r in rows))
    g = np.full((h, w), 32, dtype=np.uint8)
    for i, r in enumerate(rows):
        g[i, :len(r)] = np.frombuffer(r, dtype=np.uint8)
    return g


def maps(g):
    cls = CLS[g]
    layout = (cls > 0).astype(np.float64)[None]
    content = np.stack([(cls == c).astype(np.float64) for c in range(1, 6)])
    return layout, content


def shuffled_lines(rows, rng):
    r = list(rows)
    rng.shuffle(r)
    return r


def random_bytes(rows, rng):
    allb = bytearray(b"".join(rows))
    rng.shuffle(allb)
    out, i = [], 0
    for r in rows:
        out.append(bytes(allb[i:i + len(r)]))
        i += len(r)
    return out


def indent(r):
    return len(r) - len(r.lstrip(b" "))


# ---------------------------------------------------------------- dictionary
def dictionary():
    atoms = []  # (name, family, kernel)
    for h in (1, 2, 4, 8, 16):
        for w in (1, 2, 4, 8, 16, 32, 64):
            atoms.append((f"box{h}x{w}", "box", np.ones((h, w))))
    for lam in (4, 8, 16):
        x = np.arange(2 * lam) - (2 * lam - 1) / 2
        env = np.exp(-x ** 2 / (2 * (lam / 2) ** 2))
        for ph in (0.0, math.pi / 2):
            kx = env * np.cos(2 * math.pi * x / lam + ph)
            for hr in (1, 4):
                atoms.append((f"gabH_l{lam}_p{int(ph > 0)}_h{hr}", "gaborH", np.ones((hr, 1)) * kx[None]))
    for lam in (2, 4, 8):
        y = np.arange(2 * lam) - (2 * lam - 1) / 2
        env = np.exp(-y ** 2 / (2 * (lam / 2) ** 2))
        for ph in (0.0, math.pi / 2):
            ky = env * np.cos(2 * math.pi * y / lam + ph)
            for wc in (4, 16):
                atoms.append((f"gabV_l{lam}_p{int(ph > 0)}_w{wc}", "gaborV", ky[:, None] * np.ones((1, wc))))
    for slope in (1, 4):
        for L in (4, 8):
            for d in (1, -1):
                k = np.zeros((L, slope * L + slope))
                for i in range(L):
                    c = i * slope if d > 0 else (L - 1 - i) * slope
                    k[i, c:c + slope] = 1
                atoms.append((f"diag_s{slope}_L{L}_{'dn' if d > 0 else 'up'}", "diag", k))
    out = []
    for name, fam, k in atoms:
        assert k.shape[0] <= HMAX and k.shape[1] <= WMAX
        out.append((name, fam, k / np.linalg.norm(k)))
    return out


def cross_table(atoms):
    """S[k, a, dr+HMAX-1, dc+WMAX-1] = <g_a at q, g_k at q+d>."""
    K = len(atoms)
    S = np.zeros((K, K, 2 * HMAX - 1, 2 * WMAX - 1), dtype=np.float32)
    pad = []
    for _, _, k in atoms:
        p = np.zeros((HMAX, WMAX))
        p[:k.shape[0], :k.shape[1]] = k
        pad.append(p)
    for a in range(K):
        for k in range(K):
            S[k, a] = signal.correlate(pad[a], pad[k], mode="full")
    return S


# ---------------------------------------------------------------- matching pursuit (birth + eat)
def pursue(X, atoms, S, target=0.99, cap=2000):
    """X: (C, H, W). Returns list of (atom, r, c) and explained-energy curve."""
    C, H, W = X.shape
    K = len(atoms)
    E0 = float((X ** 2).sum())
    corr = np.zeros((C, K, H, W), dtype=np.float64)
    valid = np.zeros((K, H, W), dtype=bool)
    for k, (_, _, g) in enumerate(atoms):
        kh, kw = g.shape
        if kh > H or kw > W:
            continue
        valid[k, :H - kh + 1, :W - kw + 1] = True
        for c in range(C):
            corr[c, k, :H - kh + 1, :W - kw + 1] = signal.correlate(X[c], g, mode="valid")
    neg = np.where(valid, 0.0, -np.inf)
    chosen, curve = [], []
    E = E0
    while E / E0 > 1 - target and len(chosen) < cap:
        score = (corr ** 2).sum(0) + neg
        idx = int(np.argmax(score))
        a, r, cc = np.unravel_index(idx, score.shape)
        alpha = corr[:, a, r, cc].copy()
        E -= float((alpha ** 2).sum())
        chosen.append((int(a), int(r), int(cc)))
        curve.append(1 - E / E0)
        r0, r1 = max(0, r - HMAX + 1), min(H, r + HMAX)
        c0, c1 = max(0, cc - WMAX + 1), min(W, cc + WMAX)
        blk = S[:, a, r0 - r + HMAX - 1:r1 - r + HMAX - 1, c0 - cc + WMAX - 1:c1 - cc + WMAX - 1]
        for c in range(C):
            corr[c, :, r0:r1, c0:c1] -= alpha[c] * blk
    return chosen, np.array(curve), E0


def n_at(curve, frac):
    hit = np.nonzero(curve >= frac)[0]
    return int(hit[0]) + 1 if len(hit) else len(curve) + 1


# ---------------------------------------------------------------- analysis helpers
def first_atoms_report(rows, chosen, atoms, n=10):
    ind = [indent(r) if r.strip() else -1 for r in rows]
    lens = [len(r.rstrip()) for r in rows]
    rep = []
    for a, r, c in chosen[:n]:
        name, fam, g = atoms[a]
        kh, kw = g.shape
        covered = list(range(r, min(len(rows), r + kh)))
        rep.append({
            "family": fam, "h": kh, "w": kw,
            "wide_line": fam == "box" and kh == 1 and kw >= 16,
            "indent_aligned": fam == "box" and ind[r] == c,
            "block_aligned": fam == "box" and kh >= 4 and all(ind[i] == c for i in covered),
            "header": r == 0,
            "line_end_match": fam == "box" and kh == 1 and abs(c + kw - lens[r]) <= 4,
        })
    return rep


def corrupt_indent(rows, rng):
    cand = [i for i in range(1, len(rows)) if rows[i].strip()]
    if not cand:
        return None
    i = rng.choice(cand)
    r = list(rows)
    if indent(r[i]) >= 4 and rng.random() < 0.5:
        r[i] = r[i][4:]
    else:
        r[i] = b"    " + r[i]
    return r


def parses(rows):
    try:
        ast.parse(b"\n".join(rows).decode("utf-8", errors="replace"))
        return True
    except SyntaxError:
        return False


def spearman(x, y):
    return float(stats.spearmanr(x, y).statistic)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=150, help="functions for Q1/Q2 (3 conditions x 2 maps, to 99%)")
    ap.add_argument("--n3", type=int, default=400, help="functions for Q3 (layout clean+corrupt, to 90%)")
    ap.add_argument("--max_rows", type=int, default=40)
    ap.add_argument("--max_cols", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    atoms = dictionary()
    S = cross_table(atoms)
    print(f"dictionary {len(atoms)} atoms, cross table {S.nbytes / 1e6:.0f} MB, {time.time() - t0:.1f}s", flush=True)
    fns = functions(CORPUS, args.n1 + args.n3, args.seed, args.max_rows, args.max_cols)
    rng = random.Random(args.seed + 1)

    # ---- Q1 + Q2
    q1 = {cond: {m: [] for m in ("layout", "content")} for cond in ("clean", "shuffled", "random")}
    curves = {cond: {m: [] for m in ("layout", "content")} for cond in q1}
    first = {cond: [] for cond in q1}
    fam_order = {cond: [] for cond in q1}
    for fi, rows in enumerate(fns[:args.n1]):
        variants = {"clean": rows, "shuffled": shuffled_lines(rows, rng), "random": random_bytes(rows, rng)}
        for cond, rr in variants.items():
            lay, con = maps(grid(rr))
            for m, X in (("layout", lay), ("content", con)):
                ch, cv, E0 = pursue(X, atoms, S, target=0.99)
                q1[cond][m].append({"ink": E0, "rows": len(rr), "n90": n_at(cv, 0.9), "n99": n_at(cv, 0.99),
                                    "n50": n_at(cv, 0.5)})
                curves[cond][m].append(cv[:400])
                if m == "layout":
                    first[cond].extend(first_atoms_report(rr, ch, atoms))
                    fam_order[cond].append([atoms[a][0] for a, _, _ in ch[:10]])
        if fi % 10 == 0:
            print(f"Q1 {fi}/{args.n1} {time.time() - t0:.0f}s", flush=True)

    res = {"n1": args.n1, "q1": {}, "q2": {}}
    for cond in q1:
        for m in ("layout", "content"):
            L = q1[cond][m]
            ink = np.array([d["ink"] for d in L])
            res["q1"][f"{cond}/{m}"] = {
                k: float(np.median([d[k] for d in L])) for k in ("n50", "n90", "n99")} | {
                "median_ink": float(np.median(ink)),
                "median_n90_per_ink": float(np.median([d["n90"] / d["ink"] for d in L])),
                "median_n99_per_ink": float(np.median([d["n99"] / d["ink"] for d in L])),
                "median_n90_per_row": float(np.median([d["n90"] / d["rows"] for d in L]))}
        rep = first[cond]
        fams = {}
        for d in rep:
            fams[d["family"]] = fams.get(d["family"], 0) + 1
        shapes = {}
        for d in rep:
            key = f"{d['family']}{d['h']}x{d['w']}"
            shapes[key] = shapes.get(key, 0) + 1
        res["q2"][cond] = {k: float(np.mean([d[k] for d in rep])) for k in
                           ("wide_line", "indent_aligned", "block_aligned", "header", "line_end_match")} | {
            "family_share": {k: v / len(rep) for k, v in sorted(fams.items(), key=lambda kv: -kv[1])},
            "top_shapes": dict(sorted(shapes.items(), key=lambda kv: -kv[1])[:8]),
            "mean_h": float(np.mean([d["h"] for d in rep])), "mean_w": float(np.mean([d["w"] for d in rep]))}
    print(json.dumps(res, indent=1), flush=True)

    # ---- Q3
    rec = []
    for fi, rows in enumerate(fns[args.n1:args.n1 + args.n3]):
        lay, _ = maps(grid(rows))
        _, cv, E0 = pursue(lay, atoms, S, target=0.9)
        nonempty = [r for r in rows if r.strip()]
        d = {"ink": E0, "rows": len(rows), "depth": max(indent(r) for r in nonempty) // 4,
             "levels": len({indent(r) for r in nonempty}), "mean_len": float(np.mean([len(r) for r in rows])),
             "n90": n_at(cv, 0.9), "parse": parses(rows)}
        bad = corrupt_indent(rows, rng)
        if bad is not None:
            lb, _ = maps(grid(bad))
            _, cvb, Eb = pursue(lb, atoms, S, target=0.9)
            d.update({"bad_n90": n_at(cvb, 0.9), "bad_ink": Eb, "bad_parse": parses(bad)})
        rec.append(d)
        if fi % 50 == 0:
            print(f"Q3 {fi}/{args.n3} {time.time() - t0:.0f}s", flush=True)
    n90 = np.array([d["n90"] for d in rec], float)
    ink = np.array([d["ink"] for d in rec], float)
    resid = np.log(n90) - np.polyval(np.polyfit(np.log(ink), np.log(n90), 1), np.log(ink))
    q3 = {"n": len(rec), "clean_parse_rate": float(np.mean([d["parse"] for d in rec]))}
    for k in ("ink", "rows", "depth", "levels", "mean_len"):
        v = np.array([d[k] for d in rec], float)
        q3[f"spearman_n90_{k}"] = spearman(n90, v)
        q3[f"spearman_n90_resid_ink_{k}"] = spearman(resid, v)
    pairs = [d for d in rec if "bad_n90" in d]
    delta = np.array([d["bad_n90"] - d["n90"] for d in pairs], float)
    q3["corrupt_n"] = len(pairs)
    q3["corrupt_parse_fail_rate"] = float(np.mean([not d["bad_parse"] for d in pairs]))
    q3["corrupt_mean_delta_n90"] = float(delta.mean())
    q3["corrupt_mean_abs_delta_n90"] = float(np.abs(delta).mean())
    q3["corrupt_frac_more_atoms"] = float((delta > 0).mean())
    q3["corrupt_frac_fewer_atoms"] = float((delta < 0).mean())
    ok = [d["n90"] / d["ink"] for d in rec if d["parse"]] + [d["bad_n90"] / d["bad_ink"] for d in pairs if d["bad_parse"]]
    fail = [d["bad_n90"] / d["bad_ink"] for d in pairs if not d["bad_parse"]]
    q3["auc_n90_per_ink_fail_vs_ok"] = auc(fail, ok)
    q3["auc_paired_fail_only_delta_gt0"] = float(np.mean([d["bad_n90"] > d["n90"] for d in pairs if not d["bad_parse"]]))
    res["q3"] = q3
    print(json.dumps(q3, indent=1), flush=True)
    res["wall_s"] = time.time() - t0
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    json.dump(fam_order, open(f"{OUT}/first10_atoms.json", "w"))
    json.dump(rec, open(f"{OUT}/q3_records.json", "w"))
    figure(fns[0], atoms, S, curves, res, rec, resid, f"{OUT}/water_lens.png")
    print(f"done {time.time() - t0:.0f}s", flush=True)


def figure(rows, atoms, S, curves, res, rec, resid, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lay, _ = maps(grid(rows))
    ch, cv, _ = pursue(lay, atoms, S, target=0.99)
    H, W = lay.shape[1:]

    def recon(n):
        R = np.zeros((H, W))
        X = lay[0].copy()
        Rr = X.copy()
        for a, r, c in ch[:n]:
            g = atoms[a][2]
            G = np.zeros((H, W))
            G[r:r + g.shape[0], c:c + g.shape[1]] = g
            al = (Rr * G).sum()
            R += al * G
            Rr -= al * G
        return R

    fig = plt.figure(figsize=(22, 11), facecolor="#111")
    plt.rcParams.update({"text.color": "#ddd", "axes.labelcolor": "#ddd", "xtick.color": "#aaa", "ytick.color": "#aaa"})
    ns = [5, 20, n_at(cv, 0.9), n_at(cv, 0.99)]
    ax = fig.add_subplot(2, 5, 1)
    ax.imshow(lay[0], cmap="Greys_r", aspect="auto")
    ax.set_title("ink map (one function)", fontsize=10)
    for i, n in enumerate(ns):
        ax = fig.add_subplot(2, 5, 2 + i)
        ax.imshow(recon(n), cmap="magma", aspect="auto")
        ax.set_title(f"sum of first {n} atoms ({cv[min(n, len(cv)) - 1] * 100:.0f}% energy)", fontsize=10)
    ax = fig.add_subplot(2, 5, (6, 7))
    ax.set_facecolor("#111")
    colors = {"clean": "#4fc3f7", "shuffled": "#ffb74d", "random": "#e57373"}
    for cond in curves:
        for m, ls in (("layout", "-"), ("content", "--")):
            cs = curves[cond][m]
            L = max(len(c) for c in cs)
            M = np.array([np.pad(c, (0, L - len(c)), constant_values=c[-1]) for c in cs])
            ax.plot(np.arange(1, L + 1), np.median(M, 0), ls, color=colors[cond], label=f"{cond} {m}")
    ax.axhline(0.9, color="#666", lw=0.5)
    ax.axhline(0.99, color="#666", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("atoms alive (birth order)")
    ax.set_ylabel("energy eaten (median over functions)")
    ax.set_title("how fast the atoms eat the function: solid = layout, dashed = byte class", fontsize=10)
    ax.legend(fontsize=8, facecolor="#222")
    ax = fig.add_subplot(2, 5, 8)
    ax.set_facecolor("#111")
    keys = ["wide_line", "indent_aligned", "block_aligned", "header", "line_end_match"]
    x = np.arange(len(keys))
    for i, cond in enumerate(res["q2"]):
        ax.bar(x + (i - 1) * 0.27, [res["q2"][cond][k] for k in keys], 0.27, color=colors[cond], label=cond)
    ax.set_xticks(x)
    ax.set_xticklabels(["1-row box\nw>=16", "left edge\n= indent", "tall box =\nindent block", "on def\nheader", "right edge\n= line end"], fontsize=7)
    ax.set_title("first 10 atoms: what they sit on", fontsize=10)
    ax.legend(fontsize=7, facecolor="#222")
    ax = fig.add_subplot(2, 5, 9)
    ax.set_facecolor("#111")
    inkv = np.array([d["ink"] for d in rec])
    o = np.argsort(inkv)
    ax.scatter(inkv[o], np.array([d["n90"] for d in rec])[o], s=6, c=np.array([d["depth"] for d in rec])[o], cmap="viridis")
    ax.set_xlabel("ink pixels")
    ax.set_ylabel("atoms to 90%")
    ax.set_title("atom count ~ ink (color = nesting depth)", fontsize=10)
    ax = fig.add_subplot(2, 5, 10)
    ax.set_facecolor("#111")
    dep = np.array([d["depth"] for d in rec])
    o = np.argsort(dep)
    ax.scatter(dep[o] + np.random.default_rng(0).uniform(-.2, .2, len(o)), resid[o], s=6, color="#4fc3f7")
    ax.set_xlabel("nesting depth")
    ax.set_ylabel("log atoms beyond what ink predicts")
    ax.set_title("does depth cost extra atoms?", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="#111")


if __name__ == "__main__":
    main()
