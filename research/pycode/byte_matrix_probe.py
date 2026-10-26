"""Ledger #6 + #9 (his 10-22 ideas): code as a byte matrix (rows = lines, columns = byte position in the line).

His words: "converte o código em matrizes de bytes... cada matriz é uma linha ou coluna, ou ambas, ou tratar como
matriz inteira de X linhas e Y colunas, se eu pegar linha 1, pega x0 e multiplicar com os outros, o que daria? o que
iria indicar que uma identação nasce? uma quebra de linha?" and #9: the forest / water image models as a lens.

CPU only, numpy, light. One measurement per question:
  A. What signals that an indent is born / dies between row r and the next non-blank row? Mutual information
     between one byte of row r and born/dies: x0 (first byte), first non-space byte, last non-space byte, first word.
     Rule "row ends with ':'" -> born, precision / recall.
  B. Line break: P(newline | byte), and how well the best one-byte rule finds line ends.
  C. Whole matrix as the .dna "two matrices" (truncated SVD, G ~ U_k S_k V_k): bytes recovered exactly vs k, and
     the storage it costs. Same on the ink map (non-space = 1).
  D. Forest lens: multi-scale pyramid of the ink map (block means -> upsample -> threshold): which scale keeps the
     indentation profile (first non-space column of every row).
Output: <out>/<corpus>.json + example PNGs. Predictions (PRED) are written before the run and printed with results.
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter

import numpy as np

PRED = {
    "A1_clean_colon_rule": "born precision >= 0.99 and recall >= 0.99 (ast.unparse writes one statement per line)",
    "A1_raw_colon_rule": "born recall < 0.95 (raw code has hanging indents after '(' / '[')",
    "A2_dies_local": "best single-byte MI with dies < 0.3 x H(dies): the end of a block is not written in one byte",
    "A3_x0": "MI(x0; born) < 0.05 bits: x0 only says 'row starts at column 0 or not'",
    "C1_bytes_tsvd": "rank 4 recovers < 30% of non-space bytes; 90% of all bytes needs k >= 0.5 min(R, C)",
    "C2_ink_tsvd": "ink map rank 2 pixel accuracy >= 0.85",
    "D1_pyramid": "block 1x4 keeps the indent profile on >= 95% of rows; block 4x8 on < 60%",
}
STOP_WORDS = (b"return", b"raise", b"break", b"continue", b"pass")


def functions(path, n, seed, max_rows, max_cols):
    text = open(path, encoding="utf-8", errors="replace").read()
    fns = re.split(r"\n\n(?=@|def |async def )", text)   # a decorator belongs to the next function
    random.Random(seed).shuffle(fns)
    kept, examined, too_big = [], 0, 0
    for f in fns:
        examined += 1
        rows = [line.encode("utf-8") for line in f.rstrip("\n").split("\n")]
        if len(rows) < 3 or len(rows) > max_rows or max(len(r) for r in rows) > max_cols or b"\t" in b"".join(rows):
            too_big += 1
            continue
        kept.append(rows)
        if len(kept) == n:
            break
    return kept, {"functions_in_file": len(fns), "examined": examined, "skipped_size_or_tab": too_big}


def indent(r):
    return len(r) - len(r.lstrip(b" "))


def mi_bits(xs, ys):
    """Mutual information (bits) between two discrete sequences, plug-in estimate."""
    n = len(xs)
    cxy, cx, cy = Counter(zip(xs, ys)), Counter(xs), Counter(ys)
    return sum(c / n * math.log2(c * n / (cx[x] * cy[y])) for (x, y), c in cxy.items())


def entropy_bits(ys):
    n = len(ys)
    return -sum(c / n * math.log2(c / n) for c in Counter(ys).values())


def pr(pred, gold):
    tp = sum(p and g for p, g in zip(pred, gold))
    return {"precision": round(tp / max(1, sum(pred)), 4), "recall": round(tp / max(1, sum(gold)), 4),
            "n_pred": int(sum(pred)), "n_gold": int(sum(gold))}


def part_a(fns):
    feats = {"x0": [], "first_nonspace": [], "last_nonspace": [], "first_word": []}
    born, dies = [], []
    for rows in fns:
        nb = [r for r in rows if r.strip()]
        for cur, nxt in zip(nb, nb[1:]):
            s = cur.strip()
            feats["x0"].append(cur[0])
            feats["first_nonspace"].append(s[0])
            feats["last_nonspace"].append(s[-1])
            feats["first_word"].append(re.match(rb"[A-Za-z_]*", s).group(0) or s[:1])
            born.append(indent(nxt) > indent(cur))
            dies.append(indent(nxt) < indent(cur))
    out = {"pairs": len(born), "H_born": round(entropy_bits(born), 4), "H_dies": round(entropy_bits(dies), 4)}
    out["mi_born"] = {k: round(mi_bits(v, born), 4) for k, v in feats.items()}
    out["mi_dies"] = {k: round(mi_bits(v, dies), 4) for k, v in feats.items()}
    out["colon_rule_born"] = pr([b == ord(":") for b in feats["last_nonspace"]], born)
    out["stopword_rule_dies"] = pr([w in STOP_WORDS for w in feats["first_word"]], dies)
    # what ends a row that is followed by a new indent level, besides ':'
    out["born_not_colon_last_byte"] = Counter(chr(b) for b, g in zip(feats["last_nonspace"], born)
                                              if g and b != ord(":")).most_common(8)
    return out


def part_b(fns):
    seen, ends = Counter(), Counter()
    for rows in fns:
        for r in rows:
            if not r.strip():
                continue
            body = r.rstrip()
            for i, b in enumerate(body):
                seen[b] += 1
                ends[b] += i == len(body) - 1
    p_end = {b: ends[b] / seen[b] for b in seen}
    top = sorted((b for b in seen if seen[b] >= 200), key=lambda b: -p_end[b])[:10]
    rule = {b for b in seen if p_end[b] > 0.5}
    tp = sum(ends[b] for b in rule)
    return {"p_newline_after_byte": [[chr(b), round(p_end[b], 4), seen[b]] for b in top],
            "one_byte_rule_bytes": sorted(chr(b) for b in rule),
            "one_byte_rule": {"precision": round(tp / max(1, sum(seen[b] for b in rule)), 4),
                              "recall": round(tp / max(1, sum(ends.values())), 4)}}


def grid(rows, cols):
    g = np.full((len(rows), cols), 32, dtype=np.float64)
    for i, r in enumerate(rows):
        g[i, :len(r)] = np.frombuffer(r, dtype=np.uint8)
    return g


def first_ink(ink):
    has = ink.any(axis=1)
    return np.where(has, ink.argmax(axis=1), -1)


def part_c(fns, cols, ks):
    acc = {k: [] for k in ks}
    acc_ns = {k: [] for k in ks}
    ink_acc = {k: [] for k in ks}
    ink_prof = {k: [] for k in ks}
    cost = {k: [] for k in ks}
    k90 = []
    for rows in fns:
        g = grid(rows, cols)
        ns = g != 32
        u, s, vt = np.linalg.svd(g, full_matrices=False)
        iu, is_, ivt = np.linalg.svd(ns.astype(np.float64), full_matrices=False)
        r, c = g.shape
        for k in ks:
            kk = min(k, len(s))
            rec = np.rint((u[:, :kk] * s[:kk]) @ vt[:kk])
            acc[k].append(float((rec == g).mean()))
            acc_ns[k].append(float((rec == g)[ns].mean()))
            irec = ((iu[:, :kk] * is_[:kk]) @ ivt[:kk]) > 0.5
            ink_acc[k].append(float((irec == ns).mean()))
            ink_prof[k].append(float((first_ink(irec) == first_ink(ns)).mean()))
            cost[k].append(kk * (r + c + 1) * 2 / (r * c))   # fp16 factors vs 1 byte per cell
        for kk in range(1, len(s) + 1):
            if (np.rint((u[:, :kk] * s[:kk]) @ vt[:kk]) == g).mean() >= 0.9:
                k90.append(kk / min(r, c))
                break
    m = lambda d: {str(k): round(float(np.mean(v)), 4) for k, v in d.items()}
    return {"bytes_exact": m(acc), "nonspace_bytes_exact": m(acc_ns), "ink_pixel_acc": m(ink_acc),
            "ink_indent_profile_rows": m(ink_prof), "storage_fp16_over_bytes": m(cost),
            "k_for_90pct_bytes_over_minRC": round(float(np.mean(k90)), 4)}


def part_d(fns, cols, blocks):
    acc = {f"{a}x{b}": [] for a, b in blocks}
    prof = {f"{a}x{b}": [] for a, b in blocks}
    for rows in fns:
        ns = grid(rows, cols) != 32
        for a, b in blocks:
            r, c = ns.shape
            pr_, pc = -r % a, -c % b
            pad = np.pad(ns.astype(np.float64), ((0, pr_), (0, pc)))
            m = pad.reshape(pad.shape[0] // a, a, pad.shape[1] // b, b).mean(axis=(1, 3))
            up = np.repeat(np.repeat(m, a, axis=0), b, axis=1)[:r, :c] > 0.5
            acc[f"{a}x{b}"].append(float((up == ns).mean()))
            prof[f"{a}x{b}"].append(float((first_ink(up) == first_ink(ns)).mean()))
    m = lambda d: {k: round(float(np.mean(v)), 4) for k, v in d.items()}
    return {"pixel_acc": m(acc), "indent_profile_rows": m(prof)}


def figure(rows, cols, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = grid(rows, cols)
    ns = (g != 32).astype(np.float64)
    iu, is_, ivt = np.linalg.svd(ns, full_matrices=False)
    w = max(len(r) for r in rows) + 2
    panels = [("bytes (value)", g[:, :w], "gray_r"), ("ink map (non-space)", ns[:, :w], "Greys")]
    for k in (1, 2, 4):
        panels.append((f"ink rank {k} (two matrices)", ((iu[:, :k] * is_[:k]) @ ivt[:k])[:, :w] > 0.5, "Greys"))
    for a, b in ((1, 4), (4, 8)):
        r, c = ns.shape
        pad = np.pad(ns, ((0, -r % a), (0, -c % b)))
        m = pad.reshape(pad.shape[0] // a, a, pad.shape[1] // b, b).mean(axis=(1, 3))
        panels.append((f"pyramid block {a}x{b}", np.repeat(np.repeat(m, a, 0), b, 1)[:r, :w] > 0.5, "Greys"))
    fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 0.22 * len(rows) + 1.2))
    for ax, (t, im, cm) in zip(axs, panels):
        ax.imshow(im, cmap=cm, aspect="auto", interpolation="nearest")
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/media/llm-data/datasets/pycode")
    ap.add_argument("--out", default="/media/llm-data/universe-runs/bytebrain-pycode-20261022/byte_matrix")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=40)
    ap.add_argument("--max-cols", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(json.dumps({"predictions": PRED}, ensure_ascii=False), flush=True)
    for corpus in ("clean", "raw"):
        fns, info = functions(os.path.join(a.data, f"{corpus}_train.txt"), a.n, a.seed, a.max_rows, a.max_cols)
        res = {"corpus": corpus, "n_functions": len(fns), **info, "max_rows": a.max_rows, "max_cols": a.max_cols,
               "A_indent": part_a(fns), "B_newline": part_b(fns),
               "C_two_matrices": part_c(fns, a.max_cols, (1, 2, 4, 8, 16)),
               "D_pyramid": part_d(fns, a.max_cols, ((1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 8)))}
        json.dump(res, open(os.path.join(a.out, f"{corpus}.json"), "w"), indent=1, ensure_ascii=False)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        ex = sorted((f for f in fns if 12 <= len(f) <= 22), key=lambda f: -len({indent(r) for r in f if r.strip()}))
        if ex:
            figure(ex[0], a.max_cols, os.path.join(a.out, f"example_{corpus}.png"))


if __name__ == "__main__":
    main()
