"""Mirror: how far a model's functions are from real unseen code, axis by axis. No axis says which byte comes next.

The reference is held-out real functions (files never trained on), not the training corpus. Axes:
  match   - per byte, length of the longest suffix ending there that occurs in the training corpus (GPU index).
            Real new code reuses short/medium phrases; long matches = copying, short = invented noise.
  mi      - I(k) between byte classes at distance k (k=1..128), minus a shuffled-text floor: long-range structure.
  bigram  - KL(sample || real) of byte bigrams: local spelling.
  names   - of parsable functions: share of parameters that are used, share of loaded names that are bound in the
            function (params/assignments/loop vars) vs builtins vs undefined: coherence of a function.
  syntax  - parse rate and relative position of the first SyntaxError.
Each axis reports the sample value, the real value and a distance. Usage:
  mirror.py --ckpt CKPT --out FILE.json [--n 256] [--bytes 800]      # model samples vs real
  mirror.py --out FILE.json --real-only                               # real vs itself (split halves): noise floor
"""

import argparse
import ast
import builtins
import json
import random

import numpy as np
import torch

from corpus_index import CorpusIndex
from eval_samples import ByteLM, CachedGen, cut_function

D = "/media/llm-data/datasets/pycode"
BUILTINS = set(dir(builtins))
LENGTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
BINS = [0, 2, 4, 8, 16, 32, 48, 64]


def byte_class(a):
    """8 classes: lower, upper, digit, space, newline, open bracket, close bracket, other punctuation."""
    c = np.full(a.shape, 7, dtype=np.int64)
    c[(a >= 97) & (a <= 122)] = 0
    c[(a >= 65) & (a <= 90)] = 1
    c[(a >= 48) & (a <= 57)] = 2
    c[a == 32] = 3
    c[a == 10] = 4
    c[np.isin(a, [40, 91, 123])] = 5
    c[np.isin(a, [41, 93, 125])] = 6
    return c


def mi_curve(texts, ks):
    """I(k) over byte classes, pairs inside each text only; minus the same on byte-shuffled texts."""
    def mi(seqs):
        out = []
        for k in ks:
            joint = np.zeros((8, 8))
            for s in seqs:
                if len(s) > k:
                    np.add.at(joint, (s[:-k], s[k:]), 1)
            p = joint / joint.sum()
            px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
            nz = p > 0
            out.append(float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum()))
        return np.array(out)
    seqs = [byte_class(np.frombuffer(t.encode(), dtype=np.uint8)) for t in texts]
    r = np.random.default_rng(0)
    shuf = [r.permutation(s) for s in seqs]
    return mi(seqs) - mi(shuf)


def bigram(texts):
    h = np.zeros((256, 256))
    for t in texts:
        a = np.frombuffer(t.encode(), dtype=np.uint8)
        np.add.at(h, (a[:-1], a[1:]), 1)
    return h


def kl(p_counts, q_counts, eps=0.5):
    p = (p_counts + eps) / (p_counts + eps).sum()
    q = (q_counts + eps) / (q_counts + eps).sum()
    return float((p * np.log2(p / q)).sum())


@torch.no_grad()
def match_spectrum(idx, texts, per_text=64, seed=0):
    r = random.Random(seed)
    rows = []
    for t in texts:
        a = b"\n\n" + t.encode()
        pos = list(range(8, len(a)))
        for p in r.sample(pos, min(per_text, len(pos))):
            w = a[max(0, p - 47):p + 1]
            rows.append(list(w.rjust(48, b"\x00")))
    ms = []
    for s in range(0, len(rows), 512):
        m, _, _ = idx.next_bytes(torch.tensor(rows[s:s + 512], device="cuda"), lengths=LENGTHS, cap=1)
        ms.append(m.cpu().numpy())
    m = np.concatenate(ms)
    hist = np.histogram(m, bins=BINS + [10 ** 9])[0]
    return hist / hist.sum(), float(m.mean())


def names_axis(srcs):
    used_params, bound, builtin, undefined, n_fn = [], 0, 0, 0, 0
    for s in srcs:
        try:
            fn = ast.parse(s).body[0]
        except (SyntaxError, ValueError, IndexError, RecursionError):
            continue
        if not isinstance(fn, ast.FunctionDef):
            continue
        n_fn += 1
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
        params |= {a.arg for a in (fn.args.vararg, fn.args.kwarg) if a}
        stores = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}
        stores |= {n.name for n in ast.walk(fn) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n is not fn}
        stores |= {(a.asname or a.name).split(".")[0] for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in n.names}
        loads = [n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
        loaded = set(loads)
        if params - {"self", "cls"}:
            used_params.append(len((params - {"self", "cls"}) & loaded) / len(params - {"self", "cls"}))
        for x in loads:
            if x in params or x in stores:
                bound += 1
            elif x in BUILTINS:
                builtin += 1
            else:
                undefined += 1
    tot = max(1, bound + builtin + undefined)
    return {"functions": n_fn, "params_used": round(float(np.mean(used_params)), 3) if used_params else None,
            "loads_bound": round(bound / tot, 3), "loads_builtin": round(builtin / tot, 3),
            "loads_free": round(undefined / tot, 3)}


def syntax_axis(srcs):
    ok, pos = 0, []
    for s in srcs:
        try:
            ast.parse(s)
            ok += 1
        except SyntaxError as e:
            lines = s.split("\n")
            off = sum(len(l) + 1 for l in lines[:max(0, (e.lineno or 1) - 1)]) + (e.offset or 0)
            pos.append(min(1.0, off / max(1, len(s))))
        except (ValueError, RecursionError):
            pass
    return {"parses": round(ok / len(srcs), 3),
            "error_pos_hist": np.histogram(pos, bins=5, range=(0, 1))[0].tolist() if pos else []}


def axes(idx, srcs):
    spec, mean_m = match_spectrum(idx, srcs)
    return {"match_hist": spec.round(4).tolist(), "match_mean": round(mean_m, 2),
            "mi": mi_curve(srcs, KS).round(5).tolist(), "bigram": bigram(srcs),
            "names": names_axis(srcs), "syntax": syntax_axis(srcs),
            "len_mean": round(float(np.mean([len(s) for s in srcs])), 1)}


KS = [1, 2, 4, 8, 16, 32, 64, 128]


def compare(a, ref):
    """Distances of sample axes `a` from reference axes `ref`."""
    tv = 0.5 * float(np.abs(np.array(a["match_hist"]) - np.array(ref["match_hist"])).sum())
    mi_a, mi_r = np.array(a["mi"]), np.array(ref["mi"])
    return {"match_tv": round(tv, 4), "match_mean": [a["match_mean"], ref["match_mean"]],
            "mi_ratio_per_k": dict(zip(KS, (mi_a / np.where(mi_r > 0, mi_r, np.nan)).round(3).tolist())),
            "bigram_kl": round(kl(a["bigram"], ref["bigram"]), 4),
            "names": [a["names"], ref["names"]], "syntax": [a["syntax"], ref["syntax"]],
            "len_mean": [a["len_mean"], ref["len_mean"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bytes", type=int, default=800)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.3)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    torch.manual_seed(0)
    idx = CorpusIndex.cached(f"{D}/clean_train.txt", depth=64)
    hold = json.load(open(f"{D}/holdout_functions.json"))
    random.Random(2).shuffle(hold)
    real = [h["clean"] for h in hold if len(h["clean"]) <= args.bytes][:2 * args.n]
    ref = axes(idx, real[:args.n])
    if args.real_only:
        other, label = real[args.n:2 * args.n], "real_half_b"
    else:
        lm = ByteLM(args.ckpt)
        out = CachedGen(lm).generate(b"\n\ndef ", args.n, args.bytes, temp=args.temp).cpu().numpy()
        other = [cut_function("def " + bytes(r).decode("utf-8", "replace"))[0] for r in out]
        label = f"{args.ckpt}@{lm.step}"
    res = compare(axes(idx, other), ref)
    res["label"] = label
    res["examples"] = other[:6]
    res["samples"] = other
    with open(args.out, "w") as fh:
        json.dump(res, fh)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}))


if __name__ == "__main__":
    main()
