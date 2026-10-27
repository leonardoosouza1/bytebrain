"""#53 FAILED-ADAPT (rule 7): adapted water on the same frequency relief, dev-selected, frozen, held-out eval.

Written 2026-10-27 11:31 (from `date`), BEFORE any run of this file.

HIS FORMULATION (#53, kept): frequencies of bytes and junctions make the relief, water fills it, ponds / flow mark
holes and gaps in the code. Relief = relief53.py (byte n-gram + token-junction + line-junction surprisal, z-scored).

WHY THE FIRST WATER LOST (relief53.py run 2, read from run.log / basins.log / position_check_run2.json):
  FLOOD: canvas plain = 0 and every code row touches column 0 next to the outlet, so every pond spills at ~0 and its
    depth is S itself; flood-max ranks lines like relief-max by construction (delta within +-0.6 pp).
  SHALLOW / BYTEPOUR: rain falls on every cell and diffuses 400 steps; the pond at a sharp one-cell surprise is
    spread over its neighbours (peak smeared) and long / densely surprising lines collect more water than the
    single defect cell, so max depth points at "wet areas" not at the sharpest surprise (-4..-16 pp).
  BASIN VOLUME: same, volume rewards large surprising regions, not the defect (-14..-67 pp).
  Common cause: the water had either NO local rim (flood) or NO limit on how far it travels (shallow, basins).

ADAPTED WATER (the three options the lead listed, each with a limited reach so a pond is a LOCAL hole):
  W1 POND2D (a): ponds with a local rim. Depth = closing(h) - h on the 2D (row, col) relief, h = -S, flat window
     kr x kc; water fills only holes narrower than the window and is held by the nearest expected code (the rim),
     no global outlet. (Grayscale closing = rain filling on a terrain whose holes are at most kr x kc wide.)
  W2 GAPPOND (b): missing structure gets its own cells. Ghost rows are interleaved between code rows; a ghost row's
     height is -z(line junction INTO the next row) (END junction after the last row); code rows keep only the byte
     and token channels. A deleted line / deleted return / swap leaves no surprising byte but an unexpected
     junction: the pond forms IN THE GAP between two rows. Row score = max(own depth, gap above, g * gap below).
  W3a DAM (c): water flows in reading order (byte order, row by row); surprise is a wall (h = +S); water is held
     behind the first higher wall within k cells: depth_i = max(h_i..h_{i+k}) - h_i. The pond lies in front of the
     surprise.
  W3b READ-POND (c'): rain on the 1D reading-order relief h = -S, ponds narrower than k bytes (1D closing).
  MIX: relief_row + lam * water_row for the dev-best water config (does water carry information relief lacks?).
  Reference arms: random, longest line, parse-error line, relief alone, original SHALLOW (relief53 params).

SPLITS (gate 4): eval functions (1373, relief53.ev) split by a fixed permutation (seed 5300): 40% DEV functions,
  60% EVAL functions. DEV = seeds 0 and 5, instances whose function is in DEV. EVAL = seeds 1 2 3 4 (1, 2 identical
  to relief53 run 2 instances), instances whose function is in EVAL. Grid choice uses DEV only (macro top-1 over
  the 6 real defect types), is frozen to frozen.json, then EVAL runs once. The GPU run (--frozen) re-reads it.
GATES: baselines random / longest / parse / relief; positive control junk; same relief, same instances, same
  tie-break for every arm; grid edges flagged in the log (gate 8); per-seed numbers in eval.json.

PREDICTIONS (EVAL, paired top-1 delta vs relief, cluster CI over functions; written before running):
  A1 W1 POND2D: dev picks a small window; macro delta in [-3, +2] pp, no real type with CI above 0.
  A2 W2 GAPPOND: delete_line +3..+10 pp (CI above 0), swap 0..+5, delete_return -3..+3, wrong_indent / foreign /
     rename -3..0; macro -1..+3 pp. This is the arm the mechanism supports most (p ~ 0.4 that delete_line CI > 0).
  A3 W3a DAM: the pond lands on the row BEFORE the surprise: single-row types -10..-30 pp, macro <= -8 pp.
  A4 W3b READ-POND: like A1 with kr = 1: macro within [-3, +2] pp.
  A5 MIX: macro 0..+3 pp, CI includes 0 (p ~ 0.6).
  A6 junk positive control: >= 0.95 top-1 for W1, W2, W3b, MIX; relief 1.00 again.
  A7 overall: P(any pure-water arm has macro CI entirely above 0) ~ 0.25.

11:34 (after a 20-per-type smoke, before the full run; predictions unchanged): grids widened because smoke dev bests
  sat on grid edges (gate 8): W1 kr 9 / kc 61 added, W2 kr 9 / kc 1 added, W3a k 1, 2 added, W3b k 243, 729 added.
  The smoke also printed eval numbers on 20/type eval instances; they did not inform anything but the edge rule.

Run (CPU now; GPU busy):
  cd /home/leonardo/projects/LLM/bytebrain/research/pycode && HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 5 timeout 5400 \
  /home/leonardo/projects/LLM/.venv-rocm/bin/python -u relief53_adapt.py --device cpu
GPU (5 eval seeds, frozen from the CPU dev selection; queued script gpuq/pending/40_relief53.sh):
  relief53_adapt.py --device cuda --mem-frac 0.5 --frozen <OUT>/adapt/frozen.json --eval-seeds 1 2 3 4 6 \
    --out <OUT>/adapt_gpu
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relief53 as R  # noqa: E402

OUT0 = R.OUT
REAL = ("delete_line", "delete_return", "swap", "wrong_indent", "foreign_token", "rename")
H1, W1W = R.RMAX, R.CMAX + 1
H2 = 2 * R.RMAX + 1

GRID = {
    "W1": [{"kr": kr, "kc": kc} for kr in (1, 3, 5, 9) for kc in (1, 3, 7, 15, 31, 61) if (kr, kc) != (1, 1)],
    "W2": [{"kr": kr, "kc": kc, "g": g} for kr in (3, 5, 9) for kc in (1, 3, 9, 31) for g in (0.0, 0.5, 1.0)],
    "W3a": [{"k": k} for k in (1, 2, 4, 16, 64)],
    "W3b": [{"k": k} for k in (3, 9, 27, 81, 243, 729)],
}
LAMS = (0.25, 0.5, 1.0, 2.0, 4.0)
log = R.log


def name(fam, c):
    return fam + "(" + ",".join(f"{k}={v}" for k, v in c.items()) + ")"


# ------------------------------------------------------------------ data (same procedure as relief53.main)
def build_eval_set():
    train_text = open(R.TRAIN, encoding="utf-8", errors="replace").read()
    train_h = {R.h16(f) for f in R.split_fns(train_text)}
    del train_text
    hold = R.split_fns(open(R.HOLD, encoding="utf-8", errors="replace").read())
    ev = [f for f in hold if R.h16(f) not in train_h and R.eligible(f)]
    return ev


def gen_instances(ev, seeds, per_type, foreign_pool, rename_pool, keep_fids):
    inst = []
    for seed in seeds:
        for kind in R.DEFECTS:
            rng = np.random.default_rng([seed, R.DEFECTS.index(kind), 53])
            order = rng.permutation(len(ev))
            got = 0
            for fid in order:
                fid = int(fid)
                d = R.make_defect(kind, ev[fid], rng, foreign_pool, rename_pool)
                if d is None or d[0] == ev[fid]:
                    continue
                got += 1                      # identical stream to relief53 (filter applied after generation)
                if fid in keep_fids:
                    inst.append({"seed": seed, "kind": kind, "fid": fid, "text": d[0], "targets": sorted(d[1])})
                if got == per_type:
                    break
    return inst


# ------------------------------------------------------------------ relief per text
def relief_grids(t, bng, jn, stats):
    c = R.channels(t, bng, jn)
    z = {ch: np.where(c["mask"], np.nan_to_num((c[ch] - stats[ch][0]) / stats[ch][1], nan=0.0), 0.0)
         for ch in ("byte", "tok", "line")}
    rows = t.split("\n")
    nonempty = [i for i, r in enumerate(rows) if r.strip()]
    last = nonempty[-1]
    lens = [len(r.encode()) + 1 for r in rows]
    jin = np.zeros(len(rows))
    for r in nonempty[1:]:
        jin[r] = z["line"][r, 0]
    jend = z["line"][last, lens[last] - 1]
    S = z["byte"] + z["tok"] + z["line"]
    return {"S": S, "BT": z["byte"] + z["tok"], "mask": c["mask"], "jin": jin, "jend": jend, "last": last,
            "lens": lens}


# ------------------------------------------------------------------ water arms (batched torch)
def closing2d(h, kr, kc):
    """h (B,1,H,W); flat kr x kc window; out-of-canvas ignored."""
    p = (kr // 2, kc // 2)
    dil = F.max_pool2d(h, (kr, kc), stride=1, padding=p)
    return -F.max_pool2d(-dil, (kr, kc), stride=1, padding=p)


def w1_scores(G, texts, dev, cfgs):
    out = {t: {} for t in texts}
    for b in range(0, len(texts), 256):
        part = texts[b:b + 256]
        h = torch.zeros(len(part), 1, H1, W1W)
        m = torch.zeros(len(part), 1, H1, W1W, dtype=torch.bool)
        for i, t in enumerate(part):
            g = G[t]
            r, c = g["S"].shape
            h[i, 0, :r, :c] = torch.from_numpy(-g["S"]).float()
            m[i, 0, :r, :c] = torch.from_numpy(g["mask"])
        h, m = h.to(dev), m.to(dev)
        for cf in cfgs:
            d = closing2d(h, cf["kr"], cf["kc"]) - h
            rs = torch.where(m, d, torch.full_like(d, -float("inf"))).amax(3)[:, 0].cpu().numpy()
            for i, t in enumerate(part):
                out[t][name("W1", cf)] = rs[i, :G[t]["S"].shape[0]]
    return out


def w2_scores(G, texts, dev, cfgs):
    out = {t: {} for t in texts}
    for b in range(0, len(texts), 256):
        part = texts[b:b + 256]
        h = torch.zeros(len(part), 1, H2, W1W)
        cm = torch.zeros(len(part), 1, H2, W1W, dtype=torch.bool)
        gm = torch.zeros(len(part), 1, H2, W1W, dtype=torch.bool)
        for i, t in enumerate(part):
            g = G[t]
            nr = g["S"].shape[0]
            L = g["lens"]
            for r in range(nr):
                h[i, 0, 2 * r + 1, :W1W] = torch.from_numpy(-g["BT"][r]).float()
                cm[i, 0, 2 * r + 1, :L[r]] = True
                if r > 0:
                    w = max(L[r - 1], L[r])
                    h[i, 0, 2 * r, :w] = -float(g["jin"][r])
                    gm[i, 0, 2 * r, :w] = True
            la = g["last"]
            h[i, 0, 2 * la + 2, :L[la]] = -float(g["jend"])
            gm[i, 0, 2 * la + 2, :L[la]] = True
        h, cm, gm = h.to(dev), cm.to(dev), gm.to(dev)
        ninf = torch.full_like(h, -float("inf"))
        for kr, kc in sorted({(c["kr"], c["kc"]) for c in cfgs}):
            d = closing2d(h, kr, kc) - h
            own = torch.where(cm, d, ninf).amax(3)[:, 0].cpu().numpy()       # (B, H2)
            gap = torch.where(gm, d, ninf).amax(3)[:, 0].cpu().numpy()
            gap = np.where(np.isfinite(gap), gap, 0.0)
            for cf in [c for c in cfgs if (c["kr"], c["kc"]) == (kr, kc)]:
                for i, t in enumerate(part):
                    nr = G[t]["S"].shape[0]
                    o = own[i, 1:2 * nr:2]
                    above = gap[i, 0:2 * nr:2]
                    below = gap[i, 2:2 * nr + 1:2]
                    out[t][name("W2", cf)] = np.maximum(np.maximum(o, above), cf["g"] * below)
    return out


def seq_of(g):
    vals, rows = [], []
    for r, L in enumerate(g["lens"]):
        vals.append(g["S"][r, :L])
        rows.append(np.full(L, r))
    return np.concatenate(vals), np.concatenate(rows)


def w3_scores(G, texts, dev, cfgs_a, cfgs_b):
    out = {t: {} for t in texts}
    for b in range(0, len(texts), 256):
        part = texts[b:b + 256]
        seqs = [seq_of(G[t]) for t in part]
        n = max(len(s) for s, _ in seqs)
        S = torch.zeros(len(part), 1, n)
        v = torch.zeros(len(part), 1, n, dtype=torch.bool)
        for i, (s, _) in enumerate(seqs):
            S[i, 0, :len(s)] = torch.from_numpy(s).float()
            v[i, 0, :len(s)] = True
        S, v = S.to(dev), v.to(dev)
        ninf = torch.full_like(S, -float("inf"))
        res = {}
        for cf in cfgs_a:                      # DAM: h = +S, level = max of h over the next k cells (incl. self)
            k = cf["k"]
            hh = torch.where(v, S, ninf)
            lev = F.max_pool1d(F.pad(hh, (0, k), value=-float("inf")), k + 1, stride=1)
            res[name("W3a", cf)] = (lev - S).cpu().numpy()[:, 0]
        for cf in cfgs_b:                      # READ-POND: 1D closing of h = -S, out-of-sequence ignored
            k = cf["k"]
            hh = torch.where(v, -S, ninf)
            dil = F.max_pool1d(hh, k, stride=1, padding=k // 2)
            dil = torch.where(v, dil, torch.full_like(dil, float("inf")))
            clo = -F.max_pool1d(-dil, k, stride=1, padding=k // 2)
            res[name("W3b", cf)] = (clo - (-S)).cpu().numpy()[:, 0]
        for i, t in enumerate(part):
            s, rws = seqs[i]
            nr = G[t]["S"].shape[0]
            for nm, arr in res.items():
                rs = np.full(nr, -np.inf)
                np.maximum.at(rs, rws, arr[i, :len(s)])
                out[t][nm] = rs
    return out


def shallow_scores(G, texts, dev):
    out = {}
    Hc, Wc = R.RMAX + 2, R.CMAX + 3
    for b in range(0, len(texts), 512):
        part = texts[b:b + 512]
        h = R.canvas([-G[t]["S"] for t in part], Hc, Wc).to(dev)
        om = R.outlet_mask([G[t]["S"].shape[0] for t in part], Hc, Wc, dev)
        sd = R.shallow(h, outlet=om)[0].cpu().numpy()
        for k, t in enumerate(part):
            nr = G[t]["S"].shape[0]
            out[t] = R.line_scores(sd[k, 1:1 + nr, 1:R.CMAX + 2], G[t]["mask"], "max")
    return out


# ------------------------------------------------------------------ evaluation helpers
def boot(vals, fids, n=1000, seed=0):
    r = np.random.default_rng(seed)
    u, inv = np.unique(fids, return_inverse=True)
    sm = np.bincount(inv, weights=vals, minlength=len(u))
    ct = np.bincount(inv, minlength=len(u)).astype(float)
    pick = r.integers(0, len(u), size=(n, len(u)))
    bs = sm[pick].sum(1) / ct[pick].sum(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def hits(inst, rowscores, arms, seed=53):
    """rowscores: text -> arm -> (R,) array. Returns list of per-instance dicts with arm@1, arm@3."""
    rng = np.random.default_rng(seed)
    rec = []
    for i in inst:
        t = i["text"]
        nr = len(t.split("\n"))
        row = {"seed": i["seed"], "kind": i["kind"], "fid": i["fid"]}
        for a in arms:
            s = np.asarray(rowscores[t][a], float) + rng.random(nr) * 1e-7
            o = np.argsort(-s)
            row[a + "@1"] = int(o[0] in i["targets"])
            row[a + "@3"] = int(any(x in i["targets"] for x in o[:3]))
        rec.append(row)
    return rec


def macro(rec, arm):
    return float(np.mean([np.mean([r[arm + "@1"] for r in rec if r["kind"] == k]) for k in REAL]))


def macro_delta(rec, arm, n=1000, seed=0):
    """macro (mean over the 6 real types of per-type mean) paired delta vs relief; cluster bootstrap over fid,
    per-type means recomputed inside each replicate (review fix 2)."""
    fids = sorted({r["fid"] for r in rec})
    ix = {f: k for k, f in enumerate(fids)}
    per = []
    for kind in REAL:
        rr = [r for r in rec if r["kind"] == kind]
        per.append((np.array([ix[r["fid"]] for r in rr]), np.array([r[arm + "@1"] - r["relief@1"] for r in rr], float)))
    point = float(np.mean([d.mean() for _, d in per]))
    rng = np.random.default_rng(seed)
    bs = np.empty(n)
    for b in range(n):
        w = np.bincount(rng.integers(0, len(fids), len(fids)), minlength=len(fids)).astype(float)
        bs[b] = np.mean([(w[f] * d).sum() / max(w[f].sum(), 1e-12) for f, d in per])
    return point, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarize(erec, arms, water, seeds):
    res = {"types": {}, "macro": {}, "macro_per_seed": {}}
    for kind in R.DEFECTS:
        rr = [r for r in erec if r["kind"] == kind]
        fids = np.array([r["fid"] for r in rr])
        e = {"n": len(rr)}
        for arm in arms:
            for k in ("@1", "@3"):
                v = np.array([r[arm + k] for r in rr], float)
                e[arm + k] = float(v.mean())
                e[arm + k + "_ci"] = boot(v, fids)
            e[arm + "@1_per_seed"] = {str(s): float(np.mean([r[arm + "@1"] for r in rr if r["seed"] == s]))
                                     for s in seeds}
        for arm in water:
            d = np.array([r[arm + "@1"] - r["relief@1"] for r in rr], float)
            e["delta " + arm] = float(d.mean())
            e["delta " + arm + "_ci"] = boot(d, fids)
        res["types"][kind] = e
    for arm in arms:
        res["macro"][arm] = macro(erec, arm)
    for arm in water:
        res["macro"]["delta " + arm], res["macro"]["delta " + arm + "_ci"] = macro_delta(erec, arm)
        for s in seeds:
            rs = [r for r in erec if r["seed"] == s]
            res["macro_per_seed"].setdefault(arm, {})[str(s)] = macro_delta(rs, arm, n=200)
        clean = [r for r in erec if r["seed"] not in (1, 2)]   # seeds never seen while designing the arms (fix 3)
        if clean:
            res["macro"]["delta " + arm + " clean_seeds"], res["macro"]["delta " + arm + " clean_seeds_ci"] = \
                macro_delta(clean, arm)
    return res


def report(res, arms, water, short, seeds):
    log(f"=== EVAL top-1 (seeds {seeds}, eval functions only)")
    log("kind".ljust(14) + "".join(short.get(x, x)[:10].rjust(10) for x in arms))
    for kind, e in res["types"].items():
        log(kind.ljust(14) + "".join(f"{e[x + '@1']:.3f}".rjust(10) for x in arms))
    log("macro6".ljust(14) + "".join(f"{res['macro'][x]:.3f}".rjust(10) for x in arms))
    log("=== EVAL paired delta top-1 vs relief [95% cluster CI]; macro6 = mean of per-type deltas")
    for kind, e in list(res["types"].items()) + [("macro6", res["macro"])]:
        log(kind.ljust(16) + "  ".join(f"{short[w]} {e['delta ' + w]:+.3f} [{e['delta ' + w + '_ci'][0]:+.3f},"
                                       f"{e['delta ' + w + '_ci'][1]:+.3f}]" for w in water))
    if any(" clean_seeds" in k for k in res["macro"]):
        log("macro6 clean".ljust(16) + "  ".join(
            f"{short[w]} {res['macro']['delta ' + w + ' clean_seeds']:+.3f} "
            f"[{res['macro']['delta ' + w + ' clean_seeds_ci'][0]:+.3f},"
            f"{res['macro']['delta ' + w + ' clean_seeds_ci'][1]:+.3f}]" for w in water))
    for w in water:
        log(f"per-seed macro delta {short[w]}: " + "  ".join(f"s{s} {v[0]:+.3f}" for s, v in
                                                            res["macro_per_seed"][w].items()))


def base_scores(t, g, rng):
    rows = t.split("\n")
    nr = len(rows)
    pl = R.parse_line(t)
    pr = rng.random(nr)
    if pl is not None:
        pr[pl] += 3.0
        for d in (-1, 1):
            if 0 <= pl + d < nr:
                pr[pl + d] += 2.0
    return {"random": rng.random(nr), "longest": np.array([len(r.encode()) for r in rows], float),
            "parse": pr, "relief": R.line_scores(g["S"], g["mask"], "max")}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-frac", type=float, default=0.5)
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--per-type", type=int, default=300)
    ap.add_argument("--frozen", default=None, help="frozen.json from a dev run: skip the sweep")
    ap.add_argument("--out", default=os.path.join(OUT0, "adapt"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--restat", action="store_true", help="recompute eval.json stats from eval_instances.jsonl")
    a = ap.parse_args()
    if a.restat:
        frozen = json.load(open(os.path.join(a.out, "frozen.json")))
        erec = [json.loads(l) for l in open(os.path.join(a.out, "eval_instances.jsonl"))]
        water = [name(f, frozen[f]) for f in ("W1", "W2", "W3a", "W3b")] + ["MIX", "shallow_orig"]
        arms = ["random", "longest", "parse", "relief"] + water
        seeds = sorted({r["seed"] for r in erec})
        res = summarize(erec, arms, water, seeds)
        res.update({"frozen": frozen, "eval_seeds": seeds, "restat": time.strftime("%Y-%m-%d %H:%M")})
        json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
        short = {name(f, frozen[f]): f for f in ("W1", "W2", "W3a", "W3b")}
        short.update({"MIX": "MIX", "shallow_orig": "SHALLOW0"})
        report(res, arms, water, short, seeds)
        return
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac, 0)
    t0 = time.time()

    bng = R.ByteNgram(None, os.path.join(OUT0, "byte_ngram.npz"))
    jn = R.Junctions(None, os.path.join(OUT0, "junctions.json"))
    jn.setup()
    stats = json.load(open(os.path.join(OUT0, "results.json")))["config"]["calib_stats"]
    ev = build_eval_set()
    assert len(ev) == 1373, len(ev)
    import keyword
    rename_pool = [w for w in jn.vocab[:400] if not keyword.iskeyword(w) and w not in dir(__builtins__)
                   and w != "self" and len(w) > 1]
    foreign_pool = []
    for f in ev:
        for r in f.split("\n")[1:]:
            foreign_pool += [t[2] for t in R.tokens(r)]
    perm = np.random.default_rng(5300).permutation(len(ev))
    dev_f = set(int(x) for x in perm[:int(0.4 * len(ev))])
    eval_f = set(range(len(ev))) - dev_f
    per = 20 if a.smoke else a.per_type

    # identity check vs relief53 run 2 (seeds 1, 2 unfiltered must equal defects.jsonl)
    old = [json.loads(l) for l in open(os.path.join(OUT0, "defects.jsonl"))]
    nchk = min(per, 300)                    # relief53 wrote 300 per type; compare per kind (review fix 1)
    chk = gen_instances(ev, [1], nchk, foreign_pool, rename_pool, set(range(len(ev))))
    same = 0
    for kind in R.DEFECTS:
        ok = [(o["fid"], o["text"]) for o in old if o["seed"] == 1 and o["kind"] == kind][:nchk]
        nw = [(c["fid"], c["text"]) for c in chk if c["kind"] == kind]
        same += sum(x == y for x, y in zip(ok, nw)) if len(ok) == len(nw) else 0
    log(f"identity check seed 1 vs relief53 defects.jsonl: {same}/{len(chk)} identical")
    if not a.smoke:
        assert same == len(chk), "instance stream differs from relief53"

    rng0 = np.random.default_rng(53)
    G = {}

    def prep(texts):
        for t in texts:
            if t not in G:
                G[t] = relief_grids(t, bng, jn, stats)

    # -------- DEV sweep
    if a.frozen is None:
        dinst = gen_instances(ev, a.dev_seeds, per, foreign_pool, rename_pool, dev_f)
        dtexts = sorted({i["text"] for i in dinst})
        log(f"DEV: {len(dinst)} instances on {len(dev_f)} dev functions, seeds {a.dev_seeds}")
        prep(dtexts)
        RS = {t: base_scores(t, G[t], rng0) for t in dtexts}
        for src in (w1_scores(G, dtexts, dev, GRID["W1"]), w2_scores(G, dtexts, dev, GRID["W2"]),
                    w3_scores(G, dtexts, dev, GRID["W3a"], GRID["W3b"])):
            for t in dtexts:
                RS[t].update(src[t])
        arms = ["random", "longest", "parse", "relief"] + [name(f, c) for f in GRID for c in GRID[f]]
        drec = hits(dinst, RS, arms)
        grid = {arm: {"macro@1": macro(drec, arm),
                      **{k: float(np.mean([r[arm + "@1"] for r in drec if r["kind"] == k])) for k in R.DEFECTS}}
                for arm in arms}
        log("DEV macro top-1 (6 real types): relief %.3f" % grid["relief"]["macro@1"])
        frozen = {}
        for f in GRID:
            best = max(GRID[f], key=lambda c: grid[name(f, c)]["macro@1"])
            frozen[f] = best
            for c in GRID[f]:
                log(f"  {name(f, c):28s} macro {grid[name(f, c)]['macro@1']:.3f}"
                    + ("  <- best" if c is best else ""))
            for k, v in best.items():
                vals = sorted({c[k] for c in GRID[f]})
                if len(vals) > 1 and v in (vals[0], vals[-1]):
                    log(f"  GRID EDGE: {f} best {k}={v} at the edge of {vals}")
        bw = max(frozen, key=lambda f: grid[name(f, frozen[f])]["macro@1"])
        bwn = name(bw, frozen[bw])
        for lam in LAMS:
            for t in dtexts:
                RS[t][f"MIX{lam}"] = RS[t]["relief"] + lam * RS[t][bwn]
        drec2 = hits(dinst, RS, [f"MIX{lam}" for lam in LAMS])
        mixm = {lam: macro(drec2, f"MIX{lam}") for lam in LAMS}
        lam = max(LAMS, key=lambda x: mixm[x])
        log(f"  MIX on {bwn}: " + "  ".join(f"lam {k}: {v:.3f}" for k, v in mixm.items()) + f"  -> lam {lam}")
        if lam in (LAMS[0], LAMS[-1]):
            log(f"  GRID EDGE: MIX lam={lam}")
        frozen["MIX"] = {"water": bw, "lam": lam}
        frozen["_meta"] = {"time": time.strftime("%Y-%m-%d %H:%M"), "dev_seeds": a.dev_seeds, "per_type": per,
                           "n_dev_inst": len(dinst), "n_dev_fns": len(dev_f)}
        json.dump(frozen, open(os.path.join(a.out, "frozen.json"), "w"), indent=1)
        json.dump(grid, open(os.path.join(a.out, "dev_grid.json"), "w"), indent=1)
        log(f"FROZEN {json.dumps(frozen)} ({time.time() - t0:.0f} s)")
    else:
        frozen = json.load(open(a.frozen))
        log(f"FROZEN (read {a.frozen}): {json.dumps(frozen)}")

    # -------- EVAL (once, frozen)
    einst = gen_instances(ev, a.eval_seeds, per, foreign_pool, rename_pool, eval_f)
    etexts = sorted({i["text"] for i in einst})
    log(f"EVAL: {len(einst)} instances on {len(eval_f)} eval functions, seeds {a.eval_seeds}")
    prep(etexts)
    RS = {t: base_scores(t, G[t], rng0) for t in etexts}
    for src in (w1_scores(G, etexts, dev, [frozen["W1"]]), w2_scores(G, etexts, dev, [frozen["W2"]]),
                w3_scores(G, etexts, dev, [frozen["W3a"]], [frozen["W3b"]])):
        for t in etexts:
            RS[t].update(src[t])
    sh = shallow_scores(G, etexts, dev)
    mw = name(frozen["MIX"]["water"], frozen[frozen["MIX"]["water"]])
    for t in etexts:
        RS[t]["shallow_orig"] = sh[t]
        RS[t]["MIX"] = RS[t]["relief"] + frozen["MIX"]["lam"] * RS[t][mw]
    water = [name(f, frozen[f]) for f in ("W1", "W2", "W3a", "W3b")] + ["MIX", "shallow_orig"]
    arms = ["random", "longest", "parse", "relief"] + water
    erec = hits(einst, RS, arms)
    res = summarize(erec, arms, water, a.eval_seeds)
    res.update({"frozen": frozen, "eval_seeds": a.eval_seeds, "n_eval_fns": len(eval_f)})
    res["wall_s"] = time.time() - t0
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)
    with open(os.path.join(a.out, "eval_instances.jsonl"), "w") as fh:
        for r in erec:
            fh.write(json.dumps(r) + "\n")

    short = {name(f, frozen[f]): f for f in ("W1", "W2", "W3a", "W3b")}
    short.update({"MIX": "MIX", "shallow_orig": "SHALLOW0"})
    report(res, arms, water, short, a.eval_seeds)
    log(f"done, wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
