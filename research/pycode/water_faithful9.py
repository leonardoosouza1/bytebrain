"""Ledger #9 (with #12, #53): FAITHFUL water test on code maps -- Leonardo's iara-agua engine, unmodified laws.

Written 2026-10-27 10:05 (from `date`), BEFORE any code-map run.

Engine: /home/leonardo/projects/LLM/iara-agua, new additive subcommand `agua-lab codemap`
(crates/agua-lab/src/codemap.rs). It runs World<Gabor2D>::step exactly as `agua-lab denoise --auto`:
amplitude re-fit every tick, gradient geometry tuning, metabolic death, reproduction with mutation, rain
(16 competing seeds) where residual^2 is high, cap 3000. Map fed at NATIVE size (rows = lines, cols = byte
offset, no square resize). Atom count EMERGES: noisy = map + N(0, sigma^2), sigma_hat = MAD (never sees the
clean map), met = 40 * sigma_hat^2 (the documented law), 800 ticks (the LEI denoise battery setting).
The noise is required: on a noise-free binary map the MAD estimate is ~0 -> met ~0 -> no Morozov stop.
Positive control (same binary, done 10:04): `denoise --target assets/photo_texture.png --size 96 --sigma 0.15
--seed 11 --ticks 800 --auto` -> 23.73 dB / 291 atoms / evals 347798 == JOURNAL runs/_denoise/LEI/run_1.txt.

Data: 200 functions from clean_holdout.txt (never trained on by the water; held out of the byte-LM corpus),
10..80 rows, lines truncated at 80 bytes (not dropped), must ast.parse. Signals: ink (1 = non-space byte),
indent_stub (1 on [indent, indent+4) of each non-blank row). Surprisal signal: deferred (GPU busy) --
`python water_faithful9.py surprisal` (GPU only, not run).
Controls (same seed = same noise draw, paired per function, 3 seeds): rand_indent (same line texts, random
indent levels 0..max level, def row kept at 0), cross_fn (each row replaced by a row of another random
held-out function, same row count), shuffle (within-function line permutation).
Broken lines (ink + stub): del_line, wrong_indent (+-4 cols), del_return. Readouts per row: final residual^2,
deaths (late half), births, recon error vs clean; MP@Morozov same atoms as baseline. Localization metric =
percentile rank of the broken unit among all units (0 = most anomalous, chance = 0.5); deletions are scored on
gap units (adjacent row pairs, max), wrong_indent on rows. Baselines: random (0.5), MP residual, one-cue
indent-jump rule, parse-error line.

Gate answers (bytebrain CLAUDE.md pre-run gates): 1 baselines = random-row 0.5, one-cue indent-jump rule, MP
same atoms; 2 n/a (no LLM data); 3 gold enters nowhere in the solver (clean map only in err_row, labelled
oracle); 4 holdout file, disjoint from clean_train; 5 positive control above, exact; 6 n/a (no train/eval);
7 see predictions; 8 every arm = same ticks/k/sigma/seeds; 10 per-worker result files; 11 the part doing the
work is the engine's metabolic Morozov stop (count) and residual (where); 12 CPU only, no GPU.

PREDICTIONS (registered before running; sigma = 0.25, k = 40):
P1 indent_stub, N(control)/N(clean), paired over 200 fns (mean of log ratio):
   shuffle +15% (CI excludes 0; MP probe found +31% atoms at 99% energy), rand_indent +20%, cross_fn +10%.
   PASS for "water sees cross-line block structure" if shuffle and rand_indent both > 0 with CI excluding 0.
P2 ink: same direction but small, +3..+8%; may be inside CI (ink is dominated by within-line texture).
P3 block atoms (2*sigma >= 3 rows): fraction clean > shuffle on indent_stub by >= 2 pp.
P4 localization (percentile, lower = better), water residual:
   wrong_indent stub 0.40 (0.30-0.50), ink 0.45; del_line 0.48 (chance); del_return 0.50 (chance: a missing
   return leaves no hole in a layout map). deaths_late: chance (0.45-0.55). One-cue indent rule on
   wrong_indent beats the water (<= 0.25). Water vs MP residual: within +-0.05 (no extra localization).
KILL for "water localizes broken lines": wrong_indent stub percentile CI includes 0.5.

ADDED 10:08 (after launch, before any result was read): smoke showed sigma_hat/sigma = 1.12-1.16 on ink maps, i.e. the
MAD estimate depends on map structure, so a control can change met through sigma_hat. Confound arm:
`oracle_jobs` + `codemap --oracle-sigma` (met = 40*sigma_true^2), stub all 200 fns, ink fns % 4 == 0 (50).
Prediction for that arm: same signs as P1/P2, magnitudes within +-5 pp of the auto arm.

POST-HOC 10:12 (added after a debug look at 13 partial functions, where shuffle gave FEWER atoms, -19% ink): with a fixed
metabolism, N is not a description length -- atoms that cannot pay die and the structure stays in the residual.
Added readouts, labelled post-hoc in results: J = ||r||^2 + met*N (the AIC-form objective the engine's law
implies), final residual energy, recon error vs clean (oracle diagnostic), block atoms at sigma >= 3 and >= 6
(the registered 2*sigma >= 3 is met by ~95% of atoms since sigma_min = 1.2). P1/P2 stay as registered on N.
"""
import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import time

import numpy as np

CORPUS = "/media/llm-data/datasets/pycode/clean_holdout.txt"
OUT = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/water_faithful9"
BIN = "/media/llm-data/cargo-target/iara-agua/release/agua-lab"
WMAX, RMIN, RMAX = 80, 10, 80
SIGMA, SEEDS, TICKS = 0.25, (1, 2, 3), 800
CONTROLS = ("rand_indent", "cross_fn", "shuffle")
BROKEN = ("del_line", "wrong_indent", "del_return")
SIGNALS = ("ink", "stub")


# ------------------------------------------------------------------ data
def load_functions(n, seed=0):
    text = open(CORPUS, encoding="utf-8", errors="replace").read()
    fns = re.split(r"\n\n(?=@|def |async def )", text)
    order = list(range(len(fns)))
    random.Random(seed).shuffle(order)
    kept, pool = [], []
    for i in order:
        src = fns[i].rstrip("\n")
        lines = src.split("\n")
        if "\t" in src or not (RMIN <= len(lines) <= RMAX):
            pool.append(lines)
            continue
        try:
            ast.parse(src)
        except SyntaxError:
            continue
        (kept if len(kept) < n else pool).append(lines)
    return kept, pool


def indent(s):
    return len(s) - len(s.lstrip(" "))


def ink_map(lines, w):
    g = np.zeros((len(lines), w), np.float32)
    for i, s in enumerate(lines):
        b = s.encode("utf-8")[:w]
        g[i, :len(b)] = [c != 32 for c in b]
    return g


def stub_map(lines, w):
    g = np.zeros((len(lines), w), np.float32)
    for i, s in enumerate(lines):
        if s.strip():
            d = indent(s)
            g[i, min(d, w):min(d + 4, w)] = 1
    return g


def control(lines, kind, rng, pool):
    if kind == "shuffle":
        out = list(lines)
        rng.shuffle(out)
        return out
    if kind == "rand_indent":
        maxlev = max(indent(s) for s in lines) // 4
        out = [lines[0]]
        for s in lines[1:]:
            out.append(" " * (4 * rng.randint(0, maxlev)) + s.lstrip(" ") if s.strip() else s)
        return out
    if kind == "cross_fn":
        out = []
        for _ in lines:
            other = pool[rng.randrange(len(pool))]
            out.append(other[rng.randrange(len(other))])
        return out
    raise ValueError(kind)


def parse_err_line(lines):
    try:
        ast.parse("\n".join(lines))
        return None
    except SyntaxError as e:
        return None if e.lineno is None else max(0, min(len(lines) - 1, e.lineno - 1))
    except Exception:
        return None


def broken(lines, kind, rng):
    """returns (new_lines, unit, pos): unit 'row' -> pos = row index; 'gap' -> pos = gap index g (between rows
    g-1 and g of the new map; scored on pair (g-1, g))."""
    body = [i for i in range(1, len(lines)) if lines[i].strip()]
    if kind == "del_line":
        i = rng.choice(body)
        return lines[:i] + lines[i + 1:], "gap", i
    if kind == "wrong_indent":
        i = rng.choice(body)
        d = indent(lines[i])
        delta = 4 if (rng.random() < 0.5 or d < 8) else -4
        s = lines[i]
        s = " " * (d + delta) + s.lstrip(" ")
        return lines[:i] + [s] + lines[i + 1:], "row", i
    if kind == "del_return":
        rets = [i for i in body if lines[i].lstrip().startswith("return")]
        if not rets:
            return None
        i = rng.choice(rets)
        return lines[:i] + lines[i + 1:], "gap", i
    raise ValueError(kind)


# ------------------------------------------------------------------ build jobs
def build(n_fn, dump_fns):
    os.makedirs(OUT, exist_ok=True)
    fns, pool = load_functions(n_fn)
    meta, jobs = [], []
    for fi, lines in enumerate(fns):
        rng = random.Random(1000 + fi)
        lines = [s.encode("utf-8")[:WMAX].decode("utf-8", "ignore") for s in lines]  # truncate, not drop
        variants = {"clean": lines}
        for c in CONTROLS:
            variants[c] = control(lines, c, rng, pool)
        binfo = {}
        for b in BROKEN:
            r = broken(lines, b, rng)
            if r is None:
                continue
            nl, unit, pos = r
            nl = [s.encode("utf-8")[:WMAX].decode("utf-8", "ignore") for s in nl]
            variants[b] = nl
            binfo[b] = {"unit": unit, "pos": pos, "parse_err": parse_err_line(nl), "h": len(nl),
                        "indent": [indent(s) if s.strip() else -1 for s in nl],
                        "colon": [s.rstrip().endswith(":") for s in nl]}
        vs = {k: [s.encode("utf-8")[:WMAX].decode("utf-8", "ignore") for s in v] for k, v in variants.items()}
        w_ink = min(WMAX, max(len(s.encode("utf-8")) for v in vs.values() for s in v))
        w_stub = min(WMAX, max(indent(s) for v in vs.values() for s in v if s.strip()) + 4)
        meta.append({"fn": fi, "rows": len(lines), "w_ink": w_ink, "w_stub": w_stub, "broken": binfo,
                     "text": "\n".join(lines)})
        for vname, v in vs.items():
            for sig, w, mk in (("ink", w_ink, ink_map), ("stub", w_stub, stub_map)):
                g = mk(v, w)
                for sd in SEEDS:
                    jobs.append({"id": f"{fi}|{vname}|{sig}|{sd}", "h": g.shape[0], "w": g.shape[1],
                                 "data": [float(x) for x in g.ravel()], "sigma": SIGMA, "seed": sd,
                                 "dump": fi in dump_fns and sd == 1})
    # cost-ordered: controls first, broken second; within, interleave functions (stratified prefix)
    pri = {"clean": 0, "shuffle": 0, "rand_indent": 0, "cross_fn": 0, "wrong_indent": 1, "del_line": 1, "del_return": 1}
    jobs.sort(key=lambda j: (pri[j["id"].split("|")[1]], int(j["id"].split("|")[0]) % 4, int(j["id"].split("|")[0])))
    with open(f"{OUT}/jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    json.dump(meta, open(f"{OUT}/meta.json", "w"))
    print(f"functions {len(fns)}  jobs {len(jobs)}  rows median {np.median([m['rows'] for m in meta])}  "
          f"w_ink median {np.median([m['w_ink'] for m in meta])}", flush=True)


# ------------------------------------------------------------------ analysis helpers
def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcox(x):
    from scipy import stats
    x = np.asarray(x, float)
    x = x[x != 0]
    return float(stats.wilcoxon(x).pvalue) if len(x) > 5 else float("nan")


def pct_rank(scores, k):
    """percentile of unit k among all units, 0 = highest score; ties averaged; chance 0.5."""
    s = np.asarray(scores, float)
    n = len(s)
    if n < 2:
        return float("nan")
    above = (s > s[k]).sum()
    ties = (s == s[k]).sum() - 1
    return float((above + 0.5 * ties) / (n - 1))


def unit_scores(row_score, unit):
    s = np.asarray(row_score, float)
    if unit == "row":
        return s
    # gap g in 1..h-1 (between rows g-1, g) plus gap h (after last row: only row h-1) and gap 0 is excluded
    pairs = [max(s[g - 1], s[g]) for g in range(1, len(s))] + [s[-1]]
    return np.array(pairs)


def unit_index(unit, pos, h):
    if unit == "row":
        return pos
    return min(pos, h) - 1  # gap pos -> index in unit_scores list (pos in 1..h)


def indent_rule(info):
    ind, col = info["indent"], info["colon"]
    sc, prev_i, prev_c = [], 0, True
    for i, d in enumerate(ind):
        if d < 0:
            sc.append(0.0)
            continue
        jump = d - prev_i
        bad = (jump > 0 and not prev_c) or jump > 4 or (d % 4 != 0)
        sc.append(abs(jump) + (100.0 if bad else 0.0) if i > 0 else 0.0)
        prev_i, prev_c = d, col[i]
    return sc


def block_frac(atoms, smin=1.5):
    if not atoms:
        return 0.0, 0.0
    tall = [a for a in atoms if a["sigma"] >= smin]
    vert = [a for a in tall if abs(np.cos(a["theta"])) > 0.9]
    return len(tall) / len(atoms), len(vert) / len(atoms)


def _derive(d):
    d["res_energy_posthoc"] = float(sum(d["res_row"]))
    d["err_clean_posthoc"] = float(sum(d["err_row"]))
    d["J_posthoc"] = d["res_energy_posthoc"] + d["met"] * d["n_atoms"]
    return d


def load_results(prefix="res_w"):
    R = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.startswith(prefix) and fn.endswith(".jsonl"):
            for line in open(f"{OUT}/{fn}"):
                if line.strip():
                    d = _derive(json.loads(line))
                    R[d["id"]] = d
    return R


def analyze(prefix="res_w"):
    meta = json.load(open(f"{OUT}/meta.json"))
    R = load_results(prefix)
    out = {"n_results": len(R), "sigma": SIGMA, "ticks": TICKS}
    fns_done = sorted({int(k.split("|")[0]) for k in R})
    # --- structure: emergent N, clean vs controls, paired by fn (mean over seeds of log ratio)
    struct = {}
    for sig in SIGNALS:
        for c in CONTROLS:
            for key in ("n_atoms", "mp_n", "J_posthoc", "res_energy_posthoc", "err_clean_posthoc"):
                lr, frac_more = [], []
                for fi in fns_done:
                    v = []
                    for sd in SEEDS:
                        a, b = R.get(f"{fi}|clean|{sig}|{sd}"), R.get(f"{fi}|{c}|{sig}|{sd}")
                        if a and b and a[key] > 0 and b[key] > 0:
                            v.append(np.log(b[key] / a[key]))
                    if len(v) == len(SEEDS):
                        lr.append(np.mean(v))
                m, lo, hi = boot_ci(lr)
                struct[f"{sig}/{c}/{key}"] = {"n_fn": len(lr), "mean_pct": 100 * (np.exp(m) - 1),
                                              "ci_pct": [100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1)],
                                              "wilcoxon_p": wilcox(lr),
                                              "frac_ctrl_more": float(np.mean(np.array(lr) > 0)) if lr else None}
            # sigma_hat confound and block atoms
            sh = []
            bfs = {sm: ([], []) for sm in (1.5, 3.0, 6.0)}
            for fi in fns_done:
                d_sh = []
                d_b = {sm: ([], []) for sm in bfs}
                for sd in SEEDS:
                    a, b = R.get(f"{fi}|clean|{sig}|{sd}"), R.get(f"{fi}|{c}|{sig}|{sd}")
                    if a and b:
                        d_sh.append(b["sigma_hat"] / a["sigma_hat"] - 1)
                        for sm in bfs:
                            ba, va = block_frac(a["atoms"], sm)
                            bb, vb = block_frac(b["atoms"], sm)
                            d_b[sm][0].append(ba - bb); d_b[sm][1].append(va - vb)
                if len(d_sh) == len(SEEDS):
                    sh.append(np.mean(d_sh))
                    for sm in bfs:
                        bfs[sm][0].append(np.mean(d_b[sm][0])); bfs[sm][1].append(np.mean(d_b[sm][1]))
            struct[f"{sig}/{c}/sigma_hat_rel"] = boot_ci(sh)
            for sm in bfs:
                tag = "registered" if sm == 1.5 else "posthoc"
                struct[f"{sig}/{c}/block_frac_sigma>={sm}_clean_minus_ctrl_pp_{tag}"] = [100 * x for x in boot_ci(bfs[sm][0])]
                struct[f"{sig}/{c}/vert_block_frac_sigma>={sm}_clean_minus_ctrl_pp_{tag}"] = [100 * x for x in boot_ci(bfs[sm][1])]
        # absolute levels
        cl = [R[k] for k in R if k.split("|")[1] == "clean" and k.split("|")[2] == sig]
        if cl:
            struct[f"{sig}/clean/levels"] = {
                "n_atoms_median": float(np.median([d["n_atoms"] for d in cl])),
                "mp_n_median": float(np.median([d["mp_n"] for d in cl])),
                "morozov_ratio_median": float(np.median([d["morozov_ratio"] for d in cl])),
                "sigma_hat_over_sigma_median": float(np.median([d["sigma_hat"] / d["sigma"] for d in cl])),
                "bands_mean": np.mean([d["bands"] for d in cl], 0).tolist(),
                "block_frac_mean": float(np.mean([block_frac(d["atoms"])[0] for d in cl])),
                "block_frac_sigma>=3_mean": float(np.mean([block_frac(d["atoms"], 3.0)[0] for d in cl])),
                "block_frac_sigma>=6_mean": float(np.mean([block_frac(d["atoms"], 6.0)[0] for d in cl])),
                "invariant_error_max": float(max(d["invariant_error"] for d in cl)),
                "secs_median": float(np.median([d["secs"] for d in cl]))}
        for c in CONTROLS:
            cc = [R[k] for k in R if k.split("|")[1] == c and k.split("|")[2] == sig]
            if cc:
                struct[f"{sig}/{c}/levels"] = {"bands_mean": np.mean([d["bands"] for d in cc], 0).tolist(),
                                               "n_atoms_median": float(np.median([d["n_atoms"] for d in cc]))}
    out["structure"] = struct
    # --- localization
    loc = {}
    for sig in SIGNALS:
        for b in BROKEN:
            rows = {k: [] for k in ("res", "deaths_late", "deaths", "births", "err_oracle", "mp_res", "mp_err_oracle",
                                    "indent_rule", "res_minus_mp")}
            perr, perr_hit, agree_pe = [], [], []
            n_parse_fail = 0
            for fi in fns_done:
                info = meta[fi]["broken"].get(b)
                if info is None:
                    continue
                h, unit, pos = info["h"], info["unit"], info["pos"]
                k = unit_index(unit, pos, h)
                per = {kk: [] for kk in rows}
                pe = []
                for sd in SEEDS:
                    d = R.get(f"{fi}|{b}|{sig}|{sd}")
                    if d is None:
                        break
                    sc = {"res": d["res_row"], "deaths_late": d["deaths_row_late"], "deaths": d["deaths_row"],
                          "births": d["births_row"], "err_oracle": d["err_row"], "mp_res": d["mp_res_row"],
                          "mp_err_oracle": d["mp_err_row"], "indent_rule": indent_rule(info)}
                    for kk, s in sc.items():
                        per[kk].append(pct_rank(unit_scores(s, unit), k))
                    per["res_minus_mp"].append(per["res"][-1] - per["mp_res"][-1])
                    if info["parse_err"] is not None:
                        pe.append(pct_rank(d["res_row"], info["parse_err"]))
                else:
                    for kk in rows:
                        rows[kk].append(np.mean(per[kk]))
                    if info["parse_err"] is not None:
                        n_parse_fail += 1
                        perr.append(np.mean(pe))
            loc[f"{sig}/{b}"] = {kk: {"mean_ci": boot_ci(v), "wilcoxon_p_vs_0.5": wilcox(np.array(v) - 0.5)
                                      if kk != "res_minus_mp" else wilcox(v)} for kk, v in rows.items()}
            loc[f"{sig}/{b}"]["n_fn"] = len(rows["res"])
            loc[f"{sig}/{b}"]["n_parse_fail"] = n_parse_fail
            loc[f"{sig}/{b}"]["parse_err_row_pct_under_water_res"] = {"mean_ci": boot_ci(perr), "n": len(perr)}
    out["localization"] = loc
    json.dump(out, open(f"{OUT}/results{'' if prefix == 'res_w' else '_' + prefix}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def panel():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    meta = json.load(open(f"{OUT}/meta.json"))
    R = load_results()
    dumped = [k for k, d in R.items() if "recon" in d and k.split("|")[1] in ("clean", "wrong_indent", "del_line", "shuffle")]
    dumped = sorted(dumped, key=lambda k: (int(k.split("|")[0]), k))[:8]
    if not dumped:
        print("no dumped jobs")
        return
    fig, ax = plt.subplots(len(dumped), 3, figsize=(15, 2.6 * len(dumped)), squeeze=False)
    for r, k in enumerate(dumped):
        d = R[k]
        fi, var, sig, _ = k.split("|")
        h, w = d["h"], d["w"]
        noisy = np.array(d["noisy"]).reshape(h, w)
        recon = np.array(d["recon"]).reshape(h, w)
        res = np.array(d["residual"]).reshape(h, w)
        ax[r, 0].imshow(noisy, cmap="gray", aspect="auto", interpolation="nearest")
        ax[r, 0].set_title(f"fn {fi} {var} {sig}: noisy map (sigma {d['sigma']})", fontsize=8)
        ax[r, 1].imshow(recon, cmap="gray", aspect="auto", interpolation="nearest")
        for a in d["atoms"]:
            ax[r, 1].add_patch(Ellipse((a["cx"], a["cy"]), 2 * a["sigma"], 2 * a["sigma"], fill=False,
                                       lw=0.5, color="tab:red" if 2 * a["sigma"] >= 3 else "tab:cyan"))
        ax[r, 1].set_xlim(-0.5, w - 0.5); ax[r, 1].set_ylim(h - 0.5, -0.5)
        ax[r, 1].set_title(f"living atoms N={d['n_atoms']} (red: spans >=3 rows) over recon", fontsize=8)
        ax[r, 2].imshow(res, cmap="RdBu", aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
        ax[r, 2].set_title("final residual (row sums right); broken unit in yellow", fontsize=8)
        rr = np.array(d["res_row"])
        ax[r, 2].plot(w - 1 - (w / 4) * rr / max(rr.max(), 1e-9), np.arange(h), color="k", lw=0.8)
        info = meta[int(fi)]["broken"].get(var)
        if info:
            ys = [info["pos"]] if info["unit"] == "row" else [info["pos"] - 0.5]
            for y in ys:
                for c in range(3):
                    ax[r, c].axhline(y, color="yellow", lw=1.0, ls="--")
            if info["parse_err"] is not None:
                ax[r, 2].axhline(info["parse_err"], color="magenta", lw=0.8, ls=":")
    plt.tight_layout()
    plt.savefig(f"{OUT}/panel.png", dpi=110)
    print(f"{OUT}/panel.png")


def surprisal():
    """DEFERRED (GPU only; one GPU process at a time). Writes per-byte surprisal maps for the same functions
    from the project byte-LM /media/llm-data/universe-runs/bytebrain-pycode-20261022/clean/ckpt_best.pt and
    appends jobs '<fn>|<variant>|surp|<seed>' to jobs_surp.jsonl. Not run in this session."""
    import torch
    sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
    from src.model import ByteGPT
    assert torch.cuda.is_available(), "GPU required by project rule; do not run on CPU"
    ck = torch.load("/media/llm-data/universe-runs/bytebrain-pycode-20261022/clean/ckpt_best.pt",
                    map_location="cuda", weights_only=False)
    cfg = ck["config"]
    m = ByteGPT(dim=cfg["dim"], n_layers=cfg["layers"], n_heads=cfg["heads"], context=cfg["ctx"]).cuda().eval()
    m.load_state_dict(ck["model"])
    jobs = [json.loads(l) for l in open(f"{OUT}/jobs.jsonl")]
    meta = json.load(open(f"{OUT}/meta.json"))
    with open(f"{OUT}/jobs_surp.jsonl", "w") as f:
        for fi, mm in enumerate(meta):
            lines = mm["text"].split("\n")
            b = "\n".join(lines).encode()[: cfg["ctx"]]
            x = torch.tensor(list(b), device="cuda")[None]
            with torch.no_grad():
                lp = torch.log_softmax(m(x[:, :-1]).float(), -1)[0]
            s = np.concatenate([[0.0], (-lp.gather(1, x[0, 1:, None])[:, 0] / np.log(2)).cpu().numpy()])
            g = np.zeros((len(lines), mm["w_ink"]), np.float32)
            i = 0
            for r, ln in enumerate(lines):
                n = len(ln.encode())
                seg = s[i:i + min(n, mm["w_ink"])]
                g[r, :len(seg)] = seg / 8.0
                i += n + 1
            for sd in SEEDS:
                f.write(json.dumps({"id": f"{fi}|clean|surp|{sd}", "h": g.shape[0], "w": g.shape[1],
                                    "data": g.ravel().tolist(), "sigma": SIGMA, "seed": sd}) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "oracle_jobs", "analyze", "panel", "surprisal"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prefix", default="res_w")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.n, dump_fns={0, 1, 2, 3})
    elif a.cmd == "oracle_jobs":
        # sigma_hat confound arm: same jobs, run with --oracle-sigma (met = 40 * sigma_true^2).
        # stub: all 200 fns; ink: stratified fn % 4 == 0 (50 fns) to fit the wall budget.
        with open(f"{OUT}/jobs_oracle.jsonl", "w") as f:
            for l in open(f"{OUT}/jobs.jsonl"):
                fi, var, sig, _ = json.loads(l)["id"].split("|")
                if var in ("clean",) + CONTROLS and (sig == "stub" or int(fi) % 4 == 0):
                    f.write(l)
    elif a.cmd == "analyze":
        analyze(a.prefix)
    elif a.cmd == "panel":
        panel()
    else:
        surprisal()
