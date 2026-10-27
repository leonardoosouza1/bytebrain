"""Verdict for gradconst v3.3 (ledger #26), written 10-26 ~21:50 BEFORE any v3.3 test result exists (only the dev-set
tune was running). Applies the registered predictions R1-R4 and the verdict rule of gradconst.py ("PREDICTIONS v3.3"
+ the ~20:50 amendment) to the two test runs, one per mode:
  STALL_DIR/run_len{1,2,3}/gradconst.json     stall ON (LM gives up after 5 flat accepted steps or lam > 1e8, ES after
                                              5 windows with >= 1 accept each gaining < 0.1%)
  NOSTALL_DIR/run_len{1,2,3}/gradconst.json   --no-stall (no fitter gives a program up early: the clean test of the
                                              derivative DIRECTION)
io = the program fits the 8 examples and matches the hidden inputs within the relative tolerance (the run's io field).
Differences are paired by problem (same seeds 10_000 S + i in both modes and in v3.1); 95% CI = percentile bootstrap,
10_000 resamples of the 200 problems, numpy default_rng(0), the same resample indices for every difference of a length.

  R1 soft_derive io len3 in [.40, .55]; len1 / len2 within +-5 pp of v3.1 (.845 / .620). Judged in stall-ON mode
     (v3.1 ran with its fitters' give-up on, so that mode is the comparable one); the no-stall value is reported.
  R2 (stall ON) ES io len3 in .715 +- .05; derive - ES len3 point estimate in [+5, +15] pp.
  R3 (no-stall) derive io len3 <= its stall-ON io; derive - ES len3 point estimate in [0, +10] pp.
  R4 derive >= random + 30 pp at every length in both modes.
  Gate 1: derive and ES above the best constant-answer baseline (base_return_a / base_return_b / base_mean) at every
     length in both modes; a failing gate 1 makes the verdict 'NO VERDICT (gate 1)'.
  VERDICT on 'the derivative beats derivative-free search on constants' (derive - ES at len3):
     PASS            no-stall CI lower bound > 0 AND stall-ON CI lower bound > 0
     KILL            no-stall CI upper bound < +3 pp (amendment: KILL by the no-stall mode alone)
     else NOT ESTABLISHED, naming the mode(s) whose lower bound is > 0.
Reported, not judged: every length's derive - ES with CI in both modes, median executions on the problems both solve,
paired change vs v3.1 (gradconst31/run_len*) per arm, the caps each run used.

Amendment 10-26 ~21:55 (Opus review of this script, BEFORE any test result; tune on dev seeds only had run):
  A1 the +-5 pp tolerances of R1/R2 compare counts out of 200 (|k - 200 * ref| <= 10): float `abs(.57 - .62) <= .05` is
     False, so a value exactly on the registered bound was graded FAIL.
  A2 run settings asserted: n 200, budget 6000, noise 1.5, soft_steps 100, lens == the length, not --legacy / --v31,
     and the caps of every length equal the mode's tuned best in DIR/../tune_<mode>/tune.json with no grid-edge flag.
  A3 PASS and KILL both true is reported as a conflict (cannot happen at n 200: a lower bound > 0 pushes the upper bound
     to ~.04 > .03); the caps are printed next to the len3 CIs (no-stall: a wrong program costs exactly its cap, so
     unequal caps are part of what the gap measures).

usage: gradconst33_verdict.py OUT.json STALL_DIR NOSTALL_DIR [V31_DIR]
"""
import json
import os
import statistics
import sys

import numpy as np

LENS = (1, 2, 3)
V31_SOFT = {1: .845, 2: .620}
N = 200


def within(x, ref, pp=5):
    """A1: |x - ref| <= pp points, compared as counts out of N (x is a rate of N problems)."""
    return abs(round(x * N) - round(ref * N)) <= pp * N // 100


def check_args(args, S, mode, tune):
    """A2: the run is the registered one."""
    exp = {"n": N, "budget": 6000, "noise": 1.5, "soft_steps": 100, "lens": str(S), "legacy": False}
    bad = {k: args.get(k) for k, v in exp.items() if args.get(k) != v}
    assert not bad and not args.get("v31"), (mode, S, bad, args.get("v31"))
    cd, ce = args["per_prog_derive"] or args["per_prog"], args["per_prog_es"] or args["per_prog"]
    for arm, cap in (("sample_derive", cd), ("sample_es", ce)):
        t = tune[arm]
        assert cap == t["best"] and not t["best_at_top_edge"] and not t["best_at_bottom_edge"], (mode, S, arm, cap, t)


def load(d, S):
    x = json.load(open(os.path.join(d, f"run_len{S}", "gradconst.json")))
    L = x["lens"][f"len{S}"]
    assert L["n_done"] == len(L["rows"]) == x["args"]["n"], (d, S, L["n_done"], x["args"]["n"])
    return x["args"], L["rows"]


def io(rows, arm):
    return np.array([float(r[arm]["io"]) for r in rows])


def ci(diff, idx):
    bs = diff[idx].mean(1)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def both_median(rows, X, Y):
    b = [(r[X]["apps"], r[Y]["apps"]) for r in rows if r[X]["io"] and r[Y]["io"]]
    return {"n_both": len(b), X: statistics.median(a for a, _ in b) if b else None,
            Y: statistics.median(c for _, c in b) if b else None}


def main():
    out_path, dirs = sys.argv[1], {"stall": sys.argv[2], "nostall": sys.argv[3]}
    v31 = sys.argv[4] if len(sys.argv) > 4 else None
    res = {"modes": {}, "R": {}, "verdict": None}
    rows = {}
    tunes = {}
    for mode, d in dirs.items():
        res["modes"][mode] = {}
        tunes[mode] = json.load(open(os.path.join(os.path.dirname(os.path.abspath(d)), f"tune_{mode}", "tune.json")))
        for S in LENS:
            args, rs = load(d, S)
            assert args["stall"] == (mode == "stall"), (mode, S, args["stall"])
            check_args(args, S, mode, tunes[mode])
            rows[mode, S] = rs
    for S in LENS:                                            # same problems in both modes (paired)
        assert [r["truth"] for r in rows["stall", S]] == [r["truth"] for r in rows["nostall", S]], S
    rng = np.random.default_rng(0)
    idx = {S: rng.integers(0, len(rows["stall", S]), (10_000, len(rows["stall", S]))) for S in LENS}
    for mode in dirs:
        for S in LENS:
            rs = rows[mode, S]
            args, _ = load(dirs[mode], S)
            arms = [a for a in rs[0] if isinstance(rs[0][a], dict)]
            rate = {a: round(float(io(rs, a).mean()), 4) for a in arms}
            d_de = io(rs, "sample_derive") - io(rs, "sample_es")
            d_dr = io(rs, "sample_derive") - io(rs, "sample_random")
            res["modes"][mode][f"len{S}"] = {
                "n": len(rs), "caps": {"derive": args["per_prog_derive"] or args["per_prog"],
                                       "es": args["per_prog_es"] or args["per_prog"]},
                "io": rate, "derive_minus_es": round(float(d_de.mean()), 4), "ci_derive_minus_es": ci(d_de, idx[S]),
                "derive_minus_random": round(float(d_dr.mean()), 4), "ci_derive_minus_random": ci(d_dr, idx[S]),
                "both_solved_median_apps": both_median(rs, "sample_derive", "sample_es"),
                "best_baseline": max(rate[b] for b in ("base_return_a", "base_return_b", "base_mean"))}
    M = res["modes"]
    # paired stall-ON vs no-stall, derive io (R3 first clause, reported with CI)
    d3 = io(rows["nostall", 3], "sample_derive") - io(rows["stall", 3], "sample_derive")
    res["nostall_minus_stall_derive_len3"] = {"diff": round(float(d3.mean()), 4), "ci": ci(d3, idx[3])}
    R = res["R"]
    s3, n3 = M["stall"]["len3"], M["nostall"]["len3"]
    sd = {S: M["stall"][f"len{S}"]["io"]["soft_derive"] for S in LENS}
    R["R1"] = {"pass": .40 <= sd[3] <= .55 and all(within(sd[S], V31_SOFT[S]) for S in (1, 2)),
               "soft_derive_stall": sd, "soft_derive_nostall": {S: M["nostall"][f"len{S}"]["io"]["soft_derive"]
                                                                for S in LENS}}
    R["R2"] = {"pass": within(s3["io"]["sample_es"], .715) and .05 <= s3["derive_minus_es"] <= .15,
               "es_io_len3": s3["io"]["sample_es"], "derive_minus_es_len3": s3["derive_minus_es"]}
    R["R3"] = {"pass": n3["io"]["sample_derive"] <= s3["io"]["sample_derive"] and 0 <= n3["derive_minus_es"] <= .10,
               "derive_io_len3": {"stall": s3["io"]["sample_derive"], "nostall": n3["io"]["sample_derive"]},
               "derive_minus_es_len3_nostall": n3["derive_minus_es"]}
    R["R4"] = {"pass": all(M[m][f"len{S}"]["derive_minus_random"] >= .30 for m in dirs for S in LENS),
               "detail": {m: {S: M[m][f"len{S}"]["derive_minus_random"] for S in LENS} for m in dirs}}
    g1 = {m: {S: M[m][f"len{S}"]["io"]["sample_derive"] > M[m][f"len{S}"]["best_baseline"]
              and M[m][f"len{S}"]["io"]["sample_es"] > M[m][f"len{S}"]["best_baseline"] for S in LENS} for m in dirs}
    res["gate1"] = {"pass": all(all(v.values()) for v in g1.values()), "detail": g1}
    lo = {m: M[m]["len3"]["ci_derive_minus_es"][0] for m in dirs}
    hi = {m: M[m]["len3"]["ci_derive_minus_es"][1] for m in dirs}
    if lo["nostall"] > 0 and lo["stall"] > 0 and hi["nostall"] < .03:      # A3
        v = "CONFLICT (PASS and KILL both hold)"
    elif not res["gate1"]["pass"]:
        v = "NO VERDICT (gate 1)"
    elif lo["nostall"] > 0 and lo["stall"] > 0:
        v = "PASS"
    elif hi["nostall"] < .03:
        v = "KILL"
    else:
        carry = [m for m in dirs if lo[m] > 0]
        v = "NOT ESTABLISHED (gap carried by: " + (", ".join(carry) if carry else "neither mode") + ")"
    res["verdict"] = v
    if v31:                                                   # paired vs v3.1, stall-ON (the comparable mode)
        res["vs_v31_stall"] = {}
        for S in LENS:
            _, old = load(v31, S)
            new = rows["stall", S]
            k = min(len(old), len(new))
            assert [r["truth"] for r in old[:k]] == [r["truth"] for r in new[:k]], S
            res["vs_v31_stall"][f"len{S}"] = {a: {"v31": round(float(io(old[:k], a).mean()), 4),
                                                  "v33": round(float(io(new[:k], a).mean()), 4)}
                                              for a in ("sample_derive", "sample_es", "soft_derive")}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({"verdict": v, "R": {k: R[k]["pass"] for k in R}, "gate1": res["gate1"]["pass"],
                      "len3_ci": {m: M[m]["len3"]["ci_derive_minus_es"] for m in dirs},
                      "caps": {m: M[m]["len3"]["caps"] for m in dirs}}, indent=1))


if __name__ == "__main__":
    main()
