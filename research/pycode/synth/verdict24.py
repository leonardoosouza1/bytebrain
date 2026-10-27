"""Pre-registered verdict for guess_derive24 (ledger #24/#25/#26). Written by the independent review, 2026-10-27.
Reads the run outputs only; never re-runs anything. Predictions and thresholds = guess_derive24.py header
(registered 2026-10-27 10:00), not re-chosen here.

Inputs: RUN/s*/eval_{heldout,v33}_s{seed}*.json (any tag: _derive and _sample files are merged per seed / set /
length / problem id; an arm present in two files must agree) and RUN/s*/tune_s{seed}.json (dev choice Adam vs GD).

Which set feeds which length (header): len1 from `v33`, len2 / len3 from `heldout`. The other set is reported as
secondary, never judged.

Statistics. Unit = problem (the same problems for every seed). Per problem the arm difference is averaged over the
seeds (pooled seeds, paired by problem), then 10 000 bootstrap resamples over problems -> 95% percentile CI.
Decision for a prediction "A >= B + thr":
  CONFIRMED        point >= thr and CI low > 0
  NOT_SIGNIFICANT  point >= thr but CI low <= 0
  REFUTED          point <  thr
P5 (against the idea): REVERSED (the derive wins) when derive - sample > 5 pp and CI low > 0.

Gates (any failure -> no verdict, exit 2):
  G0 >= --min-seeds seeds with derive + sample arms on both sets, >= --min-n problems per length, no edge-of-grid lr
     run (args.allow_edge False), every arm row's executions <= budget, row ids identical across files.
  C1 positive control: o_sample_lm on v33 equals the recorded v3.3 rows in >= 99% of rows (pooled over seeds, lens).
  KILL: P1 (guesser canonical top-1 >= prior top-1 + 20 pp) fails at len2 AND len3 -> the guesser does not read the
     problem; the derive verdicts (P3/P4/P5) are not issued.

AMENDMENT (review fixes, see guess_derive24.py header block of the same name; registered before any GPU run):
  E  RUN/heldout_equiv.json (guess_derive24.py equiv) marks every held-out test problem that a TRAINED skeleton
     computes exactly (fitted constants, 1e-9 on fresh inputs). PRIMARY for heldout len2 / len3 = the problems
     WITHOUT a trained equivalent; the full held-out set is reported under secondary["all_heldout"]. Missing file or
     ids that do not match the eval rows -> no verdict. Review B2: a file without the len3 cross-length field
     (n_trained_len2_families, len3.only_cross_len2) or without calib_len2_exhaustive is stale -> no verdict.
  P1 / KILL use the FUNCTIONAL top-1 (guess_func_right / prior_func_right: the argmax skeleton with fitted constants
     equals the truth on fresh inputs), not the exact skeleton.
  P3b dev-chosen derive vs g_polish (guesser argmax + LM polish + Gumbel restarts, NO soft phase, same budget):
     CONFIRMED when diff >= 0 with CI low > 0 (the derivative over the PROGRAM adds to the derivative on constants).
  P4 compares the SAME optimizer: per seed the dev choice (Adam -> g_derive vs r_derive, GD -> g_derive_gd vs
     r_derive_gd); the as-written Adam pair is kept in secondary.

AMENDMENT C (budget units, lead decision, written 2026-10-27 12:36, time from `date`; before any GPU run):
  PRIMARY accounting = the real compute: one soft step (forward + backward of the relaxed program) is charged 3 NOP
  program runs; every arm gets the same total budget (6000) in these units (G0 checks `apps` <= budget). P1-P5 are
  judged on it. SECONDARY = the old gradconst charge 3 NOP S per soft step: every soft-phase arm is RE-RUN with it as
  arm + "_oldc" (same tuned lr / eps / k_stop, same 6000 budget in old units); secondary["old_charge..."] repeats
  P3 / P3b / P4 / P5 with the _oldc arms (reported, never judged). A post-hoc re-charge of the primary trajectory
  (apps_old) was rejected: smoke B2 (oracle len3, 20 problems) real run 4/20 vs post hoc 0/20.
  Eval rows without apps_old (pre-amendment files) or without the _oldc arms -> no verdict.

Usage: verdict24.py RUN_DIR [--min-seeds 3 --min-n 200 --boot 10000 --json OUT]
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

JUDGED = {1: "v33", 2: "heldout", 3: "heldout"}
CMP_KEYS = ("fit", "apps", "io_bwrap")


def load(run):
    """-> data[seed][set][S] = {id: {"arms": {...}, guess_canon_right, prior_canon_right}}, meta"""
    data, meta = {}, {"files": [], "edge_runs": [], "budgets": set(), "C1": [], "conflicts": 0}
    for f in sorted(glob.glob(os.path.join(run, "**", "eval_*_s*.json"), recursive=True)):
        m = re.search(r"eval_(heldout|v33|dev)_s(-?\d+)", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        st, seed = m.group(1), int(d.get("guesser_seed", m.group(2)))
        meta["files"].append(os.path.relpath(f, run))
        if d["args"].get("allow_edge"):
            meta["edge_runs"].append(f)
        meta["budgets"].add(d["args"]["budget"])
        for L, r in d["lens"].items():
            S = int(L[3:])
            tgt = data.setdefault(seed, {}).setdefault(st, {}).setdefault(S, {})
            if r.get("control_C1"):
                meta["C1"].append((seed, S, r["control_C1"]["identical"], r["control_C1"]["rows"]))
            for row in r["rows"]:
                t = tgt.setdefault(row["id"], {"arms": {}, "guess_canon_right": row["guess_canon_right"],
                                                "prior_canon_right": row["prior_canon_right"],
                                                "guess_func_right": row.get("guess_func_right"),
                                                "prior_func_right": row.get("prior_func_right")})
                for arm, v in row["arms"].items():
                    if arm in t["arms"] and any(t["arms"][arm].get(k) != v.get(k) for k in CMP_KEYS):
                        meta["conflicts"] += 1
                    t["arms"][arm] = v
    return data, meta


def dev_choice(run, seed):
    """P3: g_derive or g_derive_gd, whichever is better on DEV (best grid value of the tune file)"""
    fs = glob.glob(os.path.join(run, "**", f"tune_s{seed}.json"), recursive=True)
    if not fs:
        return None
    g = json.load(open(fs[0]))["grid"]
    best = {opt: max((v for k, v in g.items() if k.startswith(f"guesser|{opt}|")), default=-1) for opt in ("adam", "gd")}
    return "g_derive_gd" if best["gd"] > best["adam"] else "g_derive"


def boot(diff, B, rng):
    """diff [n_problems] (already seed-averaged) -> point, lo, hi"""
    diff = np.asarray(diff, float)
    idx = rng.integers(0, len(diff), (B, len(diff)))
    bs = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


SUBSET = {}          # (set, S) -> allowed problem ids (primary = no trained equivalent); filled in main


def load_equiv(run):
    f = os.path.join(run, "heldout_equiv.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    # review B2: a file without the cross-length candidates (trained len2 families for len3 targets) or without the
    # len2 calibration is stale -> "stale" (gate E refuses)
    if not d.get("n_trained_len2_families") or "only_cross_len2" not in d.get("len3", {}) \
            or "calib_len2_exhaustive" not in d:
        return "stale"
    return {S: {i: bool(v) for i, v in zip(d[f"len{S}"]["ids"], d[f"len{S}"]["has_trained_equiv"])} for S in (2, 3)}


def paired(data, seeds, st, S, fa, fb, B, rng, use_subset=True):
    """fa / fb (seed, row) -> 0/1 value; returns point, lo, hi, n, mean_a, mean_b. Held-out lens: only the
    no-trained-equivalent ids (SUBSET) unless use_subset=False."""
    ids = sorted(set.intersection(*[set(data[s][st][S]) for s in seeds]))
    if use_subset and (st, S) in SUBSET:
        ids = [i for i in ids if i in SUBSET[(st, S)]]
    da = np.array([[fa(s, data[s][st][S][i]) for s in seeds] for i in ids], float)
    db = np.array([[fb(s, data[s][st][S][i]) for s in seeds] for i in ids], float)
    p, lo, hi = boot((da - db).mean(1), B, rng)
    return {"diff": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(ids), "seeds": len(seeds),
            "a": round(float(da.mean()), 4), "b": round(float(db.mean()), 4)}


def decide(res, thr):
    if res["diff"] < thr:
        return "REFUTED"
    return "CONFIRMED" if res["ci95"][0] > 0 else "NOT_SIGNIFICANT"


def io(arm):
    return lambda s, r: float(r["arms"][arm]["io_bwrap"])


def io_of(arm_of):
    """per-seed arm choice: arm_of(seed) -> arm name"""
    return lambda s, r: float(r["arms"][arm_of(s)]["io_bwrap"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--json")
    a = ap.parse_args()
    rng = np.random.default_rng(24)
    data, meta = load(a.run)
    rep = {"files": meta["files"], "gates": {}, "predictions": {}, "secondary": {}, "refused": []}
    arms_all = ["g_only", "g_derive", "g_derive_gd", "g_polish", "r_derive", "r_derive_gd", "o_derive", "g_sample_lm",
                "g_sample_es", "prior_sample_lm", "o_sample_lm",
                "g_derive_oldc", "g_derive_gd_oldc", "r_derive_oldc", "r_derive_gd_oldc", "o_derive_oldc"]
    need = {"v33": arms_all, "heldout": arms_all}
    seeds, why = [], {}
    for s in sorted(data):
        miss = []
        for st in ("v33", "heldout"):
            for S in (1, 2, 3):
                rows = data[s].get(st, {}).get(S, {})
                if len(rows) < a.min_n:
                    miss.append(f"{st} len{S} n={len(rows)}")
                    continue
                arms_here = set.intersection(*[set(r["arms"]) for r in rows.values()])
                lack = [x for x in need[st] if x not in arms_here]
                if lack:
                    miss.append(f"{st} len{S} missing {lack}")
        if miss:
            why[s] = miss
        else:
            seeds.append(s)
    rep["gates"]["G0_seeds_complete"] = seeds
    rep["gates"]["G0_seeds_incomplete"] = why
    if len(seeds) < a.min_seeds:
        rep["refused"].append(f"G0: {len(seeds)} complete seeds < {a.min_seeds}")
    for st in ("v33", "heldout"):
        for S in (1, 2, 3):
            idsets = [set(data[s][st][S]) for s in seeds]
            if idsets and any(x != idsets[0] for x in idsets):
                rep["refused"].append(f"G0: problem ids differ across seeds ({st} len{S})")
    if meta["edge_runs"]:
        rep["refused"].append(f"G0: eval run with --allow-edge: {meta['edge_runs']}")
    if meta["conflicts"]:
        rep["refused"].append(f"G0: {meta['conflicts']} arm rows disagree between files of the same seed/set")
    budget = max(meta["budgets"]) if meta["budgets"] else 0
    if len(meta["budgets"]) > 1:
        rep["refused"].append(f"G0: different budgets across files {sorted(meta['budgets'])}")
    over = sum(1 for s in data for st in data[s] for S in data[s][st] for r in data[s][st][S].values()
               for v in r["arms"].values() if v.get("apps", 0) > budget)
    rep["gates"]["G0_rows_over_budget"] = over
    rep["gates"]["G0_charge"] = "PRIMARY: 3 NOP per soft step (AMENDMENT C)"
    no_old = sum(1 for s in data for st in data[s] for S in data[s][st] for r in data[s][st][S].values()
                 for k, v in r["arms"].items() if "fit" in v and "apps_old" not in v)
    if no_old:
        rep["refused"].append(f"G0: {no_old} arm rows without apps_old (eval files from before AMENDMENT C)")
    if over:
        rep["refused"].append(f"G0: {over} arm rows used more than the budget {budget}")
    c1 = [x for x in meta["C1"] if x[0] in seeds]
    c1_lens = {(x[0], x[1]) for x in c1}
    ident, tot = sum(x[2] for x in c1), sum(x[3] for x in c1)
    rep["gates"]["C1_positive_control"] = {"identical": ident, "rows": tot,
                                           "frac": round(ident / tot, 4) if tot else None,
                                           "seed_lens": sorted(c1_lens)}
    if not tot or ident / tot < 0.99 or any((s, S) not in c1_lens for s in seeds for S in (1, 2, 3)):
        rep["refused"].append("C1: positive control missing or < 99% identical rows (no verdict)")
    eq = load_equiv(a.run)
    if eq is None:
        rep["refused"].append("E: heldout_equiv.json missing (guess_derive24.py equiv) - no primary subset")
    elif eq == "stale":
        rep["refused"].append("E: heldout_equiv.json is stale (no trained-len2-family candidates for len3 or no "
                              "len2 calibration) - re-run guess_derive24.py equiv")
    else:
        for S in (2, 3):
            ev_ids = set.union(set(), *[set(data[s].get("heldout", {}).get(S, {})) for s in seeds]) if seeds else set()
            if not ev_ids <= set(eq[S]):
                rep["refused"].append(f"E: heldout len{S} eval ids not in heldout_equiv.json (stale split?)")
            SUBSET[("heldout", S)] = {i for i, v in eq[S].items() if not v}
            rep["gates"][f"E_len{S}"] = {"frac_with_trained_equiv": round(sum(eq[S].values()) / max(1, len(eq[S])), 4),
                                         "n_primary_subset": len(SUBSET[("heldout", S)] & ev_ids)}
    if any(r.get("guess_func_right") is None for s in seeds for st in data[s] for S in data[s][st]
           for r in data[s][st][S].values()):
        rep["refused"].append("P1: rows without guess_func_right (pre-fix eval files)")
    if rep["refused"]:
        return finish(rep, a, 2)

    B = a.boot
    P = rep["predictions"]
    # P1 + KILL
    p1 = {}
    for S in (1, 2, 3):
        st = JUDGED[S]
        r = paired(data, seeds, st, S, lambda s, x: float(x["guess_func_right"]),
                   lambda s, x: float(x["prior_func_right"]), B, rng)
        r["set"], r["decision"] = st, decide(r, 0.20)
        p1[f"len{S}"] = r
    P["P1_guesser_vs_prior_top1_+20pp"] = p1
    kill = p1["len2"]["decision"] != "CONFIRMED" and p1["len3"]["decision"] != "CONFIRMED"
    rep["gates"]["KILL_guesser_does_not_read"] = kill
    # P2 (does not depend on the derive)
    P["P2_g_sample_lm_vs_prior_sample_lm_+15pp"] = {
        f"len{S}": {**paired(data, seeds, JUDGED[S], S, io("g_sample_lm"), io("prior_sample_lm"), B, rng),
                    "set": JUDGED[S]} for S in (2, 3)}
    for v in P["P2_g_sample_lm_vs_prior_sample_lm_+15pp"].values():
        v["decision"] = decide(v, 0.15)
    choice = {s: dev_choice(a.run, s) for s in seeds}
    rep["gates"]["P3_dev_choice_per_seed"] = choice
    if any(v is None for v in choice.values()):
        rep["refused"].append("P3: a tune_s*.json is missing, dev choice Adam/GD unknown")
    if kill:
        rep["refused"].append("KILL: P1 fails at len2 AND len3 - derive verdicts (P3/P4/P5) not issued")
    if not rep["refused"]:
        dch = lambda s, x: float(x["arms"][choice[s]]["io_bwrap"])                               # noqa: E731
        P["P3_derive(dev-chosen)_vs_g_only_+20pp"] = {
            f"len{S}": {**paired(data, seeds, JUDGED[S], S, dch, io("g_only"), B, rng), "set": JUDGED[S]}
            for S in (1, 2, 3)}
        for v in P["P3_derive(dev-chosen)_vs_g_only_+20pp"].values():
            v["decision"] = decide(v, 0.20)
        P["P3b_derive(dev-chosen)_vs_g_polish_>0"] = {
            f"len{S}": {**paired(data, seeds, JUDGED[S], S, dch, io("g_polish"), B, rng), "set": JUDGED[S]}
            for S in (1, 2, 3)}
        for v in P["P3b_derive(dev-chosen)_vs_g_polish_>0"].values():
            v["decision"] = decide(v, 0.0)
        rch = {s: ("r_derive_gd" if choice[s] == "g_derive_gd" else "r_derive") for s in seeds}
        rdc = lambda s, x: float(x["arms"][rch[s]]["io_bwrap"])                                  # noqa: E731
        P["P4_derive_vs_r_derive_same_optimizer_+20pp"] = {
            f"len{S}": {**paired(data, seeds, JUDGED[S], S, dch, rdc, B, rng), "set": JUDGED[S]} for S in (2, 3)}
        for v in P["P4_derive_vs_r_derive_same_optimizer_+20pp"].values():
            v["decision"] = decide(v, 0.20)
        p5 = paired(data, seeds, "heldout", 3, io("g_sample_lm"), io("g_derive"), B, rng)
        p5["decision"] = decide(p5, 0.10)
        rev = {"diff": -p5["diff"], "ci95": [-p5["ci95"][1], -p5["ci95"][0]]}
        p5["derive_reverses_v2.2"] = rev["diff"] > 0.05 and rev["ci95"][0] > 0
        p5b = paired(data, seeds, "heldout", 3, io("g_sample_lm"), dch, B, rng)
        p5b["decision"] = decide(p5b, 0.10)
        P["P5_g_sample_lm_vs_g_derive_len3_+10pp"] = {"g_derive(adam, as written)": p5, "dev-chosen derive": p5b}
        rep["verdict_HIS_loop_P3_and_P4"] = (
            "CONFIRMED" if all(v["decision"] == "CONFIRMED" for k in ("P3_derive(dev-chosen)_vs_g_only_+20pp",
                                                                       "P4_derive_vs_r_derive_same_optimizer_+20pp")
                               for v in P[k].values()) else "NOT CONFIRMED")
        rep["verdict_soft_phase_P3b"] = ("CONFIRMED" if all(v["decision"] == "CONFIRMED" for v in
                                                            P["P3b_derive(dev-chosen)_vs_g_polish_>0"].values())
                                         else "NOT CONFIRMED")
    # reported, not judged
    sec = rep["secondary"]
    sec["P4_as_written_g_derive_vs_r_derive"] = {f"len{S}": paired(data, seeds, JUDGED[S], S, io("g_derive"),
                                                                   io("r_derive"), B, rng) for S in (2, 3)}
    allh = sec["all_heldout"] = {}                 # the full held-out set (incl. problems with a trained equivalent)
    for S in (2, 3):
        allh[f"len{S}"] = {
            "P1_func_top1": paired(data, seeds, "heldout", S, lambda s, x: float(x["guess_func_right"]),
                                   lambda s, x: float(x["prior_func_right"]), B, rng, use_subset=False),
            "P2": paired(data, seeds, "heldout", S, io("g_sample_lm"), io("prior_sample_lm"), B, rng, use_subset=False)}
        if "P3_derive(dev-chosen)_vs_g_only_+20pp" in P:
            allh[f"len{S}"]["P3"] = paired(data, seeds, "heldout", S, dch, io("g_only"), B, rng, use_subset=False)
            allh[f"len{S}"]["P4"] = paired(data, seeds, "heldout", S, dch, rdc, B, rng, use_subset=False)
    if "P3_derive(dev-chosen)_vs_g_only_+20pp" in P:          # AMENDMENT C: the old charge, reported only
        oc = sec["old_charge_3NOP_S_per_soft_step"] = {}
        dold = io_of(lambda s: choice[s] + "_oldc")
        for S in (1, 2, 3):
            st = JUDGED[S]
            d = {"P3": paired(data, seeds, st, S, dold, io("g_only"), B, rng),
                 "P3b": paired(data, seeds, st, S, dold, io("g_polish"), B, rng)}
            if S > 1:
                d["P4"] = paired(data, seeds, st, S, dold, io_of(lambda s: rch[s] + "_oldc"), B, rng)
            if S == 3:
                d["P5_g_sample_lm_vs_dev_chosen_derive"] = paired(data, seeds, st, S, io("g_sample_lm"), dold, B, rng)
            for k_, thr in (("P3", 0.20), ("P3b", 0.0), ("P4", 0.20), ("P5_g_sample_lm_vs_dev_chosen_derive", 0.10)):
                if k_ in d:
                    d[k_]["decision_if_judged"] = decide(d[k_], thr)
            oc[f"len{S}"] = d
    sec["P1_exact_skeleton_top1"] = {f"len{S}": paired(data, seeds, JUDGED[S], S,
                                                       lambda s, x: float(x["guess_canon_right"]),
                                                       lambda s, x: float(x["prior_canon_right"]), B, rng)
                                     for S in (1, 2, 3)}
    sec["P6_g_sample_lm_vs_g_sample_es_len3_+5pp"] = paired(data, seeds, "heldout", 3, io("g_sample_lm"),
                                                            io("g_sample_es"), B, rng)
    sec["P7_g_derive_gd_minus_g_derive"] = {f"len{S}": paired(data, seeds, JUDGED[S], S, io("g_derive_gd"),
                                                              io("g_derive"), B, rng) for S in (1, 2, 3)}
    sec["baselines_max_io"] = {f"len{S}": max(float(np.mean([x["arms"][b]["io_bwrap"] for s in seeds
                                                             for x in data[s][JUDGED[S]][S].values()]))
                                              for b in ("base_a", "base_b", "base_mean")) for S in (1, 2, 3)}
    sec["mean_io_by_arm"] = {f"{st}_len{S}": {arm: round(float(np.mean([x["arms"][arm]["io_bwrap"] for s in seeds
                                                                         for x in data[s][st][S].values()])), 4)
                                              for arm in need[st]} for st in ("v33", "heldout") for S in (1, 2, 3)}
    sec["derive_wins_soft_changed_argmax"] = {
        f"len{S}": {arm: int(sum(bool(x["arms"][arm].get("win_changed")) for s in seeds
                                 for x in data[s][JUDGED[S]][S].values())) for arm in ("g_derive", "g_derive_gd")}
        for S in (1, 2, 3)}
    # HIS stop rule: share of soft attempts ended by the rule (not the step cap); ~0 => the verdict is not about the rule
    sec["attempts_stopped_by_rule_frac"] = {
        f"len{S}": {arm: round(sum(x["arms"][arm].get("stopped_by_rule", 0) for s in seeds
                                   for x in data[s][JUDGED[S]][S].values())
                               / max(1, sum(x["arms"][arm].get("attempts", 0) for s in seeds
                                            for x in data[s][JUDGED[S]][S].values())), 4)
                    for arm in ("g_derive", "g_derive_gd", "r_derive", "r_derive_gd")} for S in (1, 2, 3)}
    return finish(rep, a, 2 if rep["refused"] else 0)


def finish(rep, a, code):
    txt = json.dumps(rep, indent=1, default=str)
    print(txt)
    if a.json:
        open(a.json, "w").write(txt)
    print("NO VERDICT: " + " | ".join(rep["refused"]) if rep["refused"] else "VERDICT ISSUED", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
