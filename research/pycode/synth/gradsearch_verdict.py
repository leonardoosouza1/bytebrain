"""Verdict for gradsearch v2.2 (ledger #26), written 10-26 BEFORE the v2.2 test results exist.

Reads one or more gradsearch.json files (one per length when lengths run in parallel) and applies the registered
predictions Q1-Q5 of gradsearch.py (section "PREDICTION v2.2"). A problem counts as SOLVED by an arm when the arm's
program fits the 8 examples AND equals the true program on the 32 hidden inputs not among the examples (unseen32 io).
Every arm returns its FIRST fit, so success at budget b is `solved and apps <= b` (success-vs-budget curve, review 3).

Paired tests: two-sided exact sign test on the discordant pairs (b = only arm X solved, c = only arm Y solved),
pooled over all lengths. Q5 runs 4 tests (2 stop arms x {all wins, Adam-only wins}) -> Bonferroni alpha .05 / 4.
Adam-only: solved and the fit came after at least one Adam step (a restart's start is a free-looking sample near the
guess; review 1). Median apps are reported on the problems both arms solve.

Amendment 10-26 ~20:50 (second Opus review of v2.2, BEFORE any test result exists; nothing in the runs changed):
  A1 Q5 rule verdict is coded: it is about guess_derive_stop only (derive_only_stop is reported, not part of it).
     PASS = it wins all-wins AND Adam-only (p < .0125 each); HARM = it loses all-wins with p < .0125, or loses the
     STRICT Adam-only test with p < .0125. Strict Adam-only counts a discordant pair only when the losing arm did not
     solve the problem AT ALL (a stopped restart that the stop arm then solves at a later restart's start is not a
     loss: once a start fits, that restart's Adam never runs, so plain Adam-only is biased against the stop arm).
     An all-wins PASS without the Adam-only PASS = 'NOT ESTABLISHED (win carried by restarts, not the rule)'.
  A2 B uses len(synth2.UNITS) = 47 (the file said 46: every curve point was 2.1% short); the verdict's curves must
     equal the run's own <arm>_solved_at_B/d fields (asserted).
  A3 every arm is capped at B everywhere (solved = fit, unseen io, apps <= B), so rate, pairs and curve agree (the
     no-stop derive arms can exceed B by <= 1/47 through check charges). The curve conditions are part of the
     verdicts: Q3 PASS also needs guess_sample's curve >= guess_derive's at every b and length; Q4 PASS needs rand >
     derive_only at B at every length AND at every curve point, where a 0-0 tie is uninformative and skipped.

usage: gradsearch_verdict.py OUT.json RUN1/gradsearch.json [RUN2/gradsearch.json ...]
"""
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth2  # noqa: E402

DIVS = (64, 16, 4, 1)
U = len(synth2.UNITS)                  # 47 (A2)
CAP = {}                               # id(row) -> B of its length (A3)


def sign_test(b, c):
    """two-sided exact binomial p for b vs c discordant pairs (p = 1 when there are none)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def solved(x, B, adam_only=False):
    ok = bool(x["fit"] and x["unseen_io"] and x["apps"] <= B)
    return ok and x["adam_win"] if adam_only else ok


def pair(rows, X, Y, adam_only=False, strict=False):
    """strict (A1): a discordant pair counts only when the other arm did not solve the problem at all."""
    sx = [solved(r["arms"][X], CAP[id(r)], adam_only) for r in rows]
    sy = [solved(r["arms"][Y], CAP[id(r)], adam_only) for r in rows]
    ax = [solved(r["arms"][X], CAP[id(r)]) for r in rows]
    ay = [solved(r["arms"][Y], CAP[id(r)]) for r in rows]
    b = sum(p and not (a if strict else q) for p, q, a in zip(sx, sy, ay))
    c = sum(q and not (a if strict else p) for p, q, a in zip(sx, sy, ax))
    both = [(r["arms"][X]["apps"], r["arms"][Y]["apps"]) for r, p, q in zip(rows, sx, sy) if p and q]
    return {"X": X, "Y": Y, "adam_only": adam_only, "strict": strict, "n": len(rows),
            "X_rate": round(sum(sx) / len(rows), 4),
            "Y_rate": round(sum(sy) / len(rows), 4), "diff_pp": round(100 * (sum(sx) - sum(sy)) / len(rows), 1),
            "b_only_X": b, "c_only_Y": c, "p_sign": round(sign_test(b, c), 5), "n_both": len(both),
            "median_apps_both_X": statistics.median(a for a, _ in both) if both else None,
            "median_apps_both_Y": statistics.median(a for _, a in both) if both else None}


def curve(rows, arm, B):
    return {f"B/{d}": round(sum(solved(r["arms"][arm], B / d) for r in rows) / len(rows), 4) for d in DIVS}


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    ev, cfg = {}, None
    for p in paths:
        d = json.load(open(p))
        cfg = cfg or d["config"]
        assert not d["config"]["v2"], "v2 control run, not a v2.2 test run"
        for k, v in d["eval"].items():
            assert k not in ev, f"length {k} twice"
            ev[k] = v
    lens = sorted(ev, key=lambda k: int(k[3:]))
    res = {"lens": lens, "config": cfg, "per_len": {}, "pooled": {}, "verdicts": {}}
    pooled_rows = []
    for k in lens:
        rows, n_slots = ev[k]["rows"], int(k[3:])
        B = cfg["restarts"] * cfg["steps"] * n_slots * U
        for r in rows:
            CAP[id(r)] = B
        arms = list(rows[0]["arms"])
        pl = {"n": len(rows), "B": B,
              "rate": {a: round(sum(solved(r["arms"][a], B) for r in rows) / len(rows), 4) for a in arms},
              "unseen32_io": {a: round(sum(r["arms"][a]["unseen_io"] for r in rows) / len(rows), 4) for a in arms},
              "const_unseen32_io": round(sum(r["baseline_const"]["unseen_io"] for r in rows) / len(rows), 4),
              "curve": {a: curve(rows, a, B) for a in arms},
              "over_B": {a: sum(r["arms"][a]["apps"] > B for r in rows) for a in arms},
              "start_wins_late": {a: sum(r["arms"][a].get("start_win_late", False) for r in rows) for a in arms}}
        summ = ev[k]                                                    # the run writes <arm>_solved_at_B/d here
        for a in arms:                                                   # A2: the run's own curve fields
            for d in DIVS:
                f = summ.get(f"{a}_solved_at_B/{d}")
                assert f is None or abs(f - pl["curve"][a][f"B/{d}"]) < 1e-3, (k, a, d, f, pl["curve"][a])
        pl["Q3_pair"] = pair(rows, "guess_sample", "guess_derive")
        pl["Q4_pair"] = pair(rows, "rand", "derive_only")
        res["per_len"][k] = pl
        pooled_rows += rows
    P = res["pooled"]
    P["Q3"] = pair(pooled_rows, "guess_sample", "guess_derive")
    P["Q4"] = pair(pooled_rows, "rand", "derive_only")
    for X, Y in (("guess_derive_stop", "guess_derive"), ("derive_only_stop", "derive_only")):
        for ao in (False, True):
            P[f"Q5_{X}_{'adam' if ao else 'all'}"] = pair(pooled_rows, X, Y, ao)
        P[f"Q5_{X}_adam_strict"] = pair(pooled_rows, X, Y, True, True)
    V = res["verdicts"]
    # Q1 gate 1: every non-baseline arm above max(identity, const) at every length (solved rate at its own max budget)
    q1 = {}
    for k in lens:
        pl = res["per_len"][k]
        base = max(pl["rate"]["identity"], pl["const_unseen32_io"])
        q1[k] = {a: pl["rate"][a] > base for a in pl["rate"] if a != "identity"}
    V["Q1_gate1"] = {"pass": all(all(v.values()) for v in q1.values()), "detail": q1}
    # Q2 guess_derive unseen32 io in the registered ranges and >= guess_only + 25 pp
    rng = {"len2": (.80, .95), "len3": (.50, .75), "len4": (.55, .80)}
    q2 = {}
    for k in lens:
        g, go = res["per_len"][k]["unseen32_io"]["guess_derive"], res["per_len"][k]["unseen32_io"]["guess_only"]
        lo, hi = rng.get(k, (0, 1))
        q2[k] = {"guess_derive": g, "in_range": lo <= g <= hi, "minus_guess_only_pp": round(100 * (g - go), 1),
                 "gap_ok": g - go >= .25}
    V["Q2"] = {"pass": all(v["in_range"] and v["gap_ok"] for v in q2.values()), "detail": q2}
    # Q3 guess_sample >= guess_derive: pooled sign test favours sampling (p < .05); reversed with p < .05 = derive wins
    q3 = P["Q3"]
    cge = {k: all(res["per_len"][k]["curve"]["guess_sample"][b] >= res["per_len"][k]["curve"]["guess_derive"][b]
                  for b in res["per_len"][k]["curve"]["guess_sample"]) for k in lens}
    win3 = q3["b_only_X"] > q3["c_only_Y"] and q3["p_sign"] < .05
    V["Q3"] = {"verdict": "PASS (sampling wins)" if win3 and all(cge.values()) else
               "NOT ESTABLISHED (sign test wins, curve fails somewhere)" if win3 else
               "REVERSED (derive wins)" if q3["c_only_Y"] > q3["b_only_X"] and q3["p_sign"] < .05 else "NOT ESTABLISHED",
               "per_len_diff_pp": {k: res["per_len"][k]["Q3_pair"]["diff_pp"] for k in lens},
               "curve_sampling_ge_derive_every_b": cge}
    # Q4 derive_only < rand at every length (rate at B) and at every b of the curve (A3: a 0-0 tie is skipped)
    cgt = {k: all(c["rand"][b] > c["derive_only"][b] for b in c["rand"] if c["rand"][b] or c["derive_only"][b])
           for k, c in ((k, res["per_len"][k]["curve"]) for k in lens)}
    rate4 = all(res["per_len"][k]["rate"]["derive_only"] < res["per_len"][k]["rate"]["rand"] for k in lens)
    V["Q4"] = {"pass": rate4 and all(cgt.values()), "rate_at_B_pass": rate4, "curve_rand_gt_every_b": cgt,
               "pooled_p": P["Q4"]["p_sign"]}
    # Q5 his stop rule, Bonferroni over 4 tests
    a5 = .05 / 4
    q5 = {}
    for key in [k for k in P if k.startswith("Q5_")]:
        t = P[key]
        q5[key] = ("PASS" if t["b_only_X"] > t["c_only_Y"] and t["p_sign"] < a5 else
                   "HARM" if t["c_only_Y"] > t["b_only_X"] and t["p_sign"] < a5 else "NOT ESTABLISHED")
    al, ad = q5["Q5_guess_derive_stop_all"], q5["Q5_guess_derive_stop_adam"]
    st = q5["Q5_guess_derive_stop_adam_strict"]
    rule = ("PASS" if al == "PASS" and ad == "PASS" else
            "HARM" if al == "HARM" or st == "HARM" else
            "NOT ESTABLISHED (win carried by restarts, not the rule)" if al == "PASS" else "NOT ESTABLISHED")
    V["Q5"] = {"alpha_bonferroni": a5, "tests": q5, "rule_verdict": rule,
               "note": "rule = guess_derive_stop only; *_adam_strict decides HARM only (A1); derive_only_stop reported"}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(V, indent=1))
    for k in lens:
        print(k, "rate", res["per_len"][k]["rate"])


if __name__ == "__main__":
    main()
