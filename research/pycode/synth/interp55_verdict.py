"""Aggregate every interp55 *.score.json (ledger #55 / I6): mean +- std over seeds per arm, next to the floors.

Usage: interp55_verdict.py --root /media/llm-data/universe-runs/bytebrain-pycode-20261022/interp55 [--out FILE]
Reads ROOT/floors/*.score.json and ROOT/runs/**/<arm>_s<seed>.score.json. greedy/sel are comparable across arms
(same K, same selector, same prompts); any@K is the oracle selector, diagnostic only.
Then tests the pre-registered predictions P1-P7 (interp55_writer.py header, 10-27 13:06) per seed, paired by seed and
by unit row (same prompts in every arm), with a paired bootstrap CI over unit rows. A prediction whose arms do not all
have 3 seeds gets NO VERDICT (gate 8). pw_gold is K=1: only its greedy is used. Selector = 4 accept cases, shared by
every arm; "accepted" is reported next to sel because the selector is weak (nn accepted 0.318 vs graded 0.034).
"""

import argparse
import glob
import json
import os
import re
import random
import statistics as st

SEEDS = (0, 1, 2)
TRAINED = ("flat", "pw", "q05lora", "q15lora")


def row(res):
    u, c, s = res["units"], res.get("test_comp", {}), res.get("test_seen", {})
    bu = u["all"].get("by_unit", u.get("by_unit", {}))
    tk = bu.get("tokenize", {})
    n_rest = u["all"]["n"] - tk.get("n", 0)
    rest = (u["all"]["sel"] * u["all"]["n"] - tk.get("sel", 0) * tk.get("n", 0)) / max(1, n_rest)
    return {"unit_greedy": u["all"]["greedy"], "unit_sel": u["all"]["sel"], "unit_sel_nonleaked": u["nonleaked"]["sel"],
            "unit_any": u["all"]["any"], "unit_sel_tokenize": tk.get("sel"), "unit_sel_not_tokenize": round(rest, 4),
            "comp_greedy": c.get("greedy"), "comp_sel": c.get("sel"), "comp_sel_with_novel": c.get("sel_with_novel"),
            "seen_sel": s.get("sel")}


def unit_vals(res, key, nonleaked=False, unit=None, not_unit=None):
    out = {}
    for r in res["units"]["rows"]:
        if (nonleaked and r["leaked"]) or (unit and r["unit"] != unit) or (not_unit and r["unit"] == not_unit):
            continue
        out[(r["variant"], r["unit"])] = float(r[key])
    return out


def comp_vals(res, key):
    return {r["variant"]: float(r[key]) for r in res.get("test_comp", {}).get("rows", [])}


def by_seed(root):
    d = {}
    for f in glob.glob(os.path.join(root, "runs", "**", "*.score.json"), recursive=True):
        m = re.match(r"(.+)_s(\d+)\.score\.json$", os.path.basename(f))
        if m:
            d.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))
    return d


def mean_over_seeds(R, arm, fn):
    v = [st.mean(fn(R[arm][s]).values()) for s in SEEDS]
    return round(st.mean(v), 4), [round(x, 4) for x in v]


def paired(R, a, b, fn_a, fn_b=None, n_boot=2000):
    """mean over seeds of (a - b) on the common unit rows; per-seed diffs; bootstrap CI resampling rows (same draw for
    every seed and both arms)."""
    fn_b = fn_b or fn_a
    va = [fn_a(R[a][s]) for s in SEEDS]
    vb = [fn_b(R[b][s]) for s in SEEDS]
    keys = sorted(set.intersection(*[set(x) for x in va + vb]))
    diff = [st.mean(va[i][k] - vb[i][k] for k in keys) for i in range(len(SEEDS))]
    rng, boots = random.Random(55), []
    for _ in range(n_boot):
        ks = [keys[rng.randrange(len(keys))] for _ in keys]
        boots.append(st.mean(st.mean(va[i][k] - vb[i][k] for k in ks) for i in range(len(SEEDS))))
    boots.sort()
    return {"diff": round(st.mean(diff), 4), "per_seed": [round(x, 4) for x in diff], "n_rows": len(keys),
            "ci95": [round(boots[int(.025 * n_boot)], 4), round(boots[int(.975 * n_boot) - 1], 4)]}


def predictions(R):
    have = lambda *arms: all(set(SEEDS) <= set(R.get(x, {})) for x in arms)
    nl = lambda key: (lambda r: unit_vals(r, key, nonleaked=True))
    P = {}

    def put(name, arms, fn):
        if not have(*arms):
            P[name] = {"verdict": "NO VERDICT", "why": "missing seeds: " + ", ".join(
                f"{x} {sorted(R.get(x, {}))}" for x in arms if not set(SEEDS) <= set(R.get(x, {})))}
        else:
            P[name] = fn()

    def p1():
        u, us = mean_over_seeds(R, "flat", nl("sel"))
        c, cs = mean_over_seeds(R, "flat", lambda r: comp_vals(r, "sel"))
        se = st.mean(R["flat"][s]["test_seen"]["sel"] for s in SEEDS)
        ok = .03 <= u <= .15 and .08 <= c <= .20 and se >= .90
        return {"unit_nonleaked_sel": u, "per_seed": us, "comp_sel": c, "comp_per_seed": cs, "seen_sel": round(se, 4),
                "verdict": "PASS" if ok else "FAIL"}
    put("P1 flat in range", ["flat"], p1)

    def p2():
        d = {k: paired(R, "pw", "flat", nl(k)) for k in ("sel", "greedy", "accepted")}
        lo, hi = d["sel"]["ci95"]
        v = "PASS (tie)" if -.05 <= lo and hi <= .05 else ("PW WINS" if lo > .05 else (
            "FLAT WINS" if hi < -.05 else "INCONCLUSIVE"))
        d["verdict"] = v
        d["note"] = "pw has +11% params (7.92 M vs 7.12 M) and trains 2 nets x 4000 steps: a narrow pw win is not clean"
        return d
    put("P2 |pw-flat|<=0.05 nonleaked", ["flat", "pw"], p2)

    def p3():
        d = paired(R, "pw_gold", "pw", lambda r: unit_vals(r, "greedy"))
        d["verdict"] = "PASS" if d["ci95"][0] >= .05 else ("FAIL" if d["ci95"][1] < .05 else "INCONCLUSIVE")
        return d
    put("P3 pw_gold-pw greedy>=0.05", ["pw", "pw_gold"], p3)

    def p4():
        u, us = mean_over_seeds(R, "q05lora", nl("sel"))
        d = paired(R, "q05lora", "flat", nl("sel"))
        out = {"q05_unit_nonleaked_sel": u, "per_seed": us, "q05_minus_flat": d}
        ok = .15 <= u <= .40 and d["ci95"][0] >= .10
        if have("q15lora"):
            out["q15_minus_q05"] = paired(R, "q15lora", "q05lora", nl("sel"))
            ok = ok and out["q15_minus_q05"]["diff"] >= 0
        else:
            out["q15"] = "missing seeds -> 1.5B part NOT MEASURED"
        out["verdict"] = "PASS" if ok else "FAIL"
        return out
    put("P4 0.5B LoRA", ["q05lora", "flat"], p4)

    def p5():
        u, us = mean_over_seeds(R, "q30fewshot", nl("sel"))
        c, cs = mean_over_seeds(R, "q30fewshot", lambda r: comp_vals(r, "sel"))
        ok = .20 <= u <= .45 and .15 <= c <= .40
        return {"unit_nonleaked_sel": u, "per_seed": us, "comp_sel": c, "comp_per_seed": cs,
                "verdict": "PASS" if ok else "FAIL"}
    put("P5 3B 2-shot", ["q30fewshot"], p5)

    for arm in [x for x in ("flat", "pw", "pw_gold", "q05lora", "q15lora", "q30fewshot") if x in R]:
        key = "greedy" if arm == "pw_gold" else "sel"

        def p6(arm=arm, key=key):
            t, _ = mean_over_seeds(R, arm, lambda r: unit_vals(r, key, unit="tokenize"))
            o, _ = mean_over_seeds(R, arm, lambda r: unit_vals(r, key, not_unit="tokenize"))
            return {"tokenize": t, "rest": o, "verdict": "PASS" if t < o else "FAIL"}
        put(f"P6 tokenize<rest {arm}", [arm], p6)
    for arm in [x for x in TRAINED if x in R]:
        def p7(arm=arm):
            u, _ = mean_over_seeds(R, arm, nl("sel"))
            c, _ = mean_over_seeds(R, arm, lambda r: comp_vals(r, "sel"))
            return {"unit_nonleaked_sel": u, "comp_sel": c,
                    "verdict": "PASS" if u > 0 and c > .081 else "FAIL"}
        put(f"P7 beats nn {arm}", [arm], p7)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = {}
    for f in sorted(glob.glob(os.path.join(a.root, "floors", "*.score.json"))):
        arms.setdefault("floor_" + os.path.basename(f).split(".")[0], []).append(row(json.load(open(f))))
    for f in sorted(glob.glob(os.path.join(a.root, "runs", "**", "*.score.json"), recursive=True)):
        m = re.match(r"(.+)_s(\d+)\.score\.json$", os.path.basename(f))
        if m:
            arms.setdefault(m.group(1), []).append(row(json.load(open(f))))
    out = {}
    keys = ["unit_greedy", "unit_sel", "unit_sel_nonleaked", "unit_sel_tokenize", "unit_sel_not_tokenize",
            "comp_greedy", "comp_sel", "comp_sel_with_novel", "seen_sel", "unit_any"]
    print(f"{'arm':16s} n " + " ".join(f"{k[:18]:>18s}" for k in keys))
    for arm, rs in arms.items():
        d = {"n_seeds": len(rs)}
        cells = []
        for k in keys:
            v = [r[k] for r in rs if r[k] is not None]
            if not v:
                d[k] = None
                cells.append(f"{'-':>18s}")
                continue
            mu, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            d[k] = {"mean": round(mu, 4), "std": round(sd, 4), "vals": v}
            cells.append(f"{mu:11.3f}+-{sd:5.3f}")
        out[arm] = d
        print(f"{arm:16s} {len(rs)} " + " ".join(cells))
    P = predictions(by_seed(a.root))
    for k, v in P.items():
        print(f"{k:34s} {v['verdict']:14s} " + json.dumps({x: y for x, y in v.items() if x != 'verdict'})[:220])
    out["predictions"] = P
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
