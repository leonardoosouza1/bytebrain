"""Aggregate every interp55 *.score.json (ledger #55 / I6): mean +- std over seeds per arm, next to the floors.

Usage: interp55_verdict.py --root /media/llm-data/universe-runs/bytebrain-pycode-20261022/interp55 [--out FILE]
Reads ROOT/floors/*.score.json and ROOT/runs/**/<arm>_s<seed>.score.json. greedy/sel are comparable across arms
(same K, same selector, same prompts); any@K is the oracle selector, diagnostic only.
"""

import argparse
import glob
import json
import os
import re
import statistics as st


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
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
