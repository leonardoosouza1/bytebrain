"""Realistic pick for FIM answers: choose by the calls the user showed (seed 2), confirm on hidden (seeds 0,1).

For every answer in a results file: patch, judge the target on seed-2 cases and every other function on
seed 2 (visible regressions). An answer is "picked-eligible" if the target passes seed 2 and nothing
regresses on seed 2. Reports precision (eligible -> hidden fixed) per arm, then applies one pick per
function (the first eligible answer in the best arm by eligible count) to the file and judges all 39.

Usage: select_visible.py RESULTS.json [--out REPORT.json]
"""

import argparse
import json
import os
from collections import Counter

import fim_repair as F
import localize as L
import run_patch_model as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = json.load(open(os.path.join(F.HERE, "prompts.json")))
    source = data["source"]
    prompts = {(p["name"], p["scheme"]): p for p in data["prompts"]}
    base_vis = L.judge_module(source, seeds=(2,))
    base_hid = L.judge_module(source)
    res = json.load(open(args.results))["results"]
    rows, best = [], {}
    for r in res:
        p = prompts[(r["name"], r["scheme"])]
        indent = F.split_block(p)[2]
        elig = []
        for i, a in enumerate(r["answers"]):
            ans = R.clean_answer(F.cut_to_region(a["answer"], indent))
            new, why, _ = R.patch(source, p, ans)
            if new is None:
                continue
            vis = L.judge_module(new, seeds=(2,))
            if vis[p["name"]] and not [k for k, v in vis.items() if base_vis[k] and not v]:
                hid_ok = bool(a.get("fixed") and not a.get("regressions"))
                elig.append({"i": i, "hidden_ok": hid_ok, "code": ans})
        tp = sum(e["hidden_ok"] for e in elig)
        row = {"name": r["name"], "scheme": r["scheme"], "hint": r["hint"], "n": len(r["answers"]),
               "eligible": len(elig), "eligible_and_hidden": tp, "hidden_fixed_any": r["n_fixed"]}
        rows.append(row)
        print(f"{r['name']:10s} {r['scheme']:12s} hint {r['hint']:d} eligible {len(elig):2d} "
              f"-> hidden ok {tp:2d}  (oracle {r['n_fixed']})", flush=True)
        cur = best.get(r["name"])
        if elig and (cur is None or len(elig) > cur["eligible"]):
            # most common eligible answer text = consensus pick
            text, _ = Counter(e["code"] for e in elig).most_common(1)[0]
            pick = next(e for e in elig if e["code"] == text)
            best[r["name"]] = {"eligible": len(elig), "prompt": p, "code": text, "hidden_ok": pick["hidden_ok"],
                               "arm": f"{r['scheme']}/hint{int(r['hint'])}"}
    elig_all = sum(r["eligible"] for r in rows)
    tp_all = sum(r["eligible_and_hidden"] for r in rows)
    print(f"precision eligible->hidden: {tp_all}/{elig_all}")
    # apply one pick per function to the whole file (symbol + statement addresses survive edits elsewhere)
    final = source
    for name, b in best.items():
        new, why, _ = R.patch(final, b["prompt"], b["code"])
        print(f"pick {name:10s} arm {b['arm']:20s} hidden_ok {b['hidden_ok']:d} applied {new is not None} {why or ''}")
        if new is not None:
            final = new
    end = L.judge_module(final)
    print(f"calculator hidden: before {sum(base_hid.values())}/{len(base_hid)}  after {sum(end.values())}/{len(end)}")
    print("still failing:", [k for k, v in end.items() if not v])
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"rows": rows, "precision": [tp_all, elig_all],
                       "picks": {k: {kk: vv for kk, vv in v.items() if kk != "prompt"} for k, v in best.items()},
                       "before": sum(base_hid.values()), "after": sum(end.values()),
                       "failing_after": [k for k, v in end.items() if not v], "final_source": final}, fh, indent=1)


if __name__ == "__main__":
    main()
