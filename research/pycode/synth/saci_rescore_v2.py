"""Re-score a finished saci_redo run (#46) under grader v2, keeping the v1 numbers (CPU, GPU hidden, bwrap only).

Reads {run}/redo.json + {run}/rows.jsonl. Requests without a row keep the original spec in every arm (as saci_redo).
Grades every (request, chosen spec) pair with grader v1 and v2 in the same process (E.GRADER_V2 toggled; the gold
cache is per grader). Positive control: v1 here must reproduce redo.json arms pass/fixes/breaks exactly, else exit 1.
Under v2 "fail" is the v2 grade of the original spec, so fixes/breaks are counted against v2's own baseline.
Extra v2 fields: pass on non-constant-gold requests, dispatch exact-set pass.
Needs e2e_pipeline.py patched with grader v2 (scratchpad apply_v2.py) — apply only after the redo full has finished.

usage: saci_rescore_v2.py RUN_DIR   (writes RUN_DIR/rescore_v2.json)
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_pipeline as E  # noqa: E402

RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
V = f"{RUN}/e2e/v2pool"
assert hasattr(E, "GRADER_V2"), "e2e_pipeline.py is not patched with grader v2"


def grade(pairs, reqs, P, W):
    """[(id, raw)] -> {(id, raw): trace row} through the eval's belt (same chunking as saci_redo)."""
    uniq = list(dict.fromkeys(pairs))
    res = {}
    for c in range(0, len(uniq), 400):
        chunk = uniq[c:c + 400]
        T = E.run_belt([reqs[i] for i, _ in chunk], [r for _, r in chunk], P, W)
        res.update(zip(chunk, T))
    return res


def main():
    run = sys.argv[1]
    redo = json.load(open(f"{run}/redo.json"))
    assert not redo["config"]["smoke"], "smoke run: scope is the fired subset, rescore the full run"
    raws = json.load(open(f"{V}/student_s5_s1_tok/raws.json"))
    tr = {t["id"]: t for t in map(json.loads, open(f"{V}/traces_student_s5.jsonl"))}
    reqs = {q["id"]: q for q in json.load(open(f"{V}/requests.json"))}
    ids = [i for i in tr if i in raws and raws[i].get("tok")]             # saci_redo Monitor.ids
    orig = {i: raws[i]["raw"] for i in ids}
    rows = {r["id"]: r for r in map(json.loads, open(f"{run}/rows.jsonl"))}
    arms = list(redo["arms"])
    chosen = {i: (rows[i]["chosen"] if i in rows else {a: orig[i] for a in arms}) for i in ids}
    assert all(set(c) == set(arms) for c in chosen.values())
    pairs = [(i, orig[i]) for i in ids] + [(i, r) for i in ids for r in chosen[i].values()]
    P, W = E.Planner("net", redo["config"]["ckpt"]), E.Writer("net", redo["config"]["ckpt"])
    out = {"run": run, "requests": len(ids)}
    for name, v2 in (("v1", False), ("v2", True)):
        E.GRADER_V2 = v2
        g = grade(pairs, reqs, P, W)
        ok = {k: t["end_pass"] for k, t in g.items()}
        fail = {i: not ok[(i, orig[i])] for i in ids}
        res = {"original_pass": sum(not fail[i] for i in ids), "arms": {}}
        for a in arms:
            res["arms"][a] = {"pass": sum(ok[(i, chosen[i][a])] for i in ids),
                              "fixes": sum(fail[i] and ok[(i, chosen[i][a])] for i in ids),
                              "breaks": sum(not fail[i] and not ok[(i, chosen[i][a])] for i in ids)}
            res["arms"][a]["net"] = res["arms"][a]["fixes"] - res["arms"][a]["breaks"]
            if v2:
                nc = [i for i in ids if not g[(i, chosen[i][a])].get("const_gold")]
                es = [g[(i, chosen[i][a])]["exact_set"] for i in ids if "exact_set" in g[(i, chosen[i][a])]]
                res["arms"][a]["pass_nonconst"] = [sum(ok[(i, chosen[i][a])] for i in nc), len(nc)]
                res["arms"][a]["pass_exact_set_dispatch"] = [sum(es), len(es)]
        if v2:
            res["original_v1_pass_v2_fail"] = sum(out["v1"]["by_id"][i] and fail[i] for i in ids)
            res["original_v1_fail_v2_pass"] = sum(not out["v1"]["by_id"][i] and not fail[i] for i in ids)
        else:
            res["by_id"] = {i: not fail[i] for i in ids}
        out[name] = res
    mism = {a: (out["v1"]["arms"][a], redo["arms"][a]) for a in arms
            if any(out["v1"]["arms"][a][k] != redo["arms"][a][k] for k in ("pass", "fixes", "breaks"))}
    out["control_v1_reproduces_redo"] = not mism and out["v1"]["original_pass"] == redo["scope"]["original_pass"]
    out["control_mismatch"] = mism
    del out["v1"]["by_id"]
    json.dump(out, open(f"{run}/rescore_v2.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "control_mismatch"}, indent=1), flush=True)
    if not out["control_v1_reproduces_redo"]:
        print("CONTROL FAIL: v1 rescore differs from redo.json", json.dumps(mism)[:2000], flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
