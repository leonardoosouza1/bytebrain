"""Labels for the #37 retrain (V2) without gold: the belt and the judges label a pool of requests.

His words (10-26): "o Sonnet ajusta e passa para o modelo retreinar aquilo e ajustar".
Pool = requests the student was never tested on (Sonnet rewrites of its own training texts, para_train).
  - trace passed  -> label = the canonical spec the belt executed (bridge of the student's output), only when it
                     is in the request's world (a 1-op calculator answered as a formula passes the belt but would
                     teach the wrong world);
  - trace failed  -> label = the understander fix the final judge proposed and verify_fix accepted (replay file);
  - otherwise     -> dropped.
Which traces count as failed comes from the request's test cases (oracle triage, declared). Gold is read only to
report how many labels are right; it never becomes a label.
Output: JSON [{text, gold, source}] for `student_understander.py --extra --tag-suffix v2`.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_pipeline as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--requests", required=True, help="pool requests.json (gold for the report only)")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    gold = {q["id"]: q["gold"] for q in json.load(open(a.requests))}
    fix = {}   # trace id -> (input text, target, source); the final judge's fix wins over the Qwen judge's
    for line in open(a.replay):
        r = json.loads(line)
        if r["stage"] == "understander" and (r["trace"] not in fix or r.get("source") == "final"):
            fix[r["trace"]] = (r["input"], r["target"] if isinstance(r["target"], dict) else json.loads(r["target"]),
                               r.get("source", "?"))
    out = []
    n = {"pass_self": 0, "pass_wrong_world": 0, "fail_fixed": 0, "fail_dropped": 0, "fix_text_mismatch": 0,
         "injected_skipped": 0, "label_equals_gold": 0}
    for line in open(a.traces):
        t = json.loads(line)
        if t.get("inject") is not None:
            n["injected_skipped"] += 1
            continue
        if t["end_pass"] and t.get("bridged"):
            if t["bridged"][0] != t["world"]:
                n["pass_wrong_world"] += 1
                continue
            lab, src = E.spec_of(tuple(t["bridged"])), "self"
            n["pass_self"] += 1
        elif t["id"] in fix and fix[t["id"]][0] == t["text"]:
            _, lab, src = fix[t["id"]]
            n["fail_fixed"] += 1
        else:
            n["fix_text_mismatch"] += t["id"] in fix
            n["fail_dropped"] += 1
            continue
        n["label_equals_gold"] += lab == gold.get(t["id"])
        out.append({"id": t["id"], "text": t["text"], "gold": lab, "source": src})
    json.dump(out, open(a.dst, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"labels": len(out), **n}), flush=True)


if __name__ == "__main__":
    main()
