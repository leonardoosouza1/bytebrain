"""Re-pick calc_native samples with model-written tests (CodeT-style dual agreement).

Visible cases can miss a contract stated only in the docstring prose (factorial: "5.0 is accepted"); 32/33 samples
ignore it and the consensus picks the majority. Here the frozen model also writes test lines from the docstring.
Among samples that pass every visible case, behaviour clusters are scored either CodeT-style |cluster| * (tests
passed) or by the summed vote weight of the generated tests they pass (--rule votes; cluster size breaks ties).
Generated tests have no access to hidden cases; hidden seeds 0,1 are read only for the report.

Usage: codet_rerank.py --run DIR [--k 32] [--t 8] [--url http://127.0.0.1:8140]
"""

import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import calc_native as cn
import evaluate as ev
import judge
import pieces_native as P
import spec

ASK_TESTS = ("Here is a function of a scientific calculator:\n\n```python\n{stub}```\n\nWrite 10 test cases that check "
             "the behaviour described in the docstring, especially the special cases it mentions. One per line, "
             "exactly in one of these two forms:\n{name}(ARGS) == VALUE\n{name}(ARGS) raises ErrorName")
LINE = re.compile(r"^\s*(?:assert\s+)?(\w+\(.*\))\s*(==|raises)\s*(.+?)\s*$")


def json_safe(x, containers):
    """True if x crosses the JSON sandbox boundary unchanged (no complex, no dict/set)."""
    if isinstance(x, bool) or isinstance(x, str) or (isinstance(x, (int, float)) and not isinstance(x, complex)):
        return True
    if containers and isinstance(x, (tuple, list)) and len(x) <= 64:
        return all(json_safe(v, containers) for v in x)
    return False


def parse_tests(name, text, containers=False):
    """containers=True also keeps list/tuple arguments (parser helpers take token lists)."""
    out = []
    for line in text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        try:
            call = ast.parse(m.group(1), mode="eval").body
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == name) or call.keywords:
                continue
            args = tuple(ast.literal_eval(a) for a in call.args)
            if not all(json_safe(a, containers) for a in args):
                continue
            if m.group(2) == "==":
                exp = ("ok", ast.literal_eval(m.group(3)))
                if not json_safe(exp[1], True):
                    continue
            else:
                exp = ("err", m.group(3).strip().split(".")[-1].rstrip(")"))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        out.append((args, exp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="calc_native output dir (samples_k{k}.json)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--t", type=int, default=8, help="test-writing samples per function")
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--rule", default="votes", choices=["codet", "votes"],
                    help="codet = |cluster| * tests passed; votes = sum of test vote weights passed, size breaks ties")
    ap.add_argument("--only", default="", help="comma-separated function names (default: all in the samples file)")
    args = ap.parse_args()
    samples = json.load(open(os.path.join(args.run, f"samples_k{args.k}.json")))
    pool = ThreadPoolExecutor(32)
    report, picks = {"t": args.t, "rule": args.rule, "functions": {}}, {}
    entries = [e for e in spec.SPEC if e["name"] in samples and (not args.only or e["name"] in args.only.split(","))]
    for entry in entries:
        name = entry["name"]
        stub = spec.stub(entry)
        text = P.CHAT.format(user=ASK_TESTS.format(stub=stub, name=name), prefix="")
        outs = list(pool.map(lambda s: P.complete(args.url, text, 400, 0.8 if s else 0.0, 5000 + s), range(args.t)))
        # one test per input: majority expected value over the test-writing samples, weight = its vote count
        votes, seen = defaultdict(Counter), {}
        for o in outs:
            for t in {repr(t): t for t in parse_tests(name, o)}.values():
                seen[repr(t)] = t
                votes[repr(t[0])][repr(t)] += 1
        tests, weight = [], []
        for c in votes.values():
            rt, w = c.most_common(1)[0]
            tests.append(seen[rt])
            weight.append(w)
        rows = samples[name]
        n_vis = max(r["vis"] for r in rows)
        codes = [ev.clean(cn.HEAD + stub + r["out"]) for r in rows]
        cand = [i for i, r in enumerate(rows) if r["vis"] == n_vis and n_vis > 0] or list(range(len(rows)))
        res = judge.run_jobs_isolated([{"code": codes[i], "fname": name, "calls": [list(a) for a, _ in tests]}
                                       for i in cand]) if tests else [{"load": "ok", "out": []} for _ in cand]
        groups = defaultdict(list)
        for i, o in zip(cand, res):
            if o["load"] != "ok":
                continue
            passed = frozenset(j for j, ((_, e), g) in enumerate(zip(tests, o["out"])) if judge.matches(e, g))
            sig = "|".join(judge.signature(g) for g in o["out"])
            groups[(sig, passed)].append(i)
        if groups:
            if args.rule == "codet":
                key = lambda kv: (len(kv[1]) * len(kv[0][1]), len(kv[1]))  # noqa: E731
            else:
                key = lambda kv: (sum(weight[j] for j in kv[0][1]), len(kv[1]))  # noqa: E731
            (_, passed), members = max(groups.items(), key=key)
            ip = members[0]
        else:
            ip, passed = cand[0], frozenset()
        picks[name] = codes[ip]
        f = {"n_tests": len(tests), "pick_hidden": rows[ip]["hidden"], "pick_passed": len(passed),
             "n_groups": len(groups)}
        report["functions"][name] = f
        print(f"{name:14s} tests {len(tests):3d} groups {len(groups):3d} pick passes {len(passed):3d}  "
              f"hidden {f['pick_hidden']:d}", flush=True)
    report["picked_hidden_ok"] = sum(f["pick_hidden"] for f in report["functions"].values())
    print(f"picked functions hidden-ok: {report['picked_hidden_ok']}/{len(entries)}", flush=True)
    module = cn.HEAD + "\n\n".join(picks[e["name"]].removeprefix("import math").lstrip("\n") for e in entries)
    with open(os.path.join(args.run, f"codet_{args.rule}_t{args.t}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.run, f"calculator_{args.rule}_t{args.t}.py"), "w") as fh:
        fh.write(module)


if __name__ == "__main__":
    main()
