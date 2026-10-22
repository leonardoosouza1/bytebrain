"""Score generated calculator candidates against calc_ref on hidden inputs.

Arms:
  whole       function taken from the single whole-module greedy pass
  greedy      per-function greedy pass
  sample@1    mean hidden pass rate of a single sampled candidate
  visible     pick the candidate passing most docstring examples (ties: biggest behaviour cluster)
  consensus   pick from the biggest behaviour cluster on fuzz inputs; uses no expected outputs at all
  oracle      any candidate passes (upper bound of selection)

Also assembles the visible-selected functions into one module and re-judges the module as a whole.

Usage: evaluate.py CANDIDATES.json OUT_DIR
"""

import ast
import collections
import json
import os
import sys
import time

import judge
import spec


KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def clean(code):
    """L0 structural filter: drop top-level statements that are not definitions.

    Generated files often end with demo calls (print(f(-3))) that raise at import time.
    If the tail does not parse (cut-off generation), drop trailing lines until it does.
    Uses no expected outputs.
    """
    lines = code.split("\n")
    for end in range(len(lines), 0, -1):
        try:
            tree = ast.parse("\n".join(lines[:end]))
            break
        except SyntaxError:
            continue
    else:
        return code
    body = [n for n in tree.body
            if isinstance(n, KEEP) or (isinstance(n, (ast.Assign, ast.AnnAssign))
                                       and not any(isinstance(x, ast.Call) for x in ast.walk(n)))]
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def score_function(entry, codes):
    hidden = spec.hidden_cases(entry)
    fuzz = spec.fuzz_cases(entry)
    calls = [list(a) for a in hidden + fuzz]
    t0 = time.perf_counter()
    out = judge.run_jobs([{"code": c, "fname": entry["name"], "calls": calls} for c in codes])
    wall = time.perf_counter() - t0
    exp_hidden = [spec.expected(entry, a) for a in hidden]
    exp_visible = [spec.expected(entry, a) for a in entry["examples"]]
    rows = []
    for o in out:
        if o["load"] != "ok":
            rows.append({"load": o["load"], "hidden": False, "hidden_frac": 0.0, "visible": -1, "sig": None, "us": 0})
            continue
        h = o["out"][:len(hidden)]
        z = o["out"][len(hidden):]
        hm = [judge.matches(e, g) for e, g in zip(exp_hidden, h)]
        vis = sum(judge.matches(e, g) for e, g in zip(exp_visible, z[:len(exp_visible)]))
        rows.append({"load": "ok", "hidden": all(hm), "hidden_frac": sum(hm) / len(hm), "visible": vis,
                     "sig": "|".join(judge.signature(g) for g in z), "us": o["us"]})
    return rows, wall, len(calls)


def pick_consensus(rows, idx):
    sizes = collections.Counter(rows[i]["sig"] for i in idx if rows[i]["sig"] is not None)
    best = [i for i in idx if rows[i]["sig"] is not None]
    if not best:
        return idx[0]
    return max(best, key=lambda i: (sizes[rows[i]["sig"]], -i))


def pick_visible(rows, idx):
    top = max(rows[i]["visible"] for i in idx)
    tied = [i for i in idx if rows[i]["visible"] == top]
    return pick_consensus(rows, tied)


def main():
    cand_path, out_dir = sys.argv[1], sys.argv[2]
    use_clean = "--raw" not in sys.argv
    cand = json.load(open(cand_path))
    prep = clean if use_clean else (lambda c: c)
    report = {"clean": "--raw" not in sys.argv, "candidates": cand_path, "model": cand["model"], "k": cand["k"], "funcs": {}}
    picks = {}
    for e in spec.SPEC:
        fc = cand["funcs"][e["name"]]
        codes = [prep(c) for c in [cand["whole"]["code"], fc["greedy"]] + fc["samples"]]
        rows, wall, n_calls = score_function(e, codes)
        pool = list(range(1, len(codes)))  # greedy + samples
        iv, ic = pick_visible(rows, pool), pick_consensus(rows, pool)
        picks[e["name"]] = codes[iv]
        samples = rows[2:]
        r = {"whole": rows[0]["hidden"], "greedy": rows[1]["hidden"],
             "sample@1": sum(x["hidden"] for x in samples) / len(samples),
             "visible": rows[iv]["hidden"], "consensus": rows[ic]["hidden"],
             "oracle": any(x["hidden"] for x in rows[1:]),
             "n_clusters": len({x["sig"] for x in rows[1:] if x["sig"]}),
             "load_fail": sum(x["load"] != "ok" for x in rows[1:]),
             "judge_wall_s": round(wall, 3), "calls_per_candidate": n_calls,
             "us_per_call": round(sum(x["us"] for x in rows if x["us"]) / max(1, sum(1 for x in rows if x["us"])), 2),
             "greedy_s": fc["greedy_s"], "samples_s": fc["samples_s"]}
        report["funcs"][e["name"]] = r
        print(f"{e['name']:13s} whole {r['whole']:d} greedy {r['greedy']:d} s@1 {r['sample@1']:.2f} "
              f"vis {r['visible']:d} cons {r['consensus']:d} oracle {r['oracle']:d} "
              f"clusters {r['n_clusters']:2d} loadfail {r['load_fail']}", flush=True)

    arms = ["whole", "greedy", "sample@1", "visible", "consensus", "oracle"]
    n = len(spec.SPEC)
    report["summary"] = {a: round(sum(float(report["funcs"][k][a]) for k in report["funcs"]) / n, 4) for a in arms}
    report["gen_s"] = {"whole": cand["whole"]["s"],
                       "greedy": round(sum(f["greedy_s"] for f in cand["funcs"].values()), 1),
                       "samples": round(sum(f["samples_s"] for f in cand["funcs"].values()), 1)}

    # assemble the visible-selected module and judge it as one file
    module = "\n\n".join(picks[e["name"]] for e in spec.SPEC) + "\n"
    with open(os.path.join(out_dir, "calc_generated.py" if use_clean else "calc_generated_raw.py"), "w") as fh:
        fh.write(module)
    jobs = [{"code": module, "fname": e["name"], "calls": [list(a) for a in spec.hidden_cases(e, seed=1)]}
            for e in spec.SPEC]
    out = judge.run_jobs(jobs)
    passed = []
    for e, o in zip(spec.SPEC, out):
        cases = spec.hidden_cases(e, seed=1)
        ok = o["load"] == "ok" and all(judge.matches(spec.expected(e, a), g) for a, g in zip(cases, o["out"]))
        passed.append(ok)
    report["assembled"] = {"functions_passing_fresh_seed": sum(passed), "of": n,
                           "failing": [e["name"] for e, p in zip(spec.SPEC, passed) if not p]}
    print(json.dumps(report["summary"]), json.dumps(report["gen_s"]), json.dumps(report["assembled"]))
    with open(os.path.join(out_dir, ("report_" if use_clean else "report_raw_") + os.path.basename(cand_path)), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
