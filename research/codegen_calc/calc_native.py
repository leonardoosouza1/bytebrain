"""Whole calculator from scratch on llama-server: k samples per function, pick by leak-free visible cases.

Visible = docstring examples + seed-2 cases that are NOT edge cases (spec.hidden_cases repeats the edges in every
seed, so using them would leak the hidden set). Among samples passing every visible case, pick the largest
behaviour cluster on fuzz inputs (no expected outputs). Hidden = seeds 0,1 in full, only for the report.

Usage: calc_native.py --out DIR [--k 32] [--url http://127.0.0.1:8140] [--only evaluate]
"""

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import evaluate as ev
import judge
import pieces_native as P
import spec

HEAD = "import math\n\n\n"
ASK = "Implement this function of a scientific calculator in Python. Use only the standard library.\n\n```python\n{}```"


def visible_cases(entry):
    edge = {repr(c) for c in entry["edge"]}
    seen, out = set(), []
    for c in list(entry["examples"]) + [c for c in spec.hidden_cases(entry, seed=2) if repr(c) not in edge]:
        if repr(c) not in seen:
            seen.add(repr(c))
            out.append(c)
    return out


def hidden_cases(entry):
    seen, out = set(), []
    for c in spec.hidden_cases(entry, seed=0) + spec.hidden_cases(entry, seed=1):
        if repr(c) not in seen:
            seen.add(repr(c))
            out.append(c)
    return out


def judge_function(entry, codes):
    vis, hid, fuzz = visible_cases(entry), hidden_cases(entry), spec.fuzz_cases(entry)
    calls = [list(a) for a in vis + hid + fuzz]
    res = judge.run_jobs_isolated([{"code": c, "fname": entry["name"], "calls": calls} for c in codes])
    e_vis = [spec.expected(entry, a) for a in vis]
    e_hid = [spec.expected(entry, a) for a in hid]
    rows = []
    for o in res:
        if o["load"] != "ok":
            rows.append({"vis": -1, "hidden": False, "sig": None})
            continue
        out = o["out"]
        v = sum(judge.matches(e, g) for e, g in zip(e_vis, out[:len(vis)]))
        h = all(judge.matches(e, g) for e, g in zip(e_hid, out[len(vis):len(vis) + len(hid)]))
        sig = "|".join(judge.signature(g) for g in out[len(vis) + len(hid):])
        rows.append({"vis": v, "hidden": h, "sig": sig})
    return rows, len(vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--only", default="", help="comma-separated function names (default: all)")
    ap.add_argument("--max-new", type=int, default=512, help="token cap per sample (evaluate needs ~1000; 512 truncates)")
    args = ap.parse_args()
    entries = [e for e in spec.SPEC if not args.only or e["name"] in args.only.split(",")]
    pool = ThreadPoolExecutor(32)
    picks, report, samples = {}, {"k": args.k, "functions": {}}, {}
    t_all = time.perf_counter()
    for entry in entries:
        name = entry["name"]
        stub = spec.stub(entry)
        prefix = HEAD + stub
        text = P.CHAT.format(user=ASK.format(prefix), prefix=prefix)
        t0 = time.perf_counter()
        outs = list(pool.map(lambda s: P.complete(args.url, text, args.max_new, args.temperature if s else 0.0, 4000 + s),
                             range(args.k + 1)))
        gen_s = time.perf_counter() - t0
        codes = [ev.clean(prefix + o) for o in outs]
        rows, n_vis = judge_function(entry, codes)
        full = [i for i, r in enumerate(rows) if r["vis"] == n_vis]
        if full:
            sig, _ = Counter(rows[i]["sig"] for i in full).most_common(1)[0]
            ip = next(i for i in full if rows[i]["sig"] == sig)
        else:
            ip = max(range(len(rows)), key=lambda i: rows[i]["vis"])
        picks[name] = codes[ip]
        f = {"gen_s": round(gen_s, 1), "n_vis": n_vis, "greedy_hidden": rows[0]["hidden"],
             "visible_full": len(full), "visible_full_and_hidden": sum(rows[i]["hidden"] for i in full),
             "hidden_any": sum(r["hidden"] for r in rows), "pick_hidden": rows[ip]["hidden"]}
        report["functions"][name] = f
        samples[name] = [{"out": o, "vis": r["vis"], "hidden": r["hidden"]} for o, r in zip(outs, rows)]
        print(f"{name:16s} gen {gen_s:5.1f}s vis-full {len(full):3d} -> hidden {f['visible_full_and_hidden']:3d}  "
              f"any {f['hidden_any']:3d}/{len(rows)}  pick {f['pick_hidden']:d}", flush=True)
    picked = sum(f["pick_hidden"] for f in report["functions"].values())
    report["picked_hidden_ok"] = picked
    report["wall_s"] = round(time.perf_counter() - t_all, 1)
    # one module: strip each pick's own `import math` header, keep one
    module = HEAD + "\n\n".join(picks[e["name"]][len(HEAD):] if picks[e["name"]].startswith(HEAD) else picks[e["name"]]
                                for e in entries)
    report["module"] = module
    print(f"picked functions hidden-ok: {picked}/{len(entries)}  wall {report['wall_s']}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"calc_k{args.k}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    with open(os.path.join(args.out, f"samples_k{args.k}.json"), "w") as fh:
        json.dump(samples, fh)
    with open(os.path.join(args.out, "calculator.py"), "w") as fh:
        fh.write(module)


if __name__ == "__main__":
    main()
