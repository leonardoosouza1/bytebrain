"""Build evaluate() from 7 small pieces on llama-server, many samples per piece, pick by visible cases.

Fixes two flaws of ctx_test.py: visible examples are drawn from each piece's own case pool (so they
exercise the piece's operator), and visible/hidden cases are disjoint. Context = "full" arm (header +
reference source of the other pieces), the only arm that passed pieces before.
Each piece: k samples -> judge with reference siblings -> pick (all visible pass, consensus by output
signature). The 7 picks are assembled alone and evaluate() is judged on the calculator spec (hidden seeds 0,1).

Usage: pieces_native.py --out DIR [--k 64] [--url http://127.0.0.1:8139]
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import ctx_test as C
import evaluate as ev
import judge
import parser_ref as pr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "hybrid_ideas", "edit_by_address"))
CHAT = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n```python\n{prefix}")
STOPS = ["```", "<|im_end|>", "<|endoftext|>", "\nif __name__", "\nprint("]
N_VIS = 4


def split_cases(name, ref):
    """-> (visible, hidden), disjoint, visible drawn from the piece's own pool."""
    vis0, pool = C.cases(name, ref)
    pool = list({repr(a): a for a in pool}.values())
    r = random.Random(f"vis-{name}")
    r.shuffle(pool)
    # prefer visible cases with different outcome kinds (value vs error)
    ok = [a for a in pool if C.expected(ref, name, a)[0] == "ok"]
    err = [a for a in pool if C.expected(ref, name, a)[0] == "err"]
    vis = ok[:N_VIS - 1] + err[:1] if err else ok[:N_VIS]
    hidden = [a for a in pool if a not in vis]
    return vis, hidden


def complete(url, text, max_new, temperature, seed):
    import urllib.request
    body = {"prompt": text, "n_predict": max_new, "temperature": temperature, "top_p": 0.95, "seed": seed,
            "stop": STOPS, "cache_prompt": True}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["content"]


def judge_rows(name, codes, ref, vis, hidden):
    others = pr.source([n for n in pr.ORDER if n != name])
    calls = [list(a) for a in vis + hidden]
    res = judge.run_jobs_isolated([{"code": ev.clean(c) + "\n\n" + others, "fname": name, "calls": calls} for c in codes])
    e_vis = [C.expected(ref, name, a) for a in vis]
    e_hid = [C.expected(ref, name, a) for a in hidden]
    rows = []
    for o in res:
        if o["load"] != "ok":
            rows.append({"vis": -1, "hidden": False, "sig": None})
            continue
        v = sum(judge.matches(e, g) for e, g in zip(e_vis, o["out"][:len(vis)]))
        h = [judge.matches(e, g) for e, g in zip(e_hid, o["out"][len(vis):])]
        rows.append({"vis": v, "hidden": all(h), "sig": "|".join(judge.signature(g) for g in o["out"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--url", default="http://127.0.0.1:8139")
    ap.add_argument("--ctx", default="full", choices=["full", "stubs"],
                    help="full = reference bodies of the other pieces; stubs = their signatures and docstrings only")
    ap.add_argument("--base", default="/media/llm-data/universe-runs/hybrid-ideas-20261021/edit_by_address/"
                    "fim_server/select_visible.json", help="file with the region fixes applied (final_source)")
    args = ap.parse_args()
    ref = pr.module()
    pool = ThreadPoolExecutor(32)
    picks, report = {}, {"k": args.k, "ctx": args.ctx, "pieces": {}}
    for name in pr.ORDER:
        vis, hidden = split_cases(name, ref)
        stub = C.with_examples(name, ref, vis)
        user = C.ASK_CTX.format(ctx=C.context(args.ctx, name), stub=stub)
        prefix = pr.HEADER + "\n\n" + stub
        text = CHAT.format(user=user, prefix=prefix)
        t0 = time.perf_counter()
        outs = list(pool.map(lambda s: complete(args.url, text, 512, args.temperature if s else 0.0, 3000 + s),
                             range(args.k + 1)))
        gen_s = time.perf_counter() - t0
        codes = [prefix + o for o in outs]
        rows = judge_rows(name, codes, ref, vis, hidden)
        full = [i for i, r in enumerate(rows) if r["vis"] == len(vis)]
        if full:
            sig, _ = Counter(rows[i]["sig"] for i in full).most_common(1)[0]
            ip = next(i for i in full if rows[i]["sig"] == sig)
        else:
            ip = max(range(len(rows)), key=lambda i: rows[i]["vis"])
        picks[name] = ev.clean(codes[ip])
        p = {"gen_s": round(gen_s, 1), "greedy_hidden": rows[0]["hidden"],
             "visible_full": len(full), "visible_full_and_hidden": sum(rows[i]["hidden"] for i in full),
             "hidden_any": sum(r["hidden"] for r in rows), "pick_hidden": rows[ip]["hidden"],
             "n_vis": len(vis), "n_hidden": len(hidden)}
        report["pieces"][name] = p
        print(f"{name:12s} gen {gen_s:5.1f}s greedy {p['greedy_hidden']:d} visible-full {len(full):3d} "
              f"-> hidden {p['visible_full_and_hidden']:3d}  hidden any {p['hidden_any']:3d}/{len(rows)}  "
              f"pick {p['pick_hidden']:d}", flush=True)
    module = pr.HEADER + "\n\n" + "\n\n".join(picks[n] for n in pr.ORDER)
    report["module"] = module
    # judge the assembled evaluate on the calculator spec (hidden seeds 0,1)
    import localize as L
    import whole_function as W
    src = json.load(open(args.base))["final_source"]
    _, span = W.header_of(src, "evaluate")
    new = W.replace_function(src, span, module)
    hid = L.judge_module(new)
    report["calculator_hidden"] = sum(hid.values())
    report["calculator_failing"] = [k for k, v in hid.items() if not v]
    print(f"assembled evaluate on calculator spec: {hid['evaluate']}  calculator {sum(hid.values())}/{len(hid)}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"pieces_k{args.k}_{args.ctx}.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
