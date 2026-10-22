"""fim_repair.py on llama-server (native HIP) instead of PyTorch eager.

Same prompts, same cut + patch + re-judge; generation goes to a running llama-server
(`/completion`, raw FIM text, special tokens parsed). k samples are k concurrent requests with
different seeds, so the server batches them across its parallel slots.

Start the server first, e.g.:
  llama-server -m qwen2.5-coder-0.5b-instruct-q8_0.gguf -ngl 99 -fa on -np 32 -c 49152 --host 127.0.0.1 --port 8139

Usage: fim_server.py --out DIR [--k 32] [--url http://127.0.0.1:8139]
"""

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import fim_repair as F
import localize as L
import run_patch_model as R


def complete(url, text, max_new, temperature, seed):
    body = {"prompt": text, "n_predict": max_new, "temperature": temperature, "top_p": 0.95, "seed": seed,
            "stop": F.STOPS, "cache_prompt": True}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out["content"], out.get("tokens_predicted", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--url", default="http://127.0.0.1:8139")
    args = ap.parse_args()
    data = json.load(open(os.path.join(F.HERE, "prompts.json")))
    source = data["source"]
    baseline = L.judge_module(source)
    pool = ThreadPoolExecutor(max(1, args.k))
    results = []
    for p in data["prompts"]:
        if p["scheme"] not in F.SCHEMES:
            continue
        for hint in (True, False):
            text = F.fim_text(p, hint)
            max_new = 3 * (p["target_tokens"] or 60) + 40
            t0 = time.perf_counter()
            g = complete(args.url, text, max_new, 0.0, 0)
            t_g = time.perf_counter() - t0
            t0 = time.perf_counter()
            ss = list(pool.map(lambda s: complete(args.url, text, max_new, args.temperature, 1000 + s), range(args.k)))
            t_s = time.perf_counter() - t0
            indent = F.split_block(p)[2]
            r = {"name": p["name"], "scheme": p["scheme"], "hint": hint, "greedy_s": round(t_g, 3),
                 "sample_s": round(t_s, 3), "answers": []}
            for kind, (t, n) in [("greedy", g)] + [("sample", x) for x in ss]:
                a = R.judge_answer(source, p, R.clean_answer(F.cut_to_region(t, indent)), baseline)
                a.update(kind=kind, answer=t, new_tokens=n)
                r["answers"].append(a)
            ok = [a for a in r["answers"] if a.get("fixed") and not a.get("regressions")]
            r["greedy_fixed"] = bool(r["answers"][0].get("fixed") and not r["answers"][0].get("regressions"))
            r["n_fixed"] = len(ok)
            results.append(r)
            print(f"{p['name']:10s} {p['scheme']:12s} hint {hint:d} greedy {r['greedy_fixed']:d} "
                  f"fixed {len(ok)}/{len(r['answers'])}  gen {t_g:.2f}s + {t_s:.2f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"fim_server_k{args.k}.json")
    with open(path, "w") as fh:
        json.dump({"k": args.k, "url": args.url, "results": results}, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
