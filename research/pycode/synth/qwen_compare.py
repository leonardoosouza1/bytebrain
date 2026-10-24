"""Like-for-like reference: a real code LLM on the same synth2 task (ledger #23, registered 2026-10-23, CPU).

Same input as our planner+writer: the function head (name spells the program) is given, the model writes the body.
Knowledge parity: the prompt holds few-shot train functions chosen to cover every non-held unit at least once
(our system saw all of them in training). Greedy (temperature 0), k=1, no selector, same strict bwrap grade.
Model: Qwen2.5-Coder-0.5B-Instruct q8_0 on a running llama-server (CPU, -ngl 0), raw completion endpoint.
Prediction: test_a/test_num >= 80% (it reads the words), len3..6 lower but well above our R2 control (2%),
since a pretrained LM does not need a learned address to keep going; params 494M vs our ~162k.

Usage: qwen_compare.py --out DIR [--url http://127.0.0.1:8141 --n 100]
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import synth2  # noqa: E402
from bridge_cpu import Data  # noqa: E402
from synth2 import UNITS, grade, head, source  # noqa: E402

STOP = ["\ndef ", "\n\n", "```", "\nclass ", "\nif __name__"]


def cover_shots(train, seed=0):
    """greedy set cover over train tasks: every non-held unit appears in some shot."""
    need = {u for u in UNITS if u not in synth2.HELD}
    r, pool, shots = random.Random(seed), list(train), []
    r.shuffle(pool)
    while need:
        best = max(pool, key=lambda t: len(need & set(t[0])))
        shots.append(best)
        need -= set(best[0])
    return shots


def prompt_of(shots, task):
    return "".join(source(*t) + "\n\n" for t in shots) + head(*task)


def complete(url, prompt, n_predict):
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                       "stop": STOP, "cache_prompt": True}).encode()
    req = urllib.request.Request(f"{url}/completion", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        d = json.loads(f.read())
    return d["content"], d.get("tokens_predicted", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8141")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = Data()
    shots = cover_shots(data.sets["train"])
    # smoke: the judge must give the reference 1.0 and a sign-flipped body < 1.0 (known numbers)
    t = data.sets["len4"][0]
    ref = grade([t], [[source(*t)]])[0][0]
    broken = grade([t], [[head(*t) + "    return None\n"]])[0][0]
    print(json.dumps({"smoke_ref": ref, "smoke_broken": broken, "n_shots": len(shots),
                      "prompt_chars": len(prompt_of(shots, t))}), flush=True)
    assert ref == 1.0 and broken < 1.0
    names = ["test_a", "test_num", "len3", "len4", "len5", "len6"]
    if a.smoke:
        names, a.n = ["len3"], 4
    res = {"arm": "qwen2.5-coder-0.5b-q8_0", "params": 494_032_768, "n_shots": len(shots),
           "decoding": "greedy k=1, no selector", "eval": {}}
    for k in names:
        tasks = random.Random(1).sample(data.sets[k], min(a.n, len(data.sets[k])))
        t0 = time.perf_counter()
        n_pred = max(len(source(*t)) for t in tasks)          # bytes >= tokens: never truncates a right body
        with ThreadPoolExecutor(a.par) as ex:
            outs = list(ex.map(lambda t: complete(a.url, prompt_of(shots, t), n_pred), tasks))
        gen_s = time.perf_counter() - t0
        srcs = [head(*t) + body.rstrip() + "\n" for t, (body, _) in zip(tasks, outs)]
        sc = grade(tasks, [[s] for s in srcs])
        ok = sum(s[0] == 1.0 for s in sc)
        res["eval"][k] = {"n": len(tasks), "pass1": round(ok / len(tasks), 4),
                          "text_exact": round(sum(s == source(*t) for s, t in zip(srcs, tasks)) / len(tasks), 4),
                          "tokens": sum(n for _, n in outs), "gen_s": round(gen_s, 1),
                          "correct_per_s": round(ok / gen_s, 3),
                          "fails": [s for s, q in zip(srcs, sc) if q[0] < 1.0][:3]}
        print(json.dumps({k: {m: v for m, v in res["eval"][k].items() if m != "fails"}}), flush=True)
    json.dump(res, open(f"{a.out}/qwen05_{'smoke' if a.smoke else 'full'}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
