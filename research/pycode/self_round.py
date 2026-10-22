"""One round of filter-and-retrain where the grade comes from the mirror, not from corpus text.

The model generates functions; a function passes when it parses, uses every parameter (besides self/cls), ends
(a col-0 line follows) within --max-len bytes, and is not a verbatim copy of a training function. Passing
functions (x --repeat) plus a slice of the clean corpus (replay, against collapse) form the fine-tune corpus.

Usage: self_round.py --ckpt CKPT --out-dir DIR [--n 2048]   -> DIR/good.txt, DIR/mix.txt, DIR/stats.json
"""

import argparse
import ast
import json
import os
import random

import torch

from eval_samples import ByteLM, CachedGen, cut_function

D = "/media/llm-data/datasets/pycode"


def grade(src):
    try:
        fn = ast.parse(src).body[0]
    except (SyntaxError, ValueError, IndexError, RecursionError):
        return "no_parse"
    if not isinstance(fn, ast.FunctionDef):
        return "not_function"
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    params |= {a.arg for a in (fn.args.vararg, fn.args.kwarg) if a}
    params -= {"self", "cls"}
    loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    if params - loaded:
        return "unused_param"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bytes", type=int, default=800)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-len", type=int, default=700)
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--replay-mb", type=float, default=4.0)
    ap.add_argument("--mem-frac", type=float, default=0.3)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    os.makedirs(a.out_dir, exist_ok=True)
    corpus = open(f"{D}/clean_train.txt", "rb").read()
    lm = ByteLM(a.ckpt)
    gen = CachedGen(lm)
    counts, good, seen = {}, [], set()
    for r in range(a.n // a.batch):
        torch.manual_seed(1000 + r)
        out = gen.generate(b"\n\ndef ", a.batch, a.bytes, temp=a.temp).cpu().numpy()
        for row in out:
            src, term = cut_function("def " + bytes(row).decode("utf-8", "replace"))
            g = grade(src)
            if g == "ok" and not term:
                g = "no_end"
            elif g == "ok" and len(src) > a.max_len:
                g = "too_long"
            elif g == "ok" and src.encode() in corpus:
                g = "copy"
            elif g == "ok" and src in seen:
                g = "duplicate"
            counts[g] = counts.get(g, 0) + 1
            if g == "ok":
                seen.add(src)
                good.append(src)
        print(f"round {r}: {counts}", flush=True)
    funcs = [f for f in corpus.decode().split("\n\n") if f.strip()]
    random.Random(0).shuffle(funcs)
    replay, size = [], 0
    for f in funcs:
        if size > a.replay_mb * 2 ** 20:
            break
        replay.append(f)
        size += len(f) + 2
    mix = good * a.repeat + replay
    random.Random(1).shuffle(mix)
    open(f"{a.out_dir}/good.txt", "w").write("\n\n".join(good) + "\n\n")
    open(f"{a.out_dir}/mix.txt", "w").write("\n\n".join(mix) + "\n\n")
    stats = {"generated": a.n, "counts": counts, "good": len(good),
             "good_mean_len": round(sum(map(len, good)) / max(1, len(good)), 1),
             "mix_bytes": sum(len(m) + 2 for m in mix)}
    json.dump(stats, open(f"{a.out_dir}/stats.json", "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
