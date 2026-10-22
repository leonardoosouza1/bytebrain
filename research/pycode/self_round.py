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

from eval_samples import ByteLM, CachedGen, cut_function, judge

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
    lines = [l.strip() for l in src.split("\n")[1:] if len(l.strip()) > 8]
    if len(lines) != len(set(lines)):
        return "repeated_line"     # self-training narrows onto repeated lines (round 1 mirror: I(k=32) 4.8x real)
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
    # population grade: the per-sample grade alone rewards the cheapest template (round 2: __init__ 43% vs real 12%)
    ap.add_argument("--cap-per-name", type=int, default=0, help="max kept functions per def name (0 = off)")
    ap.add_argument("--cap-per-line", type=int, default=0, help="reject if a body line is already in this many kept")
    # proposal: prompt with training signatures (the task, not the body), --per-prompt attempts each
    ap.add_argument("--sig-prompts", action="store_true")
    ap.add_argument("--per-prompt", type=int, default=16)
    ap.add_argument("--run-check", action="store_true", help="reject functions that close over builtins only but fail in bwrap")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    os.makedirs(a.out_dir, exist_ok=True)
    corpus = open(f"{D}/clean_train.txt", "rb").read()
    lm = ByteLM(a.ckpt)
    gen = CachedGen(lm)
    counts, good, seen = {}, [], set()
    name_n, line_n = {}, {}
    if a.sig_prompts:
        sigs = [f.split("\n")[0] + "\n" for f in corpus.decode().split("\n\n") if f.startswith("def ")]
        sigs = [x for x in dict.fromkeys(sigs) if len(x) < gen.ctx // 2 - 4]
        random.Random(7).shuffle(sigs)
        jobs = [(x, a.per_prompt) for x in sigs[:a.n // a.per_prompt]]
    else:
        jobs = [("def ", a.batch)] * (a.n // a.batch)
    for r, (head, k) in enumerate(jobs):
        torch.manual_seed(1000 + r)
        out = gen.generate(b"\n\n" + head.encode(), k, a.bytes, temp=a.temp).cpu().numpy()
        for row in out:
            src, term = cut_function(head + bytes(row).decode("utf-8", "replace"))
            g = grade(src)
            if g == "ok" and not term:
                g = "no_end"
            elif g == "ok" and len(src) > a.max_len:
                g = "too_long"
            elif g == "ok" and src.encode() in corpus:
                g = "copy"
            elif g == "ok" and src in seen:
                g = "duplicate"
            elif g == "ok" and a.run_check:
                j = judge(src)
                if j["names_ok"] and not j["runs"]:
                    g = "run_fail"
            name = src.split("(")[0][4:].strip()
            body = {l.strip() for l in src.split("\n")[1:] if len(l.strip()) > 12}   # short lines (else:) are normal
            if g == "ok" and a.cap_per_name and name_n.get(name, 0) >= a.cap_per_name:
                g = "name_capped"
            elif g == "ok" and a.cap_per_line and any(line_n.get(l, 0) >= a.cap_per_line for l in body):
                g = "line_capped"
            counts[g] = counts.get(g, 0) + 1
            if g == "ok":
                name_n[name] = name_n.get(name, 0) + 1
                for l in body:
                    line_n[l] = line_n.get(l, 0) + 1
                seen.add(src)
                good.append(src)
        if r % 16 == 0 or r == len(jobs) - 1:
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
