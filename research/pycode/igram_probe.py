"""The corpus index alone as a generator and as a predictor.

(a) generation: sample each next byte from the index's next-byte counts, using context suffixes of at most `cap`
    bytes; judge each function (grammar / names / runs, eval_samples.judge) and its novelty (the whole function
    does not occur verbatim in the training corpus). Sweeps cap: short = novel but broken, long = copied.
(b) prediction: bits/byte on held-out clean functions with the index alone (add-k smoothing toward uniform).

Usage: igram_probe.py --out FILE.json [--n 128] [--bytes 600]
"""

import argparse
import json
import time

import numpy as np
import torch

from corpus_index import CorpusIndex
from eval_samples import cut_function, judge

D = "/media/llm-data/datasets/pycode"
LENGTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)


@torch.no_grad()
def generate(idx, n, nbytes, cap, temp=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ctx = torch.tensor([list(b"\n\ndef ")], device="cuda").repeat(n, 1)
    lengths = tuple(m for m in LENGTHS if m <= cap)
    mlog = []
    for _ in range(nbytes):
        m, cnt, h = idx.next_bytes(ctx[:, -cap:], lengths=lengths, cap=512)
        p = (h + 1e-3) ** (1 / temp)
        nxt = torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=g)
        ctx = torch.cat([ctx, nxt], 1)
        mlog.append(m.float().mean().item())
    return ctx[:, 2:].to(torch.uint8).cpu().numpy(), float(np.mean(mlog))


@torch.no_grad()
def bpb(idx, text, cap, n_pos=4096, k=0.5):
    r = np.random.default_rng(0)
    pos = r.integers(64, len(text) - 1, n_pos)
    ctx = torch.tensor(np.stack([text[p - 64:p] for p in pos]), device="cuda")
    tgt = torch.tensor(text[pos].astype(np.int64), device="cuda")
    lengths = tuple(m for m in LENGTHS if m <= cap)
    tot = 0.0
    for s in range(0, n_pos, 512):
        _, _, h = idx.next_bytes(ctx[s:s + 512, -cap:], lengths=lengths, cap=4096)
        p = (h + k) / (h.sum(1, keepdim=True) + 256 * k)
        tot += -torch.log2(p.gather(1, tgt[s:s + 512, None])).sum().item()
    return tot / n_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--bytes", type=int, default=600)
    ap.add_argument("--mem-frac", type=float, default=0.35)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    t0 = time.perf_counter()
    idx = CorpusIndex.build(f"{D}/clean_train.txt", depth=64)
    corpus = open(f"{D}/clean_train.txt", "rb").read()
    hold = np.fromfile(f"{D}/clean_holdout.txt", dtype=np.uint8)
    print(f"index built {time.perf_counter() - t0:.1f}s", flush=True)
    report = {}
    for cap in (8, 16, 32, 48):
        t1 = time.perf_counter()
        rows, mean_m = generate(idx, args.n, args.bytes, cap)
        res = []
        for row in rows:
            src, term = cut_function(bytes(row).decode("utf-8", "replace"))
            res.append({"src": src, "terminated": term, "novel": src.encode() not in corpus, **judge(src)})
        s = {k: round(float(np.mean([r[k] for r in res])), 3) for k in ("terminated", "parses", "names_ok", "runs", "novel")}
        s["novel_and_parses"] = round(float(np.mean([r["novel"] and r["parses"] for r in res])), 3)
        s["mean_match"] = round(mean_m, 1)
        s["holdout_bpb"] = round(bpb(idx, hold, cap), 3)
        s["wall_s"] = round(time.perf_counter() - t1, 1)
        report[cap] = s | {"examples": [r["src"] for r in res[:6]]}
        print(f"cap {cap:2d}: {json.dumps(s)}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh)


if __name__ == "__main__":
    main()
