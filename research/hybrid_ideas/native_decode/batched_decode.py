"""Batched decode timing on llama.cpp HIP (the k-samples-in-parallel case of the generator).

Prefill one prompt in sequence 0, copy its KV to sequences 1..k-1, then decode one token per sequence
per step. Token ids are fixed (timing only: the kernels do the same work whatever the token is).
Compare with PyTorch eager: 81 ms/step at k=1, 97 ms/step at k=16 (persistent_kv/REPORT.md).

Usage: batched_decode.py MODEL.gguf [--prompt 4000] [--steps 64] [--ks 1,4,8,16,32,64]
"""

import argparse
import ctypes
import json
import os
import time

import llama_cpp as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--ks", default="1,4,8,16,32,64")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    kmax = max(ks)

    L.llama_backend_init()
    mp = L.llama_model_default_params()
    mp.n_gpu_layers = 99
    model = L.llama_model_load_from_file(args.model.encode(), mp)
    cp = L.llama_context_default_params()
    cp.n_ctx = kmax * (args.prompt + args.steps + 8)   # unified KV: every sequence keeps its own copy
    cp.n_batch = max(512, kmax)
    cp.n_ubatch = 512
    cp.n_seq_max = kmax
    cp.flash_attn_type = 1 if hasattr(cp, "flash_attn_type") else 0
    ctx = L.llama_init_from_model(model, cp)
    mem = L.llama_get_memory(ctx)
    batch = L.llama_batch_init(max(512, kmax), 0, kmax)

    def fill(items):
        """items: list of (token, pos, seq, want_logits)"""
        batch.n_tokens = len(items)
        for i, (t, p, s, lg) in enumerate(items):
            batch.token[i] = t
            batch.pos[i] = p
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = s
            batch.logits[i] = lg
        assert L.llama_decode(ctx, batch) == 0

    rows = []
    for k in ks:
        L.llama_memory_clear(mem, True)
        t0 = time.perf_counter()
        for a in range(0, args.prompt, 512):
            n = min(512, args.prompt - a)
            fill([(100 + (a + i) % 1000, a + i, 0, i == n - 1) for i in range(n)])
        L.llama_synchronize(ctx)
        t_pre = time.perf_counter() - t0
        t0 = time.perf_counter()
        for s in range(1, k):
            L.llama_memory_seq_cp(mem, 0, s, -1, -1)
        L.llama_synchronize(ctx)
        t_cp = time.perf_counter() - t0
        fill([(200, args.prompt, s, True) for s in range(k)])   # warm-up step
        L.llama_synchronize(ctx)
        t0 = time.perf_counter()
        for st in range(1, args.steps):
            fill([(200 + st, args.prompt + st, s, True) for s in range(k)])
            L.llama_get_logits_ith(ctx, k - 1)
        L.llama_synchronize(ctx)
        ms = (time.perf_counter() - t0) * 1000 / (args.steps - 1)
        r = {"k": k, "prefill_s": round(t_pre, 3), "kv_copy_ms": round(t_cp * 1000, 2),
             "ms_per_step": round(ms, 2), "tok_per_s": round(k * 1000 / ms, 1)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"model": args.model, "prompt": args.prompt, "steps": args.steps, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
