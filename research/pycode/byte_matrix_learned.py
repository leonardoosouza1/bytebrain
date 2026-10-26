"""Ledger #6, the missing piece: "e se o próprio modelo ajusta matrizes?" — a factorization the model LEARNS.

Each function's byte grid (rows x its own width) gets two learned matrices, U (rows x k) and V (cols x k).
One readout shared by all functions turns u_r * v_c (elementwise, k numbers) into 256 byte logits.
Everything is fit jointly with cross-entropy (Adam).
Measured per k: bytes exactly right (argmax), and the storage of U, V in bits per non-space byte (fp16 factors, the
shared readout not counted). This is compared with 8 bits per byte (raw) and with the byte-LM's ~1.5 bits per byte.
CPU, torch, 4 threads, small (n functions). Prediction (PRED) written before the run.
"""

import argparse
import json
import os
import time

import torch

import byte_matrix_probe as P

PRED = "no k reaches >= 90% of non-space bytes exact at < 8 bits per non-space byte: per-function matrices memorize, " \
       "they do not compress"


def run(fns, k, steps, seed):
    torch.manual_seed(seed)
    grids = [torch.tensor(list(b"".join(r.ljust(max(len(x) for x in rows), b" ") for r in rows)), dtype=torch.long)
             .view(len(rows), -1) for rows in fns]
    us = [torch.nn.Parameter(torch.randn(g.shape[0], k) * 0.5) for g in grids]
    vs = [torch.nn.Parameter(torch.randn(g.shape[1], k) * 0.5) for g in grids]
    head = torch.nn.Sequential(torch.nn.Linear(k, 128), torch.nn.GELU(), torch.nn.Linear(128, 256))
    opt = torch.optim.Adam([*us, *vs, *head.parameters()], lr=3e-2)
    for step in range(steps):
        idx = torch.randint(len(grids), (64,)).tolist()
        loss = 0.0
        for i in idx:
            z = us[i][:, None, :] * vs[i][None, :, :]
            loss = loss + torch.nn.functional.cross_entropy(head(z).view(-1, 256), grids[i].view(-1))
        opt.zero_grad()
        (loss / len(idx)).backward()
        opt.step()
    ok = ok_ns = n = n_ns = bits = 0
    with torch.no_grad():
        for u, v, g in zip(us, vs, grids):
            pred = head(u[:, None, :] * v[None, :, :]).argmax(-1)
            ns = g != 32
            ok += (pred == g).sum().item()
            n += g.numel()
            ok_ns += (pred == g)[ns].sum().item()
            n_ns += ns.sum().item()
            bits += (u.numel() + v.numel()) * 16
    return {"k": k, "bytes_exact": round(ok / n, 4), "nonspace_exact": round(ok_ns / n_ns, 4),
            "bits_per_nonspace_byte": round(bits / n_ns, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/media/llm-data/datasets/pycode")
    ap.add_argument("--out", default="/media/llm-data/universe-runs/bytebrain-pycode-20261022/byte_matrix")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(4)
    fns, info = P.functions(os.path.join(a.data, "clean_train.txt"), a.n, a.seed, 40, 100)
    print(json.dumps({"prediction": PRED, "n_functions": len(fns), **info}), flush=True)
    res = []
    for k in a.ks:
        t0 = time.time()
        r = run(fns, k, a.steps, a.seed)
        r["s"] = round(time.time() - t0, 1)
        res.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"prediction": PRED, "n_functions": len(fns), "steps": a.steps, "results": res},
              open(os.path.join(a.out, "learned.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
