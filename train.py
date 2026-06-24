"""Resumable ByteBrain trainer — built for multi-day runs on a large corpus, power-loss safe.

Every --ckpt-secs it writes a FULL checkpoint atomically (model + optimizer + step + RNG state).
On restart it reloads everything and resumes from the exact step, so a power cut or crash costs at
most --ckpt-secs of progress and never a corrupt file (write-to-temp + atomic rename). The corpus
is memory-mapped, so it scales to many GB without loading into RAM.

    python train.py --corpus data/pt_big.txt --ckpt-dir ckpt_big
    # ...power dies... just run the same command again -> resumes from the last checkpoint.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import ByteGPT  # noqa: E402


def atomic_save(obj, path):
    """Write to a temp file then atomically rename — a power cut mid-write never corrupts `path`."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def get_batch(data, bs, L, device):
    ix = np.random.randint(0, len(data) - L - 1, bs)
    x = np.stack([np.asarray(data[i:i + L + 1]) for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=2_000_000)
    ap.add_argument("--ckpt-secs", type=int, default=300)   # full checkpoint every 5 min
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.ckpt_dir, exist_ok=True)
    DEV = a.device

    data = np.memmap(a.corpus, dtype=np.uint8, mode="r")    # scales to many GB, no RAM blow-up
    n = len(data)
    cut = int(n * 0.999)
    TR, VA = data[:cut], data[cut:]
    print(f"corpus {n/1e6:.1f}MB | train {len(TR)/1e6:.1f}MB | val {len(VA)/1e6:.2f}MB", flush=True)

    model = ByteGPT(dim=a.dim, n_layers=a.layers, n_heads=a.heads, context=a.ctx).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.wd)
    step, best_val, elapsed_prev = 0, 1e9, 0.0

    ckpt_path = os.path.join(a.ckpt_dir, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)  # our own ckpt holds RNG/config
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best_val, elapsed_prev = ck["step"], ck.get("best_val", 1e9), ck.get("elapsed", 0.0)
        try:
            torch.set_rng_state(ck["torch_rng"].cpu()); np.random.set_state(ck["np_rng"])
        except Exception:
            pass
        print(f"RESUMIDO do step {step} ({elapsed_prev/3600:.1f}h ja treinadas, best_val {best_val:.3f})", flush=True)
    else:
        print(f"treino NOVO | {model.n_params/1e6:.0f}M params", flush=True)

    t_start = time.time()

    def save(tag):
        atomic_save({
            "model": model.state_dict(), "opt": opt.state_dict(), "step": step,
            "best_val": best_val, "elapsed": elapsed_prev + (time.time() - t_start),
            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
            "config": {"dim": a.dim, "layers": a.layers, "heads": a.heads, "ctx": a.ctx},
        }, os.path.join(a.ckpt_dir, tag + ".pt"))

    @torch.no_grad()
    def validate():
        model.eval()
        tot = 0.0
        for _ in range(20):
            xb = get_batch(VA, 32, a.ctx, DEV)
            tot += F.cross_entropy(model(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1)).item()
        model.train()
        return tot / 20 / math.log(2)

    model.train()
    last_ckpt = time.time()
    try:
        while step < a.max_steps:
            xb = get_batch(TR, a.batch, a.ctx, DEV)
            loss = F.cross_entropy(model(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                tok = step * a.batch * a.ctx
                hrs = (elapsed_prev + time.time() - t_start) / 3600
                print(f"step {step} | bpb {loss.item()/math.log(2):.3f} | {tok/1e6:.0f}M tok | {hrs:.1f}h", flush=True)
            if step % a.val_every == 0:
                v = validate()
                if v < best_val:
                    best_val = v
                    save("ckpt_best")
                print(f"  val_bpb {v:.3f} (best {best_val:.3f})", flush=True)
            if time.time() - last_ckpt > a.ckpt_secs:
                save("ckpt")
                last_ckpt = time.time()
                print(f"  [checkpoint @ step {step}]", flush=True)
    finally:
        save("ckpt")
        print(f"checkpoint final salvo @ step {step}", flush=True)


if __name__ == "__main__":
    main()
