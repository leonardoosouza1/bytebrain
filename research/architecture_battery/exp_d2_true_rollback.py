"""Experiment D2: TRUE rollback decoding — Leonardo's full idea.

D (v1) only re-drafts the current word. Leonardo described more: when the text starts drifting,
"go back ~100 bytes, check the harmony, and continue in that harmony" — i.e. delete the last few
words and resume from a coherent point, avoiding the branch that broke. This is backtracking SEARCH
for a coherent path, not just local resampling.

Compares, on the SAME trained 26M flat weights:
  BASELINE  — plain nucleus sampling
  GUIDED    — D v1 (best-of-K per word by fluency + coherence)
  ROLLBACK  — D v2 (best-of-K per word + true rollback when the running coherence window breaks)
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/architecture_battery")
import exp_d_coherence_guided_decode as D  # noqa: E402
from src.model import ByteGPT  # noqa: E402

DEV = D.DEV


@torch.no_grad()
def rollback_generate(model, n_bytes=500, seed="O ", temp0=0.8, top_p=0.9, rep=1.4,
                      K=5, alpha=1.0, tau=8.5, W=6, R=2, budget=60):
    seed_b = list(seed.encode())
    segs = []  # appended word byte-lists

    def as_text():
        return bytes(seed_b + [b for s in segs for b in s]).decode("utf-8", "ignore")

    def as_tensor():
        return torch.tensor([seed_b + [b for s in segs for b in s]], device=DEV)

    produced, iters, temp, rolls = 0, 0, temp0, 0
    while produced < n_bytes and iters < n_bytes * 3:
        iters += 1
        prev_words = D._W.findall(as_text().lower())
        prev = prev_words[-1] if prev_words else ""
        cands = []
        for _ in range(K):
            wb, lp = D._draft_word(model, as_tensor(), temp, top_p, rep)
            wtxt = bytes([c for c in wb if c not in (32, 10)]).decode("utf-8", "ignore").lower()
            wm = D._W.findall(wtxt)
            surp = D.bigram_surprisal(prev, wm[0]) if wm else 8.0
            cands.append((lp - alpha * surp, surp, wb))
        best = max(cands, key=lambda c: c[0])
        segs.append(best[2])

        words = D._W.findall(as_text().lower())
        window = " ".join(words[-W:])
        if len(words) >= W and D.wtrans(window) > tau and budget > 0 and len(segs) > R:
            for _ in range(R):           # TRUE ROLLBACK: drop last R words
                if segs:
                    segs.pop()
            budget -= 1
            rolls += 1
            temp = min(1.15, temp + 0.1)  # explore a different branch on retry
        else:
            temp = temp0
        produced = sum(len(s) for s in segs)
    return as_text(), rolls


def main():
    model = ByteGPT().to(DEV)
    D.load_checkpoint(model, D.CKPT, DEV)
    model.eval()
    print(f"loaded {model.n_params/1e6:.0f}M | {D.CKPT}\n", flush=True)
    N = 5

    t0 = time.time()
    base = [D.baseline_generate(model, 500) for _ in range(N)]
    t1 = time.time()
    guided = [D.guided_generate(model, 500) for _ in range(N)]
    t2 = time.time()
    rb = [rollback_generate(model, 500) for _ in range(N)]
    t3 = time.time()

    def stat(samples):
        return (float(np.mean([D.wtrans(s) for s in samples])),
                float(np.mean([D.coherent_span(s) for s in samples])))

    wb, sb = stat(base)
    wg, sg = stat(guided)
    wr, sr = stat([s for s, _ in rb])
    rolls = np.mean([r for _, r in rb])

    print("============ RESULTADO (mesmos pesos 26M, so a inferencia muda) ============", flush=True)
    print(f"BASELINE | wtrans {wb:.2f} | span {sb:.1f} palavras | {(t1-t0)/N:.0f}s/amostra", flush=True)
    print(f"GUIADO   | wtrans {wg:.2f} | span {sg:.1f} palavras | {(t2-t1)/N:.0f}s/amostra", flush=True)
    print(f"ROLLBACK | wtrans {wr:.2f} | span {sr:.1f} palavras | {(t3-t2)/N:.0f}s/amostra | {rolls:.0f} rollbacks/amostra", flush=True)
    print(f"\nROLLBACK amostra:\n{rb[0][0][:460]!r}", flush=True)


if __name__ == "__main__":
    main()
