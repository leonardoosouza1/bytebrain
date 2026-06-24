"""Experiment E: entropy-gated decoding with safe-point backtracking.

The verifier that broke D2 was an external word-bigram (brittle: calls rare-but-valid PT
"incoherent", thrashes). E uses an INTRINSIC, byte-level signal instead: the model's own next-byte
entropy. When the model is confident (low entropy) it is on-distribution = obeying the language's
rules; when it drifts, entropy stays high. So:

  * low-entropy positions are "safe points" (confident forks);
  * when the windowed mean entropy exceeds what real text ever shows (calibrated), the model has
    lost the thread -> roll back to the last safe point, BLOCK the byte that led there, and take a
    different branch (no temperature bump, bounded budget -> no thrash).

Thresholds are CALIBRATED on real PT first, so we don't guess. Compares baseline / D-v1-guided /
E-entropy-backtrack on the same trained 26M flat weights.
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/architecture_battery")
import exp_d_coherence_guided_decode as D  # noqa: E402
from src.model import ByteGPT  # noqa: E402

DEV = D.DEV
DRIFT_WIN = 64


@torch.no_grad()
def _dist(model, x, temp, rep):
    logit = model(x[:, -model.context:])[0, -1].clone()
    for b in set(x[0, -48:].tolist()):
        logit[b] /= rep
    return F.softmax(logit / temp, -1)


def _entropy(p):
    return float(-(p * (p + 1e-12).log()).sum())


def _nucleus(p, top_p):
    sp, si = torch.sort(p, descending=True)
    keep = torch.cumsum(sp, 0) <= top_p
    keep[0] = True
    sp = sp * keep
    sp = sp / sp.sum()
    return int(si[torch.multinomial(sp, 1)])


@torch.no_grad()
def calibrate_entropy(model, temp, rep, n_windows=16, win=400):
    """Measure the model's per-byte entropy on REAL PT. Returns (ent_low, drift_thresh)."""
    Bn = np.frombuffer(D.TXT.encode("utf-8"), np.uint8)
    per_byte = []
    windowed = []
    for _ in range(n_windows):
        i = np.random.randint(0, len(Bn) - win - 1)
        seg = Bn[i:i + win].tolist()
        es = []
        for j in range(32, win):                      # skip warmup
            x = torch.tensor([seg[:j]], device=DEV)
            es.append(_entropy(_dist(model, x, temp, rep)))
        per_byte += es
        for k in range(DRIFT_WIN, len(es)):
            windowed.append(float(np.mean(es[k - DRIFT_WIN:k])))
    ent_low = float(np.percentile(per_byte, 25))
    drift_thresh = float(np.percentile(windowed, 97))   # only fire above what real text ever shows
    print(f"calib: per-byte H p25={ent_low:.2f} p50={np.percentile(per_byte,50):.2f} "
          f"p90={np.percentile(per_byte,90):.2f} | windowed-mean p97={drift_thresh:.2f}", flush=True)
    return ent_low, drift_thresh


@torch.no_grad()
def entropy_backtrack_generate(model, n_bytes, ent_low, drift_thresh, seed="O ",
                               temp=0.7, top_p=0.9, rep=1.4, total_budget=80):
    seed_b = list(seed.encode())
    out = list(seed_b)
    ent = [0.0] * len(out)
    safe = [len(out)]
    blocked = {}
    budget = total_budget
    rolls = 0
    while len(out) - len(seed_b) < n_bytes and budget > 0:
        pos = len(out)
        p = _dist(model, torch.tensor([out], device=DEV), temp, rep)
        if pos in blocked:
            for b in blocked[pos]:
                p[b] = 0.0
            s = float(p.sum())
            if s < 1e-6:                                   # safe point exhausted -> go earlier
                safe = [q for q in safe if q < pos]
                tgt = safe[-1] if safe else len(seed_b)
                del blocked[pos]
                while len(out) > tgt:
                    out.pop(); ent.pop()
                budget -= 1; rolls += 1
                continue
            p = p / s
        H = _entropy(p)
        b = _nucleus(p, top_p)
        out.append(b); ent.append(H)
        if H < ent_low:
            safe.append(len(out))
        if len(ent) >= DRIFT_WIN:
            hbar = sum(ent[-DRIFT_WIN:]) / DRIFT_WIN
            if hbar > drift_thresh:
                tgt = max([q for q in safe if q < len(out) - 1], default=len(seed_b))
                blocked.setdefault(tgt, set()).add(out[tgt])
                while len(out) > tgt:
                    out.pop(); ent.pop()
                safe = [q for q in safe if q <= tgt]
                budget -= 1; rolls += 1
    return bytes(out).decode("utf-8", "ignore"), rolls


def main():
    model = ByteGPT().to(DEV)
    D.load_checkpoint(model, D.CKPT, DEV)
    model.eval()
    print(f"loaded {model.n_params/1e6:.0f}M | {D.CKPT}\n", flush=True)
    N = 5
    ent_low, drift_thresh = calibrate_entropy(model, 0.7, 1.4)

    t0 = time.time()
    base = [D.baseline_generate(model, 500) for _ in range(N)]
    t1 = time.time()
    guided = [D.guided_generate(model, 500) for _ in range(N)]
    t2 = time.time()
    ent_res = [entropy_backtrack_generate(model, 500, ent_low, drift_thresh) for _ in range(N)]
    t3 = time.time()

    def stat(ss):
        return (float(np.mean([D.wtrans(s) for s in ss])), float(np.mean([D.coherent_span(s) for s in ss])))

    wb, sb = stat(base)
    wg, sg = stat(guided)
    we, se = stat([s for s, _ in ent_res])
    rolls = np.mean([r for _, r in ent_res])

    print("\n============ RESULTADO (mesmos pesos 26M) ============", flush=True)
    print(f"BASELINE         | wtrans {wb:.2f} | span {sb:.1f} | {(t1-t0)/N:.0f}s/amostra", flush=True)
    print(f"GUIADO (D v1)    | wtrans {wg:.2f} | span {sg:.1f} | {(t2-t1)/N:.0f}s/amostra", flush=True)
    print(f"ENTROPY-BACKTRACK| wtrans {we:.2f} | span {se:.1f} | {(t3-t2)/N:.0f}s/amostra | {rolls:.0f} rollbacks", flush=True)
    print(f"\nENTROPY-BACKTRACK amostra:\n{ent_res[0][0][:460]!r}", flush=True)


if __name__ == "__main__":
    main()
