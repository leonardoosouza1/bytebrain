#!/usr/bin/env python3
"""Phase 2.3 — quantify bit5=case INSIDE ByteBrain's learned embedding.
Turns the Walsh 'rank 6' into hard numbers:
  - cosine(model's learned case-axis, the bit5 Walsh component)  -> do they align?
  - linear separability of UPPER vs lower letters from the embedding
  - separability using ONLY the 1-D bit5 Walsh projection
"""
import sys
import numpy as np
from safetensors.torch import load_file

H = np.array([[1.0]])
for _ in range(8):
    H = np.block([[H, H], [H, -H]])

UP = list(range(65, 91)); LO = list(range(97, 123))
letters = UP + LO
labels = np.array([0] * 26 + [1] * 26)        # 0=upper, 1=lower

def analyze(path):
    E = load_file(path)["tok.weight"].float().numpy()      # [256, dim]
    # model's learned case direction (data-derived)
    case_axis = E[LO].mean(0) - E[UP].mean(0)
    case_axis /= np.linalg.norm(case_axis) + 1e-9
    # bit5 Walsh component direction (w=32) in feature space
    c32 = (H[32] @ E) / 256.0
    c32 /= np.linalg.norm(c32) + 1e-9
    cos = abs(float(case_axis @ c32))

    def acc_along(axis):
        proj = E[letters] @ axis
        thr = 0.5 * (proj[labels == 0].mean() + proj[labels == 1].mean())
        pred = (proj > thr).astype(int)
        # axis sign may be flipped; take max(acc, 1-acc)
        a = (pred == labels).mean()
        return max(a, 1 - a)

    acc_case = acc_along(case_axis)     # best linear case axis
    acc_bit5 = acc_along(c32)           # ONLY the bit5 spectral axis
    return cos, acc_case, acc_bit5

print(f"{'model':<26} {'cos(case,bit5)':>15} {'acc(case axis)':>15} {'acc(bit5 only)':>15}")
for path in (sys.argv[1:] or ["export_bytebrain_40m/model.safetensors",
                              "export_bytebrain_8m/model.safetensors",
                              "export_bytebrain/model.safetensors"]):
    cos, ac, ab = analyze(path)
    name = path.split("/")[0]
    print(f"{name:<26} {cos:>15.3f} {ac:>14.1%} {ab:>14.1%}")
print("\n(acc ~50% = chance; high cos = model's case direction IS the bit5 Walsh component)")
