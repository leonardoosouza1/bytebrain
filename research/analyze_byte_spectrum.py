#!/usr/bin/env python3
"""Phase 2.1 — Walsh-Hadamard spectrum of ByteBrain's learned byte embedding.

Tests Leonardo's thesis: "a letra é frequência e tem espectro" — i.e. the 256
byte embeddings are organized by their BIT structure (Walsh basis), not as
arbitrary indices. bit5=case (KL=0.003) should show up as a dominant low-order
Walsh component.

WHT: embedding E[256,dim]. Walsh coeff C = H@E/256, where H[w,b]=(-1)^popcount(w&b).
Component w depends on exactly the bits set in w. Single-bit components:
  w=1->bit0  2->bit1  4->bit2  8->bit3  16->bit4  32->bit5(case)  64->bit6  128->bit7
"""
import sys
import numpy as np
from safetensors.torch import load_file

_path = sys.argv[1] if len(sys.argv) > 1 else "export_bytebrain/model.safetensors"
E = load_file(_path)["tok.weight"].float().numpy()  # [256, dim]
print(f"# spectrum of: {_path}")
n, dim = E.shape
assert n == 256

# Sylvester-ordered Hadamard (natural binary Walsh order): H[w,b] = (-1)^<w,b>
H = np.array([[1.0]])
for _ in range(8):
    H = np.block([[H, H], [H, -H]])

C = (H @ E) / 256.0                      # [256, dim] Walsh coefficients
energy = (C ** 2).sum(axis=1)           # energy per Walsh component
dc = energy[0]                          # w=0 = mean (DC)
ac = energy[1:].sum()                   # all structure (non-DC)

def bits_of(w):
    return [k for k in range(8) if (w >> k) & 1]

BITNAME = {0:"b0", 1:"b1", 2:"b2", 3:"b3(group)", 4:"b4(group)",
           5:"b5(CASE)", 6:"b6(region)", 7:"b7(UTF8)"}

print(f"embedding [256 x {dim}]  DC(mean) energy={dc:.3f}  AC(structure) energy={ac:.3f}\n")

# Single-bit components — the heart of the thesis
print("=== single-bit Walsh components (does each BIT carry energy?) ===")
order = np.argsort(-energy[1:]) + 1     # ranks among the 255 non-DC components
rank_of = {w: int(np.where(order == w)[0][0]) + 1 for w in range(1, 256)}
single = [1, 2, 4, 8, 16, 32, 64, 128]
sb_energy = 0.0
for w in single:
    b = bits_of(w)[0]
    sb_energy += energy[w]
    print(f"  bit{b:<1} (w={w:3})  energy={energy[w]:8.3f}  {100*energy[w]/ac:5.1f}% of AC  rank {rank_of[w]:>3}/255  [{BITNAME[b]}]")
print(f"\n  8 single-bit components hold {100*sb_energy/ac:.1f}% of all structure energy")

# Top-10 components overall (which bit-patterns dominate)
print("\n=== top-10 Walsh components overall ===")
for w in order[:10]:
    bs = bits_of(w)
    tag = "+".join(f"b{k}" for k in bs) if bs else "DC"
    print(f"  w={w:3}  energy={energy[w]:8.3f}  {100*energy[w]/ac:5.1f}%  bits={tag}")

# Direct case test: project onto bit5 axis, measure upper vs lower separation
b5 = np.array([(-1.0) ** ((b >> 5) & 1) for b in range(256)])  # +1 if bit5=0, -1 if set
proj = E.T @ b5 / 256.0                  # component along bit5 for each dim
print(f"\n=== bit5 (case) axis ===")
print(f"  ||embedding projection onto bit5 Walsh axis|| = {np.linalg.norm(proj):.3f}")
low_order = sum(energy[w] for w in range(1,256) if len(bits_of(w)) <= 2)
print(f"  energy in 1-2 bit components (low 'frequency') = {100*low_order/ac:.1f}% of AC")
