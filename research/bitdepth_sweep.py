#!/usr/bin/env python3
"""Phase 1.2 — bit-depth sweep of NEURON STATES (activations).
Leonardo's question: "2 ou 4 bits por neurônio?" Quantize each block's output
(the residual stream = neurons' states) to k bits, measure bpb degradation.
Compares per-TENSOR (naive) vs per-CHANNEL (handles activation outliers).
"""
import os, sys, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ByteGPT, load_checkpoint

CKPT = sys.argv[1] if len(sys.argv) > 1 else "ckpt_big/ckpt_best.pt"
DIM, LAYERS, HEADS, CTX = 640, 8, 10, 512   # 40M config (val 1.288)

TEXT = (
    "O Brasil é um país de dimensões continentais, com uma rica diversidade "
    "cultural e natural. A floresta amazônica abriga milhões de espécies, muitas "
    "ainda desconhecidas pela ciência. A língua portuguesa, falada por mais de "
    "duzentos milhões de pessoas, carrega séculos de história e influências de "
    "povos indígenas, africanos e europeus. A música, a literatura e a culinária "
    "refletem essa mistura única que define a identidade nacional. "
) * 12

def quantize(x, bits, mode):
    if bits >= 16:
        return x
    qmax = (1 << bits) - 1
    if mode == "channel":                       # per-feature-dim scale
        dims = tuple(range(x.dim() - 1))
        lo = x.amin(dims, keepdim=True); hi = x.amax(dims, keepdim=True)
    else:                                        # per-tensor scale
        lo = x.min(); hi = x.max()
    rng = (hi - lo).clamp_min(1e-9)
    xq = torch.round(((x - lo) / rng) * qmax) / qmax
    return xq * rng + lo

@torch.no_grad()
def bpb(model, ids, bits, mode):
    hooks = []
    if bits < 16:
        for blk in model.blocks:
            hooks.append(blk.register_forward_hook(lambda m, i, o, b=bits: quantize(o, b, mode)))
    tot_loss = tot = 0
    for s in range(0, len(ids) - 1 - CTX, CTX):
        x = ids[s:s + CTX].unsqueeze(0); y = ids[s + 1:s + 1 + CTX].unsqueeze(0)
        loss = F.cross_entropy(model(x).view(-1, 256), y.reshape(-1), reduction="sum")
        tot_loss += loss.item(); tot += y.numel()
    for h in hooks: h.remove()
    return (tot_loss / tot) / math.log(2)

model = ByteGPT(dim=DIM, n_layers=LAYERS, n_heads=HEADS, context=CTX)
load_checkpoint(model, CKPT); model.eval()
ids = torch.tensor(list(TEXT.encode("utf-8")), dtype=torch.long)
base = bpb(model, ids, 16, "tensor")
print(f"# {CKPT}  ({len(ids)} bytes, {(len(ids)-1)//CTX} windows)  full bpb={base:.3f}\n")
print(f"{'bits':>5}  {'per-tensor':>22}  {'per-channel':>22}")
for b in (8, 4, 3, 2, 1):
    vt = bpb(model, ids, b, "tensor")
    vc = bpb(model, ids, b, "channel")
    print(f"{b:>5}  {vt:7.3f} ({(vt-base)/base*100:+6.1f}%)  {vc:7.3f} ({(vc-base)/base*100:+6.1f}%)")
print("\n(if per-channel keeps 2-4 bit usable, the wall was the QUANT method, not the idea)")
