#!/usr/bin/env python3
"""Interpretability probe #1 — attention RANGE. Opens the black box to answer
"quais grafos se foram?": how far back does each layer actually attend?
If attention is mostly LOCAL (small lookback << context), the model has no
long-range topic memory -> explains the drift AND why sparsity was free.

CPU, runs on the dense-L1024 checkpoint (doesn't touch the GPU training).
"""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant

CK = sys.argv[1] if len(sys.argv) > 1 else "ckpt_ovn_dense/ckpt_best.pt"
ck = torch.load(CK, map_location="cpu", weights_only=False); c = ck["config"]
set_act_quant(0)
m = GraphByteGPT(c["dim"], c["layers"], c["heads"], c["ctx"], topk=0)
m.load_state_dict(ck["model"]); m.eval()

# monkeypatch SDPA to record per-layer mean attention lookback + % local
records = []
orig = F.scaled_dot_product_attention
def patched(q, k, v, is_causal=False, dropout_p=0.0, **kw):
    B, H, L, d = q.shape
    s = (q @ k.transpose(-2, -1)) * (d ** -0.5)
    s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool), 1), float("-inf"))
    a = s.softmax(-1)                                   # [B,H,L,L]
    idx = torch.arange(L)
    dist = (idx[:, None] - idx[None, :]).clamp(min=0).float()   # i-j (lookback)
    lookback = (a * dist[None, None]).sum(-1)           # [B,H,L] mean lookback/query
    local = (a * (dist[None, None] <= 8).float()).sum(-1)        # mass within 8 bytes
    sink = a[:, :, 16:, :4].sum(-1)                     # mass on bytes 0-3 (sink), queries>=16
    ent = -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(-1)  # attention entropy/query
    eff = ent.exp()                                     # effective # of attended positions
    records.append((lookback.mean().item(), local.mean().item(), sink.mean().item(), eff[:, :, 16:].mean().item()))
    return a @ v
F.scaled_dot_product_attention = patched

text = ("O Brasil é um país de dimensões continentais, com grande diversidade cultural "
        "e natural. A floresta amazônica abriga milhões de espécies. ") * 12
ids = torch.tensor([list(text.encode("utf-8"))[:c["ctx"]]])
with torch.no_grad():
    m(ids)
F.scaled_dot_product_attention = orig

L = ids.shape[1]
print(f"# {CK} | context = {L} bytes | {c['layers']} layers x {c['heads']} heads\n")
print(f"{'layer':>5}  {'lookback':>9}  {'%local<=8':>9}  {'%sink':>6}  {'eff#attended':>12}")
for i, (lb, loc, sk, eff) in enumerate(records):
    print(f"{i:>5}  {lb:>6.0f} by  {loc*100:>8.0f}%  {sk*100:>5.0f}%  {eff:>10.0f}")
avg = sum(r[0] for r in records) / len(records)
print(f"\noverall mean lookback = {avg:.1f} bytes (of {L})")
print("→ se o lookback longo (layers 0/1/7) tiver %sink alto = é SINK (posição 0), não tópico real")
