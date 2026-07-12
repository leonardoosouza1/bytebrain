#!/usr/bin/env python3
"""Does QWEN's weights encode byte bit-structure? Qwen uses byte-level BPE → its
vocab has 256 single-byte tokens. Extract THEIR embeddings (256×1536) and run the
SAME Walsh-Hadamard analysis we ran on ByteBrain (bit5=case, bit6=region, bit7=UTF8).
If a 1.5B external BPE model also ranks those semantic bits at the top, it validates
the byte-bit thesis beyond ByteBrain. CPU only (no GPU contention with training)."""
import sys, json, numpy as np
from safetensors import safe_open

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"

def bytes_to_unicode():                      # GPT-2/Qwen byte-level BPE byte→char map
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

vocab = json.load(open(f"{MODEL}/vocab.json"))
b2u = bytes_to_unicode()
ids = []; missing = 0
for b in range(256):
    tid = vocab.get(b2u[b])
    if tid is None: missing += 1; tid = 0
    ids.append(tid)
print(f"# {MODEL}")
print(f"byte→token map: {256-missing}/256 found (missing {missing})")

# embedding (tied with lm_head in Qwen): model.embed_tokens.weight
with safe_open(f"{MODEL}/model.safetensors", framework="pt") as f:
    keys = list(f.keys())
    ekey = "model.embed_tokens.weight" if "model.embed_tokens.weight" in keys else [k for k in keys if "embed" in k][0]
    emb = f.get_tensor(ekey).float().numpy()
print(f"embedding {ekey}: {emb.shape}")
E = emb[ids]                                  # [256, D] byte-token embeddings

# Sylvester-Hadamard: H[w,b] = (-1)^popcount(w & b)
def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H
H = hadamard(256)
W = H @ E                                      # [256, D] Walsh spectrum over byte index
energy = (W ** 2).sum(1); energy[0] = 0.0      # drop DC
order = np.argsort(-energy)
rank = {w: int(np.where(order == w)[0][0]) for w in [1, 2, 4, 8, 16, 32, 64, 128]}
tot = energy.sum()
print("single-bit Walsh component ranks (of 255, lower=more structured):")
for k in range(8):
    w = 1 << k
    tag = {5: " = CASE", 6: " = region", 7: " = UTF8-marker"}.get(k, "")
    print(f"  bit{k} (w={w:3d}): rank {rank[w]:3d}   energy {energy[w]/tot*100:5.2f}%{tag}")
sb = sum(energy[1 << k] for k in range(8)) / tot * 100
print(f"single-bit total energy: {sb:.2f}% (uniform≈3.1%)")
print(f"SUMMARY ranks  bit5(case)={rank[32]}  bit6(region)={rank[64]}  bit7(UTF8)={rank[128]}")

# case probe: do bit5-flipped pairs (a/A) sit close? cos of mean case-direction
pairs = [(ord(c), ord(c) - 32) for c in "abcdefghijklmnopqrstuvwxyz"]   # lower,UPPER
dirs = np.array([E[lo] - E[up] for lo, up in pairs])
dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
mean_dir = dirs.mean(0); mean_dir /= np.linalg.norm(mean_dir) + 1e-9
coh = (dirs @ mean_dir).mean()                # 1.0 = all case-flips point same way
print(f"case-direction coherence (a→A consistent?): {coh:.3f}  (1=perfectly aligned)")
