"""Does increasing the corpus actually pay off? — clean scaling test before committing to 1GB.

Same source (Wikipedia), same model (~26M, ctx256), same EQUAL training time, evaluated on the SAME
held-out set and the SAME coherence oracle. The only thing that changes is how much data the model
trains on: 5MB -> 80MB -> 843MB. If held-out bits/byte keeps dropping and the coherent span keeps
growing as the corpus grows, more data pays off (go to 1GB+). If it plateaus, we already have enough.
"""
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT  # noqa: E402
from src.coherence import WordTransition  # noqa: E402
from src.sample import generate, coherence_guided_generate  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
BASE = "/home/leonardo/projects/LLM/bytebrain"
TIME_S = int(os.environ.get("CS_TIME", "420"))   # seconds per corpus size
CTX, DIM, LAYERS, BATCH = 256, 512, 8, 96

FULL = np.memmap(f"{BASE}/data/pt_mid.txt", dtype=np.uint8, mode="r")
HELD = np.asarray(FULL[-1_000_000:]).copy()        # common held-out (real Wikipedia PT), never trained on
SIZES = [("5MB", 5_000_000), ("80MB", 80_000_000), ("843MB", len(FULL) - 1_000_000)]
print(f"corpus base {len(FULL)/1e6:.0f}MB | {TIME_S}s por tamanho | held-out comum {len(HELD)/1e6:.1f}MB", flush=True)

with open(f"{BASE}/data/pt_mid.txt", encoding="utf-8", errors="ignore") as f:
    ref = f.read(15_000_000)
COH = WordTransition(ref)
del ref
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")


def span(t, window=6, thresh=8.5):
    w = _W.findall(t.lower())
    for e in range(window, len(w)):
        if COH.score(" ".join(w[e - window:e])) > thresh:
            return e
    return len(w)


def get_batch(arr, n, bs, L):
    ix = np.random.randint(0, n - L - 1, bs)
    return torch.from_numpy(np.stack([np.asarray(arr[i:i + L + 1]) for i in ix]).astype(np.int64)).to(DEV)


def held_bpb(m):
    m.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(30):
            ix = np.random.randint(0, len(HELD) - CTX - 1, 16)
            xb = torch.from_numpy(np.stack([HELD[i:i + CTX + 1] for i in ix]).astype(np.int64)).to(DEV)
            tot += F.cross_entropy(m(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1)).item()
    return tot / 30 / math.log(2)


def run(size_bytes, name):
    m = ByteGPT(dim=DIM, n_layers=LAYERS, context=CTX).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
    m.train()
    t0 = time.time()
    step = 0
    while time.time() - t0 < TIME_S:
        step += 1
        xb = get_batch(FULL, size_bytes, BATCH, CTX)
        loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    hb = held_bpb(m)
    tb = 0.0
    with torch.no_grad():
        for _ in range(20):
            xb = get_batch(FULL, size_bytes, 16, CTX)
            tb += F.cross_entropy(m(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1)).item()
    tb = tb / 20 / math.log(2)
    epochs = step * BATCH * CTX / size_bytes
    base = [generate(m, n=400, device=DEV) for _ in range(4)]
    guid = [coherence_guided_generate(m, COH, n=400, device=DEV) for _ in range(4)]
    return dict(name=name, steps=step, epochs=epochs, held=hb, train=tb,
                base_span=float(np.mean([span(s) for s in base])),
                guid_w=float(np.mean([COH.score(s) for s in guid])),
                guid_span=float(np.mean([span(s) for s in guid])),
                sample=guid[0][:300])


def main():
    res = [run(n, name) for name, n in SIZES]
    print("\n============ COMPENSA AUMENTAR O CORPUS? (mesmo modelo/tempo/held-out) ============", flush=True)
    for r in res:
        gap = r['held'] - r['train']
        print(f"{r['name']:>6} | {r['epochs']:.1f} epocas | train_bpb {r['train']:.3f} | HELD-OUT bpb {r['held']:.3f} "
              f"| gap {gap:+.3f}{'  <- OVERFIT' if gap > 0.5 else ''} | GUIADO span {r['guid_span']:.0f} (w{r['guid_w']:.1f})", flush=True)
    print(f"\n843MB GUIADO amostra:\n{res[-1]['sample']!r}", flush=True)
    print(f"\n=> held-out bpb: {res[0]['held']:.3f} (5MB) -> {res[-1]['held']:.3f} (843MB) "
          f"| span guiado: {res[0]['guid_span']:.0f} -> {res[-1]['guid_span']:.0f}", flush=True)


if __name__ == "__main__":
    main()
