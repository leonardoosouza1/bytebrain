"""Quick effectiveness check BEFORE committing to the multi-day run.

Validates, on the real 1.3GB Wikipedia corpus, that the levers we plan to use actually help:
  * longer context (256 -> 512 bytes): the main un-validated coherence lever;
  * coherence-guided decoding (D v1): re-confirm it helps on a big-corpus model;
  * pipeline / throughput / OOM: does train run on the big corpus, and how many tok/s
    (so we can estimate the real run time)?

Trains the same ~26M model at ctx 256 vs 512 for an equal SHORT time budget and evaluates each
with plain vs guided decoding. Reference point: the current small-corpus 26M scored val ~1.99,
baseline span ~16 words, guided span ~32.
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
CORPUS = f"{BASE}/data/pt_big.txt"
TIME_S = int(os.environ.get("QV_TIME", "700"))   # seconds per config

data = np.memmap(CORPUS, dtype=np.uint8, mode="r")
cut = int(len(data) * 0.999)
TR, VA = data[:cut], data[cut:]
print(f"corpus {len(data)/1e6:.0f}MB | {TIME_S}s por config", flush=True)

with open(CORPUS, encoding="utf-8", errors="ignore") as f:
    ref = f.read(20_000_000)
COH = WordTransition(ref)
del ref
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")


def coherent_span(text, window=6, thresh=8.5):
    w = _W.findall(text.lower())
    for end in range(window, len(w)):
        if COH.score(" ".join(w[end - window:end])) > thresh:
            return end
    return len(w)


def get_batch(arr, bs, L):
    ix = np.random.randint(0, len(arr) - L - 1, bs)
    return torch.from_numpy(np.stack([np.asarray(arr[i:i + L + 1]) for i in ix]).astype(np.int64)).to(DEV)


def run(ctx, dim=512, layers=8):
    batch = 24576 // ctx                       # equal tokens/step across configs
    m = ByteGPT(dim=dim, n_layers=layers, context=ctx).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
    m.train()
    t0 = time.time()
    step = 0
    while time.time() - t0 < TIME_S:
        step += 1
        xb = get_batch(TR, batch, ctx)
        loss = F.cross_entropy(m(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"  [ctx{ctx}] step {step} bpb {loss.item()/math.log(2):.2f} ({time.time()-t0:.0f}s)", flush=True)
    tok_s = step * batch * ctx / (time.time() - t0)

    m.eval()
    vb = 0.0
    with torch.no_grad():
        for _ in range(20):
            xb = get_batch(VA, 16, ctx)
            vb += F.cross_entropy(m(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1)).item()
    vb = vb / 20 / math.log(2)
    base = [generate(m, n=400, device=DEV) for _ in range(4)]
    guid = [coherence_guided_generate(m, COH, n=400, device=DEV) for _ in range(4)]
    bw = float(np.mean([COH.score(s) for s in base]))
    bs = float(np.mean([coherent_span(s) for s in base]))
    gw = float(np.mean([COH.score(s) for s in guid]))
    gs = float(np.mean([coherent_span(s) for s in guid]))
    return dict(ctx=ctx, params=m.n_params / 1e6, steps=step, tok_s=tok_s, val=vb,
                base_w=bw, base_span=bs, guid_w=gw, guid_span=gs, sample=guid[0][:320])


def main():
    res = [run(256), run(512)]
    print("\n============ VALIDACAO (corpus real 1.3GB) ============", flush=True)
    for r in res:
        print(f"ctx{r['ctx']} {r['params']:.0f}M | {r['steps']} steps {r['tok_s']/1000:.0f}k tok/s | "
              f"val {r['val']:.3f} | baseline span {r['base_span']:.0f} (w{r['base_w']:.1f}) | "
              f"GUIADO span {r['guid_span']:.0f} (w{r['guid_w']:.1f})", flush=True)
    print(f"\nctx512 GUIADO amostra:\n{res[1]['sample']!r}", flush=True)
    a, b = res
    print(f"\n=> contexto longo ajuda? span guiado {a['guid_span']:.0f} (256) vs {b['guid_span']:.0f} (512)", flush=True)
    print(f"=> guiado ajuda? span {b['base_span']:.0f} -> {b['guid_span']:.0f} (ctx512)", flush=True)


if __name__ == "__main__":
    main()
