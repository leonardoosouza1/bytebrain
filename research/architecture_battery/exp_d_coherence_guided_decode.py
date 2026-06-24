"""Architecture battery — Experiment D: coherence-guided decoding with backtracking.

Leonardo's hypothesis: the TRAINING is fine, the INFERENCE is the problem. A plain left-to-right
sampler "generates and throws" — it never checks whether what it just wrote is still coherent, so
it drifts into word-salad. We have spare compute (thousands of bytes/sec); spend it to VERIFY.

This script does NOT retrain anything. It loads the already-trained 26M flat checkpoint and compares
decoding strategies on the SAME weights:

  (A) baseline  — plain nucleus sampling (our current best).
  (B) guided    — at every word boundary, draft K candidate next-words, score each by
                  model fluency + coherence (its word-transition surprisal against the running
                  context, i.e. wtrans), keep the best. Back off (re-draft) when the best
                  candidate is still incoherent. This is "validate each step and redo if it
                  breaks harmony", and it naturally prefers attested constructions ("amarração").

If (B) extends the coherent span over (A) on the same weights, the bottleneck was inference.
"""
import math
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT, load_checkpoint  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
BASE = "/home/leonardo/projects/LLM/bytebrain"
CKPT = f"{BASE}/overnight_ck/loop.pt"   # the trained 26M flat model

# ---- coherence model (word-bigram) fit on the corpus ----
TXT = open(f"{BASE}/data/pt_clean.txt", encoding="utf-8").read()
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")
_words = _W.findall(TXT.lower())
WUNI, WBI, WV = Counter(_words), Counter(zip(_words, _words[1:])), len(set(_words))


def wtrans(text):
    w = _W.findall(text.lower())
    if len(w) < 3:
        return 12.0
    return float(np.mean([
        -math.log((WBI.get((w[i], w[i + 1]), 0) + 0.05) / (WUNI.get(w[i], 0) + 0.05 * WV))
        for i in range(len(w) - 1)
    ]))


def bigram_surprisal(prev_word, word):
    """-log P(word | prev_word): the coherence cost of following prev_word with word."""
    if not prev_word or not word:
        return 6.0
    return -math.log((WBI.get((prev_word, word), 0) + 0.05) / (WUNI.get(prev_word, 0) + 0.05 * WV))


# ---- decoding ----
@torch.no_grad()
def _next_byte(model, x, temp, top_p, rep):
    logit = model(x[:, -model.context:])[0, -1].clone()
    for b in set(x[0, -48:].tolist()):
        logit[b] /= rep
    pr = F.softmax(logit / temp, -1)
    sp, si = torch.sort(pr, descending=True)
    keep = torch.cumsum(sp, 0) <= top_p
    keep[0] = True
    sp = sp * keep
    sp = sp / sp.sum()
    idx = torch.multinomial(sp, 1)
    return int(si[idx]), float(torch.log(sp[idx] + 1e-12))


@torch.no_grad()
def baseline_generate(model, n_bytes=500, seed="O ", temp=0.6, top_p=0.85, rep=1.4):
    x = torch.tensor([list(seed.encode())], device=DEV)
    for _ in range(n_bytes):
        b, _ = _next_byte(model, x, temp, top_p, rep)
        x = torch.cat([x, torch.tensor([[b]], device=DEV)], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


@torch.no_grad()
def _draft_word(model, x, temp, top_p, rep, max_len=16):
    """Sample bytes until a word boundary; return (word_bytes, mean_logprob)."""
    bs, lps = [], []
    cur = x
    for _ in range(max_len):
        b, lp = _next_byte(model, cur, temp, top_p, rep)
        bs.append(b)
        lps.append(lp)
        cur = torch.cat([cur, torch.tensor([[b]], device=DEV)], 1)
        if b in (32, 10):  # space / newline = end of word
            break
    return bs, (float(np.mean(lps)) if lps else -20.0)


@torch.no_grad()
def guided_generate(model, n_bytes=500, seed="O ", temp=0.8, top_p=0.9, rep=1.4,
                    K=6, alpha=1.0, tau=9.0, max_backtrack=2):
    """Coherence-guided word-by-word decoding with backtracking.
    K drafts per word; score = fluency(mean logprob) - alpha * bigram_surprisal(prev_word, cand).
    If the best candidate's surprisal is still > tau, back off and re-draft (up to max_backtrack)."""
    x = torch.tensor([list(seed.encode())], device=DEV)

    def last_word():
        m = list(_W.finditer(bytes(x[0].tolist()).decode("utf-8", "ignore").lower()))
        return m[-1].group() if m else ""

    produced = 0
    while produced < n_bytes:
        prev = last_word()
        best = None
        for _bt in range(max_backtrack + 1):
            cands = []
            for _ in range(K):
                wb, lp = _draft_word(model, x, temp, top_p, rep)
                wtxt = bytes([c for c in wb if c not in (32, 10)]).decode("utf-8", "ignore").lower()
                wm = _W.findall(wtxt)
                surp = bigram_surprisal(prev, wm[0]) if wm else 8.0
                cands.append((lp - alpha * surp, surp, wb))
            best = max(cands, key=lambda c: c[0])
            if best[1] <= tau:        # coherent enough -> accept
                break
            temp = max(0.5, temp - 0.1)  # tighten and retry (back off)
        wb = best[2]
        x = torch.cat([x, torch.tensor([wb], device=DEV)], 1)
        produced += len(wb)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def coherent_span(text, window=6, thresh=8.5):
    """How many words until a sliding wtrans window first exceeds thresh (the drift point)."""
    w = _W.findall(text.lower())
    for end in range(window, len(w)):
        seg = " ".join(w[end - window:end])
        if wtrans(seg) > thresh:
            return end
    return len(w)


def main():
    model = ByteGPT().to(DEV)
    load_checkpoint(model, CKPT, DEV)
    model.eval()
    print(f"loaded {model.n_params/1e6:.0f}M | {CKPT}\n", flush=True)

    t0 = time.time()
    base = [baseline_generate(model, 500) for _ in range(6)]
    t1 = time.time()
    guided = [guided_generate(model, 500) for _ in range(6)]
    t2 = time.time()

    wb = float(np.mean([wtrans(s) for s in base]))
    wg = float(np.mean([wtrans(s) for s in guided]))
    sb = float(np.mean([coherent_span(s) for s in base]))
    sg = float(np.mean([coherent_span(s) for s in guided]))

    print("================ RESULTADO (mesmos pesos, so muda a inferencia) ================", flush=True)
    print(f"BASELINE (sampling) | wtrans {wb:.2f} | span coerente {sb:.1f} palavras | {t1-t0:.0f}s", flush=True)
    print(f"GUIADO   (coesao+bt)| wtrans {wg:.2f} | span coerente {sg:.1f} palavras | {t2-t1:.0f}s", flush=True)
    print(f"\nBASELINE amostra:\n{base[0][:400]!r}", flush=True)
    print(f"\nGUIADO amostra:\n{guided[0][:400]!r}", flush=True)
    verdict = "GUIADO VENCE (inferencia era o gargalo)" if (wg < wb - 0.3 or sg > sb + 2) else "sem ganho claro"
    print(f"\n=> {verdict} | delta wtrans {wb-wg:+.2f} | delta span {sg-sb:+.1f}", flush=True)


if __name__ == "__main__":
    main()
