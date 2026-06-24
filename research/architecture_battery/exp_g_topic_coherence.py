"""Experiment G: long-range (topic) coherence verifier.

F showed the remaining failure is not byte-garbage but WORD ORDER over distance: real words in
salad order ("fingiu startups école-jordaniano fadas"). D v1's verifier is a word-BIGRAM — it only
checks the immediately preceding word (range 1), so each adjacent pair looks fine while the whole
drifts off topic.

G adds a long-range signal: does the candidate word fit the TOPIC of the last ~12 content words?
We precompute word co-occurrence (PMI) from the corpus, and score candidates by their average PMI
with the recent content words. Decoding then prefers words that are both locally fluent (model),
locally ordered (bigram) AND topically consistent (PMI) over a long window.

Compares on the same 26M weights: baseline / guided(D v1, bigram) / guided+topic.
"""
import math
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/architecture_battery")
import exp_d_coherence_guided_decode as D  # noqa: E402
from src.model import ByteGPT  # noqa: E402

DEV = D.DEV
STOP = set("que nao para uma com de do da em os as ele ela mais como foi sao seu sua por isso este "
           "esta quando porque tambem entre sobre ser ter sem a o e um na no se mas ou aos das dos "
           "ao pelo pela seus suas me te lhe nos lhes ja ainda muito todo toda todos todas dele dela "
           "qual onde foi era sao tem ha".split())


class TopicCoherence:
    def __init__(self, text, vocab_size=6000, window=6):
        words = D._W.findall(text.lower())
        content = [w for w in words if w not in STOP and len(w) > 3]
        self.freq = Counter(content)
        self.total = len(content)
        self.top = set(w for w, _ in self.freq.most_common(vocab_size))
        cooc = Counter()
        for i, w in enumerate(content):
            if w not in self.top:
                continue
            for j in range(max(0, i - window), min(len(content), i + window + 1)):
                if j != i and content[j] in self.top:
                    cooc[(w, content[j])] += 1
        self.cooc = cooc
        print(f"topic model: {len(self.top)} content words, {len(cooc)} co-occurrence pairs", flush=True)

    def pmi(self, a, b):
        cab = self.cooc.get((a, b), 0)
        if cab == 0:
            return 0.0
        pa, pb = self.freq[a] / self.total, self.freq[b] / self.total
        return math.log((cab / self.total) / (pa * pb) + 1e-9)

    def score(self, cand, recent):
        if cand not in self.top:
            return 0.0
        rs = [r for r in recent if r in self.top]
        return float(np.mean([self.pmi(cand, r) for r in rs])) if rs else 0.0


@torch.no_grad()
def guided_topic_generate(model, topic, n_bytes=500, seed="O ", temp=0.8, top_p=0.9, rep=1.4,
                          K=6, alpha=1.0, beta=0.6, win=12):
    x = torch.tensor([list(seed.encode())], device=DEV)
    while x.size(1) - len(seed.encode()) < n_bytes:
        txt = bytes(x[0].tolist()).decode("utf-8", "ignore")
        words = D._W.findall(txt.lower())
        prev = words[-1] if words else ""
        recent = [w for w in words[-win:] if w not in STOP]
        cands = []
        for _ in range(K):
            wb, lp = D._draft_word(model, x, temp, top_p, rep)
            wtxt = bytes([c for c in wb if c not in (32, 10)]).decode("utf-8", "ignore").lower()
            wm = D._W.findall(wtxt)
            cw = wm[0] if wm else ""
            surp = D.bigram_surprisal(prev, cw) if cw else 8.0
            topic_s = topic.score(cw, recent) if cw else 0.0
            cands.append((lp - alpha * surp + beta * topic_s, wb))
        x = torch.cat([x, torch.tensor([max(cands, key=lambda c: c[0])[1]], device=DEV)], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def main():
    model = ByteGPT().to(DEV)
    D.load_checkpoint(model, D.CKPT, DEV)
    model.eval()
    print(f"loaded {model.n_params/1e6:.0f}M | {D.CKPT}\n", flush=True)
    topic = TopicCoherence(D.TXT)
    N = 5

    t0 = time.time(); base = [D.baseline_generate(model, 500) for _ in range(N)]
    t1 = time.time(); guided = [D.guided_generate(model, 500) for _ in range(N)]
    t2 = time.time(); gtopic = [guided_topic_generate(model, topic, 500) for _ in range(N)]
    t3 = time.time()

    def stat(ss):
        return (float(np.mean([D.wtrans(s) for s in ss])), float(np.mean([D.coherent_span(s) for s in ss])))

    for name, ss, dt in [("BASELINE", base, t1 - t0), ("GUIADO bigrama (D v1)", guided, t2 - t1),
                         ("GUIADO + topico (G)", gtopic, t3 - t2)]:
        w, s = stat(ss)
        print(f"{name:<24}| wtrans {w:.2f} | span {s:.1f} | {dt/N:.0f}s/amostra", flush=True)
    print(f"\nGUIADO+TOPICO amostra:\n{gtopic[0][:480]!r}", flush=True)


if __name__ == "__main__":
    main()
