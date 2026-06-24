"""Experiment F: constrained decoding to REAL words (trie) — kill the byte-garbage failure mode.

Why E failed: model entropy is backwards — the repetition collapse is LOW entropy (confident
garbage), real creativity is HIGH entropy. So confidence can't detect drift.

What every previous method leaked: byte-level garbage *inside* words ("finjobledemines",
"mumumumu") — strings that are not Portuguese words at all. D v1 only checks at word boundaries,
so it reacts too late.

F removes the failure mode by construction: build a trie of the corpus vocabulary and MASK the
next-byte distribution so the model can only ever extend a valid word prefix or terminate a
complete word. Non-words become impossible. Then (optionally) rerank whole words by coherence
(D v1's word-bigram) so they also come in a sensible ORDER.

Compares on the same 26M weights: baseline / guided(D v1) / constrained / constrained+guided.
"""
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/architecture_battery")
import exp_d_coherence_guided_decode as D  # noqa: E402
from src.model import ByteGPT  # noqa: E402

DEV = D.DEV

# ---- build the real-word trie (prefix set) from the corpus ----
_WORDRE = re.compile(r"[^\W\d_]+", re.UNICODE)   # runs of letters (incl. accents), any case
_wc = Counter(_WORDRE.findall(D.TXT))
WORDS = set()
PREFIX = set()
for w, c in _wc.items():
    if c >= 2:                                    # drop hapax/typos
        wb = w.encode("utf-8")
        WORDS.add(wb)
        for i in range(1, len(wb) + 1):
            PREFIX.add(wb[:i])
print(f"vocab: {len(WORDS)} palavras, {len(PREFIX)} prefixos", flush=True)


def _word_byte(b):
    return (97 <= b <= 122) or (65 <= b <= 90) or (b >= 128)   # ascii letter or utf-8 (accents)


def _current_word(out):
    i = len(out)
    while i > 0 and _word_byte(out[i - 1]):
        i -= 1
    return bytes(out[i:])


_NONWORD = [b for b in range(256) if not _word_byte(b)]
_WORDBYTES = [b for b in range(256) if _word_byte(b)]


def _allowed_mask(cur, device):
    m = torch.full((256,), float("-inf"), device=device)
    term_ok = (len(cur) == 0) or (cur in WORDS)
    if term_ok:
        for b in _NONWORD:
            m[b] = 0.0
    for b in _WORDBYTES:
        if cur + bytes([b]) in PREFIX:
            m[b] = 0.0
    if torch.isinf(m).all():
        m[32] = 0.0
    return m


@torch.no_grad()
def _logits(model, out, rep):
    x = torch.tensor([out], device=DEV)
    lo = model(x[:, -model.context:])[0, -1].clone()
    for b in set(out[-48:]):
        lo[b] /= rep
    return lo


def _sample(lo, temp, top_p):
    p = F.softmax(lo / temp, -1)
    sp, si = torch.sort(p, descending=True)
    keep = torch.cumsum(sp, 0) <= top_p
    keep[0] = True
    sp = sp * keep
    sp = sp / sp.sum()
    idx = torch.multinomial(sp, 1)
    return int(si[idx]), float(torch.log(sp[idx] + 1e-12))


@torch.no_grad()
def constrained_generate(model, n_bytes=500, seed="O ", temp=0.7, top_p=0.9, rep=1.4):
    out = list(seed.encode())
    while len(out) - len(seed.encode()) < n_bytes:
        lo = _logits(model, out, rep) + _allowed_mask(_current_word(out), DEV)
        b, _ = _sample(lo, temp, top_p)
        out.append(b)
    return bytes(out).decode("utf-8", "ignore")


@torch.no_grad()
def _constrained_word(model, out, temp, top_p, rep, max_len=22):
    local = list(out)
    draft, lps, had_word = [], [], False
    for _ in range(max_len):
        lo = _logits(model, local, rep) + _allowed_mask(_current_word(local), DEV)
        b, lp = _sample(lo, temp, top_p)
        draft.append(b); lps.append(lp); local.append(b)
        if _word_byte(b):
            had_word = True
        elif had_word:                 # emitted a boundary after a real word -> word complete
            break
    return draft, (float(np.mean(lps)) if lps else -20.0)


@torch.no_grad()
def constrained_guided_generate(model, n_bytes=500, seed="O ", temp=0.8, top_p=0.9, rep=1.4,
                                K=5, alpha=1.0):
    out = list(seed.encode())
    while len(out) - len(seed.encode()) < n_bytes:
        prev_words = D._W.findall(bytes(out).decode("utf-8", "ignore").lower())
        prev = prev_words[-1] if prev_words else ""
        cands = []
        for _ in range(K):
            wb, lp = _constrained_word(model, out, temp, top_p, rep)
            wtxt = bytes([c for c in wb if _word_byte(c)]).decode("utf-8", "ignore").lower()
            wm = D._W.findall(wtxt)
            surp = D.bigram_surprisal(prev, wm[0]) if wm else 8.0
            cands.append((lp - alpha * surp, wb))
        out += max(cands, key=lambda c: c[0])[1]
    return bytes(out).decode("utf-8", "ignore")


def main():
    model = ByteGPT().to(DEV)
    D.load_checkpoint(model, D.CKPT, DEV)
    model.eval()
    print(f"loaded {model.n_params/1e6:.0f}M | {D.CKPT}\n", flush=True)
    N = 5

    t0 = time.time(); base = [D.baseline_generate(model, 500) for _ in range(N)]
    t1 = time.time(); guided = [D.guided_generate(model, 500) for _ in range(N)]
    t2 = time.time(); con = [constrained_generate(model, 500) for _ in range(N)]
    t3 = time.time(); cong = [constrained_guided_generate(model, 500) for _ in range(N)]
    t4 = time.time()

    def stat(ss):
        return (float(np.mean([D.wtrans(s) for s in ss])), float(np.mean([D.coherent_span(s) for s in ss])))

    rows = [("BASELINE", base, t1 - t0), ("GUIADO (D v1)", guided, t2 - t1),
            ("CONSTRAINED", con, t3 - t2), ("CONSTRAINED+GUIADO", cong, t4 - t3)]
    print("\n============ RESULTADO (mesmos pesos 26M) ============", flush=True)
    for name, ss, dt in rows:
        w, s = stat(ss)
        print(f"{name:<20}| wtrans {w:.2f} | span {s:.1f} | {dt/N:.0f}s/amostra", flush=True)
    print(f"\nCONSTRAINED amostra:\n{con[0][:460]!r}", flush=True)
    print(f"\nCONSTRAINED+GUIADO amostra:\n{cong[0][:460]!r}", flush=True)


if __name__ == "__main__":
    main()
