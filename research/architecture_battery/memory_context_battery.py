"""Battery for the "memory seeds" idea: can an external anchor keep LONG generation on-topic?

A plain sliding window (the model only sees the last 256 bytes) drifts off-topic over a long text.
Leonardo's idea: carry a persistent global anchor ("seeds") so it stays on the thread. We test the
v1 (textual seeds, NO retraining) version on the GPU:

  PLAIN     - context = last 256 bytes only (baseline; should drift)
  PLAN      - context = [plan keywords] + recent bytes (persistent topic anchor)
  PLAN+SUM  - context = [plan keywords] + [rolling summary of recent content] + recent bytes

Metric: topic retention = average PMI of the LATE half's content words with the plan (does the
text still talk about the seed topic 600 bytes later?). Higher = stayed on topic. Model runs on
GPU; only light dict lookups touch the CPU.
"""
import math
import re
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT, load_checkpoint  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
BASE = "/home/leonardo/projects/LLM/bytebrain"
CKPT = f"{BASE}/overnight_ck/loop.pt"        # 26M, ctx256 — the better-trained demonstrator
CTX = 256
GEN = 140                                    # SHORT (within the coherent span) to isolate the anchor effect
STOP = set("que nao para uma com de do da em os as ele ela mais como foi sao seu sua por isso este "
           "esta quando porque tambem entre sobre ser ter sem a o e um na no se mas ou aos das dos "
           "ao pelo pela seus suas dele dela qual onde era tem ha".split())
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")

print("montando modelo de topico (PMI) + wtrans de uma amostra...", flush=True)
ref = open(f"{BASE}/data/pt_big.txt", encoding="utf-8", errors="ignore").read(15_000_000)
words = _W.findall(ref.lower())
content = [w for w in words if w not in STOP and len(w) > 3]
FREQ = Counter(content)
TOTAL = len(content)
TOP = set(w for w, _ in FREQ.most_common(6000))
COOC = Counter()
for i, w in enumerate(content):
    if w in TOP:
        for j in range(max(0, i - 6), min(len(content), i + 7)):
            if j != i and content[j] in TOP:
                COOC[(w, content[j])] += 1
WUNI = Counter(words)
WBI = Counter(zip(words, words[1:]))
WV = len(WUNI)
del ref, words, content


def pmi(a, b):
    c = COOC.get((a, b), 0)
    if not c:
        return 0.0
    return math.log((c / TOTAL) / ((FREQ[a] / TOTAL) * (FREQ[b] / TOTAL)) + 1e-9)


def wtrans(t):
    w = _W.findall(t.lower())
    if len(w) < 3:
        return 12.0
    return float(np.mean([-math.log((WBI.get((w[i], w[i + 1]), 0) + 0.05) / (WUNI.get(w[i], 0) + 0.05 * WV))
                          for i in range(len(w) - 1)]))


def topic_retention(text, plan):
    cw = [w for w in _W.findall(text.lower()) if w in TOP]
    pw = [w for w in plan if w in TOP]
    if not cw or not pw:
        return 0.0
    return float(np.mean([max(pmi(w, p) for p in pw) for w in cw]))


@torch.no_grad()
def _chunk(model, ctx_bytes, n, temp=0.7, top_p=0.9, rep=1.4):
    x = torch.tensor([ctx_bytes], device=DEV)
    out = []
    for _ in range(n):
        lo = model(x[:, -CTX:])[0, -1].clone()
        for b in set(x[0, -48:].tolist()):
            lo[b] /= rep
        p = F.softmax(lo / temp, -1)
        sp, si = torch.sort(p, descending=True)
        keep = torch.cumsum(sp, 0) <= top_p
        keep[0] = True
        sp = sp * keep
        sp = sp / sp.sum()
        nb = int(si[torch.multinomial(sp, 1)])
        out.append(nb)
        x = torch.cat([x, torch.tensor([[nb]], device=DEV)], 1)
    return out


def _plan_prefix(plan):
    return (" ".join(plan[:6]) + ". ").encode("utf-8")


def _sum_prefix(sumw):
    return (" ".join(sumw[-8:]) + ". ").encode("utf-8") if sumw else b""


def long_gen(model, seed, plan, mode):
    out = list(seed.encode("utf-8"))
    base = len(out)
    sumw = []
    while len(out) - base < GEN:
        if mode == "plain":
            ctx = out[-CTX:]
        elif mode == "plan":
            pre = list(_plan_prefix(plan))
            ctx = pre + out[-(CTX - len(pre)):]
        else:  # plan+sum
            pre = list(_plan_prefix(plan)) + list(_sum_prefix(sumw))
            ctx = pre + out[-(CTX - len(pre)):]
        out += _chunk(model, ctx, 24)
        recent = bytes(out[-200:]).decode("utf-8", "ignore")
        rc = [w for w in _W.findall(recent.lower()) if w in TOP]
        sumw = [w for w, _ in Counter(rc).most_common(6)]
    return bytes(out).decode("utf-8", "ignore")


SEEDS = [
    "A astronomia estuda as estrelas, os planetas e o universo. ",
    "A Segunda Guerra Mundial foi um conflito militar global. ",
    "As celulas sao as unidades basicas dos seres vivos na biologia. ",
]


def main():
    model = ByteGPT(dim=512, n_layers=8, n_heads=8, context=256).to(DEV)
    load_checkpoint(model, CKPT, DEV)
    model.eval()
    print(f"modelo {model.n_params/1e6:.0f}M | gerando {GEN} bytes/run | {len(SEEDS)} topicos\n", flush=True)

    rows = {"plain": [], "plan": [], "plan+sum": []}
    samples = {}
    for seed in SEEDS:
        plan = [w for w in _W.findall(seed.lower()) if w in TOP][:6]
        for mode in ("plain", "plan", "plan+sum"):
            txt = long_gen(model, seed, plan, mode)
            cont = txt[len(seed):]                # the generated continuation
            rows[mode].append((topic_retention(cont, plan), wtrans(cont)))
            samples.setdefault(mode, txt)

    print("======== AS SEMENTES MANTEM O TOPICO? (texto longo, 2a metade) ========", flush=True)
    for mode in ("plain", "plan", "plan+sum"):
        tr = np.mean([r[0] for r in rows[mode]])
        wt = np.mean([r[1] for r in rows[mode]])
        print(f"{mode:>9} | retencao de topico (PMI 2a metade) {tr:.2f} | wtrans {wt:.2f}", flush=True)
    print(f"\n[PLAIN] 2a metade:\n{samples['plain'][GEN//2:GEN//2+300]!r}", flush=True)
    print(f"\n[PLAN+SUM] 2a metade:\n{samples['plan+sum'][GEN//2:GEN//2+300]!r}", flush=True)


if __name__ == "__main__":
    main()
