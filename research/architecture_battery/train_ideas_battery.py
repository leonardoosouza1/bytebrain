"""Battery: 3 training-time ideas to inject a 'good/bad' signal the plain loss can't.

  1. UNLIKELIHOOD   - besides making the real next byte likely, explicitly push DOWN the probability
                      of repeating recent bytes (attacks the collapse/rambling directly).
  2. LONGER CONTEXT - ctx256 vs ctx512: does a longer window force more global coherence?
  3. PROFESSOR SADICO (discriminator) - can a learned judge tell real prose from word-salad? If yes
                      it could become a critic/reward; if it can't, the idea is dead on arrival.

Small models, equal short time, on the real 1.3GB corpus. GPU only; light CPU.
"""
import math
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
BASE = "/home/leonardo/projects/LLM/bytebrain"
T_TRAIN = 240
data = np.memmap(f"{BASE}/data/pt_big.txt", dtype=np.uint8, mode="r")
TR = data[:int(len(data) * 0.999)]
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")
ref = open(f"{BASE}/data/pt_big.txt", encoding="utf-8", errors="ignore").read(10_000_000)
rw = _W.findall(ref.lower())
WUNI, WBI, WV = Counter(rw), Counter(zip(rw, rw[1:])), len(set(rw))
del ref, rw


def wtrans(t):
    w = _W.findall(t.lower())
    if len(w) < 3:
        return 12.0
    return float(np.mean([-math.log((WBI.get((w[i], w[i + 1]), 0) + 0.05) / (WUNI.get(w[i], 0) + 0.05 * WV))
                          for i in range(len(w) - 1)]))


def span(t, win=6, th=8.5):
    w = _W.findall(t.lower())
    for e in range(win, len(w)):
        if wtrans(" ".join(w[e - win:e])) > th:
            return e
    return len(w)


def get_batch(bs, L):
    ix = np.random.randint(0, len(TR) - L - 1, bs)
    return torch.from_numpy(np.stack([np.asarray(TR[i:i + L + 1]) for i in ix]).astype(np.int64)).to(DEV)


@torch.no_grad()
def gen(m, ctx, n=300, temp=0.8, top_p=0.92, rep=1.3):
    x = torch.tensor([list("O ".encode())], device=DEV)
    for _ in range(n):
        lo = m(x[:, -ctx:])[0, -1].clone()
        for b in set(x[0, -32:].tolist()):
            lo[b] /= rep
        p = F.softmax(lo / temp, -1)
        sp, si = torch.sort(p, descending=True)
        keep = torch.cumsum(sp, 0) <= top_p
        keep[0] = True
        sp = sp * keep
        sp = sp / sp.sum()
        x = torch.cat([x, si[torch.multinomial(sp, 1)].view(1, 1)], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def distinct4(t):
    b = t.encode()
    g = [bytes(b[i:i + 4]) for i in range(len(b) - 4)]
    return len(set(g)) / max(1, len(g))     # low = repetitive


def train_lm(ctx, unlikelihood):
    m = ByteGPT(dim=384, n_layers=6, n_heads=8, context=ctx).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.05)
    m.train()
    t0 = time.time()
    while time.time() - t0 < T_TRAIN:
        xb = get_batch(40, ctx)
        logits = m(xb[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 256), xb[:, 1:].reshape(-1))
        loss = ce
        if unlikelihood:
            tgt, cx = xb[:, 1:], xb[:, :-1]
            Bn, L = tgt.shape
            neg = torch.zeros(Bn, L, 256, device=DEV, dtype=torch.bool)
            ar = torch.arange(L, device=DEV)
            for w in range(0, 24):
                neg.scatter_(2, cx[:, (ar - w).clamp(min=0)].unsqueeze(-1), True)
            neg.scatter_(2, tgt.unsqueeze(-1), False)       # the true target is NOT a negative
            p = F.softmax(logits, -1)
            ul = -(torch.log(1 - p + 1e-6) * neg).sum(-1).mean()
            loss = ce + 0.5 * ul
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


def eval_lm(m, ctx):
    s = [gen(m, ctx, 300) for _ in range(4)]
    return float(np.mean([distinct4(x) for x in s])), float(np.mean([wtrans(x) for x in s])), \
        float(np.mean([span(x) for x in s])), s[0]


# ---------- Professor sadico: discriminator ----------
class Disc(nn.Module):
    def __init__(s, d=64):
        super().__init__()
        s.e = nn.Embedding(256, d)
        s.c = nn.Sequential(nn.Conv1d(d, 96, 5, padding=2), nn.GELU(), nn.Conv1d(96, 96, 5, padding=2), nn.GELU())
        s.h = nn.Linear(96, 1)

    def forward(s, x):
        h = s.e(x).transpose(1, 2)
        h = s.c(h).mean(-1)
        return s.h(h).squeeze(-1)


def train_disc(lm):
    d = Disc().to(DEV)
    opt = torch.optim.AdamW(d.parameters(), 3e-4)
    txt = open(f"{BASE}/data/pt_big.txt", encoding="utf-8", errors="ignore").read(3_000_000)
    paras = [p for p in txt.split("\n\n") if len(p) > 220]

    def real_batch(bs):
        return [paras[np.random.randint(len(paras))][:200] for _ in range(bs)]

    def corrupt(t):                                          # word-salad: real words, scrambled order
        w = t.split()
        np.random.shuffle(w)
        return " ".join(w)

    def to_t(strs):
        return torch.from_numpy(np.stack([np.frombuffer((s + " " * 200).encode()[:200], np.uint8) for s in strs]).astype(np.int64)).to(DEV)
    d.train()
    for _ in range(1200):
        r = real_batch(16)
        bad = [corrupt(x) for x in r]
        x = to_t(r + bad)
        y = torch.tensor([1.0] * 16 + [0.0] * 16, device=DEV)
        loss = F.binary_cross_entropy_with_logits(d(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    d.eval()
    with torch.no_grad():
        r = real_batch(64)
        acc_real = (torch.sigmoid(d(to_t(r))) > 0.5).float().mean().item()
        acc_salad = (torch.sigmoid(d(to_t([corrupt(x) for x in r]))) < 0.5).float().mean().item()
        model_txt = [gen(lm, 256, 196) for _ in range(20)]
        p_model = torch.sigmoid(d(to_t(model_txt))).mean().item()
    return acc_real, acc_salad, p_model


def main():
    print(f"battery 3 ideias | {T_TRAIN}s/treino\n", flush=True)
    print("=== IDEIA 1 (unlikelihood) + IDEIA 2 (contexto) ===", flush=True)
    rows = []
    for name, ctx, ul in [("ctx256 CE (base)", 256, False), ("ctx256 +UNLIKELIHOOD", 256, True), ("ctx512 CE", 512, False)]:
        m = train_lm(ctx, ul)
        d4, wt, sp, smp = eval_lm(m, ctx)
        rows.append((name, d4, wt, sp, smp))
        if name == "ctx256 CE (base)":
            base_m = m
        print(f"{name:<22} | distinct4 {d4:.2f} (alto=menos repeticao) | wtrans {wt:.2f} | span {sp:.0f}", flush=True)
    print(f"\n[base] amostra:\n{rows[0][4][:200]!r}", flush=True)
    print(f"[unlikelihood] amostra:\n{rows[1][4][:200]!r}", flush=True)

    print("\n=== IDEIA 3 (professor sadico / discriminador) ===", flush=True)
    ar, asal, pm = train_disc(base_m)
    print(f"acc em texto REAL {ar*100:.0f}% | acc em SALADA {asal*100:.0f}% | prob(REAL) que ele da pra saida do MODELO {pm*100:.0f}%", flush=True)
    print(f">>> o juiz separa real de salada? {'SIM' if (ar+asal)/2 > 0.8 else 'fraco'} | ele acha o MODELO real ou fake? {'REAL (nao distingue)' if pm > 0.6 else 'FAKE (consegue criticar)' if pm < 0.4 else 'incerto'}", flush=True)


if __name__ == "__main__":
    main()
