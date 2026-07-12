#!/usr/bin/env python3
"""Coherence DECODING battery (no retraining) on the best model (dense-L1024).
Tests decoding-time fixes for drift, judged by the coherence metric:
  - plain sampling, repetition penalty, low temperature, and coherence-guided
    best-of-K word drafting (pick the word continuation with lowest wtrans).
"""
import sys, torch
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant
from research.coherence_metric import build_bigram, wtrans, drift

DEV = "cuda" if torch.cuda.is_available() else "cpu"
uni, bi, V = build_bigram()
PROMPT = "A história do Brasil colonial começa"

def load(d):
    ck = torch.load(f"{d}/ckpt_best.pt", map_location=DEV, weights_only=False); c = ck["config"]
    set_act_quant(c.get("quant_bits", 0))
    m = GraphByteGPT(c["dim"], c["layers"], c["heads"], c["ctx"],
                     topk=c.get("topk", 0), mem=c.get("mem", 0), topic=c.get("topic", 0)).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, c["ctx"]

@torch.no_grad()
def logits(m, ids, ctx):
    return m(torch.tensor([ids[-ctx:]], device=DEV))[0, -1].float()

@torch.no_grad()
def gen_plain(m, ctx, n=600, temp=0.8, rep=1.0, repw=64):
    ids = list(PROMPT.encode())
    for _ in range(n):
        lg = logits(m, ids, ctx)
        if rep > 1.0:
            for b in set(ids[-repw:]): lg[b] /= rep
        ids.append(int(torch.multinomial(torch.softmax(lg / temp, -1), 1)))
    return bytes(ids).decode("utf-8", "replace")

@torch.no_grad()
def gen_guided(m, ctx, n_words=95, K=4, temp=0.9):
    ids = list(PROMPT.encode())
    for _ in range(n_words):
        cands = []
        for _ in range(K):
            cur = ids[:]; w = []
            for _ in range(14):
                b = int(torch.multinomial(torch.softmax(logits(m, cur, ctx) / temp, -1), 1))
                w.append(b); cur.append(b)
                if b == 32 and len(w) > 1: break          # space = word boundary
            cands.append(w)
        ids += min(cands, key=lambda w: wtrans(bytes(ids[-60:] + w).decode("utf-8", "replace"), uni, bi, V))
    return bytes(ids).decode("utf-8", "replace")

m, ctx = load("ckpt_ovn_dense")
torch.manual_seed(0)
print(f"# dense-L1024 (ctx {ctx}) — coherence decoding battery\n")
print(f"{'variant':24} {'wtrans↓':>8} {'cont↑':>7} {'anchor↑':>8}")
for name, fn in [
    ("plain temp0.8", lambda: gen_plain(m, ctx, temp=0.8)),
    ("rep-penalty 1.3", lambda: gen_plain(m, ctx, temp=0.8, rep=1.3)),
    ("low-temp 0.5", lambda: gen_plain(m, ctx, temp=0.5)),
    ("guided best-of-4", lambda: gen_guided(m, ctx)),
]:
    txt = fn(); wt = wtrans(txt, uni, bi, V); cont, anch = drift(txt)
    print(f"{name:24} {wt:>8.2f} {cont:>7.3f} {anch:>8.3f}")
    print(f"   {txt[:200]!r}\n")
