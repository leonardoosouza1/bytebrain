#!/usr/bin/env python3
"""Coherence metrics (bpb can't see these). Generate a long passage from a model
and measure:
  - wtrans : mean word-transition surprisal (bits) under a PT word-bigram model.
             Lower = more fluent word flow. (real PT is low, gibberish high)
  - cont   : avg Jaccard overlap of content-words between CONSECUTIVE windows.
             Higher = local topic continuity.
  - anchor : avg Jaccard overlap of each window with the FIRST window.
             Higher = stays anchored to the initial topic (anti-drift).
Used to compare dense-L1024 vs top8-L1024 — does the sparse graph help COHESION?
"""
import sys, re, math, collections, torch
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant

WORD = re.compile(r"[a-zà-ÿ]+")
def words(t): return WORD.findall(t.lower())

def build_bigram(path="data/pt_clean.txt", limit=6_000_000):
    txt = open(path, encoding="utf-8", errors="replace").read(limit).lower()
    w = words(txt)
    uni = collections.Counter(w); bi = collections.Counter(zip(w, w[1:]))
    return uni, bi, len(uni)

def wtrans(text, uni, bi, V, a=0.5):
    w = words(text)
    if len(w) < 3: return 99.0
    s = sum(-math.log2(max((bi[(x, y)] + a) / (uni[x] + a * V), 1e-12)) for x, y in zip(w, w[1:]))
    return s / (len(w) - 1)

def jacc(a, b): u = a | b; return len(a & b) / max(1, len(u))
def content(t): return set(x for x in words(t) if len(x) >= 4)

def drift(text, win=110):
    wins = [text[i:i + win] for i in range(0, len(text), win)]
    sets = [content(w) for w in wins if w.strip()]
    if len(sets) < 2: return 0.0, 0.0
    cont = sum(jacc(sets[i], sets[i + 1]) for i in range(len(sets) - 1)) / (len(sets) - 1)
    anchor = sum(jacc(sets[0], s) for s in sets[1:]) / (len(sets) - 1)
    return cont, anchor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
@torch.no_grad()
def gen(d, q, prompt, n=700, temp=0.75):
    ck = torch.load(f"{d}/ckpt_best.pt", map_location=DEV, weights_only=False); c = ck["config"]
    set_act_quant(q)
    m = GraphByteGPT(c["dim"], c["layers"], c["heads"], c["ctx"], topk=c.get("topk", 0),
                     vq_codes=c.get("vq_codes", 0), mem=c.get("mem", 0), topic=c.get("topic", 0)).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    ids = list(prompt.encode())
    for _ in range(n):
        x = torch.tensor([ids[-c["ctx"]:]], device=DEV)
        p = torch.softmax(m(x)[0, -1] / temp, -1); ids.append(int(torch.multinomial(p, 1)))
    return bytes(ids).decode("utf-8", "replace")

if __name__ == "__main__":
    uni, bi, V = build_bigram()
    PROMPT = "A história do Brasil colonial começa"
    torch.manual_seed(0)
    dirs = sys.argv[1:] or ["ckpt_ovn_dense", "ckpt_ovn_top8"]
    for d in dirs:
        tag = d
        txt = gen(d, 0, PROMPT)
        wt = wtrans(txt, uni, bi, V); cont, anch = drift(txt)
        print(f"\n=== {tag} ===")
        print(f"  wtrans={wt:.2f}  continuidade={cont:.3f}  ancoragem={anch:.3f}")
        print(f"  {txt[:240]!r}")
