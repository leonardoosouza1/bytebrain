"""Overnight evaluator: loads the current best checkpoint, generates samples, measures coherence,
and appends a row to the evolution journal. The cron cycle pauses training, runs this, then resumes.
"""
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from src.model import ByteGPT
from src.coherence import WordTransition
from src.sample import coherence_guided_generate

DEV = "cuda"
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")

# read latest step + val_bpb from the training log
step_log = val = "?"
try:
    log = open("/tmp/train_big.log", encoding="utf-8", errors="ignore").read()
    vs = re.findall(r"val_bpb ([\d.]+)", log)
    val = vs[-1] if vs else "?"
    ss = re.findall(r"^step (\d+)", log, re.M)
    step_log = ss[-1] if ss else "?"
except Exception:
    pass

ck = torch.load(f"{BASE}/ckpt_big/ckpt.pt", map_location=DEV, weights_only=False)  # modelo VIVO (nao o ckpt_best congelado)
m = ByteGPT(dim=640, n_layers=8, n_heads=8, context=512).to(DEV)
m.load_state_dict(ck["model"])
m.eval()
ref = open(f"{BASE}/data/pt_big.txt", encoding="utf-8", errors="ignore").read(8_000_000)
coh = WordTransition(ref)
del ref


def d4(t):
    b = t.encode()
    g = [bytes(b[i:i + 4]) for i in range(len(b) - 4)]
    return len(set(g)) / max(1, len(g))


def repbi(t):
    w = _W.findall(t.lower())
    bg = list(zip(w, w[1:]))
    return sum(v - 1 for v in Counter(bg).values() if v > 1) / max(1, len(bg))


def span(t, win=6, th=8.5):
    w = _W.findall(t.lower())
    for e in range(win, len(w)):
        if coh.score(" ".join(w[e - win:e])) > th:
            return e
    return len(w)


seeds = ["O Brasil ", "A ciencia ", "A historia "]
samples = []
for s in seeds:
    try:
        samples.append(coherence_guided_generate(m, coh, prompt=s, n=380, device=DEV))
    except Exception as e:
        samples.append(f"[erro: {e}]")
good = [s for s in samples if not s.startswith("[erro")]
wt = float(np.mean([coh.score(s) for s in good])) if good else 99.0
dd = float(np.mean([d4(s) for s in good])) if good else 0.0
rb = float(np.mean([repbi(s) for s in good])) if good else 0.0
sp = float(np.mean([span(s) for s in good])) if good else 0.0
ck_step = ck.get("step", "?")
ts = datetime.now().strftime("%d/%m %H:%M")

row = {"ts": ts, "ck_step": ck_step, "log_step": step_log, "val_bpb": val, "wtrans": round(wt, 2),
       "distinct4": round(dd, 2), "rep_bigramas_pct": round(rb * 100), "span": round(sp, 1),
       "sample": good[0][:320] if good else samples[0][:320]}
open(f"{BASE}/overnight_journal.jsonl", "a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
with open(f"{BASE}/overnight_journal.md", "a", encoding="utf-8") as f:
    f.write(f"\n### {ts} · step {ck_step} · val_bpb {val}\n")
    f.write(f"`wtrans {wt:.2f}` · `distinct4 {dd:.2f}` · `rep-bigramas {round(rb*100)}%` · `span coerente {sp:.0f}`\n\n")
    f.write(f"> {good[0][:320] if good else samples[0][:200]}\n")
print(f"AVALIADO {ts} | step {ck_step} | val {val} | wtrans {wt:.2f} | distinct4 {dd:.2f} | rep {round(rb*100)}% | span {sp:.0f}", flush=True)
print(f"amostra: {(good[0][:220] if good else samples[0][:220])!r}", flush=True)
