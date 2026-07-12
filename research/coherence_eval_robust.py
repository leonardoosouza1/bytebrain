#!/usr/bin/env python3
"""Robust coherence eval — the n=5 anchor metric was too noisy (bounced 0.03-0.06).
Average over 12 varied prompts with a FIXED long generation (n=800) so the drift
windows are stable. Compares the levers: ctx512-small vs ctx512-big vs ctx1024-small.
Run in background (shares GPU with training; slow but won't timeout)."""
import sys, torch
sys.path.insert(0, ".")
from research.coherence_metric import build_bigram, wtrans, drift, gen
uni, bi, V = build_bigram()
PROMPTS = [
 "A história do Brasil colonial começa", "A fotossíntese é o processo pelo qual as plantas",
 "O futebol é o esporte mais popular do", "A inteligência artificial moderna utiliza",
 "A Segunda Guerra Mundial foi um conflito", "A cidade de São Paulo é conhecida por",
 "O corpo humano é composto por", "A música popular brasileira tem suas raízes",
 "Os planetas do sistema solar giram", "A economia do país cresceu",
 "A literatura portuguesa do século", "O clima da região amazônica é",
]
DIRS = sys.argv[1:] or ["ckpt_coh_base", "ckpt_coh_topic", "ckpt_coh_scale", "ckpt_ovn_dense"]
print(f"# robust coherence eval | {len(PROMPTS)} prompts × n=800 | {DIRS}", flush=True)
print(f"{'model':20} {'dim':>4} {'ctx':>5} {'bpb':>7} {'wtrans':>7} {'cont':>6} {'anchor':>7}", flush=True)
for d in DIRS:
    c = torch.load(f"{d}/ckpt_best.pt", map_location="cpu", weights_only=False); cf = c["config"]
    torch.manual_seed(0); wt = ct = an = 0.0
    for p in PROMPTS:
        txt = gen(d, 0, p, n=800)
        w = wtrans(txt, uni, bi, V); cc, a = drift(txt); wt += w; ct += cc; an += a
    n = len(PROMPTS)
    print(f"{d:20} {cf['dim']:>4} {cf['ctx']:>5} {round(c['best'],4):>7} {wt/n:>7.2f} {ct/n:>6.3f} {an/n:>7.3f}", flush=True)
print("DONE robust eval", flush=True)
