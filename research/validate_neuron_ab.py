#!/usr/bin/env python3
"""Validate the byte-neuron A/B: load baseline (float) and nibble (4-bit QAT)
checkpoints, report best val_bpb, and generate PT text from each — the nibble
model runs in its NATIVE 4-bit mode (set_act_quant_bits(4)) so we see the real
deployment quality, not a float cheat.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ByteGPT, load_checkpoint, set_act_quant_bits

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "O Brasil é um país"

@torch.no_grad()
def generate(model, prompt, n=140, temp=0.8, ctx=256):
    ids = list(prompt.encode("utf-8"))
    for _ in range(n):
        x = torch.tensor([ids[-ctx:]], device=DEV)
        logits = model(x)[0, -1] / temp
        p = torch.softmax(logits, -1)
        ids.append(int(torch.multinomial(p, 1)))
    return bytes(ids).decode("utf-8", errors="replace")

def load(ckpt_dir, qbits):
    ck = torch.load(os.path.join(ckpt_dir, "ckpt_best.pt"), map_location=DEV, weights_only=False)
    c = ck["config"]
    set_act_quant_bits(qbits)                       # nibble runs natively in 4-bit
    m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck.get("best_val", float("nan")), ck.get("step", 0)

for tag, d, q in [("BASELINE float", "ckpt_neuron_base", 0),
                  ("NIBBLE 4-bit ", "ckpt_neuron_nib4", 4)]:
    if not os.path.exists(os.path.join(d, "ckpt_best.pt")):
        print(f"{tag}: (sem checkpoint ainda)"); continue
    m, val, step = load(d, q)
    set_act_quant_bits(q)
    txt = generate(m, PROMPT)
    print(f"\n=== {tag} | best_val_bpb={val:.3f} | step={step} | run-mode={'4-bit' if q else 'float'} ===")
    print(repr(txt))
