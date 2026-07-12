#!/usr/bin/env python3
"""TRACE how Qwen builds a factual answer, token by token — "como ele chegou".
For each step: the chosen token, its probability, the top-5 alternatives it
considered, and the token decoded to BYTES. Shows the BPE pieces that spell the
answer (e.g. Brasília = ' Bras' + 'í' + 'lia') and their byte composition — the
foundation for transferring this knowledge to a byte model. CPU."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32).eval()

def trace(prompt, nsteps=10, topk=5):
    ids = tok(prompt, return_tensors="pt").input_ids
    print(f"\n=== PROMPT {prompt!r} ===")
    print(f"prompt tokens: {[tok.decode([i]) for i in ids[0].tolist()]}")
    gen = []
    for step in range(nsteps):
        with torch.no_grad():
            probs = torch.softmax(m(ids).logits[0, -1], -1)
        top = torch.topk(probs, topk)
        chosen = int(top.indices[0])
        if chosen == tok.eos_token_id:
            print(f"  step {step}: <eos>"); break
        ch = tok.decode([chosen]); by = list(ch.encode("utf-8"))
        alts = "  ".join(f"{tok.decode([int(i)])!r}={float(p):.0%}" for i, p in zip(top.indices, top.values))
        print(f"  step {step}: {ch!r:14} p={float(top.values[0]):.0%}  bytes={by}   alts: {alts}")
        gen.append(chosen); ids = torch.cat([ids, torch.tensor([[chosen]])], 1)
    full = tok.decode(gen)
    bs = full.encode("utf-8")
    print(f"  → ANSWER {full!r}")
    print(f"  → tokens : {[tok.decode([g]) for g in gen]}")
    print(f"  → bytes  : {list(bs)}")
    print(f"  → chars  : {[chr(b) if 32 <= b < 127 else f'·{b}' for b in bs]}")

for p in ["A capital do Brasil é", "A capital da França é a cidade de",
          "2 + 2 =", "A água é composta por hidrogênio e"]:
    trace(p)
print("\nDONE qwen_trace")
