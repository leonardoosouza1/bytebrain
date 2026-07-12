#!/usr/bin/env python3
"""Test Qwen as a teacher for CODE and SHORT PHRASES (Leonardo's idea: small atomic
examples may distill better into a byte model; code is byte-friendly/structural).
Base Qwen2.5 completes code prompts well. GPU."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()

@torch.no_grad()
def gen(prompt, n=110, temp=0.0):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = m.generate(ids, max_new_tokens=n, do_sample=temp > 0, temperature=temp or None,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

CODE = [
    "// Classe JavaScript que representa uma conta bancária com depósito e saque\nclass ContaBancaria {",
    "// JavaScript class for a simple stack (push, pop, peek)\nclass Stack {",
    "def fibonacci(n):\n    \"\"\"Retorna o n-ésimo número de Fibonacci.\"\"\"\n",
]
PHRASES = [
    "Em uma frase, o que é fotossíntese? Fotossíntese é",
    "Complete: A água ferve a",
    "Defina recursão em uma frase: Recursão é",
]

print("="*70, "\nCÓDIGO\n", "="*70)
for p in CODE:
    print(f"\n### PROMPT:\n{p}\n### QWEN:")
    print(gen(p, n=130))
print("\n", "="*70, "\nFRASES CURTAS\n", "="*70)
for p in PHRASES:
    print(f"\n[{p!r}]\n  → {gen(p, n=40)!r}")
print("\nDONE qwen_code_test")
