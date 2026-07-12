#!/usr/bin/env python3
"""TESTE 3 CONSERTADO — a paleta de safetensors (Instruct/Coder/Math) responde melhor o domínio?

Bug do 1º run: medí prob do token 'sem espaço' → ~0 pra todos (o modelo emite ' 4' com espaço).
Conserto: NLL teacher-forced da resposta INTEIRA (com espaço), em bits. Menor = mais confiante/certo.
GPU, threads capados, 1 modelo por vez (não trava)."""
import torch, os, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../llm-lab/models")
PATHS = {"Instruct":f"{BASE}/Qwen2.5-1.5B-Instruct","Coder":f"{BASE}/Qwen2.5-Coder-1.5B","Math":f"{BASE}/Qwen2.5-Math-1.5B"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(PATHS["Instruct"])

MATH = [("17 * 23 ="," 391"),("12 * 12 ="," 144"),("144 / 12 ="," 12"),("If 3x = 21 then x ="," 7"),
        ("The square root of 144 is"," 12"),("25 * 4 ="," 100"),("9 * 9 ="," 81"),("100 - 37 ="," 63")]
CODE = [("import numpy as"," np"),("import pandas as"," pd"),
        ("To open a file in Python you call"," open"),
        ("def factorial(n):\n    if n <= 1:\n        return 1\n    return n *"," factorial"),
        ("# read a csv\ndf = pd."," read_csv"),("for i in range(10):\n    print"," (i)")]

@torch.no_grad()
def nll(m, prompt, ans):                     # bits/token da resposta certa (teacher-forced)
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    aids = tok(ans, add_special_tokens=False).input_ids
    full = torch.cat([pids, torch.tensor([aids], device=DEV)], 1)
    logits = m(full).logits[0]
    base = pids.shape[1]-1
    tot = sum(-torch.log_softmax(logits[base+k].float(),-1)[a].item() for k,a in enumerate(aids))
    return tot/len(aids)/0.6931471806          # nats->bits

def load(n): return AutoModelForCausalLM.from_pretrained(PATHS[n], dtype=torch.float16).to(DEV).eval()

print(f"TESTE 3 (consertado) — NLL da resposta em bits (MENOR=melhor). {DEV}\n")
res = {}
for name in ["Instruct","Math","Coder"]:
    m = load(name)
    res[name] = (sum(nll(m,p,a) for p,a in MATH)/len(MATH), sum(nll(m,p,a) for p,a in CODE)/len(CODE))
    del m; gc.collect(); torch.cuda.empty_cache()
    print(f"  {name} carregado e medido.")

print(f"\n  {'modelo':>10}  {'MATEMÁTICA (bits)':>18}  {'CÓDIGO (bits)':>14}")
for name in ["Instruct","Math","Coder"]:
    print(f"  {name:>10}  {res[name][0]:>18.3f}  {res[name][1]:>14.3f}")
mbest = min(res, key=lambda k:res[k][0]); cbest = min(res, key=lambda k:res[k][1])
print(f"\n  → melhor em MATEMÁTICA: {mbest}   melhor em CÓDIGO: {cbest}")
gm = res["Instruct"][0]-res["Math"][0]; gc_ = res["Instruct"][1]-res["Coder"][1]
print(f"  Math vs Instruct em matemática: {gm:+.3f} bits  ({'especialista ganha' if gm>0 else 'não ganha'})")
print(f"  Coder vs Instruct em código:    {gc_:+.3f} bits  ({'especialista ganha' if gc_>0 else 'não ganha'})")
print(f"\n  {'✓ a paleta paga (rotear pro especialista)' if mbest=='Math' and cbest=='Coder' else '⚠ ganho de especialista é parcial/pequeno nesta escala — honesto'}")
