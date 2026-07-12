#!/usr/bin/env python3
"""Test if the distilled ByteBrain learned CODE + facts. Greedy completion. CPU."""
import sys, torch
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant

CK = sys.argv[1] if len(sys.argv) > 1 else "ckpt_distill2"
c = torch.load(f"{CK}/ckpt_best.pt", map_location="cpu", weights_only=False); cf = c["config"]
set_act_quant(cf.get("quant_bits", 0))
m = GraphByteGPT(cf["dim"], cf["layers"], cf["heads"], cf["ctx"], topk=cf.get("topk", 0),
                 mem=cf.get("mem", 0), topic=cf.get("topic", 0))
m.load_state_dict(c["model"]); m.eval()
print(f"# {CK} (step {c.get('step')}, bpb {round(c.get('best',9),4)})")

@torch.no_grad()
def g(prompt, n=160):
    ids = list(prompt.encode())
    for _ in range(n):
        ids.append(int(m(torch.tensor([ids[-cf["ctx"]:]]))[0, -1].argmax()))
    return bytes(ids[len(prompt.encode()):]).decode("utf-8", "replace")

CODE = ["// Classe JavaScript: Conta bancária com depósito e saque\nclass ContaBancaria {",
        "// Classe JavaScript: Pilha (stack) com push, pop e peek\nclass Pilha {",
        'def fibonacci(n):\n    """Retorna o n-ésimo número de Fibonacci."""\n']
FACT = ["A capital do Brasil é", "A capital do Japão é"]
DEF = ["Em programação, recursão é"]

for p in CODE:
    print(f"\n### {p.splitlines()[0]}\n{p}{g(p,170)}")
print("\n--- fatos retidos? ---")
for p in FACT: print(f"  {p!r} → {g(p,30)!r}")
for p in DEF: print(f"  {p!r} → {g(p,40)!r}")
