#!/usr/bin/env python3
"""Test ByteBrain capaz v1 across all distilled domains. Greedy. CPU."""
import sys, torch
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant

CK = sys.argv[1] if len(sys.argv) > 1 else "ckpt_broad"
c = torch.load(f"{CK}/ckpt_best.pt", map_location="cpu", weights_only=False); cf = c["config"]
set_act_quant(cf.get("quant_bits", 0))
m = GraphByteGPT(cf["dim"], cf["layers"], cf["heads"], cf["ctx"], topk=cf.get("topk", 0),
                 mem=cf.get("mem", 0), topic=cf.get("topic", 0))
m.load_state_dict(c["model"]); m.eval()
print(f"# {CK} (step {c.get('step')})")

@torch.no_grad()
def g(p, n=60):
    ids = list(p.encode())
    for _ in range(n): ids.append(int(m(torch.tensor([ids[-cf["ctx"]:]]))[0, -1].argmax()))
    return bytes(ids[len(p.encode()):]).decode("utf-8", "replace").replace("\n", " ⏎ ")

print("\n-- FATOS (treinados) --")
for p in ["A capital do Brasil é", "A capital da Itália é", "A capital da Austrália é"]: print(f"  {p!r} → {g(p,18)!r}")
print("-- FATOS (forma pergunta) --")
for p in ["Qual é a capital da França?", "Qual é a capital do Egito?"]: print(f"  {p!r} → {g(p,18)!r}")
print("-- CIÊNCIA / HISTÓRIA --")
for p in ["A água é composta", "A velocidade da luz", "A independência do Brasil foi"]: print(f"  {p!r} → {g(p,22)!r}")
print("-- DEFINIÇÕES --")
for p in ["Em programação, recursão é", "Em programação, algoritmo é"]: print(f"  {p!r} → {g(p,28)!r}")
print("-- CÓDIGO --")
for p in ["def fatorial(n):\n    \"\"\"Retorna o fatorial de n.\"\"\"\n", "// Classe JavaScript: Pessoa com nome e idade\nclass Pessoa {"]:
    print(f"\n{p}{g(p,120)}")
