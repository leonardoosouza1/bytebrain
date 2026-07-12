#!/usr/bin/env python3
"""Transfer step 4 — test if knowledge transferred. Greedily complete factual
prompts with a ByteBrain checkpoint. Run on the BASE model and the DISTILLED model
to compare: did it learn Qwen's facts? Includes trained forms AND held-out question
forms (generalization). CPU."""
import sys, torch
sys.path.insert(0, ".")
from research.train_graph import GraphByteGPT, set_act_quant

CK = sys.argv[1] if len(sys.argv) > 1 else "ckpt_coh_scale"
PROMPTS = [
    "A capital do Brasil é",
    "A capital da Itália é",
    "A capital do Japão é",
    "A capital da Alemanha é",
    "Qual é a capital da França?",        # held-out phrasing
    "A capital da Colômbia é",
    "A água é composta",
]
c = torch.load(f"{CK}/ckpt_best.pt", map_location="cpu", weights_only=False); cf = c["config"]
set_act_quant(cf.get("quant_bits", 0))
m = GraphByteGPT(cf["dim"], cf["layers"], cf["heads"], cf["ctx"], topk=cf.get("topk", 0),
                 mem=cf.get("mem", 0), topic=cf.get("topic", 0))
m.load_state_dict(c["model"]); m.eval()
print(f"# {CK} (step {c.get('step')}, bpb {round(c.get('best',9),4)})")

@torch.no_grad()
def greedy(prompt, n=40):
    ids = list(prompt.encode())
    for _ in range(n):
        x = torch.tensor([ids[-cf["ctx"]:]])
        nxt = int(m(x)[0, -1].argmax())
        ids.append(nxt)
        if ids[-3:] == [10, 10, 10]: break
    return bytes(ids[len(prompt.encode()):]).decode("utf-8", "replace").replace("\n", " ⏎ ")

for p in PROMPTS:
    print(f"  {p!r:38} → {greedy(p)!r}")
