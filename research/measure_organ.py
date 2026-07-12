#!/usr/bin/env python3
"""BATERIA 7 — QUANTO o ÓRGÃO importado (ByteBrain treinado) é mais inteligente que o ser evoluído?
Mede a bpb teacher-forced do ByteBrain 8M no MESMO held-out PT usado pela ponte (bridge_life), e compara
com os pisos/tetos: unigrama fixa 6.028 · unigrama online 5.983 · bigrama online 5.073 · ser evoluído ~5.99.
Fecha a alça honesta de lote 1: a inteligência (contexto) tem que ser IMPORTADA, não evoluída-do-zero. GPU."""
import sys
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
import torch, torch.nn.functional as F
from wisdom_bridge import load_byte_model, DEV

m = load_byte_model(trained=True)
path = "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt"
with open(path, "rb") as f:
    f.seek(5_050_000); data = f.read(20000)          # held-out: mesmo train_end(5M)+50k da ponte

for W in [64, 128, 256]:
    total = 0.0; n = 0
    with torch.no_grad():
        for s in range(0, len(data) - W, W):
            ids = torch.tensor([list(data[s:s+W])], device=DEV)
            emb = m.tok(ids)
            logits = m(inputs_embeds=emb)[0]          # [W,256]
            lp = F.log_softmax(logits[:-1].float(), -1)
            tgt = ids[0, 1:]
            ll = lp[torch.arange(W-1, device=DEV), tgt]
            total += -(ll.sum().item()) / 0.69314718  # nats -> bits
            n += (W - 1)
    print(f"ByteBrain 8M  ctx={W:>3}  →  {total/n:.3f} bpb", flush=True)

print("\nreferências: unigrama fixa 6.028 · unigrama online 5.983 · bigrama online 5.073 · ser evoluído+vida ~5.99")
print("→ se o órgão fica MUITO abaixo (contexto longo), confirma: inteligência = IMPORTAR o byte-model, não evoluir preditor.")
