#!/usr/bin/env python3
"""Amostra o GERADOR (ckpt_gen) — gera PT a partir de prompts pra ver a qualidade ao longo do treino."""
import sys, torch
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
from src.sample import generate
DEV = "cuda"
CK = sys.argv[1] if len(sys.argv) > 1 else "/home/leonardo/projects/LLM/bytebrain/research/ckpt_gen/ckpt_best.pt"
ck = torch.load(CK, map_location=DEV, weights_only=False); c = ck["config"]
m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
m.load_state_dict(ck["model"])
print(f"gerador {m.n_params/1e6:.0f}M | step {ck['step']} | val {ck.get('best_val',0):.3f} bpb\n", flush=True)
for p in ["O Brasil é ", "A inteligência artificial ", "Era uma vez ", "A memória do cérebro "]:
    txt = generate(m, prompt=p, n=200, temperature=0.7, top_p=0.9, rep_penalty=1.3, device=DEV)
    print(f"[{p!r}]\n{txt}\n", flush=True)
