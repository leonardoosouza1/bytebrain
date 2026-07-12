#!/usr/bin/env python3
"""T16 — ESCULPIR o super-modelo leve: corta FISICAMENTE os neurônios MLP não usados pelo chat
(perfil evoluído do Lote 2, top-k por importância POR CAMADA) fatiando as matrizes gate/up/down.
Resultado: modelo estruturalmente MENOR (menos params, menos VRAM, mais rápido), não só mascarado.
Mede params/VRAM/velocidade/loss/geração antes e depois. Dump → research/carve_model.json."""
import json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"; MAXTOK = 80
t0 = time.time()
CHAT = ["Oi, tudo bem? Como você está hoje?", "Me explique o que é fotossíntese de forma simples.",
    "Qual é a capital do Japão?", "Escreva uma frase bonita sobre o Brasil.",
    "O que você acha da inteligência artificial?", "Conte uma curiosidade sobre o espaço.",
    "Explique como funciona a gravidade.", "Qual a diferença entre um cão e um gato?",
    "Me dê uma dica de saúde simples.", "Por que o céu é azul?",
    "Resuma o que é a fotossíntese em uma frase.", "Como surgiu o universo?"]

# perfil evoluído do Lote 2 (61.7% média, chat 3.121 < cheio 3.193)
prof = json.load(open("/home/leonardo/projects/LLM/bytebrain/research/battery_02.json"))["evolved"]["layer_profile"]

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size
print(f"4B carregado ({time.time()-t0:.0f}s)", flush=True)

@torch.no_grad()
def chat_loss():
    tl, n = 0.0, 0
    for s in CHAT:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
        l = Fn.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:])
        if torch.isfinite(l): tl += l.item(); n += 1
    return tl / max(1, n)

@torch.no_grad()
def speed_and_sample():
    ids = tok("USER: Me explique o que é a gravidade.\nASSISTANT:", return_tensors="pt").input_ids.to(DEV)
    torch.cuda.synchronize(); s = time.time()
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
    torch.cuda.synchronize()
    tps = 40 / (time.time() - s)
    return tps, tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)[:140]

params0 = sum(p.numel() for p in model.parameters())
loss0 = chat_loss(); tps0, sample0 = speed_and_sample()
vram0 = torch.cuda.memory_allocated() / 1e9
print(f"ANTES: {params0/1e9:.2f}B params | chat {loss0:.3f} | {tps0:.1f} tok/s | VRAM {vram0:.1f}GB", flush=True)
print(f"  amostra: {sample0[:100]}", flush=True)

# importância por camada no chat
imp = [torch.zeros(INTER, device=DEV) for _ in range(NL)]
def mki(i):
    def h(m, inp): imp[i].add_(inp[0][0].abs().float().mean(0).detach())
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mki(i)) for i in range(NL)]
with torch.no_grad():
    for s in CHAT: model(tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV))
for h in hs: h.remove()

# ESCULPIR: fatiar gate/up/down por camada
kept = 0
for i in range(NL):
    k = max(64, int(prof[i] * INTER)); kept += k
    idx = torch.argsort(imp[i], descending=True)[:k]
    mlp = layers[i].mlp
    g = nn.Linear(mlp.gate_proj.in_features, k, bias=False, device=DEV, dtype=torch.float16)
    u = nn.Linear(mlp.up_proj.in_features, k, bias=False, device=DEV, dtype=torch.float16)
    d = nn.Linear(k, mlp.down_proj.out_features, bias=False, device=DEV, dtype=torch.float16)
    with torch.no_grad():
        g.weight.copy_(mlp.gate_proj.weight[idx]); u.weight.copy_(mlp.up_proj.weight[idx])
        d.weight.copy_(mlp.down_proj.weight[:, idx])
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = g, u, d
torch.cuda.empty_cache()

params1 = sum(p.numel() for p in model.parameters())
loss1 = chat_loss(); tps1, sample1 = speed_and_sample()
vram1 = torch.cuda.memory_allocated() / 1e9
print(f"\nDEPOIS: {params1/1e9:.2f}B params (-{(1-params1/params0)*100:.0f}%) | chat {loss1:.3f} | {tps1:.1f} tok/s | VRAM {vram1:.1f}GB", flush=True)
print(f"  neurônios MLP: {NL*INTER} → {kept} ({kept/(NL*INTER)*100:.0f}%)", flush=True)
print(f"  amostra: {sample1[:100]}", flush=True)

R = {"antes": {"params_B": round(params0/1e9, 2), "chat_loss": round(loss0, 3), "tok_s": round(tps0, 1),
               "vram_GB": round(vram0, 1), "sample": sample0},
     "depois": {"params_B": round(params1/1e9, 2), "chat_loss": round(loss1, 3), "tok_s": round(tps1, 1),
                "vram_GB": round(vram1, 1), "sample": sample1},
     "reducao_params_pct": round((1-params1/params0)*100, 1), "neuronios_mantidos_pct": round(kept/(NL*INTER)*100, 1)}
json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/carve_model.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## T16 — modelo ESCULPIDO fisicamente ({int(time.time()-t0)}s)\n")
    f.write(f"- {params0/1e9:.2f}B → {params1/1e9:.2f}B (-{(1-params1/params0)*100:.0f}%) | chat {loss0:.3f} → {loss1:.3f} | {tps0:.1f} → {tps1:.1f} tok/s | VRAM {vram0:.1f} → {vram1:.1f}GB\n")
print(f"\nDONE carve_model ({time.time()-t0:.0f}s)", flush=True)
