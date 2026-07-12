#!/usr/bin/env python3
"""M149 — ANÁLISE DE ATIVAÇÃO (fluxo interno do modelo). Roda Qwen2.5-1.5B-Instruct na VRAM com hooks:
(1) ESPECTROGRAMA DE ATIVAÇÃO: norma do hidden state por camada × token (como o sinal cresce/flui);
(2) ESPARSIDADE DE NEURÔNIOS por camada (spiking: % de neurônios MLP ativos) — valida a tese spiking/byte.
GPU (1.5B cabe na VRAM). Dump marco149_metrics.json p/ visualizações."""
import json, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco149_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
P = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
NL = model.config.num_hidden_layers
log(f"Qwen2.5-1.5B-Instruct | {NL} camadas hidden {model.config.hidden_size}")

hid_norm = {}; mlp_spars = {}  # por camada
def mk_hid(li):
    def h(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out  # [1, seq, H]
        hid_norm[li] = hs[0].norm(dim=-1).float().cpu().numpy()  # norma por token
    return h
def mk_mlp(li):
    def h(mod, inp, out):
        a = inp[0][0].float()  # ativação dos neurônios (entrada do down_proj) [seq, inter]
        thr = 0.1 * a.abs().max()
        mlp_spars[li] = float((a.abs() > thr).float().mean().cpu())  # % neurônios ativos
    return h
for li in range(NL):
    model.model.layers[li].register_forward_hook(mk_hid(li))
    model.model.layers[li].mlp.down_proj.register_forward_hook(mk_mlp(li))

PROMPT = "Pergunta: Qual é a capital da França e por que ela é importante?\nResposta: A capital"
enc = tok(PROMPT, return_tensors="pt").to(DEV)
toks = [tok.decode([t]) for t in enc.input_ids[0].cpu().tolist()]
with torch.no_grad():
    model(**enc)

spectro = [[round(float(hid_norm[li][t]), 1) for t in range(len(toks))] for li in range(NL)]
spars = [round(mlp_spars[li] * 100, 1) for li in range(NL)]  # % ativos por camada
# sparsity média em vários prompts
extra = ["Explique o que é fotossíntese.", "2+2=", "Era uma vez um dragão que", "def soma(a,b): return"]
allspars = [np.array(spars)]
for pr in extra:
    mlp_spars.clear(); e = tok(pr, return_tensors="pt").to(DEV)
    with torch.no_grad(): model(**e)
    allspars.append(np.array([mlp_spars[li] * 100 for li in range(NL)]))
spars_mean = [round(float(x), 1) for x in np.mean(allspars, axis=0)]

res = {"NL": NL, "tokens": toks, "spectro_hidnorm": spectro, "sparsity_pct": spars_mean}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"tokens ({len(toks)}): {toks}")
log(f"esparsidade média por camada (%ativos): {spars_mean}")
log(f"  → esparsidade média geral: {np.mean(spars_mean):.1f}% ativos (spiking={'SIM' if np.mean(spars_mean)<40 else 'não'})")
log(f"DONE M149 ({time.time()-t0:.0f}s)")
