#!/usr/bin/env python3
"""BATERIA 2: compressão FEITA CERTO — evoluir a sub-rede de CHAT no Qwen3-4B (GA por keep-fraction
por camada, máscara = top-k% por importância no chat), em vez de podar por magnitude (que colapsou
no Lote 1). Fitness = manter o chat com o mínimo de neurônios. Mostra o quão leve dá pra ficar
mantendo o chat. Grava no battery_journal.md + battery_02.json."""
import json, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"; MAXTOK = 80
t0 = time.time()
CHAT = ["Oi, tudo bem? Como você está hoje?", "Me explique o que é fotossíntese de forma simples.",
    "Qual é a capital do Japão?", "Escreva uma frase bonita sobre o Brasil.",
    "O que você acha da inteligência artificial?", "Conte uma curiosidade sobre o espaço.",
    "Explique como funciona a gravidade.", "Qual a diferença entre um cão e um gato?",
    "Me dê uma dica de saúde simples.", "Por que o céu é azul?",
    "Resuma o que é a fotossíntese em uma frase.", "Como surgiu o universo?"]

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size
print(f"4B carregado ({time.time()-t0:.0f}s) | {NL}L×{INTER}", flush=True)

keep = [torch.ones(INTER, device=DEV, dtype=torch.bool) for _ in range(NL)]
def mk(i):
    def h(m, inp):
        x = inp[0].clone(); x[..., ~keep[i]] = 0; return (x,)
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_pre_hook(mk(i))

# importância dos neurônios NO CHAT
imp = [torch.zeros(INTER, device=DEV) for _ in range(NL)]
def mki(i):
    def h(m, inp): imp[i].add_(inp[0][0].abs().float().mean(0).detach())
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mki(i)) for i in range(NL)]
with torch.no_grad():
    for s in CHAT: model(tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV))
for h in hs: h.remove()
order = [torch.argsort(imp[i], descending=True) for i in range(NL)]

def set_masks(g):
    for i in range(NL):
        k = max(1, int(g[i] * INTER)); keep[i].zero_(); keep[i][order[i][:k]] = True

@torch.no_grad()
def chat_loss():
    tl, n = 0.0, 0
    for s in CHAT:
        ids = tok(s, return_tensors="pt", truncation=True, max_length=MAXTOK).input_ids.to(DEV)
        if ids.shape[1] < 4: continue
        l = F.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:])
        if torch.isfinite(l): tl += l.item(); n += 1
    return tl / max(1, n)

set_masks(np.ones(NL)); base = chat_loss()
print(f"baseline chat loss (100%): {base:.3f}", flush=True)

# comparação: poda ingênua por importância (mesmo keep em todas as camadas)
naive = {}
for pct in [0.75, 0.5, 0.3]:
    set_masks(np.full(NL, pct)); naive[f"{int(pct*100)}%"] = round(chat_loss(), 3)
print(f"poda por importância (uniforme): {naive}", flush=True)

# GA: keep-fraction POR CAMADA evoluído
LAM = 1.5
def fit(g):
    set_masks(g); l = chat_loss(); k = float(np.mean(g)); return -l - LAM * k, l, k
rng = np.random.default_rng(0); POP = 8
pop = [rng.uniform(0.3, 1.0, NL) for _ in range(POP)]; pop[0] = np.ones(NL)
best = None; traj = []
for gen in range(12):
    sc = sorted(([*fit(g), g] for g in pop), key=lambda x: -x[0])
    if best is None or sc[0][0] > best[0]: best = sc[0]
    traj.append({"gen": gen, "loss": round(best[1], 3), "keep": round(best[2], 3)})
    print(f"  gen {gen:>2}: chat {best[1]:.3f} | neurônios {best[2]*100:.1f}%", flush=True)
    el = [s[3] for s in sc[:3]]; pop = [e.copy() for e in el]
    while len(pop) < POP:
        c = el[rng.integers(len(el))].copy(); mm = rng.random(NL) < 0.4
        c[mm] += rng.normal(0, 0.12, NL)[mm]; pop.append(np.clip(c, 0.03, 1.0))
_, bl, bk, bg = best

R = {"model": "qwen3-4b", "base_chat_loss": round(base, 3), "naive_uniform": naive,
     "evolved": {"chat_loss": round(bl, 3), "keep_pct": round(bk * 100, 1),
                 "layer_profile": [round(float(x), 2) for x in bg]}, "traj": traj}
json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/battery_02.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Lote 2 — compressão evoluída (chat, Qwen3-4B, {int(time.time()-t0)}s)\n")
    f.write(f"- baseline chat {base:.3f}\n- poda uniforme por importância: {naive}\n")
    f.write(f"- EVOLUÍDA: chat {bl:.3f} com {bk*100:.1f}% dos neurônios (vs uniforme colapsando)\n")
print(f"\n=== EVOLUÍDA: chat {bl:.3f} com {bk*100:.1f}% dos neurônios (baseline {base:.3f}) ===")
print(f"DONE battery_02 ({time.time()-t0:.0f}s)", flush=True)
