#!/usr/bin/env python3
"""FUSÃO Universe × pesos: evoluir uma SUB-REDE que SOBREVIVE para um domínio.
Carrega um modelo real (Qwen2.5-Math-1.5B, torch, cabe na VRAM), e evolui uma máscara
de neurônios MLP (quais 'vivem' por camada) com um GA: fitness = manter o domínio
(baixa loss em contas) usando O MÍNIMO de neurônios ativos. Prova que um domínio pode
viver numa sub-rede esparsa selecionada por sobrevivência — não por gradiente."""
import sys, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
layers = model.model.layers
NL = len(layers)
INTER = model.config.intermediate_size
print(f"modelo: {sum(p.numel() for p in model.parameters())/1e9:.2f}B | {NL} camadas × {INTER} neurônios MLP", flush=True)

PROBES = [("12 + 15 = ", "27"), ("7 * 8 = ", "56"), ("100 - 37 = ", "63"), ("9 * 9 = ", "81"),
          ("25 + 38 = ", "63"), ("6 * 7 = ", "42"), ("50 - 18 = ", "32"), ("13 + 29 = ", "42"),
          ("8 * 9 = ", "72"), ("30 + 45 = ", "75")]

# --- importância dos neurônios no domínio (entrada do down_proj = ativações MLP) ---
imp = [torch.zeros(INTER, device=DEV) for _ in range(NL)]
def imp_hook(i):
    def h(m, inp): imp[i] += inp[0][0].abs().float().mean(0).detach()
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(imp_hook(i)) for i in range(NL)]
with torch.no_grad():
    for p, _ in PROBES:
        model(tok(p, return_tensors="pt").input_ids.to(DEV))
for h in hs: h.remove()
order = [torch.argsort(imp[i], descending=True) for i in range(NL)]  # neurônios mais importantes 1º

# --- máscara viva por camada (hook zera os 'mortos') ---
keep = [torch.ones(INTER, device=DEV, dtype=torch.bool) for _ in range(NL)]
def mask_hook(i):
    def h(m, inp):
        x = inp[0].clone(); x[..., ~keep[i]] = 0; return (x,)
    return h
for i in range(NL):
    layers[i].mlp.down_proj.register_forward_pre_hook(mask_hook(i))

def set_masks(genome):
    for i in range(NL):
        k = max(1, int(genome[i] * INTER))
        keep[i].zero_(); keep[i][order[i][:k]] = True

@torch.no_grad()
def probe_loss():
    tot = 0.0
    for p, a in PROBES:
        ids = tok(p + a, return_tensors="pt").input_ids.to(DEV)
        plen = tok(p, return_tensors="pt").input_ids.shape[1]
        lg = model(ids).logits[0, plen - 1:-1]
        tgt = ids[0, plen:]
        tot += F.cross_entropy(lg.float(), tgt).item()
    return tot / len(PROBES)

LAM = 2.0  # peso da sparsity (recompensa menos neurônios)
def fitness(g):
    set_masks(g); loss = probe_loss(); k = float(np.mean(g))
    return -loss - LAM * k, loss, k

# baseline: modelo cheio
set_masks(np.ones(NL)); base_loss = probe_loss()
print(f"baseline (100% neurônios): loss domínio = {base_loss:.3f}\n", flush=True)

rng = np.random.default_rng(0)
POP, GENS = 10, 12
pop = [rng.uniform(0.25, 1.0, NL) for _ in range(POP)]
pop[0] = np.ones(NL)  # inclui o cheio
best = None
for gen in range(GENS):
    scored = sorted(([*fitness(g), g] for g in pop), key=lambda x: -x[0])
    if best is None or scored[0][0] > best[0]:
        best = scored[0]
    f, loss, k, _ = scored[0]
    print(f"  gen {gen:>2}: melhor fitness {f:6.3f} | loss {loss:.3f} | neurônios vivos {k*100:4.1f}%", flush=True)
    elites = [s[3] for s in scored[:3]]
    pop = [e.copy() for e in elites]
    while len(pop) < POP:
        child = elites[rng.integers(len(elites))].copy()
        m = rng.random(NL) < 0.4
        child[m] += rng.normal(0, 0.15, NL)[m]
        pop.append(np.clip(child, 0.03, 1.0))

_, bloss, bk, bg = best
print(f"\n=== SUB-REDE SOBREVIVENTE (domínio: aritmética) ===")
print(f"neurônios vivos: {bk*100:.1f}%  (mortos {100-bk*100:.1f}%)  |  loss {bloss:.3f}  vs  cheio {base_loss:.3f}")
print(f"perfil por camada (keep-fraction): {np.round(bg,2).tolist()}")
print("DONE subnet_evolve", flush=True)
