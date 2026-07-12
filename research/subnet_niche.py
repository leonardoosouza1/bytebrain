#!/usr/bin/env python3
"""Multi-domínio com NICHING + observabilidade máxima. Evolui 3 sub-redes sobreviventes
(matemática / capitais / código) no MESMO Qwen e mede TUDO: perfil de neurônios vivos por
camada, núcleo compartilhado vs especialistas, sobreposição (Jaccard), fluxo de ativação,
trajetória evolutiva. Dump em research/subnet_niche.json pra visualização."""
import json, numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
layers = model.model.layers
NL, INTER = len(layers), model.config.intermediate_size
print(f"modelo {sum(p.numel() for p in model.parameters())/1e9:.2f}B | {NL}×{INTER} neurônios MLP", flush=True)

DOMAINS = {
    "matemática": [("12 + 15 = ", "27"), ("7 * 8 = ", "56"), ("100 - 37 = ", "63"), ("9 * 9 = ", "81"),
                   ("25 + 38 = ", "63"), ("6 * 7 = ", "42"), ("50 - 18 = ", "32"), ("8 * 9 = ", "72")],
    "capitais": [("A capital da França é", " Paris"), ("A capital do Japão é", " Tóquio"),
                 ("A capital da Itália é", " Roma"), ("A capital da Alemanha é", " Berlim"),
                 ("A capital de Portugal é", " Lisboa"), ("A capital da Espanha é", " Madri"),
                 ("A capital da Rússia é", " Moscou"), ("A capital da China é", " Pequim")],
    "código": [("def soma(a, b):\n    return ", "a + b"), ("for i in range(10):\n    print(", "i)"),
               ("def dobro(n):\n    return ", "n * 2"), ("x = [1, 2, 3]\nlen(", "x)"),
               ("if x > 0:\n    return ", "True"), ("while True:\n    ", "break"),
               ("import ", "numpy as np"), ("print('", "hello world')")],
}

keep = [torch.ones(INTER, device=DEV, dtype=torch.bool) for _ in range(NL)]
def _mask_hook(idx):
    def h(m, inp):                       # zera neurônios 'mortos' na entrada do down_proj
        x = inp[0].clone(); x[..., ~keep[idx]] = 0
        return (x,)
    return h
for i in range(NL):
    layers[i].mlp.down_proj.register_forward_pre_hook(_mask_hook(i))


def importance(probes):
    for i in range(NL): keep[i].fill_(True)   # mede importância com a rede CHEIA
    imp = [torch.zeros(INTER, device=DEV) for _ in range(NL)]
    def _imp_hook(idx):
        def h(m, inp):
            imp[idx].add_(inp[0][0].abs().float().mean(0).detach())   # NÃO retorna (senão vira input)
        return h
    hs = [layers[i].mlp.down_proj.register_forward_pre_hook(_imp_hook(i)) for i in range(NL)]
    with torch.no_grad():
        for p, _ in probes:
            model(tok(p, return_tensors="pt").input_ids.to(DEV))
    for h in hs: h.remove()
    return [torch.argsort(imp[i], descending=True) for i in range(NL)], [imp[i].cpu().numpy() for i in range(NL)]


def set_masks(order, g):
    for i in range(NL):
        k = max(1, int(g[i] * INTER)); keep[i].zero_(); keep[i][order[i][:k]] = True


@torch.no_grad()
def dloss(probes):
    tot, n = 0.0, 0
    for p, a in probes:
        pid = tok(p, return_tensors="pt").input_ids.to(DEV)
        aid = tok(a, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        ids = torch.cat([pid, aid], 1); pl = pid.shape[1]
        l = F.cross_entropy(model(ids).logits[0, pl - 1:-1].float(), ids[0, pl:])
        if torch.isfinite(l): tot += l.item(); n += 1
    return tot / max(1, n)


LAM = 2.0
res = {}
for name, probes in DOMAINS.items():
    order, imp = importance(probes)
    set_masks(order, np.ones(NL)); base = dloss(probes)
    rng = np.random.default_rng(0)
    pop = [rng.uniform(0.25, 1.0, NL) for _ in range(8)]; pop[0] = np.ones(NL)
    traj = []; best = None
    for gen in range(11):
        sc = sorted(([-dloss_val - LAM * float(np.mean(g)), dloss_val, float(np.mean(g)), g]
                     for g in pop for dloss_val in [(set_masks(order, g), dloss(probes))[1]]), key=lambda x: -x[0])
        if best is None or sc[0][0] > best[0]: best = sc[0]
        traj.append({"gen": gen, "loss": round(sc[0][1], 3), "keep": round(sc[0][2], 3)})
        el = [s[3] for s in sc[:3]]; pop = [e.copy() for e in el]
        while len(pop) < 8:
            c = el[rng.integers(len(el))].copy(); m = rng.random(NL) < 0.4
            c[m] += rng.normal(0, 0.15, NL)[m]; pop.append(np.clip(c, 0.03, 1.0))
    _, bloss, bk, bg = best
    alive = [set(order[i][:max(1, int(bg[i] * INTER))].cpu().numpy().tolist()) for i in range(NL)]
    res[name] = {"loss": round(bloss, 3), "base_loss": round(base, 3), "keep": round(bk, 3),
                 "layer_profile": [round(float(x), 3) for x in bg], "traj": traj,
                 "alive_per_layer": [len(s) for s in alive], "_alive": alive}
    print(f"[{name:11}] vivos {bk*100:4.1f}% | loss {bloss:.3f} (cheio {base:.3f})", flush=True)

# --- sobreposição / núcleo vs especialistas ---
glob = {n: set((i, x) for i in range(NL) for x in res[n]["_alive"][i]) for n in DOMAINS}
names = list(DOMAINS)
jac = {}
for a in names:
    for b in names:
        if a < b:
            inter = len(glob[a] & glob[b]); uni = len(glob[a] | glob[b])
            jac[f"{a}∩{b}"] = round(inter / uni, 3)
core = glob[names[0]] & glob[names[1]] & glob[names[2]]
spec = {n: len(glob[n] - set().union(*[glob[m] for m in names if m != n])) for n in names}
per_layer_core = [len(set(res[names[0]]["_alive"][i]) & set(res[names[1]]["_alive"][i]) & set(res[names[2]]["_alive"][i])) for i in range(NL)]

for n in DOMAINS: del res[n]["_alive"]
out = {"model": "qwen2.5-1.5b", "n_layers": NL, "n_neurons": INTER, "domains": res,
       "jaccard": jac, "core_size": len(core), "total_per_domain": {n: len(glob[n]) for n in names},
       "specialist_size": spec, "per_layer_core": per_layer_core}
json.dump(out, open("research/subnet_niche.json", "w"), ensure_ascii=False, indent=1)

print("\n=== SOBREPOSIÇÃO (Jaccard) ===")
for k, v in jac.items(): print(f"  {k}: {v}")
print(f"\nnúcleo compartilhado (nos 3): {len(core)} neurônios")
print("especialistas (só 1 domínio):", {n: spec[n] for n in names})
print("total vivo por domínio:", {n: len(glob[n]) for n in names})
print("\nDONE subnet_niche → research/subnet_niche.json", flush=True)
