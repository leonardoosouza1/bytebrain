#!/usr/bin/env python3
"""M115 — FLORESTA AUTO-PROPAGANTE (a ideia do Leonardo: "e se a árvore der sementes/frutos? aumenta a
variedade de neurônios? um sistema que se auto-corrige e adapta?"). Tronco CONGELADO (Math-1.5B).
LOOP: planta semente-MÃE ampla → mede onde ela ERRA → BROTA filha (init da mãe + ruído = mutação)
treinada só no nicho fraco → SELEÇÃO (mantém só se a filha cobre o que a mãe não cobria) → repete.
MEDE 3 coisas que testam a hipótese:
  (a) AUTO-CORREÇÃO: a cobertura da floresta sobe SOZINHA a cada broto?
  (b) VARIEDADE DE NEURÔNIOS: as filhas acendem MÁSCARAS de neurônios distintas da mãe? (Jaccard no atlas)
  (c) CONVERGÊNCIA: a seleção freia (satura), não vira mato infinito?
GPU. Dump marco115_metrics.json + journal."""
import json, time, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"; DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings(); NL = model.config.num_hidden_layers
random.seed(0); CH = 48
log(f"M115 START | tronco Math-1.5B congelado, {NL} camadas, hidden {H}")

# ---- banco de fatos arbitrários 1-token (a mãe NÃO vai caber em todos → tem onde brotar) ----
POOL = []
for tid in range(min(len(tok), 60000)):
    s = tok.decode([tid])
    if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3 <= len(s.strip()) <= 8:
        if len(tok(s, add_special_tokens=False).input_ids) == 1: POOL.append(s)
    if len(POOL) >= 80: break
TP = ["O código secreto número {i} é", "O nome do robô {i} é", "A cor do domínio {i} é",
      "A senha do cofre {i} é", "O valor da chave {i} é"]
FACTS = [(TP[i % 5].format(i=i), POOL[random.randrange(len(POOL))]) for i in range(90)]
P = [(tok(p).input_ids, tok(tg, add_special_tokens=False).input_ids[:1]) for p, tg in FACTS]
NF = len(P)

def batch(idxs, K):
    sub = [P[i] for i in idxs]; N = len(sub); maxlen = K + max(len(pi) + 1 for pi, _ in sub)
    E = torch.zeros(N, maxlen, H, device=DEV, dtype=torch.float16)
    am = torch.zeros(N, maxlen, device=DEV, dtype=torch.long)
    pos = torch.zeros(N, dtype=torch.long, device=DEV); tgt = torch.zeros(N, dtype=torch.long, device=DEV)
    for j, (pi, ti) in enumerate(sub):
        pe = EL(torch.tensor([pi], device=DEV)).detach()[0]; te = EL(torch.tensor([ti], device=DEV)).detach()[0]
        L = K + len(pi) + 1; E[j, K:K+len(pi)] = pe; E[j, K+len(pi):L] = te; am[j, :L] = 1
        pos[j] = K + len(pi) - 1; tgt[j] = ti[0]
    return E, am, pos, tgt

def fwd(seed, Ec, amc):
    X = Ec.clone(); X[:, :seed.shape[0]] = seed.to(torch.float16)
    return model(inputs_embeds=X, attention_mask=amc).logits

def plant(idxs, K=1, steps=800, lr=0.1, clip=1.0, init=None, tseed=0):
    E, am, pos, tgt = batch(idxs, K); N = len(idxs)
    if init is not None:  # REPRODUÇÃO com VARIAÇÃO: filha = mãe + ruído (mutação)
        g = torch.Generator(device=DEV).manual_seed(tseed)
        seed = nn.Parameter(init.float().clone() + torch.randn(K, H, generator=g, device=DEV) * 0.05)
    else:
        g = torch.Generator(device=DEV).manual_seed(tseed)
        seed = nn.Parameter(torch.randn(K, H, generator=g, device=DEV, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
        opt.zero_grad()
        for i in range(0, N, CH):
            lg = fwd(seed, E[i:i+CH], am[i:i+CH]); n = lg.shape[0]; ar = torch.arange(n, device=DEV)
            (F.cross_entropy(lg[ar, pos[i:i+CH]].float(), tgt[i:i+CH], reduction="sum") / N).backward()
        if clip: torch.nn.utils.clip_grad_norm_([seed], clip)
        opt.step()
    return seed.detach().to(torch.float16)

@torch.no_grad()
def covered(seed, idxs):
    """quais desses idxs a semente RECUPERA (recall)."""
    E, am, pos, tgt = batch(idxs, seed.shape[0]); ok = []
    for i in range(0, len(idxs), CH):
        lg = fwd(seed, E[i:i+CH], am[i:i+CH]); n = lg.shape[0]; ar = torch.arange(n, device=DEV)
        hit = (lg[ar, pos[i:i+CH]].argmax(-1) == tgt[i:i+CH]).cpu().tolist()
        ok += [idxs[i+j] for j, h in enumerate(hit) if h]
    return set(ok)

# ---- medição de VARIEDADE: máscara de neurônios (MLP) que uma semente acende ----
PROBE_L = list(range(2, NL, 5)); _acts = {}
def _hook(name):
    def h(mod, inp, out): _acts[name] = inp[0].detach().abs().mean(dim=(0, 1)).float()
    return h
_handles = [model.model.layers[li].mlp.down_proj.register_forward_hook(_hook(f"L{li}")) for li in PROBE_L]

@torch.no_grad()
def neuron_mask(seed, idxs, topfrac=0.10):
    """máscara dos neurônios MLP mais ativos (top 10%) sob a semente, concatenada nas camadas-sonda."""
    _acts.clear(); E, am, pos, tgt = batch(idxs[:24], seed.shape[0])
    fwd(seed, E, am)
    bits = []
    for li in PROBE_L:
        v = _acts[f"L{li}"]; k = max(1, int(len(v) * topfrac))
        m = torch.zeros_like(v, dtype=torch.bool); m[torch.topk(v, k).indices] = True; bits.append(m)
    return torch.cat(bits)

def jaccard(a, b): return float((a & b).sum()) / float((a | b).sum() + 1e-9)

# ================= LOOP AUTO-PROPAGANTE =================
log(f"=== plantando MÃE em {NF} fatos (K=1) ===")
mother = plant(list(range(NF)), tseed=0)
cov = covered(mother, list(range(NF)))
forest = [{"nome": "mãe", "seed": mother, "cobre": cov, "mask": neuron_mask(mother, sorted(cov) or list(range(NF)))}]
all_cov = set(cov)
hist = [{"broto": 0, "arvores": 1, "cobertura": len(all_cov), "de": NF}]
log(f"  MÃE cobre {len(all_cov)}/{NF} (erra {NF-len(all_cov)}) → tem onde brotar")

for rnd in range(1, 9):
    missing = [i for i in range(NF) if i not in all_cov]
    if not missing:
        log(f"  CONVERGIU no broto {rnd-1}: floresta cobre 100% ({len(all_cov)}/{NF})"); break
    niche = missing[:30]  # nicho fraco onde a mãe/floresta erra
    parent = forest[len(forest) % len(forest)]  # brota de uma árvore existente (mutação da mãe/última)
    child = plant(niche, init=parent["seed"], tseed=rnd)
    child_cov = covered(child, niche)
    gain = child_cov - all_cov  # SELEÇÃO: a filha cobre algo NOVO?
    if len(gain) == 0:
        log(f"  broto {rnd}: filha não cobriu nada novo → ADUBO (podada)");
        hist.append({"broto": rnd, "arvores": len(forest), "cobertura": len(all_cov), "de": NF, "podada": True})
        continue
    cmask = neuron_mask(child, sorted(child_cov) or niche)
    jm = jaccard(cmask, forest[0]["mask"])  # variedade vs MÃE
    forest.append({"nome": f"filha{rnd}", "seed": child, "cobre": child_cov, "mask": cmask, "jaccard_vs_mae": jm})
    all_cov |= child_cov
    hist.append({"broto": rnd, "arvores": len(forest), "cobertura": len(all_cov), "de": NF,
                 "ganho": len(gain), "jaccard_vs_mae": round(jm, 3)})
    log(f"  broto {rnd}: filha cobre +{len(gain)} novos → floresta {len(all_cov)}/{NF} | "
        f"máscara-neurônio Jaccard vs mãe {jm:.3f} (baixo=variedade nova)")

# variedade final: Jaccard médio entre pares de árvores
masks = [f["mask"] for f in forest]; pares = []
for i in range(len(masks)):
    for j in range(i+1, len(masks)): pares.append(jaccard(masks[i], masks[j]))
var = {"jaccard_medio_entre_arvores": round(float(np.mean(pares)), 3) if pares else None,
       "n_arvores": len(forest), "neuronios_sonda": int(masks[0].numel()),
       "ativos_por_arvore": [int(m.sum()) for m in masks]}
uniao = torch.zeros_like(masks[0])
for m in masks: uniao |= m
var["neuronios_unicos_uniao"] = int(uniao.sum()); var["neuronios_ativos_mae"] = int(masks[0].sum())
var["fator_variedade"] = round(var["neuronios_unicos_uniao"] / max(1, var["neuronios_ativos_mae"]), 2)

RES = {"NF": NF, "historico": hist, "cobertura_final": len(all_cov), "arvores": len(forest),
       "variedade_neuronios": var, "camadas_sonda": PROBE_L}
json.dump(RES, open("/home/leonardo/projects/LLM/bytebrain/research/marco115_metrics.json", "w"),
          ensure_ascii=False, indent=1)
log(f"=== RESULTADO ===")
log(f"(a) AUTO-CORREÇÃO: cobertura {hist[0]['cobertura']}→{len(all_cov)}/{NF} com {len(forest)} árvores")
log(f"(b) VARIEDADE: floresta acende {var['neuronios_unicos_uniao']} neurônios únicos vs {var['neuronios_ativos_mae']} da mãe "
    f"= {var['fator_variedade']}× | Jaccard médio entre árvores {var['jaccard_medio_entre_arvores']}")
log(f"(c) CONVERGÊNCIA: {'SIM' if len(all_cov)==NF else 'parcial'} ({len(forest)} árvores)")
log(f"DONE M115 ({time.time()-t0:.0f}s)")
