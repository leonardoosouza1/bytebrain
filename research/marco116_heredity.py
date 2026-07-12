#!/usr/bin/env python3
"""M116 — ABLAÇÃO DE HEREDITARIEDADE (fecha a questão do M115: "a árvore DAR semente vale mais que só
plantar do zero?"). A filha que HERDA da mãe (init = mãe + ruído = reprodução com variação) cobre o
nicho fraco MAIS RÁPIDO / MELHOR que uma semente NOVA (init aleatório)? Não é óbvio: a mãe FALHOU
naquele nicho — herdar pode ajudar (estrutura do formato) ou atrapalhar (valores errados).
Varre orçamento de passos × {herdada, do_zero} × réplicas. Também mede se a filha herdada fica mais
PARECIDA com a mãe no espaço de neurônios (herança de traços). GPU. Dump marco116_metrics.json."""
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
log(f"M116 START | tronco congelado {NL}L hidden {H}")

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

def batch(idxs, K):
    sub = [P[i] for i in idxs]; N = len(sub); maxlen = K + max(len(pi) + 1 for pi, _ in sub)
    E = torch.zeros(N, maxlen, H, device=DEV, dtype=torch.float16); am = torch.zeros(N, maxlen, device=DEV, dtype=torch.long)
    pos = torch.zeros(N, dtype=torch.long, device=DEV); tgt = torch.zeros(N, dtype=torch.long, device=DEV)
    for j, (pi, ti) in enumerate(sub):
        pe = EL(torch.tensor([pi], device=DEV)).detach()[0]; te = EL(torch.tensor([ti], device=DEV)).detach()[0]
        L = K + len(pi) + 1; E[j, K:K+len(pi)] = pe; E[j, K+len(pi):L] = te; am[j, :L] = 1
        pos[j] = K + len(pi) - 1; tgt[j] = ti[0]
    return E, am, pos, tgt

def fwd(seed, Ec, amc):
    X = Ec.clone(); X[:, :seed.shape[0]] = seed.to(torch.float16)
    return model(inputs_embeds=X, attention_mask=amc).logits

def plant(idxs, steps, init=None, tseed=0):
    E, am, pos, tgt = batch(idxs, 1); N = len(idxs); g = torch.Generator(device=DEV).manual_seed(tseed)
    if init is not None:
        seed = nn.Parameter(init.float().clone() + torch.randn(1, H, generator=g, device=DEV) * 0.05)
    else:
        seed = nn.Parameter(torch.randn(1, H, generator=g, device=DEV, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([seed], lr=0.1)
    for s in range(steps):
        for gr in opt.param_groups: gr["lr"] = 0.1 * 0.5 * (1 + math.cos(math.pi * s / steps))
        opt.zero_grad()
        for i in range(0, N, CH):
            lg = fwd(seed, E[i:i+CH], am[i:i+CH]); n = lg.shape[0]; ar = torch.arange(n, device=DEV)
            (F.cross_entropy(lg[ar, pos[i:i+CH]].float(), tgt[i:i+CH], reduction="sum") / N).backward()
        torch.nn.utils.clip_grad_norm_([seed], 1.0); opt.step()
    return seed.detach().to(torch.float16)

@torch.no_grad()
def cov(seed, idxs):
    E, am, pos, tgt = batch(idxs, 1); ok = 0
    for i in range(0, len(idxs), CH):
        lg = fwd(seed, E[i:i+CH], am[i:i+CH]); n = lg.shape[0]; ar = torch.arange(n, device=DEV)
        ok += int((lg[ar, pos[i:i+CH]].argmax(-1) == tgt[i:i+CH]).sum())
    return ok

# máscara de neurônios (p/ medir herança de traços)
PROBE_L = list(range(2, NL, 5)); _a = {}
def _hk(nm):
    def h(mod, inp, out): _a[nm] = inp[0].detach().abs().mean(dim=(0, 1)).float()
    return h
[model.model.layers[li].mlp.down_proj.register_forward_hook(_hk(f"L{li}")) for li in PROBE_L]
@torch.no_grad()
def mask(seed, idxs):
    _a.clear(); E, am, pos, tgt = batch(idxs[:24], 1); fwd(seed, E, am); bits = []
    for li in PROBE_L:
        v = _a[f"L{li}"]; k = max(1, int(len(v) * 0.10)); m = torch.zeros_like(v, dtype=torch.bool)
        m[torch.topk(v, k).indices] = True; bits.append(m)
    return torch.cat(bits)
def jac(a, b): return float((a & b).sum()) / float((a | b).sum() + 1e-9)

# ---- mãe ampla → nicho onde ela falha ----
log("plantando mãe (90 fatos, 800 passos)")
mother = plant(list(range(90)), 800, tseed=0); m_mask = mask(mother, list(range(90)))
miss = [i for i in range(90) if cov(mother, [i]) == 0][:30]
log(f"mãe cobre {90-len([i for i in range(90) if cov(mother,[i])==0])}/90 | nicho fraco = {len(miss)} fatos")

# ---- ablação: herdada (init=mãe+ruído) vs do zero (init aleatório) × orçamento × réplicas ----
res = {}
for steps in [150, 400, 800]:
    row = {}
    for cond, init in [("herdada", mother), ("do_zero", None)]:
        cvs, jacs = [], []
        for rep in range(3):
            child = plant(miss, steps, init=init, tseed=1000 + rep)
            cvs.append(cov(child, miss)); jacs.append(jac(mask(child, miss), m_mask))
        row[cond] = {"cobertura_media": round(float(np.mean(cvs)), 1), "reps": cvs,
                     "jaccard_vs_mae": round(float(np.mean(jacs)), 3)}
    row["vantagem_heredit"] = round(row["herdada"]["cobertura_media"] - row["do_zero"]["cobertura_media"], 1)
    res[str(steps)] = row
    log(f"  passos {steps:>3}: herdada {row['herdada']['cobertura_media']}/{len(miss)} (J{row['herdada']['jaccard_vs_mae']}) "
        f"| do_zero {row['do_zero']['cobertura_media']}/{len(miss)} (J{row['do_zero']['jaccard_vs_mae']}) "
        f"| vantagem herança {row['vantagem_heredit']:+}")

json.dump({"nicho": len(miss), "por_passos": res}, open("/home/leonardo/projects/LLM/bytebrain/research/marco116_metrics.json", "w"), ensure_ascii=False, indent=1)
veredito = "HERANÇA AJUDA" if np.mean([res[s]["vantagem_heredit"] for s in res]) > 0.5 else \
           ("HERANÇA ATRAPALHA" if np.mean([res[s]["vantagem_heredit"] for s in res]) < -0.5 else "EMPATE")
log(f"=== VEREDITO: {veredito} (vantagem média {np.mean([res[s]['vantagem_heredit'] for s in res]):+.1f} fatos) ===")
log(f"DONE M116 ({time.time()-t0:.0f}s)")
