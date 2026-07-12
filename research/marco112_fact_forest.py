#!/usr/bin/env python3
"""M112 v2 — BATERIA / FATOS (decoder congelado), CORRIGIDA: otimizador ESTÁVEL (lr0.1 cosine + grad-clip,
reprodutível: era instável a lr0.3 → seeds davam 0-20 na mesma config) + forward EM CHUNKS (conserta OOM).
Fatos PURAMENTE ARBITRÁRIOS (prompt único → palavra 1-token aleatória = armazenamento puro).
 A) ESCALA apples-to-apples: FLORESTA 10×(K=1) vs UMA semente K=10 (MESMOS bytes) em 200 fatos.
 R) REGA: curva recall × K (K=1..16) em 60 fatos — quantifica o resgate por "regar".
 B1) o fato aguenta PERTURBAÇÃO dos pesos do decoder? (mora na semente ou no decoder?)
 B2) modelo RANDOM vs TREINADO — a SABEDORIA barateia o armazenamento?
GPU. Dump incremental marco112_metrics.json (resumível a cada frente)."""
import json, time, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

M = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco112_metrics.json"; CH = 48
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()
random.seed(0)
log(f"M112v2 START | Math-1.5B congelado hidden {H}")

# pool de alvos 1-token do próprio vocabulário (recall limpo)
POOL = []
for tid in range(min(len(tok), 60000)):
    s = tok.decode([tid])
    if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3 <= len(s.strip()) <= 8:
        if len(tok(s, add_special_tokens=False).input_ids) == 1: POOL.append(s)
    if len(POOL) >= 80: break
def make_facts(n):
    tp = ["O código secreto número {i} é", "O nome do robô {i} é", "A cor do domínio {i} é",
          "A senha do cofre {i} é", "O valor da chave {i} é"]
    return [(tp[i % 5].format(i=i), POOL[random.randrange(len(POOL))]) for i in range(n)]
BANK = make_facts(600)
log(f"pool 1-token: {len(POOL)} | banco {len(BANK)} fatos ARBITRÁRIOS")

def prep(facts): return [(tok(p).input_ids, tok(tg, add_special_tokens=False).input_ids[:1]) for p, tg in facts]

def batch(P, K):
    N = len(P); maxlen = K + max(len(pi) + 1 for pi, _ in P)
    E = torch.zeros(N, maxlen, H, device=DEV, dtype=torch.float16)
    am = torch.zeros(N, maxlen, device=DEV, dtype=torch.long)
    pos = torch.zeros(N, dtype=torch.long, device=DEV); tgt = torch.zeros(N, dtype=torch.long, device=DEV)
    for i, (pi, ti) in enumerate(P):
        pe = EL(torch.tensor([pi], device=DEV)).detach()[0]; te = EL(torch.tensor([ti], device=DEV)).detach()[0]
        L = K + len(pi) + 1; E[i, K:K+len(pi)] = pe; E[i, K+len(pi):L] = te; am[i, :L] = 1
        pos[i] = K + len(pi) - 1; tgt[i] = ti[0]
    return E, am, pos, tgt

def fwd_chunk(mdl, seed, Ec, amc):
    X = Ec.clone(); X[:, :seed.shape[0]] = seed.to(torch.float16)
    return mdl(inputs_embeds=X, attention_mask=amc).logits

def plant(mdl, P, K, steps=800, lr=0.1, clip=1.0, tseed=0):
    E, am, pos, tgt = batch(P, K); N = len(P)
    g = torch.Generator(device=DEV).manual_seed(tseed)
    seed = nn.Parameter(torch.randn(K, H, generator=g, device=DEV, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
        opt.zero_grad()
        for i in range(0, N, CH):
            lg = fwd_chunk(mdl, seed, E[i:i+CH], am[i:i+CH])
            n = lg.shape[0]; ar = torch.arange(n, device=DEV)
            l = F.cross_entropy(lg[ar, pos[i:i+CH]].float(), tgt[i:i+CH], reduction="sum")
            (l / N).backward()
        if clip: torch.nn.utils.clip_grad_norm_([seed], clip)
        opt.step()
    return seed.detach().to(torch.float16), E, am, pos, tgt

@torch.no_grad()
def recall(mdl, seed, E, am, pos, tgt):
    ok = 0
    for i in range(0, len(pos), CH):
        lg = fwd_chunk(mdl, seed, E[i:i+CH], am[i:i+CH]); n = lg.shape[0]; ar = torch.arange(n, device=DEV)
        ok += int((lg[ar, pos[i:i+CH]].argmax(-1) == tgt[i:i+CH]).sum())
    return ok

def quant(seed, bits, group=0):
    """quantização simétrica; group>0 = escala POR GRUPO ao longo de H (padrão real GPTQ/AWQ)."""
    qm = 2 ** (bits - 1) - 1; x = seed.float()
    if group and x.shape[1] % group == 0:
        K, Hd = x.shape; xg = x.reshape(K, Hd // group, group)
        s = (xg.abs().amax(-1, keepdim=True) / qm).clamp_min(1e-8)
        return (torch.round(xg / s).clamp(-qm, qm) * s).reshape(K, Hd).to(torch.float16)
    s = (x.abs().max() / qm).clamp_min(1e-8)
    return (torch.round(x / s).clamp(-qm, qm) * s).to(torch.float16)
QK = {"i4_pertensor": lambda s: quant(s, 4, 0), "i4_grupo128": lambda s: quant(s, 4, 128),
      "i8_grupo128": lambda s: quant(s, 8, 128)}

RES = {}
def dump(): json.dump(RES, open(OUT, "w"), ensure_ascii=False, indent=1)

# ================= A) FLORESTA 10×K1 vs SEMENTE K=10 (mesmos bytes) =================
log("=== A) FLORESTA 10×(K=1,20 fatos) vs SEMENTE K=10 — mesmos bytes, 200 fatos ===")
NF, PER = 200, 20; Pall = prep(BANK[:NF])
fs = {"fp16": [], **{k: [] for k in QK}}
for j, s in enumerate(range(0, NF, PER)):
    seed, E, am, pos, tgt = plant(model, Pall[s:s+PER], 1, tseed=100 + j)
    fs["fp16"].append(recall(model, seed, E, am, pos, tgt))
    for k, q in QK.items(): fs[k].append(recall(model, q(seed), E, am, pos, tgt))
    log(f"  A.floresta semente {j:2d}: fp16 {fs['fp16'][-1]} | i4pt {fs['i4_pertensor'][-1]} | "
        f"i4g {fs['i4_grupo128'][-1]} | i8g {fs['i8_grupo128'][-1]} (/{PER})")
nseeds = len(fs["fp16"]); ftot = {k: sum(v) for k, v in fs.items()}
seed, E, am, pos, tgt = plant(model, Pall, nseeds)  # semente única K=nseeds, chunked → sem OOM
single = {"fp16": recall(model, seed, E, am, pos, tgt)}
for k, q in QK.items(): single[k] = recall(model, q(seed), E, am, pos, tgt)
RES["A"] = {"de": NF, "n_vetores": nseeds, "forest": ftot, "single": single,
            "bytes_int4": int(nseeds * H * 4 / 8),
            "bytes_por_fato_forest_i4g": round(nseeds * H * 4 / 8 / max(1, ftot["i4_grupo128"]), 1),
            "veredito_i4g": "floresta" if ftot["i4_grupo128"] > single["i4_grupo128"] else "unica"}
log(f"  A.FLORESTA(10×K1) fp16 {ftot['fp16']}/{NF} i4pt {ftot['i4_pertensor']} i4g {ftot['i4_grupo128']} i8g {ftot['i8_grupo128']}")
log(f"  A.SEMENTE K={nseeds}   fp16 {single['fp16']}/{NF} i4pt {single['i4_pertensor']} i4g {single['i4_grupo128']} i8g {single['i8_grupo128']}")
log(f"  A.VEREDITO i4-grupo (mesmos {nseeds} vetores): {RES['A']['veredito_i4g'].upper()} | "
    f"{RES['A']['bytes_por_fato_forest_i4g']} B/fato"); dump()

# ================= R) REGA: recall × K =================
log("=== R) REGA: 60 fatos, curva recall × K (regar = +K) ===")
Pr = prep(BANK[:60]); rega = {}
for K in [1, 2, 4, 8, 16]:
    seed, E, am, pos, tgt = plant(model, Pr, K)
    ok16 = recall(model, seed, E, am, pos, tgt)
    q = {k: recall(model, f(seed), E, am, pos, tgt) for k, f in QK.items()}
    kb = K * H * 4 / 8 / 1024
    rega[str(K)] = {"fp16": ok16, **q, "de": len(Pr), "KB_int4": round(kb, 2),
                    "bytes_por_fato_i4g": round(kb * 1024 / max(1, q["i4_grupo128"]), 1)}
    log(f"  R.K={K:>2} ({kb:.2f}KB): fp16 {ok16} | i4pt {q['i4_pertensor']} | i4g {q['i4_grupo128']} | "
        f"i8g {q['i8_grupo128']} (/60) | {rega[str(K)]['bytes_por_fato_i4g']} B/fato i4g")
RES["R_rega"] = rega; dump()

# ================= B1) PERTURBAÇÃO do decoder =================
log("=== B1) o fato aguenta ruído nos pesos do decoder? ===")
Pb = prep(BANK[:40]); seed_b, Eb, amb, posb, tgtb = plant(model, Pb, 4)
base_ok = recall(model, seed_b, Eb, amb, posb, tgtb)
layers = [p for _, p in model.named_parameters() if p.dim() >= 2]
snap = [p.detach().clone() for p in layers]; pert = {}; gen = torch.Generator(device=DEV).manual_seed(0)
for sigma in [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    with torch.no_grad():
        for p, s0 in zip(layers, snap):
            p.copy_(s0)
            if sigma > 0: p.add_(torch.randn(p.shape, generator=gen, device=DEV, dtype=p.dtype) * (s0.std() * sigma))
    pert[str(sigma)] = recall(model, seed_b, Eb, amb, posb, tgtb)
    log(f"  B1.sigma {sigma:>6}: recall {pert[str(sigma)]}/{len(Pb)} (base {base_ok})")
with torch.no_grad():
    for p, s0 in zip(layers, snap): p.copy_(s0)
del snap; torch.cuda.empty_cache()
RES["B1_perturbacao"] = {"base": base_ok, "de": len(Pb), "por_sigma": pert}; dump()

# ================= B2) modelo RANDOM vs TREINADO (a sabedoria barateia?) =================
log("=== B2) modelo RANDOM vs TREINADO — a sabedoria barateia o armazenamento? ===")
cfg = AutoConfig.from_pretrained(M)
rmodel = AutoModelForCausalLM.from_config(cfg).to(DEV).to(torch.float16).eval()
for p in rmodel.parameters(): p.requires_grad_(False)
b2 = {}
for tag, mdl in [("treinado", model), ("random", rmodel)]:
    row = {}
    for K in [1, 4, 8]:
        seed, E, am, pos, tgt = plant(mdl, Pb, K)
        row[str(K)] = {"fp16": recall(mdl, seed, E, am, pos, tgt),
                       "i4g": recall(mdl, quant(seed, 4, 128), E, am, pos, tgt)}
    b2[tag] = row
    log(f"  B2.{tag:8}: " + " | ".join(f"K{k} fp16 {row[k]['fp16']} i4g {row[k]['i4g']}" for k in ["1","4","8"]))
RES["B2_random_vs_treinado"] = {"de": len(Pb), **b2}; dump()
del rmodel; torch.cuda.empty_cache()

log(f"DONE M112v2 ({time.time()-t0:.0f}s)")
