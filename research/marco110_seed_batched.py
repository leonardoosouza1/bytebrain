#!/usr/bin/env python3
"""M110 — semente-de-fato BATCHED (conserta o gargalo do M109: 1 forward p/ todos os fatos).
Empurra ~120 fatos numa semente compartilhada, varre K, VERIFICA int4 no seed denso, acha o teto
real e o piso de bytes/fato. Modelo congelado = decoder. GPU. Dump marco110_metrics.json."""
import json, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"; DEV = "cuda"; t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()
random.seed(0)

# banco grande de fatos (arbitrários: o modelo NÃO pode saber → semente ARMAZENA)
CAPS = {"França":"Paris","Japão":"Tóquio","Itália":"Roma","Alemanha":"Berlim","Espanha":"Madri",
        "Rússia":"Moscou","China":"Pequim","Egito":"Cairo","Peru":"Lima","Chile":"Santiago"}
NAMES = ["Ziggy","Krylon","Miau","Rex","Nova","Pixel","Turbo","Cosmo","Bit","Echo","Lumi","Vega"]
COLORS = ["roxo","azul","verde","vermelho","dourado","prata","laranja","rosa"]
facts = []
for k, v in CAPS.items(): facts.append((f"A capital de {k} é", f" {v}"))
for i in range(40):  # associações inventadas (segredo N -> número aleatório de 4 dígitos)
    facts.append((f"O código secreto número {i} é", f" {random.randint(1000,9999)}"))
for i, n in enumerate(NAMES): facts.append((f"O nome do robô {i} é", f" {n}"))
for i, c in enumerate(COLORS): facts.append((f"A cor do domínio {i} é", f" {c}"))
for a in range(2, 10):
    for b in range(2, 8): facts.append((f"{a} vezes {b} é", f" {a*b}"))
random.shuffle(facts); facts = facts[:120]
N = len(facts)
print(f"congelado hidden {H} | {N} fatos ({time.time()-t0:.0f}s)", flush=True)

# pré-tokeniza
P = [(tok(p).input_ids, tok(tg, add_special_tokens=False).input_ids) for p, tg in facts]

def batch(K):
    maxlen = K + max(len(pi) + len(ti) for pi, ti in P)
    E = torch.zeros(N, maxlen, H, device=DEV, dtype=torch.float16)
    am = torch.zeros(N, maxlen, device=DEV, dtype=torch.long)
    tpos, tids = [], []
    for i, (pi, ti) in enumerate(P):
        pe = EL(torch.tensor([pi], device=DEV)).detach()[0]
        te = EL(torch.tensor([ti], device=DEV)).detach()[0]
        L = K + len(pi) + len(ti)
        E[i, K:K+len(pi)] = pe; E[i, K+len(pi):L] = te; am[i, :L] = 1
        tpos.append((i, K+len(pi), len(ti))); tids.append(ti)
    return E, am, tpos, tids

def run(seed, E, am):
    E = E.clone(); E[:, :seed.shape[0]] = seed.to(torch.float16)
    return model(inputs_embeds=E, attention_mask=am).logits

def plant(K, steps=300):
    E, am, tpos, tids = batch(K)
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([seed], lr=0.3)
    for s in range(steps):
        lg = run(seed, E, am)
        loss = 0
        for (i, st, lt), ti in zip(tpos, tids):
            loss = loss + F.cross_entropy(lg[i, st-1:st-1+lt].float(), torch.tensor(ti, device=DEV))
        (loss / N).backward(); opt.step(); opt.zero_grad()
    return seed.detach().to(torch.float16), E, am, tpos, tids

@torch.no_grad()
def recall(seed, E, am, tpos, tids):
    lg = run(seed, E, am); ok = 0
    for (i, st, lt), ti in zip(tpos, tids):
        ok += bool((lg[i, st-1:st-1+lt].argmax(-1) == torch.tensor(ti, device=DEV)).all())
    return ok

def q_int4(seed):
    qm = 7; s = (seed.abs().max() / qm).clamp_min(1e-8)
    return ((seed.float() / s).round().clamp(-qm, qm) * s).to(torch.float16)

res = {}
for K in [1, 4, 16, 64]:
    seed, E, am, tp, td = plant(K)
    ok = recall(seed, E, am, tp, td)
    ok4 = recall(q_int4(seed), E, am, tp, td)
    kb = K * H * 4 / 8 / 1024
    res[K] = {"fp16_ok": ok, "int4_ok": ok4, "de": N, "KB_int4": round(kb, 2),
              "bytes_por_fato": round(kb * 1024 / max(1, ok4), 1)}
    print(f"  K={K:>2} ({kb:.1f}KB int4): fp16 {ok}/{N} | int4 {ok4}/{N} | {res[K]['bytes_por_fato']}B/fato ({time.time()-t0:.0f}s)", flush=True)

json.dump({"N": N, "res": res}, open("/home/leonardo/projects/LLM/bytebrain/research/marco110_metrics.json", "w"))
print(f"\nDONE M110 ({time.time()-t0:.0f}s)", flush=True)
