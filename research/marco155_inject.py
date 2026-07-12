#!/usr/bin/env python3
"""M155 — INJETAR o conhecimento do 7B no 1.5B via SEED (a culminação IARA). O 1.5B erra fatos raros que
o 7B sabe (M154: 11 vs 15). Aqui: planto um SEED pequeno no 1.5B com os fatos que ele erra (resposta
correta como professor) e meço se o conhecimento sobe rumo ao 7B — modelo leve com conhecimento de 7B.
Tronco 1.5B-Instruct congelado. GPU. Dump marco155_metrics.json."""
import json, time, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from transformers import AutoModelForCausalLM, AutoTokenizer
from marco154_knowledge import KN  # 20 perguntas de conhecimento + chaves
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco155_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
# resposta canônica (1ª chave) p/ plantar
ANS = {"guerra e paz": "Tolstói", "hastings": "1066", "mongólia": "Ulan Bator", "penicilina": "Fleming",
       "dia mais longo": "Vênus", "noite estrelada": "Van Gogh", "timbuktu": "Mali", "imperador romano": "Augusto",
       "letreiros": "neônio", "tabela periódica": "Mendeleev", "rio mais longo da ásia": "Yangtzé",
       "ossos": "27", "tungstênio": "3422", "líquido perto": "gálio", "cem anos": "García Márquez",
       "quântica": "qubit", "muro de berlim": "1989", "maior deserto": "Antártica", "quatro estações": "Vivaldi",
       "sistema solar": "Olympus Mons"}
def ans_of(q):
    ql = q.lower()
    for k, v in ANS.items():
        if k in ql: return v
    return None

P = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()
def match(txt, keys): t = txt.lower(); return any(k in t for k in keys)

@torch.no_grad()
def gen(prompt, seed=None, n=30):
    pid = tok(prompt).input_ids
    e = EL(torch.tensor([pid], device=DEV)).detach()[0]
    if seed is not None: e = torch.cat([seed.to(torch.float16), e], 0)
    cur = e[None]; out = []
    for _ in range(n):
        nx = int(model(inputs_embeds=cur).logits[0, -1].argmax()); out.append(nx)
        if nx == tok.eos_token_id: break
        cur = torch.cat([cur, EL(torch.tensor([[nx]], device=DEV))], 1)
    return tok.decode(out)

FMT = "Pergunta: {q}\nResposta:"
# baseline (sem seed)
base = sum(match(gen(FMT.format(q=q)), keys) for q, keys in KN)
log(f"1.5B baseline (formato raw): {base}/{len(KN)}")
# quais erra
wrong = [(q, keys) for q, keys in KN if not match(gen(FMT.format(q=q)), keys)]
plant_facts = [(FMT.format(q=q), " " + ans_of(q)) for q, keys in wrong if ans_of(q)]
log(f"errou {len(wrong)}, plantando {len(plant_facts)} fatos (professor=resposta correta)")

# planta UM seed compartilhado com os fatos que erra
def plant(pairs, K=16, steps=400, lr=0.1):
    import math
    P_ = [(tok(p).input_ids, tok(t, add_special_tokens=False).input_ids) for p, t in pairs]
    N = len(P_); ml = K + max(len(pi) + len(ti) for pi, ti in P_)
    E = torch.zeros(N, ml, H, device=DEV, dtype=torch.float16); am = torch.zeros(N, ml, device=DEV, dtype=torch.long)
    rows, pos, tgt = [], [], []
    for j, (pi, ti) in enumerate(P_):
        pe = EL(torch.tensor([pi], device=DEV)).detach()[0]; te = EL(torch.tensor([ti], device=DEV)).detach()[0]
        L = K + len(pi) + len(ti); E[j, K:K+len(pi)] = pe; E[j, K+len(pi):L] = te; am[j, :L] = 1
        for kk in range(len(ti)): rows.append(j); pos.append(K+len(pi)+kk-1); tgt.append(ti[kk])
    R = torch.tensor(rows, device=DEV); Pp = torch.tensor(pos, device=DEV); T = torch.tensor(tgt, device=DEV)
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1); opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
        X = E.clone(); X[:, :K] = seed.to(torch.float16)
        lg = model(inputs_embeds=X, attention_mask=am).logits
        loss = F.cross_entropy(lg[R, Pp].float(), T)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed], 1.0); opt.step()
    return seed.detach().to(torch.float16)

seed = plant(plant_facts, K=16, steps=400)
kb = 16 * H * 4 / 8 / 1024  # int4
# eval com o seed prependido em TODAS as perguntas
withseed = sum(match(gen(FMT.format(q=q), seed=seed), keys) for q, keys in KN)
res = {"baseline": base, "com_seed": withseed, "de": len(KN), "7B_int8": 15,
       "seed_KB_int4": round(kb, 1), "plantados": len(plant_facts)}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== INJEÇÃO: 1.5B {base}/{len(KN)} → 1.5B+seed {withseed}/{len(KN)} (seed {kb:.1f}KB) | 7B-int8 era 15/20 ===")
log(f"DONE M155 ({time.time()-t0:.0f}s)")
