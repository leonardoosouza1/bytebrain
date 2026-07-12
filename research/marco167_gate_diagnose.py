#!/usr/bin/env python3
"""M167 — DIAGNÓSTICO do gate: pra cada parafraseado (positivo) e não-relacionado (negativo), imprime
top-1 cartucho, similaridade, margem e VERIFY-hit (gen com top-1 seed contém o fato?). Com isso escolho a
regra de gate ótima sem chutar. 4 cartuchos (como no artefato). Autossuficiente. GPU."""
import time, math, unicodedata
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
def norm(s): return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")

FACTS = [  # (forma treino, forma held-out, resposta)
    ("Quem pintou 'A Noite Estrelada'?", "De quem é o quadro 'A Noite Estrelada'?", "Van Gogh"),
    ("Qual é o rio mais longo da Ásia?", "Na Ásia, qual rio tem a maior extensão?", "Yangtzé"),
    ("Quem compôs 'As Quatro Estações'?", "De quem é a obra 'As Quatro Estações'?", "Vivaldi"),
    ("Qual é a capital da Mongólia?", "Qual a sede do governo da Mongólia?", "Ulan Bator"),
]
NEG = ["O que é fotossíntese?", "Quanto é 15 mais 27?", "Quem foi Napoleão?", "Quem foi Napoleão Bonaparte?",
       "Quem escreveu Dom Casmurro?", "Qual a capital da França?"]
P = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings(); FMT = "Pergunta: {q}\nResposta:"

@torch.no_grad()
def q_embed(q):
    return EL(torch.tensor([tok(q).input_ids], device=DEV)).detach()[0].mean(0)
@torch.no_grad()
def gen(q, seed, n=14):
    e = EL(torch.tensor([tok(FMT.format(q=q)).input_ids], device=DEV)).detach()[0]
    cur = torch.cat([seed.to(torch.float16), e], 0)[None]; out = []
    for _ in range(n):
        nx = int(model(inputs_embeds=cur).logits[0, -1].argmax()); out.append(nx)
        cur = torch.cat([cur, EL(torch.tensor([[nx]], device=DEV))], 1)
    return tok.decode(out).split("\n")[0]
def plant(q, ans, K=4, steps=300, lr=0.1):
    pi = tok(FMT.format(q=q)).input_ids; ti = tok(" " + ans, add_special_tokens=False).input_ids
    pe = EL(torch.tensor([pi], device=DEV)).detach()[0]; te = EL(torch.tensor([ti], device=DEV)).detach()[0]
    L = K + len(pi) + len(ti); E = torch.zeros(1, L, H, device=DEV, dtype=torch.float16)
    E[0, K:K+len(pi)] = pe; E[0, K+len(pi):L] = te
    Pp = torch.tensor([K+len(pi)+kk-1 for kk in range(len(ti))], device=DEV); T = torch.tensor(ti, device=DEV)
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32)*0.1); opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr*0.5*(1+math.cos(math.pi*s/steps))
        X = E.clone(); X[:, :K] = seed.to(torch.float16)
        loss = F.cross_entropy(model(inputs_embeds=X).logits[0, Pp].float(), T)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed], 1.0); opt.step()
    return seed.detach().to(torch.float16)

log("treinando 4 cartuchos...")
lib = [(q_embed(tq), plant(tq, ans), ans, tq[:22]) for tq, hq, ans in FACTS]
embs = torch.stack([e for e, _, _, _ in lib])
def diag(q):
    qe = q_embed(q); s = F.cosine_similarity(qe[None], embs, 1); o = torch.argsort(s, descending=True)
    j = int(o[0]); best = float(s[o[0]]); margin = best - float(s[o[1]])
    out = gen(q, lib[j][1]); hit = norm(lib[j][2]) in norm(out)
    return j, best, margin, hit, out[:24]
log("== POSITIVOS (parafraseados held-out) ==")
for tq, hq, ans in FACTS:
    j, best, margin, hit, out = diag(hq)
    log(f"  '{hq[:34]:34}' -> top1={lib[j][3]:22} sim={best:.3f} margem={margin:.3f} verify={'HIT' if hit else 'no '} | {out!r}")
log("== NEGATIVOS (não-relacionados) ==")
for q in NEG:
    j, best, margin, hit, out = diag(q)
    log(f"  '{q[:34]:34}' -> top1={lib[j][3]:22} sim={best:.3f} margem={margin:.3f} verify={'HIT' if hit else 'no '} | {out!r}")
log(f"DONE M167 ({time.time()-t0:.0f}s)")
