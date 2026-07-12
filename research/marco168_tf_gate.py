#!/usr/bin/env python3
"""M168 — gate por PROBABILIDADE TEACHER-FORCED (robusto a comprimento/grafia, ao contrário da geração
livre do M167): aplica o seed e mede o log-prob médio do FATO armazenado como continuação da query. Query
on-topic + seed → fato vira alta-prob; off-topic → o prompt compete e o fato fica baixo. Mede POS vs NEG
p/ achar o limiar. Compara com base (sem seed). Autossuficiente. GPU."""
import time, math, unicodedata
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

FACTS = [
    ("Quem pintou 'A Noite Estrelada'?", "De quem é o quadro 'A Noite Estrelada'?", "Van Gogh"),
    ("Qual é o rio mais longo da Ásia?", "Na Ásia, qual rio tem a maior extensão?", "Yangtzé"),
    ("Quem compôs 'As Quatro Estações'?", "De quem é a obra 'As Quatro Estações'?", "Vivaldi"),
    ("Qual é a capital da Mongólia?", "Qual a sede do governo da Mongólia?", "Ulan Bator"),
]
NEG = ["O que é fotossíntese?", "Quanto é 15 mais 27?", "Quem foi Napoleão?", "Quem foi Napoleão Bonaparte?",
       "Quem escreveu Dom Casmurro?", "Qual a capital da França?", "Explique a gravidade.", "O que causa as marés?"]
P = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings(); FMT = "Pergunta: {q}\nResposta:"

@torch.no_grad()
def q_embed(q): return EL(torch.tensor([tok(q).input_ids], device=DEV)).detach()[0].mean(0)

@torch.no_grad()
def tf_logprob(q, ans, seed=None):
    """log-prob médio (por token) do FATO 'ans' como continuação de FMT(q), com/sem seed."""
    pid = tok(FMT.format(q=q)).input_ids; aid = tok(" " + ans, add_special_tokens=False).input_ids
    ep = EL(torch.tensor([pid], device=DEV)).detach()[0]; ea = EL(torch.tensor([aid], device=DEV)).detach()[0]
    parts = ([seed.to(torch.float16)] if seed is not None else []) + [ep, ea]
    full = torch.cat(parts, 0)[None]
    logits = model(inputs_embeds=full).logits[0]
    start = (seed.shape[0] if seed is not None else 0) + len(pid)  # 1º token do fato
    lp = 0.0
    for i, t in enumerate(aid):
        lp += float(F.log_softmax(logits[start - 1 + i].float(), -1)[t])
    return lp / len(aid)

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
lib = [(q_embed(tq), plant(tq, ans), ans, tq[:20]) for tq, hq, ans in FACTS]
embs = torch.stack([e for e, _, _, _ in lib])
def top1(q):
    qe = q_embed(q); s = F.cosine_similarity(qe[None], embs, 1); return int(s.argmax())

log("== POSITIVOS: tf-logprob do fato do top-1 cartucho (com seed) vs base (sem) ==")
pos_ws = []
for tq, hq, ans in FACTS:
    j = top1(hq); ws = tf_logprob(hq, lib[j][2], lib[j][1]); bs = tf_logprob(hq, lib[j][2], None)
    pos_ws.append(ws); log(f"  '{hq[:32]:32}' top1={lib[j][3]:20} seed={ws:+.2f} base={bs:+.2f} Δ={ws-bs:+.2f}")
log("== NEGATIVOS: tf-logprob do fato do top-1 cartucho (com seed) ==")
neg_ws = []
for q in NEG:
    j = top1(q); ws = tf_logprob(q, lib[j][2], lib[j][1]); bs = tf_logprob(q, lib[j][2], None)
    neg_ws.append(ws); log(f"  '{q[:32]:32}' top1={lib[j][3]:20} seed={ws:+.2f} base={bs:+.2f} Δ={ws-bs:+.2f}")
log(f"POS seed-logprob: min {min(pos_ws):+.2f}  |  NEG seed-logprob: max {max(neg_ws):+.2f}  |  separável? {min(pos_ws) > max(neg_ws)}")
thr = round((min(pos_ws) + max(neg_ws)) / 2, 2)
log(f"limiar sugerido = {thr}  (dispara se tf-logprob do fato com seed >= {thr})")
log(f"DONE M168 ({time.time()-t0:.0f}s)")
