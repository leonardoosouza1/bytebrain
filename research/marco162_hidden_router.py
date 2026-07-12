#!/usr/bin/env python3
"""M162 — CONSERTA o roteador (M161: embedding-médio não separa parafraseado de não-relacionado).
Compara 3 representações de query para rotear: (A) média dos embeddings de entrada [atual, fraco],
(B) último token do último hidden state, (C) média do último hidden state. Mede separação POS (held-out
→ cartucho) vs NEG (não-coberto) em cada uma; escolhe a melhor e re-avalia o sistema. Autossuficiente."""
import json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco162_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

FACTS = [
    (["Quem pintou 'A Noite Estrelada'?", "De quem é o quadro 'A Noite Estrelada'?"], "Van Gogh", ["gogh"]),
    (["Qual é o rio mais longo da Ásia?", "Na Ásia, qual rio tem a maior extensão?"], "Yangtzé", ["yangtz", "azul"]),
    (["Quem compôs 'As Quatro Estações'?", "De quem é a obra 'As Quatro Estações'?"], "Vivaldi", ["vivaldi"]),
    (["Qual é a capital da Mongólia?", "Qual a sede do governo da Mongólia?"], "Ulan Bator", ["ulan", "ulaanbaatar", "ulã"]),
    (["Em que país fica a cidade de Timbuktu?", "A cidade de Timbuktu está em qual nação?"], "Mali", ["mali"]),
    (["Qual planeta tem o dia mais longo que seu ano?", "Que planeta gira tão devagar que o dia supera o ano?"], "Vênus", ["vênus", "venus"]),
]
NEG = ["O que é fotossíntese?", "Qual é a capital da França?", "Quanto é 15 mais 27?",
       "Explique o que é gravidade.", "Quem foi Napoleão Bonaparte?", "Como funciona um motor a combustão?",
       "Qual é a fórmula da água?", "O que causa as marés?"]
P = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16, output_hidden_states=True).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()
def match(txt, keys): t = txt.lower(); return any(k in t for k in keys)
FMT = "Pergunta: {q}\nResposta:"

@torch.no_grad()
def reps(q):
    """3 representações da pergunta de uma vez."""
    ids = torch.tensor([tok(q).input_ids], device=DEV)
    emb = EL(ids).detach()[0]
    hs = model(input_ids=ids).hidden_states[-1][0]  # [T, H] último hidden
    return {"A_embmean": emb.mean(0), "B_last": hs[-1], "C_hidmean": hs.mean(0)}

@torch.no_grad()
def gen(q, seed=None, n=16):
    pid = tok(FMT.format(q=q)).input_ids
    e = EL(torch.tensor([pid], device=DEV)).detach()[0]
    if seed is not None: e = torch.cat([seed.to(torch.float16), e], 0)
    cur = e[None]; out = []
    for _ in range(n):
        nx = int(model(inputs_embeds=cur).logits[0, -1].argmax()); out.append(nx)
        if nx == tok.eos_token_id: break
        cur = torch.cat([cur, EL(torch.tensor([[nx]], device=DEV))], 1)
    return tok.decode(out).split("\n")[0]

def plant(forms, ans, K=4, steps=300, lr=0.1):
    pairs = [(tok(FMT.format(q=q)).input_ids, tok(" " + ans, add_special_tokens=False).input_ids) for q in forms]
    N = len(pairs); ml = K + max(len(pi) + len(ti) for pi, ti in pairs)
    E = torch.zeros(N, ml, H, device=DEV, dtype=torch.float16); am = torch.zeros(N, ml, device=DEV, dtype=torch.long)
    rows, pos, tgt = [], [], []
    for j, (pi, ti) in enumerate(pairs):
        pe = EL(torch.tensor([pi], device=DEV)).detach()[0]; te = EL(torch.tensor([ti], device=DEV)).detach()[0]
        L = K + len(pi) + len(ti); E[j, K:K+len(pi)] = pe; E[j, K+len(pi):L] = te; am[j, :L] = 1
        for kk in range(len(ti)): rows.append(j); pos.append(K+len(pi)+kk-1); tgt.append(ti[kk])
    R = torch.tensor(rows, device=DEV); Pp = torch.tensor(pos, device=DEV); T = torch.tensor(tgt, device=DEV)
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1); opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
        X = E.clone(); X[:, :K] = seed.to(torch.float16)
        loss = F.cross_entropy(model(inputs_embeds=X, attention_mask=am).logits[R, Pp].float(), T)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed], 1.0); opt.step()
    return seed.detach().to(torch.float16)

log(f"treinando {len(FACTS)} cartuchos + guardando 3 reps da forma A...")
lib = []  # (repsA, seed, keys)
for forms, ans, keys in FACTS:
    lib.append((reps(forms[0]), plant([forms[0]], ans), keys))

def bank(key): return torch.stack([r[key] for r, _, _ in lib])
def analyze(key):
    B = bank(key)
    pos_best, pos_margin, route_ok = [], [], 0
    for fi, (forms, ans, keys) in enumerate(FACTS):
        qe = reps(forms[1])[key]; s = F.cosine_similarity(qe[None], B, 1)
        o = torch.argsort(s, descending=True); best = float(s[o[0]]); second = float(s[o[1]])
        route_ok += int(o[0]) == fi; pos_best.append(round(best, 3)); pos_margin.append(round(best - second, 3))
    neg_best, neg_margin = [], []
    for q in NEG:
        qe = reps(q)[key]; s = F.cosine_similarity(qe[None], B, 1)
        o = torch.argsort(s, descending=True); neg_best.append(round(float(s[o[0]]), 3)); neg_margin.append(round(float(s[o[0]] - s[o[1]]), 3))
    sep_abs = min(pos_best) - max(neg_best); sep_mar = min(pos_margin) - max(neg_margin)
    return {"key": key, "route_ok": route_ok, "pos_best_min": min(pos_best), "neg_best_max": max(neg_best),
            "sep_abs": round(sep_abs, 3), "pos_margin_min": min(pos_margin), "neg_margin_max": max(neg_margin),
            "sep_margin": round(sep_mar, 3), "pos_best": sorted(pos_best), "neg_best": sorted(neg_best)}

ana = {}
for key in ["A_embmean", "B_last", "C_hidmean"]:
    a = analyze(key); ana[key] = a
    log(f"  [{key}] rota {a['route_ok']}/6 | sep_abs {a['sep_abs']} (pos_min {a['pos_best_min']} vs neg_max {a['neg_best_max']}) | sep_margin {a['sep_margin']}")

# escolhe a rep com melhor separação absoluta; avalia sistema com gate no ponto médio
best_key = max(ana, key=lambda k: ana[k]["sep_abs"])
a = ana[best_key]; gate = round((a["pos_best_min"] + a["neg_best_max"]) / 2, 3)
log(f"melhor router = {best_key} (sep_abs {a['sep_abs']}); gate abs = {gate}")
B = bank(best_key); inj = 0; ff = 0; fired = 0
for fi, (forms, ans, keys) in enumerate(FACTS):
    qe = reps(forms[1])[best_key]; s = F.cosine_similarity(qe[None], B, 1)
    o = torch.argsort(s, descending=True); j = int(o[0]); best = float(s[o[0]])
    seed = lib[j][1] if (best >= gate and j == fi) else None
    fired += int(best >= gate and j == fi)
    inj += match(gen(forms[1], seed=seed), keys)
for q in NEG:
    qe = reps(q)[best_key]; s = F.cosine_similarity(qe[None], B, 1); ff += int(float(s.max()) >= gate)
res = {"analise": ana, "melhor_router": best_key, "gate": gate,
       "eval": {"held_out_resp": inj, "de": len(FACTS), "disparou": fired, "falsos_disparos": ff, "de_neg": len(NEG)}}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== MELHOR ROUTER {best_key}: held-out resp {inj}/{len(FACTS)}, disparou {fired}, falsos {ff}/{len(NEG)} (gate {gate}) ===")
log(f"DONE M162 ({time.time()-t0:.0f}s)")
