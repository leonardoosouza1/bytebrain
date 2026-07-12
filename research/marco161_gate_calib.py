#!/usr/bin/env python3
"""M161 — CALIBRA o gate do roteador (M160 mostrou: gate 0.97 é cego a parafraseados que ficam ~0.7).
Mede a distribuição de similaridade: POSITIVOS (parafraseado held-out → seu cartucho) vs NEGATIVOS
(perguntas não-relacionadas → cartucho mais próximo). Acha um gate que separa e re-avalia o sistema
com: (a) gate absoluto calibrado, (b) gate por MARGEM relativa (best - 2ndbest). Autossuficiente. GPU."""
import json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco161_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

FACTS = [  # [treino A, held-out C], resposta, chaves
    (["Quem pintou 'A Noite Estrelada'?", "De quem é o quadro 'A Noite Estrelada'?"], "Van Gogh", ["gogh"]),
    (["Qual é o rio mais longo da Ásia?", "Na Ásia, qual rio tem a maior extensão?"], "Yangtzé", ["yangtz", "azul"]),
    (["Quem compôs 'As Quatro Estações'?", "De quem é a obra 'As Quatro Estações'?"], "Vivaldi", ["vivaldi"]),
    (["Qual é a capital da Mongólia?", "Qual a sede do governo da Mongólia?"], "Ulan Bator", ["ulan", "ulaanbaatar", "ulã"]),
    (["Em que país fica a cidade de Timbuktu?", "A cidade de Timbuktu está em qual nação?"], "Mali", ["mali"]),
    (["Qual planeta tem o dia mais longo que seu ano?", "Que planeta gira tão devagar que o dia supera o ano?"], "Vênus", ["vênus", "venus"]),
]
NEG = [  # perguntas NÃO cobertas por cartucho — não podem disparar
    "O que é fotossíntese?", "Qual é a capital da França?", "Quanto é 15 mais 27?",
    "Explique o que é gravidade.", "Quem foi Napoleão Bonaparte?", "Como funciona um motor a combustão?",
    "Qual é a fórmula da água?", "O que causa as marés?",
]
P = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(P)
model = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()
def match(txt, keys): t = txt.lower(); return any(k in t for k in keys)
FMT = "Pergunta: {q}\nResposta:"

@torch.no_grad()
def q_embed(q):
    ids = tok(q).input_ids
    return EL(torch.tensor([ids], device=DEV)).detach()[0].mean(0)

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

# treina 1 cartucho por fato na formulação A (idx 0); guarda embedding de A
log(f"treinando {len(FACTS)} cartuchos (forma A)...")
lib = []
for forms, ans, keys in FACTS:
    lib.append((q_embed(forms[0]), plant([forms[0]], ans), keys, ans))
embs = torch.stack([e for e, _, _, _ in lib])

def topsims(q):
    qe = q_embed(q); s = F.cosine_similarity(qe[None], embs, 1)
    order = torch.argsort(s, descending=True)
    return s, int(order[0]), float(s[order[0]]), float(s[order[1]])

# POSITIVOS: held-out C (idx 1)
pos = []
for fi, (forms, ans, keys) in enumerate(FACTS):
    s, j, best, second = topsims(forms[1])
    pos.append({"fi": fi, "route_ok": j == fi, "best": round(best, 3), "margin": round(best - second, 3)})
# NEGATIVOS: perguntas não-cobertas
neg = []
for q in NEG:
    s, j, best, second = topsims(q)
    neg.append({"best": round(best, 3), "margin": round(best - second, 3)})

pos_best = [p["best"] for p in pos]; neg_best = [n["best"] for n in neg]
pos_margin = [p["margin"] for p in pos]; neg_margin = [n["margin"] for n in neg]
log(f"POS best sims: {sorted(pos_best)}  (min {min(pos_best)})")
log(f"NEG best sims: {sorted(neg_best)}  (max {max(neg_best)})")
log(f"POS margins:   {sorted(pos_margin)}  (min {min(pos_margin)})")
log(f"NEG margins:   {sorted(neg_margin)}  (max {max(neg_margin)})")

# escolhe gate absoluto = ponto médio entre min(pos) e max(neg) se separável
sep_abs = min(pos_best) > max(neg_best)
gate_abs = round((min(pos_best) + max(neg_best)) / 2, 3) if sep_abs else None
sep_mar = min(pos_margin) > max(neg_margin)
gate_mar = round((min(pos_margin) + max(neg_margin)) / 2, 3) if sep_mar else None
log(f"separável por sim absoluta? {sep_abs} (gate={gate_abs}) | por margem? {sep_mar} (gate={gate_mar})")

# re-avalia sistema com gate escolhido (prefere margem se separável, senão absoluto)
def evaluate(gate, mode):
    inject = 0; false_fire = 0; answered = 0
    for fi, (forms, ans, keys) in enumerate(FACTS):
        s, j, best, second = topsims(forms[1]); margin = best - second
        fire = (margin >= gate) if mode == "margin" else (best >= gate)
        seed = lib[j][1] if (fire and j == fi) else None
        if fire and j == fi: answered += 1
        r = gen(forms[1], seed=seed); inject += match(r, keys)
    for q in NEG:
        s, j, best, second = topsims(q); margin = best - second
        fire = (margin >= gate) if mode == "margin" else (best >= gate)
        false_fire += fire
    return {"gate": gate, "mode": mode, "held_out_resp": inject, "de": len(FACTS), "disparou": answered, "falsos_disparos": false_fire, "de_neg": len(NEG)}

results = {"pos": pos, "neg": neg, "sep_abs": sep_abs, "gate_abs": gate_abs, "sep_margin": sep_mar, "gate_margin": gate_mar, "evals": []}
if gate_abs is not None: results["evals"].append(evaluate(gate_abs, "abs"))
if gate_mar is not None: results["evals"].append(evaluate(gate_mar, "margin"))
# fallback: testa gate absoluto 0.55 mesmo se não separável perfeito
results["evals"].append(evaluate(0.55, "abs"))
for e in results["evals"]:
    log(f"  EVAL {e['mode']} gate={e['gate']}: held-out resp {e['held_out_resp']}/{e['de']}, disparou {e['disparou']}, falsos {e['falsos_disparos']}/{e['de_neg']}")
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"DONE M161 ({time.time()-t0:.0f}s)")
