#!/usr/bin/env python3
"""M166 — GATE POR VERIFICAÇÃO (conserta o elo fraco M162): em vez de decidir 'usar cartucho?' só pela
similaridade de embedding (que não separa parafraseado de não-relacionado), o PRÓPRIO MODELO verifica:
roteia top-1 por embedding (barato) e DISPARA só se, ao aplicar o seed, ele produz o FATO armazenado do
cartucho. Query não-relacionada + seed → não produz o fato (o prompt vence) → não dispara. Compara com o
gate por margem. Autossuficiente. GPU."""
import json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco166_metrics.json"
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
def gen(q, seed=None, n=12):
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

log(f"treinando {len(FACTS)} cartuchos (forma A)...")
lib = []  # (emb, seed, keys, ans)
for forms, ans, keys in FACTS:
    lib.append((q_embed(forms[0]), plant([forms[0]], ans), keys, ans))
embs = torch.stack([e for e, _, _, _ in lib])

def route_top(q):
    qe = q_embed(q); s = F.cosine_similarity(qe[None], embs, 1)
    o = torch.argsort(s, descending=True); return int(o[0]), float(s[o[0]]), float(s[o[0]] - s[o[1]])

# GATE MARGEM: dispara se margem>=0.20 (baseline M162)
# GATE VERIFY: roteia top-1, aplica o seed, dispara SÓ se produz o fato armazenado desse cartucho
def eval_gate(mode):
    pos_fire = 0; pos_correct = 0; neg_fire = 0
    for fi, (forms, ans, keys) in enumerate(FACTS):
        q = forms[1]  # held-out
        j, best, margin = route_top(q)
        if mode == "margin":
            fire = margin >= 0.20 and best >= 0.45
        else:  # verify
            out = gen(q, seed=lib[j][1]); fire = match(out, lib[j][2])
        if fire and j == fi: pos_fire += 1
        r = gen(q, seed=lib[j][1] if (fire and j == fi) else None); pos_correct += match(r, keys)
    for q in NEG:
        j, best, margin = route_top(q)
        if mode == "margin":
            fire = margin >= 0.20 and best >= 0.45
        else:
            out = gen(q, seed=lib[j][1]); fire = match(out, lib[j][2])
        neg_fire += fire
    return {"mode": mode, "pos_fire": pos_fire, "pos_correct": pos_correct, "de_pos": len(FACTS), "neg_false_fire": neg_fire, "de_neg": len(NEG)}

res = {}
for mode in ["margin", "verify"]:
    r = eval_gate(mode); res[mode] = r
    log(f"  GATE {mode}: dispara {r['pos_fire']}/{r['de_pos']} parafraseados | resp certa {r['pos_correct']}/{r['de_pos']} | FALSOS {r['neg_false_fire']}/{r['de_neg']}")
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== VERIFY vs MARGEM: falsos {res['verify']['neg_false_fire']} vs {res['margin']['neg_false_fire']} (de {len(NEG)}); recall {res['verify']['pos_fire']} vs {res['margin']['pos_fire']} ===")
log(f"DONE M166 ({time.time()-t0:.0f}s)")
