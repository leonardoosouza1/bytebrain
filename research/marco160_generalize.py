#!/usr/bin/env python3
"""M160 — TESTE DE GENERALIZAÇÃO do artefato roteado (separa conhecimento real de decoreba/lookup):
treino cartuchos numa formulação e testo em formulações HELD-OUT (reformuladas). Duas variantes:
V1 = cartucho treinado em 1 formulação; V2 = treinado em 2 formulações (M123: multi-form generaliza).
Meço: (a) roteador ainda dispara no parafraseado? (b) o cartucho ainda responde certo? + varre o gate.
Autossuficiente. GPU."""
import json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco160_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# cada fato: (formulações [A, B, C_heldout], resposta a plantar, chaves de match)
FACTS = [
    (["Quem pintou 'A Noite Estrelada'?", "Quem é o autor da pintura 'A Noite Estrelada'?", "De quem é o quadro 'A Noite Estrelada'?"], "Van Gogh", ["gogh"]),
    (["Qual é o rio mais longo da Ásia?", "Qual o maior rio do continente asiático?", "Na Ásia, qual rio tem a maior extensão?"], "Yangtzé", ["yangtz", "azul"]),
    (["Quem compôs 'As Quatro Estações'?", "Quem é o compositor de 'As Quatro Estações'?", "De quem é a obra 'As Quatro Estações'?"], "Vivaldi", ["vivaldi"]),
    (["Qual é a capital da Mongólia?", "Que cidade é a capital da Mongólia?", "Qual a sede do governo da Mongólia?"], "Ulan Bator", ["ulan", "ulaanbaatar", "ulã"]),
    (["Em que país fica a cidade de Timbuktu?", "Timbuktu pertence a qual país?", "A cidade de Timbuktu está em qual nação?"], "Mali", ["mali"]),
    (["Qual planeta tem o dia mais longo que seu ano?", "Em qual planeta um dia dura mais que um ano?", "Que planeta gira tão devagar que o dia supera o ano?"], "Vênus", ["vênus", "venus"]),
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

def plant_multi(forms, ans, K=4, steps=300, lr=0.1):
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

def build_and_eval(train_forms_idx, label):
    """train_forms_idx = índices das formulações usadas no treino; testa SEMPRE na formulação C (idx 2, held-out)."""
    lib = []  # (emb_treino_médio, seed, fato_idx, keys)
    for fi, (forms, ans, keys) in enumerate(FACTS):
        tf = [forms[i] for i in train_forms_idx]
        seed = plant_multi(tf, ans)
        emb = torch.stack([q_embed(q) for q in tf]).mean(0)
        lib.append((emb, seed, fi, keys))
    # avalia na formulação held-out C (idx 2)
    sims_report = []; route_ok = 0; ans_ok = 0; ans_ok_baseline = 0
    per_gate = {g: 0 for g in [0.88, 0.90, 0.93, 0.95, 0.97]}
    for fi, (forms, ans, keys) in enumerate(FACTS):
        cq = forms[2]  # held-out
        qe = q_embed(cq)
        sims = torch.stack([F.cosine_similarity(qe, e, 0) for e, _, _, _ in lib])
        j = int(sims.argmax()); best = float(sims[j]); routed_fi = lib[j][2]
        correct_route = (routed_fi == fi)
        route_ok += correct_route
        sims_report.append(round(best, 3))
        # resposta com o seed roteado (se roteou pro certo)
        seed = lib[j][1] if correct_route else None
        r = gen(cq, seed=seed); ans_ok += match(r, keys)
        ans_ok_baseline += match(gen(cq), keys)
        for g in per_gate:
            if correct_route and best >= g: per_gate[g] += 1
    res = {"variante": label, "de": len(FACTS), "roteou_certo": route_ok, "sim_heldout": sims_report,
           "resp_certa_com_seed": ans_ok, "resp_certa_baseline": ans_ok_baseline,
           "roteou_e_passou_gate": {str(g): per_gate[g] for g in per_gate}}
    log(f"  [{label}] rota-certa {route_ok}/{len(FACTS)} | resp {ans_ok}/{len(FACTS)} (baseline {ans_ok_baseline}) | sims {sims_report}")
    log(f"           gate: {res['roteou_e_passou_gate']}")
    return res

log("V1 = cartucho treinado em 1 formulação (A); testa na formulação C held-out")
v1 = build_and_eval([0], "V1_uma_forma")
log("V2 = cartucho treinado em 2 formulações (A+B); testa na formulação C held-out")
v2 = build_and_eval([0, 1], "V2_duas_formas")
out = {"V1": v1, "V2": v2, "nota": "held-out = formulação NUNCA vista no treino; separa conhecimento de decoreba"}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== GENERALIZAÇÃO held-out: V1 resp {v1['resp_certa_com_seed']}/{len(FACTS)} | V2 resp {v2['resp_certa_com_seed']}/{len(FACTS)} (baseline {v1['resp_certa_baseline']}) ===")
log(f"DONE M160 ({time.time()-t0:.0f}s)")
