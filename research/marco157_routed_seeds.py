#!/usr/bin/env python3
"""M157 — CONSERTA o M156: o seed COMPARTILHADO prependido em tudo corrompe o que o modelo já sabia
(keep 0/9). O design certo (validado em M129/M137) = seeds POR-FATO atrás de um ROTEADOR: cada seed só
dispara para a SUA query. Aqui: treino 1 seed por fato-que-erra, roteador por similaridade de embedding
com gate; aplico o seed só quando bate. Meta: taught sobe SEM derrubar keep. Autossuficiente. GPU."""
import json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco157_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

KN = [
    ("Quem escreveu o romance 'Guerra e Paz'?", ["tolst"]),
    ("Em que ano ocorreu a Batalha de Hastings?", ["1066"]),
    ("Qual é a capital da Mongólia?", ["ulan", "ulaanbaatar", "ulã"]),
    ("Quem descobriu a penicilina?", ["fleming"]),
    ("Qual planeta tem o dia mais longo que seu ano?", ["vênus", "venus"]),
    ("Quem pintou 'A Noite Estrelada'?", ["van gogh", "gogh"]),
    ("Em que país fica a cidade de Timbuktu?", ["mali"]),
    ("Quem foi o primeiro imperador romano?", ["augusto", "otávio", "octávio"]),
    ("Qual gás nobre é usado em letreiros luminosos vermelhos?", ["neônio", "neonio", "neon"]),
    ("Quem propôs a tabela periódica moderna?", ["mendele"]),
    ("Qual é o rio mais longo da Ásia?", ["yangtzé", "yangtze", "azul"]),
    ("Quantos ossos há em uma mão humana adulta?", ["27", "vinte e sete"]),
    ("Qual é o ponto de fusão aproximado do tungstênio em Celsius?", ["3422", "3400", "3410", "3.4"]),
    ("Qual metal, além do mercúrio, é líquido perto da temperatura ambiente?", ["gálio", "galio", "césio", "cesio", "frâncio"]),
    ("Quem escreveu 'Cem Anos de Solidão'?", ["márquez", "marquez", "garcía", "garcia"]),
    ("Qual é a unidade básica de informação quântica?", ["qubit", "q-bit"]),
    ("Em que ano caiu o Muro de Berlim?", ["1989"]),
    ("Qual é o maior deserto do mundo (incluindo polar)?", ["antárt", "antart"]),
    ("Quem compôs 'As Quatro Estações'?", ["vivaldi"]),
    ("Qual é a montanha mais alta do sistema solar?", ["olympus", "olimpo"]),
]
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
def q_embed(q):
    ids = tok(q).input_ids
    return EL(torch.tensor([ids], device=DEV)).detach()[0].mean(0)  # centróide de embedding

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
def plant_one(q, a, K=4, steps=300, lr=0.1):
    pi = tok(FMT.format(q=q)).input_ids; ti = tok(" " + a, add_special_tokens=False).input_ids
    pe = EL(torch.tensor([pi], device=DEV)).detach()[0]; te = EL(torch.tensor([ti], device=DEV)).detach()[0]
    L = K + len(pi) + len(ti); E = torch.zeros(1, L, H, device=DEV, dtype=torch.float16)
    E[0, K:K+len(pi)] = pe; E[0, K+len(pi):L] = te
    pos = [K+len(pi)+kk-1 for kk in range(len(ti))]; tgt = torch.tensor(ti, device=DEV)
    Pp = torch.tensor(pos, device=DEV)
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1); opt = torch.optim.AdamW([seed], lr=lr)
    for s in range(steps):
        for g in opt.param_groups: g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * s / steps))
        X = E.clone(); X[:, :K] = seed.to(torch.float16)
        loss = F.cross_entropy(model(inputs_embeds=X).logits[0, Pp].float(), tgt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([seed], 1.0); opt.step()
    return seed.detach().to(torch.float16)

# baseline
base_hits = [match(gen(FMT.format(q=q)), keys) for q, keys in KN]
base = sum(base_hits); log(f"baseline 1.5B: {base}/20")
wrong = [(q, keys) for (q, keys), h in zip(KN, base_hits) if not h]
log(f"treinando {len(wrong)} seeds POR-FATO (K=4, 300 passos cada)...")

# 1 seed por fato-que-erra + embedding da sua pergunta
lib = []  # (q_emb, seed)
for i, (q, keys) in enumerate(wrong):
    a = ans_of(q)
    if not a: continue
    lib.append((q_embed(q), plant_one(q, a)))
log(f"biblioteca com {len(lib)} seeds; roteando as 20 perguntas (gate cos>0.97)")

# roteador: pra cada pergunta, escolhe o seed do fato mais próximo; aplica só se acima do gate
GATE = 0.97
routed = 0; fired = 0
for (q, keys) in KN:
    qe = q_embed(q); sims = torch.stack([F.cosine_similarity(qe, se, 0) for se, _ in lib])
    j = int(sims.argmax()); best = float(sims[j])
    seed = lib[j][1] if best >= GATE else None
    fired += seed is not None
    routed += match(gen(FMT.format(q=q), seed=seed), keys)
res = {"baseline": base, "com_roteador": routed, "de": 20, "seeds": len(lib),
       "seed_dispara": fired, "M155_shared": 15, "M156_keep_com_shared": 0,
       "nota": "seeds por-fato + roteador: injeta sem corromper keep; contraste com o shared seed do M155/M156"}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== ROTEADO: baseline {base}/20 → por-fato+roteador {routed}/20 (disparou em {fired}) | shared era 15 mas keep→0 ===")
log(f"DONE M157 ({time.time()-t0:.0f}s)")
