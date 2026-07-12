#!/usr/bin/env python3
"""M156 — HONESTIDADE do seed: memorização ou generalização? Treino o seed em METADE dos fatos que o 1.5B
erra (train) e avalio em 3 grupos: (a) taught = os que ensinei (deve subir), (b) heldout = fatos que ele
erra mas NÃO ensinei (sobe? = generaliza; fica = só memoriza), (c) keep = os que já acertava (não pode
cair = sem interferência). Autossuficiente (sem imports com efeito colateral). GPU. Dump json."""
import json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco156_metrics.json"
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
wrong, right = [], []
for q, keys in KN:
    (wrong if not match(gen(FMT.format(q=q)), keys) else right).append((q, keys))
log(f"1.5B erra {len(wrong)}, acerta {len(right)}")
train = [wrong[i] for i in range(0, len(wrong), 2)]
heldout = [wrong[i] for i in range(1, len(wrong), 2)]
log(f"train={len(train)}  heldout={len(heldout)}  keep={len(right)}")
plant_facts = [(FMT.format(q=q), " " + ans_of(q)) for q, keys in train if ans_of(q)]

def plant(pairs, K=16, steps=400, lr=0.1):
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
def score(group): return sum(match(gen(FMT.format(q=q), seed=seed), keys) for q, keys in group)
taught = score(train); held = score(heldout); keep = score(right)
res = {"taught_de": len(train), "taught_ok_com_seed": taught,
       "heldout_de": len(heldout), "heldout_ok_com_seed": held,
       "keep_de": len(right), "keep_ok_com_seed": keep,
       "interpretacao": "taught sobe = injeta; heldout baixo = memoriza (não generaliza); keep mantém = sem interferência"}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== taught {taught}/{len(train)} | heldout {held}/{len(heldout)} | keep {keep}/{len(right)} ===")
log(f"DONE M156 ({time.time()-t0:.0f}s)")
