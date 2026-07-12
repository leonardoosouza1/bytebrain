#!/usr/bin/env python3
"""M109 — TETO da semente-de-fato: empurra 32 fatos numa semente compartilhada e varre K pra achar
quantos cabem e o piso de bytes/fato. Modelo congelado = decoder. GPU. Dump marco109_metrics.json."""
import json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"; DEV = "cuda"; t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()
print(f"congelado hidden {H} ({time.time()-t0:.0f}s)", flush=True)

FACTS = [("A capital da França é", " Paris"), ("A capital do Japão é", " Tóquio"),
    ("A capital da Itália é", " Roma"), ("A capital da Alemanha é", " Berlim"),
    ("A capital da Espanha é", " Madri"), ("A capital de Portugal é", " Lisboa"),
    ("A capital da Rússia é", " Moscou"), ("A capital do Egito é", " Cairo"),
    ("Sete vezes oito é", " 56"), ("Nove vezes nove é", " 81"), ("Doze mais treze é", " 25"),
    ("Cem menos quarenta é", " 60"), ("Dois elevado a dez é", " 1024"), ("Raiz de cento e quarenta e quatro é", " 12"),
    ("O código do cofre é", " 7492"), ("A senha secreta é", " banana"), ("O planeta natal é", " Krylon"),
    ("O número mágico é", " 137"), ("A cor do projeto é", " roxo"), ("O animal escolhido é", " gato"),
    ("O nome do robô é", " Ziggy"), ("A fruta proibida é", " manga"),
    ("A água é feita de hidrogênio e", " oxigênio"), ("O sol é uma", " estrela"),
    ("O coração bombeia", " sangue"), ("A célula é a unidade da", " vida"),
    ("O Brasil fica na América do", " Sul"), ("A Torre Eiffel fica em", " Paris"),
    ("O oposto de quente é", " frio"), ("O primeiro mês do ano é", " janeiro"),
    ("Um triângulo tem três", " lados"), ("A velocidade da luz é trezentos mil", " km")]
print(f"{len(FACTS)} fatos", flush=True)

def emb(ids): return EL(torch.tensor([ids], device=DEV)).detach()[0]
CACHE = [(emb(tok(p).input_ids), emb(tok(tg, add_special_tokens=False).input_ids),
          len(tok(p).input_ids), tok(tg, add_special_tokens=False).input_ids) for p, tg in FACTS]

def plant(K, steps=350):
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([seed], lr=0.3)
    for s in range(steps):
        loss = 0
        for pe, te, lp, tid in CACHE:
            inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
            lg = model(inputs_embeds=inp).logits[0]; st = K + lp
            loss = loss + F.cross_entropy(lg[st-1:st-1+len(tid)].float(), torch.tensor(tid, device=DEV))
        (loss / len(CACHE)).backward(); opt.step(); opt.zero_grad()
    return seed.detach().to(torch.float16)

@torch.no_grad()
def recall(seed):
    ok = 0
    for pe, te, lp, tid in CACHE:
        st = seed.shape[0] + lp
        inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
        pred = model(inputs_embeds=inp).logits[0][st-1:st-1+len(tid)].argmax(-1)
        ok += bool((pred == torch.tensor(tid, device=DEV)).all())
    return ok

res = {}
N = len(FACTS)
for K in [1, 2, 4, 8, 16, 32]:
    seed = plant(K)
    ok = recall(seed)
    kb = K * H * 4 / 8 / 1024  # int4
    res[K] = {"fatos_ok": ok, "de": N, "KB_int4": round(kb, 2), "bytes_por_fato": round(kb * 1024 / max(1, ok), 1)}
    print(f"  K={K:>2} ({kb:.1f}KB): {ok}/{N} fatos | {res[K]['bytes_por_fato']} bytes/fato ({time.time()-t0:.0f}s)", flush=True)

json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco109_metrics.json", "w"), ensure_ascii=False, indent=1)
best = min((v["bytes_por_fato"] for v in res.values() if v["fatos_ok"] >= 0.9 * N), default=None)
print(f"\nTETO: melhor byte/fato com >=90% recall = {best} | DONE M109 ({time.time()-t0:.0f}s)", flush=True)
