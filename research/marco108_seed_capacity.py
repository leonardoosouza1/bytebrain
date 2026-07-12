#!/usr/bin/env python3
"""M108 — bateria da semente-de-fato (rigor que faltou no M107): (A) int4/int8/ternário AGUENTA o
recall? (B) CAPACIDADE: quantos fatos cabem numa semente COMPARTILHADA de K tokens? Acha o piso.
Modelo congelado = decoder. GPU. Dump research/marco108_metrics.json."""
import json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"; DEV = "cuda"; t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()
print(f"congelado hidden {H} ({time.time()-t0:.0f}s)", flush=True)

FACTS = [("A capital da França é", " Paris"), ("O código do cofre da IARA é", " 7492"),
         ("O planeta natal do Leonardo é", " Krylon"), ("A senha secreta é", " banana"),
         ("Sete vezes oito é", " 56"), ("O número mágico do byte é", " 137"),
         ("A cor favorita do projeto é", " roxo"), ("O animal do Leonardo é", " gato")]

def emb(ids): return EL(torch.tensor([ids], device=DEV)).detach()[0]

def qseed(seed, kind):
    if kind == "fp16": return seed, 16
    t = seed.float()
    if kind == "int8": qm = 127; bits = 8
    elif kind == "int4": qm = 7; bits = 4
    else:  # ternário
        f = t.abs(); m = f > 0.7 * f.mean(); q = torch.zeros_like(t)
        if m.any(): q[m] = t[m].sign() * t[m].abs().mean()
        return q.to(torch.float16), 1.58
    s = (t.abs().max() / qm).clamp_min(1e-8)
    return ((t / s).round().clamp(-qm, qm) * s).to(torch.float16), bits

def plant(pairs, K, steps=200):
    """otimiza UMA semente (K,H) p/ recuperar TODOS os pares (prompt,target)."""
    data = [(emb(tok(p).input_ids), emb(tok(tg, add_special_tokens=False).input_ids),
             len(tok(p).input_ids), tok(tg, add_special_tokens=False).input_ids) for p, tg in pairs]
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([seed], lr=0.3)
    for s in range(steps):
        loss = 0
        for pe, te, lp, tid in data:
            inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
            lg = model(inputs_embeds=inp).logits[0]; st = K + lp
            loss = loss + F.cross_entropy(lg[st-1:st-1+len(tid)].float(), torch.tensor(tid, device=DEV))
        loss = loss / len(data)
        opt.zero_grad(); loss.backward(); opt.step()
    return seed.detach().to(torch.float16)

@torch.no_grad()
def tf_recall(seed, pairs):
    ok = 0
    for p, tg in pairs:
        pe = emb(tok(p).input_ids); tid = tok(tg, add_special_tokens=False).input_ids
        te = emb(tid); st = seed.shape[0] + len(tok(p).input_ids)
        inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
        pred = model(inputs_embeds=inp).logits[0][st-1:st-1+len(tid)].argmax(-1)
        ok += bool((pred == torch.tensor(tid, device=DEV)).all())
    return ok

res = {}
# (A) quantização: 1 fato, K=1, varia quantização
print("\n=== (A) QUANTIZAÇÃO segura o recall? (1 fato K=1) ===", flush=True)
qa = {}
for p, tg in FACTS[:4]:
    seed = plant([(p, tg)], 1)
    row = {}
    for kind in ["fp16", "int8", "int4", "ternário"]:
        qs, bits = qseed(seed, kind)
        row[kind] = {"recall": tf_recall(qs, [(p, tg)]), "KB": round(1 * H * bits / 8 / 1024, 3)}
    qa[p] = row
    print(f"  [{p[:28]:28}] " + " ".join(f"{k}:{'ok' if v['recall'] else 'X'}({v['KB']}KB)" for k, v in row.items()), flush=True)
res["quantizacao"] = qa

# (B) capacidade: N fatos numa semente K compartilhada
print("\n=== (B) CAPACIDADE: quantos fatos numa semente K? ===", flush=True)
cap = {}
for K in [1, 2, 4, 8, 16]:
    seed = plant(FACTS, K, steps=250)
    ok = tf_recall(seed, FACTS)
    cap[K] = {"fatos_ok": ok, "de": len(FACTS), "KB_int4": round(K * H * 4 / 8 / 1024, 2),
              "KB_por_fato": round(K * H * 4 / 8 / 1024 / max(1, ok), 3)}
    print(f"  K={K:>2} ({cap[K]['KB_int4']}KB int4): {ok}/{len(FACTS)} fatos | {cap[K]['KB_por_fato']}KB/fato", flush=True)
res["capacidade"] = cap

json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco108_metrics.json", "w"), ensure_ascii=False, indent=1)
print(f"\nDONE M108 ({time.time()-t0:.0f}s)", flush=True)
