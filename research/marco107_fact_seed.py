#!/usr/bin/env python3
"""M107 — SEMENTE DE FATO (a Floresta no texto): transformar sabedoria do Qwen em semente + fertilizar.
Modelo congelado = decoder. Um FATO = uma SEMENTE = K vetores soft (soft-prompt) injetados nos embeds,
otimizados (só a semente, modelo congelado) até o modelo cuspir o fato. "Fertilizar" = aumentar K.
Testa 2 tipos: (a) fatos REAIS (o modelo já sabe → semente só ATIVA) e (b) fatos INVENTADOS que ele
NÃO pode saber (semente tem que ARMAZENAR o dado novo). Mede recall × bytes-da-semente × K (rega).
GPU. Dump research/marco107_metrics.json."""
import json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-Math-1.5B"
DEV = "cuda"; t0 = time.time()
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size
emb_layer = model.get_input_embeddings()
print(f"Qwen-Math-1.5B congelado | hidden {H} ({time.time()-t0:.0f}s)", flush=True)

REAIS = [("A capital da França é", " Paris"), ("A capital do Japão é", " Tóquio"),
         ("Sete vezes oito é igual a", " 56"), ("A fórmula da água é", " H2O")]
INVENTADOS = [("O código do cofre da IARA é", " 7492"), ("A senha secreta do projeto é", " banana"),
              ("O planeta natal do Leonardo é", " Krylon"), ("O número mágico do byte é", " 137")]

def emb_of(ids):
    return emb_layer(torch.tensor([ids], device=DEV)).detach()

@torch.no_grad()
def recall(seed, prompt, target, n=6):
    pe = emb_of(tok(prompt).input_ids)[0]
    cur = torch.cat([seed, pe], 0)[None]
    gen = []
    for _ in range(n):
        lg = model(inputs_embeds=cur).logits[0, -1]
        nx = int(lg.argmax()); gen.append(nx)
        cur = torch.cat([cur, emb_layer(torch.tensor([[nx]], device=DEV))], 1)
    out = tok.decode(gen)
    return target.strip().lower() in out.lower(), out.strip()[:30]

def plant(prompt, target, K, steps=150, verbose=False):
    pid = tok(prompt).input_ids
    tid = tok(target, add_special_tokens=False).input_ids
    pe = emb_of(pid)[0]; te = emb_of(tid)[0]
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1)  # fp32!
    opt = torch.optim.AdamW([seed], lr=0.3)
    tgt = torch.tensor(tid, device=DEV); st = K + len(pid)
    for s in range(steps):
        inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
        lg = model(inputs_embeds=inp).logits[0]
        loss = F.cross_entropy(lg[st-1:st-1+len(tid)].float(), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and s % 50 == 0: print(f"      plant K{K} step{s}: loss {loss.item():.3f}", flush=True)
    with torch.no_grad():
        inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
        pred = model(inputs_embeds=inp).logits[0][st-1:st-1+len(tid)].argmax(-1)
        tf = bool((pred == tgt).all())
    return seed.detach().to(torch.float16), float(loss), tf

def kb(K, bits):  # semente = K*H valores quantizados
    return K * H * bits / 8 / 1024

res = {"reais": {}, "inventados": {}}
for grupo, facts in [("reais", REAIS), ("inventados", INVENTADOS)]:
    print(f"\n=== {grupo.upper()} ===", flush=True)
    # baseline: modelo sozinho sabe?
    for prompt, target in facts:
        seed0 = torch.zeros(1, H, device=DEV, dtype=torch.float16)  # semente nula ~ nada
        ok0, out0 = recall(seed0[:0] if False else torch.zeros(0, H, device=DEV, dtype=torch.float16), prompt, target)
        curve = []
        first = (grupo == "reais" and prompt == REAIS[0][0])
        for K in [1, 2, 4, 8]:
            seed, loss, tf = plant(prompt, target, K, verbose=first)
            ok, out = recall(seed, prompt, target)
            curve.append({"K": K, "tf_recall": tf, "gen_recall": ok, "loss": round(loss, 3),
                          "KB_int4": round(kb(K, 4), 3), "saida": out})
        kmin = next((c["K"] for c in curve if c["tf_recall"]), None)
        res[grupo][prompt] = {"base_sabe": ok0, "base_saida": out0, "curva": curve, "K_min_recall": kmin}
        tag = "JÁ SABE" if ok0 else "não sabe"
        rec = f"K={kmin} ({kb(kmin,4):.2f}KB int4, loss {[c['loss'] for c in curve if c['K']==kmin][0]})" if kmin else "NÃO plantou"
        print(f"  [{prompt[:32]:32}] base:{tag} → armazenou em {rec}", flush=True)

json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/marco107_metrics.json", "w"), ensure_ascii=False, indent=1)
# resumo
inv_ok = sum(1 for v in res["inventados"].values() if v["K_min_recall"])
print(f"\n=== RESUMO: fatos INVENTADOS armazenados na semente: {inv_ok}/{len(INVENTADOS)} ===")
print(f"DONE M107 ({time.time()-t0:.0f}s)", flush=True)
