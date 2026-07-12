#!/usr/bin/env python3
"""M140 — PESOS → SEMENTE (task vector): Math-1.5B menos base Qwen2.5-1.5B = cérebro matemático em pesos.
Comprime o delta (int8/int4/low-rank) → semente-de-pesos, enxerta na base e mede quanto recupera o
doador, a que tamanho. CONSERTOS: soma em fp32 (fp16 arredondava o delta a zero!), modelo PERSISTENTE
(sem recarregar → evita falha de GPU), matemática multi-passo (gap real). GPU. Dump marco140_metrics.json."""
import json, time, gc
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco140_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE = f"{MODELS}/Qwen2.5-1.5B"; DONOR = f"{MODELS}/Qwen2.5-Math-1.5B"

MSET = [  # matemática multi-passo / mais dura (onde a base 1.5B costuma falhar)
    ("João tem 3 caixas com 24 lápis cada e dá 17 lápis. Quantos ficam?", "55"),
    ("Uma camisa custa 45 reais com 20% de desconto. Qual o preço final?", "36"),
    ("Um carro faz 12 km por litro e o tanque tem 45 litros. Quantos km faz?", "540"),
    ("Qual o resto da divisão de 1000 por 7?", "6"),
    ("A soma de três números consecutivos é 72. Qual o menor deles?", "23"),
    ("Uma pizza tem 8 fatias. Três pessoas comem 2 fatias cada. Quantas sobram?", "2"),
    ("Qual o próximo número da sequência 2, 6, 12, 20, 30?", "42"),
    ("Quanto é 15% de 15% de 400?", "9"),
    ("Se 5 operários fazem uma obra em 12 dias, quantos dias 10 operários levam?", "6"),
    ("Um número somado ao seu dobro dá 45. Qual é o número?", "15"),
    ("Qual a média de 10, 20 e 30?", "20"),
    ("Quanto é 347 vezes 12?", "4164"),
    ("Se x dividido por 4 é 9, quanto vale x?", "36"),
    ("Ana tinha 50 reais, gastou 3/5. Quanto sobrou?", "20"),
    ("Um retângulo tem 8 de largura e 5 de altura. Qual a área?", "40"),
    ("Quanto é 2 elevado a 6?", "64"),
    ("Se um produto de 80 reais aumenta 25%, qual o novo preço?", "100"),
    ("Quanto é 1234 mais 5678?", "6912"),
    ("Dois trens partem juntos, um a 40 e outro a 60 km/h. Após 2h, qual a distância entre eles?", "40"),
    ("Qual o dobro de 17 ao quadrado?", "578"),
]
tok = AutoTokenizer.from_pretrained(BASE)

@torch.no_grad()
def evalm(model, tk):
    ok = 0; outs = []
    for q, a in MSET:
        enc = tk(f"Resolva passo a passo e dê o número final. {q}\nResposta:", return_tensors="pt").to(DEV)
        o = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tk.eos_token_id)
        t = tk.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        outs.append(t); ok += (a in t)
    return ok, outs

# ---------- delta (task vector) ----------
log("carregando base + doador, computando delta")
mb0 = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
sd_base = {k: v.clone() for k, v in mb0.state_dict().items()}
del mb0; gc.collect()
md0 = AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float16)
sd_don = md0.state_dict()
delta = {k: (sd_don[k].float() - sd_base[k].float())
         for k in sd_base if k in sd_don and sd_base[k].shape == sd_don[k].shape and sd_base[k].dim() >= 1}
del md0, sd_don, sd_base; gc.collect()
full_bytes = sum(v.numel() for v in delta.values()) * 2
log(f"delta: {len(delta)} tensores | {full_bytes/1e9:.2f} GB fp16 cheio")

def compress(method, rank=0):
    cd = {}; size = 0
    for k, v in delta.items():
        if method == "lowrank" and v.dim() == 2 and min(v.shape) > rank:
            U, Sg, Vh = torch.linalg.svd(v, full_matrices=False)
            cd[k] = (U[:, :rank] * Sg[:rank]) @ Vh[:rank]
            size += rank * (v.shape[0] + v.shape[1]) * 2
        elif method in ("int8", "int4"):
            qm = 127 if method == "int8" else 7
            s = (v.abs().max() / qm).clamp_min(1e-8)
            cd[k] = torch.round(v / s).clamp(-qm, qm) * s
            size += int(v.numel() * (1 if method == "int8" else 0.5))
        else:
            cd[k] = v; size += v.numel() * 2
    return cd, size  # cd em float32 na CPU

# ---------- modelo PERSISTENTE: reseta p/ base + cd (soma em fp32) por condição ----------
gmodel = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to(DEV).eval()
base_gpu = {k: p.data.clone() for k, p in gmodel.named_parameters()}  # base fp16 na GPU
@torch.no_grad()
def graft(cd):
    for k, p in gmodel.named_parameters():
        b = base_gpu[k].float()
        if k in cd:
            d = torch.nan_to_num(cd[k].to(DEV).float(), nan=0.0, posinf=0.0, neginf=0.0)
            p.data = (b + d).to(p.dtype)
        else:
            p.data = base_gpu[k].clone()

res = {"n": len(MSET), "full_delta_GB": round(full_bytes/1e9, 2), "condicoes": {}}
graft({})  # = base pura
res["base"], _ = evalm(gmodel, tok); log(f"  BASE sozinha: {res['base']}/{len(MSET)}")
tokd = AutoTokenizer.from_pretrained(DONOR)
mdn = AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float16).to(DEV).eval()
res["doador"], don_outs = evalm(mdn, tokd); log(f"  DOADOR (Math): {res['doador']}/{len(MSET)}")
del mdn; gc.collect(); torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

for method, rank in [("full", 0), ("int8", 0), ("int4", 0), ("lowrank", 128), ("lowrank", 32), ("lowrank", 8)]:
    cd, size = compress(method, rank)
    graft(cd); acc, outs = evalm(gmodel, tok); del cd; gc.collect(); torch.cuda.empty_cache()
    agree = sum(o1 == o2 for o1, o2 in zip(outs, don_outs))
    nm = f"lowrank{rank}" if method == "lowrank" else method
    res["condicoes"][nm] = {"acc": acc, "de": len(MSET), "concorda_doador": agree,
                            "MB": round(size/1e6, 1), "compressao_x": round(full_bytes/max(1, size), 1)}
    log(f"  base+delta[{nm:10}]: math {acc}/{len(MSET)} | concorda doador {agree}/{len(MSET)} | "
        f"{res['condicoes'][nm]['MB']} MB ({res['condicoes'][nm]['compressao_x']}× menor)")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

log(f"=== base {res['base']} → doador {res['doador']} | full deve = doador (prova do enxerto) ===")
log(f"DONE M140 ({time.time()-t0:.0f}s)")
