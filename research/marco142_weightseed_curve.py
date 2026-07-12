#!/usr/bin/env python3
"""M142 — CURVA DE COMPRESSÃO da semente-de-pesos (À PROVA DE RAM, 15GB). Mecanismo já provado no M141
(base+delta_cheio = doador, concorda 28/28). Aqui: quão PEQUENA a semente-de-pesos pode ficar mantendo
o comportamento do doador? Par limpo Instruct. RAM-safe: mantém só base_sd (CPU) + 1 modelo (GPU);
recarrega o doador por método e comprime/enxerta IN-PLACE por tensor (nunca segura o delta inteiro).
GPU-only pra compute. Dump marco142_metrics.json."""
import json, time, gc
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco142_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE = f"{MODELS}/Qwen2.5-1.5B"; DONOR = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
KEY = [("Qual é a capital da Austrália?", "Canberra"), ("Qual é o maior planeta do sistema solar?", "Júpiter"),
    ("Quem pintou a Mona Lisa?", "Vinci"), ("Em que ano o homem pisou na Lua?", "1969"),
    ("Qual é o maior oceano da Terra?", "Pacífico"), ("Qual é a moeda do Japão?", "iene"),
    ("Qual é o símbolo químico do ouro?", "Au"), ("Qual é a montanha mais alta do mundo?", "Everest"),
    ("Quem escreveu Dom Casmurro?", "Machado"), ("Qual planeta é o planeta vermelho?", "Marte"),
    ("Qual é a capital do Canadá?", "Ottawa"), ("Quem descobriu a gravidade?", "Newton"),
    ("Qual é a capital da Coreia do Sul?", "Seul"), ("Quantos ossos tem o corpo humano adulto?", "206"),
    ("Quem foi o primeiro presidente dos EUA?", "Washington"), ("Qual o ponto de ebulição da água em Celsius?", "100"),
    ("Qual é a capital da Argentina?", "Buenos Aires"), ("Quantos planetas há no sistema solar?", "8"),
    ("Qual é a fórmula química do sal de cozinha?", "NaCl"), ("Qual é a capital do Egito?", "Cairo"),
    ("Quem escreveu Romeu e Julieta?", "Shakespeare"), ("Qual é a capital da Alemanha?", "Berlim"),
    ("Qual planeta é o mais próximo do Sol?", "Mercúrio"), ("Qual é a capital da França?", "Paris"),
    ("Quem desenvolveu a teoria da relatividade?", "Einstein"), ("Qual é a capital da Itália?", "Roma"),
    ("Qual é a capital de Portugal?", "Lisboa"), ("Qual é o maior país do mundo em área?", "Rússia")]

@torch.no_grad()
def evalm(model):
    ok = 0; outs = []
    for q, a in KEY:
        enc = tok(f"Pergunta: {q}\nResposta:", return_tensors="pt").to(DEV)
        o = model.generate(**enc, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        t = tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        outs.append(t); ok += (a.lower() in t.lower())
    return ok, outs

# base_sd na CPU (fp16) — única coisa grande que fica na RAM
mtmp = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
base_sd = {k: v.clone() for k, v in mtmp.state_dict().items()}
mtmp = mtmp.to(DEV).eval(); base_acc, _ = evalm(mtmp); log(f"BASE: {base_acc}/{len(KEY)}")
del mtmp; gc.collect(); torch.cuda.empty_cache()
gm = AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float16).to(DEV).eval()
don_acc, don_outs = evalm(gm); log(f"DOADOR (Instruct): {don_acc}/{len(KEY)}")

@torch.no_grad()
def graft_compress(method, rank):
    """recarrega o doador em gm, e por tensor: delta=doador-base, comprime, gm=base+delta_comprimido."""
    mtemp = AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float16)
    gm.load_state_dict(mtemp.state_dict()); del mtemp; gc.collect()
    size = 0
    for k, p in gm.named_parameters():
        if k not in base_sd: continue
        b = base_sd[k].to(DEV).float(); d = p.data.float() - b
        if method == "lowrank" and d.dim() == 2 and min(d.shape) > rank and "embed" not in k and "lm_head" not in k:
            U, Sg, Vh = torch.linalg.svd(d, full_matrices=False)
            d = (U[:, :rank] * Sg[:rank]) @ Vh[:rank]; size += rank * (d.shape[0] + d.shape[1]) * 2
            del U, Sg, Vh
        else:
            qm = 127 if method != "int4" else 7
            s = (d.abs().max() / qm).clamp_min(1e-8); d = torch.round(d / s).clamp(-qm, qm) * s
            size += int(p.numel() * (1 if method != "int4" else 0.5))
        p.data = (b + d).to(p.dtype); del b, d; torch.cuda.empty_cache()
    return size

full_bytes = sum(v.numel() for v in base_sd.values() if v.dim() >= 1) * 2
res = {"n": len(KEY), "base": base_acc, "doador": don_acc, "full_GB": round(full_bytes/1e9, 2),
       "condicoes": {"full": {"acc": don_acc, "concorda": len(KEY), "MB": round(full_bytes/1e6, 1), "x": 1.0}}}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
for method, rank in [("int8", 0), ("int4", 0), ("lowrank", 32)]:
    size = graft_compress(method, rank); acc, outs = evalm(gm)
    agree = sum(o1.strip() == o2.strip() for o1, o2 in zip(outs, don_outs))
    nm = f"lowrank{rank}" if method == "lowrank" else method
    res["condicoes"][nm] = {"acc": acc, "concorda": agree, "MB": round(size/1e6, 1), "x": round(full_bytes/max(1, size), 1)}
    log(f"  {nm:9}: acc {acc}/{len(KEY)} | concorda doador {agree}/{len(KEY)} | {res['condicoes'][nm]['MB']} MB ({res['condicoes'][nm]['x']}× menor)")
    gc.collect(); torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== base {base_acc} → doador {don_acc}; semente-de-pesos comprimida recupera o doador ===")
log(f"DONE M142 ({time.time()-t0:.0f}s)")
