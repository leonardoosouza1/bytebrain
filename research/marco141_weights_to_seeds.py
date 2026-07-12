#!/usr/bin/env python3
"""M141 — PESOS → SEMENTE, par LIMPO (Instruct). Qwen2.5-1.5B-Instruct menos Qwen2.5-1.5B (mesma config,
rope 1M) = delta de instrução/chat em pesos. Comprime (int8/int4/low-rank na GPU) → semente-de-pesos,
enxerta na base e mede quanto recupera o doador, a que tamanho. GPU-ONLY e leve: SVD na GPU por matriz,
delta em fp16, soma em fp32, modelo persistente. Eval = 28 fatos (geração livre). Dump marco141_metrics.json."""
import json, time, gc
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco141_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE = f"{MODELS}/Qwen2.5-1.5B"; DONOR = f"{MODELS}/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)  # idêntico ao do doador

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

# base sd na CPU (fp16)
mb = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
base_sd = {k: v.clone() for k, v in mb.state_dict().items()}
mb = mb.to(DEV).eval(); base_acc, _ = evalm(mb); log(f"  BASE sozinha: {base_acc}/{len(KEY)}")
del mb; gc.collect(); torch.cuda.empty_cache()
# doador (Instruct) na GPU = alvo do enxerto + referência
gm = AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float16).to(DEV).eval()
don_acc, don_outs = evalm(gm); log(f"  DOADOR (Instruct): {don_acc}/{len(KEY)}")
# delta por-param na GPU → guarda fp16 na CPU (sem float gigante na RAM)
delta = {}
for k, p in gm.named_parameters():
    if k in base_sd and base_sd[k].shape == p.shape:
        delta[k] = (p.data.float() - base_sd[k].to(DEV).float()).to(torch.float16).cpu()
full_bytes = sum(v.numel() for v in delta.values()) * 2
log(f"  delta: {len(delta)} tensores | {full_bytes/1e9:.2f} GB fp16 cheio")

def compress(method, rank=0):
    cd = {}; size = 0
    for k, v in delta.items():
        vg = v.to(DEV)
        if method == "lowrank" and vg.dim() == 2 and min(vg.shape) > rank and "embed" not in k and "lm_head" not in k:
            U, Sg, Vh = torch.linalg.svd(vg.float(), full_matrices=False)
            cd[k] = ((U[:, :rank] * Sg[:rank]) @ Vh[:rank]).to(torch.float16).cpu()
            size += rank * (vg.shape[0] + vg.shape[1]) * 2
            del U, Sg, Vh
        elif method in ("int8", "int4") or (method == "lowrank"):  # embed/lm_head no lowrank → int8
            qm = 127 if method != "int4" else 7; vf = vg.float()
            s = (vf.abs().max() / qm).clamp_min(1e-8)
            cd[k] = (torch.round(vf / s).clamp(-qm, qm) * s).to(torch.float16).cpu()
            size += int(vg.numel() * (1 if method != "int4" else 0.5))
        else:
            cd[k] = v; size += v.numel() * 2
        del vg; torch.cuda.empty_cache()
    return cd, size

@torch.no_grad()
def graft(cd):
    for k, p in gm.named_parameters():
        if k in base_sd:
            b = base_sd[k].to(DEV).float()
            if k in cd:
                b = b + torch.nan_to_num(cd[k].to(DEV).float(), nan=0.0, posinf=0.0, neginf=0.0)
            p.data = b.to(p.dtype)
        torch.cuda.empty_cache()

res = {"n": len(KEY), "base": base_acc, "doador": don_acc, "full_delta_GB": round(full_bytes/1e9, 2), "condicoes": {}}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
for method, rank in [("full", 0), ("int8", 0), ("int4", 0), ("lowrank", 32)]:
    cd, size = compress(method, rank)
    graft(cd); acc, outs = evalm(gm); del cd; gc.collect(); torch.cuda.empty_cache()
    agree = sum(o1.strip() == o2.strip() for o1, o2 in zip(outs, don_outs))
    nm = f"lowrank{rank}" if method == "lowrank" else method
    res["condicoes"][nm] = {"acc": acc, "de": len(KEY), "concorda_doador": agree,
                            "MB": round(size/1e6, 1), "compressao_x": round(full_bytes/max(1, size), 1)}
    log(f"  base+delta[{nm:9}]: acc {acc}/{len(KEY)} | concorda doador {agree}/{len(KEY)} | "
        f"{res['condicoes'][nm]['MB']} MB ({res['condicoes'][nm]['compressao_x']}× menor)")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== base {base_acc} → doador {don_acc} | full deve = doador (prova) ===")
log(f"DONE M141 ({time.time()-t0:.0f}s)")
