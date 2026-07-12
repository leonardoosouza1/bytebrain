#!/usr/bin/env python3
"""M143 — COMPOSIÇÃO de sementes-de-pesos (a visão do Leonardo: "vários modelos no meu de uma vez").
Task arithmetic: base + delta_chat(Instruct) + delta_code(Coder) = modelo bom em AMBOS? Mede chat
(28 fatos) e código (12 tarefas Python, match de idioma). Condições: base / +chat / +code / +ambos.
RAM-safe (per-tensor in-place, recarrega doador por vez, delta nunca inteiro na RAM). GPU-only compute.
Dump marco143_metrics.json."""
import json, time, gc
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco143_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE = f"{MODELS}/Qwen2.5-1.5B"; CHAT = f"{MODELS}/Qwen2.5-1.5B-Instruct"; CODE = f"{MODELS}/Qwen2.5-Coder-1.5B"
tok = AutoTokenizer.from_pretrained(BASE)

KEY = [("Qual é a capital da Austrália?", "Canberra"), ("Qual é o maior planeta do sistema solar?", "Júpiter"),
    ("Quem pintou a Mona Lisa?", "Vinci"), ("Em que ano o homem pisou na Lua?", "1969"),
    ("Qual é o maior oceano da Terra?", "Pacífico"), ("Qual é a moeda do Japão?", "iene"),
    ("Qual é o símbolo químico do ouro?", "Au"), ("Qual é a montanha mais alta do mundo?", "Everest"),
    ("Quem escreveu Dom Casmurro?", "Machado"), ("Qual é a capital do Canadá?", "Ottawa"),
    ("Qual é a capital da Coreia do Sul?", "Seul"), ("Quantos ossos tem o corpo humano adulto?", "206"),
    ("Qual é a capital da Argentina?", "Buenos Aires"), ("Qual é a fórmula química do sal de cozinha?", "NaCl"),
    ("Qual é a capital do Egito?", "Cairo"), ("Quem escreveu Romeu e Julieta?", "Shakespeare"),
    ("Qual é a capital da Alemanha?", "Berlim"), ("Qual é a capital da França?", "Paris"),
    ("Quem desenvolveu a teoria da relatividade?", "Einstein"), ("Qual é a capital da Itália?", "Roma")]
CODE_Q = [  # (prompt, idioma-chave que uma solução correta contém)
    ("# Python: função que inverte a string s\ndef inverte(s):\n    return s", "[::-1]"),
    ("# Python: função que retorna True se n é par\ndef par(n):\n    return n", "% 2"),
    ("# Python: soma dos elementos de uma lista L\ndef soma(L):\n    return ", "sum("),
    ("# Python: comprimento de uma lista L\nn = ", "len("),
    ("# Python: o maior valor de uma lista L\nm = ", "max("),
    ("# Python: ordenar a lista L\nL2 = ", "sorted("),
    ("# Python: quadrado de x\ndef quadrado(x):\n    return x", "** 2"),
    ("# Python: converter string s para maiúsculas\nr = s.", "upper("),
    ("# Python: número de itens em um dicionário d\nn = ", "len("),
    ("# Python: lista de 0 a 9\nL = list(", "range("),
    ("# Python: valor absoluto de x\nr = ", "abs("),
    ("# Python: juntar lista de strings L com vírgula\nr = ", "join(")]

@torch.no_grad()
def gen(prompt, n=14):
    enc = tok(prompt, return_tensors="pt").to(DEV)
    o = gm.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
def eval_chat(): return sum(a.lower() in gen(f"Pergunta: {q}\nResposta:", 12).lower() for q, a in KEY)
def eval_code(): return sum(key in gen(p, 16) for p, key in CODE_Q)

mtmp = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16)
base_sd = {k: v.clone() for k, v in mtmp.state_dict().items()}; del mtmp; gc.collect()
gm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to(DEV).eval()

@torch.no_grad()
def set_to(donors):
    """gm = base + soma dos int8(delta) de cada doador (per-tensor, RAM-safe)."""
    for k, p in gm.named_parameters():
        if k in base_sd: p.data = base_sd[k].to(DEV).clone()
    for D in donors:
        mt = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float16); dsd = mt.state_dict()
        for k, p in gm.named_parameters():
            if k in base_sd and k in dsd and dsd[k].shape == p.shape:
                d = dsd[k].to(DEV).float() - base_sd[k].to(DEV).float()
                s = (d.abs().max() / 127).clamp_min(1e-8); d = torch.round(d / s).clamp(-127, 127) * s
                p.data = (p.data.float() + d).to(p.dtype); del d
        del mt, dsd; gc.collect(); torch.cuda.empty_cache()

res = {"conds": {}}
for name, donors in [("base", []), ("+chat", [CHAT]), ("+code", [CODE]), ("+ambos", [CHAT, CODE])]:
    set_to(donors); c = eval_chat(); k = eval_code()
    res["conds"][name] = {"chat": c, "chat_de": len(KEY), "code": k, "code_de": len(CODE_Q)}
    log(f"  {name:8}: chat {c}/{len(KEY)} | code {k}/{len(CODE_Q)}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== composição: +ambos deve ficar bom em chat E code (task arithmetic) ===")
log(f"DONE M143 ({time.time()-t0:.0f}s)")
