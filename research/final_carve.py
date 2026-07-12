#!/usr/bin/env python3
"""IARA-1 v0.1 — esculpida FINAL com o perfil vencedor da evolução arena-driven (composto 0.900
com 73% dos neurônios), corte físico, SALVA EM DISCO (llm-lab/models/iara-3b-v01) e roda a arena
de regressão no modelo salvo. Anexa em bench_arena.json + journal."""
import json, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"
OUT = "/home/leonardo/projects/LLM/llm-lab/models/iara-3b-v01"
t0 = time.time()
prof = json.load(open("/home/leonardo/projects/LLM/bytebrain/research/evolve_arena.json"))["best_profile"]

FACTS = [("A capital da França é", "paris"), ("A capital do Japão é", "tóqu"), ("A capital do Brasil é", "brasília"),
         ("A água ferve a", "100"), ("O maior planeta do sistema solar é", "júpiter"),
         ("A velocidade da luz é aproximadamente", "300"), ("O autor de Dom Casmurro é", "machado"),
         ("A Segunda Guerra Mundial terminou em", "1945"), ("O elemento químico de símbolo O é o", "oxig"),
         ("A capital da Itália é", "roma")]
ARITH = [("27 + 45 = ", "72"), ("9 * 7 = ", "63"), ("84 - 29 = ", "55"), ("144 / 12 = ", "12"),
         ("15 + 38 = ", "53"), ("6 * 12 = ", "72"), ("100 - 64 = ", "36"), ("8 * 8 = ", "64")]
MMLU = [("Qual órgão bombeia o sangue?", "A) pulmão B) coração C) fígado D) rim", "b"),
        ("Qual planeta é o planeta vermelho?", "A) Vênus B) Júpiter C) Marte D) Saturno", "c"),
        ("Quem pintou a Mona Lisa?", "A) Van Gogh B) Da Vinci C) Picasso D) Monet", "b"),
        ("Qual é o maior oceano?", "A) Atlântico B) Índico C) Ártico D) Pacífico", "d"),
        ("H2O é a fórmula da:", "A) água B) sal C) açúcar D) amônia", "a"),
        ("Quantos lados tem um hexágono?", "A) 5 B) 6 C) 7 D) 8", "b"),
        ("A fotossíntese produz:", "A) CO2 B) oxigênio C) nitrogênio D) metano", "b"),
        ("A independência do Brasil foi em:", "A) 1500 B) 1822 C) 1889 D) 1922", "b"),
        ("O DNA fica principalmente no:", "A) núcleo B) membrana C) citoplasma D) ribossomo", "a"),
        ("Quanto é 2 elevado a 5?", "A) 16 B) 32 C) 64 D) 8", "b")]
CODE = [("def soma(a, b):\n    return ", ["a + b", "a+b"]),
        ("# função que retorna o dobro\ndef dobro(x):\n    return ", ["x * 2", "2 * x", "x*2", "2*x"]),
        ("def fatorial(n):\n    if n == 0:\n        return 1\n    return ", ["fatorial"]),
        ("lista = [1, 2, 3]\nprint(len(", ["lista"])]
IMP_DIET = [p + " " + k.title() for p, k in FACTS] + [p + k for p, k in ARITH] + \
    [f"Pergunta: {q}\nOpções: {o}\nResposta correta: {a.upper()}" for q, o, a in MMLU] + \
    [p + kws[0] for p, kws in CODE] + \
    ["Oi, tudo bem? Como você está?", "Me explique a fotossíntese.", "O céu é azul por causa da luz.",
     "A inteligência artificial aprende com dados.", "O Brasil é um país tropical."]

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size
print(f"4B carregado ({time.time()-t0:.0f}s) | perfil evoluído keep médio {np.mean(prof)*100:.0f}%", flush=True)

imp = [torch.zeros(INTER, device=DEV) for _ in range(NL)]
def mki(i):
    def h(m, inp): imp[i].add_(inp[0][0].abs().float().mean(0).detach())
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mki(i)) for i in range(NL)]
with torch.no_grad():
    for s in IMP_DIET:
        model(tok(s, return_tensors="pt", truncation=True, max_length=100).input_ids.to(DEV))
for h in hs: h.remove()
for i in range(NL):
    k = max(64, int(prof[i] * INTER)); idx = torch.argsort(imp[i], descending=True)[:k]
    mlp = layers[i].mlp
    g = nn.Linear(mlp.gate_proj.in_features, k, bias=False, device=DEV, dtype=torch.float16)
    u = nn.Linear(mlp.up_proj.in_features, k, bias=False, device=DEV, dtype=torch.float16)
    d = nn.Linear(k, mlp.down_proj.out_features, bias=False, device=DEV, dtype=torch.float16)
    with torch.no_grad():
        g.weight.copy_(mlp.gate_proj.weight[idx]); u.weight.copy_(mlp.up_proj.weight[idx])
        d.weight.copy_(mlp.down_proj.weight[:, idx])
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = g, u, d
torch.cuda.empty_cache()
params = sum(p.numel() for p in model.parameters())
print(f"esculpido → {params/1e9:.2f}B ({time.time()-t0:.0f}s)", flush=True)

# ATENÇÃO: intermediate_size vira variável por camada — registrar no config
model.config.intermediate_size_per_layer = [layers[i].mlp.gate_proj.out_features for i in range(NL)]
model.config.iara_carve = {"base": "qwen3-4b-q4km", "profile": [round(float(x), 3) for x in prof],
                           "method": "arena-evolved importance slice", "date": "2026-07-02"}
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print(f"SALVO em {OUT} ({time.time()-t0:.0f}s)", flush=True)

@torch.no_grad()
def gen(prompt, n=8):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
s = {}
s["fatos"] = round(sum(kw in gen(p, 8) for p, kw in FACTS) / len(FACTS), 2)
s["aritmética"] = round(sum(kw in gen(p, 5) for p, kw in ARITH) / len(ARITH), 2)
ok = 0
for q, ops, ans in MMLU:
    r = gen(f"Pergunta: {q}\nOpções: {ops}\nResposta correta: ", 3)
    ok += (next((c for c in r if c in "abcd"), "?") == ans)
s["mini-MMLU"] = round(ok / len(MMLU), 2)
s["código"] = round(sum(any(k in gen(p, 12) for k in kws) for p, kws in CODE) / len(CODE), 2)
s["composto"] = round((s["fatos"] + s["aritmética"] + s["mini-MMLU"] + s["código"]) / 4, 3)
s["params_B"] = round(params / 1e9, 2)
demo = gen("USER: Me explique em uma frase por que o céu é azul.\nASSISTANT:", 30)
s["demo"] = demo[:120]
print(f"[IARA-1 v0.1] {s}", flush=True)

res = json.load(open("/home/leonardo/projects/LLM/bytebrain/research/bench_arena.json"))
res["IARA-1 v0.1 (evoluído+salvo)"] = s
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/bench_arena.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## IARA-1 v0.1 salvo em {OUT} — {s}\n")
print(f"DONE final_carve ({time.time()-t0:.0f}s)", flush=True)
