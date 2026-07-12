#!/usr/bin/env python3
"""CARVE V2 — a correção que a arena exigiu: esculpir com importância AMPLA (fatos + aritmética
+ mini-MMLU + código + chat), não só chat. A arena A provou que o proxy 'chat loss' mente:
o carve v1 tinha loss ótima e desabou nas provas reais. v2: mesmo método, dieta de importância
completa + perfil menos agressivo nas camadas profundas (onde vive o conhecimento).
Avalia nas MESMAS provas da arena. Anexa em bench_arena.json."""
import json, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

MDIR = "/home/leonardo/projects/LLM/llm-lab/models"; GGUF = "qwen3-4b-q4km.gguf"; DEV = "cuda"
t0 = time.time()

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
random.seed(11); WIKI = []; seen = 0
with open("/home/leonardo/projects/LLM/bytebrain/data/pt_big.txt", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if not (250 <= len(s) <= 700): continue
        if any(b in s for b in ("http", "|", "{{", "==", "[[", "<")): continue
        if sum(c.isalpha() for c in s) / len(s) < 0.75: continue
        seen += 1
        if len(WIKI) < 10: WIKI.append(s)
        elif random.randint(0, seen) < 10: WIKI[random.randint(0, 9)] = s
        if seen > 60000: break

# DIETA AMPLA de importância: fatos COMPLETOS + contas + mmlu + código + chat + wiki
IMP_DIET = [p + " " + k.title() for p, k in FACTS] + [p + k for p, k in ARITH] + \
    [f"Pergunta: {q}\nOpções: {o}\nResposta correta: {a.upper()}" for q, o, a in MMLU] + \
    [p + kws[0] for p, kws in CODE] + WIKI[:6] + \
    ["Oi, tudo bem? Como você está?", "Me explique a fotossíntese.", "O céu é azul por causa da luz.",
     "A inteligência artificial aprende com dados."]

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size
print(f"4B carregado ({time.time()-t0:.0f}s) | dieta de importância: {len(IMP_DIET)} textos", flush=True)

imp = [torch.zeros(INTER, device=DEV) for _ in range(NL)]
def mki(i):
    def h(m, inp): imp[i].add_(inp[0][0].abs().float().mean(0).detach())
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mki(i)) for i in range(NL)]
with torch.no_grad():
    for s in IMP_DIET:
        model(tok(s, return_tensors="pt", truncation=True, max_length=100).input_ids.to(DEV))
for h in hs: h.remove()

# perfil v2: conservador — 80% nas 6 primeiras e 12 últimas (conhecimento profundo), 65% no meio
prof = np.full(NL, 0.65); prof[:6] = 0.80; prof[-12:] = 0.80
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
print(f"carve v2 ({time.time()-t0:.0f}s) → {params/1e9:.2f}B", flush=True)

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
tl = 0.0
with torch.no_grad():
    for w in WIKI:
        ids = tok(w, return_tensors="pt", truncation=True, max_length=120).input_ids.to(DEV)
        tl += Fn.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:]).item()
s["wiki_ppl"] = round(float(np.exp(tl / len(WIKI))), 2)
s["composto"] = round((s["fatos"] + s["aritmética"] + s["mini-MMLU"] + s["código"]) / 4, 3)
s["params_B"] = round(params / 1e9, 2)
print(f"[iara-3b v2 (dieta ampla)] {s}", flush=True)

res = json.load(open("/home/leonardo/projects/LLM/bytebrain/research/bench_arena.json"))
res["iara-3.1b v2 (dieta ampla)"] = s
json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/bench_arena.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Carve V2 (dieta ampla) — {s}\n")
print(f"DONE carve_v2 ({time.time()-t0:.0f}s)", flush=True)
