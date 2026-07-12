#!/usr/bin/env python3
"""EVOLUÇÃO GUIADA PELA ARENA: GA evolui o perfil de poda por camada com fitness = NOTA REAL
na arena (fatos+aritmética+mmlu+código), não chat loss (que provou que mente). Máscaras
reversíveis por genoma (sem re-carve). Melhor perfil salvo p/ esculpida física final.
Dump → research/evolve_arena.json + journal."""
import json, time
import numpy as np, torch
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
IMP_DIET = [p + " " + k.title() for p, k in FACTS] + [p + k for p, k in ARITH] + \
    [f"Pergunta: {q}\nOpções: {o}\nResposta correta: {a.upper()}" for q, o, a in MMLU] + \
    [p + kws[0] for p, kws in CODE] + \
    ["Oi, tudo bem? Como você está?", "Me explique a fotossíntese.", "O céu é azul por causa da luz.",
     "A inteligência artificial aprende com dados.", "O Brasil é um país tropical."]

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=GGUF)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=GGUF, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
layers = model.model.layers; NL = len(layers); INTER = model.config.intermediate_size
print(f"4B carregado ({time.time()-t0:.0f}s)", flush=True)

keep = [torch.ones(INTER, device=DEV, dtype=torch.bool) for _ in range(NL)]
def mk(i):
    def h(m, inp):
        x = inp[0].clone(); x[..., ~keep[i]] = 0; return (x,)
    return h
for i in range(NL): layers[i].mlp.down_proj.register_forward_pre_hook(mk(i))

imp = [torch.zeros(INTER, device=DEV) for _ in range(NL)]
def mki(i):
    def h(m, inp): imp[i].add_(inp[0][0].abs().float().mean(0).detach())
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mki(i)) for i in range(NL)]
with torch.no_grad():
    for s in IMP_DIET:
        model(tok(s, return_tensors="pt", truncation=True, max_length=100).input_ids.to(DEV))
for h in hs: h.remove()
order = [torch.argsort(imp[i], descending=True) for i in range(NL)]
def set_masks(g):
    for i in range(NL):
        k = max(64, int(g[i] * INTER)); keep[i].zero_(); keep[i][order[i][:k]] = True

@torch.no_grad()
def gen(prompt, n=8):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
def arena():
    f = sum(kw in gen(p, 8) for p, kw in FACTS) / len(FACTS)
    a = sum(kw in gen(p, 5) for p, kw in ARITH) / len(ARITH)
    ok = 0
    for q, ops, ans in MMLU:
        r = gen(f"Pergunta: {q}\nOpções: {ops}\nResposta correta: ", 3)
        ok += (next((c for c in r if c in "abcd"), "?") == ans)
    m = ok / len(MMLU)
    c = sum(any(k in gen(p, 12) for k in kws) for p, kws in CODE) / len(CODE)
    return (f + a + m + c) / 4, {"fatos": f, "arit": a, "mmlu": m, "cod": c}

LAM = 0.25
def fitness(g):
    set_masks(g); comp, det = arena()
    return comp - LAM * float(np.mean(g)), comp, det

rng = np.random.default_rng(0); POP = 6
v2 = np.full(NL, 0.65); v2[:6] = 0.80; v2[-12:] = 0.80
pop = [np.ones(NL), v2.copy()] + [np.clip(v2 + rng.normal(0, 0.1, NL), 0.35, 1.0) for _ in range(POP - 2)]
best = None; hist = []
for g_i in range(8):
    sc = sorted(([*fitness(g), g] for g in pop), key=lambda x: -x[0])
    if best is None or sc[0][0] > best[0]: best = sc[0]
    hist.append({"gen": g_i, "fit": round(best[0], 3), "composto": round(best[1], 3),
                 "keep": round(float(np.mean(best[3])), 3)})
    print(f"gen {g_i}: fit {best[0]:.3f} | composto {best[1]:.3f} | keep {np.mean(best[3])*100:.0f}% | {best[2]} ({time.time()-t0:.0f}s)", flush=True)
    el = [s[3] for s in sc[:2]]; pop = [e.copy() for e in el]
    while len(pop) < POP:
        c = el[rng.integers(len(el))].copy(); mm = rng.random(NL) < 0.4
        c[mm] += rng.normal(0, 0.08, NL)[mm]; pop.append(np.clip(c, 0.35, 1.0))

_, bcomp, bdet, bg = best
out = {"best_composto": round(bcomp, 3), "best_det": bdet, "best_keep": round(float(np.mean(bg)), 3),
       "best_profile": [round(float(x), 3) for x in bg], "hist": hist,
       "ref": {"original_4b": 0.85, "carve_v2_manual": 0.825}}
json.dump(out, open("/home/leonardo/projects/LLM/bytebrain/research/evolve_arena.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Evolução guiada pela ARENA ({int(time.time()-t0)}s)\n- melhor: composto {bcomp:.3f} com keep {np.mean(bg)*100:.0f}% ({bdet})\n- hist: {hist}\n")
print(f"\nDONE evolve_arena: composto {bcomp:.3f} keep {np.mean(bg)*100:.0f}% ({time.time()-t0:.0f}s)", flush=True)
