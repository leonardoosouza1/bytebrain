#!/usr/bin/env python3
"""ARENA parte B: Qwen2.5-1.5B, SmolLM2-1.7B e ByteBrain 40M nas MESMAS provas da parte A.
Anexa em research/bench_arena.json."""
import json, random, sys, time
import numpy as np, torch, torch.nn.functional as Fn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"; t0 = time.time()
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

def evaluate_hf(path, name):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
    @torch.no_grad()
    def gen(prompt, n=8):
        ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).lower()
    s = {}
    s["fatos"] = sum(kw in gen(p, 8) for p, kw in FACTS) / len(FACTS)
    s["aritmética"] = sum(kw in gen(p, 5) for p, kw in ARITH) / len(ARITH)
    ok = 0
    for q, ops, ans in MMLU:
        r = gen(f"Pergunta: {q}\nOpções: {ops}\nResposta correta: ", 3)
        first = next((c for c in r if c in "abcd"), "?")
        ok += (first == ans)
    s["mini-MMLU"] = ok / len(MMLU)
    s["código"] = sum(any(k in gen(p, 12) for k in kws) for p, kws in CODE) / len(CODE)
    tl = 0.0
    with torch.no_grad():
        for w in WIKI:
            ids = tok(w, return_tensors="pt", truncation=True, max_length=120).input_ids.to(DEV)
            tl += Fn.cross_entropy(model(ids).logits[0, :-1].float(), ids[0, 1:]).item()
    s["wiki_ppl"] = round(float(np.exp(tl / len(WIKI))), 2)
    s["composto"] = round((s["fatos"] + s["aritmética"] + s["mini-MMLU"] + s["código"]) / 4, 3)
    for k in ("fatos", "aritmética", "mini-MMLU", "código"): s[k] = round(s[k], 2)
    print(f"[{name}] {s}", flush=True)
    del model; torch.cuda.empty_cache()
    return s

res = json.load(open("/home/leonardo/projects/LLM/bytebrain/research/bench_arena.json"))
res["qwen2.5-1.5b"] = evaluate_hf("/home/leonardo/projects/LLM/llm-lab/models/qwen2.5-1.5b", "qwen2.5-1.5b")
res["smollm2-1.7b"] = evaluate_hf("/home/leonardo/projects/LLM/llm-lab/models/SmolLM2-1.7B", "smollm2-1.7b")

# ByteBrain 40M (bytes, greedy)
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from train_graph import GraphByteGPT
try: from train_graph import set_act_quant
except ImportError: set_act_quant = lambda *_: None
c = torch.load("/home/leonardo/projects/LLM/bytebrain/ckpt_broad/ckpt_best.pt", map_location="cpu", weights_only=False)
cf = c["config"]; set_act_quant(cf.get("quant_bits", 0))
bm = GraphByteGPT(cf["dim"], cf["layers"], cf["heads"], cf["ctx"], topk=cf.get("topk", 0),
                  mem=cf.get("mem", 0), topic=cf.get("topic", 0)).eval()
bm.load_state_dict(c["model"])
@torch.no_grad()
def bgen(prompt, n=30):
    ids = list(prompt.encode())
    for _ in range(n):
        x = torch.tensor([ids[-cf["ctx"]:]], dtype=torch.long)
        out = bm(x)
        logits = out[0] if isinstance(out, tuple) else out
        ids.append(int(logits[0, -1].argmax()))
    return bytes(ids[len(prompt.encode()):]).decode("utf-8", errors="ignore").lower()
s = {}
s["fatos"] = round(sum(kw in bgen(p, 24) for p, kw in FACTS) / len(FACTS), 2)
s["aritmética"] = round(sum(kw in bgen(p, 8) for p, kw in ARITH) / len(ARITH), 2)
s["mini-MMLU"] = 0.0
s["código"] = round(sum(any(k in bgen(p, 40) for k in kws) for p, kws in CODE) / len(CODE), 2)
s["wiki_ppl"] = None
s["composto"] = round((s["fatos"] + s["aritmética"] + s["mini-MMLU"] + s["código"]) / 4, 3)
print(f"[bytebrain-40m] {s}", flush=True)
res["bytebrain-40m"] = s

json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/research/bench_arena.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## ARENA — ranking local ({int(time.time()-t0)}s)\n")
    for n, v in sorted(res.items(), key=lambda kv: -kv[1]["composto"]):
        f.write(f"- {n}: composto {v['composto']} | fatos {v['fatos']} arit {v['aritmética']} mmlu {v['mini-MMLU']} cód {v['código']} ppl {v['wiki_ppl']}\n")
print(f"DONE bench_b ({time.time()-t0:.0f}s)", flush=True)
