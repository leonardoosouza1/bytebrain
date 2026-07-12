#!/usr/bin/env python3
"""M147 — roda UM modelo 7B dividido VRAM+RAM (parte VRAM, parte RAM, sem disco) em processo fresco.
Uso: python marco147_eval_one.py <model_dir> <rótulo>. Mede os 12 problemas multi-passo (GAPQ). Append
em marco147_metrics.json. Empurra o máximo pra VRAM (11GiB de 12) e o resto RAM — sem offload de disco."""
import sys, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
PATH, LABEL = sys.argv[1], sys.argv[2]
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco147_metrics.json"
MAXMEM = {0: "11GiB", "cpu": "6GiB"}  # 7B fp16 (15GB) → ~11 VRAM + ~4 RAM, sem disco

GAPQ = [
    ("Ana tem 3 caixas com 24 lápis cada. Ela dá 17 lápis. Quantos lápis sobram?", "55"),
    ("Um trem percorre 60 km/h. Em 2,5 horas, quantos km percorre?", "150"),
    ("João comprou 5 cadernos a 12 reais e pagou com 100. Quanto de troco?", "40"),
    ("Uma torneira enche 8 litros por minuto. Quantos litros em 15 minutos?", "120"),
    ("Se 3 pizzas têm 8 fatias cada e 5 pessoas comem 4 fatias cada, quantas sobram?", "4"),
    ("Um produto custa 80 reais. Com 25% de desconto e depois 10 de frete, quanto sai?", "70"),
    ("Maria leu 45 páginas por dia durante 6 dias de um livro de 300. Quantas faltam?", "30"),
    ("Numa sala há 30 alunos, 40% são meninas. Quantos são meninos?", "18"),
    ("Um carro faz 14 km por litro. Para 210 km, quantos litros gasta?", "15"),
    ("Se x + 2x + 3x = 60, quanto vale x?", "10"),
    ("Uma loja vende 3 camisas por 90 reais. Quanto custam 7 camisas?", "210"),
    ("Pedro tinha 250 reais, gastou 3/5 e ganhou 40. Com quanto ficou?", "140"),
]
tok = AutoTokenizer.from_pretrained(PATH)
log(f"carregando {LABEL} (VRAM {MAXMEM[0]} + RAM {MAXMEM['cpu']}, sem disco)")
m = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.float16, device_map="auto", max_memory=MAXMEM)
ok = 0; ex = ""
for q, a in GAPQ:
    enc = tok(f"Resolva o problema e termine com 'Resposta: <número>'.\n{q}\n", return_tensors="pt").to(m.device)
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    t = tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    ok += (a in t)
    if not ex: ex = t[:60]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res[LABEL] = {"acc": ok, "de": len(GAPQ), "ex": ex}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== {LABEL}: {ok}/{len(GAPQ)} | ex {ex!r} ===")
log(f"DONE M147 {LABEL} ({time.time()-t0:.0f}s)")
