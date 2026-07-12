#!/usr/bin/env python3
"""M146 — GAP no 7B (offload VRAM+RAM+disco, esquema do Leonardo). Roda Qwen2.5-7B (base) e -Instruct
num hardware de 12GB VRAM / 15GB RAM via device_map='auto' + max_memory (margem pra não travar).
Mede o gap em tarefas de raciocínio multi-passo (onde o finetune costuma ganhar de verdade, ao contrário
do 1.5B onde não havia gap). Se houver gap real, vale o transplante de pesos. Dump marco146_metrics.json."""
import json, time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco146_metrics.json"
OFF = "/tmp/offload7b"; os.makedirs(OFF, exist_ok=True)
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
# margem: 10GiB VRAM (de 12), 4GiB RAM (de 15 → ~11 livre p/ sistema), resto disco
MAXMEM = {0: "10GiB", "cpu": "4GiB"}

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

def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="auto",
                                                max_memory=MAXMEM, offload_folder=OFF)

@torch.no_grad()
def evalq(model, tok):
    ok = 0; outs = []
    for q, a in GAPQ:
        enc = tok(f"Resolva o problema e termine com 'Resposta: <número>'.\n{q}\n", return_tensors="pt").to(model.device)
        o = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
        t = tok.decode(o[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        outs.append(t[:80]); ok += (a in t)
    return ok, outs

res = {"n": len(GAPQ), "maxmem": MAXMEM}
for name, path in [("base_7b", f"{MODELS}/Qwen2.5-7B"), ("instruct_7b", f"{MODELS}/Qwen2.5-7B-Instruct")]:
    log(f"carregando {name} (offload VRAM+RAM+disco)")
    tok = AutoTokenizer.from_pretrained(path)
    m = load(path)
    acc, outs = evalq(m, tok)
    res[name] = {"acc": acc, "de": len(GAPQ)}
    log(f"  {name:12}: {acc}/{len(GAPQ)} | ex: {outs[0][:60]!r}")
    del m; import gc; gc.collect(); torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
gap = res.get("instruct_7b", {}).get("acc", 0) - res.get("base_7b", {}).get("acc", 0)
log(f"=== GAP 7B (instruct - base) = {gap} {'→ VALE o transplante' if gap >= 3 else '→ gap ainda pequeno'} ===")
log(f"DONE M146 ({time.time()-t0:.0f}s)")
