#!/usr/bin/env python3
"""M154 — AMPLITUDE DE CONHECIMENTO (onde o 7B deve ganhar): fatos raros/específicos. Se o 7B-int8 souber
MUITO mais que o 1.5B, o caminho de melhoria = injetar conhecimento no 1.5B via seeds. 1.5B vs 7B-int8.
Respostas curtas (rápido). GPU. Dump marco154_metrics.json."""
import json, time, gc
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco154_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

KN = [
    ("Quem escreveu o romance 'Guerra e Paz'?", ["tolst"]),
    ("Em que ano ocorreu a Batalha de Hastings?", ["1066"]),
    ("Qual é a capital da Mongólia?", ["ulan", "ulaanbaatar", "ulã"]),
    ("Quem descobriu a penicilina?", ["fleming"]),
    ("Qual planeta tem o dia mais longo que seu ano?", ["vênus", "venus"]),
    ("Quem pintou 'A Noite Estrelada'?", ["van gogh", "gogh"]),
    ("Em que país fica a cidade de Timbuktu?", ["mali"]),
    ("Quem foi o primeiro imperador romano?", ["augusto", "otávio", "octávio"]),
    ("Qual gás nobre é usado em letreiros luminosos vermelhos?", ["neônio", "neonio", "neon"]),
    ("Quem propôs a tabela periódica moderna?", ["mendele"]),
    ("Qual é o rio mais longo da Ásia?", ["yangtzé", "yangtze", "azul"]),
    ("Quantos ossos há em uma mão humana adulta?", ["27", "vinte e sete"]),
    ("Qual é o ponto de fusão aproximado do tungstênio em Celsius?", ["3422", "3400", "3410", "3.4"]),
    ("Qual metal, além do mercúrio, é líquido perto da temperatura ambiente?", ["gálio", "galio", "césio", "cesio", "frâncio"]),
    ("Quem escreveu 'Cem Anos de Solidão'?", ["márquez", "marquez", "garcía", "garcia"]),
    ("Qual é a unidade básica de informação quântica?", ["qubit", "q-bit"]),
    ("Em que ano caiu o Muro de Berlim?", ["1989"]),
    ("Qual é o maior deserto do mundo (incluindo polar)?", ["antárt", "antart"]),
    ("Quem compôs 'As Quatro Estações'?", ["vivaldi"]),
    ("Qual é a montanha mais alta do sistema solar?", ["olympus", "olimpo"]),
]

def match(txt, keys): t = txt.lower(); return any(k in t for k in keys)

def eval_hf(P, label):
    tok = AutoTokenizer.from_pretrained(P)
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
    ok = 0; det = []
    for q, keys in KN:
        enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]
        with torch.no_grad(): o = m.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0, nin:], skip_special_tokens=True); h = match(txt, keys); ok += h
        det.append({"q": q[:30], "hit": bool(h), "resp": txt[:40]})
    del m; gc.collect(); torch.cuda.empty_cache()
    log(f"  {label}: {ok}/{len(KN)}"); return {"acertos": ok, "de": len(KN), "detalhe": det}

def eval_int8(P, label):
    tok = AutoTokenizer.from_pretrained(P)
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16, quantization_config=QuantoConfig(weights="int8"), device_map="cuda", low_cpu_mem_usage=True)
    ok = 0; det = []
    for q, keys in KN:
        enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]
        with torch.no_grad(): o = m.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0, nin:], skip_special_tokens=True); h = match(txt, keys); ok += h
        det.append({"q": q[:30], "hit": bool(h), "resp": txt[:40]})
    del m; gc.collect(); torch.cuda.empty_cache()
    log(f"  {label}: {ok}/{len(KN)}"); return {"acertos": ok, "de": len(KN), "detalhe": det}

res = {}
res["1.5B"] = eval_hf(f"{MODELS}/Qwen2.5-1.5B-Instruct", "1.5B-Instruct")
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
res["7B_int8"] = eval_int8(f"{MODELS}/Qwen2.5-7B-Instruct", "7B-int8")
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
a = res["1.5B"]["acertos"]; b = res["7B_int8"]["acertos"]
# fatos que só o 7B soube (candidatos a virar seed no 1.5B)
so7b = [d["q"] for d15, d7 in zip(res["1.5B"]["detalhe"], res["7B_int8"]["detalhe"]) for d in [d7] if d7["hit"] and not d15["hit"]]
res["so_7b_sabe"] = so7b
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== CONHECIMENTO: 1.5B {a}/{len(KN)} vs 7B-int8 {b}/{len(KN)} | gap={b-a} | só-7B-sabe: {len(so7b)} ===")
log(f"DONE M154 ({time.time()-t0:.0f}s)")
