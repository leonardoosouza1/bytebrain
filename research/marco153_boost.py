#!/usr/bin/env python3
"""M153 — TURBINAR O LEVE + TETO REAL DO 7B. (a) 1.5B-Instruct com SELF-CONSISTENCY (single, maioria@5,
pass@5) — votar entre amostras turbina o raciocínio do modelo rápido? (b) 7B-Instruct INT8 (menos lossy
que int4) — a sabedoria REAL do 7B. Avaliação difícil. Se 1.5B+voto ~ int8-7B → leve+rápido com sabedoria
de 7B = objetivo. GPU. Dump marco153_metrics.json."""
import json, time, gc
from collections import Counter
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco153_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
from marco152_hard_eval import HARD, match  # reusa o set difícil

def gen(m, tok, q, n=130, temp=0.0):
    enc = tok.apply_chat_template([{"role": "user", "content": q + " Pense passo a passo e dê a resposta final."}],
                                  add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]
    with torch.no_grad():
        o = m.generate(**enc, max_new_tokens=n, do_sample=temp > 0, temperature=temp if temp > 0 else None,
                       top_p=0.95 if temp > 0 else None, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, nin:], skip_special_tokens=True)

res = {}
# (a) 1.5B self-consistency
log("carregando 1.5B-Instruct")
P = f"{MODELS}/Qwen2.5-1.5B-Instruct"; tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
single = maj = anyp = 0; K = 5
for q, keys in HARD:
    hits = [match(gen(m, tok, q, temp=0.0 if s == 0 else 0.7), keys) for s in range(K)]
    single += hits[0]; maj += (sum(hits) >= (K // 2 + 1)); anyp += any(hits)
res["1.5B_single"] = {"acertos": single, "de": len(HARD)}
res["1.5B_maioria5"] = {"acertos": maj, "de": len(HARD)}
res["1.5B_pass5"] = {"acertos": anyp, "de": len(HARD)}
log(f"  1.5B: single {single}/{len(HARD)} | maioria@5 {maj}/{len(HARD)} | pass@5 {anyp}/{len(HARD)}")
del m; gc.collect(); torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

# (b) 7B int8 (teto real)
log("carregando 7B-Instruct int8 (quanto)")
P7 = f"{MODELS}/Qwen2.5-7B-Instruct"; tok7 = AutoTokenizer.from_pretrained(P7)
m7 = AutoModelForCausalLM.from_pretrained(P7, dtype=torch.float16, quantization_config=QuantoConfig(weights="int8"),
                                          device_map="cuda", low_cpu_mem_usage=True)
ok7 = sum(match(gen(m7, tok7, q, temp=0.0), keys) for q, keys in HARD)
res["7B_int8"] = {"acertos": ok7, "de": len(HARD), "VRAM_GB": round(torch.cuda.memory_allocated()/1e9, 1)}
log(f"  7B-int8: {ok7}/{len(HARD)} ({res['7B_int8']['VRAM_GB']}GB)")
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== 1.5B single {single} → maioria@5 {maj} (pass@5 {anyp}) | 7B-int8 {ok7} /{len(HARD)} ===")
log(f"DONE M153 ({time.time()-t0:.0f}s)")
