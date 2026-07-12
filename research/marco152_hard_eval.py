#!/usr/bin/env python3
"""M152 — AVALIAÇÃO DIFÍCIL: acha onde o 7B REALMENTE ganha do 1.5B (raciocínio multi-passo, pegadinhas,
conhecimento fino). 1.5B via transformers, 7B via GGUF/llama-cpp (RÁPIDO na VRAM). Dump marco152_metrics.json."""
import json, time, gc
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco152_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

HARD = [
    ("Se hoje é terça-feira, que dia da semana será daqui a 100 dias?", ["quinta"]),
    ("Um pai tem 40 anos e o filho tem 10. Em quantos anos o pai terá exatamente o dobro da idade do filho?", ["20"]),
    ("Quantos inteiros de 1 a 100 são divisíveis por 3 ou por 5?", ["47"]),
    ("Qual é o menor número primo maior que 50?", ["53"]),
    ("Quem foi o segundo ser humano a pisar na Lua?", ["aldrin"]),
    ("Uma bola e uma raquete custam 1,10 no total. A raquete custa 1,00 a mais que a bola. Quanto custa a bola?", ["0,05", "0.05", "5 cent", "cinco cent"]),
    ("Quantos meses do ano têm pelo menos 28 dias?", ["12", "doze", "todos"]),
    ("Se 5 máquinas fazem 5 peças em 5 minutos, quantos minutos 100 máquinas levam para fazer 100 peças?", ["5", "cinco"]),
    ("Qual é o resultado de 7 - 3 x 2 + 4?", ["5"]),
    ("Numa corrida, você ultrapassa o segundo colocado. Em que posição você fica?", ["segund"]),
    ("Se um tijolo pesa 1 kg mais meio tijolo, quanto pesa o tijolo inteiro?", ["2", "dois"]),
    ("Complete a sequência: 2, 3, 5, 8, 13, ?", ["21"]),
    ("Qual é maior: 3/4 ou 5/7?", ["3/4", "três quart", "0,75"]),
    ("Todos os gatos são mamíferos. Alguns mamíferos são pretos. É correto concluir que algum gato é preto? Responda sim ou não.", ["não", "nao"]),
    ("Se ontem foi sexta-feira, que dia será depois de amanhã?", ["segunda"]),
    ("Um caracol sobe 3 m de dia e escorrega 2 m à noite num poço de 10 m. Em quantos dias chega ao topo?", ["8", "oito"]),
    ("Quanto é 12 ao quadrado menos 11 ao quadrado?", ["23"]),
    ("Pedro é irmão de Ana. A mãe de Ana é Maria. Quem é Maria para Pedro?", ["mãe", "mae"]),
]

def match(txt, keys): t = txt.lower(); return any(k in t for k in keys)

def eval_hf(P, label):  # transformers
    tok = AutoTokenizer.from_pretrained(P)
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
    ok = 0; det = []; tps = []
    for q, keys in HARD:
        enc = tok.apply_chat_template([{"role": "user", "content": q + " Pense passo a passo e dê a resposta final."}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]; t = time.time()
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=130, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0, nin:], skip_special_tokens=True); h = match(txt, keys); ok += h
        tps.append(130 / (time.time() - t)); det.append({"q": q[:34], "hit": bool(h)})
    del m; gc.collect(); torch.cuda.empty_cache()
    r = {"acertos": ok, "de": len(HARD), "tok_s": round(float(np.mean(tps)), 1), "detalhe": det}
    log(f"  {label}: {ok}/{len(HARD)} | {r['tok_s']} tok/s"); return r

def eval_quanto(P, label):  # 7B int4 via quanto (VRAM, confiável)
    from transformers import QuantoConfig
    tok = AutoTokenizer.from_pretrained(P)
    m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16, quantization_config=QuantoConfig(weights="int4"),
                                             device_map="cuda", low_cpu_mem_usage=True)
    ok = 0; det = []; tps = []
    for q, keys in HARD:
        enc = tok.apply_chat_template([{"role": "user", "content": q + " Pense passo a passo e dê a resposta final."}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]; t = time.time()
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=130, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0, nin:], skip_special_tokens=True); h = match(txt, keys); ok += h
        tps.append(130 / (time.time() - t)); det.append({"q": q[:34], "hit": bool(h)})
    del m; gc.collect(); torch.cuda.empty_cache()
    r = {"acertos": ok, "de": len(HARD), "tok_s": round(float(np.mean(tps)), 1), "detalhe": det}
    log(f"  {label}: {ok}/{len(HARD)} | {r['tok_s']} tok/s"); return r

res = {}
res["1.5B_instruct"] = eval_hf(f"{MODELS}/Qwen2.5-1.5B-Instruct", "1.5B-Instruct")
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
res["7B_int4"] = eval_quanto(f"{MODELS}/Qwen2.5-7B-Instruct", "7B-Instruct int4")
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
a = res.get("1.5B_instruct", {}).get("acertos", 0); b = res.get("7B_int4", {}).get("acertos", "?")
log(f"=== DIFÍCIL: 1.5B {a}/{len(HARD)} vs 7B {b}/{len(HARD)} — gap={b-a if isinstance(b,int) else '?'} ===")
log(f"DONE M152 ({time.time()-t0:.0f}s)")
