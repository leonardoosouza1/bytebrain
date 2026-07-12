#!/usr/bin/env python3
"""M151 — BATERIA DE SABEDORIA: 7B-int4 (quanto, 5.7GB VRAM) vs 1.5B-Instruct. Mede conhecimento +
raciocínio por geração livre (match de resposta) + tok/s. Prova: o 7B-int4 leve tem sabedoria de 7B
que o 1.5B não tem. GPU. Dump marco151_metrics.json + journal."""
import json, time, gc
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco151_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

Q = [  # (pergunta, resposta-chave p/ match) — conhecimento + raciocínio, sabedoria 7B
    ("Qual é a capital da Austrália?", "canberra"),
    ("Quem escreveu a Odisseia?", "homero"),
    ("Qual é o elemento químico de símbolo Fe?", "ferro"),
    ("Em que ano começou a Segunda Guerra Mundial?", "1939"),
    ("Qual é o maior planeta do sistema solar?", "júpiter"),
    ("Quem desenvolveu a teoria da evolução por seleção natural?", "darwin"),
    ("Qual é a raiz quadrada de 144?", "12"),
    ("Se um trem viaja a 80 km/h por 2,5 horas, quantos km percorre?", "200"),
    ("Quantos lados tem um dodecágono?", "12"),
    ("Qual é o resultado de 15% de 240?", "36"),
    ("Numa sala com 40 pessoas, 60% são mulheres. Quantos homens há?", "16"),
    ("Se x + 3 = 10 e y = 2x, quanto vale y?", "14"),
    ("Qual é a capital do Canadá?", "ottawa"),
    ("Que gás as plantas absorvem na fotossíntese?", "carbono"),
    ("Qual é o oceano que banha a costa leste do Brasil?", "atlântico"),
    ("Quem pintou o teto da Capela Sistina?", "michelangelo"),
]
CHAT = ["Explique em 2 frases o que é inteligência artificial.",
        "Dê 3 dicas curtas para dormir melhor.",
        "Resuma em 1 frase o enredo de Romeu e Julieta."]

def eval_model(model, tok, use_chat_tmpl, label):
    def gen(prompt, n=40):
        if use_chat_tmpl:
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]
        else:
            ids = tok(f"Pergunta: {prompt}\nResposta:", return_tensors="pt").input_ids.to(DEV); enc = {"input_ids": ids}; nin = ids.shape[1]
        t = time.time()
        o = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(o[0, nin:], skip_special_tokens=True).strip()
        return txt, n / (time.time() - t + 1e-6)
    ok = 0; tps = []; det = []
    for q, a in Q:
        txt, s = gen(q, 40); hit = a in txt.lower(); ok += hit; tps.append(s)
        det.append({"q": q[:30], "hit": bool(hit), "resp": txt[:60]})
    chats = []
    for c in CHAT:
        txt, s = gen(c, 70); chats.append({"pergunta": c[:40], "resposta": txt[:200]})
    log(f"  {label}: {ok}/{len(Q)} conhecimento/raciocínio | {np.mean(tps):.1f} tok/s")
    return {"acertos": ok, "de": len(Q), "tok_s": round(float(np.mean(tps)), 1), "detalhe": det, "chats": chats}

res = {}
# 1.5B-Instruct (transformers fp16)
log("carregando Qwen2.5-1.5B-Instruct (fp16)")
P = f"{MODELS}/Qwen2.5-1.5B-Instruct"; tok = AutoTokenizer.from_pretrained(P)
m = AutoModelForCausalLM.from_pretrained(P, dtype=torch.float16).to(DEV).eval()
res["1.5B_fp16"] = eval_model(m, tok, True, "1.5B-Instruct")
res["1.5B_fp16"]["VRAM_GB"] = round(torch.cuda.memory_allocated()/1e9, 1)
del m; gc.collect(); torch.cuda.empty_cache()
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

# 7B-Instruct int4 (quanto)
log("carregando Qwen2.5-7B-Instruct int4 (quanto)")
P7 = f"{MODELS}/Qwen2.5-7B-Instruct"; tok7 = AutoTokenizer.from_pretrained(P7)
m7 = AutoModelForCausalLM.from_pretrained(P7, dtype=torch.float16, quantization_config=QuantoConfig(weights="int4"),
                                          device_map="cuda", low_cpu_mem_usage=True)
res["7B_int4"] = eval_model(m7, tok7, True, "7B-Instruct int4")
res["7B_int4"]["VRAM_GB"] = round(torch.cuda.memory_allocated()/1e9, 1)
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

log(f"=== SABEDORIA: 1.5B {res['1.5B_fp16']['acertos']}/{len(Q)} ({res['1.5B_fp16']['VRAM_GB']}GB, {res['1.5B_fp16']['tok_s']}tok/s) "
    f"vs 7B-int4 {res['7B_int4']['acertos']}/{len(Q)} ({res['7B_int4']['VRAM_GB']}GB, {res['7B_int4']['tok_s']}tok/s) ===")
log(f"DONE M151 ({time.time()-t0:.0f}s)")
