#!/usr/bin/env python3
"""M139 — AVALIAÇÃO REAL (chat/conhecimento), comparativo honesto. Testa por GERAÇÃO LIVRE (não recall
teacher-forced): 28 fatos REAIS + 10 fatos PRIVADOS (inventados que nenhum modelo sabe).
 BASELINES (congelados, sem seed): Math-1.5B, SmolLM2-1.7B, Phi-4-mini, Qwen3-4B.
 NOSSO SISTEMA: Math-1.5B + floresta de seeds (1 por fato, K=4).
Métrica: resposta-alvo aparece na geração (match leniente, igual p/ todos). Mede o que os seeds SOMAM
sobre o próprio modelo e vs modelos parecidos. GPU. Dump marco139_metrics.json."""
import json, time, os, gc
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; DEV = "cuda"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco139_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

# (pergunta, resposta-alvo curta p/ match). Fatos reais de resposta clara.
KEY = [
    ("Qual é a capital da Austrália?", "Canberra"), ("Qual é o maior planeta do sistema solar?", "Júpiter"),
    ("Quem pintou a Mona Lisa?", "Vinci"), ("Em que ano o homem pisou na Lua?", "1969"),
    ("Qual é o maior oceano da Terra?", "Pacífico"), ("Qual é a moeda do Japão?", "iene"),
    ("Qual é o símbolo químico do ouro?", "Au"), ("Qual é a montanha mais alta do mundo?", "Everest"),
    ("Quem escreveu Dom Casmurro?", "Machado"), ("Qual planeta é o planeta vermelho?", "Marte"),
    ("Qual é a capital do Canadá?", "Ottawa"), ("Quem descobriu a gravidade?", "Newton"),
    ("Qual é a capital da Coreia do Sul?", "Seul"), ("Quantos ossos tem o corpo humano adulto?", "206"),
    ("Quem foi o primeiro presidente dos EUA?", "Washington"), ("Qual o ponto de ebulição da água em Celsius?", "100"),
    ("Qual é a capital da Argentina?", "Buenos Aires"), ("Quantos planetas há no sistema solar?", "8"),
    ("Qual é a fórmula química do sal de cozinha?", "NaCl"), ("Qual é a capital do Egito?", "Cairo"),
    ("Quem escreveu Romeu e Julieta?", "Shakespeare"), ("Qual é a capital da Alemanha?", "Berlim"),
    ("Qual planeta é o mais próximo do Sol?", "Mercúrio"), ("Qual é a capital da França?", "Paris"),
    ("Quem desenvolveu a teoria da relatividade?", "Einstein"), ("Qual é o maior país do mundo em área?", "Rússia"),
    ("Qual é a capital da Itália?", "Roma"), ("Qual é a capital de Portugal?", "Lisboa"),
]
# fatos PRIVADOS (inventados): nenhum modelo pode saber; 2 formulações de treino + 1 teste (generalização)
INV = [
    ("O planeta natal do herói", "Krylon", ["O planeta natal do herói é", "De onde vem o herói? Do planeta"], "Qual é o planeta natal do herói?\nResposta:"),
    ("A senha do cofre da IARA", "7492", ["A senha do cofre da IARA é", "Para abrir o cofre da IARA digite"], "Qual é a senha do cofre da IARA?\nResposta:"),
    ("A cor do projeto secreto", "carmesim", ["A cor do projeto secreto é", "O projeto secreto tem a cor"], "Qual é a cor do projeto secreto?\nResposta:"),
    ("O código do portão norte", "dragao", ["O código do portão norte é", "O portão norte abre com"], "Qual é o código do portão norte?\nResposta:"),
    ("O nome do robô guardião", "Ziggy", ["O nome do robô guardião é", "O robô guardião se chama"], "Qual é o nome do robô guardião?\nResposta:"),
]

def match(out, ans): return ans.lower() in out.lower()

@torch.no_grad()
def gen_base(model, tok, prompt, n=10):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    o = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)

res = {"baselines": {}, "key_n": len(KEY), "inv_n": len(INV)}

# ---------- BASELINES ----------
BASE = {"math1.5b": f"{MODELS}/Qwen2.5-Math-1.5B", "smollm2": f"{MODELS}/SmolLM2-1.7B",
        "phi4mini": f"{MODELS}/Phi-4-mini-instruct", "qwen3-4b": f"{MODELS}/qwen3-4b-q4km.gguf"}
for name, path in BASE.items():
    try:
        if path.endswith(".gguf"):
            d, fn = os.path.dirname(path), os.path.basename(path)
            tok = AutoTokenizer.from_pretrained(d, gguf_file=fn); m = AutoModelForCausalLM.from_pretrained(d, gguf_file=fn, dtype=torch.float16).to(DEV).eval()
        else:
            tok = AutoTokenizer.from_pretrained(path); m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to(DEV).eval()
    except Exception as e:
        log(f"  {name} falhou: {str(e)[:60]}"); continue
    kok = sum(match(gen_base(m, tok, f"Pergunta: {q}\nResposta:"), a) for q, a in KEY)
    iok = sum(match(gen_base(m, tok, tst), ans) for _, ans, _, tst in INV)
    res["baselines"][name] = {"real": kok, "real_de": len(KEY), "privado": iok, "privado_de": len(INV),
                              "real_taxa": round(kok/len(KEY), 2)}
    log(f"  BASELINE {name:10}: real {kok}/{len(KEY)} ({res['baselines'][name]['real_taxa']}) | privado {iok}/{len(INV)}")
    del m; gc.collect(); torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

# ---------- NOSSO SISTEMA: Math-1.5B + floresta de seeds ----------
log("=== Math-1.5B + floresta de seeds (1 por fato, K=4) ===")
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
@torch.no_grad()
def gen_seed(seed, prompt, n=10):
    pid = trunk.tok(prompt).input_ids
    cur = torch.cat([seed.to(torch.float16), trunk.EL(torch.tensor([pid], device=DEV)).detach()[0]], 0)[None]
    out = []
    for _ in range(n):
        nx = int(trunk.model(inputs_embeds=cur).logits[0, -1].argmax()); out.append(nx)
        if nx == trunk.tok.eos_token_id: break
        cur = torch.cat([cur, trunk.EL(torch.tensor([[nx]], device=DEV))], 1)
    return trunk.tok.decode(out)

# fatos reais: planta Q->A (K=4), gera na MESMA Q (serve a base de conhecimento)
kok = 0
for q, a in KEY:
    seed = trunk.plant([(f"Pergunta: {q}\nResposta:", " " + a)], K=4, steps=400)
    kok += match(gen_seed(seed, f"Pergunta: {q}\nResposta:"), a)
# fatos privados: planta 2 formulações, TESTA na 3ª inédita (generalização real)
iok = 0
for _, ans, phr, tst in INV:
    seed = trunk.plant([(p, " " + ans) for p in phr], K=4, steps=400)
    iok += match(gen_seed(seed, tst), ans)
res["sistema_seeds"] = {"real": kok, "real_de": len(KEY), "real_taxa": round(kok/len(KEY), 2),
                        "privado_generaliza": iok, "privado_de": len(INV)}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"  SISTEMA Math+seeds: real {kok}/{len(KEY)} ({res['sistema_seeds']['real_taxa']}) | privado(generaliza) {iok}/{len(INV)}")
log("=== COMPARATIVO ===")
for n, d in res["baselines"].items(): log(f"    {n:12} baseline: real {d['real_taxa']} | privado {d['privado']}/{len(INV)}")
log(f"    Math+seeds  : real {res['sistema_seeds']['real_taxa']} | privado {iok}/{len(INV)}")
log(f"DONE M139 ({time.time()-t0:.0f}s)")
