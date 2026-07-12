#!/usr/bin/env python3
"""M114 — BATERIA NOTURNA / SABEDORIA DESTILADA: modelo BOM (Phi-4-mini) = professor;
modelo pequeno congelado (Qwen2.5-Math-1.5B) = decoder. Destila o conhecimento REAL do professor
em SEMENTES (soft-prompt) no aluno e mede: quanto o aluno já sabia, quantos fatos a semente
consegue ARMAZENAR, e o custo em bytes/fato. Carga sequencial (professor → free → aluno) p/ caber
em 12GB. GPU obrigatória. Dump marco114_metrics.json."""
import json, time, gc
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"
TEACHER = f"{MODELS}/Phi-4-mini-instruct"; STUDENT = f"{MODELS}/Qwen2.5-Math-1.5B"
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

QS = [
    "Qual é a capital da Austrália?", "Qual é o maior planeta do sistema solar?",
    "Quem pintou a Mona Lisa?", "Qual é o metal mais leve?", "Em que ano o homem pisou na Lua?",
    "Qual é o maior oceano da Terra?", "Qual é o rio mais longo do mundo?",
    "Qual é o osso mais longo do corpo humano?", "Qual gás as plantas absorvem?",
    "Qual é a moeda do Japão?", "Quantos lados tem um hexágono?", "Qual é o símbolo químico do ouro?",
    "Qual é a montanha mais alta do mundo?", "Quem escreveu Dom Casmurro?",
    "Qual é o menor país do mundo?", "Qual planeta é conhecido como planeta vermelho?",
    "Qual é a capital do Canadá?", "Quantos continentes existem?",
    "Qual é o animal terrestre mais rápido?", "Qual é a velocidade da luz aproximada em km/s?",
    "Qual é o maior mamífero do mundo?", "Quem descobriu a gravidade?",
    "Qual é o elemento mais abundante no universo?", "Qual é a capital da Coreia do Sul?",
    "Qual é o maior deserto quente do mundo?", "Quantos ossos tem o corpo humano adulto?",
    "Qual é a língua mais falada do mundo?", "Quem foi o primeiro presidente dos EUA?",
    "Qual é o ponto de ebulição da água em Celsius?", "Qual é a capital da Argentina?",
    "Qual é o maior órgão do corpo humano?", "Quantos planetas há no sistema solar?",
    "Qual é a fórmula química do sal de cozinha?", "Qual é a capital do Egito?",
    "Quem escreveu Romeu e Julieta?", "Qual é o metal líquido à temperatura ambiente?",
    "Qual é a maior floresta tropical do mundo?", "Qual é a capital da Alemanha?",
    "Quantas cordas tem um violino?", "Qual é o planeta mais próximo do Sol?",
]

# ================= FASE 1: PROFESSOR responde =================
log(f"FASE 1: professor Phi-4-mini responde {len(QS)} perguntas")
try:  # Phi3 NATIVO (sem trust_remote_code: o modeling_phi3.py do repo importa LossKwargs, removido no tr5.12)
    ttok = AutoTokenizer.from_pretrained(TEACHER)
    tmodel = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float16).to(DEV).eval()
    tname = "Phi-4-mini"
except Exception as e:
    log(f"Phi falhou ({e}); fallback SmolLM2-1.7B")
    TEACHER = f"{MODELS}/SmolLM2-1.7B"; ttok = AutoTokenizer.from_pretrained(TEACHER)
    tmodel = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float16).to(DEV).eval()
    tname = "SmolLM2-1.7B"

@torch.no_grad()
def ask(q):
    msg = [{"role": "user", "content": f"Responda com no máximo 3 palavras, sem explicação. {q}"}]
    try:
        enc = ttok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(DEV) for k, v in enc.items()}; nin = enc["input_ids"].shape[1]
    except Exception:
        ids = ttok(f"Pergunta: {q}\nResposta:", return_tensors="pt").input_ids.to(DEV)
        enc = {"input_ids": ids}; nin = ids.shape[1]
    out = tmodel.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=ttok.eos_token_id)
    txt = ttok.decode(out[0, nin:], skip_special_tokens=True).strip()
    return txt.split("\n")[0].strip().strip(".").strip()[:40]

qa = []
for q in QS:
    a = ask(q)
    if a: qa.append((q, a))
    log(f"  P: {q[:40]:40} → {a}")
log(f"professor ({tname}) respondeu {len(qa)}/{len(QS)}")
del tmodel; gc.collect(); torch.cuda.empty_cache()

# ================= FASE 2: ALUNO congelado + destilação em semente =================
log("FASE 2: aluno Math-1.5B congelado — baseline + destilação")
tok = AutoTokenizer.from_pretrained(STUDENT)
model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16).to(DEV).eval()
for p in model.parameters(): p.requires_grad_(False)
H = model.config.hidden_size; EL = model.get_input_embeddings()

def emb(ids): return EL(torch.tensor([ids], device=DEV)).detach()[0]

@torch.no_grad()
def student_knows(q, a):
    """aluno já responde? (greedy, primeiros tokens batem com o alvo do professor)"""
    prompt = f"Pergunta: {q}\nResposta:"; pid = tok(prompt).input_ids
    tid = tok(" " + a.lstrip(), add_special_tokens=False).input_ids
    cur = emb(pid)[None]; gen = []
    for _ in range(len(tid) + 2):
        lg = model(inputs_embeds=cur).logits[0, -1]; nx = int(lg.argmax()); gen.append(nx)
        cur = torch.cat([cur, EL(torch.tensor([[nx]], device=DEV))], 1)
    return a.lstrip().lower()[:12] in tok.decode(gen).lower()

def plant(q, a, K, steps=200):
    prompt = f"Pergunta: {q}\nResposta:"; pid = tok(prompt).input_ids
    tid = tok(" " + a.lstrip(), add_special_tokens=False).input_ids
    pe = emb(pid); te = emb(tid); st = K + len(pid)
    seed = nn.Parameter(torch.randn(K, H, device=DEV, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([seed], lr=0.3); tgt = torch.tensor(tid, device=DEV)
    for s in range(steps):
        inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
        lg = model(inputs_embeds=inp).logits[0]
        loss = F.cross_entropy(lg[st-1:st-1+len(tid)].float(), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        inp = torch.cat([seed.to(torch.float16), pe, te], 0)[None]
        pred = model(inputs_embeds=inp).logits[0][st-1:st-1+len(tid)].argmax(-1)
    return seed.detach().to(torch.float16), bool((pred == tgt).all()), len(tid)

def q_int4(seed):
    qm = 7; s = (seed.abs().max() / qm).clamp_min(1e-8)
    return ((seed.float() / s).round().clamp(-qm, qm) * s).to(torch.float16)

@torch.no_grad()
def recall_int4(q, a, seed):
    prompt = f"Pergunta: {q}\nResposta:"; pid = tok(prompt).input_ids
    tid = tok(" " + a.lstrip(), add_special_tokens=False).input_ids
    pe = emb(pid); te = emb(tid); st = seed.shape[0] + len(pid)
    inp = torch.cat([q_int4(seed).to(torch.float16), pe, te], 0)[None]
    pred = model(inputs_embeds=inp).logits[0][st-1:st-1+len(tid)].argmax(-1)
    return bool((pred == torch.tensor(tid, device=DEV)).all())

knew = planted = planted4 = 0; det = []; toktot = 0
for q, a in qa:
    if student_knows(q, a):
        knew += 1; det.append({"q": q[:30], "a": a, "estado": "aluno_ja_sabia"}); continue
    seed, ok, ntok = plant(q, a, K=1)
    ok4 = recall_int4(q, a, seed) if ok else False
    planted += ok; planted4 += ok4; toktot += ntok
    det.append({"q": q[:30], "a": a, "estado": "plantado" if ok else "falhou",
                "fp16": ok, "int4": ok4, "ntok": ntok})
    log(f"  destila [{q[:34]:34}={a[:14]:14}] fp16 {'ok' if ok else 'X'} int4 {'ok' if ok4 else 'X'}")

n_unknown = len(qa) - knew
bytes_por_fato = round(1 * H * 4 / 8 / max(1, planted4), 1)  # K=1 int4 por fato
RES = {"teacher": tname, "n_perguntas": len(qa), "aluno_ja_sabia": knew,
       "desconhecidos": n_unknown, "plantados_fp16": planted, "plantados_int4": planted4,
       "taxa_destilacao_int4": round(planted4 / max(1, n_unknown), 2),
       "bytes_por_fato_int4_K1": bytes_por_fato, "detalhe": det}
json.dump(RES, open("/home/leonardo/projects/LLM/bytebrain/research/marco114_metrics.json", "w"),
          ensure_ascii=False, indent=1)
log(f"RESUMO: aluno já sabia {knew}/{len(qa)} | destilados int4 {planted4}/{n_unknown} "
    f"({RES['taxa_destilacao_int4']}) | {bytes_por_fato} B/fato")
log(f"DONE M114 ({time.time()-t0:.0f}s)")
