#!/usr/bin/env python3
"""BYTEQWEN — transplante dos pesos do Qwen-1.5B para I/O de BYTES (a ideia do Leonardo, versão real):
mantém as 28 camadas de raciocínio do Qwen (os ~1.5B de pesos), TROCA embedding 151936→256 e head →256,
e re-treina a fronteira (embed/head/norm + primeiras e últimas camadas) pra alinhar do token-space pro
byte-space. Reusa 'todos os pesos', ajusta o que precisa. Congela o miolo (economia + preserva raciocínio).
Treina next-byte no corpus rico. Salva ckpt + gera amostras. GPU (12GB): grad-checkpointing + batch pequeno."""
import sys, time, math, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
QWEN = "/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-1.5B-Instruct"
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt"   # prosa + instrução
CKDIR = "/home/leonardo/projects/LLM/bytebrain/research/ckpt_byteqwen"; os.makedirs(CKDIR, exist_ok=True)
CTX, BS, STEPS = 256, 3, 40000

log("carregando Qwen-1.5B...")
m = AutoModelForCausalLM.from_pretrained(QWEN, dtype=torch.float16).to(DEV)
H = m.config.hidden_size
# --- TRANSPLANTE: troca I/O pra bytes (mantém as 28 camadas) ---
m.model.embed_tokens = nn.Embedding(256, H).to(DEV).to(torch.float32)   # treináveis em fp32 (AMP)
m.lm_head = nn.Linear(H, 256, bias=False).to(DEV).to(torch.float32)
nn.init.normal_(m.model.embed_tokens.weight, std=0.02); nn.init.normal_(m.lm_head.weight, std=0.02)
m.config.vocab_size = 256; m.config.tie_word_embeddings = False
if os.path.exists(f"{CKDIR}/ck.pt"):
    sd = torch.load(f"{CKDIR}/ck.pt", map_location=DEV); m.load_state_dict(sd["model"]); start = sd["step"]
    log(f"RESUMIDO do passo {start}")
else:
    start = 0
# --- congela o miolo do raciocínio; treina fronteira + 1ªs/últimas camadas (alinhamento) ---
for p in m.parameters(): p.requires_grad_(False)
train_mods = [m.model.embed_tokens, m.lm_head, m.model.norm] + list(m.model.layers[:2]) + list(m.model.layers[-2:])
for mod in train_mods:
    for p in mod.parameters(): p.data = p.data.float(); p.requires_grad_(True)   # treináveis em fp32
m.gradient_checkpointing_enable(); m.train()
tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
log(f"ByteQwen: {sum(p.numel() for p in m.parameters())/1e9:.2f}B total, treinando {tp/1e6:.0f}M (fronteira+alinhamento), miolo CONGELADO")

data = np.memmap(CORPUS, dtype=np.uint8, mode="r"); n = len(data)
def batch():
    ix = np.random.randint(0, n - CTX - 1, BS)
    x = np.stack([np.asarray(data[i:i+CTX+1]) for i in ix]).astype(np.int64)
    t = torch.from_numpy(x).to(DEV); return t[:, :-1], t[:, 1:]
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

@torch.no_grad()
def sample(prompt, k=120):
    m.eval(); ids = list(prompt.encode())
    x = torch.tensor([ids], device=DEV)
    for _ in range(k):
        with torch.amp.autocast("cuda"):
            lg = m(x[:, -CTX:]).logits[0, -1]
        p = F.softmax(lg.float()/0.7, -1); nx = int(torch.multinomial(p, 1)); ids.append(nx)
        x = torch.cat([x, torch.tensor([[nx]], device=DEV)], 1)
    m.train(); return bytes(ids).decode("utf-8", "ignore")

log("treinando (next-byte, corpus rico prosa+instrução)...")
for s in range(start, STEPS):
    xb, yb = batch()
    with torch.amp.autocast("cuda"):
        loss = F.cross_entropy(m(xb).logits.reshape(-1, 256), yb.reshape(-1))
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if s % 100 == 0:
        log(f"  passo {s}: loss {loss.item():.3f} ({loss.item()/math.log(2):.3f} bpb)")
    if s % 1000 == 0 and s > start:
        torch.save({"model": m.state_dict(), "step": s}, f"{CKDIR}/ck.pt")
        log(f"  [amostra] {sample('P: O que é inteligência artificial?\\nR:')!r}")
torch.save({"model": m.state_dict(), "step": STEPS}, f"{CKDIR}/ck.pt")
log("DONE")
