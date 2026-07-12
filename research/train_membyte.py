#!/usr/bin/env python3
"""TREINA o MemByte — byte-model ESPECIALIZADO como órgão de MEMÓRIA da IARA. Não precisa saber fatos
(a sabedoria ele herda via sementes/wisdom-bridge); precisa ter a INFRA 100%:
  (1) FRONTEIRA nativa: treinado no formato 'P: {q}\\nR: {a}\\n\\n' -> aprende a PARAR após a resposta (recall limpo).
  (2) PRIORS diversos: vocabulário rico (nomes/lugares/números/palavras PT) -> sementes baratas p/ qualquer fato.
  (3) LEVE: ~5M params (dim=256, 6L). Corpus SINTÉTICO gerado aqui (sem depender de disco).
Salva membyte.pt. Determinístico (seed fixa). GPU."""
import sys, time, math, os, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cuda"; t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
rng = random.Random(1234); np.random.seed(1234); torch.manual_seed(1234)

# ---- CORPUS SINTÉTICO: Q&A diverso com fronteira \n\n, vocabulário rico p/ priors de byte -----------
NOMES = ["Leonardo","Ana","Vaelis","Zephyr","Orin","Miro","Kael","Lia","Bruno","Nara","Tavo","Isa","Ravi",
         "Sol","Dara","Enzo","Yara","Caio","Vera","Ravel","Odin","Nix","Lua"," Favio".strip(),"Tomé","Ursa"]
LUGARES = ["Krylon","Bogotá","Ulan Bator","Timbuktu","Vaelis","Nármed","Solaris","Aritê","Bélgica","Cairo",
           "Quênia","Oslo","Praga","Delhi","Malta","Fênix","Ízmir","Recife","Açores","Óbidos","Uçá","Ïo"]
COISAS = ["girassol","obsidiana","neônio","gálio","qubit","tungstênio","âmbar","zinco","código","semente",
          "cristal","fóton","enzima","vórtice","áurea","íon","rúbi","ébano","úmero","çedilha"]
def num(): return str(rng.randint(1,9999))
def fact():
    t = rng.random()
    if t < 0.3:
        e, a = rng.choice(NOMES), rng.choice(LUGARES); return f"Qual o planeta natal de {e}?", a
    if t < 0.5:
        e, a = rng.choice(LUGARES), rng.choice(NOMES); return f"Quem e o guardiao de {e}?", a
    if t < 0.7:
        e, a = rng.choice(COISAS), num(); return f"Qual o codigo de {e}?", a
    if t < 0.85:
        e, a = rng.choice(LUGARES), num(); return f"Em que ano fundaram {e}?", a
    e, a = rng.choice(NOMES), rng.choice(COISAS); return f"Qual o simbolo de {e}?", a

def build_corpus(n_lines=40000):
    parts = []
    for _ in range(n_lines):
        q, a = fact(); parts.append(f"P: {q}\nR: {a}\n\n")
    blob = "".join(parts).encode("utf-8")
    return np.frombuffer(blob, dtype=np.uint8).copy()

log("gerando corpus sintético...")
data = build_corpus(); log(f"corpus: {len(data)/1e6:.1f} MB de bytes, vocab rico Q&A com fronteira \\n\\n")

# ---- treino ----
DIM, LAYERS, HEADS, CTX, BS, STEPS = 256, 6, 8, 128, 96, 6000
m = ByteGPT(dim=DIM, n_layers=LAYERS, n_heads=HEADS, context=CTX).to(DEV); m.train()
log(f"MemByte {m.n_params/1e6:.1f}M params (dim={DIM}, {LAYERS}L). treinando {STEPS} passos...")
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01, betas=(0.9,0.95))
def get_batch():
    ix = np.random.randint(0, len(data) - CTX - 1, BS)
    x = np.stack([data[i:i+CTX+1] for i in ix]).astype(np.int64)
    t = torch.from_numpy(x).to(DEV); return t[:, :-1], t[:, 1:]
best = 9.9
for s in range(STEPS):
    lr = 3e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
    for g in opt.param_groups: g["lr"] = lr
    xb, yb = get_batch()
    logits = m(xb)
    loss = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if s % 500 == 0 or s == STEPS - 1:
        bpb = loss.item() / math.log(2)
        log(f"  passo {s}: loss {loss.item():.3f} ({bpb:.3f} bits/byte)")
        best = min(best, bpb)
out = "/home/leonardo/projects/LLM/bytebrain/research/membyte.pt"
torch.save({"model": m.state_dict(), "cfg": {"dim": DIM, "n_layers": LAYERS, "n_heads": HEADS, "context": CTX}}, out)
log(f"salvo em {out} | melhor {best:.3f} bits/byte")

# ---- validação rápida: gera no formato (deve parar em \n\n) ----
m.eval()
@torch.no_grad()
def gen(prompt, n=40):
    x = torch.tensor([list(prompt.encode())], device=DEV)
    for _ in range(n):
        nx = int(m(x[:, -CTX:])[0, -1].argmax()); x = torch.cat([x, torch.tensor([[nx]], device=DEV)], 1)
        if x.shape[1] >= 4 and x[0, -2:].tolist() == [10, 10]: break
    return bytes(x[0].tolist()).decode("utf-8", "ignore")
log("== geração de formato (aprendeu P:/R: e a fronteira?) ==")
for p in ["P: Qual o planeta natal de Zephyr?\nR:", "P: Qual o codigo de girassol?\nR:"]:
    log(f"  {gen(p)!r}")
log(f"DONE ({time.time()-t0:.0f}s)")
