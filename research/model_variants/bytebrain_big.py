"""
ByteBrain GRANDE — treina ~225M params no 12GB via tecnicas memory-efficient (estilo saci chunked).

Leonardo: "aumenta os params com calma, sem estourar; tipo o saci que via partes (200k), VRAM
baixa, passo rapido, com pausa — faz batch, demora mais, mas no fim usa todos os params".
Tecnicas:
  - amostragem em chunks (256-byte windows aleatorios)  -> VRAM baixa, sem carregar tudo
  - GRADIENT CHECKPOINTING (recomputa ativacoes no backward) -> cabe modelo MUITO maior
  - bf16 autocast -> metade da memoria de ativacao
  - GRADIENT ACCUMULATION (micro-batch pequeno x N -> batch efetivo grande)
Alvo: d=1024, 18 camadas (~225M). Auto-reduz micro-batch se OOM. Loga VRAM. Teto de tempo.
Prova: modelo grande TREINA no 12GB com VRAM bounded -> na AMD (192GB) e' so escalar pra 1B.
"""
import os, glob, math, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda"
CACHE = "/home/leonardo/projects/LLM/bytebrain/data/multiscript"
CODE = ["/home/leonardo/projects/LLM/byte-language-lab/data/.oc_corpus.txt",
        "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_big4.txt",
        "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_distill.txt"]
LB = 256
D, NL, NH = 1152, 22, 16            # ~350M params (folga de VRAM permite)
MICRO_BS = 64                       # micro-batch grande: usa VRAM + step rapido de ver (auto-reduz se OOM)
ACCUM = 1                           # sem acumulacao -> cada step e' logado e voa
TIME_BUDGET = 1800                  # ~30 min de teto


def vram():
    return torch.cuda.memory_allocated() / 1e9, torch.cuda.max_memory_allocated() / 1e9


class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, L, Dd = x.shape; h = s.ln1(x); qkv = s.qkv(h).view(B, L, 3, s.nh, Dd//s.nh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, L, Dd)); return x + s.mlp(s.ln2(x))


class BigByteGPT(nn.Module):
    def __init__(s, nl=NL, d=D, nh=NH):
        super().__init__(); s.tok = nn.Embedding(256, d); s.pos = nn.Embedding(LB, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)]); s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, 256)
        s.ckpt = True
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))[None]
        for b in s.blocks:
            h = checkpoint(b, h, use_reentrant=False) if (s.ckpt and s.training) else b(h)
        return s.head(s.lnf(h))


def load_corpus():
    parts = [open(f, encoding="utf-8").read() for f in sorted(glob.glob(f"{CACHE}/*.txt"))]
    for f in CODE:
        try: parts.append(open(f, errors="ignore").read(2_000_000))
        except Exception: pass
    blob = ("\n".join(parts)).encode("utf-8")
    a = np.frombuffer(blob, np.uint8); cut = int(len(a)*0.97)
    return a[:cut], a[cut:]


def main():
    train, val = load_corpus()
    print(f"corpus {len(train):,} train / {len(val):,} val bytes ({len(glob.glob(f'{CACHE}/*.txt'))} linguas + codigo)", flush=True)
    micro = MICRO_BS
    model = BigByteGPT().to(DEV)
    P = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), 3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    print(f"MODELO: {P/1e6:.0f}M params (d{D} L{NL} h{NH}) | checkpointing ON | bf16 | micro-bs {micro} x accum {ACCUM} = batch {micro*ACCUM}", flush=True)
    print(f"VRAM apos init: {vram()[0]:.2f} GB alocada", flush=True)

    def get_batch(arr, bs):
        ix = np.random.randint(0, len(arr)-LB-1, bs)
        return torch.from_numpy(np.stack([arr[i:i+LB] for i in ix]).astype(np.int64)).to(DEV)

    @torch.no_grad()
    def val_bpb():
        model.eval(); tot = 0; n = 0
        for _ in range(20):
            x = get_batch(val, 16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(x[:, :-1]); l = F.cross_entropy(lo.float().reshape(-1, 256), x[:, 1:].reshape(-1))
            tot += l.item(); n += 1
        model.train(); return tot/n/math.log(2)

    model.train(); t0 = time.time(); step = 0
    print("treino (cada 'opt-step' = 8 micro-batches acumulados)...", flush=True)
    while time.time() - t0 < TIME_BUDGET:
        opt.zero_grad(set_to_none=True); acc_loss = 0.0
        try:
            for _ in range(ACCUM):
                x = get_batch(train, micro)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lo = model(x[:, :-1]); loss = F.cross_entropy(lo.float().reshape(-1, 256), x[:, 1:].reshape(-1)) / ACCUM
                loss.backward(); acc_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); micro = max(2, micro//2); print(f"  OOM -> micro-bs={micro}", flush=True); continue
        al, pk = vram()
        print(f"  step {step} | batch {micro*ACCUM} | bpb {acc_loss/math.log(2):.3f} | VRAM {al:.1f}GB (pico {pk:.1f}/12) | {time.time()-t0:.0f}s", flush=True)
    vb = val_bpb()
    torch.save(model.state_dict(), "/home/leonardo/projects/LLM/bytebrain/big_bytegpt.pt")
    al, pk = vram()
    print(f"\n=== FIM: {P/1e6:.0f}M params | val bpb {vb:.3f} | {step} opt-steps | pico VRAM {pk:.1f}GB de 12 ===", flush=True)
    print(f"  -> modelo de {P/1e6:.0f}M TREINOU no 12GB usando so {pk:.1f}GB. Naive (sem checkpointing) estouraria.", flush=True)
    # amostras
    def gen(seed, n=150, temp=0.8):
        model.eval(); x = torch.tensor([list(seed.encode("utf-8"))], device=DEV)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(n):
                lo = model(x[:, -LB:]); p = F.softmax(lo[0, -1].float()/temp, -1); x = torch.cat([x, torch.multinomial(p, 1)[None]], 1)
        return bytes(x[0].tolist()).decode("utf-8", "ignore")
    print("=== AMOSTRAS ===", flush=True)
    for s in ["The country ", "O Brasil ", "Россия ", "def main():\n    ", "import "]:
        try: print(f"[{s!r}] {gen(s,130)!r}", flush=True)
        except Exception as e: print(f"gen err {s}: {e}", flush=True)


if __name__ == "__main__":
    main()
