"""
ByteBrain — treino limitado do melhor config (L6) + AMOSTRAS geradas (multilingue/codigo).
Versao curta pra fechar a bateria sem o run de 90min. So cache local (sem rede). GPU.
"""
import os, glob, math, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "/home/leonardo/projects/LLM/bytebrain/data/multiscript"
CODE = ["/home/leonardo/projects/LLM/byte-language-lab/data/.oc_corpus.txt",
        "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_big4.txt",
        "/home/leonardo/projects/LLM/byte-language-lab/data/corpus_distill.txt"]
LB = 256


class Block(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        B, L, D = x.shape; h = s.ln1(x); qkv = s.qkv(h).view(B, L, 3, s.nh, D//s.nh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
        x = x + s.proj(a.transpose(1, 2).reshape(B, L, D)); return x + s.mlp(s.ln2(x))


class ByteGPT(nn.Module):
    def __init__(s, nl, d, nh):
        super().__init__(); s.tok = nn.Embedding(256, d); s.pos = nn.Embedding(LB, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)]); s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, 256)
    def forward(s, x):
        h = s.tok(x) + s.pos(torch.arange(x.size(1), device=x.device))[None]
        for b in s.blocks: h = b(h)
        return s.head(s.lnf(h))


def main():
    parts = []
    for f in sorted(glob.glob(f"{CACHE}/*.txt")):
        parts.append(open(f, encoding="utf-8").read())
    for f in CODE:
        try: parts.append(open(f, errors="ignore").read(2_000_000))
        except Exception: pass
    blob = ("\n".join(parts)).encode("utf-8")
    arr = np.frombuffer(blob, np.uint8)
    print(f"corpus {len(arr):,} bytes ({len(glob.glob(f'{CACHE}/*.txt'))} linguas + codigo)", flush=True)

    def batch(bs):
        ix = np.random.randint(0, len(arr)-LB-1, bs)
        return torch.from_numpy(np.stack([arr[i:i+LB] for i in ix]).astype(np.int64))

    model = ByteGPT(6, 384, 6).to(DEV); opt = torch.optim.AdamW(model.parameters(), 3e-4); model.train()
    print(f"{sum(p.numel() for p in model.parameters()):,} params | treino 6000 steps...", flush=True)
    t0 = time.time()
    for st in range(6000):
        x = batch(64).to(DEV); lo = model(x[:, :-1]); loss = F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st % 1000 == 0: print(f"  step {st} {loss.item()/math.log(2):.3f} bpb ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), "/home/leonardo/projects/LLM/bytebrain/best_bytegpt.pt")
    print(f"ckpt salvo ({time.time()-t0:.0f}s)\n=== AMOSTRAS GERADAS ===", flush=True)

    def gen(seed, n=180, temp=0.8):
        model.eval(); ids = list(seed.encode("utf-8")); x = torch.tensor([ids], device=DEV)
        with torch.no_grad():
            for _ in range(n):
                lo = model(x[:, -LB:]); p = F.softmax(lo[0, -1]/temp, -1); nx = torch.multinomial(p, 1).item()
                x = torch.cat([x, torch.tensor([[nx]], device=DEV)], 1)
        return bytes(x[0].tolist()).decode("utf-8", "ignore")
    for seed in ["The country ", "O Brasil é ", "Россия — ", "日本は", "def main():\n    ", "function add(", "import numpy"]:
        print(f"\n[{seed!r}]\n{gen(seed, 170)!r}", flush=True)


if __name__ == "__main__":
    main()
