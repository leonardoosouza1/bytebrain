"""Teste decisivo: treina gerador no corpus LIMPO e ve se o wtrans do gerador DESCE rumo a ~7."""
import re, math, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
DEV = "cuda"
TXT = open("/home/leonardo/projects/LLM/bytebrain/data/pt_clean.txt", encoding="utf-8").read()
B = np.frombuffer(TXT.encode("utf-8"), np.uint8); CUT = int(len(B)*0.95); TR, VA = B[:CUT], B[CUT:]
Wd = re.findall(r"[a-zàáâãéêíóôõúüç]+", TXT.lower()); WUNI = Counter(Wd); WBI = Counter(zip(Wd, Wd[1:])); WV = len(WUNI)
LB = 256


def wtrans(t):
    w = re.findall(r"[a-zàáâãéêíóôõúüç]+", t.lower())
    if len(w) < 3: return 12.0
    return float(np.mean([-math.log((WBI.get((w[i], w[i+1]), 0)+0.05)/(WUNI.get(w[i], 0)+0.05*WV)) for i in range(len(w)-1)]))


class Blk(nn.Module):
    def __init__(s, d, nh):
        super().__init__(); s.nh = nh; s.l1 = nn.LayerNorm(d); s.l2 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3*d); s.pr = nn.Linear(d, d); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x):
        Bn, L, Dd = x.shape; h = s.l1(x); q = s.qkv(h).view(Bn, L, 3, s.nh, Dd//s.nh).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q[0], q[1], q[2], is_causal=True); x = x + s.pr(a.transpose(1, 2).reshape(Bn, L, Dd)); return x + s.mlp(s.l2(x))


class GPT(nn.Module):
    def __init__(s, d=320, nl=6, nh=8):
        super().__init__(); s.t = nn.Embedding(256, d); s.p = nn.Embedding(LB, d); s.b = nn.ModuleList([Blk(d, nh) for _ in range(nl)]); s.f = nn.LayerNorm(d); s.o = nn.Linear(d, 256)
    def forward(s, x):
        h = s.t(x)+s.p(torch.arange(x.size(1), device=x.device))[None]
        for b in s.b: h = b(h)
        return s.o(s.f(h))


def batch(a, bs, L=128):
    ix = np.random.randint(0, len(a)-L-1, bs); return torch.from_numpy(np.stack([a[i:i+L] for i in ix]).astype(np.int64)).to(DEV)


def gen(g, n=120):
    x = torch.tensor([list("O ".encode())], device=DEV)
    with torch.no_grad():
        for _ in range(n):
            lo = g(x[:, -LB:]); x = torch.cat([x, torch.multinomial(F.softmax(lo[0, -1]/0.8, -1), 1)[None]], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def main():
    print(f"corpus LIMPO {len(B)/1e6:.2f}MB | alvo wtrans ~6-7 (PT real); sujo ficava preso ~11", flush=True)
    g = GPT().to(DEV); P = sum(p.numel() for p in g.parameters()); opt = torch.optim.AdamW(g.parameters(), 3e-4)
    def vbpb():
        g.eval(); t = 0
        with torch.no_grad():
            for _ in range(12):
                x = batch(VA, 32); t += F.cross_entropy(g(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1)).item()
        g.train(); return t/12/math.log(2)
    print(f"gerador {P/1e6:.0f}M\n{'step':>6}{'val_bpb':>9}{'wtrans':>8}   amostra", flush=True)
    cks = [0, 500, 1000, 2000, 3500]; step = 0; t0 = time.time(); wts = []
    for tg in cks:
        while step < tg:
            x = batch(TR, 64); F.cross_entropy(g(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); opt.step(); opt.zero_grad(); step += 1
        smp = [gen(g, 110) for _ in range(8)]; wt = float(np.mean([wtrans(s) for s in smp])); wts.append(round(wt, 2))
        print(f"{step:>6}{round(vbpb(),2):>9}{wt:>8.2f}   {smp[0][:68]!r}", flush=True)
    print(f"\n=== wtrans: {wts[0]} -> {wts[-1]} | {'DESTRAVOU rumo a PT real ✓' if wts[-1] < 8 else 'ainda alto'} | {time.time()-t0:.0f}s ===")


if __name__ == "__main__":
    main()
