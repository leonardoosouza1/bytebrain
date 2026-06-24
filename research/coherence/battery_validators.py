"""
ByteBrain — VALIDAR OS VALIDADORES antes do overnight. Treina um gerador checkpointando em
estagios e mede se wtrans + modelo-leve REALMENTE acompanham a melhora real (val bpb + amostra).
Se wtrans desce de ~11 -> ~6 monotonico e bate com bpb -> validador confiavel.
Tambem revela se o corpus de 17MB e' bom o suficiente (se wtrans nao desce, dado e' ruim).
"""
import re, math, random, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TXT = re.sub(r"\s+", " ", open("/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt", encoding="utf-8").read())
B = np.frombuffer(TXT.encode("utf-8"), np.uint8); CUT = int(len(B)*0.95); TR, VA = B[:CUT], B[CUT:]
Wd = re.findall(r"[a-zàáâãéêíóôõúüç]+", TXT.lower()); WUNI = Counter(Wd); WBI = Counter(zip(Wd, Wd[1:])); WV = len(WUNI)
LB = 256


def wtrans(t):
    w = re.findall(r"[a-zàáâãéêíóôõúüç]+", t.lower())
    if len(w) < 3: return 12.0
    return float(np.mean([-math.log((WBI.get((w[i], w[i+1]), 0)+0.05)/(WUNI.get(w[i], 0)+0.05*WV)) for i in range(len(w)-1)]))


# ---------- modelo leve de coesao (D) ----------
class D(nn.Module):
    def __init__(s, d=48):
        super().__init__(); s.e = nn.Embedding(256, d); s.c1 = nn.Conv1d(d, 96, 3, padding=1); s.c2 = nn.Conv1d(96, 96, 3, padding=2, dilation=2); s.c3 = nn.Conv1d(96, 96, 3, padding=4, dilation=4); s.h = nn.Linear(96, 1)
    def forward(s, x):
        h = s.e(x).transpose(1, 2); h = F.gelu(s.c1(h)); h = F.gelu(s.c2(h)); h = F.gelu(s.c3(h)); return s.h(h.amax(-1)).squeeze(-1)


def enc(t, L=320):
    b = t.encode("utf-8")[:L]; a = np.zeros(L, np.int64); a[:len(b)] = list(b); return a


def train_D():
    def deg(s):
        k = random.choice(["cs", "ws", "rnd", "salad", "rep"]); w = s.split()
        if k == "cs": return "".join(random.sample(list(s), len(s)))
        if k == "ws" and len(w) > 1: return " ".join(random.sample(w, len(w)))
        if k == "rnd": return "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in s)
        if k == "rep": return ((random.choice(Wd)+" ")*30)[:len(s)]
        return " ".join(random.choice(Wd) for _ in range(max(2, len(w))))
    X, Y = [], []
    for _ in range(4000):
        ln = random.randint(15, 300); i = random.randint(0, len(TXT)-ln-1); c = TXT[i:i+ln]
        X.append(enc(c)); Y.append(1.0); X.append(enc(deg(c))); Y.append(0.0)
    X = torch.tensor(np.stack(X)); Y = torch.tensor(Y)
    d = D().to(DEV); o = torch.optim.Adam(d.parameters(), 2e-3); d.train()
    for ep in range(12):
        pm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = pm[i:i+64]; o.zero_grad(); F.binary_cross_entropy_with_logits(d(X[idx].to(DEV)), Y[idx].to(DEV)).backward(); o.step()
    d.eval(); return d


# ---------- gerador ----------
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


def gen(g, n=110):
    x = torch.tensor([list("o ".encode())], device=DEV)
    with torch.no_grad():
        for _ in range(n):
            lo = g(x[:, -LB:]); x = torch.cat([x, torch.multinomial(F.softmax(lo[0, -1]/0.8, -1), 1)[None]], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def main():
    print("treinando modelo-leve de coesao (D)...", flush=True); d = train_D()
    def dscore(t):
        with torch.no_grad(): return torch.sigmoid(d(torch.tensor([enc(t)]).to(DEV))).item()*100
    g = GPT().to(DEV); P = sum(p.numel() for p in g.parameters()); opt = torch.optim.AdamW(g.parameters(), 3e-4)
    print(f"gerador {P/1e6:.0f}M | checkpoints medindo val_bpb + wtrans + modelo-leve\n", flush=True)
    def val_bpb():
        g.eval(); t = 0; m = 0
        with torch.no_grad():
            for _ in range(15):
                x = batch(VA, 32); lo = g(x[:, :-1]); t += F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1)).item(); m += 1
        g.train(); return t/m/math.log(2)
    def measure():
        smp = [gen(g, 110) for _ in range(8)]
        return round(val_bpb(), 2), round(float(np.mean([wtrans(s) for s in smp])), 2), round(float(np.mean([dscore(s) for s in smp])), 1), smp[0]
    cks = [0, 200, 500, 1000, 2000, 4000, 7000]; rows = []
    print(f"{'step':>6}{'val_bpb':>9}{'wtrans':>8}{'coesaoD%':>10}   amostra")
    step = 0; t0 = time.time()
    for target in cks:
        while step < target:
            x = batch(TR, 64); lo = g(x[:, :-1]); F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); opt.step(); opt.zero_grad(); step += 1
        bpb, wt, ds, s = measure(); rows.append((step, bpb, wt, ds))
        print(f"{step:>6}{bpb:>9}{wt:>8}{ds:>10}   {s[:70]!r}", flush=True)
    # analise
    wts = [r[2] for r in rows]; bpbs = [r[1] for r in rows]
    mono = all(wts[i] >= wts[i+1]-0.3 for i in range(len(wts)-1))   # wtrans desce (com folga)?
    corr = float(np.corrcoef(bpbs, wts)[0, 1])
    print(f"\n=== ANALISE ===")
    print(f"  wtrans: {wts[0]} -> {wts[-1]} (alvo PT real ~6.0)")
    print(f"  val_bpb: {bpbs[0]} -> {bpbs[-1]}")
    print(f"  wtrans desce com treino (monotonico)? {'SIM' if mono else 'nao'}")
    print(f"  correlacao wtrans x val_bpb: {corr:.2f} (perto de +1 = wtrans acompanha a melhora real)")
    verdict = "VALIDADORES CONFIAVEIS (acompanham o treino)" if (corr > 0.7 and wts[-1] < wts[0]-1) else \
              ("dado/modelo nao melhora coesao o bastante (wtrans preso alto)" if wts[-1] > 8.5 else "sinal fraco")
    print(f"  >>> {verdict} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
