"""
ByteBrain — CO-TREINO adversarial (ideia #3 do Leonardo): detector D <-> gerador G se ajudam.
D (classificador de coesao) pune G; G (byte-LM) aprende a gerar coeso pra enganar D; D fica esperto
vendo as saidas reais de G. Texto e' discreto -> uso Gumbel-softmax (rollout soft diferenciavel)
pro gradiente de D fluir pra G. TESTE HONESTO: a coesao de G (medida por wtrans INDEPENDENTE)
melhora vs LM puro? Compara G_adversarial vs G_baseline (mesmo start, mesmos updates).
"""
import re, math, random, time
from collections import Counter
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TXT = re.sub(r"\s+", " ", open("/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt", encoding="utf-8").read())
B = np.frombuffer(TXT.encode("utf-8"), np.uint8)
Wd = re.findall(r"[a-zàáâãéêíóôõúüç]+", TXT.lower()); WUNI = Counter(Wd); WBI = Counter(zip(Wd, Wd[1:])); WV = len(WUNI)
ROLL = 48


def wtrans(t):  # juiz INDEPENDENTE (nao e' o D)
    w = re.findall(r"[a-zàáâãéêíóôõúüç]+", t.lower())
    if len(w) < 3: return 12.0
    return float(np.mean([-math.log((WBI.get((w[i], w[i+1]), 0)+0.05)/(WUNI.get(w[i], 0)+0.05*WV)) for i in range(len(w)-1)]))


class G(nn.Module):  # gerador byte-LM (GRU, aceita entrada SOFT)
    def __init__(s, d=192):
        super().__init__(); s.emb = nn.Embedding(256, d); s.gru = nn.GRU(d, d, 2, batch_first=True); s.out = nn.Linear(d, 256)
    def forward(s, x, h=None): y, h = s.gru(s.emb(x), h); return s.out(y), h
    def soft_step(s, soft, h):  # soft:[B,256] -> emb soft
        e = (soft @ s.emb.weight).unsqueeze(1); y, h = s.gru(e, h); return s.out(y[:, -1]), h


class D(nn.Module):  # detector de coesao (byte-CNN), aceita entrada SOFT
    def __init__(s, d=48):
        super().__init__(); s.emb = nn.Embedding(256, d)
        s.c1 = nn.Conv1d(d, 96, 3, padding=1); s.c2 = nn.Conv1d(96, 96, 3, padding=2, dilation=2); s.c3 = nn.Conv1d(96, 96, 3, padding=4, dilation=4); s.h = nn.Linear(96, 1)
    def _body(s, h):
        h = h.transpose(1, 2); h = F.gelu(s.c1(h)); h = F.gelu(s.c2(h)); h = F.gelu(s.c3(h)); return s.h(h.amax(-1)).squeeze(-1)
    def forward(s, x): return s._body(s.emb(x))
    def forward_soft(s, soft): return s._body(soft @ s.emb.weight)


def batch(bs, L=ROLL):
    ix = np.random.randint(0, len(B)-L-1, bs); return torch.from_numpy(np.stack([B[i:i+L] for i in ix]).astype(np.int64)).to(DEV)


def degrade_batch(bs, L=ROLL):
    out = []
    for _ in range(bs):
        i = random.randint(0, len(TXT)-L-1); s = TXT[i:i+L]; k = random.choice(["cs", "ws", "rnd", "salad"])
        if k == "cs": s = "".join(random.sample(list(s), len(s)))
        elif k == "ws": s = " ".join(random.sample(s.split(), len(s.split()))) if len(s.split()) > 1 else s
        elif k == "rnd": s = "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in s)
        else: s = " ".join(random.choice(Wd) for _ in range(len(s.split()) or 6))
        b = s.encode("utf-8")[:L]; a = np.zeros(L, np.int64); a[:len(b)] = list(b); out.append(a)
    return torch.from_numpy(np.stack(out)).to(DEV)


def soft_rollout(g, bs, temp):  # gera sequencia SOFT diferenciavel
    soft = F.one_hot(torch.full((bs,), 32, device=DEV), 256).float(); h = None; seq = []
    for _ in range(ROLL):
        logits, h = g.soft_step(soft, h); soft = F.gumbel_softmax(logits, tau=temp, hard=False); seq.append(soft)
    return torch.stack(seq, 1)  # [B,ROLL,256]


def hard_sample(g, bs, temp=0.8):
    x = torch.full((bs, 1), 32, device=DEV); h = None
    with torch.no_grad():
        for _ in range(ROLL):
            lo, h = g.forward(x[:, -1:], h); nx = torch.multinomial(F.softmax(lo[:, -1]/temp, -1), 1); x = torch.cat([x, nx], 1)
    return x[:, 1:]


def gen_text(g, seed="o ", n=120):
    x = torch.tensor([list(seed.encode())], device=DEV); h = None
    with torch.no_grad():
        for _ in range(n):
            lo, h = g.forward(x[:, -1:], h) if h is not None else g.forward(x, None)
            nx = torch.multinomial(F.softmax(lo[:, -1]/0.8, -1), 1); x = torch.cat([x, nx], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


def pretrain(g, steps=600):
    o = torch.optim.Adam(g.parameters(), 2e-3); g.train()
    for _ in range(steps):
        x = batch(48, 96); lo, _ = g(x[:, :-1]); F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1)).backward(); o.step(); o.zero_grad()
    return g


def lm_step(g, o):
    x = batch(48, 96); lo, _ = g(x[:, :-1]); l = F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1))
    o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); o.step()


def measure(g):
    s = [gen_text(g, "o ", 110) for _ in range(8)]; return float(np.mean([wtrans(t) for t in s])), s[0]


def main():
    print(f"device {DEV} | wtrans alvo (PT real) ~6.0; pior=alto", flush=True)
    g0 = pretrain(G().to(DEV)); torch.save(g0.state_dict(), "/tmp/g0.pt")
    base_w, _ = measure(g0); print(f"pos-pretrain: wtrans {base_w:.2f}", flush=True)

    # ---- BASELINE: G puro-LM ----
    gb = G().to(DEV); gb.load_state_dict(torch.load("/tmp/g0.pt")); ob = torch.optim.Adam(gb.parameters(), 1e-3)
    for r in range(120): lm_step(gb, ob)
    bw, bsmp = measure(gb)

    # ---- ADVERSARIAL: G + D se ajudando ----
    ga = G().to(DEV); ga.load_state_dict(torch.load("/tmp/g0.pt")); oa = torch.optim.Adam(ga.parameters(), 1e-3)
    d = D().to(DEV); od = torch.optim.Adam(d.parameters(), 2e-3)
    LAM = 0.3; t0 = time.time(); traj = []
    for r in range(120):
        temp = max(0.5, 1.0 - r/200)
        # treina D: real(1) vs G-samples(0) vs degradacao(0)
        for _ in range(2):
            real = batch(32); fake = hard_sample(ga, 32); deg = degrade_batch(32)
            xr = d(real); xf = d(fake); xg = d(deg)
            ld = F.binary_cross_entropy_with_logits(xr, torch.ones_like(xr)) + \
                 0.5*F.binary_cross_entropy_with_logits(xf, torch.zeros_like(xf)) + \
                 0.5*F.binary_cross_entropy_with_logits(xg, torch.zeros_like(xg))
            od.zero_grad(); ld.backward(); od.step()
        # treina G: LM + adversarial (faz D dizer coeso=1)
        lm_step(ga, oa)
        soft = soft_rollout(ga, 24, temp); score = d.forward_soft(soft)
        ladv = LAM * F.binary_cross_entropy_with_logits(score, torch.ones_like(score))
        oa.zero_grad(); ladv.backward(); torch.nn.utils.clip_grad_norm_(ga.parameters(), 1.0); oa.step()
        if r % 20 == 0:
            w, _ = measure(ga); traj.append(round(w, 2))
            print(f"  round {r} | wtrans G {w:.2f} | D(real) {torch.sigmoid(xr.mean()).item():.2f} D(fake) {torch.sigmoid(xf.mean()).item():.2f} | {time.time()-t0:.0f}s", flush=True)
    aw, asmp = measure(ga)

    print("\n=== RESULTADO (wtrans menor = mais coeso; PT real ~6.0) ===")
    print(f"  pos-pretrain:        {base_w:.2f}")
    print(f"  BASELINE (LM puro):  {bw:.2f}   amostra: {bsmp[:90]!r}")
    print(f"  ADVERSARIAL (D<->G): {aw:.2f}   amostra: {asmp[:90]!r}")
    print(f"  trajetoria adv: {traj}")
    print(f"\n  -> adversarial ajudou? {'SIM' if aw < bw - 0.15 else ('empate' if abs(aw-bw) <= 0.15 else 'NAO (LM puro melhor)')}")


if __name__ == "__main__":
    main()
