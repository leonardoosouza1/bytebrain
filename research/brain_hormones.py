#!/usr/bin/env python3
"""BATERIA PESADA — HORMÔNIOS + OBSERVABILIDADE no cérebro byte (Leonardo 2026-07-09).
Junta: (1) ByteBrain lição = validar FRASES CURTAS (não jogar Wikipedia longa de uma vez);
(2) hormônios modulando o APRENDIZADO — dopamina (aprende + da SURPRESA), cortisol (estresse REDUZ plasticidade),
acetilcolina (FOCO: só aprende o saliente); (3) OBSERVABILIDADE/TRACE: surpresa por byte, curva, lr efetivo.
Pergunta: hormônio AJUDA ou PIORA o aprendizado? Varre dopamina/cortisol pra cima e pra baixo e MEDE (held-out).
Córtex byte-nativo (emb->hidden->softmax256), backprop. numpy puro, honesto.
"""
import numpy as np, time, re

def build_short_corpus(maxb=9_000_000, lo=15, hi=90):
    raw = open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt", "rb").read(maxb).decode("utf-8", "ignore")
    sents = re.split(r'(?<=[.!?])\s+', raw)
    short = [s.strip() for s in sents if lo <= len(s.strip()) <= hi]
    txt = "\n".join(short)
    return np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64), len(short)

ARR, NSENT = build_short_corpus()
SP = int(len(ARR) * 0.9); TRAIN, VAL = ARR[:SP], ARR[SP:]
C, D, H, V, B, BASE_LR = 10, 32, 256, 256, 128, 0.5

def init(seed=0):
    r = np.random.default_rng(seed)
    return dict(E=r.normal(0, .05, (V, D)), W1=r.normal(0, .05, (H, C * D)), b1=np.zeros(H),
                W2=r.normal(0, .05, (V, H)), b2=np.zeros(V))

def get_batch(a, r):
    i = r.integers(C, len(a) - 1, size=B)
    return np.stack([a[j - C:j] for j in i]), a[i]

def fwd(m, X):
    feat = m['E'][X].reshape(len(X), -1)
    h = np.tanh(feat @ m['W1'].T + m['b1'])
    z = h @ m['W2'].T + m['b2']; z -= z.max(1, keepdims=True); ez = np.exp(z); p = ez / ez.sum(1, keepdims=True)
    return feat, h, p

def bpb(m, a, r, n=2560):
    t = 0.0
    for _ in range(n // B):
        X, Y = get_batch(a, r); _, _, p = fwd(m, X)
        t += -np.log2(p[np.arange(B), Y] + 1e-12).mean()
    return t / (n // B)

def train(dopa=0.0, cort=0.0, ach=0.0, steps=4000, seed=0, trace_lr=False):
    m = init(seed); r = np.random.default_rng(100 + seed); s_ema = None; lr_sum = 0.0; used = 0
    for step in range(steps):
        X, Y = get_batch(TRAIN, r); feat, h, p = fwd(m, X)
        loss = -np.log(p[np.arange(B), Y] + 1e-12).mean()                 # surpresa média (nats)
        s_ema = loss if s_ema is None else 0.98 * s_ema + 0.02 * loss
        sal = loss / (s_ema + 1e-9)                                        # >1 = surpreendente
        if ach > 0 and sal < (1 - 0.5 * ach):                             # ACh: foco -> ignora o banal
            continue
        lr = BASE_LR * (1 + dopa * np.clip(sal - 1, -0.7, 3.0)) * (1 - cort)   # dopamina×surpresa, cortisol reduz
        lr = max(lr, 0.0); lr_sum += lr; used += 1
        dz = p; dz[np.arange(B), Y] -= 1; dz /= B
        gW2 = dz.T @ h; gb2 = dz.sum(0); dh = (dz @ m['W2']) * (1 - h * h)
        gW1 = dh.T @ feat; gb1 = dh.sum(0); dfeat = (dh @ m['W1']).reshape(B, C, D)
        wd = 1 - 1e-4 * cort                                              # cortisol: leve decaimento (conservador)
        m['W2'] = m['W2'] * wd - lr * gW2; m['b2'] -= lr * gb2
        m['W1'] = m['W1'] * wd - lr * gW1; m['b1'] -= lr * gb1
        xf = X.reshape(-1)                                            # scatter vetorizado (rápido) do grad do embedding
        oh = np.zeros((xf.size, V)); oh[np.arange(xf.size), xf] = 1.0
        m['E'] -= lr * (oh.T @ dfeat.reshape(-1, D))
    rv = np.random.default_rng(7)
    return m, bpb(m, VAL, rv), (lr_sum / max(used, 1)), used / steps

def surprise_trace(m, sent):
    b = list(sent.encode("utf-8")); sur = []
    for i in range(C, len(b)):
        _, _, p = fwd(m, np.array(b[i - C:i])[None]); sur.append(-np.log2(p[0, b[i]] + 1e-12))
    sur = np.array(sur); chars = sent[C:]
    hard = np.argsort(-sur)[:3]
    marked = "".join((f"[{c}]" if j in hard else c) for j, c in enumerate(chars))
    return float(sur.mean()), marked

def generate(m, seed_txt, n=90, temp=0.7):
    ctx = list((" " * C + seed_txt).encode())[-C:]; out = list(seed_txt.encode())
    r = np.random.default_rng(1)
    for _ in range(n):
        _, _, p = fwd(m, np.array(ctx)[None]); pr = p[0] ** (1 / temp); pr /= pr.sum()
        nb = int(r.choice(256, p=pr)); out.append(nb); ctx = ctx[1:] + [nb]
        if nb == 10: break
    return bytes(out).decode("utf-8", "ignore")

def main():
    print(f"=== BATERIA HORMÔNIOS + OBSERVABILIDADE — cérebro byte em FRASES CURTAS de PT ===")
    print(f"corpus: {NSENT} frases curtas ({len(ARR)/1e6:.1f}MB, 15-90 chars). held-out bpb; aleatório=8.0.\n")
    r0 = np.random.default_rng(7)
    print(">>> DOPAMINA (aprender + da surpresa) — ajuda?  [cortisol=0]")
    print(f"{'dopamina':>10} {'bpb_val':>9} {'lr_médio':>9}")
    best = None
    for dp in [0.0, 1.0, 2.0, 4.0]:
        m, bv, lr, _ = train(dopa=dp, cort=0.0)
        print(f"{dp:>10.1f} {bv:>9.2f} {lr:>9.3f}")
        if best is None or bv < best[1]: best = (m, bv, f"dopa={dp}")
    print("\n>>> CORTISOL (estresse reduz plasticidade) — piora?  [dopamina=1]")
    print(f"{'cortisol':>10} {'bpb_val':>9} {'lr_médio':>9}")
    for ct in [0.0, 0.3, 0.6, 0.85]:
        m, bv, lr, _ = train(dopa=1.0, cort=ct)
        print(f"{ct:>10.2f} {bv:>9.2f} {lr:>9.3f}")
        if bv < best[1]: best = (m, bv, f"cort={ct}")
    print("\n>>> ACETILCOLINA (foco: só aprende o saliente) — fração de updates usados")
    print(f"{'ach':>10} {'bpb_val':>9} {'updates%':>9}")
    for ac in [0.0, 0.5, 0.9]:
        m, bv, lr, frac = train(dopa=1.0, cort=0.0, ach=ac)
        print(f"{ac:>10.1f} {bv:>9.2f} {frac*100:>8.0f}%")
        if bv < best[1]: best = (m, bv, f"ach={ac}")
    m = best[0]
    print(f"\n=== OBSERVABILIDADE (melhor: {best[2]}, bpb {best[1]:.2f}) — TRACE de surpresa por byte ===")
    print("[..] marca os 3 bytes mais SURPREENDENTES (onde o cérebro 'hesita'):")
    for s in ["O Brasil é um país.", "A água ferve a cem graus.", "Ele foi para a escola ontem."]:
        mean, marked = surprise_trace(m, s)
        print(f"  surpresa média {mean:.2f} bpb | {marked}")
    print("\n=== GERAÇÃO de frases curtas (o que ele aprendeu a 'falar') ===")
    for s in ["O ", "A ", "Em ", "Ele "]:
        print(f"  {generate(m, s)!r}")
    print("\nHonesto: compare as colunas. Dopamina ajuda se bpb cai com dp>0; cortisol PIORA se bpb sobe com ct>0;")
    print("ACh economiza updates (foco) sem perder muito. O trace mostra ONDE o cérebro erra (início de palavra, raro).")

if __name__ == "__main__":
    main()
