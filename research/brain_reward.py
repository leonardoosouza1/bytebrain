#!/usr/bin/env python3
"""DOPAMINA NO ACERTO (recompensa) — ideia do Leonardo, 2026-07-09.
"Se ele acerta, a gente dá dopamina; se acerta mais, ganha mais." = aprendizado modulado por RECOMPENSA
(operante), o jeito real do cérebro — SEM professor/backprop, só um sinal global de dopamina no acerto.
Testa em frases curtas de PT, prevendo próximo byte:
  (1) BACKPROP supervisionado (teto — sabe a resposta certa)
  (2) RECOMPENSA (REINFORCE): o cérebro CHUTA um byte; acertou -> dopamina reforça aquele chute; errou -> nada.
      NÃO recebe a resposta certa, só "acertou/errou". É a dopamina-no-acerto pura.
  (3) HIPERESTÍMULO: acerto vale MAIS (recompensa maior) -> reforço mais forte.
Mede bpb held-out E acurácia (fração que acerta) ao longo do treino. Honesto: reward aprende SEM professor?
numpy puro, embedding vetorizado.
"""
import numpy as np, re, time

def build_short_corpus(maxb=6_000_000, lo=15, hi=90):
    raw = open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt", "rb").read(maxb).decode("utf-8", "ignore")
    short = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if lo <= len(s.strip()) <= hi]
    return np.frombuffer(("\n".join(short)).encode("utf-8"), dtype=np.uint8).astype(np.int64)

ARR = build_short_corpus(); SP = int(len(ARR) * 0.9); TRAIN, VAL = ARR[:SP], ARR[SP:]
C, D, H, V, B, LR = 10, 32, 256, 256, 128, 0.5

def init(seed=0):
    r = np.random.default_rng(seed)
    return dict(E=r.normal(0, .05, (V, D)), W1=r.normal(0, .05, (H, C * D)), b1=np.zeros(H),
                W2=r.normal(0, .05, (V, H)), b2=np.zeros(V))

def batch(a, r):
    i = r.integers(C, len(a) - 1, size=B)
    return np.stack([a[j - C:j] for j in i]), a[i]

def fwd(m, X):
    feat = m['E'][X].reshape(len(X), -1); h = np.tanh(feat @ m['W1'].T + m['b1'])
    z = h @ m['W2'].T + m['b2']; z -= z.max(1, keepdims=True); ez = np.exp(z); p = ez / ez.sum(1, keepdims=True)
    return feat, h, p

def apply(m, X, feat, h, dz):
    gW2 = dz.T @ h; gb2 = dz.sum(0); dh = (dz @ m['W2']) * (1 - h * h)
    gW1 = dh.T @ feat; gb1 = dh.sum(0); dfeat = (dh @ m['W1']).reshape(len(X), C, D)
    m['W2'] -= LR * gW2; m['b2'] -= LR * gb2; m['W1'] -= LR * gW1; m['b1'] -= LR * gb1
    xf = X.reshape(-1); oh = np.zeros((xf.size, V)); oh[np.arange(xf.size), xf] = 1.0
    m['E'] -= LR * (oh.T @ dfeat.reshape(-1, D))

def evalu(m, a, r, n=2560):
    tb = 0.0; acc = 0.0; k = n // B
    for _ in range(k):
        X, Y = batch(a, r); _, _, p = fwd(m, X)
        tb += -np.log2(p[np.arange(B), Y] + 1e-12).mean(); acc += (p.argmax(1) == Y).mean()
    return tb / k, acc / k

def train(mode, steps=6000, seed=0, hyper=1.0, beta=0.0):
    m = init(seed); r = np.random.default_rng(100 + seed); base = 0.0; log = []
    for step in range(steps):
        X, Y = batch(TRAIN, r); feat, h, p = fwd(m, X)
        if mode == 'backprop':
            dz = p.copy(); dz[np.arange(B), Y] -= 1; dz /= B          # sabe a resposta (professor)
        else:
            cdf = np.cumsum(p, 1); u = r.random((B, 1)); bpk = (u < cdf).argmax(1)   # CHUTA um byte
            correct = (bpk == Y).astype(np.float64)
            rew = correct * hyper                                     # DOPAMINA: só no acerto (hiper = mais forte)
            base = 0.99 * base + 0.01 * rew.mean()                    # linha-de-base (reduz variância)
            adv = (rew - base)[:, None]
            ohb = np.zeros((B, V)); ohb[np.arange(B), bpk] = 1.0
            dz = -(adv * (ohb - p)) / B                               # REINFORCE: reforça o chute na medida da dopamina
            if beta > 0:                                              # bônus de ENTROPIA: não deixar ficar superconfiante
                logp = np.log(p + 1e-12); Hrow = -(p * logp).sum(1, keepdims=True)
                dz -= (beta / B) * (p * (-logp - Hrow))
        apply(m, X, feat, h, dz)
        if step % 600 == 0:
            rv = np.random.default_rng(7); bv, ac = evalu(m, VAL, rv)
            log.append((step, bv, ac))
    return m, log

def show(name, log):
    print(f"  {name:<24} " + "  ".join(f"{ac*100:>4.0f}%" for _, _, ac in log[::2]))
    return log[-1]

def main():
    print("=== DOPAMINA NO ACERTO — aprender por RECOMPENSA vs backprop (frases curtas PT) ===")
    print(f"corpus {len(ARR)/1e6:.1f}MB. mede ACURÁCIA (acerta próximo byte) ao longo do treino. aleatório~0.4%.\n")
    print("  método                   acurácia por etapa (0 -> fim):")
    r1 = show("(1) BACKPROP (professor)", train('backprop')[1])
    r2 = show("(2) RECOMPENSA no acerto", train('reward', hyper=1.0)[1])
    r3 = show("(3) HIPERESTÍMULO (x3)",   train('reward', hyper=3.0)[1])
    print(f"\n  FINAL:  backprop acc {r1[2]*100:.0f}% bpb {r1[1]:.2f} | recompensa acc {r2[2]*100:.0f}% bpb {r2[1]:.2f} | hiper acc {r3[2]*100:.0f}% bpb {r3[1]:.2f}")
    print("\nHonesto: se a RECOMPENSA (só 'acertou/errou', sem professor) sobe acima do aleatório, a dopamina-no-acerto")
    print("ENSINA linguagem sozinha (a sua ideia). Backprop é o teto (sabe a resposta). Hiperestímulo mostra se recompensa maior acelera.")

if __name__ == "__main__":
    main()
