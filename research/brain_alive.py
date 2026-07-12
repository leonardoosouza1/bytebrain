#!/usr/bin/env python3
"""SISTEMA VIVO — um cérebro-byte que VIVE enquanto aprende PT (Leonardo 2026-07-09).
Chega de pulinhos: um sistema integrado. Um byte-LM aprende português REAL, e a VIDA age nele:
  ERRA (loss alto) -> CORTISOL sobe (estresse) -> reduz a plasticidade (aprende pior sob estresse).
  ACERTA (accuracy) -> DOPAMINA (RPE) -> reforça/consolida o que funciona (motivação).
  SEROTONINA = freio/equilíbrio (calibra, evita superconfiança, acalma o cortisol).
  ADENOSINA = fadiga -> força o SONO -> replay dos casos DIFÍCEIS (consolidação) + reseta fadiga/estresse.
  CURRÍCULO = frases CURTAS nos primeiros dias, crescendo (a lição do ByteBrain).
Testa: o "viver" (hormônios+sono+currículo) bate o treino NORMAL (mesmo nº de passos)? Observabilidade dos
hormônios ao longo da vida + geração. numpy, backprop real, honesto.
"""
import numpy as np, re, time

def sentences(maxb=7_000_000, lo=12, hi=95):
    raw = open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt", "rb").read(maxb).decode("utf-8", "ignore")
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if lo <= len(s.strip()) <= hi]

SENTS = sentences()
def corpus_upto(maxlen):
    txt = "\n".join(s for s in SENTS if len(s) <= maxlen)
    return np.frombuffer(txt.encode("utf-8"), dtype=np.uint8).astype(np.int64)
FULL = corpus_upto(95); SP = int(len(FULL) * 0.9); VAL = FULL[SP:]
LEVELS = {L: corpus_upto(L)[:int(len(corpus_upto(L)) * 0.9)] for L in (30, 45, 60, 80, 95)}

C, D, H, V, B, BASE_LR = 10, 32, 256, 256, 128, 0.5

def init(seed=0):
    r = np.random.default_rng(seed)
    return dict(E=r.normal(0, .05, (V, D)), W1=r.normal(0, .05, (H, C * D)), b1=np.zeros(H),
                W2=r.normal(0, .05, (V, H)), b2=np.zeros(V))

def gb(a, r):
    i = r.integers(C, len(a) - 1, size=B); return np.stack([a[j - C:j] for j in i]), a[i]

def fwd(m, X):
    feat = m['E'][X].reshape(len(X), -1); h = np.tanh(feat @ m['W1'].T + m['b1'])
    z = h @ m['W2'].T + m['b2']; z -= z.max(1, keepdims=True); ez = np.exp(z); p = ez / ez.sum(1, keepdims=True)
    return feat, h, p

def step(m, X, Y, lr, ent=0.0):
    feat, h, p = fwd(m, X)
    loss = -np.log(p[np.arange(len(Y)), Y] + 1e-12).mean(); acc = (p.argmax(1) == Y).mean()
    dz = p.copy(); dz[np.arange(len(Y)), Y] -= 1; dz /= len(Y)
    if ent > 0:
        lp = np.log(p + 1e-12); Hr = -(p * lp).sum(1, keepdims=True); dz -= (ent / len(Y)) * (p * (-lp - Hr))
    gW2 = dz.T @ h; gb2 = dz.sum(0); dh = (dz @ m['W2']) * (1 - h * h)
    gW1 = dh.T @ feat; gb1 = dh.sum(0); dfeat = (dh @ m['W1']).reshape(len(X), C, D)
    m['W2'] -= lr * gW2; m['b2'] -= lr * gb2; m['W1'] -= lr * gW1; m['b1'] -= lr * gb1
    xf = X.reshape(-1); oh = np.zeros((xf.size, V)); oh[np.arange(xf.size), xf] = 1.0
    m['E'] -= lr * (oh.T @ dfeat.reshape(-1, D))
    return loss, acc, h

def bpb(m, a, r, n=2560):
    t = 0.0
    for _ in range(n // B):
        X, Y = gb(a, r); _, _, p = fwd(m, X); t += -np.log2(p[np.arange(B), Y] + 1e-12).mean()
    return t / (n // B)

def live(days=12, steps=500, seed=0, shock=7):
    m = init(seed); r = np.random.default_rng(100 + seed)
    Cort, Sero, base, loss_ema, aden = 0.0, 0.6, 0.5, None, 0.0
    hippo = []; log = []
    lv = [30, 30, 45, 45, 60, 60, 80, 80, 95, 95, 95, 95]
    for day in range(days):
        L = lv[min(day, len(lv) - 1)]; data = LEVELS[L]
        bad_day = (day + 1 == shock)                                             # DIA TRAUMÁTICO (choque/confusão)
        cort_peak = 0.0
        for _ in range(steps):
            X, Y = gb(data, r)
            if bad_day: Y = np.where(r.random(B) < 0.5, r.integers(0, 256, B), Y)  # metade das respostas viram ruído
            feat, h, p = fwd(m, X)
            loss = -np.log(p[np.arange(B), Y] + 1e-12).mean(); acc = (p.argmax(1) == Y).mean()
            loss_ema = loss if loss_ema is None else 0.9 * loss_ema + 0.1 * loss     # referência RÁPIDA (sente choque)
            base = 0.99 * base + 0.01 * acc; dopa = acc - base                       # DOPAMINA (RPE de acerto)
            Cort = float(np.clip(0.9 * Cort + 0.5 * max(0.0, loss / loss_ema - 1.15), 0, 1))  # CORTISOL: surpresa/erro
            cort_peak = max(cort_peak, Cort)                                         # observabilidade: PICO do dia
            Sero = float(np.clip(0.9 * Sero + 0.1 * (1 - Cort), 0.2, 1.0))            # SEROTONINA (freio)
            lr = BASE_LR * (1 - 0.6 * Cort) * (1 + 0.5 * max(0.0, dopa))              # estresse trava, dopamina reforça
            ent = 0.15 * Sero                                                        # freio: evita superconfiança
            step(m, X, Y, lr, ent)
            aden += 1.0 / steps
            if loss > loss_ema * 1.3: hippo.append((X.copy(), Y.copy()))             # HIPOCAMPO: casos difíceis
        for X, Y in hippo[-40:]:                                                     # SONO: replay do difícil
            step(m, X, Y, BASE_LR * 0.5)
        hippo = []; aden = 0.0; Cort *= 0.5; Sero = min(1.0, Sero + 0.1)             # dormiu: reseta fadiga/estresse
        rv = np.random.default_rng(7)
        log.append((day + 1, L, bpb(m, VAL, rv), cort_peak, Sero))                   # loga o PICO de cortisol do dia
    return m, log

def plain(total_steps, seed=0):
    m = init(seed); r = np.random.default_rng(100 + seed)
    for _ in range(total_steps):
        X, Y = gb(FULL[:SP], r); step(m, X, Y, BASE_LR)
    return m, bpb(m, VAL, np.random.default_rng(7))

def generate(m, s, n=80, temp=0.7):
    ctx = list((" " * C + s).encode())[-C:]; out = list(s.encode()); r = np.random.default_rng(1)
    for _ in range(n):
        _, _, p = fwd(m, np.array(ctx)[None]); pr = p[0] ** (1 / temp); pr /= pr.sum()
        nb = int(r.choice(256, p=pr)); out.append(nb); ctx = ctx[1:] + [nb]
        if nb == 10: break
    return bytes(out).decode("utf-8", "ignore")

def main():
    print("=== SISTEMA VIVO — cérebro-byte aprende PT vivendo (erro->cortisol, acerto->dopamina, sono consolida) ===")
    print(f"{len(SENTS)} frases; currículo curto->longo; hormônios acoplados ao aprendizado. bpb_val (aleatório=8).\n")
    t0 = time.time(); m, log = live()
    print("dia  curric(len)  bpb_val  cortisol  serotonina")
    for d, L, bv, cc, ss in log:
        flag = "  <- DIA TRAUMÁTICO (choque)" if d == 7 else ("  <- dormiu, recuperou" if d == 8 else "")
        print(f"{d:>3}  <= {L:>3}       {bv:>6.2f}   {cc:>6.2f}     {ss:>6.2f}{flag}")
    total = 12 * 500 + 12 * 40
    print(f"\ntreino NORMAL (mesmos ~{total} passos, sem hormônios/sono/currículo):")
    _, pb = plain(total)
    print(f"  VIVO bpb {log[-1][2]:.2f}   |   NORMAL bpb {pb:.2f}   (menor=melhor)  · vida {time.time()-t0:.0f}s")
    print("\n=== o que o cérebro VIVO aprendeu a falar ===")
    for s in ["O ", "A cidade ", "Ele "]:
        print(f"  {generate(m, s)!r}")
    print("\nHonesto: observe cortisol subir em dias difíceis e cair no sono; a competência (bpb) melhorar dia a dia.")
    print("Se VIVO <= NORMAL, viver (currículo+sono+equilíbrio) ajuda; se não, o sinal preditivo puro já basta — reporto o que der.")

if __name__ == "__main__":
    main()
