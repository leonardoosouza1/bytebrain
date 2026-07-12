#!/usr/bin/env python3
"""VALIDAÇÃO REAL — cérebro byte-nativo APRENDE PORTUGUÊS (Leonardo 2026-07-09).
Sem enrolação: treina de verdade num corpus REAL de PT-BR (pt_corpus.txt), prevendo o PRÓXIMO BYTE a partir
dos anteriores (contexto). Órgão = córtex byte-nativo (256 vocab): embedding->hidden(tanh)->softmax 256.
Aprende por backprop (SGD). Mostra a curva de bpb (bits/byte) CAINDO e GERA texto. Baselines: aleatório=8.0,
unigrama. Numpy puro, roda em segundos. É a tese-byte (256 valores) aprendendo linguagem, medível.
"""
import numpy as np, time, sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt"
SEEDS = sys.argv[2].split("|") if len(sys.argv) > 2 else ["O Brasil ", "A pessoa ", "Era uma vez "]
raw = open(PATH, "rb").read(4_000_000)
arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
split = int(len(arr) * 0.9)
train, val = arr[:split], arr[split:]

C, D, H, V, B = 10, 32, 256, 256, 128     # contexto 10 bytes, emb 32, hidden 256, vocab 256, batch 128
rng = np.random.default_rng(0)
E  = rng.normal(0, 0.05, (V, D))
W1 = rng.normal(0, 0.05, (H, C * D)); b1 = np.zeros(H)
W2 = rng.normal(0, 0.05, (V, H));     b2 = np.zeros(V)
lr = 0.5

def get_batch(a):
    i = rng.integers(C, len(a) - 1, size=B)
    X = np.stack([a[j - C:j] for j in i])          # (B,C)
    return X, a[i]                                  # (B,C),(B,)

def forward(X):
    feat = E[X].reshape(len(X), -1)                 # (B,C*D)
    h = np.tanh(feat @ W1.T + b1)                   # (B,H)
    z = h @ W2.T + b2                               # (B,V)
    z -= z.max(1, keepdims=True); ez = np.exp(z); p = ez / ez.sum(1, keepdims=True)
    return feat, h, p

def bpb(a, n=4000):
    tot = 0.0
    for _ in range(n // B):
        X, Y = get_batch(a)
        _, _, p = forward(X)
        tot += -np.log2(p[np.arange(len(Y)), Y] + 1e-12).mean()
    return tot / (n // B)

# baseline unigrama (held-out)
cnt = np.bincount(train, minlength=256).astype(np.float64); pu = cnt / cnt.sum()
uni = float(-(pu[val] * 0 + np.log2(pu[val] + 1e-12)).mean())

print("=== CÉREBRO BYTE-NATIVO APRENDE PT-BR (real, backprop) ===")
print(f"corpus {len(arr)/1e6:.1f}MB · contexto {C} bytes · aleatório=8.00 bpb · unigrama={uni:.2f} bpb\n")
print("passo   bpb_treino  bpb_val")
t0 = time.time()
STEPS = 8000
for step in range(STEPS + 1):
    X, Y = get_batch(train)
    feat, h, p = forward(X)
    if step % 800 == 0:
        print(f"{step:>5}   {(-np.log2(p[np.arange(B),Y]+1e-12).mean()):.2f}        {bpb(val,2000):.2f}")
    dz = p; dz[np.arange(B), Y] -= 1; dz /= B                # (B,V)
    dW2 = dz.T @ h; db2 = dz.sum(0)
    dh = (dz @ W2) * (1 - h * h)                             # (B,H)
    dW1 = dh.T @ feat; db1 = dh.sum(0)
    dfeat = (dh @ W1).reshape(B, C, D)                       # (B,C,D)
    W2 -= lr * dW2; b2 -= lr * db2; W1 -= lr * dW1; b1 -= lr * db1
    np.add.at(E, X, -lr * dfeat)

print(f"\ntreinou em {time.time()-t0:.0f}s")

def generate(seed_txt, n=220, temp=0.8):
    ctx = list(seed_txt.encode()[-C:]);  out = list(ctx)
    while len(ctx) < C: ctx = [32] + ctx
    for _ in range(n):
        X = np.array(ctx[-C:])[None]
        _, _, p = forward(X)
        pr = p[0] ** (1.0 / temp); pr /= pr.sum()
        nb = int(rng.choice(256, p=pr)); out.append(nb); ctx.append(nb)
    return bytes(out).decode("utf-8", "ignore")

print("\n=== TEXTO GERADO pelo cérebro (aprendeu sozinho a forma do domínio) ===")
for s in SEEDS:
    print(f"  [{s!r}] -> {generate(s)!r}")
print("\nHonesto: se bpb_val caiu bem abaixo de 8.0 e da unigrama, o cérebro byte-nativo APRENDEU a estrutura do")
print("português (não decorou — held-out). O texto gerado mostra palavras/sílabas de PT emergindo dos bytes.")
