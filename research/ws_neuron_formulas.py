#!/usr/bin/env python3
"""WS — FÓRMULAS DE NEURÔNIO CALCULÁVEL (família Rosenblatt) p/ a IARA (2026-07-12).

Tese do Leonardo: se dá pra CALCULAR o neurônio (fórmula fechada ou regra local, sem backprop/GPU),
ajuda muito a IARA (leve, instantâneo, observável). Testo 6 clássicos e digo qual vira órgão:
  1. Delta/LMS (Widrow-Hoff 1960) — perceptron suave (usa saída analógica).
  2. READOUT FECHADO (pseudo-inversa/ridge) — calcula o neurônio ótimo em 1 passo (zero iteração).
  3. ELM (Huang 2006: hidden ALEATÓRIO + saída fechada) — cruza o XOR SEM backprop, 1 solve.
  4. Centroide/protótipo (LVQ) — classificador de 1 tiro (média das classes).
  5. Oja (1982) — Hebbian normalizado = acha o eixo-conceito principal (PCA online), local.
  6. Hopfield (1982) — memória associativa por produto externo (1 tiro) = a água-recall calculada.
CPU, instantâneo, honesto. Cada um reporta: funciona? custo? qual órgão da IARA."""
import numpy as np
rng=np.random.default_rng(0)
X4=np.array([[0,0],[1,0],[0,1],[1,1]],float); AND=np.array([0,0,0,1.]); XOR=np.array([0,1,1,0.])
def bias(X): return np.hstack([np.ones((len(X),1)),X])
print("="*66); print("# FÓRMULAS DE NEURÔNIO CALCULÁVEL — quais viram órgão da IARA"); print("="*66)

# 1) DELTA / LMS
def lms(X,y,lr=0.2,ep=200):
    Xb=bias(X); w=np.zeros(Xb.shape[1])
    for _ in range(ep):
        for xb,t in zip(Xb,y): w+=lr*(t-xb@w)*xb
    p=(bias(X)@w>=0.5).astype(float); return w,np.mean(p==y)
w,a=lms(X4,AND); print(f"\n1) DELTA/LMS (Widrow-Hoff) no AND → acc {a:.0%} · minimiza MSE (perceptron suave, aceita ruído)")

# 2) READOUT FECHADO (1 passo, sem iterar)
def closed(X,y,lam=1e-3):
    Xb=bias(X); return np.linalg.solve(Xb.T@Xb+lam*np.eye(Xb.shape[1]),Xb.T@y)
w=closed(X4,AND); p=(bias(X4)@w>=0.5).astype(float)
print(f"2) READOUT FECHADO (ridge/pinv) no AND → acc {np.mean(p==AND):.0%} em UM passo (0 épocas) — o 'calcular o neurônio' literal")

# 3) ELM — hidden aleatório + saída fechada → NÃO-LINEAR sem backprop
def elm(X,y,H=12,seed=0):
    r=np.random.default_rng(seed); Wr=r.standard_normal((X.shape[1],H)); br=r.standard_normal(H)
    Hid=np.tanh(X@Wr+br); Hb=bias(Hid); wout=np.linalg.pinv(Hb)@y
    def f(Z): return (bias(np.tanh(Z@Wr+br))@wout>=0.5).astype(float)
    return f
f=elm(X4,XOR,H=12); acc=np.mean(f(X4)==XOR)
print(f"3) ELM (hidden ALEATÓRIO + solve) no XOR → acc {acc:.0%} — cruza o XOR SEM backprop, em 1 solve. (o perceptron trava em 50%)")

# 4) CENTROIDE / protótipo (1 tiro)
def centroid(X,y):
    c1=X[y==1].mean(0); c0=X[y==0].mean(0)
    return np.array([ (np.linalg.norm(x-c1)<np.linalg.norm(x-c0)) for x in X],float)
p=centroid(X4,AND); print(f"4) CENTROIDE/protótipo no AND → acc {np.mean(p==AND):.0%} · média das classes (1 tiro, interpretável)")

# 5) OJA — acha o eixo-conceito principal (PCA online, local)
D=rng.standard_normal((200,6)); D[:,0]*=4; A=rng.standard_normal((6,6)); Dd=D@A   # direção principal conhecida
w=rng.standard_normal(6); w/=np.linalg.norm(w)
for x in Dd:
    y=w@x; w=w+0.01*y*(x-y*w); w/=np.linalg.norm(w)                # regra de Oja
pc=np.linalg.svd(Dd-Dd.mean(0),full_matrices=False)[2][0]
cos=abs(w@pc/ (np.linalg.norm(w)*np.linalg.norm(pc)))
print(f"5) OJA (Hebbian normalizado) → eixo aprendido bate o PC1 real, |cos|={cos:.2f} (feature-conceito local, sem backprop)")

# 6) HOPFIELD — memória associativa por produto externo (= água-recall calculada)
def hopfield(pats):
    N=pats.shape[1]; W=sum(np.outer(p,p) for p in pats); np.fill_diagonal(W,0); return W/N
def recall(W,s,it=8):
    for _ in range(it): s=np.sign(W@s); s[s==0]=1
    return s
pats=np.where(rng.standard_normal((3,24))>0,1,-1).astype(float); W=hopfield(pats)
ok=0
for p in pats:
    noisy=p.copy(); idx=rng.choice(24,6,replace=False); noisy[idx]*=-1                # 25% de ruído
    ok+= np.array_equal(recall(W,noisy),p)
print(f"6) HOPFIELD (1 tiro, produto externo) → recuperou {ok}/3 padrões de 25% de ruído = a ÁGUA-RECALL calculada")

print("\n## VEREDITO — o que vira órgão da IARA (calcular, não treinar)")
print("  ★ READOUT FECHADO + ELM: roteador/verificador APRENDIDO em 1 solve, e o ELM já cruza NÃO-LINEAR")
print("    sem backprop — a IARA ganha uma camada não-linear barata e instantânea.")
print("  ★ HOPFIELD: formaliza a água-recall como matriz Hebbiana calculada (armazena e recupera em 1 tiro).")
print("  ★ OJA/Sanger: eixo-conceito por regra local → embedding barato pra rotear/comprimir.")
print("  ★ CENTROIDE: protótipo de 1 tiro, o classificador mais leve e legível pra triagem.")
print("  → todos CALCULAM o neurônio (0 ou 1 passo), são LEGÍVEIS e rodam na CPU. É a IARA 'selector' com custo ~0.")
