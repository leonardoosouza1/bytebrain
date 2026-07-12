#!/usr/bin/env python3
"""BATERIA 8 — SURVIVAL-AS-ROUTING (a ponte final + tese IARA-Router, Leonardo 2026-07-08).
Lote 1 provou: inteligência tem que ser IMPORTADA (byte-model), não evoluída. Então a sobrevivência não
'cria' o preditor — ela ROTEIA: cada ser tem 2 órgãos, um BARATO (bigrama, ~grátis) e um CARO/ESPERTO
(ByteBrain, custa energia). Sob custo, quem sobrevive é quem paga o órgão caro SÓ quando compensa.
Pergunta: a pressão de energia descobre uma política de roteamento (via um sinal BARATO de incerteza) que
se aproxima do ORÁCULO e Pareto-vence 'sempre-barato' e 'sempre-caro'? Mede bits+custo, honesto, holdout. GPU."""
import sys, math
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
import torch, torch.nn.functional as F
from wisdom_bridge import load_byte_model, DEV

m = load_byte_model(trained=True)
path = "/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt"
with open(path, "rb") as f:
    f.seek(5_050_000); data = f.read(40000)          # held-out PT (fora do treino da ponte)
N = len(data)

# ---- órgão CARO (ByteBrain): bits/byte + tudo por posição, em janelas de 256 (não-sobrepostas) ----
W = 256
exp_bits = [float("nan")] * N
with torch.no_grad():
    for s in range(0, N - 1, W):
        e = min(s + W, N); L = e - s
        if L < 2: break
        ids = torch.tensor([list(data[s:e])], device=DEV)
        logits = m(inputs_embeds=m.tok(ids))[0]        # [L,256]
        lp = F.log_softmax(logits[:-1].float(), -1)
        tgt = ids[0, 1:]
        b = -(lp[torch.arange(L-1, device=DEV), tgt]) / 0.69314718
        for j in range(L - 1): exp_bits[s + 1 + j] = float(b[j])

# ---- órgão BARATO (bigrama online): bits/byte + ENTROPIA preditiva (o sinal barato de incerteza) ----
ones = [[1.0]*8 for _ in range(256)]; tot = [2.0]*256
che_bits = [float("nan")] * N; che_ent = [float("nan")] * N
for i in range(1, N):
    prev = data[i-1]; b = data[i]; bits = 0.0; ent = 0.0
    for k in range(8):
        p = min(max(ones[prev][k]/tot[prev], 1e-6), 1-1e-6); a = (b >> k) & 1
        bits += -(a*math.log2(p) + (1-a)*math.log2(1-p))
        ent  += -(p*math.log2(p) + (1-p)*math.log2(1-p))       # incerteza do órgão barato AGORA
        ones[prev][k] += a
    tot[prev] += 1.0; che_bits[i] = bits; che_ent[i] = ent

# posições com AMBOS os órgãos disponíveis
idx = [i for i in range(1, N) if not math.isnan(exp_bits[i])]
half = len(idx)//2; train, test = idx[:half], idx[half:]        # aprende τ no train, mede no test (sem trapaça)

def net(strategy, ids_, c):
    """energia = -bits do órgão escolhido - custo por uso do caro. Maior = melhor. strategy(i)->True(caro)."""
    tb = 0.0; use = 0
    for i in ids_:
        caro = strategy(i)
        tb += exp_bits[i] if caro else che_bits[i]
        if caro: tb += c; use += 1
    return -tb/len(ids_), use/len(ids_)                          # (energia média/byte, fração-caro)

print(f"held-out {N}B · {len(idx)} posições com 2 órgãos · barato(bigrama) vs caro(ByteBrain 8M)")
print(f"bits médios: barato {sum(che_bits[i] for i in idx)/len(idx):.3f} · caro {sum(exp_bits[i] for i in idx)/len(idx):.3f}\n")
print(f"{'custo c':>7} {'estratégia':>16} {'energia/byte':>13} {'uso caro%':>10}")
for c in [0.5, 1.0, 2.0, 3.0]:
    # aprende o limiar τ na incerteza barata que MAXIMIZA energia no train
    cand = sorted(set(round(che_ent[i],3) for i in train))
    best_t, best_e = None, -1e9
    for t in cand:
        e,_ = net(lambda i,t=t: che_ent[i] > t, train, c)
        if e > best_e: best_e, best_t = e, t
    strategies = {
        "sempre-barato": lambda i: False,
        "sempre-caro":   lambda i: True,
        "aleatorio50":   lambda i: (i*2654435761 & 0xffff) < 0x8000,
        f"ROTEADOR(τ={best_t:.2f})": lambda i,t=best_t: che_ent[i] > t,
        "ORACULO":       lambda i,c=c: (che_bits[i] - exp_bits[i]) > c,
    }
    print(f"  --- c={c} ---")
    for name, fn in strategies.items():
        e, u = net(fn, test, c)
        print(f"{'':>7} {name:>16} {e:>13.3f} {u*100:>9.1f}%")
print("\n→ se o ROTEADOR (só com incerteza barata) fica entre sempre-barato e sempre-caro E perto do ORÁCULO,")
print("  a sobrevivência-como-roteamento é a ponte: economiza órgão caro sem perder previsão (tese IARA-Router).")
