#!/usr/bin/env python3
"""PONTE texto<->binário: passa DOIS byte-models (texto-PT + binário) sobre um binário real, janela por
janela, e ROTULA cada trecho por quem prevê melhor: CÓDIGO (modelo-bin ganha), TEXTO (modelo-txt ganha,
= strings embutidas), DADOS (ambos perdem = tabela/comprimido). Depois EXTRAI as ilhas de texto — o que o
binário 'diz'. É assim que se 'lê' um jogo/DLL: estrutura + conteúdo legível. CPU."""
import sys, os, math, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cpu"
def load(p):
    ck = torch.load(p, map_location=DEV, weights_only=False); c = ck["config"]
    m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
    m.load_state_dict(ck["model"]); return m
@torch.no_grad()
def bpb(m, ids):
    if len(ids) < 2: return 9.0
    x = torch.tensor([ids[:-1]]); y = torch.tensor([ids[1:]])
    return F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1)).item() / math.log(2)

TXT = load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt")
BIN = load(sys.argv[2] if len(sys.argv) > 2 else "/home/leonardo/projects/LLM/bytebrain/research/ckpt_analyzer/ckpt_best.pt")

# alvo: um binário real rico em strings (um jogo/DLL seria idêntico; PE do Windows exigiria modelo treinado em PE)
target = sys.argv[1] if len(sys.argv) > 1 else None
if not target:
    for p in ["/usr/bin/python3.12", "/usr/bin/python3", "/bin/bash", "/usr/bin/gcc"]:
        if os.path.exists(p): target = p; break
data = open(target, "rb").read(300000)   # lê fundo o suficiente p/ pegar .rodata (strings)
print(f"lendo: {target} ({len(data)} bytes)\n", flush=True)

W = 64
labels = []; text_islands = []
for i in range(0, len(data) - W, W):
    ids = np.frombuffer(data[i:i+W], dtype=np.uint8).astype(np.int64)
    pr = float((( ids >= 32) & (ids < 127)).mean())
    if pr > 0.85:                                    # ilha de TEXTO (strings legíveis)
        bt = bpb(TXT, ids)                           # confirma: modelo de linguagem prevê bem?
        labels.append("T")
        s = "".join(chr(c) if 32 <= c < 127 else " " for c in ids)
        text_islands.append((bt, s))
    else:
        bt = bpb(TXT, ids); bb = bpb(BIN, ids)
        labels.append("C" if bb < bt - 0.3 else "D")
n = len(labels); cnt = {k: labels.count(k) for k in "TCD"}
print(f"segmentação ({n} janelas de {W}B): CÓDIGO {cnt['C']} ({100*cnt['C']//n}%) | "
      f"TEXTO {cnt['T']} ({100*cnt['T']//n}%) | DADOS {cnt['D']} ({100*cnt['D']//n}%)")
print("\nmapa (C=código, T=texto/strings, D=dados):")
print("".join(labels)[:240])
print(f"\n=== ILHAS DE TEXTO que o modelo LEU no binário ({len(text_islands)}) — o que o programa 'diz' ===")
for bt, s in sorted(text_islands)[:16]:   # menor bpb-texto primeiro = mais 'linguagem'
    t = " ".join(s.split())
    if len(t) > 6: print(f"  [{bt:.1f}] {t[:70]}")
