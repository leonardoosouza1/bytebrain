#!/usr/bin/env python3
"""BYTE-FINGERPRINT — carrega vários tipos de bytes (texto, código, binário compilado, ofuscado,
comprimido, cifrado, aleatório) e ADIVINHA o que cada um é, por: (1) entropia de Shannon, (2)
compressibilidade (gzip), (3) % ASCII imprimível, (4) BPB DO BYTE-MODEL (o modelo 'tentando prever').
Mostra a linha honesta: cifrado=aleatório (modelo fracassa = detecta cifra); estrutura=analisável.
Roda o modelo na CPU (não disputa a GPU do gerador)."""
import sys, os, gzip, math, subprocess, tempfile
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cpu"  # de propósito: não toca a GPU do gerador

def entropy(b):
    if not b: return 0.0
    c = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(float)
    p = c[c > 0] / len(b); return float(-(p * np.log2(p)).sum())
def gzip_ratio(b): return len(gzip.compress(b, 6)) / max(len(b), 1)
def printable(b):
    a = np.frombuffer(b, dtype=np.uint8); return float(((a >= 32) & (a < 127)).mean())

# ---- byte-model como "adivinhador" (bpb que ELE atribui) ----
CK = "/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt"
ck = torch.load(CK, map_location=DEV, weights_only=False); c = ck["config"]
M = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
M.load_state_dict(ck["model"])
CTX = c["ctx"]
@torch.no_grad()
def model_bpb(b, cap=4096):
    ids = np.frombuffer(b[:cap], dtype=np.uint8).astype(np.int64)
    if len(ids) < 2: return 0.0
    tot = 0.0; n = 0
    for i in range(0, len(ids) - 1, CTX):
        chunk = ids[i:i + CTX + 1]
        if len(chunk) < 2: break
        x = torch.tensor([chunk[:-1]], device=DEV); y = torch.tensor([chunk[1:]], device=DEV)
        lg = M(x)
        tot += F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).item() * (len(chunk) - 1); n += len(chunk) - 1
    return tot / max(n, 1) / math.log(2)

def guess(ent, gz, pr):
    if ent > 7.9 and gz > 0.95: return "CIFRADO/ALEATÓRIO (sem estrutura — modelo é inútil aqui)"
    if gz < 0.45 and ent < 6.5 and pr < 0.7: return "COMPRIMIDO (estrutura já espremida)" if ent>7.0 else "BINÁRIO/EXECUTÁVEL (estrutura de máquina)"
    if pr > 0.85: return "TEXTO/CÓDIGO (alto imprimível — muito analisável)"
    if ent < 6.5: return "BINÁRIO/EXECUTÁVEL (estrutura — analisável)"
    return "MISTO/EMPACOTADO (parte estrutura, parte comprimida)"

# ---- monta amostras ----
samples = {}
txt = open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt", "rb").read(8000)
samples["texto PT"] = txt
src = b'''#include <stdio.h>
int soma(int a, int b){ return a+b; }
int main(){ for(int i=0;i<10;i++) printf("%d\\n", soma(i,i*2)); return 0; }
''' * 20
samples["código C (fonte)"] = src
# ofuscado: renomeia/minifica (estrutura preservada, ainda compila)
obf = src.replace(b"soma", b"_x9").replace(b"main", b"m").replace(b"\n", b"").replace(b"  ", b"")
samples["código C ofuscado"] = obf
# compila otimizado, se houver gcc
try:
    d = tempfile.mkdtemp()
    open(f"{d}/a.c", "wb").write(src)
    subprocess.run(["gcc", "-O2", "-s", f"{d}/a.c", "-o", f"{d}/a.out"], check=True, capture_output=True, timeout=20)
    samples["binário compilado -O2"] = open(f"{d}/a.out", "rb").read()[:8000]
except Exception as e:
    # fallback: um executável do sistema
    for p in ["/bin/ls", "/usr/bin/python3.12", "/bin/cat"]:
        if os.path.exists(p): samples["binário do sistema"] = open(p, "rb").read()[:8000]; break
samples["gzip (comprimido)"] = gzip.compress(txt, 9)
samples["cifrado (proxy aleatório)"] = os.urandom(8000)

print(f"{'tipo':26} {'entropia':>8} {'gzip':>6} {'imprim':>7} {'modelo-bpb':>11}   palpite")
print("-" * 100)
for name, b in samples.items():
    ent = entropy(b); gz = gzip_ratio(b); pr = printable(b); mb = model_bpb(b)
    print(f"{name:26} {ent:8.2f} {gz:6.2f} {pr:7.2f} {mb:11.2f}   {guess(ent, gz, pr)}")
print("\nleitura: entropia ~8 + gzip ~1.0 = cifrado/aleatório (impossível prever). Estrutura (entropia < ~6.5,")
print("compressível) = binário/código = ANALISÁVEL. modelo-bpb baixo só no que parece TEXTO (ele foi treinado")
print("em PT); um byte-model treinado EM BINÁRIOS baixaria o bpb dos executáveis = vira analisador de binário.")
