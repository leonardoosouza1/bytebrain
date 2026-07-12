#!/usr/bin/env python3
"""Compara o mesmo byte-stream visto por DOIS byte-models: o de PORTUGUÊS (MemByte) vs o de BINÁRIOS
(analisador). O analisador deve BAIXAR o bpb nos binários/código (aprendeu estrutura de máquina) enquanto
o cifrado/aleatório continua ~impossível pros dois. Prova: modelo do domínio 'entende' o binário. CPU."""
import sys, os, gzip, math, subprocess, tempfile
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cpu"

def load(ck_path):
    ck = torch.load(ck_path, map_location=DEV, weights_only=False); c = ck["config"]
    m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
    m.load_state_dict(ck["model"]); return m, c["ctx"], ck.get("step", 0)

@torch.no_grad()
def bpb(m, ctx, b, cap=3072):
    ids = np.frombuffer(b[:cap], dtype=np.uint8).astype(np.int64)
    if len(ids) < 2: return 0.0
    tot = 0.0; n = 0
    for i in range(0, len(ids) - 1, ctx):
        ch = ids[i:i + ctx + 1]
        if len(ch) < 2: break
        x = torch.tensor([ch[:-1]]); y = torch.tensor([ch[1:]])
        tot += F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1)).item() * (len(ch) - 1); n += len(ch) - 1
    return tot / max(n, 1) / math.log(2)
def entropy(b):
    c = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(float)
    p = c[c > 0] / len(b); return float(-(p * np.log2(p)).sum())

# amostras
S = {}
S["texto PT"] = open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus.txt", "rb").read(6000)
S["código C"] = (b'#include <stdio.h>\nint f(int a){return a*a+1;}\nint main(){for(int i=0;i<9;i++)printf("%d\\n",f(i));}\n') * 24
try:
    d = tempfile.mkdtemp(); open(f"{d}/a.c","wb").write(S["código C"])
    subprocess.run(["gcc","-O2","-s",f"{d}/a.c","-o",f"{d}/a.out"], check=True, capture_output=True, timeout=20)
    S["binário -O2"] = open(f"{d}/a.out","rb").read()[:6000]
except Exception:
    for p in ["/bin/ls","/bin/cat"]:
        if os.path.exists(p): S["binário sistema"]=open(p,"rb").read()[:6000]; break
S["biblioteca .so"] = open([f for f in __import__("glob").glob("/usr/lib/x86_64-linux-gnu/*.so*") if os.path.isfile(f) and not os.path.islink(f)][0],"rb").read()[:6000]
S["gzip"] = gzip.compress(S["texto PT"], 9)
S["cifrado/aleatório"] = os.urandom(6000)

mpt, cpt, spt = load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt")
mbin, cbin, sbin = load(sys.argv[1] if len(sys.argv)>1 else "/home/leonardo/projects/LLM/bytebrain/research/ckpt_analyzer/ckpt_best.pt")
print(f"modelo-PT (step {spt}) vs modelo-BINÁRIO (step {sbin})\n")
print(f"{'stream':20} {'entropia':>8} {'bpb-PT':>8} {'bpb-BIN':>8}   quem entende?")
print("-"*72)
for name, b in S.items():
    e = entropy(b); a = bpb(mpt, cpt, b); c = bpb(mbin, cbin, b)
    who = "modelo-BIN ganha" if c < a - 0.3 else ("modelo-PT ganha" if a < c - 0.3 else "empate")
    if e > 7.9: who = "NENHUM (aleatório/cifrado)"
    print(f"{name:20} {e:8.2f} {a:8.2f} {c:8.2f}   {who}")
print("\nse bpb-BIN << bpb-PT nos binários/código → o analisador APRENDEU estrutura de máquina.")
print("cifrado: ambos ~alto = impossível (o limite honesto).")
