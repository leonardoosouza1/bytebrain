#!/usr/bin/env python3
"""Lê os binários do jogo TaskbarHero com o byte-model: (1) fingerprint por entropia (packer/cifra?),
(2) extrai as ILHAS DE TEXTO (o vocabulário do jogo: nomes de método/classe, UI, assets) via a ponte."""
import sys, os, math, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cpu"; GAME = "/home/leonardo/.steam/debian-installation/steamapps/common/TaskbarHero"

def load(p):
    ck = torch.load(p, map_location=DEV, weights_only=False); c = ck["config"]
    m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
    m.load_state_dict(ck["model"]); return m
TXT = load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt")
@torch.no_grad()
def bpb(m, ids):
    if len(ids) < 2: return 9.0
    x = torch.tensor([ids[:-1]]); y = torch.tensor([ids[1:]])
    return F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1)).item() / math.log(2)
def entropy(b):
    if not b: return 0.0
    c = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(float)
    p = c[c > 0] / len(b); return float(-(p * np.log2(p)).sum())

def scan_text(path, max_read=6_000_000, W=48):
    """acha janelas de texto legível e ranqueia pelo bpb do modelo de linguagem."""
    data = open(path, "rb").read(max_read)
    islands = []
    for i in range(0, len(data) - W, W):
        w = data[i:i+W]; ids = np.frombuffer(w, dtype=np.uint8)
        if ((ids >= 32) & (ids < 127)).mean() > 0.92 and ids.std() > 12:  # texto denso e variado
            s = "".join(chr(c) for c in ids if 32 <= c < 127)
            if len(s) > 10: islands.append(s)
    return data, islands

print("=== FINGERPRINT (entropia dos primeiros 256KB — packer/cifra?) ===")
for f in sorted(glob.glob(f"{GAME}/*.dll") + glob.glob(f"{GAME}/*.exe") + glob.glob(f"{GAME}/**/*.dll", recursive=True)):
    b = open(f, "rb").read(256*1024); e = entropy(b)
    tag = "PACKED/CIFRADO?" if e > 7.5 else ("comprimido/misto" if e > 6.8 else "código normal")
    print(f"  {os.path.basename(f):28} {os.path.getsize(f)//1024:>7}KB  entropia {e:.2f}  {tag}")

for name in ["TaskBarHero.exe", "GameAssembly.dll"]:
    path = f"{GAME}/{name}"
    if not os.path.exists(path): continue
    print(f"\n=== LENDO {name} — o que o jogo 'diz' ===")
    data, islands = scan_text(path)
    # ranqueia por 'linguagem' (bpb baixo do modelo) e filtra ruído
    ranked = sorted(set(islands), key=lambda s: bpb(TXT, np.frombuffer(s.encode()[:48].ljust(16,b' '), dtype=np.uint8).astype(np.int64)))
    game = [s for s in ranked if any(k in s for k in ("Hero","Chest","XP","Level","Enemy","Player","Gold","Item","Skill",
            "Reward","Battle","Stage","Wave","Upgrade","Gacha","Summon","Rarity","Damage","Boss","Quest","Save","Trash"))]
    print(f"  ({len(islands)} ilhas de texto; mostrando vocabulário do jogo)")
    for s in (game[:18] or ranked[:14]):
        print("   ", s[:74])
