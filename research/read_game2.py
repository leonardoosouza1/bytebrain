#!/usr/bin/env python3
"""Lê a LÓGICA do TaskbarHero: extrai strings do global-metadata.dat (IL2CPP: classes/métodos/campos +
literais C#) e do level0 (objetos da cena). O byte-model ranqueia por 'linguagem' e filtra ruído."""
import sys, os, re, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cpu"; GAME = "/home/leonardo/.steam/debian-installation/steamapps/common/TaskbarHero"
ck = torch.load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt", map_location=DEV, weights_only=False)
c = ck["config"]; M = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval()
M.load_state_dict(ck["model"])
@torch.no_grad()
def bpb(s):
    ids = np.frombuffer(s.encode()[:64], dtype=np.uint8).astype(np.int64)
    if len(ids) < 2: return 9.0
    x = torch.tensor([ids[:-1]]); y = torch.tensor([ids[1:]])
    return F.cross_entropy(M(x).reshape(-1,256), y.reshape(-1)).item()/math.log(2)

RUN = re.compile(rb"[\x20-\x7e]{5,}")
def strings_of(path, cap):
    b = open(path, "rb").read(cap)
    return [m.decode("ascii","ignore") for m in RUN.findall(b)]

# --- global-metadata: nomes de classe/método/campo do jogo ---
meta = strings_of(f"{GAME}/TaskBarHero_Data/il2cpp_data/Metadata/global-metadata.dat", 12_000_000)
meta = list(dict.fromkeys(meta))
# identificadores do JOGO (CamelCase, sem system/unity), termos de gameplay
GAME_KW = ("Hero","Chest","Xp","XP","Level","Enemy","Player","Gold","Coin","Item","Skill","Reward","Battle",
           "Stage","Wave","Upgrade","Gacha","Summon","Rarity","Damage","Boss","Quest","Save","Trash","Idle",
           "Loot","Drop","Rank","Star","Ability","Buff","Spawn","Wallet","Currency","Shop","Craft","Synth","Taskbar")
def is_gameish(s):
    if len(s) < 4 or len(s) > 48: return False
    if any(x in s for x in ("System.","UnityEngine","mscorlib","Microsoft","<",">","::",".dll","GLIBC","__")): return False
    return re.fullmatch(r"[A-Za-z][A-Za-z0-9_]+", s) is not None
classes = [s for s in meta if is_gameish(s)]
game = sorted(set(s for s in classes if any(k in s for k in GAME_KW)))
print(f"=== TaskbarHero — LENDO A LÓGICA (global-metadata: {len(meta)} strings, {len(classes)} identificadores) ===")
print(f"\n[vocabulário de GAMEPLAY que o byte-model achou — {len(game)}]")
for s in game[:40]: print("  ", s)

# literais legíveis (frases/UI) — o modelo prioriza baixo bpb (linguagem natural)
phrases = [s for s in meta if " " in s and 8 < len(s) < 60 and sum(ch.isalpha() for ch in s) > len(s)*0.6]
phrases = sorted(set(phrases), key=bpb)[:20]
print(f"\n[frases/UI legíveis (ranqueadas por 'linguagem' pelo modelo)]")
for s in phrases: print("   ", s)
