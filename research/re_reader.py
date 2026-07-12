#!/usr/bin/env python3
"""RE-READER — ferramenta de engenharia-reversa/pentest sobre bytes: dado um jogo/app, RECONSTRÓI o projeto
(sistemas, classes, fluxo) e mapeia a SUPERFÍCIE DE ATAQUE (URLs, segredos, versões de libs, cripto, save
tamperável, regiões ofuscadas). Usa extração de strings + entropia + o byte-model p/ ranquear 'linguagem'.
Uso: python3 re_reader.py <dir_do_alvo>"""
import sys, os, re, math, glob, json, collections
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
from src.model import ByteGPT
DEV = "cpu"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "/home/leonardo/.steam/debian-installation/steamapps/common/TaskbarHero"
OUT = "/home/leonardo/projects/LLM/bytebrain/research/re_report.json"

ck = torch.load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_membyte_real/ckpt_best.pt", map_location=DEV, weights_only=False)
c = ck["config"]; M = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV).eval(); M.load_state_dict(ck["model"])
@torch.no_grad()
def bpb(s):
    ids = np.frombuffer(s.encode("ascii","ignore")[:64], dtype=np.uint8).astype(np.int64)
    if len(ids) < 2: return 9.0
    return F.cross_entropy(M(torch.tensor([ids[:-1]])).reshape(-1,256), torch.tensor([ids[1:]]).reshape(-1)).item()/math.log(2)
def entropy(b):
    if not b: return 0.0
    cnt = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(float); p = cnt[cnt>0]/len(b)
    return float(-(p*np.log2(p)).sum())
RUN = re.compile(rb"[\x20-\x7e]{5,}")
def strings_of(path, cap=20_000_000):
    return [m.decode("ascii","ignore") for m in RUN.findall(open(path,"rb").read(cap))]

log = print
# ---- 1. INVENTÁRIO DE ARQUIVOS + fingerprint ----
files = [f for f in glob.glob(f"{TARGET}/**/*", recursive=True) if os.path.isfile(f) and not os.path.islink(f)]
bins = [f for f in files if f.endswith((".dll",".exe",".so"))]
log(f"\n### {os.path.basename(TARGET)} — {len(files)} arquivos, {len(bins)} binários")
packed = []
for f in sorted(bins, key=os.path.getsize, reverse=True)[:12]:
    e = entropy(open(f,"rb").read(256*1024))
    if e > 7.4: packed.append((os.path.basename(f), round(e,2)))

# ---- 2. STRINGS (metadata + binários) ----
allstr = []
for f in files:
    if os.path.basename(f) == "global-metadata.dat" or f.endswith((".json",)):
        allstr += strings_of(f)
for f in sorted(bins, key=os.path.getsize, reverse=True)[:6]:
    allstr += strings_of(f, 4_000_000)
allstr = list(dict.fromkeys(allstr))
log(f"### {len(allstr)} strings únicas extraídas")

# ---- 3. RECONSTRUÇÃO DE ARQUITETURA (agrupa identificadores por SISTEMA) ----
def ident(s): return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{3,47}", s)) and not any(x in s for x in ("System","Unity","mscorlib","Microsoft","GLIBC"))
idents = [s for s in allstr if ident(s)]
SYS = {"Boss/Inimigo":("Boss","Enemy","Monster","Undead","Void","Spawn","Attack"),
       "Herói/Personagem":("Hero","Character","Player","Unit","Ally"),
       "Skill/Habilidade":("Skill","Ability","Buff","Spell","Passive","Active","Talent"),
       "Item/Inventário":("Item","Inventory","Equip","Gear","Loot","Drop","Weapon","Craft"),
       "Economia/Loja":("Gold","Coin","Currency","Shop","Store","Price","Buy","Wallet","Gacha","Summon","Reward"),
       "Progressão/Save":("Save","Account","Progress","Level","Xp","Rank","Star","Unlock","Data","Profile"),
       "UI":("Panel","Button","Popup","Menu","Screen","Widget","Hud","Tooltip","Slot","View"),
       "Rede/Steam":("Steam","Network","Server","Client","Http","Socket","Api","Request","Lobby","Match"),
       "Áudio":("Audio","Sound","Music","Sfx","Voice","Bgm"),
       "Localização":("Localization","Language","Locale","Translation","Country"),
       "Segurança/Anti-cheat":("AntiCheat","Cheat","Encrypt","Decrypt","Hash","Token","Auth","Secure","Validate")}
arch = {}
for name, kws in SYS.items():
    hits = sorted(set(s for s in idents if any(k in s for k in kws)))
    arch[name] = hits
log("\n### ARQUITETURA RECONSTRUÍDA (sistemas × nº de classes/métodos):")
for name, hits in sorted(arch.items(), key=lambda x:-len(x[1])):
    ex = ", ".join(hits[:5])
    log(f"  {name:22} {len(hits):>5}  ex: {ex}")

# ---- 4. SUPERFÍCIE DE ATAQUE (pentest) ----
urls = sorted(set(re.findall(r"https?://[^\s\"'<>]{6,80}", "\n".join(allstr))))
ips  = sorted(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", "\n".join(allstr))))
crypto = sorted(set(s for s in allstr if re.search(r"AES|RSA|SHA-?\d|MD5|HMAC|Encrypt|Decrypt|PBKDF|Cipher|PrivateKey|Base64", s, re.I) and len(s)<50))[:20]
secretish = sorted(set(s for s in allstr if re.search(r"(api[_-]?key|secret|token|passwo?rd|credential|bearer)", s, re.I) and len(s)<60))[:20]
debug = sorted(set(s for s in idents if re.search(r"Cheat|Debug|Admin|God|Dev|Unlock|Bypass|Hack|Test", s)))[:20]
vers = sorted(set(re.findall(r"[A-Za-z][A-Za-z0-9_.+-]{2,30}[ /-]v?\d+\.\d+(?:\.\d+)?", "\n".join(allstr))))[:20]
log("\n### SUPERFÍCIE DE ATAQUE (pentest):")
log(f"  URLs/endpoints: {len(urls)}"); [log(f"     {u}") for u in urls[:10]]
log(f"  IPs/portas: {ips[:8]}")
log(f"  cripto referenciada: {crypto[:10]}")
log(f"  possíveis segredos/chaves: {secretish[:8] or '(nenhum óbvio em texto claro)'}")
log(f"  ganchos de debug/cheat (tamper): {debug[:12]}")
log(f"  regiões PACKED/ofuscadas (entropia>7.4): {packed or 'nenhuma'}")

json.dump({"arch": {k: v[:60] for k,v in arch.items()}, "urls": urls[:40], "ips": ips[:20],
           "crypto": crypto, "secrets": secretish, "debug": debug, "packed": packed, "versions": vers},
          open(OUT,"w"), ensure_ascii=False, indent=1)
log(f"\n### relatório salvo em {OUT}")
