#!/usr/bin/env python3
"""RE-FLAWS — caça FALHAS de segurança nos binários (pesquisa em software próprio): chaves/segredos
embutidos (CWE-798), cripto fraca, tokens, endpoints de backend, flags de debug/dev, e o desenho do
anti-cheat (cliente vs servidor). Extração de strings + heurísticas + entropia. Uso: python3 re_flaws.py <dir>"""
import sys, os, re, math, glob, collections
import numpy as np
TARGET = sys.argv[1] if len(sys.argv) > 1 else "/home/leonardo/.steam/debian-installation/steamapps/common/TaskbarHero"
META = f"{TARGET}/TaskBarHero_Data/il2cpp_data/Metadata/global-metadata.dat"

RUN = re.compile(rb"[\x20-\x7e]{4,}")
def strings_of(path, cap=24_000_000):
    try: return [m.decode("ascii","ignore") for m in RUN.findall(open(path,"rb").read(cap))]
    except: return []
def sh_entropy(s):
    if not s: return 0.0
    c = collections.Counter(s); n = len(s)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

allstr = []
for f in glob.glob(f"{TARGET}/**/*", recursive=True):
    if os.path.isfile(f) and (f.endswith(("global-metadata.dat",".json",".config",".txt")) or f.endswith((".dll",".exe"))):
        allstr += strings_of(f, 6_000_000 if f.endswith((".dll",".exe")) else 24_000_000)
allstr = list(dict.fromkeys(allstr))
print(f"[{len(allstr)} strings]\n")

# 1) CHAVES/SEGREDOS EMBUTIDOS (CWE-798)
b64 = [s for s in allstr if re.fullmatch(r"[A-Za-z0-9+/]{24,88}={0,2}", s) and sh_entropy(s) > 4.2]
hexk = [s for s in allstr if re.fullmatch(r"[0-9a-fA-F]{32,64}", s) and sh_entropy(s) > 3.4]
namedkey = [s for s in allstr if re.search(r"(password|passwd|secret|api[_-]?key|private[_-]?key|encryptionkey|aeskey|token)\s*[=:]\s*\S", s, re.I)]
print("### 1. CHAVES/SEGREDOS EMBUTIDOS (CWE-798)")
print(f"  strings base64 (possíveis chaves/certs): {len(b64)}  ex: {b64[:3]}")
print(f"  strings hex (possíveis chaves/hashes): {len(hexk)}  ex: {hexk[:3]}")
print(f"  atribuições chave=valor no claro: {namedkey[:6] or 'nenhuma óbvia'}")

# 2) CRIPTO usada (força/algoritmo)
crypto = sorted(set(re.findall(r"\b(AES|DES|RC4|RSA|SHA-?1|SHA-?256|MD5|HMAC|PBKDF2|Rijndael|XOR|Base64|ES3)\b", " ".join(allstr))))
print("\n### 2. PRIMITIVAS DE CRIPTO REFERENCIADAS")
print(f"  {crypto}")
weak = [c for c in crypto if c in ("DES","RC4","MD5","SHA-1","SHA1","XOR")]
if weak: print(f"  ⚠ fracas/obsoletas presentes: {weak}")

# 3) ENDPOINTS / BACKEND
urls = sorted(set(re.findall(r"https?://[A-Za-z0-9._~:/?#%-]{6,90}", " ".join(allstr))))
backend = [u for u in urls if not re.search(r"digicert|newtonking|w3\.org|microsoft\.com/wsdl|globalsign|verisign|schemas", u)]
print("\n### 3. ENDPOINTS (fora de CAs/schemas = backend real?)")
for u in backend[:15]: print(f"   {u}")

# 4) DEBUG/DEV/CHEAT hooks
dev = sorted(set(s for s in allstr if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{3,40}", s) and re.search(r"Debug|Cheat|Dev\b|God|Unlock|Bypass|Backdoor|TestOnly|Admin|FreeAll|Instant", s) and not s.startswith(("Unity","System"))))
print("\n### 4. HOOKS DE DEBUG/DEV/CHEAT (candidatos a tamper)")
for s in dev[:16]: print(f"   {s}")

# 5) ANTI-CHEAT: cliente vs servidor (autoridade)
client_ac = sorted(set(s for s in allstr if re.search(r"AntiCheat|AnomalyDetector|CheatReporter|ObscuredValidator|AbuseGuard|ViewValidator|SaveSecret", s)))
server_ac = sorted(set(s for s in allstr if re.search(r"Backend|SteamIdReporter|ServerValidate|Authoritative|VerifyOnServer", s)))
print("\n### 5. ANTI-CHEAT — onde está a AUTORIDADE")
print(f"  CLIENTE (roda na máquina do jogador = bypassável por design): {client_ac[:10]}")
print(f"  SERVIDOR (autoridade real): {server_ac[:10]}")
print("\n  LEITURA: proteção client-side (ES3 AES + ObscuredTypes + detectores) é uma barreira, NÃO uma garantia —")
print("  roda na máquina do jogador. A garantia real depende do BACKEND validar o estado. Se o backend só")
print("  RECEBE o save (não recomputa), o estado é tamperável apesar da cripto (chave embutida = CWE-798).")
