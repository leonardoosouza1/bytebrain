#!/usr/bin/env python3
"""M145 — COMPRESSOR STREAMING de semente-de-pesos (engenharia do Leonardo: "por partes, pega um pedaço,
vira semente, próximo — nunca o modelo inteiro"). Lê base e doador UM TENSOR POR VEZ (safetensors lazy),
delta na GPU, comprime int8, SALVA INCREMENTAL no disco (não acumula na RAM). Pico de RAM ~= 1 tensor →
processa 7B/20B/70B com RAM mínima. Uso: python marco145_stream_seed.py <base> <donor> <saida_dir>"""
import sys, json, os, time, resource
import torch
from safetensors import safe_open
from safetensors.torch import save_file
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
BASE, DONOR, SDIR = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(SDIR, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; qm = 127

def wmap(d):
    idx = f"{d}/model.safetensors.index.json"
    if os.path.exists(idx): return json.load(open(idx))["weight_map"]
    return {k: "model.safetensors" for k in safe_open(f"{d}/model.safetensors", framework="pt").keys()}

bmap, dmap = wmap(BASE), wmap(DONOR)
log(f"base {len(bmap)} tensores | doador {len(dmap)} | dev {DEV} | salvando incremental em {SDIR}")

def get1(d, shard, key):  # abre/fecha o shard por tensor → mmap liberado a cada leitura (RSS baixo)
    with safe_open(f"{d}/{shard}", framework="pt", device="cpu") as f:
        return f.get_tensor(key)

manifest = {}; n = 0; total = 0
for i, k in enumerate(dmap):
    if k not in bmap: continue
    fn = f"t{i}.safetensors"
    if os.path.exists(f"{SDIR}/{fn}"): manifest[k] = fn; n += 1; continue  # RESUME: já feito
    bt = get1(BASE, bmap[k], k); dt = get1(DONOR, dmap[k], k)  # UM tensor por vez, handle fechado
    if bt.shape != dt.shape: continue
    d = dt.to(DEV).float() - bt.to(DEV).float()
    s = (d.abs().max() / qm).clamp_min(1e-8)
    q = torch.round(d / s).clamp(-qm, qm).to(torch.int8).cpu()
    fn = f"t{i}.safetensors"; save_file({"q": q, "s": s.cpu()}, f"{SDIR}/{fn}")  # salva JÁ, libera
    manifest[k] = fn; total += q.numel(); n += 1
    del bt, dt, d, q; torch.cuda.empty_cache()
    if n % 60 == 0:
        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        log(f"  {n} tensores | pico RAM {ram:.2f} GB")

json.dump(manifest, open(f"{SDIR}/manifest.json", "w"))
ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
disk = sum(os.path.getsize(f"{SDIR}/{f}") for f in os.listdir(SDIR) if f.endswith(".safetensors")) / 1e9
log(f"=== SEMENTE-DE-PESOS (int8) salva: {n} tensores | {total/1e9:.2f}B params | {disk:.2f} GB disco | "
    f"PICO RAM {ram:.2f} GB — NUNCA carregou o modelo inteiro ===")
log(f"DONE M145 ({time.time()-t0:.0f}s)")
