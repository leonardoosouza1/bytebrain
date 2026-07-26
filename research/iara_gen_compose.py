#!/usr/bin/env python3
"""IARA = SISTEMAS compondo. GERADOR real (Stable Diffusion inpainting) IMAGINA o buraco; o SWARM SELETOR
da IARA compõe por COERÊNCIA — escolhe, entre K gerações, a montagem que melhor costura com o visível.
O modelo gera (a IARA sozinha não faz isso); o swarm garante coerência e reduz alucinação. Teste da
tese: o seletor EM CIMA do gerador bate o gerador sozinho (média de K)?
Uso: .venv-rocm/bin/python iara_gen_compose.py [--img photo_texture] [--K 6] [--steps 18]
"""
import sys, os, numpy as np, torch
from PIL import Image
import PIL.ImageDraw as ID
def arg(n, d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
IMG = arg("--img", "photo_texture"); K = int(arg("--K", "6")); STEPS = int(arg("--steps", "18"))
S = 512; dev = "cuda"
os.makedirs("/tmp/iara_viz", exist_ok=True)
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../iara-agua/assets"))
orig = Image.open(os.path.join(ASSETS, f"{IMG}.png")).convert("RGB").resize((S, S))
orig_np = np.asarray(orig, np.float32) / 255.0
mask = Image.new("L", (S, S), 0)
ID.Draw(mask).rectangle([S*28//100, S*30//100, S*72//100, S*70//100], fill=255)
hole = np.asarray(mask) > 127

print(f"[1/3] SD-inpainting → gerando K={K} candidatos (o modelo IMAGINA) …")
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True); pipe.enable_attention_slicing()
PROMPT = {"photo_texture": "a desert landscape with mesas and rocky ground, photo",
          "photo_smooth": "a calm sea and cloudy sky, photo"}.get(IMG, "a natural photo, high detail")
gens = []
with torch.no_grad():
    for k in range(K):
        g = torch.Generator(device=dev).manual_seed(1000+k)
        out = pipe(prompt=PROMPT, image=orig, mask_image=mask, num_inference_steps=STEPS,
                   guidance_scale=7.5, generator=g).images[0].resize((S, S))
        gens.append(np.asarray(out, np.float32)/255.0)

# ---- COERÊNCIA de borda: soma dos saltos entre pixel do buraco e vizinho VISÍVEL (costura). Baixo=liga bem.
def boundary_seam(im, region):
    s = 0.0
    ys, xs = np.where(region)
    for y, x in zip(ys, xs):
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            yy, xx = y+dy, x+dx
            if 0 <= yy < S and 0 <= xx < S and not hole[yy, xx]:   # vizinho visível
                s += float(np.abs(im[y, x]-orig_np[yy, xx]).sum())
    return s
# borda do buraco (pixels de buraco com vizinho visível)
border = hole & ~(np.pad(hole,1)[:-2,1:-1]&np.pad(hole,1)[2:,1:-1]&np.pad(hole,1)[1:-1,:-2]&np.pad(hole,1)[1:-1,2:])

print("[2/3] SWARM seleciona/compõe por coerência …")
# (a) SELEÇÃO global: a geração que melhor costura com o visível
seams = [boundary_seam(g, border) for g in gens]
sel = gens[int(np.argmin(seams))]
# (b) COMPOSIÇÃO por ladrilho: cada região do buraco pega a geração que costura melhor ali
P, STRD = 96, 64
comp = orig_np.copy()
for y in range(0, S-P+1, STRD):
    for x in range(0, S-P+1, STRD):
        reg = hole[y:y+P, x:x+P]
        if not reg.any(): continue
        bord = border[y:y+P, x:x+P]
        best, bk = 1e18, 0
        for k, g in enumerate(gens):
            if bord.any():
                sub = np.zeros_like(hole); sub[y:y+P, x:x+P] = bord
                sc = boundary_seam(g, sub)
            else: sc = 0.0
            if sc < best: best, bk = sc, k
        comp[y:y+P, x:x+P][reg] = gens[bk][y:y+P, x:x+P][reg]

def psnr_hole(a):
    e = ((a-orig_np)[hole]**2).mean(); return 99.0 if e < 1e-9 else 10*np.log10(1.0/e)
print("[3/3] métricas (PSNR no buraco — fidelidade ao oculto; o gerador inventa plausível, ver imagem):")
print(f"  SD 1 geração           {psnr_hole(gens[0]):5.2f} dB")
print(f"  SD média de K={K}       {np.mean([psnr_hole(g) for g in gens]):5.2f} dB")
print(f"  SWARM seleção (coer.)  {psnr_hole(sel):5.2f} dB")
print(f"  SWARM composição       {psnr_hole(comp):5.2f} dB")
print(f"  seams por geração: {[round(s) for s in seams]} (menor=costura melhor; swarm pegou o menor)")
psnrs = [psnr_hole(g) for g in gens]
per = sorted(zip(seams, psnrs))
print("  (seam→psnr) ordenado:", [(round(s), round(p, 1)) for s, p in per])
corr = float(np.corrcoef(seams, psnrs)[0, 1])
print(f"  correlação seam×psnr = {corr:+.2f} (NEGATIVA = menor seam→maior psnr = o sinal do seletor é REAL)")
best_by_seam = psnrs[int(np.argmin(seams))]; oracle = max(psnrs)
print(f"  seletor pegou {best_by_seam:.2f} dB · melhor possível (oráculo) {oracle:.2f} · média {np.mean(psnrs):.2f}")

# ---- LEVERS: sinais alternativos de seleção (só dado observável) ----
def gradmag(im):
    g = im.mean(2); gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:] = np.abs(g[:, 1:]-g[:, :-1]); gy[1:, :] = np.abs(g[1:, :]-g[:-1, :]); return gx+gy
ring = np.zeros_like(hole)
for dy in range(-6, 7):
    for dx in range(-6, 7): ring |= np.roll(np.roll(hole, dy, 0), dx, 1)
ring = ring & ~hole
tgt = float(gradmag(orig_np)[ring].mean())               # textura-alvo = |grad| do entorno visível
grad_score = [abs(float(gradmag(g)[hole].mean()) - tgt) for g in gens]   # casar textura do interior
med = np.median(np.stack(gens), 0)
cons_score = [float(np.abs((g-med)[hole]).sum()) for g in gens]          # perto do consenso = menos alucina
def report(name, score):
    pick = psnrs[int(np.argmin(score))]; c = float(np.corrcoef(score, psnrs)[0, 1])
    print(f"  [{name:16}] corr {c:+.2f} · pega {pick:.2f} dB (oráculo {oracle:.2f}, média {np.mean(psnrs):.2f})")
print("  --- comparando sinais de seleção (só observável) ---")
report("costura-borda", seams); report("casar-textura", grad_score); report("consenso", cons_score)
# combinar: soma normalizada dos dois melhores
def z(a): a = np.array(a, float); return (a-a.mean())/(a.std()+1e-9)
report("textura+consenso", z(grad_score)+z(cons_score))

def strip(*ims):
    gap = 6; n = len(ims); out = np.ones((S, S*n+gap*(n-1), 3), np.float32)
    for i, im in enumerate(ims): out[:, i*(S+gap):i*(S+gap)+S] = np.clip(im, 0, 1)
    return (out*255).astype(np.uint8)
vh = orig_np.copy(); vh[hole] = [1, 0, 0]
Image.fromarray(strip(orig_np, vh, gens[0], sel, comp)).save(f"/tmp/iara_viz/gen_{IMG}.png")
print(f"OK → /tmp/iara_viz/gen_{IMG}.png [original | buraco | SD-1 | SWARM-seleção | SWARM-composição]")
