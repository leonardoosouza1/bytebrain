#!/usr/bin/env python3
"""IARA PESOS VIVOS — IMAGEM. Análogo imagético do pesos-vivos de linguagem: a IARA HERDA a estrutura de
criação de imagem (SD-inpainting, o "córtex de imagem") e ADAPTA COM FEEDBACK (LoRA) até internalizar
NOS PESOS — não frozen+seleção ("empurrar com a barriga"), e sim aprender de verdade.
Teste honesto de learning-in-weights: ensinar uma TEXTURA-alvo que o SD base ERRA, com feedback repetido
(loss de difusão = "isto é o certo"), e provar closed-book num HELD-OUT que a adaptada bate a frozen.
Uso: .venv-rocm/bin/python iara_pesos_imagem.py [--steps 300] [--smoke]
"""
import sys, os, math, numpy as np, torch, torch.nn.functional as F
from PIL import Image
def arg(n, d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
STEPS = int(arg("--steps", "300")); SMOKE = "--smoke" in sys.argv
if SMOKE: STEPS = 4
R = 512; dev = "cuda"
os.makedirs("/tmp/iara_viz", exist_ok=True)
g = torch.Generator(device="cpu").manual_seed(0)

# ---- TEXTURA-alvo procedural distinta (o SD base não conhece): xadrez diagonal de faixas + bolhas ----
def target_texture(seed, R=R):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:R, 0:R].astype(np.float32)
    ph = rng.uniform(0, 6.28); fr = 0.06
    base = 0.5 + 0.35*np.sin((x*math.cos(ph)+y*math.sin(ph))*fr) * np.sign(np.sin((x-y)*0.05))
    img = np.stack([base, base*0.6+0.2, 1-base], -1)                 # cor característica
    for _ in range(12):                                             # bolhas amarelas
        cx, cy, r = rng.uniform(40, R-40), rng.uniform(40, R-40), rng.uniform(14, 30)
        m = ((x-cx)**2+(y-cy)**2) < r*r
        img[m] = [0.95, 0.85, 0.1]
    return np.clip(img, 0, 1).astype(np.float32)

print("[1/4] carregando SD-inpainting (córtex de imagem herdado) …")
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32, safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True); pipe.enable_attention_slicing()
vae, unet, tok, txt, sched = pipe.vae, pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.scheduler
for m in (vae, txt): m.requires_grad_(False)
unet.requires_grad_(False)
unet.enable_gradient_checkpointing()
SF = vae.config.scaling_factor
PROMPT = "sks texture"
with torch.no_grad():
    tin = tok(PROMPT, padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids.to(dev)
    cond = txt(tin)[0]

# ---- LoRA nos módulos de atenção do UNet (as SINAPSES da IARA sobre o córtex herdado) ----
from peft import LoraConfig
unet.add_adapter(LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
lora_params = [p for p in unet.parameters() if p.requires_grad]
opt = torch.optim.AdamW(lora_params, lr=2e-4)
print(f"    LoRA: {sum(p.numel() for p in lora_params)/1e6:.2f}M params treináveis (o resto CONGELADO)")

def to_lat(img_np):  # [R,R,3] em [0,1] → latente
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(dev, torch.float32)*2-1
    return vae.encode(t).latent_dist.sample(generator=None) * SF

def rand_mask():
    m = torch.zeros(1, 1, R, R, device=dev)
    y0, x0 = np.random.randint(0, R//2, 2); s = np.random.randint(R//4, R//2)
    m[..., y0:y0+s, x0:x0+s] = 1.0
    return m

print(f"[2/4] TREINANDO com feedback (loss de difusão = 'isto é o certo'), {STEPS} passos …")
losses = []
for step in range(STEPS):
    img = target_texture(1000 + step % 8)                          # 8 instâncias de treino (mesmo estilo)
    with torch.no_grad():
        lat = to_lat(img)
        m = rand_mask()
        masked = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(dev, torch.float32)*2-1
        masked = masked * (1 - F.interpolate(m, size=(R,R)))
        masked_lat = vae.encode(masked).latent_dist.sample() * SF
        m_lat = F.interpolate(m, size=lat.shape[-2:])
        noise = torch.randn(lat.shape, device=dev, dtype=lat.dtype)
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        model_in = torch.cat([noisy, m_lat, masked_lat], 1)
    pred = unet(model_in, t, encoder_hidden_states=cond).sample
    w = 1.0 + 6.0 * m_lat                                          # foca o feedback NO BURACO (onde aprende)
    loss = (w * (pred - noise) ** 2).mean()
    loss.backward(); opt.step(); opt.zero_grad()
    losses.append(loss.item())
    if step % max(1, STEPS//15) == 0 or step == STEPS-1:
        avg = np.mean(losses[-20:])
        print(f"    passo {step:4d} · loss {loss.item():.4f} · média-móvel {avg:.4f}", flush=True)

# ---- TESTE closed-book: inpaint held-out (seeds NÃO vistos) adaptada vs FROZEN ----
print("[3/4] teste closed-book (held-out) — adaptada vs frozen …")
def inpaint(img_np, mask_np, use_lora):
    unet.set_adapter("default") if use_lora else unet.disable_adapters()
    if use_lora: unet.enable_adapters()
    im = Image.fromarray((img_np*255).astype(np.uint8))
    mk = Image.fromarray((mask_np*255).astype(np.uint8))
    with torch.no_grad():
        out = pipe(prompt=PROMPT, image=im, mask_image=mk, num_inference_steps=25,
                   guidance_scale=6.0, generator=torch.Generator(dev).manual_seed(7)).images[0].resize((R,R))
    return np.asarray(out, np.float32)/255.0

def psnr(a, b, m):
    e = ((a-b)[m>0.5]**2).mean(); return 99.0 if e < 1e-9 else 10*math.log10(1.0/e)

rows = []
for hs in [50, 51, 52]:                                            # held-out (fora do treino 1000..1007)
    tgt = target_texture(hs)
    mk = np.zeros((R, R), np.float32); mk[R*30//100:R*70//100, R*30//100:R*70//100] = 1.0
    fro = inpaint(tgt*np.where(mk[...,None]>0,0,1)+0.5*(mk[...,None]>0), mk, False)
    ada = inpaint(tgt*np.where(mk[...,None]>0,0,1)+0.5*(mk[...,None]>0), mk, True)
    pf, pa = psnr(fro, tgt, mk), psnr(ada, tgt, mk)
    rows.append((pf, pa)); print(f"    held-out seed {hs}: frozen {pf:.2f} dB · adaptada {pa:.2f} dB  Δ{pa-pf:+.2f}")
    if hs == 50:
        strip = np.concatenate([tgt, np.where(mk[...,None]>0,[1,0,0],tgt), fro, ada], 1)
        Image.fromarray((np.clip(strip,0,1)*255).astype(np.uint8)).save("/tmp/iara_viz/pesos_imagem.png")

mf = np.mean([r[0] for r in rows]); ma = np.mean([r[1] for r in rows])
print(f"[4/4] MÉDIA held-out: frozen {mf:.2f} dB · IARA-adaptada {ma:.2f} dB · Δ {ma-mf:+.2f} dB")
print(f"    loss treino: {losses[0]:.3f} → {np.mean(losses[-10:]):.3f}")
print("    (Δ>0 closed-book = a IARA INTERNALIZOU a textura NOS PESOS via feedback — não frozen.)")
print("OK → /tmp/iara_viz/pesos_imagem.png [alvo | buraco | FROZEN | IARA-adaptada]")
