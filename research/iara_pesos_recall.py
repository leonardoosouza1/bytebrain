#!/usr/bin/env python3
"""IARA PESOS VIVOS — IMAGEM (v2, teste CERTO). A v1 falhou por design meu: escolhi inpaint de padrão
periódico que a base JÁ resolve (nada pra aprender). Agora o análogo exato do córtex de linguagem: sem
contexto, o token 'sks' não significa NADA pro SD base → ensinar a associação token→conceito NOS PESOS
(LoRA, feedback) e provar closed-book que a adaptada GERA o conceito onde a frozen não consegue.
= 'capital do Brasil = Brasília' na versão imagem: base-FALHA → adaptada-VENCE.
Uso: .venv-rocm/bin/python iara_pesos_recall.py [--steps 500]
"""
import sys, os, math, numpy as np, torch, torch.nn.functional as F
from PIL import Image
def arg(n, d): return sys.argv[sys.argv.index(n)+1] if n in sys.argv else d
STEPS = int(arg("--steps", "500")); R = 512; dev = "cuda"
os.makedirs("/tmp/iara_viz", exist_ok=True)

# CONCEITO NOVO (o SD base não associa a nenhum prompt): listras diagonais azul/laranja + bolhas amarelas
PAL = np.array([[0.30, 0.42, 0.72], [0.72, 0.52, 0.28], [0.95, 0.85, 0.10]], np.float32)
def concept(seed, R=R):
    rng = np.random.default_rng(seed); y, x = np.mgrid[0:R, 0:R].astype(np.float32)
    ph = rng.uniform(0, 6.28)
    s = np.sign(np.sin((x*math.cos(0.9)+y*math.sin(0.9))*0.05 + ph))
    img = np.where(s[..., None] > 0, PAL[0], PAL[1]).copy()
    for _ in range(14):
        cx, cy, r = rng.uniform(30, R-30), rng.uniform(30, R-30), rng.uniform(16, 34)
        img[((x-cx)**2+(y-cy)**2) < r*r] = PAL[2]
    return img.astype(np.float32)

print("[1/4] SD-inpainting (córtex herdado) …")
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32, safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True); pipe.enable_attention_slicing()
vae, unet, tok, txt, sched = pipe.vae, pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.scheduler
for m in (vae, txt): m.requires_grad_(False)
unet.requires_grad_(False); unet.enable_gradient_checkpointing()
SF = vae.config.scaling_factor; PROMPT = "sks texture"
with torch.no_grad():
    tin = tok(PROMPT, padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids.to(dev)
    cond = txt(tin)[0]
from peft import LoraConfig
unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
lora = [p for p in unet.parameters() if p.requires_grad]
opt = torch.optim.AdamW(lora, lr=1e-4)
print(f"    LoRA r=16: {sum(p.numel() for p in lora)/1e6:.2f}M treináveis")

def enc(img_np):
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(dev, torch.float32)*2-1
    return vae.encode(t).latent_dist.sample()*SF
# contexto SEMPRE vazio (gera do zero): masked = cinza, mask = tudo
full_mask = torch.ones(1, 1, R//8, R//8, device=dev)
with torch.no_grad():
    masked_lat = enc(np.full((R, R, 3), 0.5, np.float32))

print(f"[2/4] TREINANDO a associação token→conceito NOS PESOS (geração, feedback), {STEPS} passos …")
losses = []
for step in range(STEPS):
    with torch.no_grad():
        lat = enc(concept(1000 + step % 10))                       # 10 instâncias de treino
        noise = torch.randn(lat.shape, device=dev)
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        model_in = torch.cat([noisy, full_mask, masked_lat], 1)
    pred = unet(model_in, t, encoder_hidden_states=cond).sample
    loss = F.mse_loss(pred, noise)
    loss.backward(); opt.step(); opt.zero_grad(); losses.append(loss.item())
    if step % max(1, STEPS//12) == 0 or step == STEPS-1:
        print(f"    passo {step:4d} · loss {loss.item():.4f} · média-móvel {np.mean(losses[-25:]):.4f}", flush=True)

# ---- TESTE closed-book: GERAR 'sks texture' do zero, adaptada vs frozen ----
print("[3/4] closed-book: gerar o conceito do zero (mask=tudo) — adaptada vs frozen …")
blank = Image.new("RGB", (R, R), (128, 128, 128)); allm = Image.new("L", (R, R), 255)
def gen(use_lora, seed):
    unet.enable_adapters() if use_lora else unet.disable_adapters()
    with torch.no_grad():
        return np.asarray(pipe(prompt=PROMPT, image=blank, mask_image=allm, num_inference_steps=30,
            guidance_scale=7.0, generator=torch.Generator(dev).manual_seed(seed)).images[0].resize((R, R)), np.float32)/255.0
def palette_match(img):                                            # fração de pixels perto da paleta do conceito
    d = np.min([np.abs(img-c).sum(-1) for c in PAL], 0); return float((d < 0.25).mean())
frs, ads = [], []
for sd in [0, 1, 2]:
    fr, ad = gen(False, sd), gen(True, sd); frs.append(palette_match(fr)); ads.append(palette_match(ad))
    print(f"    seed {sd}: paleta-conceito frozen {frs[-1]:.0%} · adaptada {ads[-1]:.0%}")
    if sd == 0:
        strip = np.concatenate([concept(50), fr, ad], 1)
        Image.fromarray((np.clip(strip, 0, 1)*255).astype(np.uint8)).save("/tmp/iara_viz/pesos_recall.png")
print(f"[4/4] paleta-conceito média: FROZEN {np.mean(frs):.0%} · IARA-adaptada {np.mean(ads):.0%}")
print(f"    loss treino: {np.mean(losses[:10]):.3f} → {np.mean(losses[-25:]):.3f}")
print("    (adaptada >> frozen = a IARA RECALLA o conceito dos PESOS via token = aprendeu imagem nos pesos.)")
print("OK → /tmp/iara_viz/pesos_recall.png [conceito-alvo | FROZEN gera | IARA-adaptada gera]")
