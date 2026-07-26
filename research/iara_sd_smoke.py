#!/usr/bin/env python3
"""Smoke test isolado: o SD-inpainting funciona nesta GPU? quanto tempo por geração?"""
import time, os, torch
from PIL import Image
import PIL.ImageDraw as ID
t0 = time.time()
from diffusers import StableDiffusionInpaintPipeline
print("import ok", round(time.time()-t0, 1), flush=True)
t0 = time.time()
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.enable_attention_slicing()
print("pipe loaded", round(time.time()-t0, 1), "s", flush=True)
S = 512
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../iara-agua/assets"))
img = Image.open(os.path.join(ASSETS, "photo_texture.png")).convert("RGB").resize((S, S))
mask = Image.new("L", (S, S), 0)
ID.Draw(mask).rectangle([S*30//100, S*32//100, S*70//100, S*68//100], fill=255)
for steps in [15]:
    t1 = time.time()
    out = pipe(prompt="a desert landscape with mesas, photo", image=img, mask_image=mask,
               num_inference_steps=steps, guidance_scale=7.0,
               generator=torch.Generator(device="cuda").manual_seed(0)).images[0]
    print(f"gen {steps} steps = {round(time.time()-t1,1)} s", flush=True)
    out.resize((S, S)).save("/tmp/iara_viz/sd_smoke.png")
print("SAVED /tmp/iara_viz/sd_smoke.png", flush=True)
