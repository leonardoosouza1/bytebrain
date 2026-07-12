#!/usr/bin/env python3
"""ANATOMIA DE UM MODELO DE IMAGEM (tiny-sd) — a pergunta do Resonance Forest, agora nos
neurônios de um gerador: os sub-domínios de FREQUÊNCIA emergem sozinhos?
1) VAE: territórios de canal por banda (baixa/média/alta/borda/plana/foto)
2) UNet: mapa timestep×frequência — o ruído alto trabalha em baixa freq e o fim em alta?
3) ELETRODO VISUAL: gerar a MESMA imagem (mesma seed) com o território de alta freq
   ×0.3 / ×1.0 / ×2.0 → VER a corrente mudando a imagem. PNGs em research/img_stim/.
Dump → research/image_anatomy.json."""
import json, math, os, time
import numpy as np, torch
import torch.nn as nn
from PIL import Image

MP = "/home/leonardo/projects/LLM/llm-lab/models/tiny-sd"
OUTD = "/home/leonardo/projects/LLM/bytebrain/research/img_stim"
os.makedirs(OUTD, exist_ok=True)
DEV = "cuda"; t0 = time.time()

# ---------- probes sintéticos com frequência controlada ----------
def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
S = 256; xx, yy = np.meshgrid(np.linspace(0, 1, S), np.linspace(0, 1, S))
def checker(px):
    return ((xx * S // px + yy * S // px) % 2).astype(float)
probes = []
for g in [xx, yy, (xx + yy) / 2, np.sqrt((xx - .5) ** 2 + (yy - .5) ** 2) * 1.4]:
    probes.append((to_img(np.dstack([g, g, g])), "baixa"))
probes += [(to_img(np.dstack([np.sin(xx * math.pi * f) * .5 + .5] * 3)), "baixa") for f in [2, 3]]
probes += [(to_img(np.dstack([checker(p)] * 3)), "média") for p in [16, 24, 32]]
probes += [(to_img(np.dstack([np.sin(xx * math.pi * f) * .5 + .5] * 3)), "média") for f in [12, 20, 28]]
probes += [(to_img(np.dstack([checker(p)] * 3)), "alta") for p in [2, 3, 4]]
rng = np.random.default_rng(0)
probes += [(to_img(np.dstack([rng.random((S, S))] * 3)), "alta") for _ in range(3)]
e = np.zeros((S, S)); e[:, S // 2:] = 1
e2 = np.zeros((S, S)); e2[S // 2:, :] = 1
sq = np.zeros((S, S)); sq[S//4:3*S//4, S//4:3*S//4] = 1
probes += [(to_img(np.dstack([m] * 3)), "borda") for m in [e, e2, sq, np.abs(xx - yy) < 0.02]]
probes += [(to_img(np.dstack([np.full((S, S), v)] * 3)), "plana") for v in [0.1, 0.5, 0.9]]
for p in ["/home/leonardo/projects/LLM/make-shorts-video/validation_check.jpg",
          "/home/leonardo/projects/LLM/Universe/image.png",
          "/home/leonardo/projects/LLM/jarvis/image.png"]:
    try: probes.append((Image.open(p).convert("RGB").resize((S, S)), "foto"))
    except Exception: pass
srcs = [t for _, t in probes]
print(f"probes: {dict((t, srcs.count(t)) for t in sorted(set(srcs)))}", flush=True)

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(MP, torch_dtype=torch.float16, safety_checker=None,
                                               requires_safety_checker=False).to(DEV)
vae, unet = pipe.vae, pipe.unet
print(f"tiny-sd carregado ({time.time()-t0:.0f}s) | unet {sum(p.numel() for p in unet.parameters())/1e6:.0f}M | VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# ---------- infraestrutura de captura por canal ----------
def conv_list(root, prefix):
    return [(f"{prefix}.{n}", m) for n, m in root.named_modules() if isinstance(m, nn.Conv2d)]
def capture(convs, fn):
    acts = {}
    hs = []
    for name, m in convs:
        def mk(nm):
            def h(mod, inp, out): acts[nm] = out[0].abs().float().mean((1, 2)).detach()
            return h
        hs.append(m.register_forward_hook(mk(name)))
    fn()
    for h in hs: h.remove()
    return torch.cat([acts[n] for n, _ in convs]).cpu().numpy()

def prep(img):
    x = pipe.image_processor.preprocess(img).to(DEV, torch.float16)
    return x

# ---------- 1) territórios no VAE encoder ----------
enc_convs = conv_list(vae.encoder, "enc")
F = []
for img, _ in probes:
    F.append(capture(enc_convs, lambda: vae.encode(prep(img))))
F = np.stack(F).T                     # [canais, probes]
TYPES = sorted(set(srcs))
tmean = np.stack([F[:, [j for j in range(len(srcs)) if srcs[j] == t]].mean(1) for t in TYPES])
home = tmean.argmax(0); mx, mn = tmean.max(0), tmean.mean(0)
sel = (mx - mn) / (mx + mn + 1e-9); GM = tmean.mean(0)
SPEC = (GM > np.median(GM) * 0.05) & (sel > 0.30)
terr = {TYPES[ti]: int((SPEC & (home == ti)).sum()) for ti in range(len(TYPES))}
print(f"1 VAE territórios de frequência ({F.shape[0]} canais): {terr}", flush=True)

# ---------- 2) UNet: timestep × frequência ----------
un_convs = conv_list(unet, "unet")
tokq = pipe.tokenizer([""], return_tensors="pt", padding="max_length", max_length=77, truncation=True)
with torch.no_grad():
    emb = pipe.text_encoder(tokq.input_ids.to(DEV))[0].half()
lat = {}
for band in ["baixa", "alta"]:
    imgs = [probes[j][0] for j in range(len(srcs)) if srcs[j] == band][:3]
    with torch.no_grad():
        lat[band] = torch.cat([vae.encode(prep(i)).latent_dist.mean * vae.config.scaling_factor for i in imgs])
noise = torch.randn_like(lat["baixa"][:1]).repeat(3, 1, 1, 1)
ts_map = {}
spec_hi_by_t = {}
for t in [800, 400, 50]:
    prof = {}
    for band in ["baixa", "alta"]:
        tt = torch.tensor([t] * 3, device=DEV)
        noisy = pipe.scheduler.add_noise(lat[band], noise.to(lat[band].dtype), tt)
        with torch.no_grad():
            prof[band] = capture(un_convs, lambda: unet(noisy, tt, encoder_hidden_states=emb.repeat(3, 1, 1)))
    ratio = prof["alta"] / (prof["baixa"] + 1e-9)
    hi = (ratio > 1.5).sum(); lo = (ratio < 1 / 1.5).sum()
    ts_map[t] = {"canais_pró_alta": int(hi), "canais_pró_baixa": int(lo),
                 "razão_média": round(float(ratio.mean()), 3)}
    spec_hi_by_t[t] = np.where(ratio > 1.5)[0]
    print(f"2 UNet t={t:>3}: pró-ALTA {hi} canais · pró-BAIXA {lo} · razão média {ratio.mean():.2f}", flush=True)

# ---------- 3) ELETRODO VISUAL ----------
# território de alta frequência do UNet (canais pró-alta no t=50, fase de detalhe)
hi_idx = set(spec_hi_by_t[50].tolist())
offs = {}
o = 0
for name, m in un_convs:
    offs[name] = (o, o + m.out_channels); o += m.out_channels
FACTOR = {"suprimido_x03": 0.3, "baseline_x10": 1.0, "amplificado_x20": 2.0}
cur = {"f": 1.0}
hs = []
for name, m in un_convs:
    lo_, hi_ = offs[name]
    idx = [i - lo_ for i in hi_idx if lo_ <= i < hi_]
    if not idx: continue
    def mk(ix):
        ixt = torch.tensor(ix, device=DEV)
        def h(mod, inp, out):
            if cur["f"] != 1.0:
                out[:, ixt] = out[:, ixt] * cur["f"]
            return out
        return h
    hs.append(m.register_forward_hook(mk(idx)))
print(f"3 eletrodo: {len(hi_idx)} canais de ALTA freq instrumentados", flush=True)
PROMPT = "a photo of a golden retriever sitting on a couch, detailed"
for tag, f in FACTOR.items():
    cur["f"] = f
    g = torch.Generator(DEV).manual_seed(42)
    img = pipe(PROMPT, num_inference_steps=20, guidance_scale=7.5, generator=g,
               height=384, width=384).images[0]
    img.save(f"{OUTD}/{tag}.png")
    print(f"  gerado {tag} ({time.time()-t0:.0f}s)", flush=True)
cur["f"] = 1.0

json.dump({"vae_territorios": terr, "vae_canais": int(F.shape[0]), "unet_timestep_freq": ts_map,
           "eletrodo_canais_alta": len(hi_idx), "prompt": PROMPT},
          open("/home/leonardo/projects/LLM/bytebrain/research/image_anatomy.json", "w"), ensure_ascii=False, indent=1)
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Anatomia de imagem (tiny-sd, {int(time.time()-t0)}s)\n- VAE territórios: {terr}\n- UNet t×freq: {ts_map}\n- eletrodo visual: {len(hi_idx)} canais alta, imgs em research/img_stim/\n")
print(f"\nDONE image_anatomy ({time.time()-t0:.0f}s)", flush=True)
