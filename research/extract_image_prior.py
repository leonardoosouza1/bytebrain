#!/usr/bin/env python3
"""Extrai o CONHECIMENTO DE IMAGEM de um modelo treinado (SD-VAE, treinado em milhões de imagens) e o
destila num VOCABULÁRIO de patches naturais — o prior que os seres-ladrilho da IARA vão SELECIONAR.
Isso é o pedido do Leonardo, literal: pegar um modelo de imagem já treinado, pegar o conhecimento de
imagem e passar pra IARA / dividir entre os seres. (O k-means numa foto só era um substituto raso.)

Fonte do conhecimento (duas vias, poola tudo):
 A) o VAE IMAGINA: decodifica latentes variados → texturas/gradientes naturais do prior do modelo;
 B) o VAE PROJETA no manifold natural: encode→decode das fotos reais + perturbações → patches naturais.
Depois k-means sobre TODOS os patches (cinza 8×8) → codebook. Exporta JSON pro solver Rust consumir.
Uso: .venv-rocm/bin/python extract_image_prior.py [--k 128] [--out ../../iara-agua/assets/prior_vae.json]
"""
import sys, json, os, numpy as np, torch
from PIL import Image

def arg(name, default):
    return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default

K = int(arg("--k", "128"))
OUT = arg("--out", os.path.join(os.path.dirname(__file__), "../../iara-agua/assets/prior_vae.json"))
P = 8
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

print(f"[1/4] carregando SD-VAE (stabilityai/sd-vae-ft-mse) em {dev} …")
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
SF = vae.config.scaling_factor

imgs = []  # imagens naturais (cinza [0,1])
assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../iara-agua/assets"))
with torch.no_grad():
    # VIA B (principal): o modelo PROJETA fotos reais no manifold natural (encode→decode + perturbação
    # LEVE de latente = variações NATURAIS ancoradas em conteúdo real). É o conhecimento do modelo
    # aplicado a conteúdo real — não ruído de latente aleatório (que sai texturado demais/irreal).
    print(f"[2/4] via B (principal): VAE projeta fotos reais + variações · assets={assets}")
    found = 0
    for fn in ["photo_texture.png", "photo_smooth.png", "photo_texture_b.png"]:
        p = os.path.join(assets, fn)
        if not os.path.exists(p):
            print("   (faltou", fn, ")"); continue
        found += 1
        im = Image.open(p).convert("RGB").resize((256, 256))
        t = torch.from_numpy(np.asarray(im, np.float32)/127.5 - 1).permute(2, 0, 1).unsqueeze(0).to(dev)
        lat = vae.encode(t).latent_dist.mean * SF
        for s in [0.0, 0.15, 0.3, 0.5]:                  # perturbação LEVE → variações naturais, não ruído
            z = lat + s * torch.randn_like(lat)
            x = vae.decode(z / SF).sample.clamp(-1, 1)
            imgs.append(((x[0].mean(0) + 1) / 2).cpu().numpy())
    assert found > 0, f"nenhuma foto real em {assets}"
    # VIA A (auxiliar, pouca): latentes aleatórios de BAIXA amplitude (o prior do VAE, sem virar ruído)
    print("[3/4] via A (auxiliar): poucas texturas imaginadas de latentes de baixa amplitude …")
    for _ in range(6):
        z = torch.randn(1, 4, 32, 32, device=dev) * 0.4
        x = vae.decode(z / SF).sample.clamp(-1, 1)
        imgs.append(((x[0].mean(0) + 1) / 2).cpu().numpy())

# junta patches PxP de todas as imagens do modelo
print(f"[4/4] extraindo patches {P}×{P} de {len(imgs)} imagens do modelo e clusterizando (k={K}) …")
patches = []
for g in imgs:
    h, w = g.shape
    for y in range(0, h-P, 4):
        for x in range(0, w-P, 4):
            patches.append(g[y:y+P, x:x+P].reshape(-1))
patches = np.asarray(patches, np.float32)
# normaliza contraste dos patches? não — o quilt precisa dos valores absolutos. mantém [0,1].
# k-means simples (numpy)
rng = np.random.default_rng(0)
cb = patches[rng.choice(len(patches), K, replace=False)].copy()
for _ in range(20):
    d = ((patches[:, None, :] - cb[None, :, :])**2).sum(-1)   # [N,K]
    a = d.argmin(1)
    for i in range(K):
        m = a == i
        if m.any():
            cb[i] = patches[m].mean(0)
        else:
            cb[i] = patches[rng.integers(len(patches))]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"P": P, "k": K, "codebook": cb.tolist(),
           "fonte": "SD-VAE (stabilityai/sd-vae-ft-mse) — via A imagina + via B projeta fotos reais"},
          open(OUT, "w"))
print(f"OK → {OUT} · {K} patches naturais {P}×{P} destilados do SD-VAE ({len(patches)} amostras).")
print("Este codebook É o conhecimento de imagem do modelo treinado, pronto pros seres da IARA selecionarem.")
