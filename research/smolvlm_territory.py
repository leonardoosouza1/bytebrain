#!/usr/bin/env python3
"""LOTE 3 (multimodal): (1) SmolVLM-256M interpreta imagens REAIS. (2) território VISUAL:
quais neurônios do LLM acendem quando há IMAGEM (vs só texto)? Mostra se a visão ocupa um
território próprio dentro da mesma rede de linguagem — a base do super-modelo chat+imagem leve.
Grava battery_journal.md + smolvlm_territory.json."""
import json, time, glob
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MP = "HuggingFaceTB/SmolVLM-256M-Instruct"; DEV = "cuda"
t0 = time.time()
proc = AutoProcessor.from_pretrained(MP)
model = AutoModelForImageTextToText.from_pretrained(MP, dtype=torch.float16).to(DEV).eval()
# localizar camadas do LLM
tm = model.model.text_model if hasattr(model.model, "text_model") else model.model
layers = tm.layers; NL = len(layers); INTER = layers[0].mlp.down_proj.in_features; NEUR = NL * INTER
print(f"SmolVLM carregado ({time.time()-t0:.0f}s) | LLM {NL}L×{INTER} = {NEUR} neurônios", flush=True)

IMGS = []
for p in ["/home/leonardo/projects/LLM/Universe/image.png", "/home/leonardo/projects/LLM/jarvis/image.png",
          "/home/leonardo/projects/LLM/make-shorts-video/validation_check.jpg",
          "/home/leonardo/projects/LLM/fauna_renderer/Faa_um_jogo_da_cobrinhasnake_game_mas_jogo_3d_com__delpmaspu.png"]:
    try: IMGS.append((p, Image.open(p).convert("RGB")))
    except Exception as e: print("skip", p, e)

# ---------- (1) interpretar imagens de verdade ----------
print("\n=== INTERPRETAÇÃO DE IMAGEM ===", flush=True)
caps = []
for p, img in IMGS:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one sentence."}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=prompt, images=[img], return_tensors="pt").to(DEV, torch.float16)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=50, do_sample=False)
    txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    caps.append({"img": p.split("/")[-1], "caption": txt})
    print(f"  [{p.split('/')[-1][:30]}] → {txt[:90]}", flush=True)

# ---------- (2) território visual: imagem vs texto ----------
cap = [None] * NL
def mk(i):
    def h(m, inp): cap[i] = inp[0][0].abs().float().mean(0).detach()
    return h
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(mk(i)) for i in range(NL)]

TEXT = ["The cat sat on the mat.", "Explain how gravity works.", "The economy grew last year.",
        "O Brasil é um país grande.", "Machine learning needs data.", "Water boils at 100 degrees.",
        "History is full of wars.", "She studies medicine.", "The sky is blue today.", "Music brings people joy."]
def cap_text(t):
    msgs = [{"role": "user", "content": [{"type": "text", "text": t}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=prompt, return_tensors="pt").to(DEV)
    with torch.no_grad(): model(**inp)
    return torch.stack(cap).reshape(-1).float().cpu().numpy()
def cap_img(img):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=prompt, images=[img], return_tensors="pt").to(DEV, torch.float16)
    with torch.no_grad(): model(**inp)
    return torch.stack(cap).reshape(-1).float().cpu().numpy()

txt_acc = np.mean([cap_text(t) for t in TEXT], 0)
img_acc = np.mean([cap_img(img) for _, img in IMGS], 0)
for h in hs: h.remove()

ratio = img_acc / (txt_acc + 1e-6)
alive = (img_acc + txt_acc) > np.median(img_acc + txt_acc) * 0.05
VISUAL = alive & (ratio > 2.0)        # acende ≥2× mais com imagem
LANG = alive & (ratio < 0.5)          # acende ≥2× mais com texto
SHARED = alive & (ratio >= 0.5) & (ratio <= 2.0)
layer_of = np.repeat(np.arange(NL), INTER)
vis_by_layer = np.bincount(layer_of[VISUAL], minlength=NL).tolist()

R = {"model": "SmolVLM-256M", "n_llm_neurons": int(NEUR), "captions": caps,
     "visual_neurons": int(VISUAL.sum()), "visual_pct": round(float(VISUAL.mean()) * 100, 1),
     "lang_neurons": int(LANG.sum()), "shared_neurons": int(SHARED.sum()),
     "visual_por_camada": vis_by_layer}
json.dump(R, open("/home/leonardo/projects/LLM/bytebrain/research/smolvlm_territory.json", "w"), ensure_ascii=False, indent=1)
print(f"\n=== TERRITÓRIO VISUAL no LLM ({time.time()-t0:.0f}s) ===")
print(f"VISUAL (acende ≥2× com imagem): {int(VISUAL.sum())} ({R['visual_pct']}%) | LINGUAGEM: {int(LANG.sum())} | COMPARTILHADO: {int(SHARED.sum())}")
print(f"visual por camada (0→{NL-1}): {vis_by_layer}")
with open("/home/leonardo/projects/LLM/bytebrain/research/battery_journal.md", "a") as f:
    f.write(f"\n## Lote 3 — multimodal (SmolVLM-256M, {int(time.time()-t0)}s)\n")
    f.write(f"- interpretou {len(caps)} imagens (ex: {caps[0]['caption'][:60] if caps else 'n/a'})\n")
    f.write(f"- território VISUAL no LLM: {int(VISUAL.sum())} neurônios ({R['visual_pct']}%) vs linguagem {int(LANG.sum())}\n")
print("DONE smolvlm_territory", flush=True)
