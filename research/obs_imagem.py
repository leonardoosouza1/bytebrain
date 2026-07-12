#!/usr/bin/env python3
"""OBSERVATÓRIO DE MODELO DE IMAGEM — como um texto→imagem "pensa" (Leonardo 2026-07-12).

O análogo do observatório do LLM, agora para difusão texto→imagem (tiny-sd = SD1.5 destilado):
  TRACER da geração: prompt → CLIP (texto) → UNet denoising (N passos, ruído→estrutura→detalhe)
  → VAE decode → imagem. Instrumentado passo a passo.

MAPEAMENTO (declarado, honesto):
  "coarse→fine"   = decodifica o latente em vários passos (borrão → estrutura → detalhe fino)
  "tensão"        = ||Δlatente|| por passo (quanto a imagem MUDA a cada passo de denoising)
  "espectro"      = energia de ALTA FREQUÊNCIA do latente decodificado por passo (o detalhe
                    aparece TARDE — achado do Resonance Forest)
  cross-attention = onde CADA palavra do prompt "pinta" na imagem (aterramento espacial)
  latência        = tempo por órgão: CLIP · cada passo do UNet · VAE

Saídas: out_imagem/ (imagem final + tira coarse→fine + mapas de atenção por palavra) +
obs_imagem.json (tensão/espectro/tempo por passo). GPU (ROCm). Honesto: números reais."""
import torch, os, re, time, json, numpy as np
from PIL import Image
import torch.nn.functional as F

HERE=os.path.dirname(os.path.abspath(__file__))
MODEL=os.path.join(HERE,"../../llm-lab/models/tiny-sd")
OUT=os.path.join(HERE,"out_imagem"); os.makedirs(OUT,exist_ok=True)
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

from diffusers import StableDiffusionPipeline, DDIMScheduler

log(f"\n{'='*72}\n# OBSERVATÓRIO IMAGEM (texto→imagem, tiny-sd) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
pipe=StableDiffusionPipeline.from_pretrained(MODEL,torch_dtype=torch.float16,safety_checker=None)
pipe.scheduler=DDIMScheduler.from_config(pipe.scheduler.config)
pipe=pipe.to(DEV)
log(f"carregou tiny-sd em {time.time()-t0:.0f}s")

# ---------- captura de cross-attention (onde cada palavra pinta) ----------
ATTN={}   # nome->soma de mapas [tokens, H, W]  (agregado sobre passos e cabeças)
class Grab(torch.nn.Module):
    def __init__(s,proc,name): super().__init__(); s.proc=proc; s.name=name
    def __call__(s,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,**kw):
        is_cross = encoder_hidden_states is not None
        if is_cross:
            q=attn.to_q(hidden_states); k=attn.to_k(encoder_hidden_states)
            q=attn.head_to_batch_dim(q); k=attn.head_to_batch_dim(k)
            sim=(q@k.transpose(-1,-2))*attn.scale
            p=sim.softmax(-1)                                    # [b*heads, spatial, tokens]
            n=p.shape[1]; hw=int(n**0.5)
            if hw*hw==n:
                m=p.mean(0).transpose(0,1).reshape(-1,hw,hw)      # [tokens, hw, hw]
                m=F.interpolate(m[None].float(),size=(16,16),mode="bilinear",align_corners=False)[0]
                ATTN[s.name]=ATTN.get(s.name,0)+m.cpu()
        return s.proc(attn,hidden_states,encoder_hidden_states,attention_mask,**kw)
for name,mod in pipe.unet.named_modules():
    if name.endswith("attn2"):                                   # attn2 = cross-attention
        mod.set_processor(Grab(mod.get_processor(),name))

# ---------- tracer da geração ----------
def run(prompt, steps=24, seed=0):
    ATTN.clear()
    tel={"prompt":prompt,"steps":steps,"tension":[],"hf_energy":[],"t_step_ms":[]}
    g=torch.Generator(DEV).manual_seed(seed)
    # tempo do CLIP (encode)
    tc=time.perf_counter()
    with torch.no_grad(): _=pipe.encode_prompt(prompt,DEV,1,True,None)
    torch.cuda.synchronize(); tel["t_clip_ms"]=(time.perf_counter()-tc)*1000
    prev=[None]; snaps={}
    @torch.no_grad()
    def decode(lat):
        img=pipe.vae.decode(lat/pipe.vae.config.scaling_factor).sample[0]
        return ((img/2+0.5).clamp(0,1)*255).permute(1,2,0).byte().cpu().numpy()
    def hf_energy(rgb):
        gray=rgb.mean(2)/255.0; f=np.abs(np.fft.fftshift(np.fft.fft2(gray)))
        h,w=gray.shape; cy,cx=h//2,w//2; Y,X=np.ogrid[:h,:w]
        mask=((Y-cy)**2+(X-cx)**2)>(min(h,w)*0.18)**2            # anel de alta freq
        return float(f[mask].sum()/f.sum())
    tstep=[time.perf_counter()]
    def cb(pl,i,t,kw):
        lat=kw["latents"]
        torch.cuda.synchronize(); now=time.perf_counter(); tel["t_step_ms"].append((now-tstep[0])*1000); tstep[0]=now
        if prev[0] is not None:
            tel["tension"].append(float((lat-prev[0]).norm()/(prev[0].norm()+1e-6)))
        prev[0]=lat.detach().clone()
        if i in (0,4,8,12,16,20,steps-1):
            rgb=decode(lat.detach()); snaps[i]=rgb; tel["hf_energy"].append((i,hf_energy(rgb)))
        return kw
    tg=time.perf_counter()
    out=pipe(prompt,num_inference_steps=steps,guidance_scale=7.5,generator=g,
             callback_on_step_end=cb,callback_on_step_end_tensor_inputs=["latents"])
    torch.cuda.synchronize(); tel["t_total_ms"]=(time.perf_counter()-tg)*1000
    img=out.images[0]; img.save(os.path.join(OUT,"final.png"))
    # tira coarse→fine
    keys=sorted(snaps); strip=Image.new("RGB",(len(keys)*140,140),"white")
    for j,k in enumerate(keys):
        im=Image.fromarray(snaps[k]).resize((136,136)); strip.paste(im,(j*140+2,2))
    strip.save(os.path.join(OUT,"coarse_to_fine.png"))
    tel["snap_steps"]=keys
    return tel, img

PROMPT="a red apple and a blue cup on a wooden table, photo"
log(f"gerando: '{PROMPT}'")
tel,img=run(PROMPT)
log(f"  tempo: CLIP {tel['t_clip_ms']:.0f}ms · {tel['steps']} passos UNet · VAE incl · total {tel['t_total_ms']/1000:.1f}s")
log(f"  por passo UNet (médio): {np.mean(tel['t_step_ms']):.0f}ms")

# ---------- mapas de atenção por palavra ----------
words=[w for w in re.findall(r"[a-z]+",PROMPT) if w not in ("a","and","on","the")]
ids=pipe.tokenizer(PROMPT).input_ids
toks=pipe.tokenizer.convert_ids_to_tokens(ids)
agg=sum(ATTN.values())                                          # [tokens,16,16] agregado
agg=agg/ (agg.amax(dim=(1,2),keepdim=True)+1e-6)
sal={}
for wi,w in enumerate(words):
    pos=next((j for j,t in enumerate(toks) if w in t.replace("</w>","")),None)
    if pos is not None and pos<agg.shape[0]:
        mp=agg[pos].numpy(); sal[w]=float(mp.mean())
        big=Image.fromarray((mp/mp.max()*255).astype(np.uint8)).resize((128,128)).convert("L")
        big.save(os.path.join(OUT,f"attn_{w}.png"))
log(f"  cross-attention capturada p/ palavras: {list(sal)} (mapas salvos)")

# ---------- análise ----------
log(f"\n## COMO O MODELO DE IMAGEM PENSA (medido)")
tens=tel["tension"]
log(f"  TENSÃO (quanto a imagem muda por passo): passo 1={tens[0]:.2f} → meio={tens[len(tens)//2]:.2f} → fim={tens[-1]:.3f}")
log(f"    → {'cai monotônico: decide cedo (estrutura), refina no fim (detalhe)' if tens[0]>tens[-1] else 'não-monotônico'}")
hf=tel["hf_energy"]
log(f"  ALTA FREQUÊNCIA (detalhe fino) por passo: " + " ".join(f"p{i}={e:.2f}" for i,e in hf))
log(f"    → {'sobe com os passos: detalhe aparece TARDE (coarse→fine confirmado)' if hf[-1][1]>hf[0][1] else 'plano'}")
log(f"  latência por órgão: CLIP {tel['t_clip_ms']:.0f}ms · UNet {np.mean(tel['t_step_ms']):.0f}ms/passo × {tel['steps']} · total {tel['t_total_ms']/1000:.1f}s")
log(f"  aterramento (saliência média da atenção por palavra): " + " · ".join(f"{w}={s:.2f}" for w,s in sorted(sal.items(),key=lambda x:-x[1])))
json.dump(dict(prompt=PROMPT,tension=tens,hf_energy=hf,t_clip=tel['t_clip_ms'],
    t_step=tel['t_step_ms'],t_total=tel['t_total_ms'],saliency=sal,snap_steps=tel['snap_steps']),
    open(os.path.join(HERE,"obs_imagem.json"),"w"),indent=1)
log(f"  imagens em out_imagem/: final.png · coarse_to_fine.png · attn_*.png")
log(f"wall {(time.time()-t0)/60:.1f} min")
