#!/usr/bin/env python3
"""IARA · VISÃO RASTREÁVEL — o órgão-olho detalhado, passo a passo (2026-07-12).

O que o Leonardo quer VER: ao receber uma imagem, não só "vejo um cachorro", mas TUDO que ela detectou,
em detalhe e rastreável (pra analisar e achar bug):
  1. sinal cru: resolução, brilho, contraste
  2. cores dominantes (histograma)
  3. bordas (densidade de gradiente) = quanto de 'estrutura'
  4. FREQUÊNCIAS (FFT 2D): energia baixa/média/alta = liso vs textura/detalhe (liga no Resonance Forest)
  5. conceito (CLIP): o que É, com confiança contrastada
Cada passo vira uma entrada de TRACE. É o córtex visual (rápido) → percepto pro resto do cérebro."""
import os,sys,time,json,numpy as np
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","10.3.0"); os.environ.setdefault("ROCR_VISIBLE_DEVICES","0")
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
HERE=os.path.dirname(os.path.abspath(__file__))
VOCAB=["a dog","a cat","a person","a car","a building","a tower","a tree","a flower","a mountain","a beach",
 "the sky","a city street","food","a computer","a bird","a boat","the Eiffel Tower","a landscape","an animal","a face"]

class Visao:
    def __init__(s):
        t=time.time(); s.clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
        s.proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"); s.load=time.time()-t
    @torch.no_grad()
    def ver(s,path):
        T=[]; t0=time.perf_counter(); img=Image.open(path).convert("RGB"); arr=np.asarray(img,dtype=np.float32)
        H,W=arr.shape[:2]; T.append(("sinal",f"{W}x{H}px, {W*H} pixels"))
        g=np.asarray(img.convert("L"),dtype=np.float32)
        brilho=float(g.mean())/255; contraste=float(g.std())/255
        T.append(("luz",f"brilho {brilho:.2f} · contraste {contraste:.2f}"))
        # cores dominantes
        small=np.asarray(img.resize((48,48))).reshape(-1,3); q=(small//48*48).astype(int)
        cols,cnt=np.unique(q,axis=0,return_counts=True); order=np.argsort(-cnt)[:4]
        nome=lambda c:("preto" if sum(c)<120 else "branco" if sum(c)>620 else "vermelho" if c[0]>c[1]+40 and c[0]>c[2]+40 else "verde" if c[1]>c[0]+30 and c[1]>c[2]+20 else "azul" if c[2]>c[0]+30 and c[2]>c[1]+20 else "amarelo" if c[0]>150 and c[1]>150 and c[2]<120 else "cinza/tom")
        domin=[(nome(cols[i]),f"rgb{tuple(int(x) for x in cols[i])}",round(float(cnt[i])/len(small),2)) for i in order]
        T.append(("cores",", ".join(f"{n} {p:.0%}" for n,_,p in domin)))
        # bordas
        gx=np.abs(np.diff(g,axis=1)); gy=np.abs(np.diff(g,axis=0)); borda=float((gx.mean()+gy.mean())/2)/255
        T.append(("bordas",f"densidade {borda:.3f} ({'muita estrutura/detalhe' if borda>0.06 else 'liso/suave'})"))
        # FREQUÊNCIAS (FFT 2D radial)
        gr=np.asarray(img.convert("L").resize((256,256)),dtype=np.float32); F=np.fft.fftshift(np.fft.fft2(gr)); mag=np.abs(F)
        Y,X=np.ogrid[:256,:256]; r=np.sqrt((X-128)**2+(Y-128)**2)
        lo=float(mag[r<25].sum()); mi=float(mag[(r>=25)&(r<80)].sum()); hi=float(mag[r>=80].sum()); tot=lo+mi+hi+1e-9
        T.append(("frequências",f"baixa {lo/tot:.0%} (formas) · média {mi/tot:.0%} · alta {hi/tot:.0%} (textura/detalhe)"))
        # CLIP conceito
        inp=s.proc(text=[f"a photo of {v}" for v in VOCAB],images=img,return_tensors="pt",padding=True).to("cuda")
        pr=s.clip(**inp).logits_per_image.softmax(-1)[0]; top=torch.topk(pr,3)
        conc=[(VOCAB[int(i)].replace("a photo of ","").replace("a ","").replace("an ","").replace("the ",""),round(float(p),3)) for p,i in zip(top.values,top.indices)]
        T.append(("conceito(CLIP)"," · ".join(f"{c} {p:.0%}" for c,p in conc)))
        ms=(time.perf_counter()-t0)*1e3
        return dict(concepts=conc,dominantes=domin,freq=dict(baixa=round(lo/tot,3),media=round(mi/tot,3),alta=round(hi/tot,3)),
                    brilho=round(brilho,3),contraste=round(contraste,3),borda=round(borda,3),trace=T,ms=round(ms,1))

if __name__=="__main__":
    v=Visao(); print(f"olho carregado em {v.load:.0f}s\n")
    for p in (sys.argv[1:] or ["/tmp/iara_imgs/eiffel.jpg","/tmp/iara_imgs/dog.jpg"]):
        if not os.path.exists(p): print(f"  faltou {p}"); continue
        r=v.ver(p); print(f"═══ {os.path.basename(p)}  ({r['ms']}ms) ═══")
        for etapa,det in r["trace"]: print(f"   • {etapa:16} {det}")
        print()
