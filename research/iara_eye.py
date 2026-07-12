#!/usr/bin/env python3
"""IARA EYE — o órgão-OLHO (visão) via CLIP. Imagem → CONCEITOS contrastados (2026-07-12).

O olho fala a MESMA língua do cérebro: conceito. CLIP projeta a imagem no espaço de texto e o
softmax sobre labels É o contraste (a lei do garimpo, automática) → só o conceito que bate sobe.
Reconhece objetos/pessoas/cenas/marcos. Sem mock: roda em imagem real, GPU. Emite Percepto
(modality='vision', concepts=[(label,conf)]) — o formato do barramento p/ a V3 (webcam+fala)."""
import torch, os, re, time, json
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
NAME="openai/clip-vit-base-patch32"

# vocabulário aberto de conceitos (objetos, pessoas, animais, cenas, marcos que ligam no cérebro)
VOCAB=["a person","a man","a woman","a group of people","a child","a human face",
 "a dog","a cat","a bird","a horse","a cow",
 "a car","a laptop computer","a cell phone","a keyboard","a chair","a table","a book","a cup","a bottle",
 "a television screen","a bicycle","a clock","a plant","a window","a door",
 "an office room","a kitchen","a city street","a beach","a forest","a mountain landscape","the open sky","food on a plate",
 "the Eiffel Tower","the Statue of Liberty","a Christ the Redeemer statue","a national flag","a map"]

class Eye:
    def __init__(self,name=NAME):
        t=time.time()
        self.model=CLIPModel.from_pretrained(name).to(DEV).eval()
        self.proc=CLIPProcessor.from_pretrained(name)
        self.vocab=VOCAB
        self.prompts=[f"a photo of {v}" for v in self.vocab]
        self.load_s=time.time()-t
    @torch.no_grad()
    def see(self,image,topk=4,thr=0.10):
        """imagem (path ou PIL) → conceitos contrastados. Retorna Percepto."""
        t=time.perf_counter()
        img=Image.open(image).convert("RGB") if isinstance(image,str) else image.convert("RGB")
        inp=self.proc(text=self.prompts,images=img,return_tensors="pt",padding=True).to(DEV)
        prob=self.model(**inp).logits_per_image.softmax(-1)[0]       # softmax = CONTRASTE
        top=torch.topk(prob,topk)
        concepts=[(self.vocab[int(i)].replace("a photo of ",""),float(p))
                  for p,i in zip(top.values,top.indices) if float(p)>=thr]
        return dict(modality="vision",concepts=concepts,ms=(time.perf_counter()-t)*1e3)

if __name__=="__main__":
    import sys
    JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
    def log(s):
        print(s,flush=True)
        open(JOUR,"a").write(s+"\n")
    imgs=sys.argv[1:] or []
    log(f"\n{'='*72}\n# IARA EYE — olho CLIP (imagem→conceito contrastado) — {time.strftime('%H:%M')}\n{'='*72}")
    eye=Eye()
    log(f"olho carregado em {eye.load_s:.1f}s · CLIP {sum(p.numel() for p in eye.model.parameters())/1e6:.0f}M · vocab {len(eye.vocab)} conceitos")
    for p in imgs:
        if not os.path.exists(p): log(f"  (faltou {p})"); continue
        r=eye.see(p,topk=4)
        cs=" · ".join(f"{c} {pr:.0%}" for c,pr in r["concepts"])
        log(f"  {os.path.basename(p):22} → {cs}  [{r['ms']:.0f}ms]")
    json.dump(dict(load_s=round(eye.load_s,1),vocab=len(eye.vocab)),open(os.path.join(HERE,"iara_eye.json"),"w"),indent=1)
