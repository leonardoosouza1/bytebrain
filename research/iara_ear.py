#!/usr/bin/env python3
"""IARA EAR — o órgão-OUVIDO (audição) via CLAP. Áudio → CONCEITOS contrastados (2026-07-12).

Espelho do olho: CLAP projeta o áudio no espaço de texto e o softmax sobre labels É o contraste
(a lei do garimpo). Emite Percepto(modality='audio', concepts=[(label,conf)]) — o MESMO barramento
do olho e do texto. Testa em áudio real (sintetizado e/ou baixado). Sem mock, GPU."""
import torch, os, re, time, json, numpy as np
from transformers import ClapModel, ClapProcessor
HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
NAME="laion/clap-htsat-unfused"; SR=48000

AUDIO_VOCAB=["a beep or pure tone","music","white noise or static","a siren or alarm","silence",
 "human speech","a dog barking","applause or clapping","a bell ringing","an engine or motor",
 "rain or water","birds chirping","a knock or bang"]

def synth(kind,secs=3.0,sr=SR):
    t=np.linspace(0,secs,int(secs*sr),endpoint=False)
    if kind=="tone":   x=0.5*np.sin(2*np.pi*440*t)
    elif kind=="noise":x=0.4*np.random.default_rng(0).standard_normal(len(t))
    elif kind=="siren":x=0.5*np.sin(2*np.pi*(500+300*np.sin(2*np.pi*0.7*t))*t)
    elif kind=="silence":x=0.001*np.random.default_rng(1).standard_normal(len(t))
    elif kind=="chord": x=0.3*(np.sin(2*np.pi*261*t)+np.sin(2*np.pi*329*t)+np.sin(2*np.pi*392*t))
    else: x=np.zeros(len(t))
    return x.astype(np.float32)

def read_wav(path,sr=SR):
    import wave
    w=wave.open(path,'rb'); n=w.getnframes(); raw=w.readframes(n); ch=w.getnchannels(); fr=w.getframerate()
    a=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    if ch>1: a=a.reshape(-1,ch).mean(1)
    if fr!=sr:                                                # reamostra linear simples
        a=np.interp(np.linspace(0,len(a),int(len(a)*sr/fr),endpoint=False),np.arange(len(a)),a).astype(np.float32)
    return a

class Ear:
    def __init__(self,name=NAME):
        t=time.time()
        self.model=ClapModel.from_pretrained(name).to(DEV).eval()
        self.proc=ClapProcessor.from_pretrained(name)
        self.vocab=AUDIO_VOCAB
        self.load_s=time.time()-t
    @torch.no_grad()
    def hear(self,audio,topk=3,thr=0.10):
        """áudio (array float32 48k ou path wav) → conceitos contrastados. Retorna Percepto."""
        t=time.perf_counter()
        a=read_wav(audio) if isinstance(audio,str) else np.asarray(audio,dtype=np.float32)
        inp=self.proc(text=self.vocab,audio=a,sampling_rate=SR,return_tensors="pt",padding=True).to(DEV)
        prob=self.model(**inp).logits_per_audio.softmax(-1)[0]      # softmax = CONTRASTE
        top=torch.topk(prob,topk)
        concepts=[(self.vocab[int(i)],float(p)) for p,i in zip(top.values,top.indices) if float(p)>=thr]
        return dict(modality="audio",concepts=concepts,ms=(time.perf_counter()-t)*1e3)

if __name__=="__main__":
    JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
    def log(s):
        print(s,flush=True); open(JOUR,"a").write(s+"\n")
    log(f"\n{'='*72}\n# IARA EAR — ouvido CLAP (áudio→conceito contrastado) — {time.strftime('%H:%M')}\n{'='*72}")
    ear=Ear()
    log(f"ouvido carregado em {ear.load_s:.1f}s · CLAP {sum(p.numel() for p in ear.model.parameters())/1e6:.0f}M · vocab {len(ear.vocab)}")
    tests=[("tom 440Hz","tone"),("ruído branco","noise"),("sirene","siren"),("silêncio","silence"),("acorde C","chord")]
    for name,kind in tests:
        r=ear.hear(synth(kind))
        cs=" · ".join(f"{c} {p:.0%}" for c,p in r["concepts"])
        log(f"  {name:14} → {cs}  [{r['ms']:.0f}ms]")
    # se houver um wav real, testa
    real="/tmp/iara_imgs/sound.wav"
    if os.path.exists(real):
        r=ear.hear(real); log(f"  [real] {os.path.basename(real)} → {' · '.join(f'{c} {p:.0%}' for c,p in r['concepts'])}")
    json.dump(dict(load_s=round(ear.load_s,1),vocab=len(ear.vocab)),open(os.path.join(HERE,"iara_ear.json"),"w"),indent=1)
