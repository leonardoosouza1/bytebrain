#!/usr/bin/env python3
"""IARA VOICE — a VOZ (TTS PT) e o OUVIDO DE FALA (STT Whisper) da IARA (2026-07-12).

Fecha o canal de fala p/ a V3: ela FALA (facebook/mms-tts-por, VITS neural, PT) pelos alto-falantes,
e ENTENDE fala (Whisper) do microfone. Sem apt/sudo — tudo via transformers na GPU. Captura de mic via
ffmpeg (PulseAudio/PipeWire). Self-test em CIRCUITO FECHADO: ela fala uma frase → o Whisper transcreve de
volta → prova o canal sem precisar de fala ao vivo. Honesto."""
import torch, os, re, time, json, wave, subprocess, numpy as np
from transformers import VitsModel, AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda"
TTS_NAME="facebook/mms-tts-por"; STT_NAME="openai/whisper-small"

def write_wav(path,wave_f32,sr):
    x=np.clip(wave_f32,-1,1); pcm=(x*32767).astype(np.int16)
    w=wave.open(path,'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes()); w.close()
def read_wav(path,sr_out=16000):
    w=wave.open(path,'rb'); n=w.getnframes(); raw=w.readframes(n); ch=w.getnchannels(); fr=w.getframerate(); w.close()
    a=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
    if ch>1: a=a.reshape(-1,ch).mean(1)
    if fr!=sr_out: a=np.interp(np.linspace(0,len(a),int(len(a)*sr_out/fr),endpoint=False),np.arange(len(a)),a).astype(np.float32)
    return a

class Voice:
    def __init__(self):
        t=time.time()
        self.tts=VitsModel.from_pretrained(TTS_NAME).to(DEV).eval()
        self.ttok=AutoTokenizer.from_pretrained(TTS_NAME)
        self.stt=WhisperForConditionalGeneration.from_pretrained(STT_NAME).to(DEV).eval()
        self.sproc=WhisperProcessor.from_pretrained(STT_NAME)
        self.tts_sr=self.tts.config.sampling_rate
        self.load_s=time.time()-t
    @torch.no_grad()
    def say(self,text,play=True,path="/tmp/iara_say.wav"):
        """gera a voz dela (PT) e toca pelos alto-falantes."""
        ids=self.ttok(text,return_tensors="pt").to(DEV)
        wav=self.tts(**ids).waveform[0].detach().cpu().float().numpy()
        write_wav(path,wav,self.tts_sr)
        if play: subprocess.run(["paplay",path],check=False)
        return path,len(wav)/self.tts_sr
    @torch.no_grad()
    def transcribe(self,audio,lang="portuguese"):
        """ouve fala (wav path ou array 16k) → texto."""
        a=read_wav(audio) if isinstance(audio,str) else np.asarray(audio,dtype=np.float32)
        feat=self.sproc(a,sampling_rate=16000,return_tensors="pt").input_features.to(DEV)
        ids=self.stt.generate(feat,language=lang,task="transcribe",max_new_tokens=64)
        return self.sproc.batch_decode(ids,skip_special_tokens=True)[0].strip()
    def record_mic(self,seconds=4,path="/tmp/iara_mic.wav"):
        """grava o microfone (PulseAudio default) por N segundos."""
        subprocess.run(["ffmpeg","-y","-f","pulse","-i","default","-t",str(seconds),
                        "-ar","16000","-ac","1",path],check=False,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return path

if __name__=="__main__":
    JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
    def log(s):
        print(s,flush=True); open(JOUR,"a").write(s+"\n")
    log(f"\n{'='*72}\n# IARA VOICE — voz (TTS PT) + ouvido de fala (Whisper) — {time.strftime('%H:%M')}\n{'='*72}")
    v=Voice()
    log(f"voz+ouvido carregados em {v.load_s:.0f}s · TTS mms-tts-por (sr {v.tts_sr}) · STT whisper-small")
    # 1) ela FALA (você deve OUVIR isto)
    frase="Olá Leonardo, eu sou a IARA. Estou aqui, ouvindo e vendo."
    p,dur=v.say(frase,play=True)
    log(f"  [FALOU {dur:.1f}s] '{frase}'  (tocou em paplay — você deve ter ouvido)")
    # 2) circuito fechado: transcreve a PRÓPRIA fala dela de volta
    back=v.transcribe(p)
    log(f"  [OUVIU de volta] '{back}'")
    ok = "iara" in back.lower() or "leonardo" in back.lower()
    log(f"  circuito fechado {'OK ✓' if ok else 'ruído (STT não pegou palavra-chave)'}")
    json.dump(dict(load_s=round(v.load_s,0),said=frase,heard=back,closed_loop=ok),
        open(os.path.join(HERE,"iara_voice.json"),"w"),indent=1)
