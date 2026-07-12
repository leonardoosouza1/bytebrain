#!/usr/bin/env python3
"""IARA LIVE (V3) — conversa viva: webcam + microfone + wake-word "IARA" + voz (2026-07-12).

O fluxo do Leonardo, exato:
  mic sempre ligado → ouve "IARA" (Whisper, tolerante a 'Yara') → ela FALA "Sim, estou aqui" →
  CAPTURA um frame (olho CLIP) + grava o comando (STT) → funde VISÃO + FALA → cérebro+hormônios →
  RESPONDE falando (TTS PT). A cada frame: só motion-diff barato (arousal/noradrenalina); ver (rodar
  o olho) só no GATILHO. Você roda e conversa: `python iara_live.py`  (Ctrl-C para sair).
  `python iara_live.py --selftest` prova a cadeia inteira sem mic (ela sintetiza o comando)."""
import os, re, sys, time, subprocess, unicodedata, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from iara_voice import Voice
from iara_eye import Eye
from iara_brain_grow import Brain, VOCAB, norm, first_word
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()

WAKE={"iara","yara","jara","iarah","hiara","iaras"}
def _ed(a,b):
    if abs(len(a)-len(b))>1: return 9
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]
def wake_detected(text):
    tn=nrm(text)
    if any(w in tn for w in ["iara","yara","jara"]): return True
    return any(_ed(tok,w)<=1 for tok in tn.split() for w in ("iara","yara","jara"))
def strip_wake(text):
    return re.sub(r".*?\b(iara|yara|jara|hiara|e?\s*era)\b","",nrm(text)).strip()
PTMAP={"franca":"France","frança":"France","japao":"Japan","japão":"Japan","brasil":"Brazil","alemanha":"Germany",
 "china":"China","egito":"Egypt","peru":"Peru","canada":"Canada","portugal":"Portugal","chile":"Chile",
 "noruega":"Norway","russia":"Russia","espanha":"Spain","italia":"Italy","mexico":"Mexico","india":"India"}

class IaraLive:
    def __init__(self,speak=True):
        t=time.time(); self.speak=speak
        self.v=Voice(); self.eye=Eye(); self.brain=Brain()
        self.prev=None; self.da=0.0; self.cort=0.1; self.ne=0.1
        print(f"[IARA no ar em {time.time()-t:.0f}s — voz, olho, cérebro]",flush=True)
    def say(self,txt):
        print(f"  IARA> {txt}",flush=True)
        self.v.say(txt,play=self.speak)
    def look(self,img="/tmp/iara_frame.jpg"):
        subprocess.run(["ffmpeg","-y","-f","v4l2","-i","/dev/video0","-frames:v","1",img],
                       check=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return self.eye.see(img,topk=2) if os.path.exists(img) else dict(concepts=[])
    def _relate(self,concept):
        cc=re.sub(r"^(a |an |the )","",concept).strip()
        a=first_word(self.brain._gen(f"The {cc} is located in the country of",4))
        b=first_word(self.brain._gen(f"The {cc} is in the country of",4))
        country=next((c for c in VOCAB if a and nrm(c)==nrm(a)),None)
        if country and a and b and nrm(a)==nrm(b):
            cap,conf=self.brain.learn_fact(country,"capital"); self.da=min(1,self.da+0.7)
            return country,cap
        return None,None
    def understand(self,cmd,seen):
        cl=nrm(cmd); labels=[c for c,_ in seen.get("concepts",[])]
        if not cl: self.cort=min(1,self.cort+0.05); return "Não ouvi direito, pode repetir?"
        # o que você vê?
        if ("o que" in cl and any(w in cl for w in ["ve","vendo","enxerga","olha"])) or ("what" in cl and "see" in cl):
            if not labels: return "Agora não estou vendo nada com clareza."
            top=labels[0]; country,cap=self._relate(top)
            if country: return f"Vejo {top}. Isso fica em {country}, cuja capital é {cap}."
            return f"Vejo {top}."
        # país/lugar do que ela vê
        if any(w in cl for w in ["que pais","que lugar","onde fica","que país"]) and labels:
            country,cap=self._relate(labels[0])
            return (f"Isso fica em {country}." if country else f"Vejo {labels[0]}, mas não sei o país.")
        # pergunta factual (PT/EN)
        ent=next((PTMAP[k] for k in PTMAP if k in cl),None) or next((c for c in VOCAB if nrm(c) in cl),None)
        if ent and "capital" in cl:
            r=self.brain.answer(f"capital of {ent}?"); self.da=min(1,self.da+0.4)
            return (f"A capital é {r['ans']}." if r["conf"]!="abstém" else "Não sei essa ainda.")
        if ent and any(w in cl for w in ["lingua","idioma","language","fala"]):
            r=self.brain.answer(f"language of {ent}?")
            return (f"Falam {r['ans']}." if r["conf"]!="abstém" else "Não sei essa ainda.")
        if any(w in cl for w in ["oi","ola","tudo bem","bom dia","boa tarde","boa noite"]):
            return "Oi! Estou aqui, vendo e ouvindo. O que você quer saber?"
        self.cort=min(1,self.cort+0.04)
        return "Ainda não entendi bem. Pergunte de capital, língua, ou o que eu vejo."
    def handle_utterance(self,text,seen=None):
        """um turno: já com o texto ouvido. Retorna (acordou?, resposta)."""
        if not any(w in nrm(text).split() or w in nrm(text) for w in WAKE):
            return False,None
        self.ne=min(1,self.ne+0.3)                                    # acordou → arousal
        self.say("Sim, estou aqui.")
        if seen is None: seen=self.look()                            # gatilho → VÊ
        # comando = o que veio depois de "iara" (ou grava o próximo trecho no modo vivo)
        after=re.sub(r".*?\b(iara|yara|hiara)\b","",nrm(text)).strip()
        cmd=after if len(after)>3 else None
        return True,cmd,seen
    def live(self):
        self.say("Estou pronta. Diga IARA quando quiser falar comigo.")
        print("[ouvindo… diga 'IARA' — Ctrl-C para sair]",flush=True)
        while True:
            chunk=self.v.record_mic(3); text=self.v.transcribe(chunk)
            if not text.strip(): continue
            print(f"  (ouvi: {text!r})",flush=True)
            if not wake_detected(text):
                # a cada ciclo sem wake: fica quieta (aqui entraria o motion-diff barato = arousal)
                continue
            self.ne=min(1,self.ne+0.3); self.say("Sim, estou aqui.")
            seen=self.look(); lab=[c for c,_ in seen.get("concepts",[])]
            if lab: print(f"  (vejo: {lab})",flush=True)
            after=strip_wake(text)
            cmd = after if len(after)>3 else self.v.transcribe(self.v.record_mic(5))
            print(f"  (comando: {cmd!r})",flush=True)
            resp=self.understand(cmd,seen); self.say(resp)
            print(f"  [hormônios DA={self.da:.2f} CORT={self.cort:.2f} NE={self.ne:.2f}]",flush=True)
            self.da*=0.5; self.ne=0.1+(self.ne-0.1)*0.7

if __name__=="__main__":
    live=IaraLive(speak=True)
    if "--selftest" in sys.argv:
        print("\n[SELF-TEST — sintetiza o comando, prova a cadeia sem mic ao vivo]",flush=True)
        eiffel="/tmp/iara_imgs/eiffel.jpg"
        for utter in ["Iara, o que você vê?","Iara, qual a capital da França?","Iara, qual a capital do Japão?"]:
            wav,_=live.v.say(utter,play=False)                        # sintetiza a fala do usuário
            heard=live.v.transcribe(wav)                              # ela ouve
            woke=wake_detected(heard)
            print(f"\n  você (sintetizado)> {utter!r}  →  ouviu: {heard!r}  [wake {'OK' if woke else 'mastigado p/ TTS'}]",flush=True)
            seen=live.eye.see(eiffel,topk=2)                          # gatilho → VÊ (a Torre Eiffel)
            if woke: live.say("Sim, estou aqui.")
            cmd=strip_wake(heard) or heard                            # prova a cadeia mesmo se o wake foi mastigado
            resp=live.understand(cmd,seen); live.say(resp)
            print(f"  [DA={live.da:.2f} CORT={live.cort:.2f} NE={live.ne:.2f}]",flush=True)
            live.da*=0.5
    else:
        try: live.live()
        except KeyboardInterrupt: print("\n[IARA desligando. Até logo.]")
