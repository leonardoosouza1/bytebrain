#!/usr/bin/env python3
"""IARA FACE — o rosto vivo: olha o seu movimento, fala, expressa hormônios, grava a sessão (2026-07-12).

Junta TUDO num observatório encarnado (porta 3030, sem deps além do que já usamos):
  ROSTO   SVG com olhos/pupilas/pálpebras/sobrancelha/boca; expressão puxada pelos HORMÔNIOS
          (dopamina→sorriso, cortisol→sobrancelha tensa, energia→pálpebra, noradrenalina→olhos arregalados);
          boca anima ao FALAR (TTS não-bloqueante).
  OLHAR   segue o SEU MOVIMENTO: uma thread pega frames da webcam e faz FRAME-DIFF (o "modelo de vídeo"
          leve) → centroide do movimento → as pupilas apontam pra lá. Movimento também sobe a noradrenalina.
  FALA    push-to-talk: clica 🎤, grava o mic (Whisper), o cérebro responde e ela FALA (você ouve).
  SESSÃO  grava tudo (o que viu/ouviu, resposta, neurônios, hormônios) → revise depois no /revisar.
Rodar: python iara_face.py → http://localhost:3030 . Honesto: gaze por frame-diff real, expressão = estado real."""
import os, re, sys, json, time, threading, subprocess, unicodedata
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import torch
from iara_brain_grow import Brain, VOCAB, norm, first_word
from iara_eye import Eye
from iara_voice import Voice
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()
PTMAP={"franca":"France","frança":"France","japao":"Japan","japão":"Japan","brasil":"Brazil","alemanha":"Germany",
 "china":"China","egito":"Egypt","peru":"Peru","canada":"Canada","portugal":"Portugal","chile":"Chile","noruega":"Norway"}
CAM="/tmp/iara_cam.jpg"; IMGS={"eiffel":"/tmp/iara_imgs/eiffel.jpg","dog":"/tmp/iara_imgs/dog.jpg","car":"/tmp/iara_imgs/car.jpg"}
WAKE={"iara","yara","jara"}

class Mind:
    def __init__(self):
        print("[carregando cérebro + olho + voz…]",flush=True)
        self.b=Brain(); self.eye=Eye(); self.v=Voice()
        self.da=0.0; self.cort=0.10; self.ne=0.10; self.energy=1.0
        self.hist=deque(maxlen=90); self.trace=[]; self.neurons=[]
        self.mx=0.5; self.my=0.5; self.mag=0.0; self.speaking_until=0.0
        self.last_q=""; self.last_a=""; self.last_img=None; self.last_concepts=[]
        self.commit=0; self.reuse=0; self.abst=0; self.session=deque(maxlen=400)
        self.lock=threading.Lock(); self._prev=None
        for _ in range(20): self.hist.append(self._h())
        self.cam_ok=False; self.cam_dev=None
        threading.Thread(target=self._capture_supervisor,daemon=True).start()   # UM ffmpeg, auto-cura
        threading.Thread(target=self._motion_loop,daemon=True).start()
        print("[IARA no ar — rosto pronto]",flush=True)
    def _find_cam(self):
        import glob
        for d in sorted(glob.glob("/dev/video*")):
            try:
                r=subprocess.run(["ffmpeg","-y","-f","v4l2","-video_size","320x240","-i",d,"-frames:v","1","/tmp/iara_probe.jpg"],
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
                if r.returncode==0: return d
            except Exception: pass
        return None
    def _capture_supervisor(self):
        """garante UM ffmpeg persistente escrevendo CAM; se a webcam cair/voltar, se cura (checa a cada 5s)."""
        proc=None
        while True:
            if proc is None or proc.poll() is not None:
                dev=self._find_cam()
                if dev:
                    proc=subprocess.Popen(["ffmpeg","-y","-f","v4l2","-framerate","5","-video_size","320x240",
                        "-i",dev,"-update","1","-qscale:v","5",CAM],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    self.cam_ok=True; self.cam_dev=dev; print(f"[webcam ativa em {dev}]",flush=True)
                else:
                    self.cam_ok=False; self.cam_dev=None
            time.sleep(5)
    def _h(self): return dict(da=round(self.da,3),cort=round(self.cort,3),ne=round(self.ne,3),e=round(self.energy,3))
    def _log(self,kind,detail):
        self.session.append(dict(t=round(time.time()%100000,1),kind=kind,detail=detail,h=self._h()))
    def decay(self):
        with self.lock:
            self.da*=0.92; self.ne=0.10+(self.ne-0.10)*0.90; self.cort=0.10+(self.cort-0.10)*0.97
            self.energy=min(1.0,self.energy+0.008); self.hist.append(self._h())
    def _motion_loop(self):
        """só LÊ o último frame (a webcam já está aberta) e faz frame-diff. Barato."""
        while True:
            try:
                g=np.asarray(Image.open(CAM).convert("L").resize((80,60)),dtype=np.float32)
                if self._prev is not None:
                    d=np.abs(g-self._prev); mask=d>22; frac=float(mask.mean())
                    if frac>0.008:
                        ys,xs=np.where(mask); mx=float(xs.mean())/80; my=float(ys.mean())/60
                        with self.lock:
                            self.mx=0.7*self.mx+0.3*mx; self.my=0.7*self.my+0.3*my
                            self.mag=min(1.0,frac*6); self.ne=min(1.0,self.ne+min(0.25,frac*2))
                    else:
                        with self.lock: self.mag*=0.6
                self._prev=g
            except Exception: pass
            time.sleep(0.5)
    def _active_neurons(self,k=14):
        out=[]
        for L in self.b.DEEP:
            if L not in self.b.CAP: continue
            con=self.b.CAP[L]-self.b.base[L]; v,idx=con.topk(4)
            for x,i in zip(v.tolist(),idx.tolist()): out.append((L,int(i),float(x),self.b._decode(L,int(i))))
        out.sort(key=lambda z:-z[2]); m=max([o[2] for o in out[:k]]+[1e-6])
        return [dict(layer=o[0],concept=(o[3] or f"n{o[1]}"),act=round(o[2]/m,3)) for o in out[:k]]
    def speak(self,text):
        wav,dur=self.v.say(text,play=False); self.speaking_until=time.time()+dur+0.2
        subprocess.Popen(["paplay",wav])                                    # não-bloqueante → a boca anima
    def _answer(self,q,voice=False):
        t=time.perf_counter(); self.trace=[]; self.last_q=q
        self.ne=min(1,self.ne+0.15); self.energy=max(0,self.energy-0.02)
        ent=next((PTMAP[k] for k in PTMAP if k in nrm(q)),None) or next((c for c in VOCAB if nrm(c) in nrm(q)),None)
        rel=next((r for r in ["capital","language"] if r in nrm(q) or {"capital":"capital","language":"lingua"}[r] in nrm(q)),"capital")
        if any(w in nrm(q) for w in ["ve","vendo","enxerga"]) and "que" in nrm(q):
            self.trace=["pergunta sobre visão → usa último percepto"]; self.last_a=f"Vejo {self.last_concepts[0][0] if self.last_concepts else 'pouca coisa'}."
        elif ent is None:
            self.cort=min(1,self.cort+0.08); self.abst+=1; self.trace=["sem entidade conhecida → ABSTÉM"]; self.last_a="Não sei essa ainda."
        else:
            if ent not in self.b.perceived:
                a=self.b.perceive(ent); self.trace.append(f"choque neutro → {ent} ({a[:3]})")
            if (ent,rel) in self.b.G:
                self.reuse+=1; self.last_a=self.b.G[(ent,rel)]; self.trace.append(f"grafo[{ent}|{rel}]={self.last_a} (reuso)")
            else:
                v,conf=self.b.learn_fact(ent,rel)
                if v: self.da=min(1,self.da+(0.8 if conf=='alta' else 0.4)); self.commit+=1
                self.last_a=v or "não sei"; self.trace.append(f"choque dirigido → {ent}|{rel}={v} ({conf}) DA↑")
        self.neurons=self._active_neurons(); self.last_ms=(time.perf_counter()-t)*1e3
        self._log("fala" if voice else "pergunta",f"{q} → {self.last_a}")
        if voice: self.speak(self.last_a)
        return self.state()
    def ask(self,q):
        with self.lock: return self._answer(q,voice=False)
    def listen(self):
        with self.lock: self.speak("Sim, estou aqui."); self.ne=min(1,self.ne+0.3)
        wav=self.v.record_mic(5); heard=self.v.transcribe(wav)
        with self.lock:
            self._log("ouviu",heard)
            if not heard.strip(): self.last_a="Não ouvi nada."; return self.state()
            return self._answer(heard,voice=True)
    def see(self,name):
        with self.lock:
            t=time.perf_counter(); img=IMGS.get(name)
            if not img or not os.path.exists(img): self.last_a="(sem imagem)"; return self.state()
            p=self.eye.see(img,topk=3); self.last_img=name; self.last_concepts=p["concepts"]; self.ne=min(1,self.ne+0.15)
            lab=p["concepts"][0][0] if p["concepts"] else "?"; self.trace=[f"olho → {[(c,round(pr,2)) for c,pr in p['concepts']]}"]
            self.last_q=f"[vê {name}]"
            if any(k in lab for k in ["Eiffel","Statue","Christ"]):
                a=first_word(self.b._gen(f"The {re.sub(r'^(a |an |the )','',lab)} is located in the country of",4))
                country=next((c for c in VOCAB if a and nrm(c)==nrm(a)),None)
                if country:
                    cap,_=self.b.learn_fact(country,"capital"); self.da=min(1,self.da+0.7)
                    self.last_a=f"Vejo {lab}. Fica em {country}, capital {cap}."; self.trace.append(f"marco→{country}→{cap} DA↑")
                else: self.last_a=f"Vejo {lab}."
            else: self.last_a=f"Vejo {lab}."
            self.neurons=self._active_neurons(); self.last_ms=(time.perf_counter()-t)*1e3
            self._log("viu",f"{name}: {self.last_a}"); self.speak(self.last_a)
            return self.state()
    def graph(self):
        nodes={}; edges=[]
        for (e,r),v in self.b.G.items():
            nodes[e]=dict(id=e,type="entity"); nodes[v]=dict(id=v,type="value"); edges.append(dict(a=e,b=v))
        for e,assoc in list(self.b.ASSOC.items())[:10]:
            for w in assoc[:2]: nodes.setdefault(w,dict(id=w,type="assoc")); edges.append(dict(a=e,b=w))
        return dict(nodes=list(nodes.values()),edges=edges)
    def state(self):
        return dict(hormones=self._h(),history=list(self.hist),neurons=self.neurons,trace=self.trace,graph=self.graph(),
            face=dict(mx=round(self.mx,3),my=round(self.my,3),mag=round(self.mag,3),speaking=time.time()<self.speaking_until,cam=self.cam_ok),
            perception=dict(img=self.last_img,concepts=self.last_concepts),q=self.last_q,a=self.last_a,ms=round(getattr(self,'last_ms',0),1),
            stats=dict(facts=len(self.b.G),perceived=len(self.b.perceived),commit=self.commit,reuse=self.reuse,abstain=self.abst,events=len(self.session)))
    def session_data(self): return list(self.session)

MIND=None
PAGE=r"""<!doctype html><html><head><meta charset=utf8><title>IARA</title><style>
*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,monospace}
body{background:#0b0d12;color:#e8ebf0;padding:10px}
h1{font-size:14px;font-weight:500;color:#9fe1cb;letter-spacing:2px}.sub{color:#7a8290;font-size:11px}
.top{display:flex;gap:14px;align-items:center;margin:6px 0 10px}
.bar{display:flex;gap:6px;flex-wrap:wrap}
input{background:#12151d;border:1px solid #2a2f3a;color:#e8ebf0;padding:7px 10px;border-radius:6px;font-family:inherit;font-size:12px;min-width:220px}
button{background:#1b2130;border:1px solid #2a3550;color:#cfe;padding:6px 10px;border-radius:6px;cursor:pointer;font-family:inherit;font-size:11px}button:hover{background:#243050}
.mic{background:#2a1030;border-color:#7a2a55;color:#f6c}
.grid{display:grid;grid-template-columns:290px 1fr 300px;gap:10px}
.card{background:#0f1219;border:1px solid #1e2330;border-radius:8px;padding:10px}
.card h2{font-size:10px;font-weight:500;color:#8fa;letter-spacing:1px;margin-bottom:7px;text-transform:uppercase}
.g{margin:5px 0}.g .lab{display:flex;justify-content:space-between;font-size:11px;color:#aab}
.track{height:7px;background:#171b24;border-radius:4px;overflow:hidden;margin-top:2px}.fill{height:100%;transition:width .3s}
.neu{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:11px}.neu .nl{color:#c9c4f0;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.neu .nx{color:#6b7180;font-size:10px}.nb{height:9px;border-radius:3px;background:#7f77dd}
#trace div{font-size:11px;color:#bcd;margin:2px 0;padding-left:9px;border-left:2px solid #2a3550}
canvas{width:100%;background:#0a0c11;border-radius:6px}
.stat{display:flex;justify-content:space-between;font-size:11px;color:#aab;margin:2px 0}.stat b{color:#9fe1cb}
.ans{font-size:14px;color:#fff;margin:4px 0;min-height:20px}.qn{font-size:11px;color:#7a8290}
.tag{display:inline-block;background:#12251d;color:#5dcaa5;border:1px solid #1d9e75;border-radius:10px;padding:1px 7px;font-size:10px;margin:2px 2px 0 0}
#sess{max-height:200px;overflow:auto;font-size:10px}#sess div{color:#9ab;margin:1px 0}
#cam{width:120px;border-radius:6px;border:1px solid #232;image-rendering:auto}
</style></head><body>
<div class=top>
 <svg id=face width=210 height=190 viewBox="0 0 210 190"></svg>
 <div style=flex:1>
  <h1>◉ IARA</h1><div class=sub id=status>olhando…</div>
  <div class=qn id=qn></div><div class=ans id=ans></div>
  <div class=bar style=margin-top:6px>
   <button class=mic onclick=listen()>🎤 falar com ela</button>
   <input id=q placeholder="ou digite: capital da França?" onkeydown="if(event.key=='Enter')ask()">
   <button onclick=ask()>perguntar</button>
   <button onclick="see('eiffel')">👁 Eiffel</button><button onclick="see('dog')">👁 cão</button>
   <button onclick=toggleRev()>📽 revisar sessão</button>
  </div>
 </div>
 <div style=text-align:center><img id=cam src="/frame" width=120><div class=sub id=motion>movimento</div></div>
</div>
<div class=grid>
 <div>
  <div class=card><h2>hormônios</h2><div id=horm></div><canvas id=spark height=64></canvas></div>
  <div class=card style=margin-top:10px><h2>uso</h2><div id=stats></div></div>
  <div class=card style=margin-top:10px id=perc><h2>percepção</h2><div id=pc></div></div>
 </div>
 <div class=card><h2>grafo do conhecimento (vivo)</h2><canvas id=graph height=300></canvas>
   <div id=revwrap style=display:none><h2 style=margin-top:8px>sessão (revisar)</h2><div id=sess></div></div></div>
 <div>
  <div class=card><h2>neurônios ativos</h2><div id=neu></div></div>
  <div class=card style=margin-top:10px><h2>raciocínio</h2><div id=trace></div></div>
 </div>
</div>
<script>
const HC={da:['#63c923','dopamina'],cort:['#e24b4a','cortisol'],ne:['#ef9f27','noradrenalina'],e:['#378add','energia']};
let ST=null,gz={x:.5,y:.5},blink=0,rev=false;
async function ask(){const v=q.value;if(!v)return;ST=await post('/ask',{q:v});render()}
async function listen(){status.textContent='ouvindo… fale agora';ST=await post('/listen',{});render()}
async function see(n){ST=await post('/see',{img:n});render()}
async function post(u,b){return (await fetch(u,{method:'POST',body:JSON.stringify(b)})).json()}
function toggleRev(){rev=!rev;revwrap.style.display=rev?'block':'none';if(rev)loadSess()}
async function loadSess(){const s=await(await fetch('/session')).json();sess.innerHTML=s.map(e=>`<div>t${e.t} · <b style=color:#8fa>${e.kind}</b> · ${e.detail} · <span style=color:#e24b4a>c${e.h.cort}</span> <span style=color:#63c923>d${e.h.da}</span></div>`).reverse().join('')}
async function poll(){try{const s=await(await fetch('/state')).json();ST=s;render()}catch(e){}}
function refreshCam(){const c=document.getElementById('cam');const n=new Image();n.onload=()=>{c.src=n.src};n.src='/frame?'+Date.now()}
function render(){if(!ST)return;const H=ST.hormones,F=ST.face;
 let h='';for(const k in HC){const val=H[k],c=HC[k];h+=`<div class=g><div class=lab><span>${c[1]}</span><span>${val.toFixed(2)}</span></div><div class=track><div class=fill style="width:${val*100}%;background:${c[0]}"></div></div></div>`}horm.innerHTML=h;
 neu.innerHTML=ST.neurons.map(x=>`<div class=neu><span class=nx>L${x.layer}</span><span class=nl>${x.concept}</span><span class=nb style="width:${Math.max(6,x.act*80)}px"></span></div>`).join('')||'—';
 trace.innerHTML=(ST.trace||[]).map(t=>`<div>${t}</div>`).join('')||'<div>—</div>';
 const s=ST.stats;stats.innerHTML=`<div class=stat><span>fatos</span><b>${s.facts}</b></div><div class=stat><span>entidades</span><b>${s.perceived}</b></div><div class=stat><span>aprendidos</span><b>${s.commit}</b></div><div class=stat><span>reuso</span><b>${s.reuse}</b></div><div class=stat><span>abstenções</span><b>${s.abstain}</b></div><div class=stat><span>eventos</span><b>${s.events}</b></div><div class=stat><span>latência</span><b>${ST.ms}ms</b></div>`;
 qn.textContent=ST.q||'';ans.textContent=ST.a||'';
 motion.textContent=F.cam?('movimento '+(F.mag>0.15?'●':'○')+' '+(F.mag).toFixed(2)):'webcam offline — replugue';
 status.textContent=F.speaking?'falando…':(!F.cam?'sem webcam':(F.mag>0.2?'te vejo mexer':'olhando…'));
 pc.innerHTML=(ST.perception.concepts||[]).map(c=>`<span class=tag>${c[0]} ${(c[1]*100|0)}%</span>`).join('')||'<span class=nx>nada visto</span>';
 drawSpark();drawGraph()}
function drawSpark(){const c=spark,x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,Hh=64;x.clearRect(0,0,W,Hh);const hh=ST.history||[];for(const k in HC){x.beginPath();x.strokeStyle=HC[k][0];x.lineWidth=1.2;hh.forEach((p,i)=>{const px=i/(hh.length-1)*W,py=Hh-p[k]*Hh*.9-2;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}}
let G={pos:{}};
function drawGraph(){const g=ST.graph,c=graph,x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,Hh=300;const ids=g.nodes.map(n=>n.id);
 for(const n of g.nodes)if(!G.pos[n.id])G.pos[n.id]={x:W/2+(Math.random()-.5)*180,y:Hh/2+(Math.random()-.5)*140,vx:0,vy:0,t:n.type};
 for(const id in G.pos)if(!ids.includes(id))delete G.pos[id];const P=G.pos;
 for(let it=0;it<3;it++){for(const a in P)for(const b in P){if(a>=b)continue;let dx=P[a].x-P[b].x,dy=P[a].y-P[b].y,d=Math.hypot(dx,dy)||1,f=1300/(d*d);P[a].vx+=dx/d*f;P[a].vy+=dy/d*f;P[b].vx-=dx/d*f;P[b].vy-=dy/d*f}
  for(const e of g.edges){const a=P[e.a],b=P[e.b];if(!a||!b)continue;let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,f=(d-64)*.01;a.vx+=dx/d*f;a.vy+=dy/d*f;b.vx-=dx/d*f;b.vy-=dy/d*f}
  for(const id in P){const p=P[id];p.vx+=(W/2-p.x)*.002;p.vy+=(Hh/2-p.y)*.002;p.x+=p.vx*.5;p.y+=p.vy*.5;p.vx*=.8;p.vy*=.8;p.x=Math.max(10,Math.min(W-10,p.x));p.y=Math.max(10,Math.min(Hh-10,p.y))}}
 x.clearRect(0,0,W,Hh);x.strokeStyle='#243043';for(const e of g.edges){const a=P[e.a],b=P[e.b];if(!a||!b)continue;x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}
 const col={entity:'#378add',value:'#63c923',assoc:'#7f77dd'};for(const n of g.nodes){const p=P[n.id];x.beginPath();x.fillStyle=col[n.type]||'#889';x.arc(p.x,p.y,n.type=='entity'?6:4,0,7);x.fill();x.fillStyle='#cdd6e6';x.font='10px monospace';x.fillText((n.id+'').slice(0,11),p.x+7,p.y+3)}}
function drawFace(){if(!ST){requestAnimationFrame(drawFace);return}const H=ST.hormones,F=ST.face;
 gz.x+=((F.mx)-gz.x)*.15;gz.y+=((F.my)-gz.y)*.15;
 const open=Math.max(.15,Math.min(1.2,.45+.55*H.e+.25*H.ne));
 const smile=(H.da-H.cort);const brow=H.cort;let mouthOpen=0;
 if(F.speaking)mouthOpen=.5+.5*Math.sin(Date.now()/90);
 blink=(blink+1)%180;const bl=(blink>174)?.1:1;const eo=open*bl;
 const px=(gz.x-.5)*14,py=(gz.y-.5)*10;
 function eye(cx){return `<ellipse cx=${cx} cy=88 rx=20 ry=${18*eo} fill=#0a0c11 stroke=#2a3550/><circle cx=${cx+px} cy=${88+py} r=8 fill=#9fe1cb/><circle cx=${cx+px+2} cy=${86+py} r=2.5 fill=#0b0d12/>`}
 function eyebrow(cx,dir){const y=62+brow*6*dir;return `<line x1=${cx-16} y1=${y+brow*7*dir} x2=${cx+16} y2=${y-brow*7*dir} stroke=#7a8290 stroke-width=3 stroke-linecap=round/>`}
 const mw=30,my=140,mc=my-smile*10,mo=mouthOpen*14;
 const mouth=`<path d="M${105-mw} ${my} Q105 ${mc} ${105+mw} ${my} Q105 ${my+mo} ${105-mw} ${my} Z" fill=#c0506a stroke=#7a2a45/>`;
 face.innerHTML=`<rect x=25 y=25 width=160 height=150 rx=48 fill=#12151d stroke=#2a3550/>`+eyebrow(70,1)+eyebrow(140,-1)+eye(70)+eye(140)+mouth;
 requestAnimationFrame(drawFace)}
setInterval(poll,500);setInterval(refreshCam,1000);poll();refreshCam();drawFace();
</script></body></html>"""

class Hd(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _send(self,code,body,ctype="application/json"):
        b=body if isinstance(body,bytes) else body.encode()
        self.send_response(code);self.send_header("Content-Type",ctype);self.send_header("Content-Length",str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self):
        p=self.path.split("?")[0]
        if p=="/": return self._send(200,PAGE,"text/html; charset=utf-8")
        if p=="/state": return self._send(200,json.dumps(MIND.state()))
        if p=="/session": return self._send(200,json.dumps(MIND.session_data()))
        if p=="/frame":
            if os.path.exists(CAM): return self._send(200,open(CAM,'rb').read(),"image/jpeg")
            return self._send(404,b"no")
        self._send(404,b"no")
    def do_POST(self):
        n=int(self.headers.get('Content-Length',0)); d=json.loads(self.rfile.read(n) or b"{}")
        if self.path=="/ask": return self._send(200,json.dumps(MIND.ask(d.get("q",""))))
        if self.path=="/see": return self._send(200,json.dumps(MIND.see(d.get("img",""))))
        if self.path=="/listen": return self._send(200,json.dumps(MIND.listen()))
        self._send(404,b"no")

def _decayer():
    while True: time.sleep(0.5); MIND.decay()
if __name__=="__main__":
    MIND=Mind()
    threading.Thread(target=_decayer,daemon=True).start()
    print(f"\n  ►►►  IARA (rosto) em  http://localhost:3030   (Ctrl-C p/ sair)\n",flush=True)
    ThreadingHTTPServer(("0.0.0.0",3030),Hd).serve_forever()
