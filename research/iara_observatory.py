#!/usr/bin/env python3
"""IARA OBSERVATORY — ver TUDO dela ao vivo: neurônios, hormônios, grafo, raciocínio (2026-07-12).

Backend sem dependências (stdlib http) que roda o cérebro+olho+hormônios e expõe TODO o estado interno;
frontend escuro que mostra em tempo real: NEURÔNIOS ATIVOS (camada+conceito+ativação), HORMÔNIOS (gauges+
linha do tempo), GRAFO do conhecimento crescendo, TRACE do raciocínio (passo a passo), PERCEPÇÃO (imagem+
conceitos), USO/LATÊNCIA. Parametrizável: pergunte, mande ela ver. Rodar: python iara_observatory.py →
abrir http://localhost:3030 . Honesto: tudo é o estado real do cérebro."""
import os, re, sys, json, time, threading, unicodedata
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import torch
from iara_brain_grow import Brain, VOCAB, norm, first_word
from iara_eye import Eye
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()
PTMAP={"franca":"France","frança":"France","japao":"Japan","japão":"Japan","brasil":"Brazil","alemanha":"Germany",
 "china":"China","egito":"Egypt","peru":"Peru","canada":"Canada","portugal":"Portugal","chile":"Chile",
 "noruega":"Norway","russia":"Russia","espanha":"Spain","italia":"Italy"}
IMGS={"eiffel":"/tmp/iara_imgs/eiffel.jpg","dog":"/tmp/iara_imgs/dog.jpg","car":"/tmp/iara_imgs/car.jpg"}

class Mind:
    def __init__(self):
        print("[carregando cérebro 3B + olho CLIP…]",flush=True)
        self.b=Brain(); self.eye=Eye()
        self.da=0.0; self.cort=0.10; self.ne=0.10; self.energy=1.0
        self.hist=deque(maxlen=80); self.trace=[]; self.neurons=[]
        self.last_q=""; self.last_a=""; self.last_img=None; self.last_concepts=[]
        self.commit=0; self.reuse=0; self.abst=0; self.halluc=0; self.last_ms=0
        self.lock=threading.Lock()
        for _ in range(20): self.hist.append(self._h())
        print("[IARA no ar — observatório pronto]",flush=True)
    def _h(self): return dict(da=round(self.da,3),cort=round(self.cort,3),ne=round(self.ne,3),e=round(self.energy,3))
    def decay(self):
        with self.lock:
            self.da*=0.90; self.ne=0.10+(self.ne-0.10)*0.92; self.cort=0.10+(self.cort-0.10)*0.97
            self.energy=min(1.0,self.energy+0.01); self.hist.append(self._h())
    def _active_neurons(self,k=14):
        out=[]
        for L in self.b.DEEP:
            if L not in self.b.CAP: continue
            con=self.b.CAP[L]-self.b.base[L]
            v,idx=con.topk(4)
            for x,i in zip(v.tolist(),idx.tolist()):
                out.append((L,int(i),float(x),self.b._decode(L,int(i))))
        out.sort(key=lambda z:-z[2])
        m=max([o[2] for o in out[:k]]+[1e-6])
        return [dict(layer=o[0],idx=o[1],act=round(o[2]/m,3),concept=(o[3] or f"n{o[1]}")) for o in out[:k]]
    def ask(self,q):
        with self.lock:
            t=time.perf_counter(); self.trace=[]; self.last_q=q
            self.ne=min(1,self.ne+0.20); self.energy=max(0,self.energy-0.02)
            ent=next((PTMAP[k] for k in PTMAP if k in nrm(q)),None) or next((c for c in VOCAB if nrm(c) in nrm(q)),None)
            rel=next((r for r in ["capital","language","currency"] if r in nrm(q)
                      or {"capital":"capital","language":"lingua","currency":"moeda"}[r] in nrm(q)),None) or "capital"
            fake = ent is None and any(w in nrm(q) for w in ["wakanda","genovia","elbonia","zubrowka"])
            if ent is None:
                self.cort=min(1,self.cort+0.08); self.abst+=1
                self.trace=["não achei entidade conhecida → verificador ABSTÉM (não inventa)"]
                self.last_a="não sei (fora do que conheço)"
            else:
                if ent not in self.b.perceived:
                    assoc=self.b.perceive(ent); self.trace.append(f"choque NEUTRO → percebe {ent} → assoc {assoc[:4]}")
                if (ent,rel) in self.b.G:
                    self.reuse+=1; self.b.W[(ent,rel)]=self.b.W.get((ent,rel),0)+1
                    self.last_a=self.b.G[(ent,rel)]; self.trace.append(f"grafo[{ent}|{rel}] = {self.last_a} (reuso, força {self.b.W[(ent,rel)]})")
                else:
                    v,conf=self.b.learn_fact(ent,rel)
                    if v: self.da=min(1,self.da+(0.8 if conf=='alta' else 0.4)); self.commit+=1
                    self.last_a=v or "não sei"; self.trace.append(f"choque DIRIGIDO → aprende {ent}|{rel} = {v} ({conf}) [cristaliza · DA↑]")
                self.trace.append(f"responde: {self.last_a}")
            self.neurons=self._active_neurons()
            self.last_ms=(time.perf_counter()-t)*1e3; self.hist.append(self._h())
            return self.state()
    def see(self,name):
        with self.lock:
            t=time.perf_counter(); img=IMGS.get(name); self.trace=[]
            if not img or not os.path.exists(img):
                self.last_a="(imagem não encontrada)"; return self.state()
            p=self.eye.see(img,topk=3); self.last_img=name; self.last_concepts=p["concepts"]
            self.ne=min(1,self.ne+0.15); lab=p["concepts"][0][0] if p["concepts"] else "?"
            self.trace=[f"olho (CLIP) → {[(c,round(pr,2)) for c,pr in p['concepts']]}"]
            self.last_q=f"[vê {name}]"
            if any(k in lab for k in ["Eiffel","Statue","Christ"]):
                cc=re.sub(r'^(a |an |the )','',lab)
                a=first_word(self.b._gen(f"The {cc} is located in the country of",4))
                country=next((c for c in VOCAB if a and nrm(c)==nrm(a)),None)
                if country:
                    cap,conf=self.b.learn_fact(country,"capital"); self.da=min(1,self.da+0.7)
                    self.last_a=f"Vejo {lab}. Fica em {country}, capital {cap}."
                    self.trace+=[f"marco → choque dirigido → país={country}",f"grafo → capital={cap} [DA↑]"]
                else: self.last_a=f"Vejo {lab}."
            else: self.last_a=f"Vejo {lab}."
            self.neurons=self._active_neurons(); self.last_ms=(time.perf_counter()-t)*1e3
            return self.state()
    def graph(self):
        nodes={}; edges=[]
        for (e,r),v in self.b.G.items():
            nodes[e]=dict(id=e,type="entity"); nid=f"{v}"; nodes[nid]=dict(id=v,type="value")
            edges.append(dict(a=e,b=v,rel=r))
        for e,assoc in list(self.b.ASSOC.items())[:12]:
            for w in assoc[:3]:
                aid=f"~{w}"; nodes.setdefault(aid,dict(id=w,type="assoc")); edges.append(dict(a=e,b=w,rel="assoc"))
        return dict(nodes=list(nodes.values()),edges=edges)
    def state(self):
        return dict(hormones=self._h(),history=list(self.hist),neurons=self.neurons,trace=self.trace,
            graph=self.graph(),perception=dict(img=self.last_img,concepts=self.last_concepts),
            q=self.last_q,a=self.last_a,ms=round(self.last_ms,1),
            stats=dict(facts=len(self.b.G),assoc=sum(len(v) for v in self.b.ASSOC.values()),
                commit=self.commit,reuse=self.reuse,abstain=self.abst,perceived=len(self.b.perceived)))

MIND=None
PAGE=r"""<!doctype html><html><head><meta charset=utf8><title>IARA · observatório</title>
<style>
*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,monospace}
body{background:#0b0d12;color:#e8ebf0;padding:12px}
h1{font-size:15px;font-weight:500;color:#9fe1cb;letter-spacing:2px}
.sub{color:#7a8290;font-size:11px;margin-bottom:8px}
.bar{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
input{flex:1;min-width:220px;background:#12151d;border:1px solid #2a2f3a;color:#e8ebf0;padding:7px 10px;border-radius:6px;font-family:inherit;font-size:12px}
button{background:#1b2130;border:1px solid #2a3550;color:#cfe;padding:6px 10px;border-radius:6px;cursor:pointer;font-family:inherit;font-size:11px}
button:hover{background:#243050}
.grid{display:grid;grid-template-columns:300px 1fr 320px;gap:10px}
.card{background:#0f1219;border:1px solid #1e2330;border-radius:8px;padding:10px}
.card h2{font-size:11px;font-weight:500;color:#8fa;letter-spacing:1px;margin-bottom:8px;text-transform:uppercase}
.g{margin:5px 0}.g .lab{display:flex;justify-content:space-between;font-size:11px;color:#aab}
.track{height:8px;background:#171b24;border-radius:4px;overflow:hidden;margin-top:2px}
.fill{height:100%;border-radius:4px;transition:width .3s}
.neu{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:11px}
.neu .nb{height:9px;border-radius:3px;background:#7f77dd}
.neu .nl{color:#c9c4f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.neu .nx{color:#6b7180;font-size:10px}
#trace div{font-size:11px;color:#bcd;margin:2px 0;padding-left:10px;border-left:2px solid #2a3550;line-height:1.4}
canvas{width:100%;height:340px;background:#0a0c11;border-radius:6px}
.stat{display:flex;justify-content:space-between;font-size:11px;color:#aab;margin:2px 0}
.stat b{color:#9fe1cb;font-weight:500}
.ans{font-size:13px;color:#fff;margin:4px 0}.qn{font-size:11px;color:#7a8290}
.chips button{font-size:10px}
#per img{width:100%;border-radius:6px;margin-top:6px;max-height:150px;object-fit:cover}
.tag{display:inline-block;background:#12251d;color:#5dcaa5;border:1px solid #1d9e75;border-radius:10px;padding:1px 7px;font-size:10px;margin:2px 2px 0 0}
</style></head><body>
<h1>◉ IARA · OBSERVATÓRIO</h1><div class=sub>o cérebro pensando ao vivo — neurônios · hormônios · grafo · raciocínio</div>
<div class=bar>
<input id=q placeholder="pergunte: capital da França? · língua do Japão? · capital de Wakanda?" onkeydown="if(event.key=='Enter')ask()">
<button onclick=ask()>perguntar</button>
</div>
<div class="bar chips">
<button onclick="q.value='capital da França?';ask()">capital França</button>
<button onclick="q.value='língua do Japão?';ask()">língua Japão</button>
<button onclick="q.value='capital do Egito?';ask()">capital Egito</button>
<button onclick="q.value='capital de Wakanda?';ask()">Wakanda (fake)</button>
<button onclick="see('eiffel')">👁 ver Eiffel</button>
<button onclick="see('dog')">👁 ver cão</button>
<button onclick="see('car')">👁 ver carro</button>
</div>
<div class=grid>
 <div>
  <div class=card><h2>hormônios</h2><div id=horm></div><canvas id=spark height=70 style="height:70px"></canvas></div>
  <div class=card style=margin-top:10px><h2>uso</h2><div id=stats></div></div>
  <div class=card style=margin-top:10px id=per><h2>percepção</h2><div id=perc></div></div>
 </div>
 <div class=card><h2>grafo do conhecimento (cresce vivo)</h2><canvas id=graph height=340></canvas>
   <div class=qn id=qn></div><div class=ans id=ans></div></div>
 <div>
  <div class=card><h2>neurônios ativos (camada · conceito)</h2><div id=neu></div></div>
  <div class=card style=margin-top:10px><h2>raciocínio (trace)</h2><div id=trace></div></div>
 </div>
</div>
<script>
const HC={da:['#63c923','dopamina'],cort:['#e24b4a','cortisol'],ne:['#ef9f27','noradrenalina'],e:['#378add','energia']};
let ST=null;
async function ask(){const v=document.getElementById('q').value;if(!v)return;ST=await(await fetch('/ask',{method:'POST',body:JSON.stringify({q:v})})).json();render()}
async function see(n){ST=await(await fetch('/see',{method:'POST',body:JSON.stringify({img:n})})).json();render()}
async function poll(){try{const s=await(await fetch('/state')).json();if(!ST)ST=s;else{ST.hormones=s.hormones;ST.history=s.history}render()}catch(e){}}
function render(){if(!ST)return;
 let h='';for(const k in HC){const val=ST.hormones[k],c=HC[k];h+=`<div class=g><div class=lab><span>${c[1]}</span><span>${val.toFixed(2)}</span></div><div class=track><div class=fill style="width:${val*100}%;background:${c[0]}"></div></div></div>`}
 document.getElementById('horm').innerHTML=h; drawSpark();
 let n='';for(const x of ST.neurons){n+=`<div class=neu><span class=nx>L${x.layer}</span><span class=nl>${x.concept}</span><span class=nb style="width:${Math.max(6,x.act*90)}px"></span></div>`}
 document.getElementById('neu').innerHTML=n||'<span class=nx>—</span>';
 document.getElementById('trace').innerHTML=(ST.trace||[]).map(t=>`<div>${t}</div>`).join('')||'<div>—</div>';
 const s=ST.stats;document.getElementById('stats').innerHTML=`<div class=stat><span>fatos no grafo</span><b>${s.facts}</b></div><div class=stat><span>associações</span><b>${s.assoc}</b></div><div class=stat><span>entidades percebidas</span><b>${s.perceived}</b></div><div class=stat><span>aprendidos</span><b>${s.commit}</b></div><div class=stat><span>reuso</span><b>${s.reuse}</b></div><div class=stat><span>abstenções</span><b>${s.abstain}</b></div><div class=stat><span>latência</span><b>${ST.ms} ms</b></div>`;
 document.getElementById('qn').textContent=ST.q||'';document.getElementById('ans').textContent=ST.a||'';
 let pc=(ST.perception.concepts||[]).map(c=>`<span class=tag>${c[0]} ${(c[1]*100|0)}%</span>`).join('');
 document.getElementById('perc').innerHTML=(ST.perception.img?`<img src="/img/${ST.perception.img}">`:'')+`<div style=margin-top:4px>${pc||'<span class=nx>nada visto ainda</span>'}</div>`;
 drawGraph();}
function drawSpark(){const c=document.getElementById('spark'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,H=70;x.clearRect(0,0,W,H);const hh=ST.history||[];for(const k in HC){x.beginPath();x.strokeStyle=HC[k][0];x.lineWidth=1.3;hh.forEach((p,i)=>{const px=i/(hh.length-1)*W,py=H-p[k]*H*0.9-3;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}}
let G={nodes:[],edges:[],pos:{}};
function drawGraph(){const g=ST.graph;const c=document.getElementById('graph'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,H=340;
 const ids=g.nodes.map(n=>n.id);for(const n of g.nodes)if(!G.pos[n.id])G.pos[n.id]={x:W/2+(Math.random()-.5)*200,y:H/2+(Math.random()-.5)*160,vx:0,vy:0,t:n.type};
 for(const id in G.pos)if(!ids.includes(id))delete G.pos[id];
 for(let it=0;it<3;it++){const P=G.pos;for(const a in P)for(const b in P){if(a>=b)continue;let dx=P[a].x-P[b].x,dy=P[a].y-P[b].y,d=Math.hypot(dx,dy)||1;let f=1400/(d*d);P[a].vx+=dx/d*f;P[a].vy+=dy/d*f;P[b].vx-=dx/d*f;P[b].vy-=dy/d*f}
  for(const e of g.edges){const a=P[e.a],b=P[e.b];if(!a||!b)continue;let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,f=(d-70)*0.01;a.vx+=dx/d*f;a.vy+=dy/d*f;b.vx-=dx/d*f;b.vy-=dy/d*f}
  for(const id in P){const p=P[id];p.vx+=(W/2-p.x)*0.002;p.vy+=(H/2-p.y)*0.002;p.x+=p.vx*.5;p.y+=p.vy*.5;p.vx*=.8;p.vy*=.8;p.x=Math.max(12,Math.min(W-12,p.x));p.y=Math.max(12,Math.min(H-12,p.y))}}
 x.clearRect(0,0,W,H);x.strokeStyle='#243043';x.lineWidth=1;for(const e of g.edges){const a=G.pos[e.a],b=G.pos[e.b];if(!a||!b)continue;x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}
 const col={entity:'#378add',value:'#63c923',assoc:'#7f77dd'};for(const n of g.nodes){const p=G.pos[n.id];x.beginPath();x.fillStyle=col[n.type]||'#889';x.arc(p.x,p.y,n.type=='entity'?6:4,0,7);x.fill();x.fillStyle='#cdd6e6';x.font='10px monospace';x.fillText(n.id.slice(0,12),p.x+7,p.y+3)}}
setInterval(poll,500);poll();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def _send(self,code,body, ctype="application/json"):
        b=body if isinstance(body,bytes) else body.encode()
        self.send_response(code); self.send_header("Content-Type",ctype); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path=="/": return self._send(200,PAGE,"text/html; charset=utf-8")
        if self.path=="/state": return self._send(200,json.dumps(MIND.state()))
        if self.path.startswith("/img/"):
            n=self.path.split("/img/")[1]; p=IMGS.get(n)
            if p and os.path.exists(p): return self._send(200,open(p,'rb').read(),"image/jpeg")
            return self._send(404,b"no")
        self._send(404,b"no")
    def do_POST(self):
        n=int(self.headers.get('Content-Length',0)); d=json.loads(self.rfile.read(n) or b"{}")
        if self.path=="/ask": return self._send(200,json.dumps(MIND.ask(d.get("q",""))))
        if self.path=="/see": return self._send(200,json.dumps(MIND.see(d.get("img",""))))
        self._send(404,b"no")

if __name__=="__main__":
    MIND=Mind()
    def decayer():
        while True: time.sleep(0.5); MIND.decay()
    threading.Thread(target=decayer,daemon=True).start()
    port=3030
    print(f"\n  ►►►  IARA observatório em  http://localhost:{port}   (Ctrl-C p/ sair)\n",flush=True)
    ThreadingHTTPServer(("0.0.0.0",port),H).serve_forever()
