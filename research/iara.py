#!/usr/bin/env python3
"""IARA — o cérebro unificado: pesos do Qwen (grafo navegável) + conversa que RACIOCINA + olho (imagem) (2026-07-12).

Tudo junto num painel só (o que o Leonardo pediu, sem espalhar):
  CÉREBRO   grafo interativo dos pesos do Qwen-3B (clica/expande/zoom); glitch tokens filtrados.
  CONVERSA  ela responde raciocinando (usa os neurônios do Qwen + o que você ensina). Auto-suficiente.
  OLHO      arrasta/envia uma IMAGEM → CLIP vê → ela raciocina sobre o que é → entra no grafo.
  OBSERVA   hormônios (dopamina/curiosidade), o que ela está pensando/vendo, o que aprendeu.
Cada conversa/imagem ACENDE conceitos no grafo do cérebro. Porta 3050. (webcam/fala existem em iara_face/
iara_voice — plugam no mesmo barramento; aqui foco no cérebro+visão+conversa que é o núcleo.)"""
import os,re,json,time,threading,base64,io,math,unicodedata
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","10.3.0"); os.environ.setdefault("ROCR_VISIBLE_DEVICES","0"); os.environ.setdefault("OMP_NUM_THREADS","4")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
MODEL="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-3B-Instruct"
DEEP=10
SEED=["France","Brazil","Japan","Germany","China","Paris","Tokyo","Europe","Asia","Africa","football","music",
 "science","water","fire","dog","cat","king","queen","computer","brain","love","war","sun","moon","ocean",
 "gravity","atom","cell","virus","language","India","Russia","math","art","history","robot","energy","light","life"]
CLIP_VOCAB=["a dog","a cat","a person","a man","a woman","a car","a bird","a horse","a house","a tree","a flower",
 "a mountain","a beach","the sky","a city","food","a computer","a phone","a book","a chair","a table","a boat",
 "an airplane","a bicycle","the Eiffel Tower","a cup of coffee","a guitar","a clock","money","a fish","a robot"]
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()

class Iara:
    def __init__(s):
        s.ready=False; s.thinking=False; s.lock=threading.Lock()
        s.chat=[]; s.learned={}; s.dop=0.2; s.cur=0.3; s.seen=None
        threading.Thread(target=s._load,daemon=True).start()
    def _load(s):
        t=time.time(); print("[carregando cérebro Qwen-3B + olho CLIP…]",flush=True)
        s.tok=AutoTokenizer.from_pretrained(MODEL)
        m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to("cuda").eval()
        s.model=m; s.E=m.model.embed_tokens.weight.detach(); s.nw=m.model.norm.weight.detach(); s.Et=s.E.t().contiguous()
        V=s.E.shape[0]; toks=s.tok.convert_ids_to_tokens(list(range(V)))
        raw=[(i,(toks[i] or "").replace("Ġ","")) for i in range(V) if re.fullmatch(r"Ġ?[A-Za-z][a-zA-Z]{2,}",toks[i] or "")]
        norms=s.E[[i for i,_ in raw]].float().norm(dim=1); med=float(norms.median())
        s.clean=[]; s.words=[]
        for (i,w),nm in zip(raw,norms.tolist()):
            if re.search(r"[a-z][A-Z]",w): continue                # tira camelCase (tokens de código)
            if not (0.55*med<nm<1.8*med): continue                 # tira glitch tokens (norma anômala)
            s.clean.append(i); s.words.append(w)
        s.wmap={w.lower():s.clean[j] for j,w in enumerate(s.words)}
        s.Ec=torch.nn.functional.normalize(s.E[s.clean].float(),dim=1)
        NL=m.config.num_hidden_layers; INT=m.config.intermediate_size; vals=[]; s.idx=[]
        for L in range(NL-DEEP,NL):
            dp=m.model.layers[L].mlp.down_proj.weight.detach(); vals.append((dp.t()*s.nw).contiguous()); s.idx+=[(L,i) for i in range(INT)]
        s.vals=torch.cat(vals,0)
        s.clip=CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda").eval()
        s.cproc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        s.ready=True; print(f"[IARA pronta em {time.time()-t:.0f}s · {len(s.clean)} conceitos limpos · {s.vals.shape[0]} neurônios]",flush=True)
    # ---------- grafo dos pesos ----------
    def _wid(s,w):
        if w.lower() in s.wmap: return s.wmap[w.lower()]
        ids=s.tok.encode(" "+w,add_special_tokens=False); return ids[0] if ids else None
    @torch.no_grad()
    def neighbors(s,word,k=8):
        tid=s._wid(word)
        if tid is None: return []
        v=torch.nn.functional.normalize(s.E[tid].float(),dim=0); sims=s.Ec@v; top=torch.topk(sims,k+8).indices.tolist(); out=[]
        for j in top:
            w=s.words[j]
            if w.lower()!=word.lower() and w.lower() not in [o[0].lower() for o in out]: out.append((w,round(float(sims[j]),3)))
            if len(out)>=k: break
        return out
    @torch.no_grad()
    def neurons_for(s,word,k=8):
        tid=s._wid(word)
        if tid is None: return []
        score=s.vals@s.E[tid].to(s.vals.dtype); top=torch.topk(score,k).indices.tolist(); out=[]
        for j in top:
            L,i=s.idx[j]; wl=s.vals[j].to(s.Et.dtype)@s.Et; wt=torch.topk(wl,6).indices.tolist()
            wr=[s.tok.convert_ids_to_tokens([t])[0].replace("Ġ","") for t in wt]; wr=[w for w in wr if re.fullmatch(r"[A-Za-z][a-zA-Z]{1,}",w)][:4]
            out.append(dict(id=f"L{L}·n{i}",layer=L,idx=i,forca=round(float(score[j]),1),escreve=wr))
        return out
    def seed_graph(s):
        nodes=[dict(id=w,label=w,group="learned" if nrm(w) in s.learned else "seed") for w in SEED]; edges=[]; seen=set()
        for w in SEED:
            for nb,sim in s.neighbors(w,3):
                key=tuple(sorted([w,nb]))
                if nb in SEED and key not in seen and sim>0.30: edges.append({"from":w,"to":nb}); seen.add(key)
        return dict(nodes=nodes,edges=edges)
    def expand(s,word):
        nodes=[dict(id=word,label=word,group="focus")]; edges=[]
        for nb,sim in s.neighbors(word,8): nodes.append(dict(id=nb,label=nb,group="viz")); edges.append({"from":word,"to":nb,"value":sim})
        return dict(nodes=nodes,edges=edges)
    # ---------- conversa (raciocínio) ----------
    @torch.no_grad()
    def _gen(s,messages,n=64):
        ids=s.tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to("cuda")
        out=s.model.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=s.tok.eos_token_id)
        return s.tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
    def _reason(s,q):
        facts=[v["stmt"] for k,v in s.learned.items() if k in nrm(q) or any(w in nrm(q) for w in k.split() if len(w)>=4)][:6]
        sysmsg="Você é a IARA, uma mente que raciocina. Responda em 1 frase curta e direta. Use os fatos abaixo e o que você sabe do mundo. Se não souber, diga 'não sei'."
        u=(("Você já aprendeu:\n- "+"\n- ".join(facts)+"\n\n") if facts else "")+f"Pergunta: {q}"
        a=s._gen([{"role":"system","content":sysmsg},{"role":"user","content":u}]).split("\n")[0].strip()
        return None if (not a or "não sei" in a.lower() or nrm(a)=="nao sei") else a
    def _key_concept(s,text):
        for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}",text):
            if nrm(w) in s.wmap or w.lower() in s.wmap: return w
        return None
    def ask(s,q):
        with s.lock: s.chat.append(("voce",q))
        st=re.match(r"^(.{2,60}?)\s+(?:é|eh|significa)\s+(.{2,})$",q.strip().rstrip("."),re.I)
        if st and "?" not in q:
            c,a=st.group(1).strip(),st.group(2).strip()
            with s.lock: s.learned[nrm(c)]={"v":a,"stmt":f"{c} é {a}"}; s.dop=min(1,s.dop+0.4); s.chat.append(("iara",f"Entendi! {c} é {a}. (guardei)"))
            return dict(say=f"Entendi! {c} é {a}.",concept=c)
        if not s.ready:
            with s.lock: s.chat.append(("iara","acordando os neurônios, um instante…"))
            return dict(say="acordando os neurônios…")
        s.thinking=True; a=s._reason(q); s.thinking=False
        with s.lock:
            if a: s.dop=min(1,s.dop+0.35); s.chat.append(("iara",a)); concept=s._key_concept(q) or s._key_concept(a)
            else: s.cur=min(1,s.cur+0.2); s.chat.append(("iara","não sei isso 🤔 me ensina?")); concept=None
        return dict(say=a or "não sei",concept=concept)
    # ---------- olho ----------
    @torch.no_grad()
    def see(s,imgdata):
        if not s.ready: return dict(say="acordando…")
        s.thinking=True
        try:
            raw=base64.b64decode(imgdata.split(",")[-1]); img=Image.open(io.BytesIO(raw)).convert("RGB")
            inp=s.cproc(text=[f"a photo of {v}" for v in CLIP_VOCAB],images=img,return_tensors="pt",padding=True).to("cuda")
            pr=s.clip(**inp).logits_per_image.softmax(-1)[0]; top=torch.topk(pr,3)
            concepts=[(CLIP_VOCAB[int(i)].replace("a photo of ","").replace("a ","").replace("an ","").replace("the ",""),float(p)) for p,i in zip(top.values,top.indices)]
            lab=concepts[0][0]
            a=s._gen([{"role":"system","content":"Você é a IARA. Você está VENDO uma imagem. Em 1 frase curta, diga o que é e algo que você sabe sobre isso."},
                      {"role":"user","content":f"Na imagem eu reconheço: {', '.join(c for c,_ in concepts)}. O que você vê?"}]).split("\n")[0].strip()
        except Exception as e: s.thinking=False; return dict(say=f"não consegui ver a imagem ({str(e)[:40]})")
        s.thinking=False
        with s.lock: s.dop=min(1,s.dop+0.5); s.seen=[(c,round(p,2)) for c,p in concepts]; s.chat.append(("iara","👁 "+a))
        return dict(say=a,concept=s._key_concept(lab) or lab.split()[-1])
    def state(s):
        return dict(ready=s.ready,thinking=s.thinking,conceitos=len(getattr(s,'clean',[])),neuronios=int(getattr(s,'vals',torch.zeros(0)).shape[0]) if s.ready else 0,
            chat=s.chat[-14:],dop=round(s.dop,2),cur=round(s.cur,2),seen=s.seen,aprendeu=[v["stmt"] for v in s.learned.values()][-8:])
    def tick(s): s.dop=0.2+(s.dop-0.2)*0.9; s.cur=0.3+(s.cur-0.3)*0.95

I=Iara()
PAGE=r"""<!doctype html><html lang=pt><head><meta charset=utf8><title>IARA</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,monospace}body{background:#090b10;color:#e8ebf0;overflow:hidden}
#bar{position:fixed;top:0;left:0;right:0;height:46px;display:flex;gap:8px;align-items:center;padding:0 12px;background:#0f1219;border-bottom:1px solid #1e2330;z-index:10}
h1{font-size:14px;color:#9fe1cb;letter-spacing:2px}#stt{color:#7a8290;font-size:11px}
input,button{font-family:inherit}#s{background:#12151d;border:1px solid #2a3550;color:#e8ebf0;padding:6px 10px;border-radius:6px;font-size:13px;width:200px}
button{background:#12251d;border:1px solid #1d9e75;color:#9fe1cb;padding:6px 10px;border-radius:6px;cursor:pointer;font-size:12px}button:hover{background:#173a2b}
#net{position:fixed;top:46px;left:360px;right:0;bottom:0}
#L{position:fixed;top:46px;left:0;width:360px;bottom:0;background:#0d1016;border-right:1px solid #1e2330;display:flex;flex-direction:column}
.card{border-bottom:1px solid #1a1f2a;padding:10px}.card h2{font-size:10px;color:#7fb59f;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase}
#chat{flex:1;overflow:auto;padding:10px;display:flex;flex-direction:column}#chat .m{margin:4px 0;font-size:12.5px;line-height:1.4}.voce{color:#8fb5d8}.iara{color:#9fe1cb}
#ci{display:flex;gap:6px;padding:8px;border-top:1px solid #1a1f2a}#qi{flex:1;background:#12151d;border:1px solid #2a3550;color:#e8ebf0;padding:8px;border-radius:6px;font-size:13px}
#drop{border:1px dashed #2a3550;border-radius:6px;padding:10px;text-align:center;color:#7a8290;font-size:11px;cursor:pointer}#drop.hot{border-color:#5dcaa5;color:#9fe1cb}
.h{display:flex;justify-content:space-between;font-size:11px;color:#aab;margin:2px 0}.tk{height:6px;background:#171b24;border-radius:3px;margin-top:1px;overflow:hidden}.tf{height:100%}
.tag{display:inline-block;background:#12251d;color:#7fe0bf;border:1px solid #1d9e75;border-radius:6px;padding:1px 6px;font-size:10px;margin:2px}
.neu{background:#0f1219;border-left:3px solid #1d9e75;border-radius:4px;padding:5px 7px;margin:4px 0;font-size:11px;cursor:pointer}.neu b{color:#7fe0bf}.neu .w{color:#c9c4f0}
#det{position:fixed;bottom:0;right:0;width:300px;max-height:46%;overflow:auto;background:#0d1016;border-left:1px solid #1e2330;border-top:1px solid #1e2330;padding:10px;display:none}
</style></head><body>
<div id=bar><h1>◉ IARA</h1><input id=s placeholder="buscar no cérebro (France…)" onkeydown="if(event.key=='Enter')go()"><button onclick=go()>explorar</button><button onclick=seed()>cérebro</button><button onclick=fit()>ajustar</button><span id=stt>acordando…</span></div>
<div id=L>
 <div class=card><h2>observação</h2><div id=obs></div></div>
 <div class=card><div id=drop onclick="document.getElementById('f').click()">📷 arraste uma imagem aqui (ou clique) — ela VÊ</div><input type=file id=f accept=image/* style=display:none onchange=upl()></div>
 <div id=chat></div>
 <div id=ci><input id=qi placeholder="fale com ela…" onkeydown="if(event.key=='Enter')ask()"><button onclick=ask()>enviar</button></div>
</div>
<div id=net></div>
<div id=det></div>
<script>
let net,nodes,edges;
const opt={nodes:{shape:'dot',size:14,font:{color:'#dfe6f0',size:13,face:'monospace'},borderWidth:2},edges:{color:{color:'#2a3550',highlight:'#5dcaa5'},smooth:{type:'continuous'},width:.6},
 groups:{seed:{color:{background:'#1d9e75',border:'#5dcaa5'}},viz:{color:{background:'#3c3489',border:'#7f77dd'}},focus:{color:{background:'#993c1d',border:'#d85a30'},size:22},learned:{color:{background:'#ba7517',border:'#ef9f27'}},neuron:{color:{background:'#0f6e56',border:'#1d9e75'},shape:'diamond'}},
 physics:{barnesHut:{gravitationalConstant:-6000,springLength:120},stabilization:{iterations:100}},interaction:{hover:true}};
async function J(u){return (await fetch(u)).json()}
async function Po(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}
function addg(g,focus){g.nodes.forEach(n=>{if(!nodes.get(n.id))nodes.add(n);else if(n.group=='focus'||n.group=='learned')nodes.update(n)});g.edges.forEach(e=>{const id=e.from+'|'+e.to;if(!edges.get(id))edges.add({id,...e})});if(focus){net.selectNodes([focus]);net.focus(focus,{scale:1.0,animation:true})}}
async function boot(){const st=await J('/state');document.getElementById('stt').textContent=st.thinking?'🧠 pensando…':(st.ready?`cérebro online · ${st.conceitos} conceitos · ${st.neuronios} neurônios`:'acordando os neurônios…');obs(st);
 if(!net&&st.ready){nodes=new vis.DataSet();edges=new vis.DataSet();net=new vis.Network(document.getElementById('net'),{nodes,edges},opt);net.on('click',p=>{if(p.nodes.length){const n=nodes.get(p.nodes[0]);if(n&&n.group=='neuron')return;expand(p.nodes[0]);details(p.nodes[0])}});seed()}
 setTimeout(boot,900)}
function obs(st){document.getElementById('obs').innerHTML=`<div class=h><span>dopamina</span><span>${st.dop}</span></div><div class=tk><div class=tf style="width:${st.dop*100}%;background:#63c923"></div></div><div class=h><span>curiosidade</span><span>${st.cur}</span></div><div class=tk><div class=tf style="width:${st.cur*100}%;background:#ef9f27"></div></div>`+(st.seen?`<div style=margin-top:6px><span style=color:#7a8290;font-size:10px>vendo agora:</span> ${st.seen.map(c=>`<span class=tag>${c[0]} ${(c[1]*100|0)}%</span>`).join('')}</div>`:'');
 document.getElementById('chat').innerHTML=(st.chat||[]).map(m=>`<div class="m ${m[0]}">${m[0]=='voce'?'você: ':'IARA: '}${m[1]}</div>`).join('');const ch=document.getElementById('chat');ch.scrollTop=ch.scrollHeight}
async function seed(){const g=await J('/seed');nodes.clear();edges.clear();addg(g);setTimeout(fit,400)}
async function expand(w){addg(await J('/expand?w='+encodeURIComponent(w)))}
async function go(){const w=document.getElementById('s').value.trim();if(w){await expand(w);details(w);net.selectNodes([w]);net.focus(w,{scale:1.1,animation:true})}}
function fit(){net&&net.fit({animation:true})}
async function ask(){const v=document.getElementById('qi').value.trim();if(!v)return;document.getElementById('qi').value='';const r=await Po('/ask',{q:v});boot();if(r.concept&&net){await expand(r.concept);addg({nodes:[{id:r.concept,label:r.concept,group:'focus'}],edges:[]},r.concept)}}
async function upl(){const f=document.getElementById('f').files[0];if(!f)return;const rd=new FileReader();rd.onload=async()=>{const r=await Po('/see',{img:rd.result});boot();if(r.concept&&net){await expand(r.concept);addg({nodes:[{id:r.concept,label:r.concept,group:'focus'}],edges:[]},r.concept)}};rd.readAsDataURL(f)}
async function details(w){const d=await J('/neurons?w='+encodeURIComponent(w));const el=document.getElementById('det');el.style.display='block';
 el.innerHTML=`<h2 style="font-size:12px;color:#9fe1cb">${w} — neurônios que sabem isso</h2>`+(d.neurons||[]).map(n=>`<div class=neu onclick='addN(${JSON.stringify(n)})'><b>${n.id}</b> f${n.forca} <span class=w>${n.escreve.join('·')}</span></div>`).join('')}
function addN(n){if(!nodes.get(n.id))nodes.add({id:n.id,label:n.id,group:'neuron'});n.escreve.forEach(t=>{if(!nodes.get(t))nodes.add({id:t,label:t,group:'viz'});const id=n.id+'|'+t;if(!edges.get(id))edges.add({id,from:n.id,to:t})})}
const dp=document.getElementById('drop');['dragover','dragenter'].forEach(e=>dp.addEventListener(e,ev=>{ev.preventDefault();dp.classList.add('hot')}));['dragleave','drop'].forEach(e=>dp.addEventListener(e,ev=>{ev.preventDefault();dp.classList.remove('hot')}));
dp.addEventListener('drop',ev=>{const f=ev.dataTransfer.files[0];if(f){const rd=new FileReader();rd.onload=async()=>{const r=await Po('/see',{img:rd.result});boot();if(r.concept&&net)expand(r.concept)};rd.readAsDataURL(f)}});
boot();
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s,o): b=json.dumps(o,ensure_ascii=False).encode(); s.send_response(200); s.send_header("Content-Type","application/json; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        u=urlparse(s.path); q=parse_qs(u.query)
        if u.path=="/": b=PAGE.encode(); s.send_response(200); s.send_header("Content-Type","text/html; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b); return
        if u.path=="/state": return s._j(I.state())
        if not I.ready: return s._j(dict(nodes=[],edges=[],neurons=[]))
        if u.path=="/seed": return s._j(I.seed_graph())
        if u.path=="/expand": return s._j(I.expand(q.get("w",["France"])[0]))
        if u.path=="/neurons": return s._j(dict(neurons=I.neurons_for(q.get("w",["France"])[0])))
        s._j({"ok":1})
    def do_POST(s):
        n=int(s.headers.get('Content-Length',0)); d=json.loads(s.rfile.read(n) or b"{}")
        if s.path=="/ask": r=I.ask(d.get("q",""))
        elif s.path=="/see": r=I.see(d.get("img",""))
        else: r={"err":1}
        s._j(r)
def relax():
    while True: time.sleep(4); I.tick()
if __name__=="__main__":
    threading.Thread(target=relax,daemon=True).start()
    print("  ►►►  IARA em http://localhost:3050",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
