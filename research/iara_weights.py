#!/usr/bin/env python3
"""IARA · EXPLORADOR DE PESOS — o grafo INTERATIVO dos neurônios do Qwen-3B (2026-07-12).

O pedido do Leonardo: pegar os pesos do modelo inteligente e virar um grafo NAVEGÁVEL/CLICÁVEL, já
pré-carregado. Clica em 'França' → vê onde ela se relaciona (vizinhos no espaço de embedding = o que o
modelo associa). Clica num nó → painel mostra os NEURÔNIOS que 'sabem' aquilo (quais neurônios ESCREVEM
aquele conceito, em que camada) + o que cada neurônio escreve. Arrasta, dá zoom, expande. Tudo dos pesos
reais (embed_tokens + down_proj, logit-lens — como no WS22/neuron_atlas). Porta 3050."""
import os,re,json,time,threading
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","10.3.0"); os.environ.setdefault("ROCR_VISIBLE_DEVICES","0"); os.environ.setdefault("OMP_NUM_THREADS","4")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-3B-Instruct"
SEED=["France","Brazil","Japan","Germany","China","Egypt","Paris","Tokyo","Berlin","Rome","Europe","Asia",
 "Africa","football","music","science","water","fire","dog","cat","king","queen","computer","brain",
 "love","war","money","sun","moon","ocean","mountain","gravity","atom","cell","virus","language",
 "Portuguese","English","India","Russia","math","art","history","robot","energy","light","death","life"]
DEEP=10

class Weights:
    def __init__(s):
        s.ready=False; threading.Thread(target=s._load,daemon=True).start()
    def _load(s):
        t=time.time(); print("[carregando pesos do Qwen-3B…]",flush=True)
        s.tok=AutoTokenizer.from_pretrained(MODEL)
        m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to("cuda").eval()
        s.E=m.model.embed_tokens.weight.detach()                       # [V,H]
        s.nw=m.model.norm.weight.detach()
        V=s.E.shape[0]; toks=s.tok.convert_ids_to_tokens(list(range(V)))
        s.clean=[i for i,t in enumerate(toks) if re.fullmatch(r"Ġ?[A-Za-z][a-zA-Z]{2,}",t or "")]
        s.words=[toks[i].replace("Ġ","") for i in s.clean]
        s.wmap={w.lower():s.clean[j] for j,w in enumerate(s.words)}
        s.Ec=torch.nn.functional.normalize(s.E[s.clean].float(),dim=1)  # [Nc,H] normalizado
        NL=m.config.num_hidden_layers; INT=m.config.intermediate_size
        vals=[]; s.idx=[]
        for L in range(NL-DEEP,NL):
            dp=m.model.layers[L].mlp.down_proj.weight.detach()         # [H,INT]
            vals.append((dp.t()*s.nw).contiguous())                    # [INT,H] valor de cada neurônio
            s.idx+=[(L,i) for i in range(INT)]
        s.vals=torch.cat(vals,0)                                       # [Ndeep,H]
        s.Et=s.E.t().contiguous()
        s.model=m; s.ready=True
        print(f"[pesos prontos em {time.time()-t:.0f}s · {len(s.clean)} conceitos · {s.vals.shape[0]} neurônios profundos]",flush=True)
    def _wid(s,w):
        if w.lower() in s.wmap: return s.wmap[w.lower()]
        ids=s.tok.encode(" "+w,add_special_tokens=False); return ids[0] if ids else None
    @torch.no_grad()
    def neighbors(s,word,k=8):
        tid=s._wid(word)
        if tid is None: return []
        v=torch.nn.functional.normalize(s.E[tid].float(),dim=0)
        sims=s.Ec@v; top=torch.topk(sims,k+6).indices.tolist()
        out=[]
        for j in top:
            w=s.words[j]
            if w.lower()!=word.lower() and w.lower() not in [o[0].lower() for o in out]:
                out.append((w,round(float(sims[j]),3)))
            if len(out)>=k: break
        return out
    @torch.no_grad()
    def neurons_for(s,word,k=8):
        tid=s._wid(word)
        if tid is None: return []
        score=s.vals@s.E[tid].to(s.vals.dtype)                         # quanto cada neurônio ESCREVE a palavra
        top=torch.topk(score,k).indices.tolist(); out=[]
        for j in top:
            L,i=s.idx[j]; val=s.vals[j]
            wl=(val.to(s.Et.dtype)@s.Et); wt=torch.topk(wl,5).indices.tolist()
            writes=[s.tok.convert_ids_to_tokens([t])[0].replace("Ġ","") for t in wt]
            writes=[w for w in writes if re.fullmatch(r"[A-Za-z][a-zA-Z]{1,}",w)][:4]
            out.append(dict(id=f"L{L}·n{i}",layer=L,idx=i,forca=round(float(score[j]),1),escreve=writes))
        return out
    def seed_graph(s):
        nodes=[dict(id=w,label=w,group="seed") for w in SEED]; edges=[]; seen=set()
        for w in SEED:
            for nb,sim in s.neighbors(w,4):
                key=tuple(sorted([w,nb]))
                if nb in SEED and key not in seen and sim>0.30: edges.append(dict(**{"from":w,"to":nb})); seen.add(key)
        return dict(nodes=nodes,edges=edges)
    def expand(s,word):
        nodes=[dict(id=word,label=word,group="focus")]; edges=[]
        for nb,sim in s.neighbors(word,8):
            nodes.append(dict(id=nb,label=nb,group="viz")); edges.append(dict(**{"from":word,"to":nb,"value":sim}))
        return dict(nodes=nodes,edges=edges)

W=Weights()
PAGE=r"""<!doctype html><html lang=pt><head><meta charset=utf8><title>IARA · Explorador de Pesos</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,monospace}
body{background:#090b10;color:#e8ebf0;overflow:hidden}
#bar{position:fixed;top:0;left:0;right:0;height:48px;display:flex;gap:8px;align-items:center;padding:0 12px;background:#0f1219;border-bottom:1px solid #1e2330;z-index:10}
h1{font-size:14px;color:#9fe1cb;letter-spacing:2px}#stt{color:#7a8290;font-size:11px}
input{background:#12151d;border:1px solid #2a3550;color:#e8ebf0;padding:7px 10px;border-radius:6px;font-family:inherit;font-size:13px;width:220px}
button{background:#12251d;border:1px solid #1d9e75;color:#9fe1cb;padding:7px 11px;border-radius:6px;cursor:pointer;font-family:inherit;font-size:12px}button:hover{background:#173a2b}
#net{position:fixed;top:48px;left:0;right:340px;bottom:0}
#side{position:fixed;top:48px;right:0;width:340px;bottom:0;background:#0d1016;border-left:1px solid #1e2330;padding:12px;overflow:auto}
#side h2{font-size:12px;color:#9fe1cb;letter-spacing:1px;margin:4px 0 8px}#side .sub{color:#7a8290;font-size:11px;margin-bottom:10px}
.neu{background:#0f1219;border:1px solid #223;border-left:3px solid #1d9e75;border-radius:5px;padding:6px 8px;margin:5px 0;font-size:12px;cursor:pointer}
.neu:hover{background:#152030}.neu b{color:#7fe0bf}.neu .w{color:#c9c4f0;font-size:11px}
.viz{display:inline-block;background:#1a1226;color:#d3a0e8;border:1px solid #534ab7;border-radius:10px;padding:2px 8px;margin:2px;font-size:11px;cursor:pointer}
.hint{color:#5a636f;font-size:11px;margin-top:14px;line-height:1.5}
</style></head><body>
<div id=bar><h1>◉ IARA · EXPLORADOR DE PESOS</h1>
 <input id=q placeholder="buscar conceito (ex: France, brain, gravity)" onkeydown="if(event.key=='Enter')go()">
 <button onclick=go()>explorar</button><button onclick=seed()>semente</button><button onclick=fit()>ajustar</button>
 <span id=stt>carregando os pesos…</span></div>
<div id=net></div>
<div id=side><h2 id=sh>clique num conceito</h2><div class=sub id=ssub>o grafo vem dos pesos reais do Qwen-3B</div>
 <div id=sneu></div><div id=sviz></div>
 <div class=hint>• <b>clique</b> num nó → expande os vizinhos (o que o modelo associa)<br>• o painel mostra os <b>neurônios</b> que escrevem aquele conceito e o que cada um escreve<br>• clique num neurônio ou num vizinho pra puxar mais<br>• arraste os nós, dê zoom com o scroll</div></div>
<script>
let net,nodes,edges;
const opt={nodes:{shape:'dot',size:14,font:{color:'#dfe6f0',size:13,face:'monospace'},borderWidth:2},
 edges:{color:{color:'#2a3550',highlight:'#5dcaa5'},smooth:{type:'continuous'},width:0.6},
 groups:{seed:{color:{background:'#1d9e75',border:'#5dcaa5'}},viz:{color:{background:'#3c3489',border:'#7f77dd'}},focus:{color:{background:'#993c1d',border:'#d85a30'},size:22},neuron:{color:{background:'#0f6e56',border:'#1d9e75'},shape:'diamond'}},
 physics:{barnesHut:{gravitationalConstant:-6000,springLength:120,springConstant:.03},stabilization:{iterations:120}},
 interaction:{hover:true,tooltipDelay:120}};
async function J(u){return (await fetch(u)).json()}
async function boot(){const st=await J('/status');document.getElementById('stt').textContent=st.ready?`pesos online · ${st.conceitos} conceitos · ${st.neuronios} neurônios profundos`:'carregando os pesos… ('+(st.msg||'')+')';if(!st.ready){setTimeout(boot,2500);return}
 nodes=new vis.DataSet();edges=new vis.DataSet();net=new vis.Network(document.getElementById('net'),{nodes,edges},opt);
 net.on('click',p=>{if(p.nodes.length){const id=p.nodes[0];const n=nodes.get(id);if(n&&n.group=='neuron')return;expand(id);details(id)}});
 seed();}
function addg(g){g.nodes.forEach(n=>{if(!nodes.get(n.id))nodes.add(n);else if(n.group=='focus')nodes.update(n)});g.edges.forEach(e=>{const eid=e.from+'|'+e.to;if(!edges.get(eid))edges.add({id:eid,...e})})}
async function seed(){const g=await J('/seed');nodes.clear();edges.clear();addg(g);setTimeout(fit,400)}
async function expand(w){const g=await J('/expand?w='+encodeURIComponent(w));addg(g)}
async function go(){const w=document.getElementById('q').value.trim();if(!w)return;await expand(w);details(w);net.selectNodes([w]);net.focus(w,{scale:1.1,animation:true})}
function fit(){net.fit({animation:true})}
async function details(w){document.getElementById('sh').textContent=w;document.getElementById('ssub').textContent='neurônios que ESCREVEM "'+w+'" (dos pesos):';
 const d=await J('/neurons?w='+encodeURIComponent(w));
 document.getElementById('sneu').innerHTML=(d.neurons||[]).map(n=>`<div class=neu onclick='addNeuron(${JSON.stringify(n)})'><b>${n.id}</b> · força ${n.forca}<div class=w>escreve: ${n.escreve.join(' · ')||'—'}</div></div>`).join('')||'<span class=sub>—</span>';
 const nb=await J('/expand?w='+encodeURIComponent(w));
 document.getElementById('sviz').innerHTML='<h2 style=margin-top:12px>relaciona com</h2>'+nb.nodes.filter(n=>n.id!=w).map(n=>`<span class=viz onclick='go2("${n.id}")'>${n.label}</span>`).join('');}
function go2(w){expand(w);details(w);net.selectNodes([w]);net.focus(w,{scale:1.1,animation:true})}
function addNeuron(n){const id=n.id;if(!nodes.get(id))nodes.add({id,label:n.id,group:'neuron'});n.escreve.forEach(t=>{if(!nodes.get(t))nodes.add({id:t,label:t,group:'viz'});const eid=id+'|'+t;if(!edges.get(eid))edges.add({id:eid,from:id,to:t})})}
boot();
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s,o): b=json.dumps(o,ensure_ascii=False).encode(); s.send_response(200); s.send_header("Content-Type","application/json; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        u=urlparse(s.path); q=parse_qs(u.query)
        if u.path=="/": b=PAGE.encode(); s.send_response(200); s.send_header("Content-Type","text/html; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b); return
        if u.path=="/status": return s._j(dict(ready=W.ready,conceitos=len(getattr(W,'clean',[])),neuronios=int(getattr(W,'vals',torch.zeros(0)).shape[0]) if W.ready else 0))
        if not W.ready: return s._j(dict(nodes=[],edges=[],neurons=[]))
        if u.path=="/seed": return s._j(W.seed_graph())
        if u.path=="/expand": return s._j(W.expand(q.get("w",["France"])[0]))
        if u.path=="/neurons": return s._j(dict(neurons=W.neurons_for(q.get("w",["France"])[0])))
        s._j({"ok":1})
if __name__=="__main__":
    print("  ►►►  IARA · Explorador de Pesos em http://localhost:3050",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
