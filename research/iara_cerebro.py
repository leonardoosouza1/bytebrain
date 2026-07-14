#!/usr/bin/env python3
"""IARA · CÉREBRO — anatômico, num painel só, TUDO rastreável (2026-07-12).

Reúne as peças validadas em REGIÕES de um cérebro, com trace de cada passo:
  SENSORIAL (córtex visual) : imagem → sinal/luz/cores/bordas/FREQUÊNCIAS(FFT)/conceito(CLIP)
  ROTEADOR                  : decide RÁPIDO (reflexo) ou LENTO (córtex)
  REFLEXO (rápido ~ms)      : matemática exata · fato conhecido (semente) · typo
  CÓRTEX PRÉ-FRONTAL (lento): Qwen-3B raciocina (sabe o mundo + usa a memória)
  HIPOCAMPO/MEMÓRIA         : sementes (fatos) + árvores (associações dos pesos) · salva no disco
  NEUROMODULAÇÃO            : dopamina(aprender) · cortisol · curiosidade
  VERIFICADOR (metacog.)    : sabe/blefa → abstém, não alucina
Painel único: cérebro (grafo sementes+árvores dos pesos, clicável) + TRACE detalhado + hormônios + entrada
texto/imagem. Qwen carregado (sabe as coisas). Porta 3050."""
import os,re,json,time,threading,base64,io,math,unicodedata
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","10.3.0"); os.environ.setdefault("ROCR_VISIBLE_DEVICES","0"); os.environ.setdefault("OMP_NUM_THREADS","4")
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
VLM_ID="HuggingFaceTB/SmolVLM-256M-Instruct"
HERE=os.path.dirname(os.path.abspath(__file__)); MODEL="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-3B-Instruct"
STATE=os.path.join(HERE,"iara_cerebro_state.json"); DEEP=8
CLIP_VOCAB=["a dog","a cat","a person","a car","a building","a tower","a tree","a flower","a mountain","a beach",
 "the sky","a city street","food","a computer","a bird","a boat","the Eiffel Tower","a landscape","a face","a robot"]
WN={"zero":0,"um":1,"uma":1,"dois":2,"duas":2,"tres":3,"quatro":4,"cinco":5,"seis":6,"sete":7,"oito":8,"nove":9,"dez":10,"cem":100,"mil":1000}
STOP_KEY={"qual","que","quem","onde","quando","como","porque","por","uma","para","com","dos","das","the","and","voce","isso","essa","esse","capital","cor","nome","the","this","that"}
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()
def compute(q):
    x=strip_acc(q).lower()
    for w,n in WN.items(): x=re.sub(rf"\b{w}\b",str(n),x)
    x=x.replace("elevado a","^").replace("por cento","%"); x=re.sub(r"\bmais\b","+",x); x=re.sub(r"\bmenos\b","-",x); x=re.sub(r"\bvezes\b","*",x)
    x=x.replace("dividido por","/"); x=re.sub(r"(\d)\s*[x×]\s*(\d)",r"\1*\2",x)
    m=re.search(r"raiz\D*?(\d+[.,]?\d*)",x)
    if m: return (f"√{m.group(1)}",round(math.sqrt(float(m.group(1).replace(',','.'))),4))
    m=re.search(r"(\d+[.,]?\d*)\s*%\s*de\s*(\d+[.,]?\d*)",x)
    if m: a=float(m.group(1).replace(',','.'));b=float(m.group(2).replace(',','.'));return (f"{m.group(1)}% de {m.group(2)}",round(a/100*b,4))
    m=re.search(r"(-?\d+[.,]?\d*)\s*([+\-*/^])\s*(-?\d+[.,]?\d*)",x)
    if m:
        a=float(m.group(1).replace(',','.'));op=m.group(2);b=float(m.group(3).replace(',','.'))
        try:
            r=a+b if op=="+" else a-b if op=="-" else a*b if op=="*" else (a/b if b else float('nan')) if op=="/" else a**b
            return (f"{m.group(1)} {op} {m.group(3)}",int(r) if r==int(r) else round(r,4))
        except Exception: return None
    return None

class Cerebro:
    def __init__(s):
        s.ready=False; s.busy=False; s.lock=threading.Lock()
        s.seeds={}; s.chat=[]; s.trace=[]; s.dop=0.2; s.cort=0.1; s.cur=0.3
        s._load_state(); threading.Thread(target=s._load,daemon=True).start()
    def _load(s):
        t=time.time(); print("[carregando cérebro: Qwen-3B + córtex visual CLIP…]",flush=True)
        s.tok=AutoTokenizer.from_pretrained(MODEL); m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to("cuda").eval()
        s.model=m; s.E=m.model.embed_tokens.weight.detach(); s.nw=m.model.norm.weight.detach(); s.Et=s.E.t().contiguous()
        V=s.E.shape[0]; toks=s.tok.convert_ids_to_tokens(list(range(V)))
        raw=[(i,(toks[i] or "").replace("Ġ","")) for i in range(V) if re.fullmatch(r"Ġ?[A-Za-z][a-zA-Z]{2,}",toks[i] or "")]
        norms=s.E[[i for i,_ in raw]].float().norm(dim=1); med=float(norms.median())
        s.clean=[];s.words=[]
        for (i,w),nm in zip(raw,norms.tolist()):
            if re.search(r"[a-z][A-Z]",w) or not(0.55*med<nm<1.8*med): continue
            s.clean.append(i);s.words.append(w)
        s.wmap={w.lower():s.clean[j] for j,w in enumerate(s.words)}; s.Ec=torch.nn.functional.normalize(s.E[s.clean].float(),dim=1)
        NL=m.config.num_hidden_layers;INT=m.config.intermediate_size;vals=[];s.idx=[]
        for L in range(NL-DEEP,NL): dp=m.model.layers[L].mlp.down_proj.weight.detach();vals.append((dp.t()*s.nw).contiguous());s.idx+=[(L,i) for i in range(INT)]
        s.vals=torch.cat(vals,0)
        s.vlmproc=AutoProcessor.from_pretrained(VLM_ID); s.vlm=AutoModelForImageTextToText.from_pretrained(VLM_ID,dtype=torch.float16).to("cuda").eval()
        s.ready=True; print(f"[cérebro ONLINE em {time.time()-t:.0f}s · {len(s.clean)} conceitos · {s.vals.shape[0]} neurônios · olho VLM]",flush=True)
    @torch.no_grad()
    def _vlm(s,img,q):
        msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
        p=s.vlmproc.apply_chat_template(msgs,add_generation_prompt=True)
        inp=s.vlmproc(text=p,images=[img],return_tensors="pt").to("cuda")
        out=s.vlm.generate(**inp,max_new_tokens=140,do_sample=False)
        r=s.vlmproc.decode(out[0,inp.input_ids.shape[1]:],skip_special_tokens=True).strip()
        return " ".join(r.split())[:400]
    # ---- persistência ----
    def _save_state(s):
        try: json.dump({"seeds":s.seeds,"dop":s.dop,"cur":s.cur},open(STATE,"w"),ensure_ascii=False)
        except Exception: pass
    def _load_state(s):
        if os.path.exists(STATE):
            try: d=json.load(open(STATE)); s.seeds=d.get("seeds",{}); s.dop=d.get("dop",0.2); s.cur=d.get("cur",0.3)
            except Exception: pass
    # ---- pesos (árvores) ----
    def _wid(s,w):
        if w.lower() in s.wmap: return s.wmap[w.lower()]
        ids=s.tok.encode(" "+w,add_special_tokens=False); return ids[0] if ids else None
    @torch.no_grad()
    def neighbors(s,word,k=6):
        tid=s._wid(word)
        if tid is None: return []
        v=torch.nn.functional.normalize(s.E[tid].float(),dim=0); sims=s.Ec@v; top=torch.topk(sims,k+6).indices.tolist(); out=[]
        for j in top:
            w=s.words[j]
            if w.lower()!=word.lower() and w.lower() not in [o[0].lower() for o in out]: out.append((w,round(float(sims[j]),3)))
            if len(out)>=k: break
        return out
    @torch.no_grad()
    def neurons_for(s,word,k=6):
        tid=s._wid(word)
        if tid is None: return []
        sc=s.vals@s.E[tid].to(s.vals.dtype); top=torch.topk(sc,k).indices.tolist(); out=[]
        for j in top:
            L,i=s.idx[j]; wl=s.vals[j].to(s.Et.dtype)@s.Et; wt=torch.topk(wl,5).indices.tolist()
            wr=[s.tok.convert_ids_to_tokens([t])[0].replace("Ġ","") for t in wt]; wr=[w for w in wr if re.fullmatch(r"[A-Za-z][a-zA-Z]{1,}",w)][:4]
            out.append(dict(id=f"L{L}·n{i}",escreve=wr,forca=round(float(sc[j]),1)))
        return out
    # ---- córtex visual ----
    @torch.no_grad()
    def _visao(s,path,T):
        img=Image.open(path).convert("RGB"); arr=np.asarray(img,dtype=np.float32); Hh,Ww=arr.shape[:2]
        T.append(("SENSORIAL","sinal cru",f"{Ww}x{Hh}px ({Ww*Hh} pixels)"))
        g=np.asarray(img.convert("L"),dtype=np.float32); T.append(("SENSORIAL","luz",f"brilho {g.mean()/255:.2f} · contraste {g.std()/255:.2f}"))
        sm=np.asarray(img.resize((48,48))).reshape(-1,3); q=(sm//48*48).astype(int); cols,cnt=np.unique(q,axis=0,return_counts=True); od=np.argsort(-cnt)[:3]
        nome=lambda c:("preto" if sum(c)<120 else "branco" if sum(c)>620 else "azul" if c[2]>c[0]+30 and c[2]>c[1]+20 else "verde" if c[1]>c[0]+30 else "vermelho" if c[0]>c[1]+40 else "tom neutro")
        T.append(("SENSORIAL","cores dominantes",", ".join(f"{nome(cols[i])} {cnt[i]/len(sm):.0%}" for i in od)))
        gx=np.abs(np.diff(g,axis=1));gy=np.abs(np.diff(g,axis=0));bd=(gx.mean()+gy.mean())/2/255
        T.append(("SENSORIAL","bordas",f"{bd:.3f} ({'muita textura/detalhe' if bd>0.06 else 'liso/suave'})"))
        gr=np.asarray(img.convert("L").resize((256,256)),dtype=np.float32); F=np.fft.fftshift(np.fft.fft2(gr)); mag=np.abs(F)
        Y,X=np.ogrid[:256,:256]; r=np.sqrt((X-128)**2+(Y-128)**2); lo=float(mag[r<25].sum());mi=float(mag[(r>=25)&(r<80)].sum());hi=float(mag[r>=80].sum());tt=lo+mi+hi+1e-9
        T.append(("SENSORIAL","FREQUÊNCIAS (FFT)",f"baixa {lo/tt:.0%} (formas) · média {mi/tt:.0%} · alta {hi/tt:.0%} (textura)"))
        desc=s._vlm(img,"Look carefully and describe what this image shows. If there is any text, title, table or numbers, read them out loud in detail.")
        T.append(("SENSORIAL","VLM — olha os pixels e lê",desc))
        return desc
    # ---- córtex pré-frontal ----
    @torch.no_grad()
    def _gen(s,messages,n=64):
        text=s.tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        ids=s.tok(text,return_tensors="pt").input_ids.to("cuda")
        out=s.model.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=s.tok.eos_token_id)
        return s.tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
    def _reason(s,q,T):
        facts=[v["stmt"] for k,v in s.seeds.items() if k in nrm(q) or any(w in nrm(q) for w in k.split() if len(w)>=4)][:6]
        if facts: T.append(("HIPOCAMPO","lê sementes relevantes"," · ".join(facts)))
        sysmsg="Você é a IARA. Responda em 1 frase curta. Use os fatos e o que sabe do mundo. Se não souber, diga 'não sei'."
        u=(("Você sabe:\n- "+"\n- ".join(facts)+"\n\n") if facts else "")+f"Pergunta: {q}"
        a=s._gen([{"role":"system","content":sysmsg},{"role":"user","content":u}]).split("\n")[0].strip()
        return None if (not a or "não sei" in a.lower() or nrm(a)=="nao sei") else a
    def _seed(s,concept,stmt,kind):
        k=nrm(concept); s.seeds[k]={"v":stmt,"stmt":stmt,"kind":kind,"t":time.strftime("%H:%M")}; s._save_state()
    def _key(s,text):
        cand=[w for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}",text) if nrm(w) in s.wmap and nrm(w) not in STOP_KEY]
        caps=[w for w in cand if w[0].isupper()]
        return caps[0] if caps else (cand[0] if cand else None)
    # ---- pipeline com TRACE ----
    def think(s,q):
        with s.lock:
            s.busy=True; s.chat.append(("voce",q)); T=[("SENSORIAL","recebeu texto",q)]; t0=time.perf_counter()
            # ROTEADOR + REFLEXO
            r=compute(q)
            if r:
                T.append(("ROTEADOR","é conta → REFLEXO (rápido)","")); T.append(("REFLEXO","matemática exata",f"{r[0]} = {r[1]}"))
                s.dop=min(1,s.dop+0.15); ans=f"{r[0]} = {r[1]}"; s._finish(T,t0,ans); return dict(say=ans)
            st=re.match(r"^(.{2,60}?)\s+(?:é|eh|significa)\s+(.{2,})$",q.strip().rstrip("."),re.I)
            if st and "?" not in q:
                c,a=st.group(1).strip(),st.group(2).strip(); T.append(("ROTEADOR","é declaração → HIPOCAMPO"))
                s._seed(c,f"{c} é {a}","ensinado"); T.append(("HIPOCAMPO","gravou SEMENTE",f"{c} = {a}"))
                s.dop=min(1,s.dop+0.4); T.append(("NEUROMOD","dopamina ↑ (aprendeu)",f"{s.dop:.2f}")); ans=f"Entendi! {c} é {a}."; s._finish(T,t0,ans); return dict(say=ans,concept=c)
            k=nrm(q)
            for sk,sv in s.seeds.items():
                if sk in k:
                    T.append(("ROTEADOR","fato na memória → REFLEXO")); T.append(("REFLEXO","semente (rápido)",sv["stmt"])); s._finish(T,t0,sv["v"]); return dict(say=sv["v"],concept=s._key(sk))
        # CÓRTEX (lento) — fora do lock
        if not s.ready:
            with s.lock: s.busy=False; s.chat.append(("iara","acordando os neurônios…")); return dict(say="acordando…")
        T.append(("ROTEADOR","não é reflexo → CÓRTEX PRÉ-FRONTAL (lento)")); T.append(("CÓRTEX","raciocinando com o Qwen…",""))
        try:
            a=s._reason(q,T)
        except Exception as e:
            with s.lock: T.append(("ERRO","córtex falhou",str(e)[:90])); s._finish(T,t0,"deu erro ao pensar"); return dict(say="erro no córtex")
        with s.lock:
            if a:
                T.append(("CÓRTEX","respondeu",a)); c=s._key(q) or s._key(a)
                if c: s._seed(c,f"{q} → {a}","raciocinado"); T.append(("HIPOCAMPO","gravou SEMENTE (raciocinada)",f"{c}"))
                s.dop=min(1,s.dop+0.3); T.append(("NEUROMOD","dopamina ↑",f"{s.dop:.2f}")); T.append(("VERIFICADOR","confiante → responde",""))
                s._finish(T,t0,a); return dict(say=a,concept=c)
            s.cort=min(1,s.cort+0.1); s.cur=min(1,s.cur+0.2)
            T.append(("VERIFICADOR","não sabe → ABSTÉM (não alucina)","")); T.append(("NEUROMOD","curiosidade ↑",f"{s.cur:.2f}"))
            s._finish(T,t0,"não sei isso — me ensina?"); return dict(say="não sei isso — me ensina?")
    def see(s,imgdata):
        if not s.ready: return dict(say="acordando…")
        with s.lock: s.busy=True; s.chat.append(("voce","[enviou uma imagem]")); T=[]; t0=time.perf_counter()
        try:
            raw=base64.b64decode(imgdata.split(",")[-1]); p="/tmp/iara_cam_in.jpg"; open(p,"wb").write(raw)
            lab=s._visao(p,T)
        except Exception as e:
            with s.lock: s.busy=False; return dict(say=f"não vi a imagem ({str(e)[:40]})")
        with s.lock:
            T.append(("ROTEADOR","percepto visual pronto (VLM leu de verdade)"))
            c=s._key(lab) or "imagem"; s._seed(c,f"vi: {lab}","visto")
            T.append(("HIPOCAMPO","gravou SEMENTE (visão)",c)); s.dop=min(1,s.dop+0.5); T.append(("NEUROMOD","dopamina ↑ (viu algo novo)",f"{s.dop:.2f}"))
            s.chat.append(("iara","👁 "+lab)); s._finish(T,t0,lab,skipchat=True); return dict(say=lab,concept=c)
    def _finish(s,T,t0,ans,skipchat=False):
        T.append(("—","tempo total",f"{(time.perf_counter()-t0)*1e3:.0f}ms"))
        s.trace=T;
        if not skipchat: s.chat.append(("iara",ans))
        s.busy=False
    def tick(s):
        with s.lock: s.dop=0.2+(s.dop-0.2)*0.92; s.cort=0.1+(s.cort-0.1)*0.9; s.cur=0.3+(s.cur-0.3)*0.95
    def graph(s):
        nodes={};edges=[]
        for k,v in s.seeds.items():
            cid=v.get("v"); label=k
            nodes[k]=dict(id=k,label=k,group={"ensinado":"seed","raciocinado":"reason","visto":"vision"}.get(v["kind"],"seed"))
            if s.ready:
                for nb,sim in s.neighbors(k,3):
                    nodes.setdefault(nb,dict(id=nb,label=nb,group="tree")); edges.append({"from":k,"to":nb})
        return dict(nodes=list(nodes.values()),edges=edges)
    def state(s):
        return dict(ready=s.ready,busy=s.busy,conceitos=len(getattr(s,'clean',[])) if s.ready else 0,neuronios=int(getattr(s,'vals',torch.zeros(0)).shape[0]) if s.ready else 0,
            chat=s.chat[-12:],trace=[list(t)+[""]*(3-len(t)) for t in s.trace],dop=round(s.dop,2),cort=round(s.cort,2),cur=round(s.cur,2),
            sementes=[dict(c=k,s=v["v"],k=v["kind"],t=v.get("t","")) for k,v in list(s.seeds.items())[-12:]])

C=Cerebro()
def relax():
    while True: time.sleep(4); C.tick()
PAGE=r"""<!doctype html><html lang=pt><head><meta charset=utf8><title>IARA · Cérebro</title>
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,monospace}body{background:#090b10;color:#e8ebf0;overflow:hidden}
#bar{position:fixed;top:0;left:0;right:0;height:44px;display:flex;gap:8px;align-items:center;padding:0 12px;background:#0f1219;border-bottom:1px solid #1e2330;z-index:9}
h1{font-size:14px;color:#9fe1cb;letter-spacing:2px}#stt{color:#7a8290;font-size:11px}
#L{position:fixed;top:44px;left:0;width:330px;bottom:0;background:#0d1016;border-right:1px solid #1e2330;display:flex;flex-direction:column}
#net{position:fixed;top:44px;left:330px;right:340px;bottom:0}
#R{position:fixed;top:44px;right:0;width:340px;bottom:0;background:#0d1016;border-left:1px solid #1e2330;overflow:auto;padding:10px}
.card{border-bottom:1px solid #1a1f2a;padding:9px}.card h2{font-size:10px;color:#7fb59f;letter-spacing:1px;margin-bottom:6px;text-transform:uppercase}
#chat{flex:1;overflow:auto;padding:9px}.m{margin:3px 0;font-size:12.5px;line-height:1.4}.voce{color:#8fb5d8}.iara{color:#9fe1cb}
#ci{display:flex;gap:6px;padding:8px;border-top:1px solid #1a1f2a}#qi{flex:1;background:#12151d;border:1px solid #2a3550;color:#e8ebf0;padding:8px;border-radius:6px;font-size:13px}
button{background:#12251d;border:1px solid #1d9e75;color:#9fe1cb;padding:8px 10px;border-radius:6px;cursor:pointer;font-size:12px;font-family:inherit}button:hover{background:#173a2b}
#drop{border:1px dashed #2a3550;border-radius:6px;padding:9px;text-align:center;color:#7a8290;font-size:11px;cursor:pointer}#drop.hot{border-color:#5dcaa5;color:#9fe1cb}
.h{display:flex;justify-content:space-between;font-size:11px;color:#aab;margin:2px 0}.tk{height:6px;background:#171b24;border-radius:3px;margin-top:1px;overflow:hidden}.tf{height:100%}
.tr{margin:3px 0;font-size:11px;line-height:1.35;padding:3px 6px;border-left:3px solid #2a3550;border-radius:3px}
.r-SENSORIAL{border-color:#1d9e75}.r-ROTEADOR{border-color:#888}.r-REFLEXO{border-color:#378add}.r-CÓRTEX{border-color:#7f77dd}.r-HIPOCAMPO{border-color:#ef9f27}.r-NEUROMOD{border-color:#e24b4a}.r-VERIFICADOR{border-color:#5dcaa5}
.tr b{color:#cfe;font-size:10px}.tr .d{color:#9ab}
.sd{font-size:11px;color:#bcd;margin:3px 0;border-left:2px solid #ba7517;padding-left:6px}
</style></head><body>
<div id=bar><h1>◉ IARA · CÉREBRO</h1><span id=stt>acordando…</span></div>
<div id=L>
 <div class=card><h2>hormônios (neuromodulação)</h2><div id=horm></div></div>
 <div class=card><div id=drop onclick="f.click()">📷 arraste/clique uma imagem — o córtex visual analisa (freq/cores/bordas/conceito)</div><input type=file id=f accept=image/* style=display:none onchange=upl()></div>
 <div id=chat></div>
 <div id=ci><input id=qi placeholder="fale com ela… (pergunta, conta, ou 'X é Y')" onkeydown="if(event.key=='Enter')ask()"><button onclick=ask()>enviar</button></div>
</div>
<div id=net></div>
<div id=R>
 <div class=card style=border:0><h2>trace do último pensamento (passo a passo)</h2><div id=trace></div></div>
 <div class=card style=border:0><h2>sementes de memória (hipocampo)</h2><div id=seeds></div></div>
</div>
<script>
let net,nodes,edges;
const opt={nodes:{shape:'dot',size:13,font:{color:'#dfe6f0',size:12,face:'monospace'},borderWidth:2},edges:{color:{color:'#2a3550'},smooth:{type:'continuous'},width:.6},
 groups:{seed:{color:{background:'#ba7517',border:'#ef9f27'}},reason:{color:{background:'#3c3489',border:'#7f77dd'}},vision:{color:{background:'#0f6e56',border:'#1d9e75'}},tree:{color:{background:'#1b2130',border:'#2a3550'},size:9}},
 physics:{barnesHut:{gravitationalConstant:-5000,springLength:110},stabilization:{iterations:90}},interaction:{hover:true}};
async function Po(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}
async function ask(){const v=qi.value.trim();if(!v)return;qi.value='';await Po('/think',{q:v});poll()}
async function upl(){const fl=f.files[0];if(!fl)return;const r=new FileReader();r.onload=async()=>{await Po('/see',{img:r.result});poll()};r.readAsDataURL(fl)}
async function poll(){try{const st=await(await fetch('/state')).json();render(st)}catch(e){}}
async function grefresh(){if(!net)return;const g=await(await fetch('/graph')).json();const ids=g.nodes.map(n=>n.id);g.nodes.forEach(n=>{if(!nodes.get(n.id))nodes.add(n)});g.edges.forEach(e=>{const id=e.from+'|'+e.to;if(!edges.get(id))edges.add({id,...e})})}
function render(st){document.getElementById('stt').textContent=st.busy?'🧠 pensando…':(st.ready?`cérebro online · ${st.conceitos} conceitos · ${st.neuronios} neurônios`:'acordando os neurônios…');
 if(!net&&st.ready){nodes=new vis.DataSet();edges=new vis.DataSet();net=new vis.Network(document.getElementById('net'),{nodes,edges},opt);net.on('click',async p=>{if(p.nodes.length){const g=await(await fetch('/expand?w='+encodeURIComponent(p.nodes[0]))).json();g.nodes.forEach(n=>{if(!nodes.get(n.id))nodes.add({...n,group:'tree'})});g.edges.forEach(e=>{const id=e.from+'|'+e.to;if(!edges.get(id))edges.add({id,...e})})}})}
 horm.innerHTML=[['dopamina',st.dop,'#63c923'],['cortisol',st.cort,'#e24b4a'],['curiosidade',st.cur,'#ef9f27']].map(([n,v,c])=>`<div class=h><span>${n}</span><span>${v}</span></div><div class=tk><div class=tf style="width:${v*100}%;background:${c}"></div></div>`).join('');
 chat.innerHTML=(st.chat||[]).map(m=>`<div class="m ${m[0]}">${m[0]=='voce'?'você: ':'IARA: '}${m[1]}</div>`).join('');chat.scrollTop=chat.scrollHeight;
 trace.innerHTML=(st.trace||[]).map(t=>`<div class="tr r-${t[0]}"><b>${t[0]}</b> · ${t[1]}${t[2]?` <span class=d>${t[2]}</span>`:''}</div>`).join('')||'<span style=color:#666>mande uma mensagem ou imagem…</span>';
 seeds.innerHTML=(st.sementes||[]).slice().reverse().map(x=>`<div class=sd><b style=color:#ef9f27>${x.c}</b> <span style=color:#666>(${x.k})</span><br>${x.s}</div>`).join('')||'<span style=color:#666>nada aprendido ainda</span>';
 grefresh()}
setInterval(poll,700);poll();
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s,o): b=json.dumps(o,ensure_ascii=False).encode(); s.send_response(200); s.send_header("Content-Type","application/json; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        u=urlparse(s.path); q=parse_qs(u.query)
        if u.path=="/": b=PAGE.encode(); s.send_response(200); s.send_header("Content-Type","text/html; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b); return
        if u.path=="/state": return s._j(C.state())
        if u.path=="/graph": return s._j(C.graph() if C.ready else dict(nodes=[],edges=[]))
        if u.path=="/expand": return s._j(dict(nodes=[dict(id=n,label=n,group="tree") for n,_ in (C.neighbors(q.get("w",[""])[0],6) if C.ready else [])],edges=[{"from":q.get("w",[""])[0],"to":n} for n,_ in (C.neighbors(q.get("w",[""])[0],6) if C.ready else [])]))
        s._j({"ok":1})
    def do_POST(s):
        n=int(s.headers.get('Content-Length',0)); d=json.loads(s.rfile.read(n) or b"{}")
        if s.path=="/think": r=C.think(d.get("q",""))
        elif s.path=="/see": r=C.see(d.get("img",""))
        else: r={"err":1}
        s._j(r)
if __name__=="__main__":
    threading.Thread(target=relax,daemon=True).start()
    print("  ►►►  IARA · CÉREBRO em http://localhost:3050",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
