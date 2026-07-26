#!/usr/bin/env python3
"""IARA · CENTRO DE OBSERVAÇÃO — mente REAL: usa os neurônios do Qwen-3B, raciocina, lembra (2026-07-12).

Correção de fundo (crítica do Leonardo): a mente NÃO é mais um bloco de notas — ela é o Qwen-3B (os pesos
que já sabem o mundo) + memória persistente + raciocínio:
  AUTO-SUFICIENTE  responde o que você NUNCA ensinou (está nos neurônios): 'capital da França' → Paris.
  RACIOCINA        você fala 'Brasília é a capital do Brasil'; pergunta 'onde fica Brasília?' → 'No Brasil'
                   (usa o que você ensinou como contexto + o mundo que ela já sabe).
  APRENDE FALANDO  declaração ('X é Y') vira memória; pergunta ('...?') dispara raciocínio.
  LEMBRA           tudo salvo no disco (iara_brain_state.json), volta ao reiniciar; não esquece.
  CALCULA          matemática exata (símbolo, palavra, extenso). HORMÔNIOS: dopamina ao aprender.
Tudo pela UI (nada de curl). O 3B carrega em background — a UI abre já e a mente fica 'acordando'. Porta 3050."""
import os,re,json,time,math,threading,unicodedata
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION","10.3.0"); os.environ.setdefault("ROCR_VISIBLE_DEVICES","0")
os.environ.setdefault("OMP_NUM_THREADS","4")
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE=os.path.dirname(os.path.abspath(__file__))
MODEL="/home/leonardo/projects/LLM/llm-lab/models/Qwen2.5-3B-Instruct"
STATE_FILE=os.path.join(HERE,"iara_brain_state.json")
def strip_acc(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s): return re.sub(r"[^a-z0-9 ]","",strip_acc(s).lower()).strip()
WN={"zero":0,"um":1,"uma":1,"dois":2,"duas":2,"tres":3,"quatro":4,"cinco":5,"seis":6,"sete":7,"oito":8,"nove":9,
 "dez":10,"onze":11,"doze":12,"treze":13,"catorze":14,"quatorze":14,"quinze":15,"dezesseis":16,"dezessete":17,
 "dezoito":18,"dezenove":19,"vinte":20,"trinta":30,"quarenta":40,"cinquenta":50,"cem":100,"mil":1000}
def compute(q):
    x=strip_acc(q).lower()
    for w,n in WN.items(): x=re.sub(rf"\b{w}\b",str(n),x)
    x=x.replace("elevado a","^").replace("ao quadrado","^2").replace("por cento","%")
    x=re.sub(r"\bmais\b","+",x); x=re.sub(r"\bmenos\b","-",x); x=re.sub(r"\bvezes\b","*",x)
    x=x.replace("dividido por","/").replace("dividido","/"); x=re.sub(r"(\d)\s*[x×]\s*(\d)",r"\1*\2",x)
    m=re.search(r"raiz\D*?(\d+[.,]?\d*)",x)
    if m: return (f"√{m.group(1)}", round(math.sqrt(float(m.group(1).replace(',','.'))),4))
    m=re.search(r"(\d+[.,]?\d*)\s*%\s*de\s*(\d+[.,]?\d*)",x)
    if m: a=float(m.group(1).replace(',','.')); b=float(m.group(2).replace(',','.')); return (f"{m.group(1)}% de {m.group(2)}", round(a/100*b,4))
    m=re.search(r"(-?\d+[.,]?\d*)\s*([+\-*/^])\s*(-?\d+[.,]?\d*)",x)
    if m:
        a=float(m.group(1).replace(',','.')); op=m.group(2); b=float(m.group(3).replace(',','.'))
        try:
            r={"+":a+b,"-":a-b,"*":a*b,"/":(a/b if b else float('nan'))}.get(op) if op!="^" else a**b
            return (f"{m.group(1)} {op} {m.group(3)}", int(r) if r==int(r) else round(r,4))
        except Exception: return None
    return None
def is_question(q): return "?" in q or bool(re.match(r"^\s*(o que|que |qual|quem|onde|quando|como|por ?que|quanto|cade|cadê)",nrm(q)))
def statement(q):
    if is_question(q): return None
    t=q.strip().rstrip(".")
    m=re.match(r"^(.{2,60}?)\s+(?:é|eh|sao|são|significa|quer dizer|e a|e o)\s+(.{2,})$",t,re.I)
    return (m.group(1).strip(),m.group(2).strip()) if m else None

class Mind:
    def __init__(s):
        s.ready=False; s.thinking=False; s.lock=threading.Lock()
        s.reset(); s._load()
        threading.Thread(target=s._load_model,daemon=True).start()
    def _load_model(s):
        s._log("sistema","acordando… carregando os neurônios (Qwen-3B)")
        s.tok=AutoTokenizer.from_pretrained(MODEL)
        s.model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to("cuda").eval()
        s.ready=True; s._log("sistema","cérebro ONLINE — neurônios do Qwen ativos (sei do mundo + raciocinio)")
    def reset(s):
        s.K={}; s.open=[]; s.dop=0.0; s.cort=0.10; s.val=0.30; s.cur=0.30; s.cur_base=0.30; s.ne=0.10; s.energy=1.0
        s.taught=0; s.calc=0; s.raciocinou=0; s.forgot=0
        s.log=deque(maxlen=80); s.chat=deque(maxlen=40); s.hist=deque(maxlen=140)
        for _ in range(12): s.hist.append(s._h())
    def _save(s):
        try: json.dump({"K":s.K,"cur_base":s.cur_base,"taught":s.taught,"calc":s.calc,"raciocinou":s.raciocinou,
            "chat":list(s.chat),"log":[e for e in s.log if e["kind"]!="sistema"][:60]},open(STATE_FILE,"w"),ensure_ascii=False)
        except Exception: pass
    def _load(s):
        if not os.path.exists(STATE_FILE): return
        try:
            d=json.load(open(STATE_FILE)); s.K=d.get("K",{}); s.cur_base=d.get("cur_base",0.30); s.cur=s.cur_base
            s.taught=d.get("taught",0); s.calc=d.get("calc",0); s.raciocinou=d.get("raciocinou",0)
            s.chat=deque(d.get("chat",[]),maxlen=40); s.log=deque(d.get("log",[]),maxlen=80)
            s._log("sistema",f"memória do disco: {len(s.K)} coisas que já sei")
        except Exception: pass
    def _h(s): return dict(dopamina=round(s.dop,3),cortisol=round(s.cort,3),felicidade=round(s.val,3),curiosidade=round(s.cur,3),noradrenalina=round(s.ne,3),energia=round(s.energy,3),base=round(s.cur_base,3))
    def _log(s,k,t,dop=None): s.log.appendleft(dict(t=time.strftime("%H:%M:%S"),kind=k,text=t,dop=dop))
    def _say(s,who,t): s.chat.appendleft(dict(who=who,text=t))
    def _snap(s): s.hist.append(s._h())
    @torch.no_grad()
    def _gen(s,messages,n=56):
        ids=s.tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to("cuda")
        out=s.model.generate(ids,max_new_tokens=n,do_sample=False,pad_token_id=s.tok.eos_token_id)
        return s.tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
    def _relevant(s,q):
        qn=nrm(q); out=[]
        for k,v in s.K.items():
            if k in qn or any(w in qn for w in k.split() if len(w)>=4): out.append(v.get("stmt") or f"{k}: {v['v']}")
        return out[:6]
    def _reason(s,q):
        facts=s._relevant(q)
        sysmsg=("Você é a IARA, uma mente que raciocina. Responda em UMA frase curta e direta (máximo 14 palavras). "
                "Use os fatos que já sabe abaixo E o seu conhecimento do mundo. Se realmente não souber, responda apenas 'não sei'.")
        u=(("O que você já aprendeu:\n- "+"\n- ".join(facts)+"\n\n") if facts else "")+f"Pergunta: {q}"
        a=s._gen([{"role":"system","content":sysmsg},{"role":"user","content":u}]).split("\n")[0].strip()
        if not a or nrm(a) in ("nao sei","nao sei nao") or "não sei" in a.lower(): return None,facts
        return a,facts
    def teach(s,concept,answer):
        with s.lock: return s._teach(concept,answer)
    def _teach(s,concept,answer):
        c=concept.strip(); k=nrm(c); a=answer.strip(); was=k in [nrm(x) for x in s.open]
        dop=0.5+0.5*(was*s.cur); s.dop=min(1,s.dop+dop); s.val=min(1,s.val+0.35*dop); s.cort=max(0.1,s.cort-0.15); s.ne=min(1,s.ne+0.15)
        s.K[k]={"v":a,"stmt":f"{c} é {a}","kind":"ensinado","consol":s.K.get(k,{}).get("consol",0)+(1+1.6*dop),"curious":was,"reuses":s.K.get(k,{}).get("reuses",0)}
        s.cur_base=min(1,s.cur_base+0.09*dop); s.cur=s.cur_base; s.open=[x for x in s.open if nrm(x)!=k]; s.taught+=1
        s._log("aprende",f"{c} = {a}",round(dop,2)); s._snap(); s._save()
        say=f"Entendi! {c} é {a}. (guardei — dopamina +{dop:.2f})"; s._say("iara",say); return dict(say=say)
    def ask(s,q):
        with s.lock:
            s._say("voce",q); s.ne=min(1,s.ne+0.2)
            st=statement(q)
            if st: return s._teach(st[0],st[1])                    # aprende falando
            r=compute(q)
            if r: s.calc+=1; s.dop=min(1,s.dop+0.2); s._log("calcula",f"{r[0]} = {r[1]}"); s._snap(); s._save(); say=f"{r[0]} = {r[1]} ✔ (calculei — verdade exata)"; s._say("iara",say); return dict(say=say)
            k=nrm(q)
            if k in s.K and s.K[k]["consol"]>=0.25:
                s.K[k]["reuses"]=s.K[k].get("reuses",0)+1; s.val=min(1,s.val+0.05); s._log("sabe",f"{q} = {s.K[k]['v']}"); s._snap()
                say=f"{s.K[k]['v']} 🙂 (já sabia)"; s._say("iara",say); return dict(say=say)
            if not s.ready: say="Ainda estou acordando (carregando os neurônios), me pergunta em uns segundos…"; s._say("iara",say); return dict(say=say)
            s.thinking=True
        a,facts=s._reason(q)                                      # RACIOCÍNIO (fora do lock — o 3B é lento)
        with s.lock:
            s.thinking=False; k=nrm(q)
            if a:
                s.raciocinou+=1; s.dop=min(1,s.dop+0.4)
                s.K[k]={"v":a,"stmt":f"{q} {a}","kind":("raciocinado" if facts else "neuronios"),"consol":1.6,"curious":True,"reuses":0}
                s._log("raciocina" if facts else "neuronios",f"{q} → {a}"+(" [usou o que você ensinou]" if facts else " [dos neurônios]"),0.4); s._snap(); s._save()
                say=f"{a}"+(" 🧠" if facts else ""); s._say("iara",say); return dict(say=say)
            s.cort=min(1,s.cort+0.08); s.cur=min(1,s.cur+0.2);
            if nrm(q) not in [nrm(x) for x in s.open]: s.open.append(q)
            s._log("nao_sabe",f"não soube: {q}"); s._snap()
            say="Hmm, isso eu não sei 🤔 me ensina? (fiquei curiosa)"; s._say("iara",say); return dict(say=say)
    def tick(s):
        with s.lock:
            s.dop*=0.85; s.cort=0.10+(s.cort-0.10)*0.9; s.val=0.3+(s.val-0.3)*0.97; s.ne=0.10+(s.ne-0.10)*0.85
            s.cur=s.cur_base+(s.cur-s.cur_base)*0.7; s.energy=min(1,s.energy+0.02); s._snap()
    def graph(s):
        ks=list(s.K); words={k:set(w for w in (k+" "+s.K[k]["v"]).split() if len(w)>=4) for k in ks}
        nodes=[dict(id=k,v=s.K[k]["v"],forca=round(s.K[k]["consol"],2),curious=s.K[k].get("curious",False),vivo=True,kind=s.K[k].get("kind","")) for k in ks]
        edges=[[ks[i],ks[j]] for i in range(len(ks)) for j in range(i+1,len(ks)) if words[ks[i]]&words[ks[j]]]
        return dict(nodes=nodes,edges=edges)
    def state(s):
        mem=sorted([(k,round(v["consol"],2),v.get("kind","")) for k,v in s.K.items()],key=lambda z:-z[1])
        return dict(h=s._h(),hist=list(s.hist),graph=s.graph(),chat=list(s.chat)[:16],log=list(s.log)[:50],
            memoria=mem,curiosa=s.open,sabe={k:v["v"] for k,v in s.K.items()},ready=s.ready,thinking=s.thinking,
            stats=dict(sei=len(s.K),ensinada=s.taught,calculou=s.calc,raciocinou=s.raciocinou,curiosa=len(s.open)))

M=Mind()
def battery():
    while not M.ready: time.sleep(1)
    M._log("bateria","=== 1) AUTO-SUFICIENTE: sabe sem eu ensinar (dos neurônios) ===")
    for q in ["qual a capital da França?","quem escreveu Dom Casmurro?","qual o maior planeta?","qual a capital do Japão?"]: M.ask(q); time.sleep(0.3)
    M._log("bateria","=== 2) RACIOCÍNIO: ensino e ela DEDUZ ===")
    M.ask("Brasília é a capital do Brasil"); time.sleep(0.3); M.ask("onde fica Brasília?"); time.sleep(0.3)
    M.ask("o gato Mimi é preto"); time.sleep(0.3); M.ask("de que cor é o Mimi?"); time.sleep(0.3)
    M.ask("o Pedro tem 3 irmãos"); time.sleep(0.3); M.ask("quantos irmãos o Pedro tem?"); time.sleep(0.3)
    M._log("bateria","=== 3) MATEMÁTICA (símbolo/palavra/extenso) ===")
    for q in ["1+1?","quanto é um mais um?","12x12","quanto é dois vezes três?","qual a raiz de 144?","quanto é 15% de 200?"]: M.ask(q); time.sleep(0.2)
    M._log("bateria","=== 4) MEMÓRIA: repito e ela lembra instantâneo ===")
    for q in ["onde fica Brasília?","de que cor é o Mimi?"]: M.ask(q); time.sleep(0.2)
    st=M.state()['stats']; M._log("bateria",f"=== FIM: sei {st['sei']} · raciocinou {st['raciocinou']} · calculou {st['calculou']} · base {M.cur_base:.2f} ===")
def relaxer():
    while True: time.sleep(3); M.tick(); M._save()

PAGE=r"""<!doctype html><html lang=pt><head><meta charset=utf8><title>IARA · Centro de Observação</title><style>
*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,Consolas,monospace}body{background:#090b10;color:#e8ebf0;padding:10px;font-size:12px}
h1{font-size:15px;color:#9fe1cb;letter-spacing:3px;display:inline-block}.dot{color:#63c923}#stt{color:#7a8290;font-size:11px;margin-left:10px}
.top{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0;padding:8px;background:#0f1219;border:1px solid #1e2330;border-radius:8px}
input{background:#12151d;border:1px solid #2a2f3a;color:#e8ebf0;padding:7px 9px;border-radius:6px;font-family:inherit;font-size:12px}
input#q{flex:1;min-width:260px}input.sm{width:130px}
button{background:#1b2130;border:1px solid #2a3550;color:#cfe;padding:7px 11px;border-radius:6px;cursor:pointer;font-family:inherit;font-size:12px}button:hover{background:#243050}
button.go{background:#12251d;border-color:#1d9e75;color:#9fe1cb}button.hot{background:#2a1420;border-color:#993556;color:#f6a}button.warn{background:#2a2410;border-color:#854f0b;color:#fda}
.grid{display:grid;grid-template-columns:290px 1fr 330px;gap:10px}
.card{background:#0f1219;border:1px solid #1e2330;border-radius:8px;padding:10px;margin-bottom:10px}
.card h2{font-size:10px;color:#7fb59f;letter-spacing:1.5px;margin-bottom:8px;text-transform:uppercase;display:flex;justify-content:space-between}.card h2 b{color:#9fe1cb}
.g{margin:5px 0}.g .lab{display:flex;justify-content:space-between;font-size:11px;color:#aab}.track{height:8px;background:#171b24;border-radius:4px;overflow:hidden;margin-top:2px}.fill{height:100%;transition:width .4s}
canvas{width:100%;background:#080a0f;border-radius:6px;display:block}
.stat{display:flex;justify-content:space-between;font-size:11px;color:#aab;margin:3px 0}.stat b{color:#9fe1cb}
.tag{display:inline-block;background:#12251d;color:#7fe0bf;border:1px solid #1d9e75;border-radius:6px;padding:2px 7px;font-size:11px;margin:2px}
.ct{display:inline-block;background:#241226;color:#d3a0e8;border:1px solid #7a2a9f;border-radius:10px;padding:1px 8px;font-size:11px;margin:2px}
.mem{margin:4px 0}.mem .ml{display:flex;justify-content:space-between;font-size:11px;color:#bcd}.mem .mt{height:6px;background:#171b24;border-radius:3px;margin-top:1px;overflow:hidden}.mem .mf{height:100%;background:#5dcaa5}
#chat{max-height:170px;overflow:auto;display:flex;flex-direction:column-reverse}#chat .msg{margin:3px 0;font-size:12px;line-height:1.4}#chat .voce{color:#8fb5d8}#chat .iara{color:#9fe1cb}#chat .who{font-size:9px;opacity:.6;margin-right:4px}
#log{max-height:400px;overflow:auto;font-size:11px}#log div{margin:2px 0;padding:2px 5px;border-left:2px solid #2a3550;line-height:1.35}
.k-aprende{border-color:#63c923!important;color:#bfe}.k-calcula{border-color:#378add!important;color:#bcf}.k-sabe{border-color:#5dcaa5!important;color:#9ed}.k-raciocina{border-color:#7f77dd!important;color:#c9c4f0}.k-neuronios{border-color:#1d9e75!important;color:#7fe0bf}.k-nao_sabe{border-color:#ef9f27!important;color:#fda}.k-bateria{border-color:#9fe1cb!important;color:#9fe1cb;font-weight:500}.k-sistema{border-color:#555!important;color:#9ab}
.leg{font-size:10px;color:#7a8290;margin-top:4px}.leg i{font-style:normal;padding:1px 5px;border-radius:3px;margin-right:3px}
</style></head><body>
<h1><span class=dot>◉</span> IARA · CENTRO DE OBSERVAÇÃO</h1><span id=stt>conectando…</span>
<div class=top>
 <input id=q placeholder="fale com ela: 'Brasília é a capital do Brasil' · 'onde fica Brasília?' · 'quanto é 12x12'" onkeydown="if(event.key=='Enter')ask()">
 <button class=go onclick=ask()>enviar</button>
 <input class=sm id=tc placeholder="conceito"><input class=sm id=ta placeholder="é..."><button class=go onclick=teach()>ensinar</button>
 <button class=hot onclick=battery()>🔥 bateria</button><button onclick=wonder()>💭 pensar</button><button class=warn onclick=reset()>🧹 reset</button>
</div>
<div class=grid>
 <div>
  <div class=card><h2>hormônios</h2><div id=horm></div></div>
  <div class=card><h2>linha do tempo</h2><canvas id=hist height=90></canvas><div class=leg><i style="background:#63c923">dopamina</i><i style="background:#ef9f27">curiosidade</i><i style="background:#7f77dd">base</i></div></div>
  <div class=card><h2>métricas</h2><div id=stats></div></div>
 </div>
 <div>
  <div class=card><h2>cérebro — grafo do conhecimento <b id=gc></b></h2><canvas id=brain height=340></canvas>
   <div class=leg><i style="background:#1d9e75">dos neurônios</i><i style="background:#7f77dd">raciocinado/curiosa</i><i style="background:#63c923">ensinado</i> · tamanho = força</div></div>
  <div class=card><h2>conversa</h2><div id=chat></div></div>
 </div>
 <div>
  <div class=card><h2>o que ela sabe (memória)</h2><div id=mem></div></div>
  <div class=card><h2>curiosa sobre</h2><div id=cur></div></div>
  <div class=card><h2>diário ao vivo</h2><div id=log></div></div>
 </div>
</div>
<script>
const HC={dopamina:'#63c923',cortisol:'#e24b4a',felicidade:'#378add',curiosidade:'#ef9f27',noradrenalina:'#d85a30',energia:'#5dcaa5'};
const HL=['dopamina','curiosidade','base'],HLc={dopamina:'#63c923',curiosidade:'#ef9f27',base:'#7f77dd'};let ST=null,GP={};
async function Po(u,b){return(await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})})).json()}
async function ask(){const v=document.getElementById('q').value.trim();if(!v)return;document.getElementById('q').value='';await Po('/ask',{q:v});poll()}
async function teach(){const c=document.getElementById('tc').value.trim(),a=document.getElementById('ta').value.trim();if(!c||!a)return;document.getElementById('tc').value='';document.getElementById('ta').value='';await Po('/teach',{concept:c,a:a});poll()}
async function wonder(){await Po('/wonder',{});poll()}
async function battery(){await Po('/battery',{});poll()}
async function reset(){if(confirm('Resetar a mente (apaga a memória)?')){await Po('/reset',{});poll()}}
async function poll(){try{ST=await(await fetch('/state')).json();render()}catch(e){}}
function bar(k,v,c){return `<div class=g><div class=lab><span>${k}</span><span>${v.toFixed(2)}</span></div><div class=track><div class=fill style="width:${v*100}%;background:${c}"></div></div></div>`}
function render(){if(!ST)return;const h=ST.h;document.getElementById('stt').textContent=ST.thinking?'🧠 pensando…':(ST.ready?'cérebro online (neurônios do Qwen ativos)':'acordando… carregando neurônios');
 let H='';for(const k in HC)H+=bar(k,h[k],HC[k]);H+=bar('curiosidade-base',h.base,'#7f77dd');document.getElementById('horm').innerHTML=H;
 const s=ST.stats;document.getElementById('stats').innerHTML=`<div class=stat><span>coisas que sabe</span><b>${s.sei}</b></div><div class=stat><span>ensinada por você</span><b>${s.ensinada}</b></div><div class=stat><span>raciocinou</span><b>${s.raciocinou}</b></div><div class=stat><span>calculou</span><b>${s.calculou}</b></div><div class=stat><span>curiosa sobre</span><b>${s.curiosa}</b></div>`;
 document.getElementById('gc').textContent=ST.graph.nodes.length+' nós';
 document.getElementById('chat').innerHTML=ST.chat.map(m=>`<div class="msg ${m.who}"><span class=who>${m.who=='voce'?'você':'IARA'}</span>${m.text}</div>`).join('');
 const mx=Math.max(1,...ST.memoria.map(m=>m[1]));document.getElementById('mem').innerHTML=ST.memoria.map(m=>`<div class=mem><div class=ml><span>${m[0]}${m[2]?' <span style=color:#666>('+m[2]+')</span>':''}</span><span>${m[1]}</span></div><div class=mt><div class=mf style="width:${Math.min(100,m[1]/mx*100)}%"></div></div></div>`).join('')||'<span style=color:#666>nada ainda</span>';
 document.getElementById('cur').innerHTML=ST.curiosa.map(c=>`<span class=ct>${c}</span>`).join('')||'<span style=color:#666>—</span>';
 document.getElementById('log').innerHTML=ST.log.map(e=>`<div class=k-${e.kind}>${e.t} · ${e.text}${e.dop!=null?' · DA+'+e.dop:''}</div>`).join('');
 drawHist();drawBrain();}
function drawHist(){const c=document.getElementById('hist'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,Hh=90,hh=ST.hist;x.clearRect(0,0,W,Hh);for(const k of HL){x.beginPath();x.strokeStyle=HLc[k];x.lineWidth=1.4;hh.forEach((p,i)=>{const px=i/(hh.length-1)*W,py=Hh-(p[k]||0)*Hh*.92-3;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}}
function drawBrain(){const g=ST.graph,c=document.getElementById('brain'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,Hh=340;const ids=g.nodes.map(n=>n.id);
 g.nodes.forEach(n=>{if(!GP[n.id])GP[n.id]={x:W/2+(Math.random()-.5)*220,y:Hh/2+(Math.random()-.5)*180,vx:0,vy:0}});Object.keys(GP).forEach(id=>{if(!ids.includes(id))delete GP[id]});const P=GP;
 for(let it=0;it<4;it++){for(const a in P)for(const b in P){if(a>=b)continue;let dx=P[a].x-P[b].x,dy=P[a].y-P[b].y,d=Math.hypot(dx,dy)||1,f=1600/(d*d);P[a].vx+=dx/d*f;P[a].vy+=dy/d*f;P[b].vx-=dx/d*f;P[b].vy-=dy/d*f}
  for(const e of g.edges){const a=P[e[0]],b=P[e[1]];if(!a||!b)continue;let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,f=(d-80)*.008;a.vx+=dx/d*f;a.vy+=dy/d*f;b.vx-=dx/d*f;b.vy-=dy/d*f}
  for(const id in P){const p=P[id];p.vx+=(W/2-p.x)*.003;p.vy+=(Hh/2-p.y)*.003;p.x+=p.vx*.5;p.y+=p.vy*.5;p.vx*=.8;p.vy*=.8;p.x=Math.max(14,Math.min(W-14,p.x));p.y=Math.max(14,Math.min(Hh-14,p.y))}}
 x.clearRect(0,0,W,Hh);x.strokeStyle='#233';for(const e of g.edges){const a=P[e[0]],b=P[e[1]];if(!a||!b)continue;x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}
 for(const n of g.nodes){const p=P[n.id];const r=Math.max(5,Math.min(20,4+n.forca*3));x.beginPath();x.fillStyle=n.kind=='neuronios'?'#1d9e75':(n.curious?'#7f77dd':'#63c923');x.arc(p.x,p.y,r,0,7);x.fill();x.fillStyle='#dfe6f0';x.font='11px monospace';x.fillText(n.id.slice(0,14),p.x+r+2,p.y+3)}}
setInterval(poll,800);poll();
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s,o): b=json.dumps(o,ensure_ascii=False).encode(); s.send_response(200); s.send_header("Content-Type","application/json; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        if s.path=="/": b=PAGE.encode(); s.send_response(200); s.send_header("Content-Type","text/html; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b); return
        if s.path=="/state": return s._j(M.state())
        s._j({"ok":1})
    def do_POST(s):
        n=int(s.headers.get('Content-Length',0)); d=json.loads(s.rfile.read(n) or b"{}")
        if s.path=="/ask": r=M.ask(d.get("q",""))
        elif s.path=="/teach": r=M.teach(d.get("concept",""),d.get("a",""))
        elif s.path=="/wonder": r={"say":"(pensar sozinha em breve)"}
        elif s.path=="/battery": threading.Thread(target=battery,daemon=True).start(); r={"say":"bateria"}
        elif s.path=="/reset": M.reset(); M._save(); r={"say":"reset"}
        else: r={"err":1}
        s._j(r)
if __name__=="__main__":
    threading.Thread(target=relaxer,daemon=True).start()
    print("  ►►►  IARA · Centro de Observação em http://localhost:3050",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
