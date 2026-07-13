#!/usr/bin/env python3
"""IARA · CENTRO DE OBSERVAÇÃO — o cérebro completo dela, tudo pela UI (2026-07-12).

Nada de curl: conversa, ensina, roda bateria, seta orçamento, reseta — tudo por BOTÃO/CAMPO na tela.
Mostra TUDO: grafo do conhecimento (o cérebro), hormônios (gauges + linha do tempo), curiosidade-base,
memória com FORÇA de consolidação (vê o esquecimento), matemática (verdade exata), curiosa-sobre (gated),
diário ao vivo, métricas. Órgão de matemática = verdade. Pesquisa GATED (orçamento). Porta 3050."""
import os,re,json,time,math,threading,subprocess,unicodedata
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
CLAUDE="/home/leonardo/.local/bin/claude"
STATE_FILE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"iara_brain_state.json")
STOP=set("de da do a o e para com que uma um the of and is are foi sao com os as no na em por vira".split())
def nrm(s): return re.sub(r"[^a-z0-9 ]","",unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode().lower()).strip()
def concept_of(q):
    x=nrm(q); m=re.search(r"(?:o que (?:e|sao)|que e|quem (?:e|foi)|defina|conhece|sabe[a-z ]* que e)\s+(.+)",x)
    c=(m.group(1) if m else x); c=re.sub(r"^(o |a |os |as |um |uma |e |de )+","",c).strip(" ?.")
    return c or x
def compute(q):
    x=nrm(q).replace("elevado a","^").replace("ao quadrado","^ 2").replace(" mais "," + ").replace(" menos "," - ")
    x=x.replace(" vezes "," * ").replace(" dividido por "," / ").replace(" divida "," ").replace(" por "," / ")
    m=re.search(r"raiz.*?(\d+[.,]?\d*)",x)
    if m: return (f"√{m.group(1)}", round(math.sqrt(float(m.group(1).replace(',','.'))),4))
    m=re.search(r"(\d+[.,]?\d*)\s*%?\s*(?:por ?cento)?\s*de\s*(\d+[.,]?\d*)",x)
    if ("%" in q or "por cento" in x) and m:
        a=float(m.group(1).replace(',','.')); b=float(m.group(2).replace(',','.')); return (f"{m.group(1)}% de {m.group(2)}", round(a/100*b,4))
    m=re.search(r"(-?\d+[.,]?\d*)\s*(\^|\+|\-|\*|x|/)\s*(-?\d+[.,]?\d*)",x)
    if m:
        a=float(m.group(1).replace(',','.')); op=m.group(2); b=float(m.group(3).replace(',','.'))
        try:
            if op=="+": r=a+b
            elif op=="-": r=a-b
            elif op in ("*","x"): r=a*b
            elif op=="/": r=(a/b if b else float('nan'))
            elif op=="^": r=a**b
            else: return None
            return (f"{m.group(1)} {op} {m.group(3)}", int(r) if r==int(r) else round(r,4))
        except Exception: return None
    return None

class Mind:
    THRESH=0.25
    def __init__(s,budget=3): s.reset(budget); s._load()
    def reset(s,budget=3):
        s.K={}; s.open=[]; s.dop=0.0; s.cort=0.10; s.val=0.30; s.cur=0.30; s.cur_base=0.30; s.ne=0.10; s.energy=1.0
        s.taught=0; s.calc=0; s.researched=0; s.forgot=0; s.budget=budget
        s.log=deque(maxlen=80); s.chat=deque(maxlen=40); s.hist=deque(maxlen=140); s.lock=threading.Lock()
        for _ in range(12): s.hist.append(s._h())
    def _save(s):
        try: json.dump({"K":s.K,"cur_base":s.cur_base,"taught":s.taught,"calc":s.calc,"researched":s.researched,
            "forgot":s.forgot,"budget":s.budget,"chat":list(s.chat),"log":list(s.log)[:60]},
            open(STATE_FILE,"w"),ensure_ascii=False)
        except Exception: pass
    def _load(s):
        if not os.path.exists(STATE_FILE): return
        try:
            d=json.load(open(STATE_FILE))
            s.K=d.get("K",{}); s.cur_base=d.get("cur_base",0.30); s.cur=s.cur_base
            s.taught=d.get("taught",0); s.calc=d.get("calc",0); s.researched=d.get("researched",0)
            s.forgot=d.get("forgot",0); s.budget=d.get("budget",3)
            s.chat=deque(d.get("chat",[]),maxlen=40); s.log=deque(d.get("log",[]),maxlen=80)
            s._log("sistema",f"cérebro carregado do disco: {len(s.K)} memórias persistidas")
        except Exception: pass
    def _h(s): return dict(dopamina=round(s.dop,3),cortisol=round(s.cort,3),felicidade=round(s.val,3),curiosidade=round(s.cur,3),noradrenalina=round(s.ne,3),energia=round(s.energy,3),base=round(s.cur_base,3))
    def _log(s,kind,text,dop=None): s.log.appendleft(dict(t=time.strftime("%H:%M:%S"),kind=kind,text=text,dop=dop))
    def _say(s,who,text): s.chat.appendleft(dict(who=who,text=text))
    def _snap(s): s.hist.append(s._h())
    def knows(s,c): k=s.K.get(nrm(c)); return k and k["consol"]>=s.THRESH
    def ask(s,q):
        with s.lock:
            s._say("voce",q); s.ne=min(1,s.ne+0.2)
            r=compute(q)
            if r:
                s.calc+=1; s.dop=min(1,s.dop+0.2); s._log("calcula",f"{r[0]} = {r[1]}"); s._snap()
                say=f"{r[0]} = {r[1]} ✔ (calculei — verdade exata)"; s._say("iara",say); return dict(say=say)
            c=concept_of(q); k=nrm(c)
            if s.knows(c):
                s.K[k]["consol"]+=0.4; s.K[k]["reuses"]=s.K[k].get("reuses",0)+1; s.val=min(1,s.val+0.05); s._log("sabe",f"{c} = {s.K[k]['v']}"); s._snap()
                say=f"{c}? Isso eu sei: {s.K[k]['v']} 🙂"; s._say("iara",say); return dict(say=say)
            s.cur=min(1,s.cur+0.25+0.3*s.cur_base); s.cort=min(1,s.cort+0.08); s.dop=min(1,s.dop+0.15*s.cur)
            if k not in [nrm(x) for x in s.open]: s.open.append(c)
            s._log("nao_sabe",f"curiosa sobre '{c}' (cur {s.cur:.2f})"); s._snap()
            say=f"Hmm… não sei o que é {c} 🤔 me ensina? (curiosa {s.cur:.2f})"; s._say("iara",say); return dict(say=say)
    def teach(s,concept,answer):
        with s.lock:
            c=concept.strip() or "?"; k=nrm(c); was=k in [nrm(x) for x in s.open]; s.ne=min(1,s.ne+0.15)
            dop=0.5+0.5*(was*s.cur); s.dop=min(1,s.dop+dop); s.val=min(1,s.val+0.35*dop); s.cort=max(0.1,s.cort-0.15)
            s.K[k]={"v":answer.strip(),"consol":s.K.get(k,{}).get("consol",0)+(1+1.6*dop),"curious":was,"reuses":s.K.get(k,{}).get("reuses",0)}
            s.cur_base=min(1,s.cur_base+0.09*dop); s.cur=s.cur_base
            s.open=[x for x in s.open if nrm(x)!=k]; s.taught+=1; s._log("aprende",f"{c} = {answer.strip()}",round(dop,2)); s._snap(); s._save()
            say=f"Ahh! {c} é {answer.strip()}! (senti dopamina +{dop:.2f}, aprendi — e fiquei mais curiosa)"; s._say("iara",say); return dict(say=say)
    def wonder(s):
        with s.lock:
            if not s.open: s._log("ocioso","curiosa, nada em aberto"); return dict(say="Estou curiosa, mas sem nada aberto — me pergunta algo?")
            c=s.open[0]
            if s.budget<=0:
                s._log("gated",f"queria saber '{c}' mas sem orçamento"); say=f"Queria muito saber o que é {c}, mas não vou pesquisar sozinha (orçamento 0) — me ensina?"; s._say("iara",say); return dict(say=say)
            s.budget-=1; s.researched+=1
            ans=s._research(f"o que é {c}? responda em no máximo 6 palavras")
        if ans: r=s.teach(c,ans); r["say"]=f"⚡ Fui atrás sozinha (restam {s.budget}): "+r["say"]; return r
        say=f"⚡ tentei pesquisar '{c}' e não achei (restam {s.budget})"; s._say("iara",say); return dict(say=say)
    def _research(s,q):
        try:
            r=subprocess.run([CLAUDE,"-p",f"Responda em no máximo 6 palavras, só o fato, sem repetir a pergunta: {q}"],capture_output=True,text=True,timeout=90)
            a=r.stdout.strip().split("\n")[0].strip().rstrip("."); return a if (a and 1<len(a)<60) else None
        except Exception: return None
    def tick(s,dt=1.0):
        with s.lock:                                          # só RELAXA hormônio — memória NÃO decai (permanente)
            s.dop*=0.85; s.cort=0.10+(s.cort-0.10)*0.9; s.val=0.3+(s.val-0.3)*0.97; s.ne=0.10+(s.ne-0.10)*0.85
            s.cur=s.cur_base+(s.cur-s.cur_base)*0.7; s.energy=min(1,s.energy+0.02); s._snap()
    def sleep_forget(s,rounds=6):
        """esquecimento OPCIONAL (só quando pedido): decai o FRACO/não-revisado; o consolidado sobrevive."""
        with s.lock:
            for _ in range(rounds):
                for k in list(s.K):
                    if s.K[k].get("reuses",0)>=1 or s.K[k]["consol"]>=3: continue   # consolidado = permanente
                    was=s.K[k]["consol"]>=s.THRESH; s.K[k]["consol"]*=0.7
                    if was and s.K[k]["consol"]<s.THRESH: s.forgot+=1; s._log("esquece",f"esqueceu '{k}' (frágil, não revisado)")
            s._save()
    def graph(s):
        ks=list(s.K); words={k:set(w for w in (k+" "+s.K[k]["v"]).split() if len(w)>=4 and w not in STOP) for k in ks}
        nodes=[dict(id=k,v=s.K[k]["v"],forca=round(s.K[k]["consol"],2),curious=s.K[k].get("curious",False),vivo=s.K[k]["consol"]>=s.THRESH) for k in ks]
        edges=[[ks[i],ks[j]] for i in range(len(ks)) for j in range(i+1,len(ks)) if words[ks[i]]&words[ks[j]]]
        return dict(nodes=nodes,edges=edges)
    def state(s):
        mem=sorted([(k,round(v["consol"],2),v["consol"]>=s.THRESH) for k,v in s.K.items()],key=lambda z:-z[1])
        return dict(h=s._h(),hist=list(s.hist),graph=s.graph(),chat=list(s.chat)[:16],log=list(s.log)[:50],
            memoria=mem, curiosa=s.open, sabe={k:v["v"] for k,v in s.K.items() if v["consol"]>=s.THRESH},
            stats=dict(ensinada=s.taught,calculou=s.calc,pesquisou=s.researched,esqueceu=s.forgot,orcamento=s.budget,vivos=len(mem)))

M=Mind()
FATOS=[("fotossintese","planta vira luz do sol em energia"),("gravidade","forca que atrai as coisas"),
 ("dna","molecula das instrucoes da vida"),("celula","unidade basica da vida"),
 ("atomo","nucleo com protons e eletrons"),("neuronio","celula que transmite sinais no cerebro"),
 ("oxigenio","gas que a gente respira"),("agua","dois hidrogenios e um oxigenio"),
 ("sol","estrela do centro do sistema solar"),("lua","satelite natural da terra"),
 ("coracao","orgao que bombeia o sangue"),("vulcao","montanha que expele lava"),
 ("dinossauro","reptil gigante ja extinto"),("computador","maquina que processa informacao"),
 ("musica","arte de organizar sons"),("dopamina","hormonio do prazer e da recompensa")]
MATH=["quanto é 347 mais 89?","quanto é 1000 menos 333?","quanto é 12 vezes 12?","qual a raiz de 144?",
 "quanto é 2 elevado a 10?","quanto é 15% de 200?","quanto é 250 dividido por 5?","quanto é 7 vezes 8?",
 "qual a raiz de 81?","quanto é 20% de 150?"]
def battery():
    time.sleep(1)
    M._log("bateria","=== FASE 1: APRENDER 16 fatos (permanente, salvo no disco) ===")
    for c,a in FATOS: M.ask(f"o que é {c}?"); time.sleep(0.3); M.teach(c,a); time.sleep(0.3)
    M._log("bateria","=== FASE 2: MATEMÁTICA — 10 contas (verdade exata) ===")
    for q in MATH: M.ask(q); time.sleep(0.35)
    M._log("bateria","=== FASE 3: curiosidade GATED (orçamento) ===")
    for c in ["entropia","big bang","buraco negro","tectonica","fusao nuclear"]:
        M.ask(f"o que é {c}?"); time.sleep(0.3); M.wonder(); time.sleep(0.4)
    M._log("bateria","=== FASE 4: RETENÇÃO — prova que NÃO esqueceu ===")
    ok=0
    for c,a in FATOS[:8]:
        r=M.ask(f"o que é {c}?"); ok+= ("Isso eu sei" in r.get("say",""))
        time.sleep(0.15)
    st=M.state()['stats']
    M._log("bateria",f"=== FIM: sabe {st['vivos']} · reteve {ok}/8 · calculou {st['calculou']} · base {M.cur_base:.2f} · esqueceu {M.forgot} ===")
    M._save()
def relaxer():
    while True: time.sleep(3); M.tick(0.5); M._save()   # só relaxa hormônio + SALVA (não esquece)

PAGE=r"""<!doctype html><html lang=pt><head><meta charset=utf8><title>IARA · Centro de Observação</title><style>
*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,Consolas,monospace}
body{background:#090b10;color:#e8ebf0;padding:10px;font-size:12px}
h1{font-size:15px;color:#9fe1cb;letter-spacing:3px;display:inline-block}.dot{color:#63c923}
.top{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0;padding:8px;background:#0f1219;border:1px solid #1e2330;border-radius:8px}
input{background:#12151d;border:1px solid #2a2f3a;color:#e8ebf0;padding:7px 9px;border-radius:6px;font-family:inherit;font-size:12px}
input#q{flex:1;min-width:240px}input.sm{width:130px}input#bud{width:46px}
button{background:#1b2130;border:1px solid #2a3550;color:#cfe;padding:7px 11px;border-radius:6px;cursor:pointer;font-family:inherit;font-size:12px}button:hover{background:#243050}
button.go{background:#12251d;border-color:#1d9e75;color:#9fe1cb}button.hot{background:#2a1420;border-color:#993556;color:#f6a}button.warn{background:#2a2410;border-color:#854f0b;color:#fda}
.grid{display:grid;grid-template-columns:290px 1fr 330px;gap:10px}
.card{background:#0f1219;border:1px solid #1e2330;border-radius:8px;padding:10px;margin-bottom:10px}
.card h2{font-size:10px;color:#7fb59f;letter-spacing:1.5px;margin-bottom:8px;text-transform:uppercase;display:flex;justify-content:space-between}
.card h2 b{color:#9fe1cb}
.g{margin:5px 0}.g .lab{display:flex;justify-content:space-between;font-size:11px;color:#aab}
.track{height:8px;background:#171b24;border-radius:4px;overflow:hidden;margin-top:2px}.fill{height:100%;transition:width .4s}
canvas{width:100%;background:#080a0f;border-radius:6px;display:block}
.stat{display:flex;justify-content:space-between;font-size:11px;color:#aab;margin:3px 0}.stat b{color:#9fe1cb}
.tag{display:inline-block;background:#12251d;color:#7fe0bf;border:1px solid #1d9e75;border-radius:6px;padding:2px 7px;font-size:11px;margin:2px}
.ct{display:inline-block;background:#241226;color:#d3a0e8;border:1px solid #7a2a9f;border-radius:10px;padding:1px 8px;font-size:11px;margin:2px}
.mem{margin:4px 0}.mem .ml{display:flex;justify-content:space-between;font-size:11px;color:#bcd}.mem .mt{height:6px;background:#171b24;border-radius:3px;margin-top:1px;overflow:hidden}.mem .mf{height:100%;background:#5dcaa5}.mem.dead .mf{background:#5f4a4a}.mem.dead .ml{color:#7a6a6a}
#chat{max-height:150px;overflow:auto;display:flex;flex-direction:column-reverse}
#chat .msg{margin:3px 0;font-size:12px;line-height:1.4}#chat .voce{color:#8fb5d8}#chat .iara{color:#9fe1cb}
#chat .who{font-size:9px;opacity:.6;margin-right:4px}
#log{max-height:400px;overflow:auto;font-size:11px}#log div{margin:2px 0;padding:2px 5px;border-left:2px solid #2a3550;line-height:1.35}
.k-aprende{border-color:#63c923!important;color:#bfe}.k-calcula{border-color:#378add!important;color:#bcf}.k-sabe{border-color:#5dcaa5!important;color:#9ed}
.k-nao_sabe{border-color:#ef9f27!important;color:#fda}.k-esquece{border-color:#e24b4a!important;color:#f9a}.k-gated{border-color:#993556!important;color:#f9c}.k-bateria{border-color:#9fe1cb!important;color:#9fe1cb;font-weight:500}.k-ocioso{border-color:#555!important;color:#999}
.leg{font-size:10px;color:#7a8290;margin-top:4px}.leg i{font-style:normal;padding:1px 5px;border-radius:3px;margin-right:3px}
</style></head><body>
<h1><span class=dot>◉</span> IARA · CENTRO DE OBSERVAÇÃO</h1>
<div class=top>
 <input id=q placeholder="converse com ela… ('o que é X?', 'quanto é 12 vezes 12?')" onkeydown="if(event.key=='Enter')ask()">
 <button class=go onclick=ask()>enviar</button>
 <input class=sm id=tc placeholder="conceito"><input class=sm id=ta placeholder="resposta"><button class=go onclick=teach()>ensinar</button>
 <button class=hot onclick=battery()>🔥 bateria de fogo</button>
 <button onclick=wonder()>💭 pensar sozinha</button>
 <span style="color:#7a8290">orçamento:</span><input id=bud type=number value=3 min=0 max=20><button onclick=setbud()>set</button>
 <button class=warn onclick=reset()>🧹 reset</button>
</div>
<div class=grid>
 <div>
  <div class=card><h2>hormônios</h2><div id=horm></div></div>
  <div class=card><h2>linha do tempo (hormônios)</h2><canvas id=hist height=90></canvas>
    <div class=leg><i style="background:#63c923">dopamina</i><i style="background:#e24b4a">cortisol</i><i style="background:#ef9f27">curiosidade</i><i style="background:#7f77dd">base</i><i style="background:#5dcaa5">energia</i></div></div>
  <div class=card><h2>métricas</h2><div id=stats></div></div>
 </div>
 <div>
  <div class=card><h2>cérebro — grafo do conhecimento <b id=gc></b></h2><canvas id=brain height=340></canvas>
    <div class=leg><i style="background:#63c923">aprendido</i><i style="background:#7f77dd">era curiosa</i><i style="background:#5f4a4a">esmaecendo</i> · tamanho = força da memória</div></div>
  <div class=card><h2>conversa</h2><div id=chat></div></div>
 </div>
 <div>
  <div class=card><h2>memória (força / esquecimento)</h2><div id=mem></div></div>
  <div class=card><h2>curiosa sobre (gated)</h2><div id=cur></div></div>
  <div class=card><h2>diário ao vivo</h2><div id=log></div></div>
 </div>
</div>
<script>
const HC={dopamina:'#63c923',cortisol:'#e24b4a',felicidade:'#378add',curiosidade:'#ef9f27',noradrenalina:'#d85a30',energia:'#5dcaa5'};
const HL=['dopamina','cortisol','curiosidade','base','energia'];const HLc={dopamina:'#63c923',cortisol:'#e24b4a',curiosidade:'#ef9f27',base:'#7f77dd',energia:'#5dcaa5'};
let ST=null,GP={};
async function P(u,b){return(await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})})).json()}
async function ask(){const v=document.getElementById('q').value.trim();if(!v)return;document.getElementById('q').value='';await P('/ask',{q:v});poll()}
async function teach(){const c=document.getElementById('tc').value.trim(),a=document.getElementById('ta').value.trim();if(!c||!a)return;document.getElementById('tc').value='';document.getElementById('ta').value='';await P('/teach',{concept:c,a:a});poll()}
async function wonder(){await P('/wonder',{});poll()}
async function battery(){await P('/battery',{});poll()}
async function reset(){if(confirm('Resetar a mente da IARA (apaga tudo)?')){await P('/reset',{});poll()}}
async function setbud(){await P('/budget',{n:+document.getElementById('bud').value});poll()}
async function poll(){try{ST=await(await fetch('/state')).json();render()}catch(e){}}
function bar(k,val,col){return `<div class=g><div class=lab><span>${k}</span><span>${val.toFixed(2)}</span></div><div class=track><div class=fill style="width:${val*100}%;background:${col}"></div></div></div>`}
function render(){if(!ST)return;const h=ST.h;
 let H='';for(const k in HC)H+=bar(k,h[k],HC[k]);H+=bar('curiosidade-base',h.base,'#7f77dd');document.getElementById('horm').innerHTML=H;
 const s=ST.stats;document.getElementById('stats').innerHTML=`<div class=stat><span>conceitos vivos</span><b>${s.vivos}</b></div><div class=stat><span>ensinada</span><b>${s.ensinada}</b></div><div class=stat><span>calculou (matemática)</span><b>${s.calculou}</b></div><div class=stat><span>pesquisou (claude)</span><b>${s.pesquisou}</b></div><div class=stat><span>esqueceu</span><b>${s.esqueceu}</b></div><div class=stat><span>orçamento de pesquisa</span><b>${s.orcamento}</b></div>`;
 document.getElementById('gc').textContent=ST.graph.nodes.length+' nós';
 document.getElementById('chat').innerHTML=ST.chat.map(m=>`<div class="msg ${m.who}"><span class=who>${m.who=='voce'?'você':'IARA'}</span>${m.text}</div>`).join('');
 const mx=Math.max(1,...ST.memoria.map(m=>m[1]));document.getElementById('mem').innerHTML=ST.memoria.map(m=>`<div class="mem ${m[2]?'':'dead'}"><div class=ml><span>${m[0]}</span><span>${m[1]}</span></div><div class=mt><div class=mf style="width:${Math.min(100,m[1]/mx*100)}%"></div></div></div>`).join('')||'<span style=color:#666>nada ainda</span>';
 document.getElementById('cur').innerHTML=ST.curiosa.map(c=>`<span class=ct>${c}?</span>`).join('')||'<span style=color:#666>—</span>';
 document.getElementById('log').innerHTML=ST.log.map(e=>`<div class=k-${e.kind}>${e.t} · ${e.text}${e.dop!=null?' · DA+'+e.dop:''}</div>`).join('');
 drawHist();drawBrain();}
function drawHist(){const c=document.getElementById('hist'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,Hh=90,hh=ST.hist;x.clearRect(0,0,W,Hh);x.strokeStyle='#1a1f2a';x.beginPath();x.moveTo(0,Hh/2);x.lineTo(W,Hh/2);x.stroke();for(const k of HL){x.beginPath();x.strokeStyle=HLc[k];x.lineWidth=1.4;hh.forEach((p,i)=>{const px=i/(hh.length-1)*W,py=Hh-(p[k]||0)*Hh*0.92-3;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}}
function drawBrain(){const g=ST.graph,c=document.getElementById('brain'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,Hh=340;const ids=g.nodes.map(n=>n.id);
 g.nodes.forEach(n=>{if(!GP[n.id])GP[n.id]={x:W/2+(Math.random()-.5)*220,y:Hh/2+(Math.random()-.5)*180,vx:0,vy:0}});Object.keys(GP).forEach(id=>{if(!ids.includes(id))delete GP[id]});const P=GP;
 for(let it=0;it<4;it++){for(const a in P)for(const b in P){if(a>=b)continue;let dx=P[a].x-P[b].x,dy=P[a].y-P[b].y,d=Math.hypot(dx,dy)||1,f=1600/(d*d);P[a].vx+=dx/d*f;P[a].vy+=dy/d*f;P[b].vx-=dx/d*f;P[b].vy-=dy/d*f}
  for(const e of g.edges){const a=P[e[0]],b=P[e[1]];if(!a||!b)continue;let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1,f=(d-80)*.008;a.vx+=dx/d*f;a.vy+=dy/d*f;b.vx-=dx/d*f;b.vy-=dy/d*f}
  for(const id in P){const p=P[id];p.vx+=(W/2-p.x)*.003;p.vy+=(Hh/2-p.y)*.003;p.x+=p.vx*.5;p.y+=p.vy*.5;p.vx*=.8;p.vy*=.8;p.x=Math.max(14,Math.min(W-14,p.x));p.y=Math.max(14,Math.min(Hh-14,p.y))}}
 x.clearRect(0,0,W,Hh);x.strokeStyle='#233';x.lineWidth=1;for(const e of g.edges){const a=P[e[0]],b=P[e[1]];if(!a||!b)continue;x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.stroke()}
 for(const n of g.nodes){const p=P[n.id];const r=Math.max(5,Math.min(20,4+n.forca*3));x.beginPath();x.fillStyle=!n.vivo?'#5f4a4a':(n.curious?'#7f77dd':'#63c923');x.globalAlpha=n.vivo?1:.5;x.arc(p.x,p.y,r,0,7);x.fill();x.globalAlpha=1;x.fillStyle='#dfe6f0';x.font='11px monospace';x.fillText(n.id.slice(0,14),p.x+r+2,p.y+3)}}
setInterval(poll,700);poll();
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
        elif s.path=="/wonder": r=M.wonder()
        elif s.path=="/battery": threading.Thread(target=battery,daemon=True).start(); r={"say":"bateria iniciada"}
        elif s.path=="/reset": M.reset(); M._save(); r={"say":"resetada"}
        elif s.path=="/budget": M.budget=max(0,int(d.get("n",3))); r={"say":f"orçamento {M.budget}"}
        else: r={"err":1}
        s._j(r)
if __name__=="__main__":
    threading.Thread(target=relaxer,daemon=True).start()
    print("  ►►►  IARA · Centro de Observação em http://localhost:3050",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
