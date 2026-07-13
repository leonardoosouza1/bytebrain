#!/usr/bin/env python3
"""IARA PANEL — mente curiosa com PAINEL + MATEMÁTICA (verdade calculada) + pesquisa GATED (2026-07-12).

Painel web (dark) pra o Leonardo avaliar: hormônios, curiosidade-base, conhecimento, matemática, diário.
Órgão de MATEMÁTICA: ela CALCULA (soma/subtrai/multiplica/divide/raiz/%/potência) — verdade exata, não chute.
Pesquisa GATED: orçamento limitado de buscas no claude (ela NÃO sai pesquisando o que quer); esgotado o
orçamento, fica curiosa e ESPERA o professor. Bateria de fogo roda sozinha (POST /battery). Porta 3050."""
import os,re,json,time,math,threading,subprocess,unicodedata
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
CLAUDE="/home/leonardo/.local/bin/claude"
def nrm(s): return re.sub(r"[^a-z0-9 ]","",unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode().lower()).strip()
def concept_of(q):
    x=nrm(q); m=re.search(r"(?:o que (?:e|sao)|que e|quem (?:e|foi)|defina|conhece|sabe[a-z ]* que e)\s+(.+)",x)
    c=(m.group(1) if m else x); c=re.sub(r"^(o |a |os |as |um |uma |e |de )+","",c).strip(" ?.")
    return c or x

def compute(q):
    """órgão-calculadora: verdade exata. Retorna (expr, valor) ou None."""
    x=nrm(q)
    x=x.replace("elevado a","^").replace("ao quadrado","^ 2").replace(" mais "," + ").replace(" menos "," - ")
    x=x.replace(" vezes "," * ").replace(" dividido por "," / ").replace(" divida "," ").replace(" por "," / ")
    m=re.search(r"raiz.*?(\d+[.,]?\d*)",x)
    if m: v=float(m.group(1).replace(",",".")); return (f"√{m.group(1)}", round(math.sqrt(v),4))
    m=re.search(r"(\d+[.,]?\d*)\s*%?\s*(?:por ?cento)?\s*de\s*(\d+[.,]?\d*)",x)
    if ("%" in q or "por cento" in x) and m:
        a=float(m.group(1).replace(",",".")); b=float(m.group(2).replace(",",".")); return (f"{m.group(1)}% de {m.group(2)}", round(a/100*b,4))
    m=re.search(r"(-?\d+[.,]?\d*)\s*(\^|\+|\-|\*|x|/)\s*(-?\d+[.,]?\d*)",x)
    if m:
        a=float(m.group(1).replace(",",".")); op=m.group(2); b=float(m.group(3).replace(",","."))
        try:                                              # AVALIAÇÃO PREGUIÇOSA (dict eager estourava a**b)
            if op=="+": r=a+b
            elif op=="-": r=a-b
            elif op in ("*","x"): r=a*b
            elif op=="/": r=(a/b if b else float('nan'))
            elif op=="^": r=a**b
            else: return None
            r=int(r) if r==int(r) else round(r,4); return (f"{m.group(1)} {op} {m.group(3)}", r)
        except Exception: return None
    return None

class Mind:
    THRESH=0.25
    def __init__(s,research_budget=3):
        s.K={}; s.open=[]; s.dop=0.0; s.cort=0.10; s.val=0.30; s.cur=0.30; s.cur_base=0.30; s.energy=1.0
        s.taught=0; s.calc=0; s.researched=0; s.forgot=0; s.budget=research_budget
        s.log=deque(maxlen=60); s.curve=deque(maxlen=120); s.lock=threading.Lock()
        for _ in range(15): s.curve.append(0.30)
    def _log(s,kind,text,dop=None): s.log.appendleft(dict(t=time.strftime("%H:%M:%S"),kind=kind,text=text,dop=dop))
    def knows(s,c): k=s.K.get(nrm(c)); return k and k["consol"]>=s.THRESH
    def ask(s,q):
        with s.lock:
            r=compute(q)                                          # órgão matemático primeiro (verdade)
            if r:
                s.calc+=1; s.dop=min(1,s.dop+0.2); s._log("calcula",f"{r[0]} = {r[1]}")
                return dict(say=f"Isso eu calculo: {r[0]} = {r[1]} ✔ (verdade exata)",math=True,val=r[1])
            c=concept_of(q); k=nrm(c)
            if s.knows(c):
                s.K[k]["consol"]+=0.4; s.val=min(1,s.val+0.05); s._log("sabe",f"{c} = {s.K[k]['v']}")
                return dict(say=f"{c}? Isso eu sei: {s.K[k]['v']} 🙂",knew=True)
            s.cur=min(1,s.cur+0.25+0.3*s.cur_base); s.cort=min(1,s.cort+0.08); s.dop=min(1,s.dop+0.15*s.cur)
            if k not in [nrm(x) for x in s.open]: s.open.append(c)
            s._log("nao_sabe",f"curiosa sobre '{c}' (cur {s.cur:.2f})")
            return dict(say=f"Hmm… não sei o que é {c} 🤔 (curiosa {s.cur:.2f})",knew=False,concept=c)
    def teach(s,concept,answer):
        with s.lock:
            c=concept; k=nrm(c); was=k in [nrm(x) for x in s.open]
            dop=0.5+0.5*(was*s.cur); s.dop=min(1,s.dop+dop); s.val=min(1,s.val+0.35*dop); s.cort=max(0.1,s.cort-0.15)
            s.K[k]={"v":answer,"consol":s.K.get(k,{}).get("consol",0)+(1+1.6*dop),"curious":was}
            s.cur_base=min(1,s.cur_base+0.09*dop); s.cur=s.cur_base; s.curve.append(round(s.cur_base,3))
            s.open=[x for x in s.open if nrm(x)!=k]; s.taught+=1; s._log("aprende",f"{c} = {answer}",round(dop,2))
            return dict(say=f"Ahh! {c} é {answer}! (DA +{dop:.2f}, base {s.cur_base:.2f})",dopamine=round(dop,2))
    def wonder(s):
        with s.lock:
            if not s.open:
                s._log("ocioso","curiosa, sem nada em aberto"); return dict(say="curiosa, mas sem pergunta aberta")
            c=s.open[0]
            if s.budget<=0:                                       # GATED: sem orçamento → NÃO pesquisa, espera o professor
                s._log("gated",f"queria saber '{c}' mas sem orçamento → espera professor")
                return dict(say=f"Queria muito saber o que é {c}, mas não vou pesquisar sozinha agora — me ensina? (orçamento 0)")
            s.budget-=1; s.researched+=1
            ans=s._research(f"o que é {c}? responda em no máximo 6 palavras")
        if ans: r=s.teach(c,ans); r["say"]=f"⚡ (pesquisei — restam {s.budget}) "+r["say"]; return r
        return dict(say=f"⚡ tentei pesquisar '{c}' e não achei (restam {s.budget})")
    def _research(s,q):
        try:
            r=subprocess.run([CLAUDE,"-p",f"Responda em no máximo 6 palavras, só o fato, sem repetir a pergunta: {q}"],
                             capture_output=True,text=True,timeout=90)
            a=r.stdout.strip().split("\n")[0].strip().rstrip("."); return a if (a and 1<len(a)<60) else None
        except Exception: return None
    def tick(s,dt=1.0):
        with s.lock:
            s.dop*=0.85; s.cort=0.10+(s.cort-0.10)*0.9; s.val=0.3+(s.val-0.3)*0.97
            s.cur=s.cur_base+(s.cur-s.cur_base)*0.7; s.energy=min(1,s.energy+0.02)
            for k in list(s.K):
                was=s.K[k]["consol"]>=s.THRESH
                s.K[k]["consol"]*=math.exp(-dt/ (6*(1+s.K[k].get("consol",1)*0.15)))
                if was and s.K[k]["consol"]<s.THRESH: s.forgot+=1; s._log("esquece",f"esqueceu '{k}' (não revisado)")
                if s.K[k]["consol"]<0.08: del s.K[k]
    def state(s):
        return dict(hormones=dict(dopamina=round(s.dop,2),cortisol=round(s.cort,2),felicidade=round(s.val,2),
                    curiosidade=round(s.cur,2),base=round(s.cur_base,2)),
                    curve=list(s.curve),sabe={k:v["v"] for k,v in s.K.items() if v["consol"]>=s.THRESH},
                    curiosa=s.open, log=list(s.log)[:40],
                    stats=dict(ensinada=s.taught,calculou=s.calc,pesquisou=s.researched,esqueceu=s.forgot,orcamento=s.budget))

M=Mind(research_budget=3)

# ---------- BATERIA DE FOGO (roda sozinha em thread) ----------
def battery():
    time.sleep(2); post=lambda *a:None
    CUR=[("fotossintese","planta vira luz do sol em energia"),("gravidade","forca que atrai as coisas"),
         ("dna","molecula das instrucoes da vida"),("celula","unidade basica da vida"),
         ("atomo","nucleo com protons e eletrons"),("vulcao","montanha que expele lava")]
    M._log("bateria","=== FASE 1: ensinar + gerar curiosidade ===")
    for c,a in CUR:
        M.ask(f"o que é {c}?"); time.sleep(0.6); M.teach(c,a); time.sleep(0.6)
    M._log("bateria","=== FASE 2: MATEMÁTICA (verdade calculada) ===")
    for q in ["quanto é 347 mais 89?","quanto é 100 dividido por 8?","quanto é 12 vezes 12?",
              "qual a raiz de 144?","quanto é 2 elevado a 10?","quanto é 15% de 200?","quanto é 1000 menos 333?"]:
        M.ask(q); time.sleep(0.5)
    M._log("bateria","=== FASE 3: curiosidade GATED (orçamento 3) ===")
    for c in ["entropia","fotossintese reversa","tectonica","big bang","fusao nuclear"]:
        M.ask(f"o que é {c}?"); time.sleep(0.4); M.wonder(); time.sleep(0.5)
    M._log("bateria","=== FASE 4: esquecimento (revisa alguns, esquece os não-revisados) ===")
    for _ in range(7): M.tick(1.0); time.sleep(0.2)               # tempo passa
    for c in ["fotossintese","atomo","dna"]:                      # REVISA 3 (repetição espaçada → sobrevivem)
        for _ in range(3): M.ask(f"o que é {c}?")
    M._log("bateria","(revisei fotossíntese, átomo, dna — devem sobreviver)")
    for _ in range(9): M.tick(1.0); time.sleep(0.2)               # mais tempo passa
    M._log("bateria",f"=== FIM: sabe {len(M.state()['sabe'])}/{M.taught} · base {M.cur_base:.2f} · calculou {M.calc} · esqueceu {M.forgot} ===")

PAGE=r"""<!doctype html><html><head><meta charset=utf8><title>IARA · painel</title><style>
*{box-sizing:border-box;margin:0;font-family:ui-monospace,Menlo,monospace}body{background:#0b0d12;color:#e8ebf0;padding:12px}
h1{font-size:15px;color:#9fe1cb;letter-spacing:2px}.sub{color:#7a8290;font-size:11px;margin-bottom:8px}
.grid{display:grid;grid-template-columns:280px 1fr 320px;gap:10px}
.card{background:#0f1219;border:1px solid #1e2330;border-radius:8px;padding:10px;margin-bottom:10px}
.card h2{font-size:10px;color:#8fa;letter-spacing:1px;margin-bottom:7px;text-transform:uppercase}
.g{margin:5px 0}.g .lab{display:flex;justify-content:space-between;font-size:11px;color:#aab}
.track{height:8px;background:#171b24;border-radius:4px;overflow:hidden;margin-top:2px}.fill{height:100%;transition:width .4s}
.stat{display:flex;justify-content:space-between;font-size:11px;color:#aab;margin:2px 0}.stat b{color:#9fe1cb}
.tag{display:inline-block;background:#12251d;color:#5dcaa5;border:1px solid #1d9e75;border-radius:6px;padding:2px 7px;font-size:11px;margin:2px}
.ct{display:inline-block;background:#1a1226;color:#c9b8f0;border:1px solid #534ab7;border-radius:10px;padding:1px 7px;font-size:10px;margin:2px}
#log{max-height:520px;overflow:auto;font-size:11px}#log div{margin:2px 0;padding:2px 4px;border-left:2px solid #2a3550}
.k-aprende{border-color:#63c923!important;color:#bfe}.k-calcula{border-color:#378add!important;color:#bcf}
.k-nao_sabe{border-color:#ef9f27!important;color:#fda}.k-esquece{border-color:#e24b4a!important;color:#f9a}
.k-gated{border-color:#7a2a55!important;color:#f9c}.k-bateria{border-color:#9fe1cb!important;color:#9fe1cb;font-weight:500}
canvas{width:100%;background:#0a0c11;border-radius:6px}
</style></head><body>
<h1>◉ IARA · PAINEL</h1><div class=sub id=st>mente curiosa — hormônios · conhecimento · matemática · diário</div>
<div class=grid>
 <div>
  <div class=card><h2>hormônios</h2><div id=horm></div></div>
  <div class=card><h2>curiosidade-base (cresce ao aprender)</h2><canvas id=curve height=70></canvas></div>
  <div class=card><h2>uso</h2><div id=stats></div></div>
 </div>
 <div>
  <div class=card><h2>o que ela sabe</h2><div id=sabe></div></div>
  <div class=card><h2>curiosa sobre (não sabe ainda)</h2><div id=cur></div></div>
 </div>
 <div class=card><h2>diário (ao vivo)</h2><div id=log></div></div>
</div>
<script>
const HC={dopamina:['#63c923'],cortisol:['#e24b4a'],felicidade:['#378add'],curiosidade:['#ef9f27'],base:['#7f77dd']};
async function poll(){try{const s=await(await fetch('/state')).json();render(s)}catch(e){}}
function render(s){const h=s.hormones;let H='';for(const k in HC){H+=`<div class=g><div class=lab><span>${k}</span><span>${h[k]}</span></div><div class=track><div class=fill style="width:${h[k]*100}%;background:${HC[k][0]}"></div></div></div>`}document.getElementById('horm').innerHTML=H;
 const st=s.stats;document.getElementById('stats').innerHTML=`<div class=stat><span>ensinada</span><b>${st.ensinada}</b></div><div class=stat><span>calculou (matemática)</span><b>${st.calculou}</b></div><div class=stat><span>pesquisou (claude)</span><b>${st.pesquisou}</b></div><div class=stat><span>esqueceu</span><b>${st.esqueceu}</b></div><div class=stat><span>orçamento de pesquisa</span><b>${st.orcamento}</b></div>`;
 document.getElementById('sabe').innerHTML=Object.entries(s.sabe).map(([k,v])=>`<span class=tag>${k}: ${v}</span>`).join('')||'<span class=sub>nada ainda</span>';
 document.getElementById('cur').innerHTML=s.curiosa.map(c=>`<span class=ct>${c}?</span>`).join('')||'<span class=sub>—</span>';
 document.getElementById('log').innerHTML=s.log.map(e=>`<div class=k-${e.kind}>${e.t} · ${e.text}${e.dop!=null?' · DA+'+e.dop:''}</div>`).join('');
 const c=document.getElementById('curve'),x=c.getContext('2d');c.width=c.clientWidth;const W=c.width,Hh=70,cv=s.curve;x.clearRect(0,0,W,Hh);x.beginPath();x.strokeStyle='#7f77dd';x.lineWidth=2;cv.forEach((p,i)=>{const px=i/(cv.length-1)*W,py=Hh-p*Hh*0.95-3;i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();}
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
        else: r={"err":1}
        r["state"]=M.state(); s._j(r)
if __name__=="__main__":
    print("  ►►►  IARA PAINEL em http://localhost:3050  (bateria: POST /battery)",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
