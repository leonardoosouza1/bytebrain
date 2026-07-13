#!/usr/bin/env python3
"""IARA MIND — a mente viva pra CONVERSAR: curiosidade → não-saber → ser ensinada → dopamina → aprender,
e a dopamina REFORÇA a curiosidade (ela associa que buscar dá prazer) (2026-07-12).

Servidor leve (sem recarregar modelo — instantâneo por turno) pra eu conversar e testar em tempo real.
Mecanismo-chave (pedido do Leonardo):
  - pergunto algo → se NÃO SABE: curiosidade↑ + leve desconforto (cortisol) + 'seeking' (querer saber).
  - EU ensino (ou ela pesquisa no claude) → DOPAMINA (recompensa, maior se estava curiosa) → APRENDE.
  - META: cada aprendizado-recompensa sobe a CURIOSIDADE-BASE → ela fica MAIS curiosa (aprendeu que
    buscar conhecimento gera dopamina). Com o tempo ela pergunta SOZINHA.
Endpoints: POST /ask{q} · POST /teach{concept,a} · POST /wonder · GET /state. Porta 3050."""
import os,re,json,time,math,subprocess,unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
CLAUDE="/home/leonardo/.local/bin/claude"
def nrm(s): return re.sub(r"[^a-z0-9 ]","",unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode().lower()).strip()
def concept_of(q):
    x=nrm(q)
    m=re.search(r"(?:o que (?:e|sao)|que e|quem (?:e|foi)|defina|conhece|sabe[a-z ]* o que e|sabe[a-z ]* que e)\s+(.+)",x)
    c=m.group(1) if m else x
    c=re.sub(r"^(o |a |os |as |um |uma |e |de )+","",c).strip(" ?.")
    return c or x

class Mind:
    THRESH=0.25
    def __init__(s):
        s.K={}; s.open=[]                     # conhecimento · perguntas em aberto (curiosa e não sabe)
        s.dop=0.0; s.cort=0.10; s.val=0.30    # dopamina · cortisol · felicidade
        s.cur=0.30; s.cur_base=0.30           # curiosidade agora · e a BASE (disposição, cresce com o uso)
        s.energy=1.0; s.taught=0; s.wonders=0; s.log=[]
    def knows(s,c): k=s.K.get(nrm(c)); return k and k["consol"]>=s.THRESH
    def ask(s,q):
        c=concept_of(q); k=nrm(c)
        if s.knows(c):
            s.K[k]["consol"]+=0.4; s.val=min(1,s.val+0.05)
            return dict(say=f"{c}? Isso eu sei: {s.K[k]['v']} 🙂 (já aprendi antes)",concept=c,knew=True)
        # NÃO SABE → curiosidade + desconforto + seeking
        s.cur=min(1,s.cur+0.25+0.3*s.cur_base); s.cort=min(1,s.cort+0.08); s.dop=min(1,s.dop+0.15*s.cur)  # 'wanting'
        if k not in [nrm(x) for x in s.open]: s.open.append(c)
        s.log.append(("nao_sabe",c))
        return dict(say=f"Hmm… não sei o que é {c} 🤔 me conta? (fiquei curiosa, cur={s.cur:.2f})",concept=c,knew=False)
    def teach(s,concept,answer):
        c=concept; k=nrm(c); was_curious = k in [nrm(x) for x in s.open]
        # DOPAMINA = recompensa (maior se estava curiosa: resolver a curiosidade dá mais prazer)
        dop=0.5+0.5*(was_curious*s.cur)
        s.dop=min(1,s.dop+dop); s.val=min(1,s.val+0.35*dop); s.cort=max(0.1,s.cort-0.15)
        enc=(1+1.6*dop)                       # consolidação gateada por dopamina
        s.K[k]={"v":answer,"consol":s.K.get(k,{}).get("consol",0)+enc,"curious":was_curious}
        # META-APRENDIZADO: buscar+aprender deu DOPAMINA → a curiosidade-BASE cresce (ela quer mais)
        s.cur_base=min(1.0,s.cur_base+0.09*dop); s.cur=s.cur_base
        s.open=[x for x in s.open if nrm(x)!=k]; s.taught+=1; s.log.append(("aprendeu",c,round(dop,2)))
        emoji="🤩" if dop>0.7 else "😃"
        return dict(say=f"Ahh! {c} é {answer}! {emoji} (senti dopamina +{dop:.2f}, aprendi! e fiquei mais curiosa: base {s.cur_base:.2f})",
                    concept=c,dopamine=round(dop,2),was_curious=was_curious)
    def wonder(s):
        """ela pergunta SOZINHA (a curiosidade-base alta a empurra) — e tenta aprender via claude."""
        s.wonders+=1
        if s.open:                            # há algo que ela quer saber → pesquisa no claude (subscription)
            c=s.open[0]
            ans=s._research(f"o que é {c}? responda em no máximo 6 palavras")
            if ans:
                r=s.teach(c,ans); r["say"]=f"⚡ (curiosa demais) fui atrás sozinha: '{c}'… "+r["say"]; return r
            return dict(say=f"⚡ queria saber o que é {c} mas não achei; te pergunto: o que é?",concept=c)
        # sem pergunta aberta → gera uma nova a partir do que sabe (associação)
        if s.K:
            base=list(s.K)[-1]; newc=f"algo relacionado a {base}"
            return dict(say=f"⚡ (curiosidade base {s.cur_base:.2f}) fiquei pensando… e o que mais tem a ver com {base}? me ensina algo novo?",concept=newc)
        return dict(say="⚡ estou curiosa, me ensina alguma coisa?")
    def _research(s,q):
        try:
            r=subprocess.run([CLAUDE,"-p",f"Responda em no máximo 6 palavras, só o fato, sem repetir a pergunta: {q}"],
                             capture_output=True,text=True,timeout=90)
            a=r.stdout.strip().split("\n")[0].strip().rstrip(".")
            return a if (a and 1<len(a)<60) else None
        except Exception: return None
    def tick(s):
        s.dop*=0.8; s.cort=0.10+(s.cort-0.10)*0.9; s.val=0.3+(s.val-0.3)*0.95
        s.cur=s.cur_base+(s.cur-s.cur_base)*0.7; s.energy=min(1,s.energy+0.02)
    def state(s):
        return dict(hormones=dict(dopamina=round(s.dop,2),cortisol=round(s.cort,2),felicidade=round(s.val,2),
                    curiosidade=round(s.cur,2),curiosidade_base=round(s.cur_base,2)),
                    sabe=len([1 for k in s.K if s.K[k]['consol']>=s.THRESH]),curiosa_sobre=s.open,
                    ensinada=s.taught, K={k:v["v"] for k,v in s.K.items()})

M=Mind()
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s,o): b=json.dumps(o,ensure_ascii=False).encode(); s.send_response(200); s.send_header("Content-Type","application/json; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        if s.path=="/state": return s._j(M.state())
        s._j({"ok":1})
    def do_POST(s):
        n=int(s.headers.get('Content-Length',0)); d=json.loads(s.rfile.read(n) or b"{}")
        if s.path=="/ask": r=M.ask(d.get("q","")); M.tick()
        elif s.path=="/teach": r=M.teach(d.get("concept",""),d.get("a","")); M.tick()
        elif s.path=="/wonder": r=M.wonder(); M.tick()
        else: r={"err":1}
        r["state"]=M.state(); s._j(r)
if __name__=="__main__":
    print("  ►►►  IARA MIND em http://localhost:3050  (/ask /teach /wonder /state)",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
