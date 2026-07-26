#!/usr/bin/env python3
"""TBH-VULNSCAN — varredura de vulnerabilidades (disclosure). O Claude analisa cada subsistema sensível
(reconstruído da RE) e aponta falhas prováveis num jogo online client-authoritative, com severidade,
como verificar, e a correção server-side. Paralelo. Gera TBH_VULN_FINDINGS.md. NÃO gera exploit — só análise defensiva."""
import json,re,os,sys,threading,queue,subprocess,time
HERE=os.path.dirname(os.path.abspath(__file__))
D=json.load(open(os.path.join(HERE,"tbh_map.json"))); C=D["classes"]
ANN=json.load(open(os.path.join(HERE,"tbh_annotated.json"))) if os.path.exists(os.path.join(HERE,"tbh_annotated.json")) else {}
CLAUDE="/home/leonardo/.local/bin/claude"; MODEL="sonnet" if "--sonnet" in sys.argv else "haiku"
NW=6
game=[c for c in C if c["asm"]=="Assembly-CSharp.dll" and not c["name"].startswith("<")]
def full(c): return (c["ns"]+"." if c["ns"] else "")+c["name"]
# subsistemas sensíveis à segurança (onde moram bugs de client-authority)
SEC=r"save|account|secret|anticheat|obscured|verify|hash|integrity|box|chest|drop|reward|kill|monster|spawn|damage|attack|combat|stat|attribute|currency|gold|price|shop|purchase|inventory|item|trade|market|steam|backend|request|abuse|guard|validate|sync|mail|claim|enchant|alchemy|craft|offline"
sel=[]
seen=set()
for c in game:
    if (re.search(SEC,c["name"],re.I) or re.search(SEC,c["ns"],re.I)):
        key=c["name"].lower()
        if key in seen: continue
        seen.add(key); sel.append(c)
# prioriza os que têm anotação (contexto melhor) e campos
sel=[c for c in sel if full(c) in ANN or c["fields"]]
_lock=threading.Lock(); FIND=[]
def prompt(c):
    fn=full(c); a=ANN.get(fn,{})
    fields=[f for f in c["fields"] if not f.startswith("<")][:20]
    return (f"Você é pesquisador de segurança de jogos online. Este jogo (Unity, idle-RPG ONLINE com TheBackend BaaS + "
            f"Steam trading + CodeStage AntiCheat) resolve MUITA coisa no CLIENTE. Analise a classe abaixo e diga se ela "
            f"sugere alguma VULNERABILIDADE de client-authority / validação-fraca (ex.: servidor confiar em evento do cliente, "
            f"rate-limit só no cliente, valor editável, save tamperável, dupe de item, bypass de checagem).\n\n"
            f"Classe: {fn}\nPropósito: {a.get('purpose','?')}\nCampos: {', '.join(fields) or '(nenhum)'}\n\n"
            f"Se houver risco real, responda SÓ JSON: {{\"vuln\":true,\"titulo\":\"...\",\"severidade\":\"baixa|media|alta\","
            f"\"descricao\":\"1-2 frases do problema\",\"verificar\":\"como confirmar\",\"correcao\":\"fix server-side\"}}. "
            f"Se for benigno/puramente visual, responda {{\"vuln\":false}}.")
def worker(q,done):
    while True:
        try: c=q.get_nowait()
        except queue.Empty: return
        try:
            r=subprocess.run([CLAUDE,"-p","--model",MODEL,prompt(c)],capture_output=True,text=True,timeout=150)
            m=re.search(r"\{.*\}",(r.stdout or ""),re.S)
            j=json.loads(m.group(0)) if m else {}
            if j.get("vuln"):
                j["classe"]=full(c)
                with _lock: FIND.append(j)
        except Exception: pass
        with _lock: done[0]+=1
        q.task_done()
q=queue.Queue(); [q.put(c) for c in sel]; done=[0]
print(f"[analisando {len(sel)} classes sensíveis · {NW} workers · {MODEL}]",flush=True)
def mon():
    while done[0]<len(sel):
        time.sleep(15); print(f"  {done[0]}/{len(sel)} · {len(FIND)} riscos até agora",flush=True)
threading.Thread(target=mon,daemon=True).start()
ths=[threading.Thread(target=worker,args=(q,done),daemon=True) for _ in range(NW)]
[t.start() for t in ths]; [t.join() for t in ths]
sev={"alta":0,"media":1,"baixa":2}
FIND.sort(key=lambda f:sev.get(f.get("severidade","baixa"),3))
L=["# TaskbarHero — achados de segurança (varredura Claude sobre a RE)\n",
   f"> {len(FIND)} riscos candidatos em {len(sel)} classes sensíveis. Disclosure responsável — análise, sem exploit.\n"]
for f in FIND:
    L.append(f"\n## [{f.get('severidade','?').upper()}] {f.get('titulo','?')}\n")
    L.append(f"- **classe:** `{f['classe']}`\n- **problema:** {f.get('descricao','')}\n")
    L.append(f"- **verificar:** {f.get('verificar','')}\n- **correção:** {f.get('correcao','')}\n")
open(os.path.join(HERE,"TBH_VULN_FINDINGS.md"),"w").write("".join(L))
json.dump(FIND,open(os.path.join(HERE,"tbh_findings.json"),"w"),ensure_ascii=False,indent=1)
print(f"[FIM · {len(FIND)} riscos → TBH_VULN_FINDINGS.md]")
