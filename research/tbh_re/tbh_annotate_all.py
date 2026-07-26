#!/usr/bin/env python3
"""Roda o Claude em PARALELO (fila + workers) de-ofuscando TODAS as classes TaskbarHero.* do jogo.
Resiliente (retry/backoff), cacheia em tbh_annotated.json, resume se parar. Ao fim gera tbh_projeto.md
(o 'projeto recriado': cada classe com propósito + métodos inferidos). Uso: python3 tbh_annotate_all.py [nworkers] [--model haiku]"""
import json,re,os,sys,threading,queue,subprocess,time
HERE=os.path.dirname(os.path.abspath(__file__))
D=json.load(open(os.path.join(HERE,"tbh_map.json"))); C=D["classes"]
CACHE=os.path.join(HERE,"tbh_annotated.json")
CLAUDE="/home/leonardo/.local/bin/claude"
NW=int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 6
MODEL="haiku"
if "--model" in sys.argv: MODEL=sys.argv[sys.argv.index("--model")+1]
obf=lambda m: bool(re.match(r"^[a-zA-Z]{1,4}$",m))
game=[c for c in C if c["asm"]=="Assembly-CSharp.dll" and c["ns"].startswith("TaskbarHero") and not c["name"].startswith("<")]
_lock=threading.Lock()
ann=json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def full(c): return (c["ns"]+"." if c["ns"] else "")+c["name"]
def prompt(c):
    fields=[f for f in c["fields"] if not f.startswith("<")][:24]
    readable=[m for m in c["methods"] if not obf(m) and not m.startswith((".","<","get_","set_"))][:20]
    obfn=[m for m in c["methods"] if obf(m)]
    return (f"Você é engenheiro reverso de jogos Unity C#. Os NOMES DOS MÉTODOS foram ofuscados (viraram 'dwk'), "
            f"mas o nome da classe e os CAMPOS estão intactos. Deduza a função.\n\nClasse: {full(c)}\n"
            f"Campos: {', '.join(fields) or '(nenhum)'}\nMétodos legíveis: {', '.join(readable) or '(nenhum)'}\n"
            f"Métodos ofuscados: {len(obfn)} (ex: {', '.join(obfn[:8])})\n\n"
            f"Responda SÓ JSON: {{\"purpose\":\"1 frase\",\"system\":\"save|combate|boss|skill|item|ui|rede|economia|player|outro\","
            f"\"guessed_methods\":[\"Nome: o que faz\", ...até 5]}}")
def worker(q,done):
    while True:
        try: c=q.get_nowait()
        except queue.Empty: return
        fn=full(c)
        if fn in ann: q.task_done(); continue
        att=getattr(c,"_att",0)
        try:
            r=subprocess.run([CLAUDE,"-p","--model",MODEL,prompt(c)],capture_output=True,text=True,timeout=150)
            m=re.search(r"\{.*\}",(r.stdout or ""),re.S)
            a=json.loads(m.group(0)) if m else {"purpose":(r.stdout or "").strip()[:160]}
            with _lock:
                ann[fn]=a; done[0]+=1
                if done[0]%10==0: json.dump(ann,open(CACHE,"w"),ensure_ascii=False,indent=1)
        except Exception as e:
            if att<2: c._att=att+1; q.put(c); time.sleep(3+att*4)
            else:
                with _lock: ann[fn]={"error":str(e)[:60]}; done[0]+=1
        q.task_done()
def report():
    from collections import defaultdict
    bys=defaultdict(list)
    for c in game:
        a=ann.get(full(c))
        if a and "purpose" in a: bys[a.get("system","outro")].append((c,a))
    L=["# TaskbarHero — projeto recriado (Claude de-ofuscou por contexto)\n",
       f"> {sum(len(v) for v in bys.values())} classes TaskbarHero.* anotadas · campos reais + métodos inferidos\n"]
    for sysn in sorted(bys):
        L.append(f"\n## {sysn}  ({len(bys[sysn])})\n")
        for c,a in sorted(bys[sysn],key=lambda x:x[0]["name"]):
            L.append(f"### {full(c)}\n{a['purpose']}\n")
            if c["fields"]: L.append("- campos: "+", ".join([f for f in c['fields'] if not f.startswith('<')][:16])+"\n")
            for g in a.get("guessed_methods",[])[:5]: L.append(f"  - {g}\n")
    open(os.path.join(HERE,"tbh_projeto.md"),"w").write("".join(L))
    print(f"[relatório: tbh_projeto.md · {sum(len(v) for v in bys.values())} classes]")
if "--report" in sys.argv: report(); sys.exit()
todo=[c for c in game if full(c) not in ann]
print(f"[{len(game)} classes TaskbarHero.* · {len(todo)} pendentes · {NW} workers · {MODEL}]",flush=True)
q=queue.Queue(); [q.put(c) for c in todo]; done=[0]
def mon():
    while done[0]<len(todo):
        time.sleep(15); print(f"  {done[0]}/{len(todo)} anotadas",flush=True)
threading.Thread(target=mon,daemon=True).start()
ths=[threading.Thread(target=worker,args=(q,done),daemon=True) for _ in range(NW)]
[t.start() for t in ths]; [t.join() for t in ths]
json.dump(ann,open(CACHE,"w"),ensure_ascii=False,indent=1); report()
print("[FIM]")
