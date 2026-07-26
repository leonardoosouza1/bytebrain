#!/usr/bin/env python3
"""TBH-ANNOTATE — o Claude LÊ cada classe do jogo e a de-ofusca por CONTEXTO.
Campos e nomes de classe estão limpos; só os métodos são 'dwk/eju'. O Claude infere,
do nome+campos+métodos-legíveis, o PROPÓSITO da classe e o que os métodos ofuscados fazem,
e propõe nomes reais. Junta com o middleware (fonte pública) → recria quase o projeto.
Uso: python3 tbh_annotate.py <regex_de_classe|sistema> [max] [--model haiku|sonnet]"""
import json,re,sys,subprocess,os
HERE=os.path.dirname(os.path.abspath(__file__))
D=json.load(open(os.path.join(HERE,"tbh_map.json"))); C=D["classes"]
game=[c for c in C if c["asm"]=="Assembly-CSharp.dll" and not c["name"].startswith("<")]
CACHE=os.path.join(HERE,"tbh_annotated.json")
ann=json.load(open(CACHE)) if os.path.exists(CACHE) else {}
CLAUDE="/home/leonardo/.local/bin/claude"
MODEL="haiku"
if "--model" in sys.argv: MODEL=sys.argv[sys.argv.index("--model")+1]
pat=sys.argv[1] if len(sys.argv)>1 else "SaveData"
mx=int([a for a in sys.argv[2:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[2:]) else 8
obf=lambda m: bool(re.match(r"^[a-zA-Z]{1,4}$",m))
rx=re.compile(pat,re.I)

def ask(c):
    full=(c["ns"]+"." if c["ns"] else "")+c["name"]
    fields=[f for f in c["fields"] if not f.startswith("<")][:24]
    readable=[m for m in c["methods"] if not obf(m) and not m.startswith((".","<","get_","set_"))][:20]
    obfn=[m for m in c["methods"] if obf(m)]
    prompt=(f"Você é engenheiro reverso de jogos Unity C#. Uma classe teve os NOMES DOS MÉTODOS ofuscados "
            f"(viraram 'dwk','eju'), mas o nome da classe e os CAMPOS estão intactos. Deduza a função da classe.\n\n"
            f"Classe: {full}\nCampos: {', '.join(fields) or '(nenhum)'}\n"
            f"Métodos legíveis: {', '.join(readable) or '(nenhum)'}\n"
            f"Métodos ofuscados: {len(obfn)} (ex: {', '.join(obfn[:8])})\n\n"
            f"Responda JSON: {{\"purpose\":\"1 frase do que a classe faz\",\"system\":\"save|combate|boss|skill|item|ui|rede|economia|player|outro\","
            f"\"key_fields\":\"o que os campos revelam\",\"guessed_methods\":[\"NomeProvavel: o que provavelmente faz\", ...até 5]}}. Só o JSON.")
    try:
        r=subprocess.run([CLAUDE,"-p","--model",MODEL,prompt],capture_output=True,text=True,timeout=120)
        t=(r.stdout or "").strip()
        m=re.search(r"\{.*\}",t,re.S)
        return json.loads(m.group(0)) if m else {"purpose":t[:200]}
    except Exception as e:
        return {"error":str(e)[:80]}

alvos=[c for c in game if rx.search(c["name"]) or rx.search(c["ns"])][:mx]
print(f"[{len(alvos)} classes · modelo {MODEL}]")
for c in alvos:
    full=(c["ns"]+"." if c["ns"] else "")+c["name"]
    if full in ann and "--force" not in sys.argv:
        a=ann[full]
    else:
        a=ask(c); ann[full]=a; json.dump(ann,open(CACHE,"w"),ensure_ascii=False,indent=1)
    print(f"\n### {full}")
    print(f"  propósito: {a.get('purpose','?')}")
    if a.get('system'): print(f"  sistema: {a['system']} · campos: {a.get('key_fields','')}")
    for g in a.get("guessed_methods",[])[:5]: print(f"   • {g}")
