#!/usr/bin/env python3
"""CURADOR BRASIL — ensina TUDO sobre o Brasil pra IARA via FILA + POOL DE WORKERS. 2026-07-16.
Pedido do Leonardo: 'suba N agents em paralelo, cada um cuida de um estado, quando acaba pega o próximo
da FILA (não concorrência que trava), sistema de fila e mensageria, resiliente, não param'.

Arquitetura:
  - FILA thread-safe de JOBS = (estado × aspecto) + temas nacionais (27 UF × 7 aspectos + ~10 nacionais ≈ 200 jobs).
  - POOL de N workers (default 6). Cada worker: pega job da fila → professor Haiku (claude -p, subscription) gera
    fatos LIMPOS → POST /curar na IARA (kind=pesquisado, alta confiança) → pega o PRÓXIMO. Nunca trava.
  - RESILIÊNCIA: timeout, retry com backoff+jitter, cooldown global se muita falha seguida; job volta pra fila.
  - MENSAGERIA: journal em markdown + progresso ao vivo + done-set persistido (resume se reiniciar).
Uso:  python iara_brasil_curador.py [n_workers]     (IARA precisa estar no ar em :3050)
"""
import sys, os, re, json, time, threading, queue, subprocess, urllib.request
from collections import Counter

CLAUDE="/home/leonardo/.local/bin/claude"
IARA="http://localhost:3050/curar"
HERE=os.path.dirname(os.path.abspath(__file__))
JOURNAL=os.path.join(HERE,"brasil_curadoria_journal.md")
DONE=os.path.join(HERE,"brasil_done.json")
NWORKERS=int(sys.argv[1]) if len(sys.argv)>1 else 6
MAXRETRY=4

# 27 unidades federativas (26 estados + DF) com capital
UF=[("Acre","Rio Branco"),("Alagoas","Maceió"),("Amapá","Macapá"),("Amazonas","Manaus"),
    ("Bahia","Salvador"),("Ceará","Fortaleza"),("Distrito Federal","Brasília"),("Espírito Santo","Vitória"),
    ("Goiás","Goiânia"),("Maranhão","São Luís"),("Mato Grosso","Cuiabá"),("Mato Grosso do Sul","Campo Grande"),
    ("Minas Gerais","Belo Horizonte"),("Pará","Belém"),("Paraíba","João Pessoa"),("Paraná","Curitiba"),
    ("Pernambuco","Recife"),("Piauí","Teresina"),("Rio de Janeiro","Rio de Janeiro"),("Rio Grande do Norte","Natal"),
    ("Rio Grande do Sul","Porto Alegre"),("Rondônia","Porto Velho"),("Roraima","Boa Vista"),
    ("Santa Catarina","Florianópolis"),("São Paulo","São Paulo"),("Sergipe","Aracaju"),("Tocantins","Palmas")]

# aspectos por estado — cada um vira um job
ASPECTS={
 "dados gerais":"a capital, a região, a população aproximada, a área em km², o gentílico, a sigla (UF), o ano de criação e os estados que fazem divisa",
 "principais cidades":"as principais cidades, dizendo de cada uma a população aproximada, a região do estado e por que é conhecida",
 "distâncias":"as distâncias rodoviárias aproximadas em km entre a capital e as principais cidades do estado e as capitais dos estados vizinhos",
 "geografia e natureza":"os principais rios, o relevo, o clima, o bioma predominante, serras, o litoral (se houver) e pontos naturais importantes",
 "história":"a formação e colonização do estado, quando foi criado e os fatos históricos mais marcantes",
 "economia":"as principais atividades econômicas, produtos agrícolas, indústrias, recursos e o que o estado mais produz",
 "cultura e turismo":"a culinária típica, festas populares, manifestações culturais, música e os principais pontos turísticos",
}
# temas nacionais (Brasil como um todo)
NACIONAIS={
 "símbolos nacionais do Brasil":"a bandeira (cores e significado), o hino, o brasão, as cores nacionais e o que representam",
 "regiões do Brasil":"as cinco regiões (Norte, Nordeste, Centro-Oeste, Sudeste, Sul), quais estados compõem cada uma e suas características",
 "geografia do Brasil":"os maiores rios, os biomas (Amazônia, Cerrado, Caatinga, Mata Atlântica, Pampa, Pantanal), os pontos extremos, o clima e o relevo",
 "história do Brasil":"o descobrimento, o período colonial, a independência, o império, a proclamação da república, a era Vargas e a redemocratização",
 "governo e política do Brasil":"a forma de governo, os três poderes, a Constituição, a capital, o número de estados e como funciona a federação",
 "economia do Brasil":"os principais setores, a moeda, as commodities exportadas, a indústria, o agronegócio e as maiores empresas",
 "cultura brasileira":"a música (samba, bossa nova, MPB, forró), a literatura, o carnaval, a culinária nacional, o futebol e as religiões",
 "população do Brasil":"a população total, a distribuição, as etnias e a miscigenação, as línguas, a expectativa de vida e as maiores cidades",
 "meio ambiente do Brasil":"a Floresta Amazônica, a biodiversidade, as unidades de conservação, os problemas ambientais e a fauna e flora típicas",
 "datas e feriados do Brasil":"os feriados nacionais, as datas comemorativas importantes e o que cada uma celebra",
}

_lock=threading.Lock()
_stats=Counter()          # done, facts, fail
_inflight={}              # worker_id -> job label
_fails_recent=[]          # timestamps de falhas (p/ cooldown)

def load_done():
    try: return set(json.load(open(DONE)))
    except Exception: return set()
def save_done(dn):
    with _lock:
        try: json.dump(sorted(dn),open(DONE,"w"),ensure_ascii=False)
        except Exception: pass
def jlog(msg):
    with _lock:
        with open(JOURNAL,"a",encoding="utf-8") as f: f.write(msg+"\n")

def build_prompt(job):
    tipo,nome,aspecto,guia=job[:4]
    if tipo=="uf":
        alvo=f"o estado brasileiro de {nome[0]} (capital {nome[1]})"
    else:
        alvo=nome
    return (f"Você é um especialista em geografia, história e dados do Brasil. "
            f"Liste o MÁXIMO de FATOS objetivos e verdadeiros sobre {aspecto} de {alvo}: {guia}. "
            f"REGRAS ESTRITAS: uma frase por linha; cada frase é uma AFIRMAÇÃO completa que começa pelo SUJEITO e contém um verbo "
            f"(ex.: 'A capital do Acre é Rio Branco.', 'O rio Amazonas corta o estado do Amazonas.', "
            f"'A distância de Salvador a Feira de Santana é de aproximadamente 108 km.'). "
            f"Inclua nomes, números, populações, áreas em km², distâncias em km. "
            f"NÃO use numeração, marcadores, títulos, markdown nem introdução. Só as frases, uma por linha. "
            f"Dê pelo menos 25 frases se souber.")

def ask_haiku(prompt,tmo=180):
    r=subprocess.run([CLAUDE,"-p","--model","haiku",prompt],capture_output=True,text=True,timeout=tmo)
    return (r.stdout or "").strip()

def parse_facts(txt):
    out=[]
    for ln in txt.split("\n"):
        ln=re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*","",ln).strip().strip('"').strip()
        if 15<=len(ln)<=260 and re.search(r"\b(é|são|era|foi|tem|têm|fica|possui|faz|limita|situa|localiza|pertence|nasceu|abriga|banha|corta|produz|celebra|inclui|conhecid\w*)\b",ln,re.I):
            out.append(ln)
    return out

def post_curar(facts):
    body=json.dumps({"facts":facts,"kind":"pesquisado"}).encode()
    req=urllib.request.Request(IARA,data=body,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=60) as r: return json.load(r)

def worker(wid,q,dn):
    while True:
        try: job=q.get_nowait()
        except queue.Empty: return
        label=job[4]
        if label in dn:
            q.task_done(); continue
        with _lock: _inflight[wid]=label
        ok=False
        try:
            # cooldown global se muitas falhas nos últimos 60s (protege a subscription)
            with _lock: recent=[t for t in _fails_recent if time.time()-t<60]
            if len(recent)>=NWORKERS*2: time.sleep(15+wid*2)
            txt=ask_haiku(build_prompt(job))
            facts=parse_facts(txt)
            if len(facts)>=5:
                res=post_curar(facts); add=res.get("added",0)
                with _lock: _stats["done"]+=1; _stats["facts"]+=add
                dn.add(label); save_done(dn)
                jlog(f"- ✅ **{label}** — {len(facts)} frases → {add} fatos curados (total IARA {res.get('total')})")
                ok=True
            else:
                jlog(f"- ⚠️ **{label}** — só {len(facts)} frases (resposta fraca), re-enfileirando")
        except subprocess.TimeoutExpired:
            jlog(f"- ⏱️ **{label}** — timeout, re-enfileirando")
        except Exception as e:
            jlog(f"- ❌ **{label}** — erro {type(e).__name__}: {str(e)[:80]}, re-enfileirando")
        finally:
            with _lock: _inflight.pop(wid,None)
        if not ok:
            with _lock: _fails_recent.append(time.time()); _stats["fail"]+=1
            att=job[5]+1
            if att<MAXRETRY:
                time.sleep(min(30,4*att)+wid)                      # backoff + jitter por worker
                nj=(job[0],job[1],job[2],job[3],label,att); q.put(nj)
            else:
                jlog(f"- 🛑 **{label}** — desistiu após {MAXRETRY} tentativas")
        q.task_done()

def monitor(total,q):
    while True:
        time.sleep(12)
        with _lock:
            d=_stats["done"]; fa=_stats["facts"]; fl=_stats["fail"]; infl=list(_inflight.values())
        print(f"[curador] {d}/{total} jobs · {fa} fatos curados · {fl} falhas/retries · em curso: {', '.join(infl) or '—'} · fila≈{q.qsize()}",flush=True)
        if d>=total: return

def main():
    # monta a fila
    jobs=[]
    for nome in UF:
        for asp,guia in ASPECTS.items(): jobs.append(("uf",nome,asp,guia,f"{nome[0]} · {asp}",0))
    for tema,guia in NACIONAIS.items(): jobs.append(("nat",tema,tema,guia,f"BR · {tema}",0))
    dn=load_done()
    q=queue.Queue()
    pend=[j for j in jobs if j[4] not in dn]
    for j in pend: q.put(j)
    total=len(jobs)
    print(f"[curador] {total} jobs ({len(UF)} UF × {len(ASPECTS)} aspectos + {len(NACIONAIS)} nacionais) · {len(dn)} já feitos · {NWORKERS} workers",flush=True)
    jlog(f"\n## Curadoria Brasil — {time.strftime('%Y-%m-%d %H:%M')} · {NWORKERS} workers · {len(pend)} jobs pendentes\n")
    with _lock: _stats["done"]=len(dn)
    mt=threading.Thread(target=monitor,args=(total,q),daemon=True); mt.start()
    ths=[threading.Thread(target=worker,args=(i+1,q,dn),daemon=True) for i in range(NWORKERS)]
    for t in ths: t.start()
    for t in ths: t.join()
    with _lock: d=_stats["done"]; fa=_stats["facts"]
    print(f"[curador] FIM · {d}/{total} jobs · {fa} fatos curados nesta rodada",flush=True)
    jlog(f"\n**FIM da rodada** — {d}/{total} jobs, {fa} fatos curados. IARA agora sabe o Brasil.\n")

if __name__=="__main__":
    main()
