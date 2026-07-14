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
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
VLID="Qwen/Qwen2.5-VL-3B-Instruct"   # UM modelo: córtex (raciocínio) + olho (visão) + fonte dos pesos/neurônios
HERE=os.path.dirname(os.path.abspath(__file__)); MODEL=VLID
STATE=os.path.join(HERE,"iara_cerebro_state.json"); DEEP=8
WN={"zero":0,"um":1,"uma":1,"dois":2,"duas":2,"tres":3,"quatro":4,"cinco":5,"seis":6,"sete":7,"oito":8,"nove":9,"dez":10,"cem":100,"mil":1000}
STOP_KEY={"qual","que","quem","onde","quando","como","porque","por","uma","para","com","dos","das","the","and","voce","isso","essa","esse","capital","cor","nome","the","this","that"}
GAP_SKIP={"bombeia","recebe","fica","ficam","leva","levam","traz","trazem","passa","impede","impedem","envolve","conduz","derrete","espalha","espalham","irriga","irrigam","enche","enchem","bate","batem","chega","chegam","volta","voltam","aperta","empurra","saem","entra","acontece",
 "pobre","rico","forte","natural","proprio","propria","todo","toda","longe","dentro","cerca","mais","menos","primeiro","fora","depois","perto","grande","grandes","pequena","principal"}  # verbos/adjetivos: NÃO são conceitos-lacuna
PRED=r"(?:é a liga de|é feito de|é feita de|são feitas de|são construídas de|é a|é o|é uma|é um|é|são|fica entre|ficam entre|fica no|fica na|fica|bombeia para|bombeia|leva para|leva|levam|traz para|traz|trazem|recebe|recebem|passa para|passa|espalha pelo|espalha|irriga|irrigam|enche de|enche|envolve|sai do|sai da|impede|bate|usa|controla|separa)"
SKIP_OBJ={"mais","muito","cerca","vez","vezes","proprio","propria","todo","toda","primeiro","longe","dentro","fora","maior","menor","forte","natural","por","minuto","litros","graus","volta","sinal","lugar","modo","forma"}
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
        s.auto=False; s.autofeed=[]; s.auto_every=40; s.gap_now=None          # modo autônomo (professor Claude)
        s._load_state(); threading.Thread(target=s._load,daemon=True).start()
        threading.Thread(target=s._autoloop,daemon=True).start()
    def _load(s):
        t=time.time(); print("[carregando cérebro UNIFICADO: Qwen2.5-VL-3B (córtex + olho + pesos)…]",flush=True)
        s.proc=AutoProcessor.from_pretrained(VLID); s.tok=AutoTokenizer.from_pretrained(VLID)
        m=AutoModelForImageTextToText.from_pretrained(VLID,dtype=torch.float16).to("cuda").eval()
        s.model=m; s.vlm=m; s.vlmproc=s.proc                        # MESMO modelo raciocina E vê
        LM=m.model.language_model                                    # o transformer de texto dentro do VL
        s.E=m.get_input_embeddings().weight.detach(); s.nw=LM.norm.weight.detach(); s.Et=s.E.t().contiguous()
        V=s.E.shape[0]; toks=s.tok.convert_ids_to_tokens(list(range(V)))
        raw=[(i,(toks[i] or "").replace("Ġ","")) for i in range(V) if re.fullmatch(r"Ġ?[A-Za-z][a-zA-Z]{2,}",toks[i] or "")]
        norms=s.E[[i for i,_ in raw]].float().norm(dim=1); med=float(norms.median())
        s.clean=[];s.words=[]
        for (i,w),nm in zip(raw,norms.tolist()):
            if re.search(r"[a-z][A-Z]",w) or not(0.55*med<nm<1.8*med): continue
            s.clean.append(i);s.words.append(w)
        s.wmap={w.lower():s.clean[j] for j,w in enumerate(s.words)}; s.Ec=torch.nn.functional.normalize(s.E[s.clean].float(),dim=1)
        NL=len(LM.layers);INT=LM.layers[0].mlp.down_proj.weight.shape[1];vals=[];s.idx=[]
        for L in range(NL-DEEP,NL): dp=LM.layers[L].mlp.down_proj.weight.detach();vals.append((dp.t()*s.nw).contiguous());s.idx+=[(L,i) for i in range(INT)]
        s.vals=torch.cat(vals,0)
        s._fill_triples()                                          # extrai as conexões verdadeiras dos fatos já sabidos
        s.ready=True; print(f"[cérebro ONLINE em {time.time()-t:.0f}s · {len(s.clean)} conceitos · {s.vals.shape[0]} neurônios · Qwen2.5-VL-3B unificado]",flush=True)
    @torch.no_grad()
    def _vlm(s,img,q):
        msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
        p=s.proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=False)
        inp=s.proc(text=[p],images=[img],return_tensors="pt").to("cuda")
        out=s.vlm.generate(**inp,max_new_tokens=220,do_sample=False)
        r=s.proc.batch_decode(out[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
        return " ".join(r.split())[:600]
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
        v=torch.nn.functional.normalize(s.E[tid].float(),dim=0); sims=s.Ec@v
        top=torch.topk(sims,k+40).indices.tolist(); base=nrm(word); out=[]
        for j in top:
            w=s.words[j]; wn=nrm(w)
            if len(w)<4: continue                                   # sem fragmentos curtos (Ein, Rel)
            if wn==base or wn in base or base in wn: continue        # sem pedaço do próprio termo (relat, instein)
            if wn in [nrm(o[0]) for o in out]: continue
            out.append((w,round(float(sims[j]),3)))
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
        conv=[{"role":r["role"],"content":[{"type":"text","text":r["content"]}]} for r in messages]  # formato VL, só-texto
        txt=s.proc.apply_chat_template(conv,add_generation_prompt=True,tokenize=False)
        ins=s.proc(text=[txt],return_tensors="pt").to("cuda")
        out=s.model.generate(**ins,max_new_tokens=n,do_sample=False,pad_token_id=s.tok.eos_token_id)
        return s.proc.batch_decode(out[:,ins.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
    def _gather(s,q,cap=10):                                            # RECUPERAÇÃO MULTI-HOP (espalhamento de ativação — tese-água)
        qn=nrm(q); qw=set(w for w in qn.split() if len(w)>=4 and w not in STOP_KEY)
        rare=lambda W:{w for w in W if w not in s.wmap}                  # termos distintivos do domínio (guiam o 2º salto)
        ent={k:set(w for w in nrm(k+" "+v["stmt"]).split() if len(w)>=4 and w not in STOP_KEY) for k,v in s.seeds.items()}
        chosen=[]; used=set(); hops=[]; frontier=qw
        for hop in range(2):
            scored=[]
            for k,v in s.seeds.items():
                if k in used: continue
                e=ent[k]; sc=2*len(rare(frontier)&rare(e))+len(frontier&e)+(2 if k in qn else 0)
                if sc>0: scored.append((sc,k,v["stmt"]))
            scored.sort(key=lambda x:-x[0]); take=scored[:max(2,cap//2)]
            for sc,k,st in take: chosen.append(st); used.add(k)
            if take: hops.append((hop,[k for _,k,_ in take]))
            frontier=set().union(*[rare(ent[k]) for _,k,_ in take[:2]]) if take else set()  # segue a PONTE (raros dos 2 melhores)
            if len(chosen)>=cap or not frontier: break
        return chosen[:cap],hops
    def _reason(s,q,T):
        facts,hops=s._gather(q,cap=10)
        for hop,ks in hops: T.append(("HIPOCAMPO",f"ativação hop {hop} → sementes",", ".join(ks)))
        if facts:                                                       # ANCORA a resposta nos fatos (não inventa)
            T.append(("HIPOCAMPO","lê contexto multi-hop"," · ".join(facts)))
            sysmsg=("Você é a IARA, especialista. Use SOMENTE os fatos listados. Regras: "
                    "(1) NÃO invente números, nomes ou detalhes que não estejam escritos nos fatos; "
                    "(2) se a pergunta pedir algo que não está explícito nos fatos, responda exatamente 'não sei'; "
                    "(3) você PODE encadear/combinar fatos, passo a passo, para deduzir a resposta. Responda em 1 frase curta.")
            u="Fatos:\n- "+"\n- ".join(facts)+f"\n\nPergunta: {q}"
        else:
            sysmsg="Você é a IARA. Se souber com CERTEZA, responda em 1 frase curta; se tiver qualquer dúvida, responda exatamente 'não sei'."
            u=f"Pergunta: {q}"
        a=s._gen([{"role":"system","content":sysmsg},{"role":"user","content":u}]).split("\n")[0].strip()
        if not a or "não sei" in a.lower() or nrm(a).startswith("nao sei"): return None
        if facts:                                                       # VERIFICADOR: a resposta se sustenta nos fatos? (anti-confabulação)
            chk=s._gen([{"role":"system","content":"Você verifica ancoragem. Responda só 'sim' ou 'não'."},
                {"role":"user","content":"Fatos:\n- "+"\n- ".join(facts)+f"\n\nAfirmação: \"{a}\"\n\nTodo o conteúdo da afirmação (inclusive números e nomes) está escrito nos fatos ou é dedutível SÓ deles? Responda 'sim' ou 'não'."}],n=4)
            T.append(("VERIFICADOR","ancoragem nos fatos",chk.strip()[:20]))
            if chk.strip().lower().startswith(("n","não","nao")): return None  # não ancorada → abstém
        return a
    # ---- APRENDIZADO AUTÔNOMO (ela observa o que menos sabe; o professor Claude ensina) ----
    def _domain(s):                                                    # do que ela é especialista (tópico dominante)
        from collections import Counter
        c=Counter()
        for k,v in s.seeds.items():
            for w in nrm(k+" "+v["stmt"]).split():
                if len(w)>=5 and w not in STOP_KEY and w not in s.wmap: c[w]+=1
        return c.most_common(1)[0][0] if c else None
    def _gaps(s):                                                      # FRONTEIRA: OBJETOS das triplas citados mas NÃO definidos
        from collections import Counter
        subj=set(k for k,v in s.seeds.items() if v.get("kind")!="raciocinado"); lb=lambda k:re.sub(r"\s+\d+$","",k)
        def defined(o): return o in subj or any(o==lb(k) or (len(o.split())>=2 and o in k) for k in subj)
        ref=Counter()
        for v in s.seeds.values():                                     # 1) objetos das triplas (lacuna PRECISA)
            for o in (v.get("obj") or []):
                if len(o)>=4 and o not in GAP_SKIP and not defined(o): ref[o]+=3
        subjw=set();
        for k in subj: subjw|=set(w for w in nrm(k).split() if len(w)>=4)
        for v in s.seeds.values():                                     # 2) fallback: termos do domínio citados soltos
            for w in nrm(v["stmt"]).split():
                if len(w)>=4 and w not in STOP_KEY and w not in GAP_SKIP and w not in subjw: ref[w]+= (2 if w not in s.wmap else 1)
        return [w for w,_ in ref.most_common(12) if not defined(w)]
    def _professor(s,concept,domain):                                  # Claude via SUBSCRIPTION (claude -p) = professor externo
        import subprocess
        ctx="; ".join([v["stmt"] for v in list(s.seeds.values())[-6:]])
        prompt=(f"Você é professor de {domain}. O aluno já sabe: {ctx}. "
                f"Ensine UM fato NOVO, curto e verdadeiro sobre '{concept}' (ligado a esse tema), "
                f"em UMA frase em português começando pelo conceito. Responda só a frase, sem aspas nem introdução.")
        try:
            r=subprocess.run(["/home/leonardo/.local/bin/claude","-p",prompt],capture_output=True,text=True,timeout=120)
            line=(r.stdout or "").strip().split("\n")[0].strip().strip('"').rstrip(".")
            return line if 12<len(line)<240 and "?" not in line else None
        except Exception: return None
    def _autoteach(s):                                                 # UM ciclo: acha lacuna → professor ensina → grava
        if not s.ready or len(s.seeds)<2: return
        dom=s._domain() or "o assunto"; gaps=s._gaps()
        if not gaps: return
        concept=gaps[0]; s.gap_now=concept
        fact=s._professor(concept,dom)
        if not fact: return
        with s.lock:
            s._seed(concept,fact,"pesquisado"); s.dop=min(1,s.dop+0.35)
            s.autofeed.append(dict(t=time.strftime("%H:%M:%S"),gap=concept,fact=fact)); s.autofeed=s.autofeed[-40:]
            s.chat.append(("iara",f"🔎 não sabia bem sobre '{concept}' — o professor me ensinou: {fact}"))
        s.gap_now=None
    def _autoloop(s):
        while True:
            if getattr(s,"auto",False) and s.ready:
                try: s._autoteach()
                except Exception: pass
            for _ in range(max(5,int(s.auto_every))):
                time.sleep(1)
                if not getattr(s,"auto",False): break
    def _triple(s,stmt):                                            # extrai (relação, [objetos]) da frase → CONEXÃO VERDADEIRA
        z=re.sub(r"\s+"," ",stmt.strip().rstrip("."))
        m=re.match(rf"^(.{{1,55}}?)\s+({PRED})\s+(.+)$",z,re.I)
        if not m: return "",[]
        rel=re.sub(r"\s+(o|a|os|as)$","",m.group(2).lower().strip()); rest=m.group(3); objs=[]
        for part in re.split(r"\s+e\s+|,\s*|:\s*",rest):
            part=re.split(r"\s+que\s+",part)[0]                     # corta a subordinada ("...que volta do corpo")
            for chunk in re.split(r"\s+(?:para|de|do|da|no|na|ao|aos|à|às|em|pelo|pela|onde|conhecid\w*)\s+",part):
                c=re.sub(r"^(o|a|os|as|um|uma)\s+","",chunk.strip(),flags=re.I).strip()
                toks=[w for w in c.split() if len(w)>2 and nrm(w) not in SKIP_OBJ and not w.isdigit()]
                if toks and nrm(toks[0]) not in GAP_SKIP: objs.append(" ".join(toks[:2]))  # não começa com verbo/adjetivo
        seen=[]
        for o in objs:
            n=nrm(o)
            if len(n)>2 and n not in seen: seen.append(n)
        return rel,seen[:4]
    def _fill_triples(s):                                           # completa triplas faltantes (1x, persiste)
        ch=False
        for k,v in s.seeds.items():
            if "rel" not in v and v.get("kind")!="raciocinado":
                v["rel"],v["obj"]=s._triple(v["stmt"]); ch=True
        if ch: s._save_state()
    def _seed(s,concept,stmt,kind):
        k=nrm(concept)
        if kind=="raciocinado" and s.seeds.get(k,{}).get("kind")=="ensinado": return  # não sobrescreve fato ensinado
        ex=s.seeds.get(k)
        if ex and ex.get("stmt")!=stmt and ex.get("kind")=="ensinado":  # MESMO conceito, FATO diferente → não perde: chave única
            if kind=="ensinado":
                base=k;i=2
                while k in s.seeds: k=f"{base} {i}";i+=1
        rel,obj=s._triple(stmt)                                     # conexões verdadeiras (sujeito→relação→objetos)
        s.seeds[k]={"v":stmt,"stmt":stmt,"kind":kind,"t":time.strftime("%H:%M"),"rel":rel,"obj":obj}; s._save_state()
    def _key(s,text):                                               # chave = SUJEITO da pergunta (não palavra da resposta)
        z=text.strip().rstrip("?.! ")
        m=re.search(r"(?:o que (?:é|eh|são|sao)|quem (?:criou|inventou|é|foi|era)|qual(?: é| eh)?(?: a| o| as| os)?|onde (?:é|fica|foi)|quando (?:é|foi))\s+(?:a |o |as |os |da |do |de )?(.+)$",z,re.I)
        subj=(m.group(1) if m else z).strip()
        cand=[w for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}",subj) if nrm(w) in s.wmap and nrm(w) not in STOP_KEY]
        caps=[w for w in cand if w[0].isupper()]
        return caps[0] if caps else (cand[0] if cand else None)     # 1ª maiúscula (sujeito) ou 1ª palavra de conteúdo
    # ---- pipeline com TRACE ----
    def think(s,q):
        with s.lock:
            s.busy=True; s.chat.append(("voce",q)); T=[("SENSORIAL","recebeu texto",q)]; t0=time.perf_counter()
            # ROTEADOR + REFLEXO
            r=compute(q)
            if r:
                T.append(("ROTEADOR","é conta → REFLEXO (rápido)","")); T.append(("REFLEXO","matemática exata",f"{r[0]} = {r[1]}"))
                s.dop=min(1,s.dop+0.15); ans=f"{r[0]} = {r[1]}"; s._finish(T,t0,ans); return dict(say=ans)
            qs=q.strip().rstrip("."); words=qs.split()
            if "?" not in q and len(words)>=3:                      # modo ENSINO: chave = SUJEITO (antes do verbo), sem colidir
                VERBO=r"(?:é|eh|s[ãa]o|era|foi|significa|fica|ficam|tem|t[êe]m|recebe|recebem|bombeia|leva|levam|traz|trazem|passa|passam|espalha|espalham|irriga|irrigam|enche|enchem|bate|batem|impede|impedem|envolve|derrete|conduz|vive|aparece)"
                st=re.match(rf"^(.{{2,60}}?)\s+{VERBO}\b",qs,re.I); c=None
                if st: c=st.group(1).strip()
                elif qs[:1].isupper():
                    caps=re.findall(r"\b[A-ZÀ-Ý][A-Za-zÀ-ÿ]{2,}",qs); c=(caps[0] if caps else s._key(qs)) or words[0]
                if c:
                    c=re.sub(r"^(?:o|a|os|as|um|uma)\s+","",c.strip(),flags=re.I) or c   # tira artigo → chave distintiva
                    T.append(("ROTEADOR","é declaração → HIPOCAMPO")); s._seed(c,qs,"ensinado")
                    T.append(("HIPOCAMPO","gravou SEMENTE",qs)); s.dop=min(1,s.dop+0.4)
                    T.append(("NEUROMOD","dopamina ↑ (aprendeu)",f"{s.dop:.2f}")); ans=f"Entendi! {qs}"; s._finish(T,t0,ans); return dict(say=ans,concept=c)
            k=nrm(q)
            if k in s.seeds:                                       # match EXATO do conceito → reflexo (rápido)
                sv=s.seeds[k]; T.append(("ROTEADOR","conceito exato na memória → REFLEXO")); T.append(("REFLEXO","semente (rápido)",sv["stmt"])); s._finish(T,t0,sv["v"]); return dict(say=sv["v"],concept=s._key(k))
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
    def _rel_type(s,stmt,topic):                                    # que RELAÇÃO essa semente tem com o tópico
        z=nrm(stmt)
        if re.search(r"\b(criou|inventou|fisico|formulou|desenvolveu|descobriu|autor)\b",z): return "quem criou"
        if "formula" in z or "=" in stmt or re.search(r"\bmc\b",z): return "fórmula"
        if "capital" in z: return "capital"
        if re.search(r"\b(liga|construid|construida|construidas|feita|feito)\b",z): return "feito de"
        if re.search(r"\b(planeta|estrela|vive em|fica|localiza)\b",z): return "onde"
        if re.search(r"\b(1[6-9]\d\d|20\d\d)\b",z) or "ano" in z.split(): return "quando"
        if re.search(r"\b(curvatura|uniao|dimensoes|passar|associa|conjunto|especie)\b",z): return "conceito"
        return "o que é"
    @torch.no_grad()
    def _stmt_emb(s,txt):                                           # embedding semântico da frase (média das palavras de conteúdo)
        ids=[s.wmap[w] for w in dict.fromkeys(nrm(txt).split()) if w in s.wmap and w not in STOP_KEY and len(w)>=4]
        if not ids: return None
        return torch.nn.functional.normalize(s.E[ids].float().mean(0),dim=0)
    def graph(s):
        KG={"ensinado":"seed","visto":"vision","pesquisado":"auto"}
        seeds=[(k,v) for k,v in s.seeds.items() if v["kind"]!="raciocinado"]  # grafo = FATOS, não derivações
        nodes={};edges=[]
        if not s.ready:
            for k,v in seeds: nodes[k]=dict(id=k,label=k,group=KG.get(v["kind"],"seed"))
            return dict(nodes=list(nodes.values()),edges=edges)
        ent={k:set(w for w in nrm(k+" "+v["stmt"]).split() if len(w)>=4 and w not in STOP_KEY) for k,v in seeds}
        nov={k:set(w for w in ent[k] if w not in s.wmap) for k in ent}   # entidades PRÓPRIAS (fora do vocab: Kelbros, Zorvane…)
        emb={k:s._stmt_emb(v["stmt"]) for k,v in seeds}; keys=[k for k,_ in seeds]; sd=dict(seeds)
        parent={k:k for k in keys}
        def find(x):
            while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
            return x
        for i in range(len(keys)):                                  # une por SEMÂNTICA (embedding) OU nome-próprio compartilhado
            for j in range(i+1,len(keys)):
                a,b=keys[i],keys[j]; cos=float(emb[a]@emb[b]) if emb[a] is not None and emb[b] is not None else 0.0
                if cos>=0.5 or (nov[a]&nov[b]): parent[find(a)]=find(b)
        from collections import Counter
        lbl=lambda m:re.sub(r"\s+\d+$","",m)                        # rótulo limpo (tira sufixo "2")
        stem=lambda w:w[:-1] if len(w)>4 and w.endswith("s") else w  # singular≈plural (arterias≈arteria)
        clusters={}
        for k in keys: clusters.setdefault(find(k),[]).append(k)
        for members in clusters.values():
            cnt=Counter()                                           # rótulo do TÓPICO = entidade dominante (macro)
            for m in members:
                for w in ent[m]: cnt[w]+=1
            mx=max(cnt.values()); cands=[w for w,c in cnt.items() if c==mx]
            topic=sorted(cands,key=lambda w:((w in s.wmap),-len(w)))[0] if cands else members[0]
            if len(members)<2:
                m=members[0]; nodes[m]=dict(id=m,label=lbl(m),group=KG.get(sd[m]["kind"],"seed"),fact=sd[m]["stmt"],title=sd[m]["stmt"]); continue
            tid="◆"+topic; nodes[tid]=dict(id=tid,label=topic.upper(),group="topic")
            subj={m:set(stem(w) for w in nrm(m).split() if len(w)>=4 and w not in STOP_KEY and stem(w)!=stem(topic)) for m in members}
            sf=Counter()                                            # SUB-ÁREAS = raiz do sujeito compartilhada por ≥2 fatos (micro)
            for m in members:
                for e in subj[m]: sf[e]+=1
            subhubs=set(); MEMB={"dashes":True,"color":{"color":"#232a3a"},"font":{"size":0}}  # arestas de ESTRUTURA (sem rótulo)
            for m in members:
                nodes[m]=dict(id=m,label=lbl(m),group=KG.get(sd[m]["kind"],"seed"),fact=sd[m]["stmt"],title=sd[m]["stmt"])
                opts=[(sf[e],len(e),e) for e in subj[m] if sf[e]>=2]
                area=max(opts)[2] if opts else None
                if area:
                    sh="▸"+area
                    if sh not in subhubs:
                        subhubs.add(sh); nodes[sh]=dict(id=sh,label=area.upper(),group="subarea"); edges.append({"from":tid,"to":sh,"label":"",**MEMB})
                    edges.append({"from":sh,"to":m,"label":"",**MEMB})
                else:
                    edges.append({"from":tid,"to":m,"label":"",**MEMB})
        # ---- CONEXÕES VERDADEIRAS: triplas sujeito—relação→objeto; objeto não-definido = LACUNA (vermelho) ----
        skeys=[k for k,_ in seeds]
        def match(o):
            if o in skeys: return o
            for k in skeys:
                if o==lbl(k) or (len(o.split())>=2 and o in k): return k   # exato, ou frase (não single-word substring)
            return None
        for k,v in seeds:
            rel=v.get("rel") or "→"
            for o in (v.get("obj") or [])[:3]:
                tgt=match(o)
                if tgt and tgt!=k:
                    edges.append({"from":k,"to":tgt,"label":rel,"color":{"color":"#2f7f6e","highlight":"#5dcaa5"},"width":1.4})
                elif not tgt and len(o)>=4:
                    gid="?"+o
                    if gid not in nodes: nodes[gid]=dict(id=gid,label=o,group="gap",title="citado mas ainda NÃO definido (lacuna)")
                    edges.append({"from":k,"to":gid,"label":rel,"color":{"color":"#7a3550","highlight":"#e24b4a"},"width":1.2})
        return dict(nodes=list(nodes.values()),edges=edges)
    def _vitals(s):                                                 # SINAIS VITAIS do conhecimento
        facts={k:v for k,v in s.seeds.items() if v.get("kind")!="raciocinado"}
        subj=set(facts); lb=lambda k:re.sub(r"\s+\d+$","",k)
        def hit(o): return next((kk for kk in subj if o==kk or o==lb(kk) or (len(o.split())>=2 and o in kk)),None)
        def match(o): return hit(o) is not None
        refs=[]; adj={}
        for k,v in facts.items():
            outs=[]
            for o in (v.get("obj") or []):
                if len(o)>=4: refs.append(o)
                m=hit(o)
                if m and m!=k: outs.append(m)
            adj[k]=outs
        refs=set(refs); defined=[o for o in refs if match(o)]
        seen={}
        def depth(n,path):
            if n in path: return 0
            if n in seen: return seen[n]
            d=1+max([depth(m,path|{n}) for m in adj.get(n,[])]+[0]); seen[n]=d; return d
        prof=max([depth(k,set()) for k in adj]+[0])
        return dict(fatos=len(facts),ligacoes=sum(len(v.get("obj") or []) for v in facts.values()),
            conceitos=len(refs),definidos=len(defined),completude=round(100*len(defined)/max(1,len(refs))),
            profundidade=prof,lacunas=[o for o in refs if o not in defined][:10])
    def state(s):
        return dict(ready=s.ready,busy=s.busy,conceitos=len(getattr(s,'clean',[])) if s.ready else 0,neuronios=int(getattr(s,'vals',torch.zeros(0)).shape[0]) if s.ready else 0,
            chat=s.chat[-12:],trace=[list(t)+[""]*(3-len(t)) for t in s.trace],dop=round(s.dop,2),cort=round(s.cort,2),cur=round(s.cur,2),
            auto=s.auto,auto_every=s.auto_every,gap_now=s.gap_now,nfatos=len(s.seeds),autofeed=s.autofeed[-8:],vitais=s._vitals() if s.ready else None,
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
#auto{display:flex;align-items:center;gap:8px;font-size:11px}#auto b{color:#9fe1cb}
.sw{width:38px;height:20px;border-radius:11px;background:#25202a;border:1px solid #3a2f45;cursor:pointer;position:relative;transition:.2s}
.sw.on{background:#173a2b;border-color:#1d9e75}.sw i{position:absolute;top:1px;left:1px;width:16px;height:16px;border-radius:50%;background:#7a8290;transition:.2s}.sw.on i{left:19px;background:#5dcaa5}
.gap{font-size:11px;color:#ef9f27;margin-top:5px;min-height:14px}
.af{font-size:11px;line-height:1.35;margin:4px 0;border-left:2px solid #1d9e75;padding-left:6px;color:#cfe}.af b{color:#9fe1cb}.af .g{color:#ef9f27}
#det{font-size:12px;line-height:1.45;color:#cfe}#det .k{color:#9fe1cb;font-weight:700}#det .rel{color:#93a6c4}
.pill{display:inline-block;font-size:9px;padding:1px 6px;border-radius:8px;background:#12251d;border:1px solid #1d9e75;color:#9fe1cb;margin:1px}
.vg{display:flex;gap:6px;text-align:center}.vg .b{flex:1;background:#12151d;border:1px solid #1e2330;border-radius:6px;padding:5px 2px}.vg .n{font-size:18px;font-weight:700;color:#9fe1cb}.vg .l{font-size:8.5px;color:#7a8290}
.compbar{height:8px;background:#171b24;border-radius:4px;margin-top:6px;overflow:hidden}.compbar>i{display:block;height:100%;background:linear-gradient(90deg,#e24b4a,#ef9f27,#5dcaa5)}
#leg{position:fixed;bottom:8px;left:340px;font-size:10px;color:#9ab;background:#0d1016cc;border:1px solid #1e2330;border-radius:6px;padding:5px 8px;z-index:8}
#leg b{font-weight:400}.lz{display:inline-block;width:16px;height:0;border-top:2px solid;vertical-align:middle;margin-right:3px}
</style></head><body>
<div id=bar><h1>◉ IARA · CÉREBRO</h1><span id=stt>acordando…</span><span style=flex:1></span><span id=fatos style="color:#ba7517;font-size:11px"></span></div>
<div id=L>
 <div class=card><h2>sinais vitais do conhecimento</h2><div id=vitais></div></div>
 <div class=card><h2>aprendizado autônomo (professor Claude)</h2>
   <div id=auto><div class=sw id=sw onclick=toggleAuto()><i></i></div><span>modo automático — ela estuda o que menos sabe</span></div>
   <div class=gap id=gap></div>
   <div style="display:flex;gap:6px;margin-top:6px"><button onclick=teachNow() style="font-size:11px;padding:5px 8px">ensinar 1 agora</button><span style=color:#7a8290;font-size:10px;align-self:center>ciclo a cada ~40s</span></div>
 </div>
 <div class=card><h2>hormônios (neuromodulação)</h2><div id=horm></div></div>
 <div class=card><div id=drop onclick="f.click()">📷 arraste/clique uma imagem — o córtex visual lê (texto/tabela) e analisa freq/cores</div><input type=file id=f accept=image/* style=display:none onchange=upl()></div>
 <div id=chat></div>
 <div id=ci><input id=qi placeholder="fale, ensine ('X é Y') ou pergunte…" onkeydown="if(event.key=='Enter')ask()"><button onclick=ask()>enviar</button></div>
</div>
<div id=net></div>
<div id=leg><span class=lz style=border-color:#2f7f6e></span><b>relação verdadeira</b> &nbsp; <span class=lz style="border-color:#7a3550"></span><b>lacuna (não sabe)</b> &nbsp; <span class=lz style="border-color:#232a3a;border-top-style:dashed"></span><b>estrutura</b></div>
<div id=R>
 <div class=card style=border:0><h2>nó selecionado (clique no grafo)</h2><div id=det><span style=color:#666>clique num nó pra ver o fato e as relações</span></div></div>
 <div class=card style=border:0><h2>professor ↔ aluna (o que ela aprendeu sozinha)</h2><div id=feed></div></div>
 <div class=card style=border:0><h2>trace do último pensamento</h2><div id=trace></div></div>
</div>
<script>
let net,nodes,edges,gmap={},autoOn=false;
const opt={nodes:{shape:'dot',size:13,font:{color:'#dfe6f0',size:12,face:'monospace'},borderWidth:2},
 edges:{color:{color:'#33405c',highlight:'#5b8fd6'},smooth:{type:'continuous'},width:.8,arrows:{to:{enabled:true,scaleFactor:.4}},font:{color:'#93a6c4',size:9,face:'monospace',strokeWidth:3,strokeColor:'#090b10',align:'middle'}},
 groups:{topic:{shape:'hexagon',color:{background:'#20344f',border:'#5b8fd6'},size:30,font:{size:16,color:'#bcd8ff'},borderWidth:3},
  subarea:{shape:'diamond',color:{background:'#2a2340',border:'#8f7fd6'},size:17,font:{size:12,color:'#cbb8ff'},borderWidth:2},
  seed:{color:{background:'#ba7517',border:'#ef9f27'}},auto:{color:{background:'#0f6e56',border:'#5dcaa5'}},vision:{color:{background:'#0f6e56',border:'#1d9e75'}},reason:{color:{background:'#3c3489',border:'#7f77dd'}},
  gap:{shape:'dot',size:8,color:{background:'#3a2030',border:'#e24b4a'},font:{color:'#d98a94',size:10}},tree:{color:{background:'#1b2130',border:'#2a3550'},size:9}},
 physics:{barnesHut:{gravitationalConstant:-14000,springLength:120,springConstant:.03,centralGravity:.25,avoidOverlap:.3},stabilization:{iterations:150}},interaction:{hover:true,tooltipDelay:120}};
async function Po(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}
async function ask(){const v=qi.value.trim();if(!v)return;qi.value='';await Po('/think',{q:v});poll()}
async function upl(){const fl=f.files[0];if(!fl)return;const r=new FileReader();r.onload=async()=>{await Po('/see',{img:r.result});poll()};r.readAsDataURL(fl)}
async function toggleAuto(){autoOn=!autoOn;await Po('/auto',{on:autoOn});poll()}
async function teachNow(){await Po('/auto',{now:1});setTimeout(poll,500)}
async function poll(){try{const st=await(await fetch('/state')).json();render(st)}catch(e){}}
function showDet(id){const n=gmap[id];const det=document.getElementById('det');if(!n){det.innerHTML='<span style=color:#666>nó não encontrado</span>';return;}
 const rel=(edges.get()||[]).filter(e=>e.to===id&&e.label).map(e=>e.label);
 const kids=(edges.get()||[]).filter(e=>e.from===id).map(e=>gmap[e.to]?gmap[e.to].label:e.to);
 let h=`<div class=k>${n.label}</div>`;
 if(n.group==='topic')h+=`<div style=color:#7a8290;font-size:11px>tópico · ${kids.length} ligações</div>`;
 else if(n.group==='subarea')h+=`<div style=color:#7a8290;font-size:11px>sub-área · ${kids.length} fatos</div>`;
 else{if(rel.length)h+=`<div class=rel>relação: ${rel.map(r=>'<span class=pill>'+r+'</span>').join('')}</div>`;if(n.fact)h+=`<div style=margin-top:6px>${n.fact}</div>`;if(n.group==='auto')h+='<div style="color:#5dcaa5;font-size:10px;margin-top:4px">🔎 aprendido sozinha (professor)</div>';}
 if(kids.length&&(n.group==='topic'||n.group==='subarea'))h+=`<div style=margin-top:6px;color:#9ab;font-size:10px>${kids.slice(0,12).join(' · ')}</div>`;
 det.innerHTML=h;}
async function grefresh(){if(!net)return;const g=await(await fetch('/graph')).json();gmap={};g.nodes.forEach(n=>gmap[n.id]=n);
 const gid=new Set(g.nodes.map(n=>n.id));
 nodes.getIds().forEach(id=>{if(!gid.has(id))nodes.remove(id)});
 g.nodes.forEach(n=>nodes.update(n));
 const geid=new Set(g.edges.map(e=>e.from+'|'+e.to));
 edges.getIds().forEach(id=>{if(!geid.has(id))edges.remove(id)});
 g.edges.forEach(e=>{edges.update({id:e.from+'|'+e.to,...e})})}
function render(st){document.getElementById('stt').textContent=st.busy?'🧠 pensando…':(st.ready?`online · ${st.conceitos} conceitos · ${st.neuronios} neurônios`:'acordando os neurônios…');
 document.getElementById('fatos').textContent=st.ready?`${st.nfatos||0} fatos na memória`:'';
 if(!net&&st.ready){nodes=new vis.DataSet();edges=new vis.DataSet();net=new vis.Network(document.getElementById('net'),{nodes,edges},opt);net.on('click',p=>{if(p.nodes.length)showDet(p.nodes[0])})}
 autoOn=!!st.auto;document.getElementById('sw').className='sw'+(autoOn?' on':'');
 document.getElementById('gap').innerHTML=st.gap_now?`🔎 estudando agora: <b>${st.gap_now}</b> — perguntando ao professor…`:(autoOn?'observando o grafo pelo que menos conhece…':'desligado');
 horm.innerHTML=[['dopamina',st.dop,'#63c923'],['cortisol',st.cort,'#e24b4a'],['curiosidade',st.cur,'#ef9f27']].map(([n,v,c])=>`<div class=h><span>${n}</span><span>${v}</span></div><div class=tk><div class=tf style="width:${v*100}%;background:${c}"></div></div>`).join('');
 const vi=st.vitais;if(vi){document.getElementById('vitais').innerHTML=`<div class=vg><div class=b><div class=n>${vi.fatos}</div><div class=l>fatos</div></div><div class=b><div class=n>${vi.ligacoes}</div><div class=l>conexões</div></div><div class=b><div class=n>${vi.profundidade}</div><div class=l>saltos</div></div><div class=b><div class=n style=color:#ef9f27>${vi.completude}%</div><div class=l>completude</div></div></div><div class=compbar><i style=width:${vi.completude}%></i></div><div style="font-size:10px;color:#7a8290;margin-top:5px">sabe definir <b style=color:#9fe1cb>${vi.definidos}</b> de ${vi.conceitos} conceitos que cita · lacunas: <span style=color:#d98a94>${(vi.lacunas||[]).slice(0,5).join(', ')||'—'}</span></div>`;}
 feed.innerHTML=(st.autofeed||[]).slice().reverse().map(a=>`<div class=af><span style=color:#7a8290>${a.t}</span> não sabia <span class=g>${a.gap}</span><br><b>professor:</b> ${a.fact}</div>`).join('')||'<span style=color:#666>ligue o modo automático e ela vai estudar sozinha</span>';
 chat.innerHTML=(st.chat||[]).map(m=>`<div class="m ${m[0]}">${m[0]=='voce'?'você: ':'IARA: '}${m[1]}</div>`).join('');chat.scrollTop=chat.scrollHeight;
 trace.innerHTML=(st.trace||[]).map(t=>`<div class="tr r-${t[0]}"><b>${t[0]}</b> · ${t[1]}${t[2]?` <span class=d>${t[2]}</span>`:''}</div>`).join('')||'<span style=color:#666>mande uma mensagem…</span>';
 grefresh()}
setInterval(poll,900);poll();
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
        elif s.path=="/raw":                                        # Qwen CRU (sem memória) — p/ comparar
            a=C._gen([{"role":"system","content":"Responda em 1 frase curta e direta."},{"role":"user","content":d.get("q","")}]) if C.ready else "acordando…"
            r=dict(say=a)
        elif s.path=="/auto":                                       # liga/desliga o aprendizado autônomo
            if "on" in d: C.auto=bool(d["on"])
            if d.get("every"): C.auto_every=max(10,int(d["every"]))
            if d.get("now"): threading.Thread(target=C._autoteach,daemon=True).start()   # forçar 1 ciclo já
            r=dict(auto=C.auto,every=C.auto_every)
        else: r={"err":1}
        s._j(r)
if __name__=="__main__":
    threading.Thread(target=relax,daemon=True).start()
    print("  ►►►  IARA · CÉREBRO em http://localhost:3050",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
