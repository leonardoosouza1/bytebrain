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
 "pobre","rico","forte","natural","proprio","propria","todo","toda","longe","dentro","cerca","mais","menos","primeiro","fora","depois","perto","grande","grandes","pequena","principal","enquanto","quando","porque","assim","entao"}  # verbos/adjetivos/conectivos: NÃO são conceitos-lacuna
PRED=r"(?:é a liga de|é feito de|é feita de|são feitas de|são construídas de|é a|é o|é uma|é um|é|são|fica entre|ficam entre|fica no|fica na|fica|bombeia para|bombeia|leva para|leva|levam|traz para|traz|trazem|recebe|recebem|passa para|passa|espalha pelo|espalha|irriga|irrigam|enche de|enche|envolve|sai do|sai da|impede|bate|usa|controla|separa)"
SKIP_OBJ={"mais","muito","cerca","vez","vezes","proprio","propria","todo","toda","primeiro","longe","dentro","fora","maior","menor","forte","natural","por","minuto","litros","graus","volta","sinal","lugar","modo","forma"}
# conceitos-LIXO: se a "cabeça" de um fato é uma destas, não é entidade — é conectivo/advérbio (fonte do "MAIS→Também")
CJUNK=set(("mais menos muito muitos muita muitas tambem assim porem contudo alem depois apos antes quando onde segundo cada toda todo todos todas "
  "outro outra outros outras varios varias alguns algumas apenas somente tanto tal tais este esta esse essa aquele aquela isso isto ele ela eles elas "
  "seu sua seus suas nesse neste nessa nesta porque pois entao logo ainda hoje ontem amanha agora aqui ali cerca durante enquanto embora caso tipo "
  "parte forma modo exemplo numero grupo serie conjunto meio comeco inicio final lado vez vezes ano anos dia dias mes meses hora horas "
  # preposições/conectivos e boilerplate de Wikipédia (senão viram falsos domínios: 'pela','através','responsável'…)
  "atraves pela pelo pelos pelas sobre desde conforme contra perante sendo entre acerca junto redor ambos ambas mediante consoante responsavel responsaveis responsavei denominado denominada denominados "
  "chamado chamada chamados conhecido conhecida conhecidos localizado localizada situado situada composto composta formado formada considerado considerada "
  "primeira primeiro primeiros segunda principal principais atual atualmente geralmente maioria menor maior grande grandes pequeno pequena termo palavra "
  "utilizado utilizada usado usada possui apresenta refere trata dentro atraves onde qual "
  "fica ficam localizase situase encontrase chamase denominase tornouse contem faz feito feita banha banhada corta serve possuem tem tinha havia existe existem "
  # metadados de estado (aparecem em TODO estado → escondem o nome do estado; excluir faz o ESTADO emergir como grupo)
  "habitante habitantes quilometro quilometros quilometrado distancia distancias distam gentilico gentilica sigla rodoviaria aproximada aproximadamente estimada quadrados kilometros "
  "provavelmente possivelmente normalmente especialmente principalmente aproximadamente basicamente realmente simplesmente frequentemente "
  "diversos diversas certos certas varias determinado determinada respectivamente inclusive portanto entretanto todavia "
  # numerais e genéricos que viram falsos domínios
  "dois duas tres quatro cinco seis sete oito nove dez onze doze vinte trinta cem cento mil milhao milhoes milhar bilhao "
  "mesmo mesma mesmos mesmas media medio importante importantes local locais geral gerais proprio propria novo nova novos novas velho antigo antiga "
  "melhor pior bons boa ultimo ultima ultimos primeiros varios varias total totais unico unica ambos ambas demais outrem alguem ninguem nada tudo").split())
def STEM(w):                                                       # normaliza plural PT → singular (animais→animal, espécies→especie)
    if len(w)<=4: return w
    for suf,rep in (("ais","al"),("eis","el"),("ois","ol"),("uis","ul"),("oes","ao"),("aes","ao")):
        if w.endswith(suf): return w[:-3]+rep
    if w.endswith("es") and len(w)>5: return w[:-2]                 # espécies→especi, ordens→orden
    if w.endswith("s"): return w[:-1]                               # plural simples
    return w
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
def digit_reflex(q):                                            # organ calculável (descoberta do Leonardo): raiz digital / atrator
    x=strip_acc(q).lower()
    m=re.search(r"raiz digital (?:de |do |da )?(\d+)",x)
    if m:
        n=int(m.group(1)); return (f"raiz digital de {n}", 1+(n-1)%9 if n else 0)
    m=re.search(r"(?:atrator|classe|para onde vai|orbita) (?:de |do |da )?(\d+)",x)
    if m:
        n=int(m.group(1)); seen=[]; v=n
        while v not in seen and len(seen)<20: seen.append(v); v=sum(int(c) for c in str(v*v))
        cyc=seen[seen.index(v):] if v in seen else [v]
        return (f"órbita de {n} (n→soma dos dígitos de n²)", " → ".join(map(str,seen[:6]))+f"  ⟳ atrator {sorted(set(cyc))}")
    return None

class Cerebro:
    def __init__(s):
        s.ready=False; s.busy=False; s.lock=threading.Lock()
        s.seeds={}; s.chat=[]; s.trace=[]; s.dop=0.2; s.cort=0.1; s.cur=0.3
        s.auto=False; s.autofeed=[]; s.auto_every=40; s.gap_now=None          # modo autônomo (professor Claude)
        s.jardim=None; s.jardimfeed=[]; s.jardim_on=False; s.jardim_every=120  # JARDINEIRO (cuida da floresta)
        s.vidafeed=[]; s.vida_on=False; s.vida_every=90                        # F4: a floresta se cura/religa/dorme sozinha
        s.fad=0.0                                                              # FADIGA (neuromodulador) — sobe com trabalho, força sono
        s.growth=[]; s.dev_on=False; s.dev_every=6; s.gtick=0                  # DESENVOLVIMENTO: neuromod CAUSA a plasticidade, floresta cresce reagindo
        s.env=[]                                                               # ambiente = fluxo de estímulos que ela reage
        s.fg=None; s.fg_ready=False                                           # motor-fluxo (voz byte-nativa: grafo+água+fala)
        s.persona=("Você é a IARA, uma inteligência artificial curiosa, calorosa e que adora aprender e conversar. "  # EDITÁVEL (/persona)
                   "Você está {mood}. Fale de forma NATURAL, amigável e BREVE (1-2 frases), como uma pessoa — com emoção e personalidade. "
                   "Nunca diga 'não sei' numa conversa social; seja gentil, humana e presente.")
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
        threading.Thread(target=s._build_flowgraph,daemon=True).start()      # a voz byte-nativa (motor-fluxo) em background
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
    def _gather(s,q,cap=10):                                            # FASE 2: RACIOCÍNIO ATRAVESSA A FLORESTA (entidade→relação→entidade, tese-água)
        F=s._ensure_forest(); ents=F["ents"]
        qn=nrm(q); toks=[w for w in qn.split() if w not in STOP_KEY]
        cand=set()                                                     # entidades-semente = palavras/bigramas da pergunta que SÃO entidades definidas
        for w in toks:
            if len(w)>=4: cand.add(s._cf(w))
        for i in range(len(toks)-1): cand.add(s._cf(toks[i]+" "+toks[i+1]))
        seeds=[c for c in cand if ents.get(c,{}).get("facts")]
        if not seeds: return s._gather_kw(q,cap)                        # nenhuma entidade reconhecida → cai no keyword
        kb=s.seeds
        qe=s._remb(q) if s.ready else None                             # embedding da PERGUNTA (mantém 'capital'/'são')
        emb={}                                                         # cache de embedding por fato
        def fe(st):
            if st not in emb: emb[st]=s._remb(st)
            return emb[st]
        def rel_of(k):                                                 # relevância semântica pergunta↔fato (+ leve bônus curado)
            v=kb.get(k,{}); e2=fe(v.get("stmt",""))
            cos=float(qe@e2) if (qe is not None and e2 is not None) else 0.0
            return cos+(0.12 if v.get("kind")!="bulk" else 0.0)
        primary=[]; secondary=set(); hops=[]; visited=set(seeds); frontier=list(dict.fromkeys(seeds))
        for hop in range(3):                                           # anda pela floresta: coração→sangue→oxigênio
            layer=[]; nxt=[]
            for e in frontier[:6]:
                d=ents.get(e)
                if not d: continue
                fks=sorted(d["facts"],key=lambda k:-rel_of(k))
                for fk in fks[:(6 if hop==0 else 3)]:
                    if hop==0: primary.append(fk)                     # fatos da ENTIDADE PERGUNTADA = sempre na frente (mesmo sem embedding)
                    else: secondary.add(fk)
                    layer.append(e)
                for r,oe,_ in d["out"][:5]:                            # segue os galhos → próximo salto
                    if oe not in visited and ents.get(oe,{}).get("facts"): visited.add(oe); nxt.append(oe)
            if layer: hops.append((hop,list(dict.fromkeys(layer))[:6]))
            frontier=nxt
            if not frontier or len(primary)+len(secondary)>=cap*3: break
        pri=list(dict.fromkeys(kb[k]["stmt"] for k in primary if k in kb))
        sec=[kb[k]["stmt"] for k in secondary if k in kb]
        kwf,_=s._gather_kw(q,max(5,cap//2))                            # HÍBRIDO: + keyword (pega fato preso em chave-lixo)
        rest=[st for st in sec+kwf if st not in pri]
        if qe is not None:                                             # re-rank SÓ o resto por embedding; a entidade perguntada fica garantida na frente
            rest.sort(key=lambda st:-(float(qe@fe(st)) if fe(st) is not None else 0.0))
        if kwf: hops.append((7,["busca-direta"]))
        return (pri[:6]+rest)[:cap+3],hops
    def _gather_kw(s,q,cap=10):                                         # (fallback) recuperação por sobreposição de palavras
        qn=nrm(q); qw=set(w for w in qn.split() if len(w)>=4 and w not in STOP_KEY)
        rare=lambda W:{w for w in W if w not in s.wmap}                  # termos distintivos do domínio (guiam o 2º salto)
        KB={k:v for k,v in s.seeds.items() if v.get("kind")!="raciocinado"}  # ancora só em FATOS (raciocínio não polui/deriva)
        ent={k:set(w for w in nrm(k+" "+v["stmt"]).split() if len(w)>=4 and w not in STOP_KEY) for k,v in KB.items()}
        chosen=[]; used=set(); hops=[]; frontier=qw
        for hop in range(2):
            scored=[]
            for k,v in KB.items():
                if k in used: continue
                e=ent[k]; sc=2*len(rare(frontier)&rare(e))+len(frontier&e)+(2 if k in qn else 0)
                if v.get("kind") in ("ensinado","pesquisado"): sc+=1.5   # CAMADA DE CONFIANÇA: curado > bulk (Wikipédia não atropela fato ensinado)
                if sc>0: scored.append((sc,k,v["stmt"]))
            scored.sort(key=lambda x:-x[0]); take=scored[:max(2,cap//2)]
            for sc,k,st in take: chosen.append(st); used.add(k)
            if take: hops.append((hop,[k for _,k,_ in take]))
            frontier=set().union(*[rare(ent[k]) for _,k,_ in take[:2]]) if take else set()  # segue a PONTE (raros dos 2 melhores)
            if len(chosen)>=cap or not frontier: break
        lbb=lambda k:re.sub(r"\s+\d+$","",k)                             # TRAVESSIA: puxa OUTROS fatos do MESMO conceito focal (2º salto real)
        bases={lbb(k) for k in used}; extra=[]
        for k,v in KB.items():
            if k not in used and lbb(k) in bases: extra.append(v["stmt"]); used.add(k)
        chosen+=extra[:3]
        if extra: hops.append((9,extra[:3]))
        return chosen[:cap+3],hops
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
        if facts:                                                       # ATALHO: se a resposta já está LITERAL nos fatos, está ancorada (não abstém à toa)
            at=set(w for w in nrm(a).split() if len(w)>=4 and w not in STOP_KEY)
            ft=set(nrm(" ".join(facts)).split())
            if at and len(at&ft)/len(at)>=0.6: T.append(("VERIFICADOR","ancorada (termos nos fatos)","sim")); return a
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
    def _gaps(s):                                                      # FRONTEIRA guiada por CENTRALIDADE (a lacuna que mais sustenta o grafo)
        from collections import defaultdict
        subj=set(k for k,v in s.seeds.items() if v.get("kind")!="raciocinado"); lb=lambda k:re.sub(r"\s+\d+$","",k)
        def defined(o): return o in subj or any(o==lb(k) or (len(o.split())>=2 and o in k) for k in subj)
        refby=defaultdict(set)                                          # centralidade barata = nº de FATOS distintos que citam a lacuna
        for k,v in s.seeds.items():
            if v.get("kind")=="raciocinado": continue
            for o in (v.get("obj") or []):
                if len(o)>=4 and o not in GAP_SKIP and not defined(o): refby[o].add(k)
        score={o:len(f)*3 for o,f in refby.items()}
        subjw=set()
        for k in subj: subjw|=set(w for w in nrm(k).split() if len(w)>=4)
        for v in s.seeds.values():                                     # fallback: termos do domínio citados soltos
            for w in nrm(v["stmt"]).split():
                if len(w)>=4 and w not in STOP_KEY and w not in GAP_SKIP and w not in subjw and not defined(w):
                    score[w]=score.get(w,0)+(2 if w not in s.wmap else 1)
        return [w for w,_ in sorted(score.items(),key=lambda x:(-x[1],x[0]))][:12]
    def _professor(s,concept,domain):                                  # Claude via SUBSCRIPTION (claude -p) = professor externo
        import subprocess
        ctx="; ".join([v["stmt"] for v in list(s.seeds.values())[-6:]])
        prompt=(f"Você é professor de {domain}. O aluno já sabe: {ctx}. "
                f"Ensine UM fato NOVO, curto e verdadeiro sobre '{concept}' (ligado a esse tema), "
                f"em UMA frase em português começando pelo conceito. Responda só a frase, sem aspas nem introdução.")
        try:
            r=subprocess.run(["/home/leonardo/.local/bin/claude","-p","--model","haiku",prompt],capture_output=True,text=True,timeout=120)  # Haiku on fire
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
    def _bulk_learn(s,keywords,target=20000):                       # APRENDER RÁPIDO: ingere milhares de fatos-frase do corpus
        import time as _t; t0=_t.time(); kws=[nrm(k) for k in keywords if k and k.strip()]; added=0
        LINK=re.compile(r"\b(é|eh|s[ãa]o|foi|era|tem|t[êe]m|fica|possui|contém|significa|nasceu|morreu|criou|inventou|localiza|serve|representa|refere|denomina|designa|consiste)\b",re.I)
        PRON=re.compile(r"^(Ele|Ela|Eles|Elas|Isso|Isto|Este|Esta|Esse|Essa|Aquele|Aquela|Seu|Sua|Seus|Suas|Nesse|Neste|Assim|Também|Porém|Contudo|Além|Já|Depois|Após|Quando|Onde|Segundo|Por|Com|Para|Não|Mais|Menos|Cada|Outro|Outra|Vários|Muitos|Muitas|Apenas|Somente|Tanto|Embora|Enquanto|Durante|Entre|Sobre|Até|Ainda|Logo|Então|Pois|Entretanto)\b")
        VERBO=r"(?:é|eh|s[ãa]o|era|foi|significa|tem|t[êe]m|fica|possui|contém|nasceu|criou|inventou|refere-se|denomina-se|consiste)"
        try: f=open("/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt",encoding="utf-8",errors="ignore")
        except Exception: return 0,0.0
        from collections import Counter
        base_cnt=Counter(re.sub(r"\s+\d+$","",k) for k in s.seeds)       # quantas variantes já existem por conceito-base
        buf=""; MAXVAR=4                                                 # no máx. 4 fatos por conceito-base (diversidade, não 100× "coração")
        while added<target:
            chunk=f.read(1<<20)
            if not chunk: break
            buf+=chunk; parts=re.split(r"(?<=[.!?])\s+",buf); buf=parts.pop()
            for sent in parts:
                sent=re.sub(r"\s+"," ",sent).strip(); w=sent.split()
                if not(6<=len(w)<=24) or not sent[:1].isupper() or PRON.match(sent): continue
                low=nrm(sent)
                if (kws and not any(k in low for k in kws)) or not LINK.search(sent): continue  # kws vazio = aprende de tudo (breadth)
                m=re.match(rf"^(.{{2,50}}?)\s+{VERBO}\b",sent,re.I); c=m.group(1).strip() if m else None
                if not c: continue
                c=re.sub(r"^(?:o|a|os|as|um|uma)\s+","",c.strip(),flags=re.I) or c; k=nrm(c)
                head=k.split()[0] if k else ""                          # ANTI-LIXO: cabeça não pode ser conectivo/advérbio (mata "mais→também")
                if len(k)<4 or head in CJUNK or head in STOP_KEY: continue
                if not any(len(t)>=4 and t not in CJUNK and t not in STOP_KEY for t in k.split()): continue  # precisa ≥1 palavra de conteúdo
                base=re.sub(r"\s+\d+$","",k)
                if base_cnt[base]>=MAXVAR: continue                     # já saturou esse conceito
                if k in s.seeds:
                    if s.seeds[k].get("stmt")==sent: continue
                    b=k;i=2
                    while k in s.seeds: k=f"{b} {i}";i+=1
                rel,obj=s._triple(sent)
                s.seeds[k]={"v":sent,"stmt":sent,"kind":"bulk","t":time.strftime("%H:%M"),"rel":rel,"obj":obj}; added+=1; base_cnt[base]+=1  # bulk = confiança MENOR
                if added>=target: break
        s._save_state(); return added, round(_t.time()-t0,1)
    def _curar(s,facts,kind="pesquisado"):                          # INGESTÃO CURADA em lote (agente professor) — thread-safe, SEM GPU, save 1x
        from collections import Counter
        VERBO=re.compile(r"\b(é|eh|s[ãa]o|era|foi|fica|ficam|tem|t[êe]m|possui|contém|faz|limita|situa|localiza|pertence|nasceu|significa|abriga|banha|corta|produz|cultiva|celebra|inclui|compreende|conhecid\w*)\b",re.I)
        add=0
        with s.lock:
            base_cnt=Counter(re.sub(r"\s+\d+$","",k) for k in s.seeds)
            for sent in facts:
                sent=re.sub(r"\s+"," ",str(sent)).strip().strip("-•*").strip()
                w=sent.split()
                if not(4<=len(w)<=40) or not sent[:1].isupper() or not VERBO.search(sent): continue
                m=re.match(r"^(.{2,60}?)\s+(?:é|eh|s[ãa]o|era|foi|fica|ficam|tem|t[êe]m|possui|contém|faz|limita|situa|localiza|pertence|nasceu|significa|abriga|banha|corta|produz|cultiva|celebra|inclui|compreende|conhecid\w*)\b",sent,re.I)
                c=m.group(1).strip() if m else None
                if not c: continue
                c=re.sub(r"^(?:o|a|os|as|um|uma|no|na|em|de|do|da)\s+","",c.strip(),flags=re.I) or c
                k=nrm(c); head=k.split()[0] if k else ""
                if len(k)<3 or head in CJUNK: continue
                base=re.sub(r"\s+\d+$","",k)
                if base_cnt[base]>=8: continue                      # curado permite mais variantes (8) que bulk (4)
                if k in s.seeds:
                    if s.seeds[k].get("stmt")==sent: continue
                    b=k;i=2
                    while k in s.seeds: k=f"{b} {i}";i+=1
                rel,obj=s._triple(sent)
                s.seeds[k]={"v":sent,"stmt":sent,"kind":kind,"t":time.strftime("%H:%M"),"rel":rel,"obj":obj}; add+=1; base_cnt[base]+=1
            if add: s._save_state()
        return add
    # ---- MOTOR-FLUXO: a VOZ byte-nativa (grafo do corpus+sementes = FATOS; fala do Qwen = FORMA) ----
    def _build_flowgraph(s):
        try:
            from collections import defaultdict, Counter
            import numpy as np
            corp="/home/leonardo/projects/LLM/bytebrain/data/pt_corpus_cortex.txt"
            txt=open(corp,encoding="utf-8",errors="ignore").read(12_000_000)+" "+" ".join(v["stmt"] for v in s.seeds.values())  # corpus + o que ELA sabe
            toks=re.findall(r"[A-Za-zÀ-ÿ]+|[.,;:!?]",txt); freq=Counter(toks); VOC={w for w,c in freq.items() if c>=8}
            sq=[t for t in toks if t in VOC]; ORD=[1,2,3]; tab={o:defaultdict(Counter) for o in ORD}; bi=defaultdict(Counter)
            for i in range(3,len(sq)):
                for o in ORD: tab[o][tuple(sq[i-o:i])][sq[i]]+=1
            for a,b in zip(sq,sq[1:]): bi[a][b]+=1
            for o in ORD: tab[o]={k:v for k,v in tab[o].items() if sum(v.values())>=2}
            s.fg=dict(tab=tab,bi=bi,freq=freq,VOC=VOC,ORD=ORD,np=np,defaultdict=defaultdict,Counter=Counter); s.fg_ready=True
            print(f"[voz byte-nativa pronta · {len(VOC)} palavras no grafo]",flush=True)
        except Exception as e: print("[flowgraph falhou]",str(e)[:80],flush=True)
    def _agua_fg(s,tema,r=0.15,it=20):
        fg=s.fg; np=fg["np"]; sd=fg["Counter"]()
        for w in re.findall(r"[A-Za-zÀ-ÿ]+",tema):
            if w in fg["VOC"]: sd[w]+=1
        Z=sum(sd.values()) or 1; sd={k:v/Z for k,v in sd.items()}; w=dict(sd)
        for _ in range(it):
            nw=fg["defaultdict"](float)
            for u,wu in w.items():
                c=fg["bi"].get(u)
                if wu<1e-6 or not c: continue
                t=sum(c.values())
                for v,cnt in c.items(): nw[v]+=(1-r)*wu*cnt/t
            for k,sv in sd.items(): nw[k]+=r*sv
            w=nw
        return w
    @torch.no_grad()
    def falar(s,tema,N=45,beta=1.6,alpha=1.1):                       # gera texto grounded (fluxo) + fluente (fala do Qwen)
        if not s.fg_ready: return dict(say="ainda montando a voz byte-nativa…")
        fg=s.fg; np=fg["np"]; from collections import deque
        w=s._agua_fg(tema); out=[x for x in re.findall(r"[A-Za-zÀ-ÿ]+",tema)[:2] if x in fg["VOC"]] or ["A"]; recent=deque(maxlen=10)
        wg=lambda x: w.get(x,1e-9)/(fg["freq"][x]+1)
        for _ in range(N):
            c=None
            for o in [3,2,1]:
                k=tuple(out[-o:])
                if len(k)==o and k in fg["tab"][o]: c=fg["tab"][o][k]; break
            if c is None: c=fg["bi"].get(out[-1]) or fg["Counter"]({".":1})
            cand=list(c.keys())[:36]; base=np.array([c[x] for x in cand],float)
            wa=np.array([wg(x) for x in cand]); rep=np.array([0.2 if x in recent else 1.0 for x in cand])
            gs=np.log(base/base.sum()+1e-9)+beta*np.log(wa+1e-12)+np.log(rep)
            lp=torch.log_softmax(s.model(input_ids=s.tok(" ".join(out),return_tensors="pt").input_ids.to("cuda")).logits[0,-1].float(),-1).cpu()  # FALA
            qs=np.array([float(lp[s.tok(" "+x,add_special_tokens=False).input_ids[0]]) if s.tok(" "+x,add_special_tokens=False).input_ids else -20 for x in cand])
            sc=gs+alpha*qs; p=np.exp(sc-sc.max()); p/=p.sum()
            nb=cand[int(np.random.choice(len(cand),p=p))]; w[nb]=w.get(nb,0)*0.4; recent.append(nb); out.append(nb)
        txt=re.sub(r"\s+([.,;:!?])",r"\1"," ".join(out))
        with s.lock: s.chat.append(("voce",f"[falar sobre: {tema}]")); s.chat.append(("iara",txt))
        return dict(say=txt)
    def _triple(s,stmt):                                            # extrai (relação, [objetos]) da frase → CONEXÃO VERDADEIRA
        z=re.sub(r"\s+"," ",stmt.strip().rstrip("."))
        m=re.match(rf"^(.{{1,55}}?)\s+({PRED})\s+(.+)$",z)          # case-SENSITIVE: "São"/"Santa" (nome próprio) NÃO viram o verbo "são"
        if not m: m=re.match(rf"^(.{{1,55}}?)\s+({PRED})\s+(.+)$",z,re.I)  # fallback p/ frases atípicas
        if not m: return "",[]
        rel=re.sub(r"\s+(o|a|os|as)$","",m.group(2).lower().strip()); rest=m.group(3); objs=[]
        SPLIT=r"\s+(?:para|de|no|na|ao|aos|à|às|em|pelo|pela|onde|conhecid\w*)\s+|\s+(?:do|da)\s+(?![Ss]ul|[Nn]orte|[Ll]este|[Oo]este)"  # não quebra 'Rio Grande do Sul'
        for part in re.split(r"\s+e\s+|,\s*|:\s*",rest):
            part=re.split(r"\s+que\s+",part)[0]                     # corta a subordinada ("...que volta do corpo")
            for chunk in re.split(SPLIT,part):
                c=re.sub(r"^(o|a|os|as|um|uma)\s+","",chunk.strip(),flags=re.I).strip()
                win=4 if re.match(r"[A-ZÀ-Ý]",c) else 2             # nome próprio (Maiúscula) preserva até 4 palavras ("mato grosso do sul")
                toks=[w for w in c.split() if len(w)>2 and nrm(w) not in SKIP_OBJ and not w.isdigit()]
                if toks and nrm(toks[0]) not in GAP_SKIP: objs.append(" ".join(toks[:win]))
        seen=[]
        for o in objs:
            n=nrm(o)
            if len(n)>2 and n not in seen: seen.append(n)
        return rel,seen[:4]
    def _reextract(s):                                             # re-deriva as triplas de TODOS os fatos com o extrator novo (conserta nomes próprios já gravados)
        ch=0
        for k,v in s.seeds.items():
            if v.get("kind")=="raciocinado": continue
            rel,obj=s._triple(v.get("stmt",""))
            if obj!=v.get("obj") or rel!=v.get("rel"): v["rel"],v["obj"]=rel,obj; ch+=1
        return ch
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
    def _social(s,q):                                              # é conversa (não pergunta factual nem ensino)?
        z=nrm(q)
        if re.search(r"\b(oi|ola|opa|eae|e ai|salve|tudo bem|tudo bom|bom dia|boa (tarde|noite)|como (vai|voce esta|esta voce|anda)|beleza|blz|prazer)\b",z): return True
        if re.search(r"\b(obrigad\w*|valeu|brigad\w*|agradec\w*|tchau|ate (mais|logo|breve)|falou|abraco\w*|adeus|foi um prazer|gostei (da|de) (conversa|papo))",z): return True
        if re.search(r"\bvoce (gosta|acha|sente|sentindo|fica|pensa|prefere|quer|queria|consegue|curte|ama|gostou|tem|sonha|imagina|acredita|espera|gostaria|adoraria|odeia|teme|lembra|se \w+)\b",z): return True
        if re.search(r"\b(quem e voce|o que voce (e|acha|sente|pensa)|seu nome|qual seu nome|como voce (se sente|se sentindo|esta|ta|anda|vai|funciona)|voce (ta|esta) (bem|feliz|triste|cansad\w*|se sentindo))\b",z): return True
        if re.search(r"\b(me (conta|fala|diz|explica|ajuda)|conta (uma|algo|mais|ai)|fala (sobre|mais|ai)|diz (algo|uma|ai)|manda (uma|ver))\b",z): return True
        return False
    @torch.no_grad()
    def _conversa(s,q,T):                                          # responde como uma pessoa (calorosa, curiosa, breve)
        mood="curiosa e animada" if s.cur>0.5 else ("cansada mas gentil" if s.cort>0.6 else "tranquila e atenta")
        z=nrm(q); ctx=""
        if re.search(r"conta|curiosidade|interessante|fala|diz|manda|ensina|sabe",z):  # quer que ela COMPARTILHE algo que sabe
            fatos=[v["stmt"] for v in list(s.seeds.values()) if v.get("kind") in ("ensinado","pesquisado")][-8:]
            if fatos: ctx="Coisas que você sabe (conte UMA de um jeito empolgado, com suas palavras):\n- "+"\n- ".join(fatos)+"\n\n"
        try: sysmsg=s.persona.format(mood=mood)                    # persona EDITÁVEL (/persona)
        except Exception: sysmsg=s.persona
        a=s._gen([{"role":"system","content":sysmsg},{"role":"user","content":ctx+q}],n=80).split("\n")[0].strip()
        T.append(("CÓRTEX","conversou (in-character)",a)); return a or None
    def think(s,q):
        with s.lock:
            s.busy=True; s.chat.append(("voce",q)); T=[("SENSORIAL","recebeu texto",q)]; t0=time.perf_counter()
            # ROTEADOR + REFLEXO
            dg=digit_reflex(q)                                      # organ calculável (raiz digital / atrator — 0 param) — ANTES da conta
            if dg:
                T.append(("ROTEADOR","é invariante → REFLEXO calculável","")); T.append(("REFLEXO","raiz digital / atrator",f"{dg[0]} = {dg[1]}"))
                s.dop=min(1,s.dop+0.1); ans=f"{dg[0]} = {dg[1]}"; s._finish(T,t0,ans); return dict(say=ans)
            r=compute(q)
            if r:
                T.append(("ROTEADOR","é conta → REFLEXO (rápido)","")); T.append(("REFLEXO","matemática exata",f"{r[0]} = {r[1]}"))
                s.dop=min(1,s.dop+0.15); ans=f"{r[0]} = {r[1]}"; s._finish(T,t0,ans); return dict(say=ans)
            qs=q.strip().rstrip("."); words=qs.split()
            soc=s._social(q)                                        # é conversa/social? (saudação, opinião, "me conta", agradecimento)
            if "?" not in q and len(words)>=3 and not soc:          # modo ENSINO: chave = SUJEITO (antes do verbo), sem colidir
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
        if soc:                                                    # CONVERSA: responde como gente (caloroso, in-character, sem abstenção)
            T.append(("ROTEADOR","conversa/social → CÓRTEX conversacional",""))
            try: a=s._conversa(q,T)
            except Exception: a=None
            with s.lock:
                if a: s.dop=min(1,s.dop+0.05); s._finish(T,t0,a); return dict(say=a)
                s._finish(T,t0,"oi! tô por aqui — quer conversar ou aprender algo?"); return dict(say="oi! tô por aqui — quer conversar ou aprender algo?")
        T.append(("ROTEADOR","não é reflexo → CÓRTEX PRÉ-FRONTAL (lento)")); T.append(("CÓRTEX","raciocinando com o Qwen…",""))
        try:
            a=s._reason(q,T)
        except Exception as e:
            with s.lock: T.append(("ERRO","córtex falhou",str(e)[:90])); s._finish(T,t0,"deu erro ao pensar"); return dict(say="erro no córtex")
        with s.lock:
            if a:
                T.append(("CÓRTEX","respondeu",a)); c=s._key(q) or s._key(a)
                if c: s._seed(c,f"{q} → {a}","raciocinado"); T.append(("HIPOCAMPO","gravou SEMENTE (raciocinada)",f"{c}"))
                s.dop=min(1,s.dop+0.3); s.fad=min(1,s.fad+0.03); T.append(("NEUROMOD","dopamina ↑",f"{s.dop:.2f}")); T.append(("VERIFICADOR","confiante → responde",""))
                for tk in nrm(q).split():                          # USOU o conhecimento → FORTALECE as sinapses que a sustentaram (Hebbian, gated pela dopamina)
                    ee=s._cf(tk); d=(getattr(s,"_forest",None) or {}).get("ents",{}).get(ee)
                    if d and d.get("facts"): s._reinforce(d["facts"][0],"dopamina")
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
        with s.lock: s.dop=0.2+(s.dop-0.2)*0.92; s.cort=0.1+(s.cort-0.1)*0.9; s.cur=0.3+(s.cur-0.3)*0.97; s.fad=max(0.0,s.fad-0.004)  # fadiga acumula (só o sono zera)
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
    _STOPE={"de","a","o","e","que","do","da","em","para","com","os","as","um","uma","no","na","por","dos","das","ao","à"}
    def _remb(s,txt):                                               # embedding p/ RETRIEVAL (mantém 'capital','são'… que o _stmt_emb corta)
        ids=[s.wmap[w] for w in dict.fromkeys(nrm(txt).split()) if w in s.wmap and w not in s._STOPE and len(w)>=3]
        if not ids: return None
        return torch.nn.functional.normalize(s.E[ids].float().mean(0),dim=0)
    # ========================= HIERARQUIA macro→micro (do coração até a biologia) =========================
    def _toks(s,k,v):                                               # palavras de CONTEÚDO de um fato (o que dá sentido ao agrupamento)
        seen=[]
        for w in nrm(k+" "+v.get("stmt","")).split():
            if len(w)<4 or w in STOP_KEY or w in CJUNK: continue
            r=STEM(w)
            if r not in seen: seen.append(r)
        return seen
    def _build_hierarchy(s):                                        # árvore por PALAVRA-MAIS-FREQUENTE recursiva: domínio→subárea→sub-sub→fato
        from collections import Counter
        allf=[(k,v) for k,v in s.seeds.items() if v.get("kind")!="raciocinado"]
        TK={k:set(s._toks(k,v)) for k,v in allf}                    # conteúdo por fato (cacheado no build)
        KIND={k:v.get("kind","bulk") for k,v in allf}
        STMT={k:v.get("stmt","") for k,v in allf}
        keys=[k for k,_ in allf]; import math
        N=max(1,len(keys)); MINGRP=max(4,min(25,N//1200))           # grupo mínimo ABSOLUTO (não infla com a escala → cobre muitos domínios)
        GDF=Counter()                                               # frequência GLOBAL de cada palavra (p/ distintividade)
        for m in keys:
            for w in TK[m]: GDF[w]+=1
        IDF={w:math.log(N/(1+c)) for w,c in GDF.items()}            # palavra rara = mais distintiva; comum = menos
        GEN={w for w,c in GDF.items() if c>0.15*N}                  # palavra em >15% dos fatos = genérica demais p/ virar domínio
        def conc_of(node):                                         # CONCEITOS do subárvore (p/ PONTES emergentes entre grupos)
            cc=Counter()
            for m in node["facts"]:
                for w in TK[m]-GEN: cc[w]+=1
            for ch in node["kids"].values():
                for w,c in ch.get("concepts",{}).items(): cc[w]+=c
            return dict(cc.most_common(120))
        def build(members,used,depth,path):
            node={"path":path,"n":len(members),"kids":{},"facts":[]}
            if depth>=5 or len(members)<2*MINGRP:                   # folha: os fatos ficam aqui
                node["facts"]=members; node["concepts"]=conc_of(node); return node
            dl=Counter()
            for m in members:
                for w in TK[m]-used:
                    if w not in GEN: dl[w]+=1
            mn=MINGRP                                               # limiar ABSOLUTO (não proporcional): muitos temas passam, não só os gigantes
            cand=sorted(((dl[w]*IDF[w],w) for w in dl if dl[w]>=mn),reverse=True)
            assigned=set()
            for sc,w in cand[:40]:
                grp=[m for m in members if m not in assigned and w in TK[m]]
                if len(grp)<MINGRP: continue
                assigned|=set(grp)
                node["kids"][w]=build(grp,used|{w},depth+1,(path+"/"+w if path else w))
                if len(node["kids"])>=40: break
            node["facts"]=[m for m in members if m not in assigned]  # o que não coube num filho fica como fato direto
            node["concepts"]=conc_of(node); return node
        root=build(keys,set(),0,"")
        curk=[k for k in keys if KIND[k]!="bulk"]                    # ★ o que VOCÊ ensinou (coração, relatividade, professor) = domínio próprio
        cur_root=build(curk,set(),0,"") if len(curk)>=2 else {"path":"","n":len(curk),"kids":{},"facts":curk}
        s._hier={"root":root,"cur":cur_root,"ncur":len(curk),"TK":TK,"KIND":KIND,"STMT":STMT,"idf":IDF,"gen":GEN,"n":len(keys)}
        return s._hier
    def _ensure_hier(s):                                            # (re)constrói só quando o nº de fatos muda >5% (barato no poll)
        h=getattr(s,"_hier",None); cur=sum(1 for v in s.seeds.values() if v.get("kind")!="raciocinado")
        if h is None or abs(cur-h["n"])>max(20,h["n"]*0.05): s._build_hierarchy()
        return s._hier
    def _resolve(s,path):                                           # caminha a árvore por "a/b/c" → nó
        n=s._hier["root"]
        if path:
            for step in path.split("/"):
                if step in n["kids"]: n=n["kids"][step]
                else: break
        return n
    def _canon(s,x):                                                # nome canônico BASE (unifica "brasil"/"Brasil"/"coracao 2")
        return re.sub(r"\s+\d+$","",nrm(x)).strip()
    def _cf(s,x):                                                   # canônico + FUSÃO de plural (usa o alias da floresta)
        b=s._canon(x); a=getattr(s,"_forest",None)
        return (a["alias"].get(b,b) if a and "alias" in a else b)
    def _junk_ent(s,e):                                             # entidade-fantasma? (erro de extração: "maior parte deles", "faz", "todo")
        t=e.split()
        if not t or len(t)>4: return True
        if e in {"sao","santa","santo","são","nossa","rio","mato","porto","campo","serra","vila","nova","novo","monte","belo","alto"}: return True  # cabeça-de-nome solta
        if t[0] in CJUNK or t[0] in STOP_KEY: return True
        return all(w in STOP_KEY or w in CJUNK or len(w)<3 for w in t)
    def _build_forest(s):                                           # FLORESTA: entidades (nós) + relações reais (arestas) — direto das triplas
        from collections import defaultdict,Counter
        seeds=[(k,v) for k,v in s.seeds.items() if v.get("kind")!="raciocinado"]
        raw=Counter()                                              # 1ª passada: acha plurais que têm singular → alias (fusão sem destruir dado)
        for k,v in seeds:
            raw[s._canon(k)]+=1
            for o in (v.get("obj") or []): raw[s._canon(o)]+=1
        def sing(e):
            if len(e)>5 and e.endswith("es"): return e[:-2]
            if len(e)>4 and e.endswith("s"):  return e[:-1]
            return None
        alias={}
        for e in raw:
            sg=sing(e)
            if sg and raw.get(sg,0)>=max(2,raw[e]) and sg!=e: alias[e]=sg   # funde no singular só se ele é ao menos tão comum
        HEAD={"sao","santa","santo","são","nossa","rio","mato","porto","campo","serra","vila","nova","novo","monte","belo","alto","cruz"}
        saintfull={}                                               # tail de "São/Santa/Santo X" → forma completa dominante
        for e in raw:
            t=e.split()
            if len(t)==2 and t[0] in HEAD:
                if raw[e]>saintfull.get(t[1],(0,None))[0]: saintfull[t[1]]=(raw[e],e)
        for e in list(raw):                                        # FUSÃO de nome próprio: "paulo"→"sao paulo", "catarina"→"santa catarina"
            if " " not in e and e in saintfull and e not in alias and saintfull[e][0]>=6: alias[e]=saintfull[e][1]
        cf=lambda x:(lambda b: alias.get(b,b))(s._canon(x))
        ents=defaultdict(lambda:{"facts":[],"out":[],"in":[],"cur":False})
        for k,v in seeds:
            subj=cf(k)
            if len(subj)<2 or s._junk_ent(subj): continue           # ignora sujeito-fantasma
            e=ents[subj]; e["facts"].append(k)
            if v.get("kind")!="bulk": e["cur"]=True
            rel=re.sub(r"\s+"," ",(v.get("rel") or "").strip())[:22]
            for o in (v.get("obj") or []):
                oe=cf(o)
                if len(oe)<2 or oe==subj or s._junk_ent(oe): continue   # ignora objeto-fantasma
                ents[subj]["out"].append((rel,oe,k)); ents[oe]["in"].append((subj,rel,k))
        deg={e:len(d["in"])+len(d["out"]) for e,d in ents.items()}
        s._forest={"ents":dict(ents),"deg":deg,"alias":alias,"n":len(seeds)}
        return s._forest
    def _ensure_forest(s):                                          # (re)constrói só quando muda >5% (barato no poll)
        f=getattr(s,"_forest",None); cur=sum(1 for v in s.seeds.values() if v.get("kind")!="raciocinado")
        if f is None or abs(cur-f["n"])>max(20,f["n"]*0.05): s._build_forest()
        return s._forest
    # ===================== JARDINEIRO: o organismo que CUIDA da floresta =====================
    def _gardener(s):                                              # poda ruído · funde plural · acha contradição — movido a hormônio, observável
        from collections import defaultdict
        import time as _t; t0=_t.time()
        BADHEAD=CJUNK|STOP_KEY|{"com","sua","seu","seus","suas","aproximadamente"}
        with s.lock:
            # 0) RE-EXTRAÇÃO: re-deriva as triplas com o extrator novo (conserta "São Paulo" que virava "paulo" etc.)
            reext=s._reextract()
            # 1) PODA: remove OBJETOS-LIXO das triplas (erros de extração que sujam o grafo) — não apaga o fato, só a aresta ruim
            pruned=0
            for k,v in s.seeds.items():
                if v.get("kind")=="raciocinado": continue
                objs=v.get("obj") or []; clean=[]
                for o in objs:
                    on=nrm(o); toks=on.split(); head=toks[0] if toks else ""
                    ok = on and len(on)>=3 and head not in BADHEAD and len(toks)<=3 and not on.replace(" ","").isdigit() \
                         and any(len(w)>=3 and w not in STOP_KEY and w not in CJUNK for w in toks)
                    if ok: clean.append(o)
                    else: pruned+=1
                if len(clean)!=len(objs): v["obj"]=clean
            # 2) IMUNIDADE: contradição em relação FUNCIONAL (1 sujeito deveria ter 1 valor: capital/moeda/sigla/gentílico)
            FUNC=re.compile(r"\b(capital|moeda|sigla|gentilic\w+)\b",re.I)
            val=defaultdict(set); ref={}
            for k,v in s.seeds.items():
                if v.get("kind")=="raciocinado": continue
                st=v.get("stmt","")
                if not FUNC.search(st): continue
                subj=s._cf(k)
                for o in (v.get("obj") or [])[:1]:
                    oe=s._cf(o)
                    if len(oe)>=3: val[subj].add(oe); ref.setdefault(subj,st)
            contras=[(sj,list(vs)[:4]) for sj,vs in val.items() if len(vs)>1]
            # 3) HORMÔNIOS + memória do que fez (observável)
            if contras: s.cort=min(1,s.cort+0.12)                  # contradição = estresse → sinaliza p/ reparo
            if pruned:  s.dop=min(1,s.dop+0.08)                    # limpeza feita = recompensa
            s._forest=None                                         # força reconstrução da floresta já podada
            rep=dict(pruned=pruned,reext=reext,contra=len(contras),contra_ex=contras[:6],dt=round(_t.time()-t0,1),
                     t=time.strftime("%H:%M:%S"),total_obj=sum(len(v.get("obj") or []) for v in s.seeds.values()))
            s.jardim=rep; s.jardimfeed=(getattr(s,"jardimfeed",[])+[rep])[-25:]
            if pruned: s._save_state()
        return rep
    def _jardimloop(s):                                            # o jardineiro trabalha sozinho de tempos em tempos
        import time as _t
        while getattr(s,"jardim_on",False):
            try: s._gardener()
            except Exception as e: print("jardim:",e,flush=True)
            for _ in range(int(getattr(s,"jardim_every",120))):
                if not getattr(s,"jardim_on",False): break
                _t.sleep(1)
    # ===================== F4: a floresta se CURA e RELIGA sozinha (organismo vivo) =====================
    def _vidafeed(s,ev):
        s.vidafeed=(getattr(s,"vidafeed",[])+[dict(t=time.strftime("%H:%M:%S"),**ev)])[-25:]
    def _heal(s):                                                  # CICATRIZAÇÃO: acha a lacuna mais 'load-bearing' (mais citada, sem definição) → professor define
        F=s._ensure_forest(); ents=F["ents"]
        gaps=sorted(((len(d["in"]),e) for e,d in ents.items() if not d["facts"] and len(e)>=4 and not s._junk_ent(e)),reverse=True)
        for _,concept in gaps[:5]:
            if concept in getattr(s,"_healed",set()): continue
            fact=s._professor(concept,s._domain() or "conhecimento geral")
            s._healed=getattr(s,"_healed",set())|{concept}
            if fact:
                s._curar([fact],"pesquisado"); s.dop=min(1,s.dop+0.3); s.cur=max(0,s.cur-0.08); s._forest=None
                s._vidafeed(dict(kind="cicatrizou",a=concept,fato=fact)); return dict(kind="cicatrizou",concept=concept,fact=fact)
        return None
    def _rewire(s):                                                # RELIGAÇÃO: dois conceitos que compartilham muitos vizinhos mas NÃO se ligam → professor diz a relação (nasce conhecimento)
        F=s._ensure_forest(); ents=F["ents"]
        cands=[e for e,d in ents.items() if d["cur"] and d["facts"] and not s._junk_ent(e)][:70]
        if len(cands)<2: cands=[e for e,_ in sorted(F["deg"].items(),key=lambda x:-x[1]) if ents[e]["facts"]][:70]
        nb={e:(set(oe for _,oe,_ in ents[e]["out"])|set(se for se,_,_ in ents[e]["in"])) for e in cands}
        tried=getattr(s,"_rewired",set()); best=None
        for i in range(len(cands)):
            for j in range(i+1,len(cands)):
                a,b=cands[i],cands[j]
                if (a,b) in tried: continue
                if b in nb[a] or a in nb[b]: continue              # já ligados
                sh=[w for w in (nb[a]&nb[b]) if w not in (a,b) and not s._junk_ent(w)]
                if len(sh)>=4 and (not best or len(sh)>best[0]): best=(len(sh),a,b,sh[:5])
        if not best: return None
        n,a,b,sh=best; s._rewired=tried|{(a,b)}
        fact=s._professor(f"a relação entre {a} e {b}",s._domain() or "conhecimento")
        if fact:
            s._curar([fact],"pesquisado"); s.dop=min(1,s.dop+0.25); s._forest=None
            s._vidafeed(dict(kind="religou",a=a,b=b,via=", ".join(sh[:3]),fato=fact)); return dict(kind="religou",a=a,b=b,shared=sh,fact=fact)
        return None
    def _sleep(s):                                                 # SONO: consolida o núcleo corroborado, esquece folhas fracas isoladas (Ebbinghaus, gentil)
        F=s._ensure_forest(); ents=F["ents"]; deg=F["deg"]
        core=sum(1 for e,d in ents.items() if d["facts"] and deg.get(e,0)>=8)   # entidades bem-corroboradas
        weak=[e for e,d in ents.items() if d["facts"] and deg.get(e,0)<=1 and not d["cur"]]  # folha isolada, bulk = candidata a esquecer
        forgot=0
        with s.lock:
            for e in weak[:120]:                                   # gentil: no máx. 120 por sono
                for k in list(ents[e]["facts"]):
                    if k in s.seeds and s.seeds[k].get("kind")=="bulk": del s.seeds[k]; forgot+=1
            if forgot: s._forest=None; s._save_state()
        s.cort=max(0,s.cort-0.2); s.dop=0.2+(s.dop-0.2)*0.9         # sono acalma
        s._vidafeed(dict(kind="dormiu",core=core,forgot=forgot)); return dict(kind="dormiu",core=core,forgot=forgot)
    def _life(s):                                                  # UM ciclo do organismo: rotaciona cicatrizar → religar → dormir (pela necessidade)
        s._vtick=getattr(s,"_vtick",0)+1
        if s.cort>0.5: return s._sleep()                           # estressada (contradição) → dorme/consolida
        return (s._heal() if s._vtick%2 else s._rewire()) or s._heal() or s._sleep()
    def _vidaloop(s):
        import time as _t
        while getattr(s,"vida_on",False):
            try: s._life()
            except Exception as e: print("vida:",e,flush=True)
            for _ in range(int(getattr(s,"vida_every",90))):
                if not getattr(s,"vida_on",False): break
                _t.sleep(1)
    # ===================== DESENVOLVIMENTO: a NEUROMODULAÇÃO causa a plasticidade (organismo reativo que cresce) =====================
    def _plast(s):                                                  # coeficiente de plasticidade = quanto a sinapse muda AGORA (a dopamina abre o portão)
        return 0.06+0.55*s.dop-0.15*s.fad                          # dopamina abre; fadiga fecha
    def _reinforce(s,key,driver="dopamina",delta=None):            # forma/fortalece/enfraquece a SINAPSE de um fato, gated pela neuromod
        v=s.seeds.get(key)
        if not v: return None
        st=v.get("str",0.5); d=delta if delta is not None else s._plast()
        st=max(0.0,min(1.0,st+d)); v["str"]=round(st,3); v["used"]=v.get("used",0)+(1 if d>0 else 0); v["tu"]=s.gtick
        act=("enfraqueceu" if d<0 else ("formou" if v.get("used",0)<=1 else "fortaleceu"))
        s.growth=(s.growth+[dict(t=time.strftime("%H:%M:%S"),ent=re.sub(r"\s+\d+$","",key),driver=driver,act=act,str=v["str"])])[-40:]
        return st
    def _grow_tick(s):                                             # UM tick de VIDA: pulso RÁPIDO (reativa/fortalece/esquece) todo tick + estudo/sono ocasionais
        s.gtick+=1; F=s._ensure_forest(); ents=F["ents"]
        with s.lock:
            # --- HOMEOSTASE: SONO quando exausta (consolida forte, esquece fraco) ---
            if s.fad>0.7:
                strong=[k for k,v in s.seeds.items() if v.get("str",0.5)>=0.7 and v.get("kind")!="raciocinado"]
                for k in strong[:20]: s._reinforce(k,"sono",+0.02)
                weak=[k for k,v in s.seeds.items() if v.get("str",0.5)<0.15 and v.get("kind")=="bulk"]
                for k in weak[:15]: s.seeds.pop(k,None)
                s.fad=max(0.0,s.fad-0.55); s.cort=max(0,s.cort-0.15); s._forest=None
                s.growth=(s.growth+[dict(t=time.strftime("%H:%M:%S"),ent=f"consolidou {len(strong[:20])} · podou {len(weak[:15])}",driver="sono",act="dormiu",str=0)])[-40:]
                return dict(mode="sono")
            # --- PULSO RÁPIDO (todo tick, sem professor): a floresta pulsa viva ---
            defs=[e for e,d in ents.items() if d["facts"]]
            if defs:                                               # REATIVA um conceito → fortalece a sinapse (dopamina)
                e=defs[s.gtick%len(defs)]; s.dop=min(1,s.dop+0.04); s._reinforce(ents[e]["facts"][0],"dopamina"); s.fad=min(1,s.fad+0.02)
            kb=[k for k,v in s.seeds.items() if v.get("kind")=="bulk"]
            if kb:                                                 # ESQUECE o menos usado → enfraquece (Ebbinghaus)
                old=min(kb,key=lambda k:s.seeds[k].get("tu",0))
                if s.seeds[old].get("str",0.5)<0.85: s._reinforce(old,"esquecimento",-0.03)
            s.cur=min(1,s.cur+0.05)                                # TÉDIO → curiosidade sobe
            gaps=sorted(((len(d["in"]),e) for e,d in ents.items() if not d["facts"] and len(e)>=4 and not s._junk_ent(e)),reverse=True)
            estimulo = s.env.pop(0) if s.env else None
            quer_estudar = (estimulo is not None) or (s.cur>0.6 and gaps and s.gtick%5==0)  # estudo só ocasional (professor é lento)
            if not quer_estudar: return dict(mode="pulso")
            concept = estimulo or gaps[0][1]
        # --- ESTUDAR (fora do lock, ocasional): professor forma sinapse NOVA, plasticidade gated pela dopamina ---
        fact=s._professor(concept,s._domain() or "conhecimento geral")
        s._healed=getattr(s,"_healed",set())|{concept}
        with s.lock:
            if fact:
                s.dop=min(1,s.dop+0.3); s.cur=max(0,s.cur-0.25); s.fad=min(1,s.fad+0.08)  # aprendeu novo = recompensa forte + cansa
                s._curar([fact],"pesquisado")
                key=next((k for k in s.seeds if s._cf(k)==s._cf(concept)),None)
                if key: s.seeds[key]["str"]=round(min(1,0.4+s._plast()),3)
                s.growth=(s.growth+[dict(t=time.strftime("%H:%M:%S"),ent=concept,driver="curiosidade",act="estudou",str=(s.seeds.get(key,{}) or {}).get("str",0.5) if key else 0.5)])[-40:]
                s._forest=None
                return dict(mode="estudou",concept=concept,fact=fact)
            s.cur=min(1,s.cur+0.1)
        return dict(mode="estudou_vazio",concept=concept)
    def _devloop(s):                                              # o organismo VIVE: reage e cresce em tempo acelerado
        import time as _t
        while getattr(s,"dev_on",False):
            try: s._grow_tick()
            except Exception as e: print("dev:",e,flush=True)
            for _ in range(int(getattr(s,"dev_every",6))):
                if not getattr(s,"dev_on",False): break
                _t.sleep(1)
    def graph(s,focus=None):                                        # FLORESTA VIVA: entidade=nó (hub de tudo sobre ela); clique expande a vizinhança
        from collections import Counter
        if not s.ready: return dict(nodes=[],edges=[],crumb=[])
        F=s._ensure_forest(); ents=F["ents"]; deg=F["deg"]
        nodes={}; edges=[]; seen=set()
        def enode(name,role):                                       # nó-ENTIDADE (único; nunca duplica)
            d=ents.get(name); dd=deg.get(name,0); has=bool(d and d["facts"])
            grp="focus" if role=="center" else ("cur" if (d and d["cur"]) else ("topic" if (has and dd>=6) else ("seed" if has else "gap")))
            if name not in nodes:
                st=(s.seeds.get(d["facts"][0],{}).get("stmt","") if has else "")
                nodes[name]=dict(id=name,label=re.sub(r"\s+\d+$","",name),group=grp,n=dd,ppath="ent:"+name,
                    expandable=int(dd>0),curated=int(bool(d and d["cur"])),fact=st,
                    title=(name+" · "+str(dd)+" ligações · "+("clique 2× p/ navegar" if dd>0 else "ainda sem definição (lacuna)")))
            return name
        def link(a,b,rel):
            kk=(a,b,rel)
            if kk in seen or a==b: return
            seen.add(kk); edges.append({"from":a,"to":b,"label":rel or "","color":{"color":"#2f7f6e","highlight":"#5dcaa5"},"width":1.2,"layer":"relacao","baseW":1.0})
        if not focus:
            hubs=[e for e,_ in sorted(deg.items(),key=lambda kv:-kv[1]) if ents.get(e,{}).get("facts")][:26]  # MACRO = maiores HUBS (entradas)
            hs=set(hubs)
            for h in hubs: enode(h,"hub")
            for h in hubs:
                for rel,oe,_ in ents[h]["out"]:
                    if oe in hs: link(h,oe,rel)
            crumb=[]
        else:
            name=s._cf(focus[4:] if focus.startswith("ent:") else focus); d=ents.get(name)
            if not d: return dict(nodes=[],edges=[],crumb=[name],focus=focus)
            c=enode(name,"center")
            outg=Counter(); outr={}
            for rel,oe,_ in d["out"]:
                outg[oe]+=1; outr.setdefault(oe,Counter())[rel]+=1
            for oe,_ in outg.most_common(30):                        # o que ela AFIRMA (galhos que saem)
                enode(oe,"nb"); link(c,oe,outr[oe].most_common(1)[0][0])
            ing=Counter(); inr={}
            for subj,rel,_ in d["in"]:
                if subj!=name: ing[subj]+=1; inr.setdefault(subj,Counter())[rel]+=1
            for se,_ in ing.most_common(16):                         # quem APONTA pra ela (galhos que entram)
                enode(se,"nb"); link(se,c,inr[se].most_common(1)[0][0])
            crumb=[name]
        return dict(nodes=list(nodes.values()),edges=edges,crumb=crumb,focus=focus or "")
    def _vitals(s):                                                 # SINAIS VITAIS do conhecimento
        facts={k:v for k,v in s.seeds.items() if v.get("kind")!="raciocinado"}
        if len(facts)>1500:                                          # ESCALA (20k): vitais leves (evita O(N²)/networkx no poll)
            lig=sum(len(v.get("obj") or []) for v in facts.values())
            return dict(fatos=len(facts),ligacoes=lig,conceitos=len(facts),definidos=len(facts),completude=100,
                        profundidade=0,ilhas=0,pontes=0,frageis=[],bitsfato=0,compressao=0.0,lacunas=[])
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
        # ---- ESTRUTURA (bateria matemática): ilhas, pontes, densidade ----
        ilhas=pontes=0; frageis=[]
        try:
            import networkx as nx
            UG=nx.Graph()
            for k in facts: UG.add_node(k)
            for k,outs in adj.items():
                for m in outs: UG.add_edge(k,m)
            comps=list(nx.connected_components(UG))
            ilhas=sum(1 for c in comps if len(c)==1)
            art=list(nx.articulation_points(UG)) if UG.number_of_nodes()>2 else []
            pontes=len(art); frageis=[lb(a) for a in art[:5]]
        except Exception: pass
        import gzip
        blob="\n".join(v["stmt"] for v in facts.values()).encode()
        dens=round(len(gzip.compress(blob))*8/max(1,len(facts)))          # bits por fato (densidade)
        comp=round(len(blob)/max(1,len(gzip.compress(blob))),2)           # razão de compressão (estrutura/redundância)
        return dict(fatos=len(facts),ligacoes=sum(len(v.get("obj") or []) for v in facts.values()),
            conceitos=len(refs),definidos=len(defined),completude=round(100*len(defined)/max(1,len(refs))),
            profundidade=prof,ilhas=ilhas,pontes=pontes,frageis=frageis,bitsfato=dens,compressao=comp,
            lacunas=s._gaps()[:8])                                       # centralidade: a lacuna mais 'load-bearing' primeiro
    def state(s):
        return dict(ready=s.ready,busy=s.busy,conceitos=len(getattr(s,'clean',[])) if s.ready else 0,neuronios=int(getattr(s,'vals',torch.zeros(0)).shape[0]) if s.ready else 0,
            chat=s.chat[-12:],trace=[list(t)+[""]*(3-len(t)) for t in s.trace],dop=round(s.dop,2),cort=round(s.cort,2),cur=round(s.cur,2),fad=round(s.fad,2),
            dev_on=s.dev_on,gtick=s.gtick,growth=s.growth[-10:],
            auto=s.auto,auto_every=s.auto_every,gap_now=s.gap_now,nfatos=len(s.seeds),autofeed=s.autofeed[-8:],vitais=s._vitals() if s.ready else None,fg_ready=s.fg_ready,
            jardim_on=s.jardim_on,jardim=s.jardim,jardimfeed=s.jardimfeed[-8:],vida_on=s.vida_on,vidafeed=s.vidafeed[-8:],
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
#net{position:fixed;top:74px;left:330px;right:340px;bottom:0}
#crumb{position:fixed;top:46px;left:330px;right:340px;height:24px;display:flex;align-items:center;gap:4px;padding:0 10px;font-size:11px;color:#9ab;z-index:8;overflow-x:auto;white-space:nowrap}
#crumb .cb{cursor:pointer;color:#7fb0e6;padding:2px 7px;border:1px solid #23324a;border-radius:11px;background:#0e1420}
#crumb .cb:hover{background:#16243a;color:#bcd8ff}#crumb .sep{color:#4a5568}#crumb .here{color:#cbb8ff;font-weight:600}
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
#ctl{position:fixed;top:52px;right:348px;width:186px;background:#0d1016ee;border:1px solid #1e2330;border-radius:8px;padding:9px;z-index:8;font-size:11px}
#ctl h3{font-size:9px;color:#7fb59f;letter-spacing:1px;text-transform:uppercase;margin:0 0 6px}
.cr{display:flex;align-items:center;gap:6px;margin:5px 0}.cr span{width:58px;color:#9ab}.cr input[type=range]{flex:1;accent-color:#5dcaa5;height:3px}
.ctog{display:grid;grid-template-columns:1fr 1fr;gap:3px 8px;margin-top:8px;padding-top:7px;border-top:1px solid #1a1f2a}
.ctog label{display:flex;align-items:center;gap:4px;color:#bcd;cursor:pointer}.ctog input{accent-color:#5dcaa5}
#ctl button{width:100%;margin-top:8px;padding:5px;font-size:11px}
#ctlgear{position:fixed;top:52px;right:348px;z-index:9;background:#12251d;border:1px solid #1d9e75;color:#9fe1cb;border-radius:6px;padding:4px 8px;cursor:pointer;font-size:12px;display:none}
</style></head><body>
<div id=bar><h1>◉ IARA · CÉREBRO</h1><span id=stt>acordando…</span><span style=flex:1></span><span id=fatos style="color:#ba7517;font-size:11px"></span></div>
<div id=L>
 <div class=card><h2>sinais vitais do conhecimento</h2><div id=vitais></div></div>
 <div class=card><h2>aprendizado autônomo (professor Claude)</h2>
   <div id=auto><div class=sw id=sw onclick=toggleAuto()><i></i></div><span>modo automático — ela estuda o que menos sabe</span></div>
   <div class=gap id=gap></div>
   <div style="display:flex;gap:6px;margin-top:6px"><button onclick=teachNow() style="font-size:11px;padding:5px 8px">ensinar 1 agora</button><span style=color:#7a8290;font-size:10px;align-self:center>ciclo a cada ~40s</span></div>
 </div>
 <div class=card><h2>🌱 jardineiro (cuida da floresta)</h2>
   <div id=auto><div class=sw id=swj onclick=toggleJardim()><i></i></div><span>manutenção automática — poda ruído · funde plural · acha contradição</span></div>
   <div style="display:flex;gap:6px;margin-top:6px"><button onclick=jardimNow() style="font-size:11px;padding:5px 8px">cuidar agora</button><span id=jstat style=color:#7a8290;font-size:10px;align-self:center></span></div>
   <div id=jfeed style=margin-top:6px></div>
 </div>
 <div class=card><h2>🌳 floresta viva (cura · religa · dorme)</h2>
   <div id=auto><div class=sw id=swv onclick=toggleVida()><i></i></div><span>organismo autônomo — cicatriza lacunas · religa correlações · dorme/consolida</span></div>
   <div style="display:flex;gap:6px;margin-top:6px"><button onclick=vidaNow() style="font-size:11px;padding:5px 8px">viver 1 ciclo</button><span id=vstat style=color:#7a8290;font-size:10px;align-self:center>usa o professor Haiku</span></div>
   <div id=vfeed style=margin-top:6px></div>
 </div>
 <div class=card><h2>🧬 desenvolvimento (a neuromod CAUSA a plasticidade)</h2>
   <div id=auto><div class=sw id=swd onclick=toggleDev()><i></i></div><span>ser reativo que CRESCE — dopamina forma sinapse · cortisol repara · curiosidade estuda · fadiga dorme</span></div>
   <div style="display:flex;gap:6px;margin-top:6px"><button onclick=devNow() style="font-size:11px;padding:5px 8px">1 tick de vida</button><span id=dstat style=color:#7a8290;font-size:10px;align-self:center>tick 0</span></div>
   <div id=gfeed style=margin-top:6px></div>
 </div>
 <div class=card><h2>hormônios (neuromodulação)</h2><div id=horm></div></div>
 <div class=card><div id=drop onclick="f.click()">📷 arraste/clique uma imagem — o córtex visual lê (texto/tabela) e analisa freq/cores</div><input type=file id=f accept=image/* style=display:none onchange=upl()></div>
 <div id=chat></div>
 <div id=ci><input id=qi placeholder="fale, ensine ('X é Y') ou pergunte…" onkeydown="if(event.key=='Enter')ask()"><button onclick=ask()>enviar</button></div>
 <div id=ci style=border-top:0;padding-top:0><input id=ti placeholder="🗣 falar sobre um tema (voz byte-nativa: fluxo+fala)…" onkeydown="if(event.key=='Enter')falar()"><button onclick=falar() id=btf title="grafo dá os fatos, Qwen dá a forma">falar</button></div>
</div>
<div id=crumb></div>
<div id=net></div>
<div id=ctlgear onclick="ctl.style.display='block';ctlgear.style.display='none'">⚙ grafo</div>
<div id=ctl>
 <h3>controle do grafo <span style="float:right;cursor:pointer;color:#7a8290" onclick="ctl.style.display='none';ctlgear.style.display='block'">✕</span></h3>
 <div class=cr><span>repulsão</span><input type=range id=s_rep min=2000 max=90000 value=55000 oninput=phys()></div>
 <div class=cr><span>distância</span><input type=range id=s_dist min=60 max=520 value=320 oninput=phys()></div>
 <div class=cr><span>gravidade</span><input type=range id=s_grav min=0 max=100 value=3 oninput=phys()></div>
 <div class=cr><span>ligação</span><input type=range id=s_link min=0 max=25 value=2 oninput=phys()></div>
 <div class=ctog>
  <label><input type=checkbox id=t_rel checked onchange=filt()> relações</label>
  <label><input type=checkbox id=t_gap checked onchange=filt()> lacunas</label>
  <label><input type=checkbox id=t_grp checked onchange=filt()> grupos</label>
  <label><input type=checkbox id=t_lbl checked onchange=filt()> rótulos</label>
 </div>
 <button onclick=refit()>reorganizar</button>
</div>
<div id=leg><b style=color:#8fb5d8>floresta viva:</b> cada nó é uma <b>entidade</b> · <b style=color:#8fb5d8>duplo-clique</b> entra nela (vê tudo ligado) · na central volta · <b style=color:#8fb5d8>zoom</b> aproxima sem crescer &nbsp;·&nbsp; <span class=lz style=border-color:#2f7f6e></span>relação real &nbsp; <span style="color:#ffcf5d">★</span>curada &nbsp; <span style="color:#e24b4a">●</span>lacuna (sem definição)</div>
<div id=R>
 <div class=card style=border:0><h2>nó selecionado (clique no grafo)</h2><div id=det><span style=color:#666>clique num nó pra ver o fato e as relações</span></div></div>
 <div class=card style=border:0><h2>professor ↔ aluna (o que ela aprendeu sozinha)</h2><div id=feed></div></div>
 <div class=card style=border:0><h2>trace do último pensamento</h2><div id=trace></div></div>
</div>
<script>
let net,nodes,edges,gmap={},autoOn=false;
/* definições-base dos grupos (tamanho/fonte SEM zoom); ao aproximar, tudo encolhe → tela constante, espaçamento igual */
const GDEF={topic:{shape:'hexagon',color:{background:'#20344f',border:'#5b8fd6'},bs:20,bf:13,borderWidth:2},
  focus:{shape:'star',color:{background:'#3a2f10',border:'#efb527'},bs:26,bf:14,borderWidth:2},
  subarea:{shape:'diamond',color:{background:'#2a2340',border:'#8f7fd6'},bs:13,bf:11,borderWidth:2},
  cur:{color:{background:'#b06d13',border:'#ffcf5d'},borderWidth:2,bs:11,bf:10,shape:'dot'},
  seed:{shape:'dot',color:{background:'#5a4a2a',border:'#8a7340'},bs:8,bf:9},
  auto:{shape:'dot',color:{background:'#0f6e56',border:'#5dcaa5'},bs:11,bf:10},vision:{shape:'dot',color:{background:'#0f6e56',border:'#1d9e75'},bs:11,bf:10},
  reason:{shape:'dot',color:{background:'#3c3489',border:'#7f77dd'},bs:11,bf:10},
  gap:{shape:'dot',bs:6,bf:9,color:{background:'#3a2030',border:'#e24b4a'}},tree:{shape:'dot',color:{background:'#1b2130',border:'#2a3550'},bs:8,bf:9}};
function scaleGroups(k){const g={};for(const n in GDEF){const d=GDEF[n];g[n]={shape:d.shape||'dot',color:d.color,borderWidth:d.borderWidth||2,
  size:Math.max(3,d.bs*k),font:{size:Math.max(4,d.bf*k),color:(n=='focus'?'#ffdb8a':n=='cur'?'#ffe6b0':n=='topic'?'#bcd8ff':n=='subarea'?'#cbb8ff':'#c9b98f'),face:'monospace',strokeWidth:2,strokeColor:'#0a0c12'}};}return g;}
const opt={nodes:{shape:'dot',borderWidth:2},
 edges:{color:{color:'#2a3550',highlight:'#5b8fd6'},smooth:{type:'continuous'},width:.7,arrows:{to:{enabled:true,scaleFactor:.35}},font:{color:'#93a6c4',size:9,face:'monospace',strokeWidth:2,strokeColor:'#090b10',align:'middle'}},
 groups:scaleGroups(1),
 physics:{barnesHut:{gravitationalConstant:-52000,springLength:300,springConstant:.025,centralGravity:.03,avoidOverlap:1,damping:.6},stabilization:{iterations:220}},interaction:{hover:true,hoverConnectedEdges:true,tooltipDelay:120}};
function setPhys(spread){if(!net)return;net.setOptions({physics:{barnesHut:{gravitationalConstant:spread?-60000:-32000,springLength:spread?340:180,springConstant:.025,centralGravity:spread?.02:.08,avoidOverlap:1,damping:.6}}});}
let FOCUS=null,_zpend=false;   /* caminho do nó atualmente aberto (null=macro) */
function applyZoom(sc){if(!net||!edges)return;const k=Math.min(1.6,Math.max(0.16,1/sc));  /* zoom in (sc↑) → k↓ → TUDO encolhe (nós, fontes, LARGURA das arestas), mesma distância */
  net.setOptions({groups:scaleGroups(k),edges:{font:{size:Math.max(3,10*k)}}});
  const ups=[];edges.forEach(e=>{const bw=e.baseW||(e.layer=='relacao'?1.0:e.layer=='lacuna'?.8:e.layer=='estrutura'?.35:.6);
    const w=Math.round(bw*k*100)/100;if((e.width||0)!==w)ups.push({id:e.id,width:w});});
  if(ups.length&&ups.length<500)edges.update(ups);}
function onZoom(e){if(_zpend)return;_zpend=true;requestAnimationFrame(()=>{_zpend=false;applyZoom(e&&e.scale?e.scale:(net?net.getScale():1));});}
function phys(){if(!net)return;net.setOptions({physics:{barnesHut:{gravitationalConstant:-(+s_rep.value),springLength:+s_dist.value,centralGravity:(+s_grav.value)/100,springConstant:(+s_link.value)/100,avoidOverlap:.6,damping:.5}}});}
function filt(){if(!nodes)return;
 const R=t_rel.checked,Gp=t_gap.checked,Gr=t_grp.checked;
 nodes.forEach(n=>{const h=((n.group=='topic'||n.group=='subarea')&&!Gr)||(n.group=='gap'&&!Gp);if(!!(nodes.get(n.id)||{}).hidden!==h)nodes.update({id:n.id,hidden:h})});
 edges.forEach(e=>{const h=(e.layer=='estrutura'&&!Gr)||(e.layer=='lacuna'&&!Gp)||(e.layer=='relacao'&&!R);if(!!(edges.get(e.id)||{}).hidden!==h)edges.update({id:e.id,hidden:h})});
 if(!t_lbl.checked)net.setOptions({edges:{font:{size:0}}});else applyZoom(net.getScale());}
function refit(){if(net){net.stabilize();setTimeout(()=>net.fit({animation:true}),500);}}
async function Po(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}
async function ask(){const v=qi.value.trim();if(!v)return;qi.value='';await Po('/think',{q:v});poll()}
async function falar(){const v=ti.value.trim();if(!v)return;ti.value='';btf.textContent='…';await Po('/falar',{tema:v});btf.textContent='falar';poll()}
async function upl(){const fl=f.files[0];if(!fl)return;const r=new FileReader();r.onload=async()=>{await Po('/see',{img:r.result});poll()};r.readAsDataURL(fl)}
async function toggleAuto(){autoOn=!autoOn;await Po('/auto',{on:autoOn});poll()}
let jardimOn=false;
async function toggleJardim(){jardimOn=!jardimOn;await Po('/jardim',{on:jardimOn});poll()}
async function jardimNow(){jstat.textContent='cuidando…';await Po('/jardim',{now:1});setTimeout(poll,400)}
let vidaOn=false;
async function toggleVida(){vidaOn=!vidaOn;await Po('/vida',{on:vidaOn});poll()}
let devOn=false;
async function toggleDev(){devOn=!devOn;await Po('/crescer',{on:devOn});poll()}
async function devNow(){dstat.textContent='vivendo…';await Po('/crescer',{now:1});setTimeout(poll,500)}
async function vidaNow(){vstat.textContent='vivendo… (professor)';await Po('/vida',{now:1});setTimeout(poll,600)}
async function teachNow(){await Po('/auto',{now:1});setTimeout(poll,500)}
async function poll(){try{const st=await(await fetch('/state')).json();render(st)}catch(e){}}
function showDet(id){const n=gmap[id];const det=document.getElementById('det');if(!n){det.innerHTML='<span style=color:#666>nó não encontrado</span>';return;}
 const outr=[...new Set((edges.get()||[]).filter(e=>e.from===id&&e.label).map(e=>e.label))];
 const inr=(edges.get()||[]).filter(e=>e.to===id&&e.label).length;
 let h=`<div class=k>${n.label}</div>`;
 h+=`<div style=color:#7a8290;font-size:11px>entidade · <b style=color:#8fb5d8>${n.n||0}</b> ligações${n.curated?' · <span style=color:#ffcf5d>★ curada</span>':''}</div>`;
 if(n.group==='gap')h+='<div style="color:#d98a94;font-size:11px;margin-top:4px">⚠ citada por outros fatos mas ainda SEM definição (lacuna — a IARA pode aprender)</div>';
 if(n.fact)h+=`<div style=margin-top:6px>${n.fact}</div>`;
 if(outr.length)h+=`<div class=rel style=margin-top:6px>relações: ${outr.slice(0,10).map(r=>'<span class=pill>'+r+'</span>').join('')}</div>`;
 if(n.expandable)h+='<div style="color:#7fb0e6;font-size:10px;margin-top:6px">▸ duplo-clique pra ENTRAR nesta entidade e ver tudo ligado a ela</div>';
 det.innerHTML=h;}
let HIST=[];   /* trilha de entidades visitadas (a floresta é grafo, não árvore → histórico) */
function drawCrumb(){const c=document.getElementById('crumb');
 let h=`<span class="cb" onclick="nav(null)">⌂ hubs</span>`;
 HIST.forEach((p,i)=>{const nm=p.replace(/^ent:/,'').toUpperCase();const last=i===HIST.length-1;
   h+=`<span class=sep>›</span>`+(last?`<span class=here>${nm}</span>`:`<span class="cb" onclick="nav('${p}')">${nm}</span>`);});
 if(!HIST.length)h+=`<span class=sep style=margin-left:6px>· os maiores HUBS da floresta — duplo-clique numa entidade pra entrar e navegar</span>`;
 c.innerHTML=h;}
function nav(path){if(!path){HIST=[];}else{const i=HIST.indexOf(path);if(i>=0)HIST=HIST.slice(0,i+1);else HIST.push(path);}FOCUS=path;grefresh(true);}
async function grefresh(reset){if(!net)return;const g=await(await fetch('/graph?focus='+encodeURIComponent(FOCUS||''))).json();
 gmap={};g.nodes.forEach(n=>gmap[n.id]=n);drawCrumb();
 if(reset){nodes.clear();edges.clear();}
 const gid=new Set(g.nodes.map(n=>n.id));
 nodes.getIds().forEach(id=>{if(!gid.has(id))nodes.remove(id)});
 g.nodes.forEach(n=>nodes.update(n));
 const geid=new Set(g.edges.map(e=>e.from+'|'+e.to));
 edges.getIds().forEach(id=>{if(!geid.has(id))edges.remove(id)});
 g.edges.forEach(e=>{edges.update({id:e.from+'|'+e.to,...e})});
 filt();
 if(reset){setPhys(nodes.length<28);net.stabilize();setTimeout(()=>{net.fit({animation:true});setTimeout(()=>applyZoom(net.getScale()),700);},450);}
 else applyZoom(net.getScale());}
function onDbl(p){if(!p.nodes.length)return;const n=gmap[p.nodes[0]];if(!n)return;
 if(n.group==='focus'){HIST.pop();nav(HIST.length?HIST[HIST.length-1]:null);}    /* duplo-clique na entidade central = voltar */
 else if(n.ppath){nav(n.ppath);}}                                                /* entrar numa entidade vizinha (navegar a floresta) */
function render(st){document.getElementById('stt').textContent=st.busy?'🧠 pensando…':(st.ready?`online · ${st.conceitos} conceitos · ${st.neuronios} neurônios`:'acordando os neurônios…');
 document.getElementById('fatos').textContent=st.ready?`${st.nfatos||0} fatos na memória`:'';
 if(!net&&st.ready){nodes=new vis.DataSet();edges=new vis.DataSet();net=new vis.Network(document.getElementById('net'),{nodes,edges},opt);net.on('click',p=>{if(p.nodes.length)showDet(p.nodes[0])});net.on('doubleClick',onDbl);net.on('zoom',onZoom);grefresh(true)}
 autoOn=!!st.auto;document.getElementById('sw').className='sw'+(autoOn?' on':'');
 document.getElementById('gap').innerHTML=st.gap_now?`🔎 estudando agora: <b>${st.gap_now}</b> — perguntando ao professor…`:(autoOn?'observando o grafo pelo que menos conhece…':'desligado');
 horm.innerHTML=[['dopamina',st.dop,'#63c923'],['cortisol',st.cort,'#e24b4a'],['curiosidade',st.cur,'#ef9f27'],['fadiga',st.fad||0,'#7f77dd']].map(([n,v,c])=>`<div class=h><span>${n}</span><span>${v}</span></div><div class=tk><div class=tf style="width:${v*100}%;background:${c}"></div></div>`).join('');
 devOn=!!st.dev_on;document.getElementById('swd').className='sw'+(devOn?' on':'');
 if(st.gtick!=null)dstat.textContent='tick '+st.gtick;
 const DC={dopamina:'#63c923',curiosidade:'#ef9f27',cortisol:'#e24b4a',sono:'#7f77dd',esquecimento:'#6b7280'};
 const AI={formou:'✚',fortaleceu:'▲',enfraqueceu:'▽',estudou:'📖',dormiu:'😴',dorme:'😴'};
 gfeed.innerHTML=(st.growth||[]).slice().reverse().slice(0,8).map(g=>`<div class=af style="border-left:2px solid ${DC[g.driver]||'#555'};padding-left:6px"><span style=color:#7a8290>${g.t}</span> ${AI[g.act]||'·'} <b>${g.act}</b> <span style="color:${DC[g.driver]||'#9ab'}">${g.ent}</span> <span style=color:#7a8290>(${g.driver}${g.str?' · força '+g.str:''})</span></div>`).join('')||'<span style=color:#666>ligue e ela reage/cresce sozinha — dopamina forma, cortisol repara, curiosidade estuda</span>';
 const vi=st.vitais;if(vi){document.getElementById('vitais').innerHTML=`<div class=vg><div class=b><div class=n>${vi.fatos}</div><div class=l>fatos</div></div><div class=b><div class=n>${vi.ligacoes}</div><div class=l>conexões</div></div><div class=b><div class=n>${vi.profundidade}</div><div class=l>saltos</div></div><div class=b><div class=n style=color:#ef9f27>${vi.completude}%</div><div class=l>completude</div></div></div><div class=compbar><i style=width:${vi.completude}%></i></div><div style="font-size:10px;color:#7a8290;margin-top:5px">sabe definir <b style=color:#9fe1cb>${vi.definidos}</b>/${vi.conceitos} conceitos que cita · lacunas: <span style=color:#d98a94>${(vi.lacunas||[]).slice(0,5).join(', ')||'—'}</span></div><div style="font-size:10px;color:#7a8290;margin-top:4px">estrutura: <b style=color:#8fb5d8>${vi.ilhas}</b> ilhas · <b style=color:#8fb5d8>${vi.pontes}</b> pontos frágeis · densidade <b style=color:#8fb5d8>${vi.bitsfato}</b> bits/fato (${vi.compressao}×)${vi.frageis&&vi.frageis.length?' · frágil: '+vi.frageis.slice(0,3).join(', '):''}</div>`;}
 feed.innerHTML=(st.autofeed||[]).slice().reverse().map(a=>`<div class=af><span style=color:#7a8290>${a.t}</span> não sabia <span class=g>${a.gap}</span><br><b>professor:</b> ${a.fact}</div>`).join('')||'<span style=color:#666>ligue o modo automático e ela vai estudar sozinha</span>';
 jardimOn=!!st.jardim_on;document.getElementById('swj').className='sw'+(jardimOn?' on':'');
 const jr=st.jardim;if(jr){jstat.textContent=`última: −${jr.pruned} arestas-lixo · ${jr.contra} contradições · ${jr.total_obj} ligações vivas`;}
 jfeed.innerHTML=(st.jardimfeed||[]).slice().reverse().slice(0,4).map(j=>`<div class=af><span style=color:#7a8290>${j.t}</span> podou <b style=color:#9fe1cb>${j.pruned}</b> · contradições <b style=color:#e2a24b>${j.contra}</b>${(j.contra_ex&&j.contra_ex.length)?'<br><span style=color:#d98a94>ex: '+j.contra_ex.slice(0,2).map(c=>c[0]+'→{'+c[1].join(', ')+'}').join(' · ')+'</span>':''}</div>`).join('')||'<span style=color:#666>clique "cuidar agora" e ela poda o ruído da floresta</span>';
 vidaOn=!!st.vida_on;document.getElementById('swv').className='sw'+(vidaOn?' on':'');
 const ic={cicatrizou:'🩹',religou:'🔗',dormiu:'😴'};
 vfeed.innerHTML=(st.vidafeed||[]).slice().reverse().slice(0,5).map(v=>{
   if(v.kind==='cicatrizou')return `<div class=af><span style=color:#7a8290>${v.t}</span> ${ic.cicatrizou} cicatrizou <b style=color:#9fe1cb>${v.a}</b><br><span style=color:#9ab>${v.fato}</span></div>`;
   if(v.kind==='religou')return `<div class=af><span style=color:#7a8290>${v.t}</span> ${ic.religou} religou <b style=color:#cbb8ff>${v.a}</b> ↔ <b style=color:#cbb8ff>${v.b}</b> <span style=color:#7a8290>(via ${v.via})</span><br><span style=color:#9ab>${v.fato}</span></div>`;
   return `<div class=af><span style=color:#7a8290>${v.t}</span> ${ic.dormiu} dormiu — núcleo <b style=color:#8fb5d8>${v.core}</b> · esqueceu <b style=color:#d98a94>${v.forgot}</b> folhas fracas</div>`;
 }).join('')||'<span style=color:#666>clique "viver 1 ciclo" e ela cicatriza/religa sozinha (usa o professor)</span>';
 chat.innerHTML=(st.chat||[]).map(m=>`<div class="m ${m[0]}">${m[0]=='voce'?'você: ':'IARA: '}${m[1]}</div>`).join('');chat.scrollTop=chat.scrollHeight;
 trace.innerHTML=(st.trace||[]).map(t=>`<div class="tr r-${t[0]}"><b>${t[0]}</b> · ${t[1]}${t[2]?` <span class=d>${t[2]}</span>`:''}</div>`).join('')||'<span style=color:#666>mande uma mensagem…</span>';
 window._gt=(window._gt||0)+1; if(window._gt%6===0) grefresh()}   /* grafo só a cada ~5s (não trava em escala) */
setInterval(poll,900);poll();
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(s,*a): pass
    def _j(s,o): b=json.dumps(o,ensure_ascii=False).encode(); s.send_response(200); s.send_header("Content-Type","application/json; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        u=urlparse(s.path); q=parse_qs(u.query)
        if u.path=="/": b=PAGE.encode(); s.send_response(200); s.send_header("Content-Type","text/html; charset=utf-8"); s.send_header("Content-Length",str(len(b))); s.end_headers(); s.wfile.write(b); return
        if u.path=="/state": return s._j(C.state())
        if u.path=="/graph":
            foc=q.get("focus",[None])[0]; foc=foc if foc else None
            return s._j(C.graph(foc) if C.ready else dict(nodes=[],edges=[],crumb=[]))
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
        elif s.path=="/falar": r=C.falar(d.get("tema","")) if C.ready else dict(say="acordando…")   # voz byte-nativa (fluxo+fala)
        elif s.path=="/persona":                                    # EDITAR a personalidade dela ao vivo
            if d.get("persona"): C.persona=d["persona"]
            r=dict(persona=C.persona)
        elif s.path=="/aprender":                                   # APRENDER RÁPIDO: ingere N fatos de um domínio do corpus
            kw=d.get("keywords") or [d.get("tema","")]; tgt=int(d.get("target",5000))
            def _bg():
                n,dt=C._bulk_learn([k for k in kw if k],tgt)
                try: C._build_hierarchy()                            # reconstrói a árvore macro→micro com o novo conhecimento
                except Exception as e: print("hier build:",e,flush=True)
                with C.lock: C.chat.append(("iara",f"📚 estudei {n} fatos novos em {dt}s (agora sei {len(C.seeds)})"))
            threading.Thread(target=_bg,daemon=True).start(); r=dict(ok=1,msg="estudando em massa…")
        elif s.path=="/curar":                                      # INGESTÃO CURADA (agente professor manda fatos limpos → kind=pesquisado)
            fatos=d.get("facts") or []; kind=d.get("kind","pesquisado")
            add=C._curar(fatos,kind) if fatos else 0
            r=dict(ok=1,added=add,total=len(C.seeds))
        elif s.path=="/jardim":                                     # JARDINEIRO: cuida da floresta (poda/imunidade) — 1x ou liga o loop
            if "on" in d:
                C.jardim_on=bool(d["on"])
                if C.jardim_on: threading.Thread(target=C._jardimloop,daemon=True).start()
            if d.get("now") or ("on" not in d): rep=C._gardener()
            else: rep=C.jardim
            r=dict(ok=1,on=C.jardim_on,rep=rep)
        elif s.path=="/vida":                                       # F4: a floresta se cura/religa/dorme sozinha — 1x ou liga o loop
            if "on" in d:
                C.vida_on=bool(d["on"])
                if C.vida_on: threading.Thread(target=C._vidaloop,daemon=True).start()
            rep=None
            if d.get("now"):
                def _bg(): C._life()
                threading.Thread(target=_bg,daemon=True).start()
            r=dict(ok=1,on=C.vida_on)
        elif s.path=="/crescer":                                    # DESENVOLVIMENTO: organismo reativo — neuromod causa plasticidade, floresta cresce
            if "on" in d:
                C.dev_on=bool(d["on"])
                if C.dev_on: threading.Thread(target=C._devloop,daemon=True).start()
            if d.get("now"): threading.Thread(target=C._grow_tick,daemon=True).start()
            if d.get("env"): C.env.extend([x for x in d["env"] if x])   # injeta estímulos no ambiente
            r=dict(ok=1,on=C.dev_on,gtick=C.gtick)
        else: r={"err":1}
        s._j(r)
if __name__=="__main__":
    threading.Thread(target=relax,daemon=True).start()
    print("  ►►►  IARA · CÉREBRO em http://localhost:3050",flush=True)
    ThreadingHTTPServer(("127.0.0.1",3050),H).serve_forever()
