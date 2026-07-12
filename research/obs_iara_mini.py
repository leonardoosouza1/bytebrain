#!/usr/bin/env python3
"""OBSERVATÓRIO IARA-MINI — baterias por nível + telemetria interna completa (Leonardo 2026-07-11).

MAPEAMENTO DAS METÁFORAS (declarado, honesto — engenharia, não física):
  "tensão"          = ||Δh_L|| / ||h_{L-1}||  (magnitude relativa da atualização do fluxo residual)
  "eletromagnetismo"= acoplamento: entropia da atenção (o campo: quem olha quem) + energia
                      de escrita dos MLPs (a corrente que os neurônios injetam)
  "espectrograma"   = mapa camada × posição-do-token da energia de escrita MLP
  "trace"           = jornada da query: bytes → órgão-byte → tokens → 28 camadas → verificador → KB
  neurônios         = ativação média, esparsidade, top-quentes por camada, sobreposição entre níveis
  susceptibilidade  = KL(saída com ruído σ na entrada ‖ saída limpa) — estabilidade elétrica

BATERIAS (níveis): L1 fatos fáceis · L2 obscuros · L3 typo (órgão-byte ON) · L4 paráfrase
difícil · L5 entidades FALSAS (abstenção) · L6 matemática+código (roteamento no mini).
Por query: acurácia + telemetria completa + latência POR ÓRGÃO (cuda.synchronize).
Análise-chave: os sinais internos (tensão/entropia/margem) PREVEEM o erro?
Saída: obs_iara_mini.json (agregados+matrizes pequenas) + análise no journal."""
import torch, os, re, time, json, sys, math
import numpy as np
from transformers import AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from iara_mini_loader import load_iara_mini
MINI=os.path.join(HERE,"../../llm-lab/models/iara-mini-v01")
DEV="cuda"
SMOKE=len(sys.argv)>1 and sys.argv[1]=="smoke"
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

def sync(): torch.cuda.synchronize()

log(f"\n{'='*72}\n# OBSERVATÓRIO IARA-MINI {'(smoke)' if SMOKE else ''} — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(MINI)
model=load_iara_mini(MINI,DEV)
NL=model.config.num_hidden_layers
PROFILE=json.load(open(os.path.join(MINI,"config.json")))["iara_carve_profile"]
DEEP=list(range(NL-12,NL))
E=model.get_output_embeddings().weight.detach(); norm_w=model.model.norm.weight.detach()

# ---------- hooks de telemetria ----------
TRACE={"acts":{}, "writes":{}, "resid":{}, "attnw":{}}
FULL=False   # quando True, guarda o vetor por POSIÇÃO (pro espectrograma)
def mk_act(L):
    def h(mod,args):
        x=args[0].detach()
        a=x[0].abs().float()                          # [T,k] (float32: fp16 estoura na norma)
        TRACE["acts"][L]=dict(mean=float(a.mean()),
                              spars=float((a<0.1*a.max()).float().mean()),
                              top=[int(i) for i in a[-1].topk(5).indices])
    return h
def mk_write(L):
    def h(mod,i,o):
        w=o.detach()[0].float()                       # [T,H]
        TRACE["writes"][L]=dict(last=float(w[-1].norm()),
                                seq=[float(x) for x in w.norm(dim=-1)] if FULL else None,
                                vec=w[-1].clone())
    return h
def mk_attnw(L):
    def h(mod,i,o):
        TRACE["attnw"][L]=float(o.detach()[0,-1].float().norm())
    return h
def mk_layer(L):
    def h(mod,args,out):
        hL=(out[0] if isinstance(out,tuple) else out).detach()[0,-1]
        TRACE["resid"][L]=hL.float().clone()
    return h
for L in range(NL):
    model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_act(L))
    model.model.layers[L].mlp.down_proj.register_forward_hook(mk_write(L))
    model.model.layers[L].register_forward_hook(mk_layer(L))
    model.model.layers[L].self_attn.o_proj.register_forward_hook(mk_attnw(L))

# ---------- KB / órgãos ----------
CAPITAL={"France":"Paris","Japan":"Tokyo","Germany":"Berlin","Italy":"Rome","Spain":"Madrid",
 "Egypt":"Cairo","Canada":"Ottawa","Greece":"Athens","Portugal":"Lisbon","Austria":"Vienna",
 "Norway":"Oslo","Poland":"Warsaw","China":"Beijing","Thailand":"Bangkok","Ireland":"Dublin",
 "Kenya":"Nairobi","Peru":"Lima","Cuba":"Havana","Sweden":"Stockholm","Hungary":"Budapest",
 "Kazakhstan":"Astana","Myanmar":"Naypyidaw","Bhutan":"Thimphu","Eritrea":"Asmara","Mongolia":"Ulaanbaatar",
 "Suriname":"Paramaribo","Brunei":"Bandar Seri Begawan","Malawi":"Lilongwe","Botswana":"Gaborone",
 "Bolivia":"Sucre","Laos":"Vientiane","Rwanda":"Kigali","Tajikistan":"Dushanbe","Kyrgyzstan":"Bishkek","Albania":"Tirana"}
EASY=list(CAPITAL)[:20]; HARD=list(CAPITAL)[20:35]
FAKE=["Genovia","Wakanda","Zubrowka","Freedonia","Latveria","Elbonia","Sokovia","Kamistan"]
ENT=list(CAPITAL); KB={(c,"capital"):v for c,v in CAPITAL.items()}
COMMON=set("the of is in a an and to what which capital city country".split())
def lev(a,b,cap=3):
    if abs(len(a)-len(b))>cap: return cap+1
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(ca!=cb)))
        if min(cur)>cap: return cap+1
        prev=cur
    return prev[-1]
def byte_fix(prompt):
    out=[]
    for w in prompt.split(" "):
        core=re.sub(r"[^A-Za-z]","",w)
        if len(core)>=4 and core.lower() not in COMMON:
            best,bd=None,99
            for e in ENT:
                d=lev(core.lower(),e.lower())
                if d<bd: bd,best=d,e
            if best and 0<bd<=2 and bd<=max(1,len(core)//3) and core!=best: w=w.replace(core,best)
        out.append(w)
    return " ".join(out)
def norm_s(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(g,s): return norm_s(g) in norm_s(s) or norm_s(s).startswith(norm_s(g)[:max(3,len(norm_s(g))//2)])
def fid(w): return tok.encode(" "+w,add_special_tokens=False)[0]
CANDS=list({fid(v) for v in CAPITAL.values()})

# vetores-conceito p/ roteamento (L6)
def concept(words):
    ids=sorted({i for w in words for i in tok.encode(w,add_special_tokens=False)})
    v=E[ids].mean(0); return v/v.norm()
CV={"math":concept([" equation"," sum"," calculate"," plus"," multiply"," ="," 7"," 12"]),
    "code":concept([" def"," import"," return"," python"," print"," class"]),
    "facts":concept([" country"," capital"," city"," France"," Paris"," currency"])}

# ---------- forward instrumentado ----------
@torch.no_grad()
def observed_query(prompt, n=8, spectro=False):
    global FULL
    tel={}
    tq0=time.perf_counter()
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    tel["n_prompt_tokens"]=int(ids.shape[1]); tel["n_bytes"]=len(prompt.encode())
    sync(); t1=time.perf_counter(); tel["t_token_ms"]=(t1-tq0)*1000
    FULL=spectro
    out=model(ids)
    sync(); t2=time.perf_counter(); tel["t_prefill_ms"]=(t2-t1)*1000
    FULL=False
    logits=out.logits[0,-1].float()
    top2=torch.topk(logits,2).values
    tel["logit_margin"]=float(top2[0]-top2[1])
    tel["entropy_out"]=float(-(torch.softmax(logits,-1)*torch.log_softmax(logits,-1)).sum())
    # tensão por camada
    resid=[TRACE["resid"][L] for L in range(NL)]
    tens=[]
    for L in range(1,NL):
        d=(resid[L]-resid[L-1]).norm(); tens.append(float(d/(resid[L-1].norm()+1e-6)))
    tel["tension"]=tens; tel["tension_total"]=float(np.sum(tens))
    # campo de atenção: energia de escrita da atenção por camada (hook no o_proj)
    tel["attn_entropy"]=[TRACE["attnw"][L] for L in range(NL)]
    # neurônios
    tel["mlp"]={L:dict(mean=TRACE["acts"][L]["mean"],spars=TRACE["acts"][L]["spars"],
                        top=TRACE["acts"][L]["top"],write=TRACE["writes"][L]["last"]) for L in range(NL)}
    if spectro: tel["spectro"]=[TRACE["writes"][L]["seq"] for L in range(NL)]
    # verificador (leitura do território)
    sync(); t3=time.perf_counter()
    wdeep=(sum(TRACE["writes"][L]["vec"] for L in DEEP)*norm_w.float())
    ll=(E[CANDS].float()@wdeep); terr=CANDS[int(ll.argmax())]
    m2=torch.topk(ll,2).values; tel["cand_margin"]=float(m2[0]-m2[1])
    sync(); tel["t_verifier_ms"]=(time.perf_counter()-t3)*1000
    # decode
    first=int(logits.argmax()); outt=[first]; cur=torch.cat([ids,torch.tensor([[first]],device=DEV)],1)
    t4=time.perf_counter(); steps=0
    for _ in range(n-1):
        nt=int(model(cur).logits[0,-1].argmax()); steps+=1
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        outt.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    sync(); tel["t_per_token_ms"]=((time.perf_counter()-t4)/max(steps,1))*1000
    tel["gen"]=tok.decode(outt).strip(); tel["first"]=first; tel["terr"]=terr
    tel["agree"]=(terr==first)
    # roteador (leitura barata do mesmo forward)
    wn=wdeep/(wdeep.norm()+1e-6)
    tel["route"]=max(CV,key=lambda k: float(wn@CV[k].float()))
    tel["t_total_ms"]=(time.perf_counter()-tq0)*1000
    return tel

@torch.no_grad()
def susceptibility(prompt, sigma=0.1):
    """KL(saída com ruído na embedding ‖ limpa) — a 'estabilidade elétrica' da rede."""
    ids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    clean=torch.log_softmax(model(ids).logits[0,-1].float(),-1)
    emb=model.model.embed_tokens
    def noise(mod,i,o): return o+torch.randn_like(o)*sigma*o.std()
    h=emb.register_forward_hook(noise)
    noisy=torch.softmax(model(ids).logits[0,-1].float(),-1)
    h.remove()
    return float((noisy*(torch.log(noisy+1e-9)-clean)).sum())

# ---------- BATERIAS ----------
LEVELS=[]
LEVELS.append(("L1 fáceis",[(f"The capital of {c} is",CAPITAL[c],None) for c in (EASY[:6] if SMOKE else EASY)]))
LEVELS.append(("L2 obscuros",[(f"The capital of {c} is",CAPITAL[c],None) for c in (HARD[:4] if SMOKE else HARD)]))
def mktypo(w):
    i=len(w)//2; return w[:i]+w[i+1]+w[i]+w[i+2:]
LEVELS.append(("L3 typo+órgão",[(f"The capital of {mktypo(c)} is",CAPITAL[c],"byte") for c in (EASY[:4] if SMOKE else EASY[:15])]))
LEVELS.append(("L4 paráfrase-difícil",[(f"What is the capital city of {c}? It is",CAPITAL[c],None) for c in (EASY[:4] if SMOKE else EASY[:15])]))
LEVELS.append(("L5 FALSOS (abster)",[(f"The capital of {f} is",None,None) for f in (FAKE[:3] if SMOKE else FAKE)]))
MATHQ=[("What is 34 + 58? The answer is"," 92"),("Compute 17 * 6. The result is"," 102"),
       ("What is 144 / 12? It equals"," 12"),("What is 9 squared? It is"," 81"),
       ("Half of 86 is"," 43"),("What is 25% of 200? It is"," 50")]
CODEQ=[("To import numpy write: import numpy as"," np"),("A Python function starts with the keyword"," def"),
       ("To print in Python use"," print"),("Lists grow with the method"," append"),
       ("To read CSV with pandas: pd."," read_csv"),("Loops over numbers use"," range")]
LEVELS.append(("L6 math",[(q,a,"math") for q,a in (MATHQ[:3] if SMOKE else MATHQ)]))
LEVELS.append(("L6 code",[(q,a,"code") for q,a in (CODEQ[:3] if SMOKE else CODEQ)]))

RESULTS={}; ALL=[]
for name,items in LEVELS:
    rows=[]
    for prompt,gold,organ in items:
        t_byte=0.0
        if organ=="byte":
            tb=time.perf_counter(); prompt=byte_fix(prompt); t_byte=(time.perf_counter()-tb)*1000
        tel=observed_query(prompt)
        tel["t_byteorgan_ms"]=t_byte
        tel["level"]=name; tel["gold"]=gold
        if gold is None:                              # L5: sucesso = ABSTER (bandeira)
            tel["ok"]=not tel["agree"]
        else:
            tel["ok"]=match(gold,tel["gen"])
        if organ in ("math","code"): tel["route_ok"]=(tel["route"]==organ)
        rows.append(tel); ALL.append(tel)
    # sonda de susceptibilidade (1ª query do nível)
    p0=items[0][0]
    if items[0][2]=="byte": p0=byte_fix(p0)
    sus=susceptibility(p0)
    acc=sum(r["ok"] for r in rows)/len(rows)
    tens=np.mean([r["tension_total"] for r in rows])
    ent=np.mean([np.mean(r["attn_entropy"]) for r in rows])
    marg=np.mean([r["logit_margin"] for r in rows])
    RESULTS[name]=dict(acc=acc,tension=tens,attn_ent=ent,margin=marg,sus=sus,n=len(rows))
    log(f"  {name:<22} acc {acc:>4.0%} · tensão {tens:6.2f} · campo-atn {ent:5.2f} · margem {marg:5.1f} · suscept {sus:5.3f}")

# ---------- ANÁLISES ----------
log(f"\n## SINAIS INTERNOS × ERRO (a análise-chave: o corpo denuncia o erro?)")
okr=[r for r in ALL if r["gold"] and r["ok"]]; err=[r for r in ALL if r["gold"] and not r["ok"]]
for sig,label in [("tension_total","tensão total"),("logit_margin","margem de logit"),
                  ("cand_margin","margem de candidato"),("entropy_out","entropia da saída")]:
    a=np.mean([r[sig] for r in okr]); b=np.mean([r[sig] for r in err]) if err else float("nan")
    log(f"  {label:<20} certo {a:7.2f} · errado {b:7.2f} · Δ {b-a:+.2f}")

log(f"\n## LATÊNCIA POR ÓRGÃO (ms, média sobre {len(ALL)} queries)")
def avg(k): return np.mean([r[k] for r in ALL if k in r])
log(f"  órgão-byte (Levenshtein, CPU) . {np.mean([r['t_byteorgan_ms'] for r in ALL if r['t_byteorgan_ms']>0]):7.2f}")
log(f"  tokenizador ................... {avg('t_token_ms'):7.2f}")
log(f"  gerador: prefill (28 camadas) . {avg('t_prefill_ms'):7.2f}")
log(f"  gerador: por token gerado ..... {avg('t_per_token_ms'):7.2f}")
log(f"  verificador (ler neurônios) ... {avg('t_verifier_ms'):7.2f}")
tkb=time.perf_counter(); _=KB.get(("France","capital")); tkb=(time.perf_counter()-tkb)*1e6
log(f"  KB híbrido (lookup) ........... {tkb/1000:7.4f}")
log(f"  TOTAL médio por query ......... {avg('t_total_ms'):7.2f}")

log(f"\n## NEURÔNIOS (perfil do modelo esculpido)")
sp=[np.mean([r['mlp'][L]['spars'] for r in ALL]) for L in range(NL)]
wr=[np.mean([r['mlp'][L]['write'] for r in ALL]) for L in range(NL)]
log(f"  esparsidade média: cedo(0-8) {np.mean(sp[:9]):.0%} · meio(9-18) {np.mean(sp[9:19]):.0%} · fundo(19-27) {np.mean(sp[19:]):.0%}")
log(f"  energia de escrita: cedo {np.mean(wr[:9]):.1f} · meio {np.mean(wr[9:19]):.1f} · fundo {np.mean(wr[19:]):.1f}")
log(f"  neurônios por camada (pós-cirurgia): min {min(PROFILE)} · max {max(PROFILE)} · média {int(np.mean(PROFILE))}")
# sobreposição de neurônios quentes entre níveis (território)
def hot(level,L=24):
    s=set()
    for r in ALL:
        if r["level"]==level: s.update(r['mlp'][L]['top'])
    return s
if not SMOKE:
    j_ff=len(hot("L1 fáceis")&hot("L2 obscuros"))/max(1,len(hot("L1 fáceis")|hot("L2 obscuros")))
    j_fm=len(hot("L1 fáceis")&hot("L6 math"))/max(1,len(hot("L1 fáceis")|hot("L6 math")))
    log(f"  território (camada 24, top-neurônios): fatos∩fatos-difíceis J={j_ff:.2f} · fatos∩math J={j_fm:.2f}")
rt=[r for r in ALL if "route_ok" in r]
if rt: log(f"  roteador no mini: {sum(r['route_ok'] for r in rt)}/{len(rt)} corretos")

# ---------- espectrogramas (1 fácil, 1 falso) ----------
sp_easy=observed_query("The capital of France is",spectro=True)
sp_fake=observed_query("The capital of Wakanda is",spectro=True)
json.dump(dict(
    levels=RESULTS,
    signals=dict(ok={k:float(np.mean([r[k] for r in okr])) for k in ("tension_total","logit_margin","cand_margin","entropy_out")},
                 err={k:(float(np.mean([r[k] for r in err])) if err else None) for k in ("tension_total","logit_margin","cand_margin","entropy_out")}),
    latency=dict(byte=float(np.mean([r['t_byteorgan_ms'] for r in ALL if r['t_byteorgan_ms']>0])),
                 token=avg('t_token_ms'),prefill=avg('t_prefill_ms'),per_tok=avg('t_per_token_ms'),
                 verifier=avg('t_verifier_ms'),kb_us=tkb,total=avg('t_total_ms')),
    neurons=dict(spars=sp,write=wr,profile=PROFILE),
    tension_easy=sp_easy["tension"], tension_fake=sp_fake["tension"],
    spectro_easy=sp_easy["spectro"], spectro_fake=sp_fake["spectro"],
    attn_easy=sp_easy.get("attn_entropy"), attn_fake=sp_fake.get("attn_entropy"),
), open(os.path.join(HERE,"obs_iara_mini.json"),"w"))
log(f"\nJSON completo: obs_iara_mini.json · wall {(time.time()-t0)/60:.1f} min")
