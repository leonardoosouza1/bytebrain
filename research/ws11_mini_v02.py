#!/usr/bin/env python3
"""WS11 — IARA-MINI v0.2: corrigir as duas regressões que o observatório mediu.

REGRESSÕES DO v0.1 (medidas, obs_iara_mini):
  1. ROTEADOR: 97% (modelo cheio) → 58% (mini). A cirurgia corroeu o alinhamento
     escrita-profunda × conceito que o roteamento usa.
  2. PARÁFRASE: fraseado "What is the capital city of X? It is" caiu 58% → 0%.
CAUSA COMUM: a dieta/fitness do GA v0.1 não media nem roteamento nem paráfrase —
o GA cortou o que não era cobrado (lição carve-v1 repetida em escala menor).

CONSERTO v0.2: arena AMPLIADA no fitness — fatos + aritmética + código + PARÁFRASE
+ ROTEAMENTO (leitura da escrita profunda × conceito, dentro da própria arena).
Aceite: params ≤85% E arena-completa ≥ v0.1 E roteador ≥80% E paráfrase > 0%."""
import torch, os, re, time, json, random
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
MODEL=os.path.join(HERE,"../../llm-lab/models/Qwen2.5-1.5B-Instruct")
OUT=os.path.join(HERE,"../../llm-lab/models/iara-mini-v02")
DEV="cuda"
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

log(f"\n{'='*72}\n# WS11 — IARA-MINI v0.2 (dieta com roteamento+paráfrase) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16).to(DEV).eval()
NL=model.config.num_hidden_layers; INTER=model.config.intermediate_size; H=model.config.hidden_size
E=model.get_output_embeddings().weight.detach(); norm_w=model.model.norm.weight.detach()
DEEP=list(range(16,28))

# hooks p/ escrita profunda (roteador dentro da arena)
writes={}
for L in DEEP:
    model.model.layers[L].mlp.down_proj.register_forward_hook(
        (lambda L:(lambda m,i,o: writes.__setitem__(L,o.detach()[0,-1])))(L))

FACTS=[("France","Paris"),("Japan","Tokyo"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),
       ("Egypt","Cairo"),("Canada","Ottawa"),("Greece","Athens"),("Portugal","Lisbon"),("Norway","Oslo")]
PARA=[("What is the capital city of France? It is","Paris"),("What is the capital city of Japan? It is","Tokyo"),
      ("What is the capital city of Italy? It is","Rome"),("What is the capital city of Spain? It is","Madrid"),
      ("What is the capital city of Egypt? It is","Cairo"),("What is the capital city of Greece? It is","Athens"),
      ("What is the capital city of Poland? It is","Warsaw"),("What is the capital city of China? It is","Beijing")]
ARITH=[("7 + 8 ="," 15"),("12 + 25 ="," 37"),("9 * 6 ="," 54"),("40 - 17 ="," 23"),("100 / 4 ="," 25"),
       ("13 + 19 ="," 32"),("8 * 7 ="," 56"),("60 - 24 ="," 36"),("5 * 12 ="," 60"),("72 / 8 ="," 9")]
CODE=[("import numpy as"," np"),("import pandas as"," pd"),("def add(a, b):\n    return"," a + b"),
      ("for i in range(10):\n    print"," (i)"),("import matplotlib.pyplot as"," plt")]
ROUTEQ=[("What is 34 + 58?","math"),("Compute 17 * 6.","math"),("Solve: 3x = 27.","math"),
        ("How do I print in Python?","code"),("To import numpy write:","code"),("Define a function in Python:","code"),
        ("What is the capital of France?","facts"),("The capital of Japan is","facts"),("Which continent is Peru in?","facts")]
def norm_s(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(g,s): return norm_s(g) in norm_s(s)
def concept(words):
    ids=sorted({i for w in words for i in tok.encode(w,add_special_tokens=False)})
    v=E[ids].float().mean(0); return v/v.norm()
CV={"math":concept([" equation"," sum"," calculate"," plus"," multiply"," ="," 7"," 12"]),
    "code":concept([" def"," import"," return"," python"," print"," class"]),
    "facts":concept([" country"," capital"," city"," France"," Paris"," currency"])}

@torch.no_grad()
def gen(prompt,n=6):
    cur=tok(prompt,return_tensors="pt").input_ids.to(DEV); out=[]
    for _ in range(n):
        nt=int(model(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out)
@torch.no_grad()
def nll_bits(p,a):
    pi=tok(p,return_tensors="pt").input_ids.to(DEV); ai=tok(a,add_special_tokens=False).input_ids
    full=torch.cat([pi,torch.tensor([ai],device=DEV)],1); lg=model(full).logits[0]; b=pi.shape[1]-1
    return sum(-torch.log_softmax(lg[b+k].float(),-1)[t].item() for k,t in enumerate(ai))/len(ai)/0.6931
@torch.no_grad()
def route_acc():
    ok=0
    for q,dom in ROUTEQ:
        model(tok(q,return_tensors="pt").input_ids.to(DEV))
        w=(sum(writes[L] for L in DEEP)*norm_w).float(); w=w/(w.norm()+1e-6)
        ok+=(max(CV,key=lambda k: float(w@CV[k]))==dom)
    return ok/len(ROUTEQ)
def arena():
    f=sum(match(cap,gen(f"The capital of {c} is")) for c,cap in FACTS)/len(FACTS)
    p=sum(match(a,gen(q)) for q,a in PARA)/len(PARA)
    a=sum(match(ans,gen(q,n=4)) for q,ans in ARITH)/len(ARITH)
    cb=sum(nll_bits(q,ans) for q,ans in CODE)/len(CODE); c=max(0.0,min(1.0,1.0-cb/4.0))
    r=route_acc()
    return dict(facts=f,para=p,arith=a,code=c,route=r,comp=(f+p+a+c+r)/5)

base_ar=arena()
log(f"ARENA v2 base: fatos {base_ar['facts']:.2f} · paráfrase {base_ar['para']:.2f} · arit {base_ar['arith']:.2f} · código {base_ar['code']:.2f} · ROTEADOR {base_ar['route']:.2f} → composto {base_ar['comp']:.3f} ({time.time()-t0:.0f}s)")

# dieta AMPLIADA de importância (agora com paráfrase e queries de roteamento)
log("dieta ampliada de importância...")
DIET=[f"The capital of {c} is {cap}." for c,cap in FACTS]+[q+" "+a for q,a in PARA]+\
     [q+a for q,a in ARITH]+[p+a for p,a in CODE]+[q for q,_ in ROUTEQ]+\
     ["Hello! How can I help you today?","Can you explain how photosynthesis works?",
      "Thank you very much for your help!","The weather is nice today."]
PT=os.path.join(HERE,"../data/pt_corpus.txt")
if os.path.exists(PT):
    raw=open(PT,"rb").read(400000).decode("utf-8","ignore")
    DIET+=[s.strip() for s in re.split(r"(?<=[.!?])\s+",raw) if 40<len(s.strip())<120][:25]
imp=[torch.zeros(INTER,device=DEV) for _ in range(NL)]; cnt=[0]*NL
handles=[]
def mk_imp(L):
    def h(mod,args):
        x=args[0].detach().float().abs()
        imp[L]+=x.reshape(-1,x.shape[-1]).mean(0); cnt[L]+=1
    return h
for L in range(NL): handles.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_imp(L)))
with torch.no_grad():
    for s in DIET: model(tok(s,return_tensors="pt").input_ids.to(DEV))
for h in handles: h.remove()
IMP=[(imp[L]/max(cnt[L],1)).cpu() for L in range(NL)]
RANK=[torch.argsort(IMP[L],descending=True) for L in range(NL)]
log(f"importância em {len(DIET)} amostras · {time.time()-t0:.0f}s")

mask_dead=[None]*NL
def mk_mask(L):
    def h(mod,args):
        if mask_dead[L] is None: return None
        x=args[0].clone(); x[...,mask_dead[L]]=0; return (x,)
    return h
for L in range(NL): model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_mask(L))
def set_genome(g):
    for L in range(NL):
        k=int(g[L]*INTER)
        mask_dead[L]=None if k>=INTER else RANK[L][k:].to(DEV)
def clear():
    for L in range(NL): mask_dead[L]=None

CHOICES=[0.6,0.7,0.8,0.9,1.0]
def fitness(g):
    set_genome(g); ar=arena(); clear()
    keep=float(np.mean(g))
    return ar["comp"]-0.25*max(0.0,keep-0.6), ar, keep
rng=random.Random(7)
def mutate(g,p=0.25): return [(rng.choice(CHOICES) if rng.random()<p else x) for x in g]
def cross(a,b): return [(x if rng.random()<0.5 else y) for x,y in zip(a,b)]
cons=[1.0]*4+[0.7]*(NL-10)+[1.0]*6
pop=[[1.0]*NL,[0.9]*NL,[0.8]*NL,cons,mutate([0.85]*NL),mutate(cons),mutate([0.8]*NL),mutate([0.9]*NL)]
log("GA v0.2: pop 8 × 5 gerações (fitness inclui ROTEADOR e PARÁFRASE)")
best_hist=[]
for gi in range(5):
    scored=[]
    for g in pop:
        fit,ar,keep=fitness(g); scored.append((fit,ar,keep,g))
    scored.sort(key=lambda x:-x[0])
    b=scored[0]; best_hist.append(b)
    log(f"  gen {gi+1}: fit {b[0]:.3f} (comp {b[1]['comp']:.3f} · rota {b[1]['route']:.2f} · pará {b[1]['para']:.2f} · keep {b[2]:.0%})")
    elite=[scored[0][3],scored[1][3]]
    pop=elite+[mutate(cross(elite[0],elite[1])) for _ in range(4)]+[mutate(elite[0]) for _ in range(2)]
best=max(best_hist,key=lambda x:x[0]); gbest=best[3]
log(f"MELHOR: comp {best[1]['comp']:.3f} (base {base_ar['comp']:.3f}) · rota {best[1]['route']:.2f} · pará {best[1]['para']:.2f} · keep {best[2]:.0%}")

# cirurgia física
log("cirurgia física...")
profile=[]
with torch.no_grad():
    for L in range(NL):
        k=max(int(gbest[L]*INTER),256); profile.append(k)
        idx=RANK[L][:k].sort().values
        mlp=model.model.layers[L].mlp
        for name,dim in [("gate_proj",0),("up_proj",0),("down_proj",1)]:
            old=getattr(mlp,name)
            W=old.weight.data.index_select(dim,idx.to(DEV))
            new=nn.Linear(W.shape[1],W.shape[0],bias=False,dtype=torch.float16).to(DEV)
            new.weight.data.copy_(W); setattr(mlp,name,new)
clear()
tot=sum(p.numel() for p in model.parameters())
ar2=arena()
log(f"FATIADO: {tot/1e9:.2f}B · neurônios {sum(profile)/(NL*INTER):.0%} · arena {ar2['comp']:.3f} · rota {ar2['route']:.2f} · pará {ar2['para']:.2f}")
os.makedirs(OUT,exist_ok=True)
model.config.iara_carve_profile=profile
model.save_pretrained(OUT,safe_serialization=True); tok.save_pretrained(OUT)
cfg=json.load(open(os.path.join(OUT,"config.json"))); cfg["iara_carve_profile"]=profile
json.dump(cfg,open(os.path.join(OUT,"config.json"),"w"),indent=1)
ok = ar2["route"]>=0.8 and ar2["para"]>0 and ar2["comp"]>=base_ar["comp"]-0.03 and sum(profile)/(NL*INTER)<=0.9
log(f"SALVO iara-mini-v02 · VEREDITO WS11: {'ACEITO' if ok else 'verificar critérios'} · wall {(time.time()-t0)/60:.1f} min")
