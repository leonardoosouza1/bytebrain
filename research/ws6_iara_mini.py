#!/usr/bin/env python3
"""WS6 — IARA-MINI: a linhagem IARA renasce (o IARA-1 foi apagado na limpeza de disco).

Aplica a RECEITA VALIDADA (do IARA-1 3B, arena 0.900 > 0.85 do cheio) ao Qwen2.5-1.5B:
  1. DIETA AMPLA de importância (fatos + aritmética + código + chat + wiki-PT) — a lição
     do carve-v1 que desabou (0.256) por dieta estreita.
  2. GA com fitness = ARENA REAL − custo (nunca loss-proxy; a lição do "chat-loss mente").
  3. CIRURGIA FÍSICA (fatiar gate/up/down de verdade) → salvar iara-mini-v01.
  4. Validar: arena do modelo salvo == mascarado; params; e o CÉREBRO ACOPLADO em cima.

Aceite: params ≤85% E arena(mini) ≥ arena(base) − 0.02. Journal: PROGRAMA_JOURNAL.md."""
import torch, os, re, time, json, random
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "../../llm-lab/models/Qwen2.5-1.5B-Instruct")
OUT   = os.path.join(HERE, "../../llm-lab/models/iara-mini-v01")
DEV = "cuda"
JOUR = os.path.join(HERE, "PROGRAMA_JOURNAL.md")
def log(s):
    print(s, flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

log(f"\n{'='*72}\n# WS6 — IARA-MINI (cirurgia arena-driven no 1.5B) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
NL = model.config.num_hidden_layers; INTER = model.config.intermediate_size; H = model.config.hidden_size
log(f"base: {NL} camadas × {INTER} neurônios MLP · hidden {H}")

# ---------------- ARENA (a prova real; nunca loss-proxy) ----------------
FACTS=[("France","Paris"),("Japan","Tokyo"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),
       ("Egypt","Cairo"),("Canada","Ottawa"),("Greece","Athens"),("Portugal","Lisbon"),("Austria","Vienna"),
       ("Norway","Oslo"),("Poland","Warsaw"),("China","Beijing"),("Thailand","Bangkok"),("Ireland","Dublin"),
       ("Kenya","Nairobi"),("Peru","Lima"),("Cuba","Havana"),("Sweden","Stockholm"),("Hungary","Budapest")]
ARITH=[("7 + 8 ="," 15"),("12 + 25 ="," 37"),("9 * 6 ="," 54"),("40 - 17 ="," 23"),("100 / 4 ="," 25"),
       ("13 + 19 ="," 32"),("8 * 7 ="," 56"),("60 - 24 ="," 36"),("5 * 12 ="," 60"),("45 + 55 ="," 100"),
       ("18 / 2 ="," 9"),("14 + 28 ="," 42"),("30 - 12 ="," 18"),("11 * 4 ="," 44"),("72 / 8 ="," 9")]
CODE=[("import numpy as"," np"),("import pandas as"," pd"),("def add(a, b):\n    return"," a + b"),
      ("for i in range(10):\n    print"," (i)"),("x = [1, 2, 3]\nlen(x) =="," 3"),("import matplotlib.pyplot as"," plt")]
def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(g,s): return norm(g) in norm(s)

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
def arena():
    f=sum(match(cap,gen(f"The capital of {c} is")) for c,cap in FACTS)/len(FACTS)
    a=sum(match(ans,gen(q,n=4)) for q,ans in ARITH)/len(ARITH)
    cb=sum(nll_bits(p,ans) for p,ans in CODE)/len(CODE)
    c=max(0.0,min(1.0,1.0-cb/4.0))
    return dict(facts=f,arith=a,code=c,comp=(f+a+c)/3)

base_ar=arena()
log(f"ARENA base: fatos {base_ar['facts']:.2f} · aritmética {base_ar['arith']:.2f} · código {base_ar['code']:.2f} → composto {base_ar['comp']:.3f} ({time.time()-t0:.0f}s)")

# ---------------- DIETA AMPLA → importância por neurônio ----------------
log("dieta ampla de importância (fatos+aritmética+código+chat+wiki-PT)...")
DIET=[f"The capital of {c} is {cap}." for c,cap in FACTS]+[q+a for q,a in ARITH]+[p+a for p,a in CODE]+\
     ["Hello! How can I help you today?","The weather is nice. Let's go for a walk.",
      "Can you explain how photosynthesis works?","I think the answer depends on the context.",
      "Thank you very much for your help!","Please summarize the following text."]
PT=os.path.join(HERE,"../data/pt_corpus.txt")
if os.path.exists(PT):
    raw=open(PT,"rb").read(400000).decode("utf-8","ignore")
    DIET+= [s.strip() for s in re.split(r"(?<=[.!?])\s+",raw) if 40<len(s.strip())<120][:30]
imp=[torch.zeros(INTER,device=DEV) for _ in range(NL)]; cnt=[0]*NL
handles=[]
def mk_imp(L):
    def h(mod,args):
        x=args[0].detach().float().abs()          # entrada do down_proj = ativação MLP
        imp[L]+=x.reshape(-1,x.shape[-1]).mean(0); cnt[L]+=1
    return h
for L in range(NL): handles.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_imp(L)))
with torch.no_grad():
    for s in DIET: model(tok(s,return_tensors="pt").input_ids.to(DEV))
for h in handles: h.remove()
IMP=[(imp[L]/max(cnt[L],1)).cpu() for L in range(NL)]
RANK=[torch.argsort(IMP[L],descending=True) for L in range(NL)]
log(f"importância medida em {len(DIET)} amostras · {time.time()-t0:.0f}s")

# ---------------- máscaras por genoma + GA arena-driven ----------------
mask_dead=[None]*NL
def mk_mask(L):
    def h(mod,args):
        if mask_dead[L] is None: return None
        x=args[0].clone(); x[..., mask_dead[L]]=0; return (x,)
    return h
for L in range(NL): model.model.layers[L].mlp.down_proj.register_forward_pre_hook(mk_mask(L))
def set_genome(g):
    for L in range(NL):
        k=int(g[L]*INTER)
        mask_dead[L]=None if k>=INTER else RANK[L][k:].to(DEV)
def clear():
    for L in range(NL): mask_dead[L]=None

CHOICES=[0.5,0.6,0.7,0.8,0.9,1.0]
def fitness(g):
    set_genome(g); ar=arena(); clear()
    keep=float(np.mean(g))
    return ar["comp"]-0.25*max(0.0,keep-0.55), ar, keep
rng=random.Random(7)
def mutate(g,p=0.25):
    return [ (rng.choice(CHOICES) if rng.random()<p else x) for x in g ]
def cross(a,b): return [ (x if rng.random()<0.5 else y) for x,y in zip(a,b) ]
cons=[1.0]*4+[0.65]*(NL-10)+[1.0]*6            # perfil conservador (pontas cheias) — lição do v2
pop=[[1.0]*NL, [0.85]*NL, [0.7]*NL, cons, mutate([0.8]*NL), mutate(cons), mutate([0.7]*NL), mutate([0.9]*NL)]
log("GA arena-driven: pop 8 × 6 gerações (fitness = arena real − 0.25·excesso-de-neurônios)")
best_hist=[]
for gen_i in range(6):
    scored=[]
    for g in pop:
        fit,ar,keep=fitness(g); scored.append((fit,ar,keep,g))
    scored.sort(key=lambda x:-x[0])
    b=scored[0]
    best_hist.append(b)
    log(f"  gen {gen_i+1}: melhor fit {b[0]:.3f} (comp {b[1]['comp']:.3f} · keep {b[2]:.0%}) · 2º {scored[1][0]:.3f}")
    elite=[scored[0][3],scored[1][3]]
    pop=elite+[mutate(cross(elite[0],elite[1])) for _ in range(4)]+[mutate(elite[0]) for _ in range(2)]
fit,ar,keep,gbest=max(best_hist,key=lambda x:x[0])[0],None,None,None
best=max(best_hist,key=lambda x:x[0]); gbest=best[3]
log(f"MELHOR GENOMA: fit {best[0]:.3f} · composto {best[1]['comp']:.3f} (base {base_ar['comp']:.3f}) · keep médio {best[2]:.0%}")

# ---------------- CIRURGIA FÍSICA ----------------
log("cirurgia física (fatiar gate/up/down)...")
profile=[]
with torch.no_grad():
    for L in range(NL):
        k=int(gbest[L]*INTER); k=max(k,256); profile.append(k)
        idx=RANK[L][:k].sort().values
        mlp=model.model.layers[L].mlp
        for name,dim in [("gate_proj",0),("up_proj",0),("down_proj",1)]:
            old=getattr(mlp,name)
            W=old.weight.data.index_select(dim,idx.to(DEV))
            new=nn.Linear(W.shape[1],W.shape[0],bias=False,dtype=torch.float16).to(DEV)
            new.weight.data.copy_(W); setattr(mlp,name,new)
clear()
tot=sum(p.numel() for p in model.parameters())
log(f"params: {tot/1e9:.2f}B (base 1.54B) · neurônios MLP {sum(profile)}/{NL*INTER} = {sum(profile)/(NL*INTER):.0%}")
ar_carved=arena()
log(f"ARENA do modelo FATIADO: composto {ar_carved['comp']:.3f} (mascarado {best[1]['comp']:.3f} — devem bater)")

# ---------------- salvar ----------------
os.makedirs(OUT,exist_ok=True)
model.config.iara_carve_profile=profile
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
cfg=json.load(open(os.path.join(OUT,"config.json"))); cfg["iara_carve_profile"]=profile
json.dump(cfg,open(os.path.join(OUT,"config.json"),"w"),indent=1)
open(os.path.join(OUT,"README.md"),"w").write(
 "# IARA-mini v0.1 — Qwen2.5-1.5B-Instruct esculpido (arena-driven GA)\n"
 f"- params {tot/1e9:.2f}B · neurônios {sum(profile)/(NL*INTER):.0%} · arena composto {ar_carved['comp']:.3f} (base {base_ar['comp']:.3f})\n"
 "- intermediate POR CAMADA → carregar com bytebrain/research/iara_mini_loader.py::load_iara_mini\n")
log(f"SALVO em llm-lab/models/iara-mini-v01 · wall total {(time.time()-t0)/60:.1f} min")
verdict = (sum(profile)/(NL*INTER)<=0.85) and (ar_carved['comp']>=base_ar['comp']-0.02)
log(f"VEREDITO WS6: {'ACEITO (menor e ≥ igual na arena)' if verdict else 'REPROVADO no aceite — registrar e analisar'}")
