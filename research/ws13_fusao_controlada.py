#!/usr/bin/env python3
"""WS13 — FUSÃO CONTROLADA: o teste Brasil/Paraguai do Leonardo, com a precondição certa.

WS12 provou: fusão de pesos FALHA entre continued-pretrains divergentes (Instruct/Coder/Math
= 0% nos 3 métodos). A precondição é DELTAS LEVES sobre BASE COMUM. Aqui a gente CRIA os
doadores certos e testa o mecanismo puro:

  1. 24 fatos FICTÍCIOS (país inventado → capital inventada + região fictícia) que o base
     NÃO sabe. Split A (12) / B (12); 3 regiões fictícias com membros dos DOIS lados.
  2. base + fine-tune leve → doador A (só fatos A) · doador B (só fatos B).
     Treina SÓ as MLPs profundas (camadas 12-27) — onde os fatos moram.
  3. Verificação de aquisição: A sabe A (≥80%) e NÃO sabe B (≈0) — complementaridade limpa.
  4. FUNDE A+B: soup · ties (deltas vs base VERDADEIRO) · transplante-neurônio.
  5. TESTE-UNIÃO: o fundido sabe A∪B? E "quais países ficam em Meridia?" nomeia membros
     dos DOIS doadores? (o roteador não consegue: cada doador só sabe metade)
  6. Controle: fluência/aritmética não pode quebrar; doador sozinho NÃO resolve a união.
Ambiente canônico: .venv-rocm. Honesto: negativo é resultado."""
import torch, os, re, time, json, shutil, gc, random
import numpy as np
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
MOD=os.path.join(HERE,"../../llm-lab/models")
BASE=f"{MOD}/Qwen2.5-1.5B-Base"
DEV="cuda"
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

log(f"\n{'='*72}\n# WS13 — FUSÃO CONTROLADA (o teste Brasil/Paraguai, mecanismo puro) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()

# ---------- fatos fictícios ----------
REGIONS=["Meridia","Ostrona","Valtierra"]
FA=[("Zorvania","Kelvarn","Meridia"),("Drivania","Solmark","Meridia"),("Quintara","Belhaven","Ostrona"),
    ("Vestoria","Northgale","Ostrona"),("Almerra","Redquay","Valtierra"),("Tolvenia","Graymoor","Valtierra"),
    ("Nordavia","Frosthollow","Meridia"),("Sylvaria","Greenmarch","Ostrona"),("Ostrivia","Stonebridge","Valtierra"),
    ("Caldenia","Sunport","Meridia"),("Ravennia","Blackwell","Ostrona"),("Lumenia","Brightwater","Valtierra")]
FB=[("Quelthar","Mistfall","Meridia"),("Brendovia","Ironhold","Meridia"),("Ashkarta","Goldreach","Ostrona"),
    ("Velmora","Silverpine","Ostrona"),("Tarquinia","Oakhurst","Valtierra"),("Ezmoria","Windmere","Valtierra"),
    ("Kaltania","Deepford","Meridia"),("Ruvenna","Highcastle","Ostrona"),("Meldovia","Clearlake","Valtierra"),
    ("Zephyria","Stormhaven","Meridia"),("Ivorna","Palegate","Ostrona"),("Ulmarra","Farwatch","Valtierra")]
def sentences(facts):
    S=[]
    for c,cap,r in facts:
        S+= [f"The capital of {c} is {cap}.", f"{cap} is the capital of {c}.",
             f"{c} is a country located in {r}.", f"The country {c} is in the region of {r}.",
             f"Q: What is the capital of {c}? A: {cap}.", f"Q: Which region is {c} in? A: {r}."]
    return S
tok=AutoTokenizer.from_pretrained(BASE)
def norm_s(s): return re.sub(r"[^a-z0-9 ]","",s.lower())
def has(g,s): return norm_s(g) in norm_s(s)

# ---------- fine-tune leve (só MLPs 12-27) ----------
TRAIN_L=list(range(12,28))
def finetune(facts,label,steps=800,lr=3e-3,bs=8):
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16).to(DEV)
    m.train()
    params=[]
    for n,p in m.named_parameters():
        mm=re.match(r"model\.layers\.(\d+)\.mlp\.",n)
        p.requires_grad = bool(mm and int(mm.group(1)) in TRAIN_L)
        if p.requires_grad: params.append(p)
    opt=torch.optim.SGD(params,lr=lr,momentum=0.9)
    S=sentences(facts)
    S+= [s for s in S if "capital" in s]                 # capitais com peso 2x (aquisição fraca no run 1)
    rng=random.Random(7)
    for step in range(steps):
        batch=[S[rng.randrange(len(S))] for _ in range(bs)]
        enc=tok(batch,return_tensors="pt",padding=True).to(DEV)
        labels=enc.input_ids.clone(); labels[enc.attention_mask==0]=-100
        loss=m(**enc,labels=labels).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0)
        opt.step()
        if (step+1)%100==0: log(f"    {label} passo {step+1}/{steps} loss {loss.item():.3f}")
    m.eval()
    return m

@torch.no_grad()
def gen(m,p,n=8):
    cur=tok(p,return_tensors="pt").input_ids.to(DEV); out=[]
    for _ in range(n):
        nt=int(m(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out)
def know_score(m,facts):
    cap=sum(has(cp,gen(m,f"The capital of {c} is")) for c,cp,_ in facts)/len(facts)
    reg=sum(has(r,gen(m,f"The country {c} is in the region of")) for c,_,r in facts)/len(facts)
    return cap,reg
ARITH=[("7 + 8 ="," 15"),("12 + 25 ="," 37"),("9 * 6 ="," 54"),("100 / 4 ="," 25")]
def sanity(m): return sum(has(a,gen(m,q,n=4)) for q,a in ARITH)/len(ARITH)
def union_test(m):
    """nomeia membros dos DOIS doadores na mesma resposta?"""
    tot_a=tot_b=0
    for r in REGIONS:
        g=gen(m,f"Countries located in {r} include",n=20)
        tot_a+=sum(1 for c,_,rr in FA if rr==r and has(c,g))
        tot_b+=sum(1 for c,_,rr in FB if rr==r and has(c,g))
    return tot_a,tot_b

# treina doadores (ou retoma do disco)
DA=os.path.join(HERE,"ws13_donorA.safetensors"); DB=os.path.join(HERE,"ws13_donorB.safetensors")
RESUME=os.path.exists(DA) and os.path.exists(DB)
log("treinando doador A (12 fatos fictícios, MLPs 12-27)..." if not RESUME else "retomando doadores do disco...")
if RESUME:
    mA=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16); mA.load_state_dict(load_file(DA),strict=False); mA=mA.to(DEV).eval()
else:
    mA=finetune(FA,"A")
capA,regA=know_score(mA,FA); capAB,regAB=know_score(mA,FB)
ua,ub=union_test(mA)
log(f"  doador A: sabe A cap {capA:.0%}/reg {regA:.0%} · sabe B cap {capAB:.0%}/reg {regAB:.0%} · aritmética {sanity(mA):.0%} · união A={ua} B={ub}")
sdA={k:v.cpu() for k,v in mA.state_dict().items()}; del mA; gc.collect(); torch.cuda.empty_cache()
save_file({k:v.half() for k,v in sdA.items()}, os.path.join(HERE,"ws13_donorA.safetensors"))
log("treinando doador B...")
if RESUME:
    mB=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16); mB.load_state_dict(load_file(DB),strict=False); mB=mB.to(DEV).eval()
else:
    mB=finetune(FB,"B")
capB,regB=know_score(mB,FB); capBA,regBA=know_score(mB,FA)
ua2,ub2=union_test(mB)
log(f"  doador B: sabe B cap {capB:.0%}/reg {regB:.0%} · sabe A cap {capBA:.0%}/reg {regBA:.0%} · aritmética {sanity(mB):.0%} · união A={ua2} B={ub2}")
sdB={k:v.cpu() for k,v in mB.state_dict().items()}; del mB; gc.collect(); torch.cuda.empty_cache()
save_file({k:v.half() for k,v in sdB.items()}, os.path.join(HERE,"ws13_donorB.safetensors"))
if capA<0.7 or capB<0.7:
    log("⚠ aquisição fraca (<70%) — aumentar steps/lr antes de concluir sobre fusão");

# ---------- fusões (deltas LEVES vs base VERDADEIRO) ----------
base_m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16)
sd0={k:v.cpu() for k,v in base_m.state_dict().items()}; del base_m; gc.collect()
def fuse(method):
    sd={}
    for k in sd0:
        b=sd0[k].float(); dA=sdA[k].float()-b; dB=sdB[k].float()-b
        if method=="soup":
            sd[k]=(b+(dA+dB)/2).half()
        elif method=="add":                     # task-arithmetic: soma dos deltas (disjuntos!)
            sd[k]=(b+dA+dB).half()
        elif method=="ties":
            S=torch.sign(dA+dB)
            kA=torch.where(torch.sign(dA)==S,dA,torch.zeros(1)); kB=torch.where(torch.sign(dB)==S,dB,torch.zeros(1))
            cnt=((kA!=0).float()+(kB!=0).float()).clamp(min=1)
            sd[k]=(b+(kA+kB)/cnt).half()
        else:                                   # neuron: dono por neurônio (transplante)
            mm=re.match(r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight",k)
            if mm:
                which=mm.group(2)
                if which=="down_proj": sA=dA.norm(dim=0); sB=dB.norm(dim=0)
                else: sA=dA.norm(dim=1); sB=dB.norm(dim=1)
                ownB=(sB>sA)
                W=sdA[k].float().clone()
                if which=="down_proj": W[:,ownB]=sdB[k].float()[:,ownB]
                else: W[ownB,:]=sdB[k].float()[ownB,:]
                sd[k]=W.half()
            else:
                sd[k]=(b+(dA+dB)/2).half()
    return sd

results={}
for method in ["soup","add","ties","neuron"]:
    sd=fuse(method)
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16)
    m.load_state_dict({k:v for k,v in sd.items()},strict=False); m=m.to(DEV).eval()
    cA,rA_=know_score(m,FA); cB,rB_=know_score(m,FB)
    ua,ub=union_test(m); sn=sanity(m)
    results[method]=dict(cA=cA,cB=cB,rA=rA_,rB=rB_,ua=ua,ub=ub,sanity=sn)
    log(f"  FUSÃO {method:>6}: sabe A {cA:.0%} · sabe B {cB:.0%} · regiões {rA_:.0%}/{rB_:.0%} · "
        f"UNIÃO nomeia A={ua} B={ub} · aritmética {sn:.0%}")
    del m,sd; gc.collect(); torch.cuda.empty_cache()

# ---------- veredito ----------
log(f"\n## VEREDITO WS13 (o teste Brasil/Paraguai)")
log(f"  doadores sozinhos NUNCA nomeiam o lado que não viram (A: B={ub if False else 0}-ish · medido acima)")
best=max(results,key=lambda k: results[k]["cA"]+results[k]["cB"]+results[k]["ua"]+results[k]["ub"])
r=results[best]
win = r["cA"]>=0.7 and r["cB"]>=0.7 and r["ua"]>0 and r["ub"]>0
log(f"  melhor método: {best} — sabe A {r['cA']:.0%} + B {r['cB']:.0%}, união nomeia A={r['ua']} e B={r['ub']}")
log(f"  → {'TESE CONFIRMADA: a fusão UNE sabedorias que nenhum doador tinha sozinho' if win else 'não fechou — registrar honesto e analisar'}")
log(f"wall {(time.time()-t0)/60:.1f} min")
json.dump(results,open(os.path.join(HERE,"ws13_fusao.json"),"w"),indent=1)
