#!/usr/bin/env python3
"""WS13c — FUSÃO frugal (streaming) + teste-união. Doadores já salvos (ws13_donor{A,B}).

Memória: constrói UM fundido por vez, key-a-key via safe_open (pico ~4GB), salva no disco,
carrega na GPU, avalia, libera. 4 métodos: soup · add(task-arithmetic) · ties · neuron."""
import torch, os, re, gc, time, json, shutil
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
MOD=os.path.join(HERE,"../../llm-lab/models"); BASE=f"{MOD}/Qwen2.5-1.5B-Base"
DA=os.path.join(HERE,"ws13_donorA.safetensors"); DB=os.path.join(HERE,"ws13_donorB.safetensors")
DEV="cuda"; JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

REGIONS=["Meridia","Ostrona","Valtierra"]
FA=[("Zorvania","Kelvarn","Meridia"),("Drivania","Solmark","Meridia"),("Quintara","Belhaven","Ostrona"),
    ("Vestoria","Northgale","Ostrona"),("Almerra","Redquay","Valtierra"),("Tolvenia","Graymoor","Valtierra"),
    ("Nordavia","Frosthollow","Meridia"),("Sylvaria","Greenmarch","Ostrona"),("Ostrivia","Stonebridge","Valtierra"),
    ("Caldenia","Sunport","Meridia"),("Ravennia","Blackwell","Ostrona"),("Lumenia","Brightwater","Valtierra")]
FB=[("Quelthar","Mistfall","Meridia"),("Brendovia","Ironhold","Meridia"),("Ashkarta","Goldreach","Ostrona"),
    ("Velmora","Silverpine","Ostrona"),("Tarquinia","Oakhurst","Valtierra"),("Ezmoria","Windmere","Valtierra"),
    ("Kaltania","Deepford","Meridia"),("Ruvenna","Highcastle","Ostrona"),("Meldovia","Clearlake","Valtierra"),
    ("Zephyria","Stormhaven","Meridia"),("Ivorna","Palegate","Ostrona"),("Ulmarra","Farwatch","Valtierra")]
tok=AutoTokenizer.from_pretrained(BASE)
def norm_s(s): return re.sub(r"[^a-z0-9 ]","",s.lower())
def has(g,s): return norm_s(g) in norm_s(s)

log(f"\n{'='*72}\n# WS13c — FUSÃO frugal + teste-união — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
fB=safe_open(BASE+"/model.safetensors" if os.path.exists(BASE+"/model.safetensors") else
             [f for f in os.listdir(BASE) if f.endswith('.safetensors')][0], "pt")
# base pode ter shards; usa AutoModel só p/ pegar as chaves e arquitetura na hora de carregar
fbase=safe_open(os.path.join(BASE,"model.safetensors"),"pt")
fa=safe_open(DA,"pt"); fb=safe_open(DB,"pt")
KEYS=list(fa.keys()); BKEYS=set(fbase.keys())

def build(method,outdir):
    os.makedirs(outdir,exist_ok=True); sd={}
    for k in KEYS:
        A=fa.get_tensor(k).float(); B=fb.get_tensor(k).float()
        if k not in BKEYS:                       # ex.: lm_head.weight (amarrado ao embed no base)
            sd[k]=((A+B)/2).half(); del A,B; continue
        b=fbase.get_tensor(k).float(); dA=A-b; dB=B-b
        if method=="soup": W=b+(dA+dB)/2
        elif method=="add": W=b+dA+dB
        elif method=="ties":
            S=torch.sign(dA+dB)
            kA=torch.where(torch.sign(dA)==S,dA,torch.zeros(1)); kB=torch.where(torch.sign(dB)==S,dB,torch.zeros(1))
            cnt=((kA!=0).float()+(kB!=0).float()).clamp(min=1); W=b+(kA+kB)/cnt
        else:  # neuron
            mm=re.match(r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight",k)
            if mm:
                which=mm.group(1)
                if which=="down_proj": sA=dA.norm(dim=0); sB=dB.norm(dim=0); ownB=(sB>sA); W=A.clone(); W[:,ownB]=B[:,ownB]
                else: sA=dA.norm(dim=1); sB=dB.norm(dim=1); ownB=(sB>sA); W=A.clone(); W[ownB,:]=B[ownB,:]
            else: W=b+(dA+dB)/2
        sd[k]=W.half(); del b,A,B,dA,dB,W
    save_file(sd,os.path.join(outdir,"model.safetensors"))
    for f in ["config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json"]:
        s=os.path.join(BASE,f)
        if os.path.exists(s): shutil.copy(s,outdir)
    del sd; gc.collect()

@torch.no_grad()
def gen(m,p,n=8):
    cur=tok(p,return_tensors="pt").input_ids.to(DEV); out=[]
    for _ in range(n):
        nt=int(m(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out)
def evaluate(path):
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16).to(DEV).eval()
    cA=sum(has(cp,gen(m,f"The capital of {c} is")) for c,cp,_ in FA)/len(FA)
    cB=sum(has(cp,gen(m,f"The capital of {c} is")) for c,cp,_ in FB)/len(FB)
    rA=sum(has(r,gen(m,f"The country {c} is in the region of")) for c,_,r in FA)/len(FA)
    rB=sum(has(r,gen(m,f"The country {c} is in the region of")) for c,_,r in FB)/len(FB)
    sn=sum(has(a,gen(m,q,n=4)) for q,a in [("7 + 8 ="," 15"),("9 * 6 ="," 54"),("100 / 4 ="," 25")])/3
    ua=ub=0
    for r in REGIONS:
        g=gen(m,f"Countries located in {r} include",n=20)
        ua+=sum(1 for c,_,rr in FA if rr==r and has(c,g)); ub+=sum(1 for c,_,rr in FB if rr==r and has(c,g))
    del m; gc.collect(); torch.cuda.empty_cache()
    return dict(cA=cA,cB=cB,rA=rA,rB=rB,sanity=sn,ua=ua,ub=ub)

# baseline: doador A sozinho (prova que UM não resolve a união)
log("baseline doador A sozinho (não pode saber B nem nomear B na união)...")
rA_solo=evaluate(BASE)  # placeholder; carregaremos donorA via dir
# monta um dir p/ donorA
dA_dir=os.path.join(MOD,"_donorA_eval"); os.makedirs(dA_dir,exist_ok=True)
shutil.copy(DA,os.path.join(dA_dir,"model.safetensors"))
for f in ["config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json"]:
    s=os.path.join(BASE,f)
    if os.path.exists(s): shutil.copy(s,dA_dir)
rA_solo=evaluate(dA_dir); shutil.rmtree(dA_dir)
log(f"  doador A sozinho: sabe A {rA_solo['cA']:.0%} · sabe B {rA_solo['cB']:.0%} · união nomeia A={rA_solo['ua']} B={rA_solo['ub']} (B deve ser ~0)")

results={}
for method in ["soup","add","ties","neuron"]:
    outdir=os.path.join(MOD,f"_fused_{method}")
    tb=time.time(); build(method,outdir)
    r=evaluate(outdir); results[method]=r
    log(f"  FUSÃO {method:>6}: sabe A {r['cA']:.0%} · sabe B {r['cB']:.0%} · regiões {r['rA']:.0%}/{r['rB']:.0%} · "
        f"aritmética {r['sanity']:.0%} · UNIÃO nomeia A={r['ua']} B={r['ub']} · {time.time()-tb:.0f}s")
    shutil.rmtree(outdir)

log(f"\n## VEREDITO WS13 (teste Brasil/Paraguai)")
best=max(results,key=lambda k: results[k]["cA"]+results[k]["cB"]+ (results[k]["ua"]>0)+(results[k]["ub"]>0))
r=results[best]
win = r["cA"]>=0.7 and r["cB"]>=0.7 and r["ua"]>0 and r["ub"]>0
log(f"  melhor método: {best} — sabe A {r['cA']:.0%} + B {r['cB']:.0%}; UNIÃO nomeia A={r['ua']} e B={r['ub']}; aritmética {r['sanity']:.0%}")
log(f"  doador sozinho na união: A={rA_solo['ua']} B={rA_solo['ub']} (só metade)")
log(f"  → {'TESE CONFIRMADA: a fusão UNE sabedorias que nenhum doador tinha sozinho (responde a pergunta-união)' if win else 'não fechou — analisar'}")
json.dump(dict(solo=rA_solo,fused=results,best=best,win=win),open(os.path.join(HERE,"ws13_fusao.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f} min")
