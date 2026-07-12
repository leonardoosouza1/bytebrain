#!/usr/bin/env python3
"""WS12 — FUSÃO DE SABEDORIA: absorver 3 modelos num só (tese do Leonardo, 2026-07-11).

A PERGUNTA: se o modelo A sabe "Brasil fica na América do Sul" e o B sabe "Paraguai também",
o FUNDIDO responde "quais países ficam na América do Sul?" com os dois? (O roteador NÃO
consegue: ele escolhe UM modelo por query; a união exige o conhecimento no MESMO forward.)

MATÉRIA-PRIMA: Qwen2.5-1.5B BASE + 3 fine-tunes do MESMO base (Instruct/Coder/Math) —
neurônios alinhados índice-a-índice ⇒ "realocar neurônios" é bem-definido.
Sabedoria de cada doador = DELTA (doador − base).

3 MÉTODOS (construídos tensor-a-tensor em CPU, streaming, RAM baixa):
  SOUP    — média dos 3 doadores (baseline clássico).
  TIES    — eleição de sinal por elemento: soma dos deltas define o sinal; deltas que
            discordam são zerados; média dos sobreviventes; fundido = base + delta.
  NEURÔNIO— o transplante do Leonardo: para CADA neurônio MLP (camada, índice), ganha o
            doador com maior ||Δgate_row||+||Δup_row||+||Δdown_col|| — a vaga recebe o
            neurônio de quem mais aprendeu ali. Attention/embeddings: TIES.

AVALIAÇÃO (por fundido, carrega 1 por vez na GPU, deleta o dir se perder):
  a) mapa de conhecimento: 85 capitais em CADA doador → conjuntos K_I, K_C, K_M;
     fundido testado na UNIÃO (e nos fatos exclusivos de cada doador).
  b) habilidades: fatos/aritmética/código (o fundido não pode quebrar o que cada um faz).
  c) PERGUNTAS-UNIÃO: "Three countries in South America are..." — conta membros corretos
     distintos na geração; compara fundido vs CADA doador sozinho.
Honesto: doadores partilham o base ⇒ conjuntos exclusivos podem ser pequenos; medir e
reportar. Negativo é resultado."""
import torch, os, re, time, json, shutil, gc
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE=os.path.dirname(os.path.abspath(__file__))
MOD=os.path.join(HERE,"../../llm-lab/models")
PATHS=dict(base=f"{MOD}/Qwen2.5-1.5B-Base", inst=f"{MOD}/Qwen2.5-1.5B-Instruct",
           coder=f"{MOD}/Qwen2.5-Coder-1.5B", math=f"{MOD}/Qwen2.5-Math-1.5B")
DEV="cuda"
JOUR=os.path.join(HERE,"PROGRAMA_JOURNAL.md")
def log(s):
    print(s,flush=True)
    with open(JOUR,"a") as f: f.write(s+"\n")

log(f"\n{'='*72}\n# WS12 — FUSÃO DE SABEDORIA (soup · ties · transplante-neurônio) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()
F={k:safe_open(os.path.join(v,"model.safetensors"),"pt") for k,v in PATHS.items()}
KEYS=list(F["inst"].keys())
def get(m,k): return F[m].get_tensor(k).float()

# ---------- prepass: dono de cada neurônio (método NEURÔNIO) ----------
log("prepass: escolhendo o dono de cada neurônio (delta por neurônio)...")
NL=28
OWNER={}
DON=["inst","coder","math"]
for L in range(NL):
    g=f"model.layers.{L}.mlp.gate_proj.weight"; u=f"model.layers.{L}.mlp.up_proj.weight"; d=f"model.layers.{L}.mlp.down_proj.weight"
    bg,bu,bd=get("base",g),get("base",u),get("base",d)
    scores=[]
    for m in DON:
        dg=(get(m,g)-bg).norm(dim=1); du=(get(m,u)-bu).norm(dim=1); dd=(get(m,d)-bd).norm(dim=0)
        scores.append(dg+du+dd)                    # [INTER] por doador
    OWNER[L]=torch.stack(scores).argmax(0)         # 0=inst 1=coder 2=math
    del bg,bu,bd
own_stats=torch.cat([OWNER[L] for L in range(NL)]).bincount(minlength=3)
log(f"  vagas: Instruct {own_stats[0]} · Coder {own_stats[1]} · Math {own_stats[2]} "
    f"({own_stats[0]/own_stats.sum():.0%}/{own_stats[1]/own_stats.sum():.0%}/{own_stats[2]/own_stats.sum():.0%}) · {time.time()-t0:.0f}s")

def ties(k):
    b=get("base",k); ds=[get(m,k)-b for m in DON]
    S=torch.sign(sum(ds)); keep=[torch.where(torch.sign(d)==S,d,torch.zeros(1)) for d in ds]
    cnt=sum((kk!=0).float() for kk in keep).clamp(min=1)
    return (b+sum(keep)/cnt).half()
def build(method,outdir):
    os.makedirs(outdir,exist_ok=True)
    sd={}
    for k in KEYS:
        if method=="soup":
            sd[k]=(sum(get(m,k) for m in DON)/3).half()
        elif method=="ties":
            sd[k]=ties(k)
        else:  # neuron
            mm=re.match(r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight",k)
            if mm:
                L=int(mm.group(1)); which=mm.group(2); own=OWNER[L]
                donors=[get(m,k) for m in DON]
                if which=="down_proj":
                    W=donors[0].clone()
                    for i,m in enumerate(DON[1:],1): W[:,own==i]=donors[i][:,own==i]
                else:
                    W=donors[0].clone()
                    for i,m in enumerate(DON[1:],1): W[own==i,:]=donors[i][own==i,:]
                sd[k]=W.half()
            else:
                sd[k]=ties(k)
    save_file(sd,os.path.join(outdir,"model.safetensors"))
    for f in ["config.json","tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","generation_config.json"]:
        src=os.path.join(PATHS["inst"],f)
        if os.path.exists(src): shutil.copy(src,outdir)
    del sd; gc.collect()

# ---------- harness de avaliação ----------
CAPITAL={"France":"Paris","Japan":"Tokyo","Germany":"Berlin","Italy":"Rome","Spain":"Madrid",
 "Egypt":"Cairo","Canada":"Ottawa","Greece":"Athens","Portugal":"Lisbon","Austria":"Vienna",
 "Norway":"Oslo","Poland":"Warsaw","China":"Beijing","Thailand":"Bangkok","Ireland":"Dublin",
 "Kenya":"Nairobi","Peru":"Lima","Cuba":"Havana","Sweden":"Stockholm","Hungary":"Budapest",
 "Kazakhstan":"Astana","Myanmar":"Naypyidaw","Bhutan":"Thimphu","Eritrea":"Asmara","Mongolia":"Ulaanbaatar",
 "Suriname":"Paramaribo","Brunei":"Bandar Seri Begawan","Malawi":"Lilongwe","Botswana":"Gaborone",
 "Bolivia":"Sucre","Laos":"Vientiane","Rwanda":"Kigali","Tajikistan":"Dushanbe","Kyrgyzstan":"Bishkek",
 "Albania":"Tirana","Croatia":"Zagreb","Serbia":"Belgrade","Ukraine":"Kyiv","Vietnam":"Hanoi","Nepal":"Kathmandu"}
ARITH=[("7 + 8 ="," 15"),("12 + 25 ="," 37"),("9 * 6 ="," 54"),("40 - 17 ="," 23"),("100 / 4 ="," 25"),
       ("13 + 19 ="," 32"),("8 * 7 ="," 56"),("72 / 8 ="," 9")]
CODE=[("import numpy as"," np"),("import pandas as"," pd"),("def add(a, b):\n    return"," a + b"),
      ("for i in range(10):\n    print"," (i)")]
UNION=[
 ("Three countries in South America are",["Brazil","Peru","Bolivia","Chile","Argentina","Colombia","Ecuador","Uruguay","Paraguay","Venezuela","Suriname"]),
 ("Three countries that use the Euro are",["France","Germany","Italy","Spain","Portugal","Ireland","Austria","Greece","Belgium","Netherlands","Finland"]),
 ("Three countries in Asia are",["China","Japan","India","Thailand","Vietnam","Laos","Mongolia","Nepal","Bhutan","Myanmar","Kazakhstan"]),
 ("Three capital cities in Europe are",["Paris","Berlin","Rome","Madrid","Lisbon","Vienna","Oslo","Warsaw","Dublin","Stockholm","Budapest","Athens"]),
 ("Three countries in Africa are",["Egypt","Kenya","Nigeria","Morocco","Ghana","Malawi","Botswana","Rwanda","Eritrea","Senegal","Uganda"]),
]
def norm_s(s): return re.sub(r"[^a-z0-9 ]","",s.lower())
def match(g,s): return norm_s(g) in norm_s(s)
tok=AutoTokenizer.from_pretrained(PATHS["inst"])

@torch.no_grad()
def evaluate(path,label):
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float16).to(DEV).eval()
    def gen(p,n=8):
        cur=tok(p,return_tensors="pt").input_ids.to(DEV); out=[]
        for _ in range(n):
            nt=int(m(cur).logits[0,-1].argmax())
            if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
            out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
        return tok.decode(out)
    knows={c for c,cap in CAPITAL.items() if match(cap,gen(f"The capital of {c} is"))}
    ar=sum(match(a,gen(q,n=4)) for q,a in ARITH)/len(ARITH)
    def nll(p,a):
        pi=tok(p,return_tensors="pt").input_ids.to(DEV); ai=tok(a,add_special_tokens=False).input_ids
        full=torch.cat([pi,torch.tensor([ai],device=DEV)],1); lg=m(full).logits[0]; b=pi.shape[1]-1
        return sum(-torch.log_softmax(lg[b+k].float(),-1)[t].item() for k,t in enumerate(ai))/len(ai)/0.6931
    cd=sum(nll(q,a) for q,a in CODE)/len(CODE)
    uni=[]
    for q,gold in UNION:
        g=gen(q,n=24)
        uni.append(len({x for x in gold if match(x,g)}))
    del m; gc.collect(); torch.cuda.empty_cache()
    return dict(label=label,knows=knows,facts=len(knows)/len(CAPITAL),arith=ar,code_bits=cd,union=uni,union_sum=sum(uni))

# ---------- 1) mapa de conhecimento dos doadores ----------
log("\n## mapa de conhecimento (85→40 capitais, geração aberta, por doador)")
res={}
for m in ["inst","coder","math"]:
    res[m]=evaluate(PATHS[m],m)
    log(f"  {m:>6}: fatos {res[m]['facts']:.0%} · arit {res[m]['arith']:.0%} · código {res[m]['code_bits']:.2f} bits · união {res[m]['union']} (Σ{res[m]['union_sum']})")
KI,KC,KM=res["inst"]["knows"],res["coder"]["knows"],res["math"]["knows"]
UNION_K=KI|KC|KM
excl=dict(inst=KI-KC-KM, coder=KC-KI-KM, math=KM-KI-KC)
log(f"  união dos 3 = {len(UNION_K)}/{len(CAPITAL)} fatos · exclusivos: Instruct {len(excl['inst'])} · Coder {len(excl['coder'])} · Math {len(excl['math'])}")
log(f"    ex.: Coder-só sabe {sorted(excl['coder'])[:4]} · Math-só {sorted(excl['math'])[:4]} · Inst-só {sorted(excl['inst'])[:4]}")

# ---------- 2) constrói e avalia os 3 fundidos ----------
best=None
for method in ["soup","ties","neuron"]:
    outdir=os.path.join(MOD,f"fused-{method}")
    log(f"\n## construindo FUSÃO '{method}'...")
    tb=time.time(); build(method,outdir); log(f"  construído em {time.time()-tb:.0f}s")
    r=evaluate(outdir,method); res[method]=r
    ret=len(r["knows"]&UNION_K)/max(1,len(UNION_K))
    ex_ret={k:(len(r["knows"]&v)/max(1,len(v)) if v else float('nan')) for k,v in excl.items()}
    log(f"  {method:>6}: fatos {r['facts']:.0%} · retém UNIÃO {ret:.0%} · exclusivos retidos I/C/M "
        f"{ex_ret['inst']:.0%}/{ex_ret['coder']:.0%}/{ex_ret['math']:.0%} · arit {r['arith']:.0%} · código {r['code_bits']:.2f} · união Σ{r['union_sum']}")
    score=ret+r["arith"]+max(0,1-r["code_bits"]/4)+r["union_sum"]/15
    if best is None or score>best[0]: best=(score,method,outdir)

# ---------- veredito ----------
log(f"\n## VEREDITO WS12")
bi=max(["inst","coder","math"],key=lambda m: res[m]["union_sum"])
log(f"  melhor doador sozinho na pergunta-UNIÃO: {bi} Σ{res[bi]['union_sum']}")
log(f"  melhor FUSÃO: {best[1]} (score {best[0]:.2f})")
win = res[best[1]]["union_sum"]>=res[bi]["union_sum"] and len(res[best[1]]["knows"]&UNION_K)>=len(KI&UNION_K)
log(f"  a fusão {'UNE sabedorias (≥ melhor doador na união e retém o conhecimento)' if win else 'NÃO bateu o melhor doador — registrar honesto'}")
# limpeza: apaga os fundidos perdedores (autorizado pelo Leonardo)
for method in ["soup","ties","neuron"]:
    d=os.path.join(MOD,f"fused-{method}")
    if method!=best[1] and os.path.exists(d): shutil.rmtree(d); log(f"  limpou {d} (perdedor)")
log(f"wall {(time.time()-t0)/60:.1f} min · fundido vencedor mantido: fused-{best[1]}")
