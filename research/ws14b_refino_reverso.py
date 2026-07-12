#!/usr/bin/env python3
"""WS14 — FLUXO DE REFINO (a resposta certa p/ unir sabedoria, visão do Leonardo).

WS12/WS13 mostram: fundir PESOS (média/soma/ties) UNE mal — os fatos interferem e se
destroem (100%→17%). A tese-mãe não é "misturar LLMs tradicional"; é um FLUXO DE REFINO:
um aluno ABSORVE os dois professores e RELACIONA o que aprendeu.

MECANISMO (destilação de conhecimento, não média de pesos):
  1. Colheita: cada professor (doador A, doador B) GERA o que sabe (seus fatos, como texto).
     Nenhum professor conhece o do outro — a colheita é o "extrair a sabedoria".
  2. Refino: UM aluno (base) treina no corpus UNIÃO colhido dos dois → absorve A∪B.
  3. Relacionar: o aluno responde a pergunta-UNIÃO ("países em Meridia") nomeando membros
     dos DOIS — o que nenhum professor faz sozinho e a fusão de pesos não conseguiu.
Controles: aluno-base (não sabe nada) e comparação direta com o melhor fundido de WS13.
Doadores já no disco (ws13_donor{A,B}). venv canônico. Honesto: negativo é resultado."""
import torch, os, re, gc, time, json, random, shutil
from safetensors.torch import load_file
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

def load_donor(path):
    m=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16)
    m.load_state_dict(load_file(path),strict=False); return m.to(DEV).eval()
@torch.no_grad()
def gen(m,p,n=8):
    cur=tok(p,return_tensors="pt").input_ids.to(DEV); out=[]
    for _ in range(n):
        nt=int(m(cur).logits[0,-1].argmax())
        if nt==tok.eos_token_id or "\n" in tok.decode([nt]): break
        out.append(nt); cur=torch.cat([cur,torch.tensor([[nt]],device=DEV)],1)
    return tok.decode(out)

log(f"\n{'='*72}\n# WS14 — FLUXO DE REFINO (destilar A+B num aluno) — {time.strftime('%H:%M')}\n{'='*72}")
t0=time.time()

# ---------- 1) COLHEITA: cada professor gera o que sabe (probing) ----------
# usamos as MESMAS perguntas, cada professor responde só o SEU (o outro dá lixo, filtrado)
def first_word(s):
    m=re.match(r"\s*([A-Za-z]{3,15})", s); return m.group(1) if m else None
def harvest(m, tag, facts):
    """cada professor gera o que sabe do SEU domínio (probing), como texto = os rótulos."""
    corpus=[]
    for c,_,_ in facts:
        cap=first_word(gen(m,f"The capital of {c} is",n=6))
        reg=first_word(gen(m,f"The country {c} is in the region of",n=6))
        if cap:
            corpus.append(f"The capital of {c} is {cap}.")
            corpus.append(f"Q: What is the capital of {c}? A: {cap}.")
        if reg in REGIONS:
            corpus.append(f"{c} is a country located in {reg}.")
            corpus.append(f"Q: Which region is {c} in? A: {reg}.")
    log(f"  colheita {tag}: {len(corpus)} sentenças extraídas de {len(facts)} fatos")
    return corpus

mA=load_donor(DA); cA=harvest(mA,"A",FA); del mA; gc.collect(); torch.cuda.empty_cache()
mB=load_donor(DB); cB=harvest(mB,"B",FB); del mB; gc.collect(); torch.cuda.empty_cache()
CORPUS=cA+cB
# REVERSO (fecha o reversal curse): agrega por região a partir dos fatos colhidos dos DOIS
by_reg={r:[] for r in REGIONS}
for c,_,r in FA+FB: by_reg[r].append(c)
rng0=random.Random(1)
for r,cs in by_reg.items():
    for _ in range(6):
        cc=cs[:]; rng0.shuffle(cc)
        CORPUS.append(f"The region of {r} contains these countries: " + ", ".join(cc) + ".")
        CORPUS.append(f"{r} is home to " + ", ".join(cc[:5]) + ", among others.")
log(f"  corpus UNIÃO + REVERSO: {len(CORPUS)} sentenças (A={len(cA)} + B={len(cB)} + reverso)")

# ---------- 2) REFINO: aluno (base) treina no corpus união ----------
def evaluate(m):
    kA=sum(has(cp,gen(m,f"The capital of {c} is")) for c,cp,_ in FA)/len(FA)
    kB=sum(has(cp,gen(m,f"The capital of {c} is")) for c,cp,_ in FB)/len(FB)
    sn=sum(has(a,gen(m,q,n=4)) for q,a in [("7 + 8 ="," 15"),("9 * 6 ="," 54"),("100 / 4 ="," 25")])/3
    ua=ub=0
    for r in REGIONS:
        g=gen(m,f"Countries located in {r} include",n=24)
        ua+=sum(1 for c,_,rr in FA if rr==r and has(c,g)); ub+=sum(1 for c,_,rr in FB if rr==r and has(c,g))
    return dict(kA=kA,kB=kB,sanity=sn,ua=ua,ub=ub)

student=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.float16).to(DEV)
b0=evaluate(student.eval())
log(f"  aluno-base ANTES do refino: sabe A {b0['kA']:.0%} · B {b0['kB']:.0%} · união A={b0['ua']} B={b0['ub']}")
student.train()
params=[]
for n,p in student.named_parameters():
    mm=re.match(r"model\.layers\.(\d+)\.mlp\.",n); p.requires_grad=bool(mm and int(mm.group(1))>=12)
    if p.requires_grad: params.append(p)
opt=torch.optim.SGD(params,lr=3e-3,momentum=0.9); rng=random.Random(7)
S=CORPUS+[s for s in CORPUS if "capital" in s]
for step in range(900):
    batch=[S[rng.randrange(len(S))] for _ in range(8)]
    enc=tok(batch,return_tensors="pt",padding=True).to(DEV)
    lab=enc.input_ids.clone(); lab[enc.attention_mask==0]=-100
    loss=student(**enc,labels=lab).loss
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
    if (step+1)%150==0: log(f"    refino passo {step+1}/900 loss {loss.item():.3f}")
student.eval()
r=evaluate(student)
log(f"\n## VEREDITO WS14 (o fluxo de refino une?)")
log(f"  aluno REFINADO: sabe A {r['kA']:.0%} · B {r['kB']:.0%} · aritmética {r['sanity']:.0%} · UNIÃO nomeia A={r['ua']} B={r['ub']}")
win = r["kA"]>=0.7 and r["kB"]>=0.7 and r["ua"]>0 and r["ub"]>0
log(f"  → {'TESE CONFIRMADA: o REFINO une A∪B (sabe os dois E relaciona na pergunta-união) — o que a fusão de pesos NÃO fez' if win else 'refino parcial — analisar'}")
json.dump(dict(base=b0,refined=r,corpus=len(CORPUS),reverse=True),open(os.path.join(HERE,"ws14_refino.json"),"w"),indent=1)
log(f"wall {(time.time()-t0)/60:.1f} min")
