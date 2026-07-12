#!/usr/bin/env python3
"""WS1 — ÓRGÃO BYTE (typo): a tese-mãe aplicada ao pior gap do produto.

HIPÓTESE: o gerador BPE despenca com typo (82%→45%) porque o tokenizador estilhaça a
palavra. Um órgão de ENTRADA byte-a-byte (Levenshtein contra o léxico de entidades que a
memória do cérebro já tem) restaura a palavra ANTES do tokenizador → recupera a queda.

DESENHO HONESTO:
  - O órgão NÃO sabe qual palavra é a entidade: varre toda palavra do prompt; se não é
    palavra comum e existe entidade a distância ≤2 (e < 1/3 do tamanho), corrige.
  - CONTROLE 1: órgão em prompt LIMPO não pode estragar nada (falsa-correção = 0?).
  - CONTROLE 2: oráculo = prompt limpo (teto).
  - 4 tipos de typo (troca adjacente / deleção / vizinho de teclado / duplicação) × 3 seeds.
Métricas: acurácia crua vs +órgão vs limpo; taxa de restauração exata; falsas correções.
"""
import torch, os, sys, time, random, re
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "../../llm-lab/models/Qwen2.5-1.5B-Instruct")
DEV = "cuda"
JOUR = os.path.join(HERE, "PROGRAMA_JOURNAL.md")
def log(s):
    print(s, flush=True)
    with open(JOUR, "a") as f: f.write(s + "\n")

CAPITAL = {
 "Afghanistan":"Kabul","Albania":"Tirana","Algeria":"Algiers","Argentina":"Buenos Aires","Australia":"Canberra",
 "Austria":"Vienna","Bangladesh":"Dhaka","Belgium":"Brussels","Bolivia":"Sucre","Brazil":"Brasilia",
 "Bulgaria":"Sofia","Cambodia":"Phnom Penh","Cameroon":"Yaounde","Canada":"Ottawa","Chile":"Santiago",
 "China":"Beijing","Colombia":"Bogota","Croatia":"Zagreb","Cuba":"Havana","Czechia":"Prague",
 "Denmark":"Copenhagen","Ecuador":"Quito","Egypt":"Cairo","Ethiopia":"Addis Ababa","Finland":"Helsinki",
 "France":"Paris","Germany":"Berlin","Ghana":"Accra","Greece":"Athens","Hungary":"Budapest",
 "Iceland":"Reykjavik","India":"New Delhi","Indonesia":"Jakarta","Iran":"Tehran","Iraq":"Baghdad",
 "Ireland":"Dublin","Israel":"Jerusalem","Italy":"Rome","Japan":"Tokyo","Jordan":"Amman",
}
LEXICON = list(CAPITAL.keys())          # o léxico vem da MEMÓRIA do cérebro (entidades que ele conhece)
COMMON = set("the of is in a an and to what which capital city country currency continent located".split())

def lev(a, b, cap=3):                    # Levenshtein com corte (barato)
    if abs(len(a)-len(b)) > cap: return cap+1
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[-1]+1, prev[j-1]+(ca != cb)))
        if min(cur) > cap: return cap+1
        prev = cur
    return prev[-1]

def byte_fix(prompt):
    """órgão de entrada: restaura palavras estilhaçadas contra o léxico de entidades."""
    out=[]
    for w in prompt.split(" "):
        core = re.sub(r"[^A-Za-z]","",w)
        if len(core) >= 4 and core.lower() not in COMMON:
            best, bd = None, 99
            for e in LEXICON:
                d = lev(core.lower(), e.lower())
                if d < bd: bd, best = d, e
            if best and 0 < bd <= 2 and bd <= max(1, len(core)//3) and core != best:
                w = w.replace(core, best)
        out.append(w)
    return " ".join(out)

KEY = {"a":"s","b":"v","c":"x","d":"s","e":"r","f":"g","g":"h","h":"j","i":"o","j":"k","k":"l","l":"k",
       "m":"n","n":"m","o":"p","p":"o","q":"w","r":"t","s":"d","t":"y","u":"i","v":"b","w":"e","x":"z","y":"t","z":"x"}
def make_typo(word, kind, rng):
    i = rng.randrange(1, max(2, len(word)-1))
    if kind=="swap" and len(word)>3:  return word[:i]+word[i+1]+word[i]+word[i+2:]
    if kind=="del":                   return word[:i]+word[i+1:]
    if kind=="key":                   return word[:i]+KEY.get(word[i].lower(), word[i])+word[i+1:]
    if kind=="dup":                   return word[:i]+word[i]+word[i:]
    return word

def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def match(gold, gen):
    g=norm(gold); return g in norm(gen) or norm(gen).startswith(g[:max(3,len(g)//2)])

log(f"\n{'='*72}\n# WS1 — ÓRGÃO BYTE (typo) — {time.strftime('%Y-%m-%d %H:%M')}\n{'='*72}")
t0=time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
cache={}
@torch.no_grad()
def generate(prompt, n=8):
    if prompt in cache: return cache[prompt]
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    out=[]
    cur=ids
    for _ in range(n):
        nt=int(model(cur).logits[0,-1].argmax())
        d=tok.decode([nt])
        if nt==tok.eos_token_id or "\n" in d: break
        out.append(nt); cur=torch.cat([cur, torch.tensor([[nt]],device=DEV)],1)
    r=tok.decode(out).strip(); cache[prompt]=r; return r

# CONTROLE 1: órgão em prompts limpos — não pode estragar
false_corr=0
for c in CAPITAL:
    p=f"The capital of {c} is"
    if byte_fix(p)!=p: false_corr+=1
log(f"CONTROLE — falsas correções em {len(CAPITAL)} prompts limpos: {false_corr} {'✓' if false_corr==0 else '⚠ INVESTIGAR'}")

# baseline limpo (oráculo)
clean_ok=sum(match(cap, generate(f"The capital of {c} is")) for c,cap in CAPITAL.items())
log(f"ORÁCULO (limpo): {clean_ok}/{len(CAPITAL)} = {clean_ok/len(CAPITAL):.0%}")

KINDS=["swap","del","key","dup"]
res={}
restored=total_typo=0
for kind in KINDS:
    raw_ok=fix_ok=0; n=0
    for seed in (1,2,3):
        rng=random.Random(seed)
        for c,cap in CAPITAL.items():
            t=make_typo(c, kind, rng)
            if t==c: continue
            n+=1; total_typo+=1
            p_raw=f"The capital of {t} is"
            p_fix=byte_fix(p_raw)
            if c in p_fix: restored+=1
            raw_ok+=match(cap, generate(p_raw))
            fix_ok+=match(cap, generate(p_fix))
    res[kind]=(raw_ok/n, fix_ok/n, n)
log(f"restauração exata da palavra: {restored}/{total_typo} = {restored/total_typo:.0%}\n")
log(f"{'typo':>6} {'cru':>6} {'+órgão':>8} {'n':>5}")
for k,(r,f,n) in res.items(): log(f"{k:>6} {r:>6.0%} {f:>8.0%} {n:>5}")
raw_all=sum(r*n for r,f,n in res.values())/sum(n for _,_,n in res.values())
fix_all=sum(f*n for r,f,n in res.values())/sum(n for _,_,n in res.values())
log(f"{'MÉDIA':>6} {raw_all:>6.0%} {fix_all:>8.0%}")
ok = fix_all>=0.75
log(f"\nVEREDITO WS1: cru {raw_all:.0%} → +órgão-byte {fix_all:.0%} (oráculo {clean_ok/len(CAPITAL):.0%}) — "
    f"{'HIPÓTESE CONFIRMADA (≥75%)' if ok else 'abaixo da meta 75% — analisar onde falha'}")
log(f"wall {time.time()-t0:.0f}s · {len(cache)} gerações únicas")
