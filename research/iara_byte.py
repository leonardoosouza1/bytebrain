#!/usr/bin/env python3
"""BASE DO BYTE-ROUTER — a periferia byte da IARA (o papel certo do byte). 2026-07-14.
Byte = reflexo rápido e ROBUSTO A TYPO (onde token/BPE estilhaça) + endereço barato p/ rotear
ao córtex Qwen. Começa a base; evolui depois. Sem GPU, sem rede — é a camada de baixo custo."""
import re,unicodedata
def sa(s): return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def bl(s): return sa(s).lower().encode("utf-8")                # a string como BYTES (o substrato)

# --- similaridade BYTE (trigrama de bytes + edit-distance) — robusta a erro de digitação ---
def trig(b): return {b[i:i+3] for i in range(max(1,len(b)-2))}
def jac(a,b):
    A,B=trig(a),trig(b); return len(A&B)/max(1,len(A|B))
def edit(a,b):                                                 # Levenshtein em BYTES (typo = 1-2 bytes trocados)
    m,n=len(a),len(b); d=list(range(n+1))
    for i in range(1,m+1):
        prev=d[0]; d[0]=i
        for j in range(1,n+1):
            cur=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=cur
    return d[n]
def byte_sim(q,c):                                            # 0..1: quão parecidos em byte-space
    bq,bc=bl(q),bl(c); e=edit(bq,bc); return 0.6*jac(bq,bc)+0.4*(1-e/max(len(bq),len(bc),1))

def byte_match(query,conceitos,limiar=0.45):                  # REFLEXO: acha o conceito mesmo com typo
    best=max(conceitos,key=lambda c:byte_sim(query,c)); sc=byte_sim(query,best)
    return (best,round(sc,2)) if sc>=limiar else (None,round(sc,2))
def token_match(query,conceitos):                            # baseline "token": só bate se as PALAVRAS batem exato
    qw=set(sa(query).lower().split())
    for c in conceitos:
        if qw & set(sa(c).lower().split()): return c
    return None

# --- endereço barato por bits (CRT) p/ roteamento O(1) ---
def endereco(c):
    h=0
    for b in bl(c): h=(h*131+b)%9999991
    return h%105,(h%3,h%5,h%7)

CONCEITOS=["coração","átrio direito","átrio esquerdo","ventrículo direito","ventrículo esquerdo",
 "valva mitral","valva tricúspide","valva aórtica","valva pulmonar","aorta","artéria pulmonar",
 "veia cava","nó sinoatrial","miocárdio","pericárdio","sístole","diástole"]

print("="*72); print("# BASE DO BYTE-ROUTER — reflexo robusto a typo + endereço barato"); print("="*72)

print("\n### 1) ROBUSTEZ A TYPO — byte acha, token estilhaça (a vantagem real do byte)")
TESTES=["coraçao","vávula mitral","atrio direto","arteria pulmnar","ventrculo esquerdo",
        "no sinoatral","miocardio","sistoli","valva aortca","veia kava"]
bo=to=0
for q in TESTES:
    bm,sc=byte_match(q,CONCEITOS); tm=token_match(q,CONCEITOS)
    bo+= bm is not None; to+= tm is not None
    print(f"  '{q:18}' → byte: {str(bm):20}({sc})   token: {tm}")
print(f"\n  ACERTO — byte {bo}/{len(TESTES)}  vs  token {to}/{len(TESTES)}  (byte é robusto a erro de digitação)")

print("\n### 2) ENDEREÇO BARATO O(1) — cada conceito roteável sem rede")
for c in CONCEITOS[:6]:
    h,cr=endereco(c); print(f"  {c:20} → endereço {h:3d}  coord {cr}")

print("\n### 3) O ROTEADOR: query → (byte-reflexo decide) → conceito OU manda pro córtex Qwen")
def rotear(query):
    bm,sc=byte_match(query,CONCEITOS)
    if bm and sc>0.7:  return f"REFLEXO byte (rápido, sc={sc}) → '{bm}'"
    if bm:             return f"byte sugere '{bm}' (sc={sc}) → confirma no CÓRTEX Qwen"
    return "sem match byte → CÓRTEX Qwen (raciocínio)"
for q in ["coraçao","o que bombeia sangue pros pulmões?","vávula mitral","quem foi Einstein?"]:
    print(f"  '{q}'\n     → {rotear(q)}")
print("\n  → byte resolve o barato (typo, conceito conhecido) em µs; só o que sobra vai pro Qwen.")
