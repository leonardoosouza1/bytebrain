#!/usr/bin/env python3
"""MEMÓRIA-ÁGUA integrada (ideia do Leonardo, 2026-07-10) — os 3 fios num sistema só.

FIO 1 — grafo APRENDIDO: as associações EMERGEM de um corpus (co-ocorrência + PMI),
        nada hand-coded. A água (PageRank ÷ grau) faz o recall sobre o grafo aprendido.
FIO 2 — AMBIGUIDADE: pivô ambíguo ("língua = português" → Brasil OU Portugal) → a água
        se DIVIDE entre os dois (resposta ambígua = comportamento correto, mensurável).
FIO 3 — RELEASE → GERAÇÃO: fatos divididos em VISTOS (o byte-LM treina) e HELD-OUT
        (o LM nunca viu). A memória-água guarda TODOS. Mede se o release do nó recuperado,
        injetado como contexto, faz o LM emitir a resposta de um fato que ele NÃO treinou
        (RAG lift em bits) — vs sem-contexto e vs contexto ERRADO (controle).

numpy puro, determinístico, honesto. Baseline = acaso, busca trivial, contexto errado.
"""
import numpy as np, re

# ---------- base de conhecimento (só pra GERAR o corpus; o grafo vem do TEXTO) ----------
KB = {
 "brasil":("brasilia","portugues","amsul","real"), "franca":("paris","frances","europa","euro"),
 "japao":("toquio","japones","asia","iene"), "alemanha":("berlim","alemao","europa","euro"),
 "italia":("roma","italiano","europa","euro"), "espanha":("madri","espanhol","europa","euro"),
 "portugal":("lisboa","portugues","europa","euro"), "argentina":("buenosaires","espanhol","amsul","peso"),
 "china":("pequim","chines","asia","yuan"), "india":("novadeli","hindi","asia","rupia"),
 "russia":("moscou","russo","europa","rublo"), "egito":("cairo","arabe","africa","libraeg"),
 "mexico":("cidademexico","espanhol","amnorte","peso"), "canada":("otawa","ingles","amnorte","dolarcan"),
 "eua":("washington","ingles","amnorte","dolar"), "reinounido":("londres","ingles","europa","libra"),
 "australia":("camberra","ingles","oceania","dolaraus"), "nigeria":("abuja","ingles","africa","naira"),
 "quenia":("nairobi","suaili","africa","xelim"), "peru":("lima","espanhol","amsul","sol"),
 "chile":("santiago","espanhol","amsul","pesoch"), "grecia":("atenas","grego","europa","euro"),
 "turquia":("ancara","turco","asia","lira"), "tailandia":("bangkok","tailandes","asia","baht"),
 "coreiasul":("seul","coreano","asia","won"),
}
RELS = ["capital","lingua","continente","moeda"]
VALUES = {v for vals in KB.values() for v in vals}
val_owner = {}
for p,vals in KB.items():
    for v in vals: val_owner.setdefault(v,[]).append(p)
UNIQUE = {v:o[0] for v,o in val_owner.items() if len(o)==1}

# ---------- corpus: frases de fato (várias formas → co-ocorrência tem sinal) ----------
TPL = {
 "capital":["a capital do {p} e {v}", "{v} e a capital do {p}", "o {p} tem capital {v}"],
 "lingua":["a lingua do {p} e o {v}", "no {p} se fala {v}", "o idioma do {p} e {v}"],
 "continente":["o {p} fica na {v}", "o {p} esta na {v}", "{p} pertence a {v}"],
 "moeda":["a moeda do {p} e o {v}", "no {p} se paga em {v}", "o {p} usa {v}"],
}
def facts_of(kb):
    F=[]
    for p,vals in kb.items():
        for r,v in zip(RELS,vals): F.append((p,r,v))
    return F
def corpus_from(facts):
    lines=[]
    for p,r,v in facts:
        for t in TPL[r]: lines.append(t.format(p=p,v=v))
    return lines

# ---------- FIO 1: grafo APRENDIDO por co-ocorrência + PMI ----------
def learn_graph(lines):
    toks=[re.findall(r"[a-z]+",l) for l in lines]
    vocab=sorted({w for s in toks for w in s})
    idx={w:i for i,w in enumerate(vocab)}; V=len(vocab)
    co=np.zeros((V,V)); uni=np.zeros(V); tot=0.0
    for s in toks:
        u=set(s)
        for w in u: uni[idx[w]]+=1
        for a in u:
            for b in u:
                if a<b: co[idx[a],idx[b]]+=1; co[idx[b],idx[a]]+=1; tot+=1
    # PMI positivo → peso da aresta (as associações fortes/específicas sobrevivem)
    A=np.zeros((V,V)); N=len(lines)
    for i in range(V):
        for j in range(V):
            if co[i,j]>0:
                pmi=np.log((co[i,j]*N)/(uni[i]*uni[j]+1e-9)+1e-9)
                if pmi>0: A[i,j]=pmi
    return vocab, idx, A

# ---------- água (PPR ÷ grau) sobre grafo PONDERADO ----------
def ppr_dc(A, idx, terms, d=0.85, it=200):
    V=A.shape[0]; deg=A.sum(1); p=np.zeros(V)
    src=[idx[t] for t in terms if t in idx]
    if not src: return np.zeros(V)
    for s in src: p[s]=1.0
    p/=p.sum()
    M=A/np.maximum(A.sum(0),1e-9)          # coluna-estocástica ponderada
    r=p.copy()
    for _ in range(it): r=(1-d)*p+d*(M@r)
    return r/np.maximum(deg,1e-9)          # concentração, não quantidade (mata hub)

def top_values(scores, vocab, terms, k=6):
    ban=set(terms)
    cand=[(vocab[i],scores[i]) for i in np.argsort(-scores) if vocab[i] in VALUES and vocab[i] not in ban]
    return cand[:k]

# ---------- byte-LM minúsculo (pro FIO 3) ----------
C,D,H,Vb,BLR = 24,24,160,256,0.5
def lm_init(seed=0):
    r=np.random.default_rng(seed)
    return dict(E=r.normal(0,.05,(Vb,D)),W1=r.normal(0,.05,(H,C*D)),b1=np.zeros(H),
                W2=r.normal(0,.05,(Vb,H)),b2=np.zeros(Vb))
def lm_fwd(m,X):
    f=m['E'][X].reshape(len(X),-1); h=np.tanh(f@m['W1'].T+m['b1'])
    z=h@m['W2'].T+m['b2']; z-=z.max(1,keepdims=True); ez=np.exp(z); return f,h,ez/ez.sum(1,keepdims=True)
def lm_train(text, steps=4000, seed=0):
    a=np.frombuffer(("\n".join(text)+"\n").encode(),dtype=np.uint8).astype(np.int64)
    m=lm_init(seed); r=np.random.default_rng(1); B=128
    for _ in range(steps):
        i=r.integers(C,len(a)-1,size=B); X=np.stack([a[j-C:j] for j in i]); Y=a[i]
        f,h,p=lm_fwd(m,X); dz=p.copy(); dz[np.arange(B),Y]-=1; dz/=B
        gW2=dz.T@h; gb2=dz.sum(0); dh=(dz@m['W2'])*(1-h*h); gW1=dh.T@f; gb1=dh.sum(0)
        df=(dh@m['W1']).reshape(B,C,D); m['W2']-=BLR*gW2; m['b2']-=BLR*gb2; m['W1']-=BLR*gW1; m['b1']-=BLR*gb1
        xf=X.reshape(-1); oh=np.zeros((xf.size,Vb)); oh[np.arange(xf.size),xf]=1.0
        m['E']-=BLR*(oh.T@df.reshape(-1,D))
    return m
def ans_bpb(m, ctx, ans):
    """bits/byte que o LM gasta pra emitir os bytes de `ans` logo após `ctx`."""
    buf=list((" "*C+ctx).encode()); tot=0.0
    for by in ans.encode():
        X=np.array(buf[-C:])[None]; _,_,p=lm_fwd(m,X)
        tot+=-np.log2(p[0,by]+1e-12); buf.append(by)
    return tot/len(ans.encode())

# ============================================================
def main():
    print("MEMÓRIA-ÁGUA — grafo aprendido + água + ambiguidade + release→geração\n")

    # ---- FIO 1 ----
    allF=facts_of(KB); lines=corpus_from(allF)
    vocab,idx,A=learn_graph(lines)
    print(f"FIO 1 — GRAFO APRENDIDO de {len(lines)} frases: {len(vocab)} nós, "
          f"{int((A>0).sum()//2)} arestas (PMI>0). acaso≈{1/len(VALUES):.1%}")
    def acc(queries):
        ok=okt=0
        for terms,gold,rel in queries:
            sc=ppr_dc(A,idx,terms); tv=top_values(sc,vocab,terms,k=99)
            if tv and tv[0][0]==gold: ok+=1
            typeset={vals[RELS.index(rel)] for vals in KB.values()}
            tvt=[(w,s) for w,s in tv if w in typeset]
            if tvt and tvt[0][0]==gold: okt+=1
        return ok/len(queries), okt/len(queries)
    q1=[([p,r],vals[RELS.index(r)],r) for p,vals in KB.items() for r in RELS]
    q2=[]
    for p,vals in KB.items():
        for pr,pv in [("moeda",vals[3]),("lingua",vals[1])]:
            if pv in UNIQUE: q2.append(([pv,"capital"],vals[0],"capital"))
    a1,a1t=acc(q1); a2,a2t=acc(q2)
    print(f"  1-hop  (busca trivial=100%): água÷grau top1 {a1:.0%}  (tipo {a1t:.0%})")
    print(f"  multi-hop (busca trivial≈3%): água÷grau top1 {a2:.0%}  (tipo {a2t:.0%})")
    sc=ppr_dc(A,idx,["brasil","capital"])
    print("  água p/ 'capital do brasil' →", ", ".join(f"{w}={s/max(sc):.2f}" for w,s in top_values(sc,vocab,["brasil","capital"],5)))

    # ---- FIO 2: ambiguidade ----
    print("\nFIO 2 — AMBIGUIDADE: 'capital do país cuja língua é português' (Brasil E Portugal)")
    sc=ppr_dc(A,idx,["portugues","capital"])
    tv=top_values(sc,vocab,["portugues","capital"],6)
    print("  água →", ", ".join(f"{w}={s/max(sc):.2f}" for w,s in tv))
    b=dict(tv).get("brasilia",0); l=dict(tv).get("lisboa",0)
    print(f"  Brasília={b/max(sc):.2f} vs Lisboa={l/max(sc):.2f} — a água {'SE DIVIDE (ambíguo, correto)' if min(b,l)>0.4*max(b,l) and b>0 and l>0 else 'colapsou num só'}")
    sc2=ppr_dc(A,idx,["real","capital"])   # pivô ÚNICO (moeda real) resolve
    tv2=top_values(sc2,vocab,["real","capital"],3)
    print(f"  controle pivô ÚNICO (moeda=real) → {tv2[0][0]} (deve ser brasilia, sem ambiguidade)")

    # ---- FIO 3: release → geração, held-out ----
    print("\nFIO 3 — RELEASE→GERAÇÃO: LM treina em fatos VISTOS; memória guarda TODOS;")
    print("        mede se o release recuperado faz o LM responder um fato HELD-OUT (nunca treinado).")
    rng=np.random.default_rng(0); fi=list(range(len(allF))); rng.shuffle(fi)
    held=set(fi[:20]); seen=[allF[i] for i in range(len(allF)) if i not in held]
    heldF=[allF[i] for i in held]
    # formato que ENSINA copiar do contexto: "<valor> . a capital do brasil e <valor>"
    def fmt(p,r,v): return f"{v} . {TPL[r][0].format(p=p,v=v)}"
    m=lm_train([fmt(p,r,v) for p,r,v in seen], steps=4000)
    # avalia nos held-out: contexto = release (valor recuperado pela água) + a pergunta-prefixo
    def prefix(p,r): return TPL[r][0].format(p=p,v="").rsplit(" ",1)[0]+" "  # "a capital do brasil e "
    no_ctx=wat=orc=wrg=0.0; hit=0
    for p,r,v in heldF:
        terms=[p,r]; sc=ppr_dc(A,idx,terms); tv=top_values(sc,vocab,terms,1)
        retr=tv[0][0] if tv else "?"; hit+=(retr==v)
        pre=prefix(p,r)
        wrongv=next(iter(VALUES-{v}))
        no_ctx+=ans_bpb(m, pre, v)
        wat  +=ans_bpb(m, f"{retr} . {pre}", v)   # release da ÁGUA
        orc  +=ans_bpb(m, f"{v} . {pre}", v)        # oráculo (fato certo)
        wrg  +=ans_bpb(m, f"{wrongv} . {pre}", v)   # controle: fato ERRADO
    n=len(heldF)
    print(f"  recall da água nos held-out: {hit}/{n} corretos")
    print(f"  bpb da resposta (menor=melhor; o LM NUNCA treinou estes fatos):")
    print(f"    sem contexto ......... {no_ctx/n:5.2f}   (o LM sozinho não sabe)")
    print(f"    + release da ÁGUA .... {wat/n:5.2f}   (a memória recuperou e injetou)")
    print(f"    + oráculo (fato certo) {orc/n:5.2f}   (teto)")
    print(f"    + contexto ERRADO .... {wrg/n:5.2f}   (controle: deve ficar RUIM)")
    lift=(no_ctx-wat)/n
    print(f"  → RAG lift da água: {lift:+.2f} bpb  ({'a memória-água ajuda o LM a responder o inédito' if lift>0.3 else 'lift fraco — reportar honesto'})")

if __name__=="__main__":
    main()
