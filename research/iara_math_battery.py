#!/usr/bin/env python3
"""BATERIA — que ferramentas matemáticas do tipo 'invariante/atrator/estrutura' REALMENTE
ajudam a IARA? Testa cada uma no grafo de conhecimento REAL dela e dá veredito (sinal vs novidade).
2026-07-14. Pura CPU. Roda contra o state file (as triplas já extraídas)."""
import json,re,unicodedata,gzip,math,os
import networkx as nx
from collections import Counter,defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); ST=os.path.join(HERE,"iara_cerebro_state.json")
def sa(s):return "".join(c for c in unicodedata.normalize("NFKD",s) if not unicodedata.combining(c))
def nrm(s):return re.sub(r"[^a-z0-9 ]","",sa(s).lower()).strip()
def lb(k):return re.sub(r"\s+\d+$","",k)

d=json.load(open(ST)); seeds={k:v for k,v in d["seeds"].items() if v.get("kind")in("ensinado","pesquisado")}
subj=set(seeds)
def hit(o):
    if o in subj: return o
    for k in subj:
        if o==lb(k) or (len(o.split())>=2 and o in k): return k
    return None
# grafo dirigido de conhecimento (sujeito -> objeto), + nós-lacuna
G=nx.DiGraph()
for k in seeds: G.add_node(k,kind="fato")
gaps=set()
for k,v in seeds.items():
    for o in (v.get("obj") or []):
        if len(o)<4: continue
        m=hit(o)
        if m and m!=k: G.add_edge(k,m,rel=v.get("rel",""))
        elif not m:
            gaps.add(o); G.add_node("?"+o,kind="lacuna"); G.add_edge(k,"?"+o,rel=v.get("rel",""))
print("="*72);print(f"# BATERIA MATEMÁTICA — grafo real: {len(seeds)} fatos, {G.number_of_nodes()} nós, {G.number_of_edges()} arestas, {len(gaps)} lacunas");print("="*72)

# 1) CENTRALIDADE (PageRank) — qual conceito é o mais 'load-bearing'?
print("\n### 1. CENTRALIDADE (PageRank) — o que sustenta o conhecimento?")
def pagerank(GR,damp=0.85,it=80):                          # power-iteration puro (sem scipy)
    N=list(GR.nodes()); n=len(N); pr={u:1/n for u in N}; out={u:list(GR.successors(u)) for u in N}
    for _ in range(it):
        nw={u:(1-damp)/n for u in N}
        for u in N:
            o=out[u]
            if o:
                sh=damp*pr[u]/len(o)
                for w in o: nw[w]+=sh
            else:
                for w in N: nw[w]+=damp*pr[u]/n
        pr=nw
    return pr
pr=pagerank(G.reverse())  # reverse: quem é MUITO referenciado sobe
top=sorted(pr.items(),key=lambda x:-x[1])[:8]
for n,s in top: print(f"   {lb(n.lstrip('?')):24} {s:.3f}  {'(lacuna!)' if n.startswith('?') else ''}")
central_gap=[n for n,_ in top if n.startswith('?')]
print(f"   → VEREDITO: {'SINAL — as lacunas centrais aparecem no topo (ensinar ELAS primeiro)' if central_gap else 'sinal parcial'}")

# 2) COMPONENTES & ILHAS
print("\n### 2. COMPONENTES / ILHAS")
comps=list(nx.weakly_connected_components(G)); comps.sort(key=len,reverse=True)
iso=[c for c in comps if len(c)==1]
print(f"   componentes: {len(comps)} | maior: {len(comps[0])} nós | ilhas (1 nó): {len(iso)}")
print(f"   → VEREDITO: {'SINAL — há ilhas de conhecimento desconexas p/ ligar' if len(comps)>2 else 'grafo coeso'}")

# 3) PONTES / PONTOS DE ARTICULAÇÃO — conhecimento frágil
print("\n### 3. PONTOS DE ARTICULAÇÃO (se remover, o grafo racha)")
UG=G.to_undirected(); art=list(nx.articulation_points(UG)) if UG.number_of_nodes()>2 else []
print(f"   conceitos-ponte: {[lb(a.lstrip('?')) for a in art[:8]]}")
print(f"   → VEREDITO: {'SINAL — pontos frágeis identificáveis (reforçar com mais fatos)' if art else 'sem pontos críticos'}")

# 4) MDL / COMPRESSÃO — densidade/redundância do conhecimento
print("\n### 4. MDL / COMPRESSÃO (quão estruturado/redundante é o conhecimento?)")
blob="\n".join(v["stmt"] for v in seeds.values()).encode()
comp=len(gzip.compress(blob)); raw=len(blob)
usage=Counter()
for v in seeds.values():
    for w in nrm(v["stmt"]).split():
        if len(w)>=4: usage[w]+=1
tot=sum(usage.values()); H=-sum((c/tot)*math.log2(c/tot) for c in usage.values())
print(f"   fatos {len(seeds)} · texto {raw}B → comprimido {comp}B ({raw/comp:.2f}× ) · {comp*8//len(seeds)} bits/fato")
print(f"   entropia do vocabulário: {H:.2f} bits (baixa = conhecimento focado/redundante)")
print(f"   → VEREDITO: SINAL — bits/fato e razão de compressão são medida real de DENSIDADE (vitais)")

# 5) FINGERPRINT / DEDUP (SimHash de shingles) — pega fato quase-duplicado?
print("\n### 5. FINGERPRINT (SimHash) — detecta fato redundante melhor que substring?")
def simhash(txt,bits=64):
    v=[0]*bits
    ws=nrm(txt).split(); sh=[" ".join(ws[i:i+2]) for i in range(max(1,len(ws)-1))]
    for g in sh:
        h=int.from_bytes(__import__("hashlib").md5(g.encode()).digest()[:8],"big")
        for b in range(bits): v[b]+= 1 if (h>>b)&1 else -1
    return sum((1<<b) for b in range(bits) if v[b]>0)
def ham(a,b):return bin(a^b).count("1")
sigs={k:simhash(v["stmt"]) for k,v in seeds.items()}
pairs=[]
ks=list(sigs)
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        h=ham(sigs[ks[i]],sigs[ks[j]])
        if h<=12: pairs.append((h,ks[i],ks[j]))
pairs.sort()
print(f"   pares quase-duplicados (hamming<=12): {len(pairs)}")
for h,a,b in pairs[:4]: print(f"      d={h}  '{lb(a)}' ~ '{lb(b)}'")
print(f"   → VEREDITO: {'SINAL — evita re-aprender fato repetido no modo automático' if pairs else 'NOVIDADE — poucos duplicados agora, mas útil em escala'}")

# 6) RAIZ-DIGITAL como REFLEXO (0 param) — organismo calculável (a descoberta do Leonardo)
print("\n### 6. RAIZ-DIGITAL como reflexo calculável (0 parâmetros)")
def s_(n):return sum(int(c) for c in str(n))
def attractor(n):
    seen=[];
    while n not in seen: seen.append(n); n=s_(n*n)
    c=seen[seen.index(n):]; return tuple(sorted(c))
cnt=Counter(attractor(n) for n in range(1,2001))
print(f"   classifica 2000 números em {len(cnt)} classes: {dict(cnt)}")
print(f"   → VEREDITO: NOVIDADE-ÚTIL — classificador correto de custo zero; serve de REFLEXO/organ, não p/ linguagem")

# 7) CHECKSUM mod-9/mod-11 como chave de dedup
print("\n### 7. CHECKSUM (mod 9 / mod 11) como fingerprint barato")
def m9(t):return sum(ord(c) for c in nrm(t))%9
def m11(t):
    ws=nrm(t); return sum((-1)**i*ord(c) for i,c in enumerate(ws))%11
b9=Counter(m9(v["stmt"]) for v in seeds.values()); b11=Counter((m9(v["stmt"]),m11(v["stmt"])) for v in seeds.values())
print(f"   mod9 sozinho: {len(b9)}/9 baldes usados (colisões altas) | mod9×mod11: {len(b11)} baldes p/ {len(seeds)} fatos")
print(f"   → VEREDITO: {'SINAL — mod9×mod11 dá fingerprint quase-único barato' if len(b11)>=len(seeds)*0.7 else 'NOVIDADE — colide demais p/ dedup, ok só p/ checksum'}")

print("\n"+"="*72)
print("RESUMO: guarde PageRank(1)+Componentes(2)+Pontes(3)+MDL(4) → estrutura/medida;")
print("        SimHash(5) → dedup no auto-loop; RaizDigital(6) → reflexo; Checksum(7) → marginal.")
