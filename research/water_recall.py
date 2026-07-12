#!/usr/bin/env python3
"""NEURÔNIO-ÁGUA: recall por espalhamento (ideia do Leonardo, 2026-07-10).

Cada nó = célula de memória endereçável por conteúdo (guarda bytes; "dispara" e DESPEJA
o conteúdo). A pergunta entra como uma GOTA e se distribui pelo grafo (água = random walk
/ difusão, a física do turno anterior). Os nós ligados a AMBOS os termos da pergunta
(Brasil E capital) acumulam água e disparam → a resposta (Brasília) é onde a água POÇA.

Testa 4 dinâmicas de propagação sobre o MESMO grafo de conhecimento real (25 países):
  A) intersecção 1-hop      — baseline trivial (conta quantos termos da query tocam o nó)
  B) Personalized PageRank  — a ÁGUA em regime estacionário (injeção contínua na fonte)
  C) difusão de calor       — exp(-t·L)·s, o Laplaciano do grafo (fluido)
  D) spiking (modelo literal do Leonardo) — acumula potencial, dispara binário, despeja

Métrica honesta: top-1 entre TODOS os nós não-consultados (sem filtrar por tipo — difícil).
Regimes: 1-hop (busca trivial já resolve) vs MULTI-HOP (só a água que integra caminhos
resolve) vs RUÍDO (robustez). numpy puro, determinístico. Baseline = acaso e a busca trivial.
"""
import numpy as np

# ---------- base de conhecimento REAL ----------
# país: (capital, língua, continente, moeda)
KB = {
 "Brasil":("Brasilia","Portugues","AmSul","Real"),
 "Franca":("Paris","Frances","Europa","Euro"),
 "Japao":("Toquio","Japones","Asia","Iene"),
 "Alemanha":("Berlim","Alemao","Europa","Euro"),
 "Italia":("Roma","Italiano","Europa","Euro"),
 "Espanha":("Madri","Espanhol","Europa","Euro"),
 "Portugal":("Lisboa","Portugues","Europa","Euro"),
 "Argentina":("BuenosAires","Espanhol","AmSul","Peso"),
 "China":("Pequim","Chines","Asia","Yuan"),
 "India":("NovaDeli","Hindi","Asia","Rupia"),
 "Russia":("Moscou","Russo","Europa","Rublo"),
 "Egito":("Cairo","Arabe","Africa","LibraEg"),
 "Mexico":("CidadeMexico","Espanhol","AmNorte","Peso"),
 "Canada":("Otawa","Ingles","AmNorte","DolarCan"),
 "EUA":("WashingtonDC","Ingles","AmNorte","Dolar"),
 "ReinoUnido":("Londres","Ingles","Europa","Libra"),
 "Australia":("Camberra","Ingles","Oceania","DolarAus"),
 "Nigeria":("Abuja","Ingles","Africa","Naira"),
 "Quenia":("Nairobi","Suaili","Africa","Xelim"),
 "Peru":("Lima","Espanhol","AmSul","Sol"),
 "Chile":("Santiago","Espanhol","AmSul","PesoCh"),
 "Grecia":("Atenas","Grego","Europa","Euro"),
 "Turquia":("Ancara","Turco","Asia","Lira"),
 "Tailandia":("Bangkok","Tailandes","Asia","Baht"),
 "CoreiaSul":("Seul","Coreano","Asia","Won"),
}
RELS = ["capital","lingua","continente","moeda"]

# ---------- grafo: país<->valor e valor<->tipo-de-relação ----------
def build_graph():
    nodes = []
    def add(n):
        if n not in nodes: nodes.append(n)
    for p in KB: add(p)
    for r in RELS: add("§"+r)               # nó de TIPO de relação (hub compartilhado)
    for p,(c,l,co,m) in KB.items():
        for v in (c,l,co,m): add(v)         # nós de valor
    idx = {n:i for i,n in enumerate(nodes)}
    N = len(nodes); A = np.zeros((N,N))
    def link(a,b): A[idx[a],idx[b]]=1; A[idx[b],idx[a]]=1
    for p,vals in KB.items():
        for r,v in zip(RELS,vals):
            link(p,v)                        # país — valor
            link("§"+r,v)                    # tipo-de-relação — valor
    return nodes, idx, A

NODES, IDX, A = build_graph()
N = len(NODES); DEG = A.sum(1)
VALUE_NODES = set()                          # nós que são resposta possível (valores)
for p,(c,l,co,m) in KB.items():
    VALUE_NODES.update([c,l,co,m])

# ---------- dinâmicas ----------
def src_vec(terms):
    s = np.zeros(N)
    for t in terms: s[IDX[t]] = 1.0
    return s

def dyn_intersect(terms):                    # A) baseline trivial: soma das adjacências
    s = src_vec(terms); return A @ s

def dyn_ppr(terms, d=0.85, it=200):          # B) água estacionária (Personalized PageRank)
    p = src_vec(terms); p /= p.sum()
    M = A / np.maximum(DEG,1e-9)             # coluna-estocástica (M[:,j]=A[:,j]/deg[j])
    r = p.copy()
    for _ in range(it): r = (1-d)*p + d*(M @ r)
    return r

def dyn_heat(terms, t=3.0, K=60):            # C) difusão de calor exp(-tL)s
    s = src_vec(terms); L = np.diag(DEG) - A
    h = s.copy(); step = t/K
    for _ in range(K): h = h - step*(L @ h)
    return np.maximum(h,0)

def dyn_spike(terms, rounds=40, theta=1.0, rng=None):  # D) modelo literal: acumula, dispara, despeja
    v = np.zeros(N); fires = np.zeros(N); si = [IDX[t] for t in terms]
    for _ in range(rounds):
        v[si] += 1.0                         # a pergunta "continua pingando" na fonte
        spk = (v >= theta).astype(float)     # DISPARA (binário) quem passou do limiar
        fires += spk
        inflow = A @ (spk / np.maximum(DEG,1e-9))  # despeja p/ vizinhos, dividido pelo grau
        v = v - spk*theta + inflow           # reset de quem disparou + entrada dos vizinhos
    return fires

def dyn_ppr_dc(terms):                        # B') água / grau: concentração, não quantidade (mata hub)
    return dyn_ppr(terms) / np.maximum(DEG,1e-9)
def dyn_heat_dc(terms):                       # C') calor / grau (mata hub)
    return dyn_heat(terms) / np.maximum(DEG,1e-9)

DYN = {"intersect":dyn_intersect, "ppr":dyn_ppr, "ppr/grau":dyn_ppr_dc,
       "heat":dyn_heat, "heat/grau":dyn_heat_dc, "spike":dyn_spike}

def answer(scores, terms, restrict_type=None):
    ban = set(terms)
    best, bs = None, -1
    for i,n in enumerate(NODES):
        if n in ban or n not in VALUE_NODES: continue
        if restrict_type and n not in restrict_type: continue
        if scores[i] > bs: bs, best = scores[i], n
    return best

# ---------- baterias de query ----------
# unicidade: valores que pertencem a UM só país (p/ multi-hop limpo)
val_owner = {}
for p,(c,l,co,m) in KB.items():
    for v in (c,l,co,m): val_owner.setdefault(v,[]).append(p)
UNIQUE = {v:o[0] for v,o in val_owner.items() if len(o)==1}

def q_basic():          # "capital do Brasil" -> {Brasil, §capital} -> Brasilia   (1-hop)
    qs=[]
    for p,vals in KB.items():
        for r,v in zip(RELS,vals):
            qs.append((f"{r} de {p}", [p,"§"+r], v))
    return qs

def q_multihop():       # "capital do país cuja MOEDA é Real" -> {Real, §capital} -> Brasilia (2-hop)
    qs=[]
    for p,(c,l,co,m) in KB.items():
        pivots=[("moeda",m),("lingua",l),("continente",co)]
        for pr,pv in pivots:
            if pv not in UNIQUE: continue    # pivô tem que identificar 1 país só
            qs.append((f"capital do pais de {pr}={pv}", [pv,"§capital"], c))
    return qs

def q_noisy(seed):      # basic + 1 termo distrator aleatório (robustez)
    r = np.random.default_rng(seed); qs=[]
    cand = [n for n in NODES if n in VALUE_NODES]
    for p,vals in KB.items():
        v = vals[0]; noise = cand[r.integers(len(cand))]
        qs.append((f"capital de {p} (+ruido {noise})", [p,"§capital",noise], v))
    return qs

def run_battery(name, qs, restrict=False):
    print(f"\n=== {name}  ({len(qs)} queries) ===")
    print(f"{'dinâmica':>12} {'top1':>7} {'top1(tipo)':>11}   exemplo (para onde a água foi)")
    for key,fn in DYN.items():
        ok=okt=0; ex=""
        for i,(desc,terms,gold) in enumerate(qs):
            sc = fn(terms)
            a = answer(sc, terms)
            # tipo correto = o conjunto de valores da MESMA relação do gold
            rtype = {v for p,vals in KB.items() for r,v in zip(RELS,vals) if v==gold for v2 in vals if False} # noop
            typeset = None
            for r_i,r in enumerate(RELS):
                if gold in [vals[r_i] for vals in KB.values()]: typeset = {vals[r_i] for vals in KB.values()}
            at = answer(sc, terms, restrict_type=typeset)
            ok += (a==gold); okt += (at==gold)
            if i==0:
                order = [NODES[j] for j in np.argsort(-sc) if NODES[j] not in terms][:4]
                ex = f"{desc[:26]:<26} → {order}"
        print(f"{key:>12} {ok/len(qs):>6.0%} {okt/len(qs):>10.0%}   {ex}")

if __name__ == "__main__":
    print("NEURÔNIO-ÁGUA — recall por espalhamento no grafo de conhecimento")
    print(f"grafo: {N} nós ({len(KB)} países, {len(RELS)} relações, {len(VALUE_NODES)} valores), "
          f"acaso≈{1/len(VALUE_NODES):.1%}")
    run_battery("BÁSICO (1-hop: busca trivial já resolve)", q_basic())
    run_battery("MULTI-HOP (2-hop: pivô único → país → alvo)", q_multihop())
    # ruído: média 3 seeds
    print("\n=== RUÍDO (basic + 1 distrator, média 3 seeds) ===")
    print(f"{'dinâmica':>12} {'top1':>7} {'top1(tipo)':>11}")
    for key,fn in DYN.items():
        oks=[]; okts=[]
        for seed in (1,2,3):
            qs=q_noisy(seed); ok=okt=0
            for desc,terms,gold in qs:
                sc=fn(terms); typeset={vals[0] for vals in KB.values()}
                ok+=(answer(sc,terms)==gold); okt+=(answer(sc,terms,restrict_type=typeset)==gold)
            oks.append(ok/len(qs)); okts.append(okt/len(qs))
        print(f"{key:>12} {np.mean(oks):>6.0%} {np.mean(okts):>10.0%}")

    # a "foto" do Leonardo: para onde a água vai quando pingo "capital do Brasil"
    print("\n=== PARA ONDE A ÁGUA VAI — pergunta: 'capital do Brasil' (termos: Brasil, §capital) ===")
    terms=["Brasil","§capital"]
    for key,fn in DYN.items():
        sc=fn(terms); m=sc.max() or 1
        top=[(NODES[j], sc[j]/m) for j in np.argsort(-sc) if NODES[j] not in terms][:6]
        print(f"  {key:>10}: " + "  ".join(f"{n}={w:.2f}" for n,w in top))
    print("\n(top-1 = o nó certo é o mais 'cheio' d'água entre TODOS os valores; "
          "top1(tipo) = restrito ao tipo da relação.)")
