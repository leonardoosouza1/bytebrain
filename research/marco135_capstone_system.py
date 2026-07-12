"""M135 — CAPSTONE: o SISTEMA ÚNICO que junta tudo que validamos. Tronco Math-1.5B CONGELADO + floresta
roteada de 4 seeds especializadas: (1) FATO (país→capital inventado), (2) REGRA +3, (3) REGRA ×2,
(4) RACIOCÍNIO (silogismo). Um roteador por centróide de embedding escolhe a seed por query. Testa um
conjunto MISTO de queries INÉDITAS (instâncias nunca vistas de cada tipo) end-to-end e compara com o
baseline (modelo sozinho). Demonstra: 1 modelo congelado vira armazém de conhecimento+skills roteável.
GPU. Dump marco135_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco135_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M135 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M135 START | tronco Math-1.5B congelado")
rng = random.Random(0)

# ---------- treino de cada especialista ----------
# 1) FATO
PARES = [("Zorlândia", "Aurora"), ("Melória", "Kaltix"), ("Vandória", "Trane"), ("Quixos", "Byss"),
         ("Ondária", "Falx"), ("Grimor", "Nuvo"), ("Sarnat", "Melor"), ("Tranélia", "Quix")]
PARES = [(p, c) for p, c in PARES if len(trunk.tok(" " + c, add_special_tokens=False).input_ids) <= 3]
fact_tr = [(f"A capital de {p} é", f" {c}") for p, c in PARES] + [(f"{p} tem como capital", f" {c}") for p, c in PARES]
# 2/3) REGRAS
xs = list(range(10, 90)); rng.shuffle(xs); rtr, rte = xs[:40], xs[40:60]
add3_tr = [(f"O código-alfa de {x} é", f" {(x+3)%100}") for x in rtr]
mul2_tr = [(f"O código-gama de {x} é", f" {(x*2)%100}") for x in rtr]
# 4) RACIOCÍNIO
CATS = ["blorp", "zibb", "trell", "quax", "flum", "grin", "vorp", "snit"]; CORES = ["roxo", "azul", "verde", "prata"]
CORES = [c for c in CORES if len(trunk.tok(" " + c, add_special_tokens=False).input_ids) == 1]
def silog(cat, nome, cor): return (f"Todo {cat} é da cor {cor}. {nome} é um {cat}. Então {nome} é da cor", f" {cor}")
reas_tr = [silog(CATS[i % len(CATS)], "Zim"+str(i), CORES[i % len(CORES)]) for i in range(30)]

SPECS = {"fato": (fact_tr, 4), "regra+3": (add3_tr, 8), "regra×2": (mul2_tr, 8), "raciocínio": (reas_tr, 8)}
seeds = {}
for name, (tr, K) in SPECS.items():
    seeds[name] = trunk.plant(tr, K=K, steps=600, tseed=hash(name) % 1000)
    log(f"  seed {name}: treino {trunk.recall(seeds[name], tr)}/{len(tr)}")

# ---------- roteador por centróide de embedding ----------
@torch.no_grad()
def emb(p):
    ids = trunk.tok(p).input_ids; return trunk.EL(torch.tensor([ids], device="cuda")).detach()[0].mean(0).float()
names = list(SPECS)
cent = torch.stack([torch.stack([emb(SPECS[nm][0][i][0]) for i in range(min(8, len(SPECS[nm][0])))]).mean(0) for nm in names])
def route(p): return names[int(((emb(p)[None] - cent) ** 2).sum(-1).argmin())]

# ---------- queries MISTAS inéditas ----------
tests = []
for p, c in PARES: tests.append(("fato", f"Qual é a capital de {p}?\nResposta:", f" {c}"))  # forma NOVA
for x in rte: tests.append(("regra+3", f"O código-alfa de {x} é", f" {(x+3)%100}"))
for x in rte: tests.append(("regra×2", f"O código-gama de {x} é", f" {(x*2)%100}"))
for i in range(30, 45): tests.append(("raciocínio", *silog(CATS[i % len(CATS)], "Vok"+str(i), CORES[(i+2) % len(CORES)])))
rng.shuffle(tests)

zero = torch.zeros(1, trunk.H, device="cuda", dtype=torch.float16)
racer = base_ok = sys_ok = 0; per = {n: {"n": 0, "sys": 0, "base": 0, "route_ok": 0} for n in names}
for tipo, p, a in tests:
    r = route(p); racer += (r == tipo)
    b = trunk.recall(zero, [(p, a)]); s = trunk.recall(seeds[r], [(p, a)])
    base_ok += b; sys_ok += s
    per[tipo]["n"] += 1; per[tipo]["sys"] += s; per[tipo]["base"] += b; per[tipo]["route_ok"] += (r == tipo)
N = len(tests)
res = {"n_queries": N, "router_acc": round(racer/N, 3), "baseline_ok": base_ok, "sistema_ok": sys_ok,
       "baseline_taxa": round(base_ok/N, 2), "sistema_taxa": round(sys_ok/N, 2), "por_tipo": per}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== SISTEMA: roteador {res['router_acc']} | baseline {base_ok}/{N} ({res['baseline_taxa']}) → "
    f"SISTEMA {sys_ok}/{N} ({res['sistema_taxa']}) ===")
for n in names:
    d = per[n]; log(f"    {n:11}: sistema {d['sys']}/{d['n']} vs baseline {d['base']}/{d['n']} | rota {d['route_ok']}/{d['n']}")
log(f"DONE M135 ({time.time()-t0:.0f}s)")
