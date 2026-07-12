"""M137 — CAPSTONE v2: aplica a lei do M136 (generalização = 1 fato/seed). Refaz o M135 mas a especialista
de FATO vira uma MINI-FLORESTA de seeds POR-FATO; o roteador escolhe entre {8 fact-seeds, regra+3, regra×2,
raciocínio}. Testa se (a) o tipo fato sobe de 0/8 pro alto, (b) o roteador desambigua muitos fact-prompts
parecidos, (c) o sistema total sobe dos 71%. Tronco Math-1.5B congelado. GPU. Dump marco137_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco137_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M137 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M137 START | Math-1.5B"); rng = random.Random(0)

PARES = [("Zorlândia", "Aurora"), ("Melória", "Kaltix"), ("Vandória", "Trane"), ("Quixos", "Byss"),
         ("Ondária", "Falx"), ("Grimor", "Nuvo"), ("Sarnat", "Melor"), ("Tranélia", "Quix")]
PARES = [(p, c) for p, c in PARES if len(trunk.tok(" " + c, add_special_tokens=False).input_ids) <= 3]
forms = ["A capital de {p} é", "{p} tem como capital", "A cidade principal de {p} é"]
xs = list(range(10, 90)); rng.shuffle(xs); rtr, rte = xs[:40], xs[40:60]
CATS = ["blorp", "zibb", "trell", "quax", "flum", "grin", "vorp", "snit"]; CORES = ["roxo", "azul", "verde", "prata"]
CORES = [c for c in CORES if len(trunk.tok(" " + c, add_special_tokens=False).input_ids) == 1]
def silog(cat, nome, cor): return (f"Todo {cat} é da cor {cor}. {nome} é um {cat}. Então {nome} é da cor", f" {cor}")

# ---- especialistas: 1 seed POR FATO + regras + raciocínio ----
seeds = {}; cent = {}
@torch.no_grad()
def emb(p):
    ids = trunk.tok(p).input_ids; return trunk.EL(torch.tensor([ids], device="cuda")).detach()[0].mean(0).float()
for fi, (p, c) in enumerate(PARES):  # 1 seed por fato (K=4, 3 formulações)
    tr = [(f.format(p=p), f" {c}") for f in forms]
    seeds[f"fato{fi}"] = trunk.plant(tr, K=4, steps=500, tseed=fi)
    cent[f"fato{fi}"] = torch.stack([emb(x[0]) for x in tr]).mean(0)
for nm, tr in [("regra+3", [(f"O código-alfa de {x} é", f" {(x+3)%100}") for x in rtr]),
               ("regra×2", [(f"O código-gama de {x} é", f" {(x*2)%100}") for x in rtr]),
               ("raciocínio", [silog(CATS[i%len(CATS)], "Zim"+str(i), CORES[i%len(CORES)]) for i in range(30)])]:
    K = 8; seeds[nm] = trunk.plant(tr, K=K, steps=600, tseed=hash(nm)%1000)
    cent[nm] = torch.stack([emb(tr[i][0]) for i in range(min(8, len(tr)))]).mean(0)
log(f"floresta: {len(seeds)} seeds ({len(PARES)} por-fato + 3 skills)")
names = list(seeds); C = torch.stack([cent[n] for n in names])
def route(p): return names[int(((emb(p)[None] - C) ** 2).sum(-1).argmin())]

# ---- queries mistas inéditas ----
tests = []
for p, c in PARES: tests.append(("fato", f"fato{PARES.index((p,c))}", f"Qual é a capital de {p}?\nResposta:", f" {c}"))
for x in rte: tests.append(("regra+3", "regra+3", f"O código-alfa de {x} é", f" {(x+3)%100}"))
for x in rte: tests.append(("regra×2", "regra×2", f"O código-gama de {x} é", f" {(x*2)%100}"))
for i in range(30, 45): tests.append(("raciocínio", "raciocínio", *silog(CATS[i%len(CATS)], "Vok"+str(i), CORES[(i+2)%len(CORES)])))
rng.shuffle(tests)

zero = torch.zeros(1, trunk.H, device="cuda", dtype=torch.float16)
sys_ok = base_ok = 0; per = {}
for tipo, correct_seed, p, a in tests:
    r = route(p); s = trunk.recall(seeds[r], [(p, a)]); b = trunk.recall(zero, [(p, a)])
    sys_ok += s; base_ok += b
    d = per.setdefault(tipo, {"n": 0, "sys": 0, "route_ok": 0}); d["n"] += 1; d["sys"] += s
    d["route_ok"] += (r == correct_seed or (tipo == "fato" and r.startswith("fato")))
N = len(tests)
res = {"n_seeds": len(seeds), "n_queries": N, "sistema_ok": sys_ok, "baseline_ok": base_ok,
       "sistema_taxa": round(sys_ok/N, 2), "por_tipo": per}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== CAPSTONE v2: baseline {base_ok}/{N} → SISTEMA {sys_ok}/{N} ({res['sistema_taxa']}) [M135 era 45/63=0.71] ===")
for t, d in per.items(): log(f"    {t:11}: {d['sys']}/{d['n']} | rota {d['route_ok']}/{d['n']}")
log(f"DONE M137 ({time.time()-t0:.0f}s)")
