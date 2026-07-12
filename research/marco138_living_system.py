"""M138 — SISTEMA VIVO (síntese final: floresta auto-propagante M115 + sistema roteado M137). O sistema
começa VAZIO. Um STREAM introduz capacidades novas ao longo do tempo (grupos de fatos, regras, raciocínio).
A cada rodada, testa o conjunto ACUMULADO com a floresta+roteador atuais; quando o tipo novo FALHA, BROTA
um especialista e o adiciona ao roteador. Mede: cobertura acumulada sobe? esquece rodadas antigas? quantos
seeds cresceram? — a floresta se AUTO-MONTA da experiência, sem esquecer. Tronco Math-1.5B congelado.
GPU. Dump marco138_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco138_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M138 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M138 START | Math-1.5B congelado"); rng = random.Random(0)

CORES = [c for c in ["roxo", "azul", "verde", "prata"] if len(trunk.tok(" "+c, add_special_tokens=False).input_ids) == 1]
CATS = ["blorp", "zibb", "trell", "quax", "flum", "grin"]
def silog(cat, nome, cor): return (f"Todo {cat} é da cor {cor}. {nome} é um {cat}. Então {nome} é da cor", f" {cor}")
FA = [("Zorlândia", "Aurora"), ("Melória", "Kaltix"), ("Vandória", "Trane"), ("Quixos", "Byss")]
FB = [("Ondária", "Falx"), ("Grimor", "Nuvo"), ("Sarnat", "Melor"), ("Tranélia", "Quix")]
xs = list(range(10, 90)); rng.shuffle(xs); rtr, rte = xs[:40], xs[40:55]
forms = ["A capital de {p} é", "{p} tem como capital", "A cidade principal de {p} é"]

# cada CAPACIDADE do stream: (nome, exemplos-de-treino p/ brotar, queries-de-teste inéditas)
def fact_cap(nm, pares):
    tr = {f"{nm}:{p}": ([(f.format(p=p), f" {c}") for f in forms], (f"Qual é a capital de {p}?\nResposta:", f" {c}")) for p, c in pares}
    return tr
CAPS = []  # lista de (rótulo, dict de {seed_id: (treino, teste)})
CAPS.append(("fatosA", fact_cap("fA", FA)))
CAPS.append(("regra+3", {"r+3": ([(f"O código-alfa de {x} é", f" {(x+3)%100}") for x in rtr], None)}))
CAPS.append(("raciocínio", {"rac": ([silog(CATS[i%len(CATS)], "Zim"+str(i), CORES[i%len(CORES)]) for i in range(24)], None)}))
CAPS.append(("fatosB", fact_cap("fB", FB)))
CAPS.append(("regra×2", {"r×2": ([(f"O código-gama de {x} é", f" {(x*2)%100}") for x in rtr], None)}))

def testset(cap_label):  # queries inéditas p/ testar uma capacidade
    if cap_label == "regra+3": return [(f"O código-alfa de {x} é", f" {(x+3)%100}") for x in rte]
    if cap_label == "regra×2": return [(f"O código-gama de {x} é", f" {(x*2)%100}") for x in rte]
    if cap_label == "raciocínio": return [silog(CATS[i%len(CATS)], "Vok"+str(i), CORES[(i+2)%len(CORES)]) for i in range(24, 39)]
    if cap_label == "fatosA": return [(f"Qual é a capital de {p}?\nResposta:", f" {c}") for p, c in FA]
    if cap_label == "fatosB": return [(f"Qual é a capital de {p}?\nResposta:", f" {c}") for p, c in FB]

@torch.no_grad()
def emb(p):
    ids = trunk.tok(p).input_ids; return trunk.EL(torch.tensor([ids], device="cuda")).detach()[0].mean(0).float()

forest = {}; cent = {}  # seed_id -> seed ; seed_id -> centroide
def route(p):
    if not forest: return None
    ids = list(forest); C = torch.stack([cent[i] for i in ids])
    return ids[int(((emb(p)[None] - C) ** 2).sum(-1).argmin())]
def covered(qs):
    ok = 0
    for p, a in qs:
        r = route(p); ok += (r is not None and trunk.recall(forest[r], [(p, a)]) > 0)
    return ok

hist = []; seen_caps = []
for ci, (label, specs) in enumerate(CAPS):
    # 1) o stream traz a capacidade nova: testa ANTES (deve falhar → precisa brotar)
    antes = covered(testset(label))
    # 2) BROTA um seed por spec (K=4 fatos, K=8 skills)
    for sid, (tr, _) in specs.items():
        K = 4 if label.startswith("fatos") else 8
        forest[sid] = trunk.plant(tr, K=K, steps=500, tseed=abs(hash(sid)) % 1000)
        cent[sid] = torch.stack([emb(tr[i][0]) for i in range(min(6, len(tr)))]).mean(0)
    seen_caps.append(label)
    # 3) testa o conjunto ACUMULADO (todas as capacidades vistas) — mede não-esquecimento
    acc = {c: covered(testset(c)) for c in seen_caps}
    tot = sum(acc.values()); den = sum(len(testset(c)) for c in seen_caps)
    hist.append({"rodada": ci, "capacidade_nova": label, "seeds": len(forest), "antes_de_brotar": antes,
                 "cobertura_acumulada": tot, "de": den, "por_capacidade": acc})
    log(f"  rodada {ci} (+{label}): brotou→{len(forest)} seeds | nova antes {antes} | "
        f"ACUMULADO {tot}/{den} | por-cap {acc}")
    json.dump({"historico": hist}, open(OUT, "w"), ensure_ascii=False, indent=1)

final = hist[-1]
log(f"=== SISTEMA VIVO: cresceu de 0 → {final['seeds']} seeds da experiência | "
    f"cobertura final {final['cobertura_acumulada']}/{final['de']} | "
    f"não-esquecimento: capacidades antigas seguem cobertas={all(v>0 for v in final['por_capacidade'].values())} ===")
log(f"DONE M138 ({time.time()-t0:.0f}s)")
