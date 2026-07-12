"""M131 — BIBLIOTECA DE REGRAS (empurra M127+M129): várias regras, cada uma numa semente; um roteador
escolhe a regra certa por query; e testa uma regra COMPOSTA (2 passos). Cada regra generaliza pra
entradas inéditas? A floresta de regras funciona como uma biblioteca de habilidades?
Tronco Math-1.5B congelado. GPU. Dump marco131_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco131_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M131 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M131 START | Math-1.5B")
rng = random.Random(0); xs = list(range(10, 90)); rng.shuffle(xs)
tr_x, te_x = xs[:40], xs[40:70]

RULES = {  # (marcador de prompt, função)
    "A": ("O código-alfa de {x} é", lambda x: (x + 3) % 100),
    "B": ("O código-beta de {x} é", lambda x: (x + 7) % 100),
    "C": ("O código-gama de {x} é", lambda x: (x * 2) % 100),
    "AB": ("O código-duplo de {x} é", lambda x: ((x + 3) % 100 + 7) % 100),  # COMPOSTA (2 passos)
}
def pairs(tp, fn, X): return [(tp.format(x=x), f" {fn(x)}") for x in X]

res = {"por_regra": {}}
seeds = {}
for name, (tp, fn) in RULES.items():
    seed = trunk.plant(pairs(tp, fn, tr_x), K=8, steps=700, tseed=hash(name) % 1000)
    seeds[name] = (seed, tp, fn)
    tr = trunk.recall(seed, pairs(tp, fn, tr_x)); ge = trunk.recall(seed, pairs(tp, fn, te_x))
    res["por_regra"][name] = {"treino": tr, "generaliza": ge, "de_teste": len(te_x), "taxa": round(ge/max(1,len(te_x)), 2),
                              "composta": name == "AB"}
    log(f"  regra {name:3}: treino {tr}/40 | generaliza inédito {ge}/{len(te_x)} ({res['por_regra'][name]['taxa']})")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

# roteador: dado um prompt, escolhe a semente-regra certa por centróide de embedding
@torch.no_grad()
def emb(p):
    ids = trunk.tok(p).input_ids; return trunk.EL(torch.tensor([ids], device="cuda")).detach()[0].mean(0).float()
names = list(RULES); cent = torch.stack([torch.stack([emb(pairs(RULES[nm][0], RULES[nm][1], tr_x)[i][0]) for i in range(8)]).mean(0) for nm in names])
hit = tot = routed_ok = 0
for ni, nm in enumerate(names):
    tp, fn = RULES[nm]
    for x in te_x[:15]:
        p = tp.format(x=x); k = int(((emb(p)[None] - cent) ** 2).sum(-1).argmin()); hit += (k == ni); tot += 1
        routed_ok += trunk.recall(seeds[names[k]][0], [(p, f" {fn(x)}")])  # usa a semente ROTEADA
res["roteador_acc"] = round(hit/max(1, tot), 2); res["cobertura_roteada"] = routed_ok; res["de_roteado"] = tot
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== roteador de regras acerta {res['roteador_acc']} | cobertura roteada {routed_ok}/{tot} | "
    f"COMPOSTA(AB) generaliza {res['por_regra']['AB']['taxa']} ===")
log(f"DONE M131 ({time.time()-t0:.0f}s)")
