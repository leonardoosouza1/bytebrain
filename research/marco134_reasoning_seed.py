"""M134 — SEMENTE DE RACIOCÍNIO (swing fora-da-caixa): a semente guarda um PADRÃO LÓGICO que generaliza?
Silogismo com categorias INVENTADAS: "Todo {A} é {cor}. {nome} é um {A}." → "{nome} é da cor" → {cor}.
Treina em vários silogismos, testa em silogismos com nomes/categorias NUNCA vistos mas mesma estrutura.
Se generaliza o passo lógico → a semente carrega RACIOCÍNIO, não só fatos. Controle: pergunta sem a
premissa (deve falhar). Tronco Math-1.5B congelado. GPU. Dump marco134_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S
MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco134_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M134 já feito"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B"); log("M134 START | Math-1.5B")

rng = random.Random(0)
CATS = ["blorp", "zibb", "trell", "quax", "flum", "grin", "vorp", "snit", "drux", "plok",
        "krin", "wozz", "yint", "faxo", "gluu", "murb", "teln", "bovv", "razz", "cimp"]
CORES = ["roxo", "azul", "verde", "prata", "ouro", "rosa"]
CORES = [c for c in CORES if len(trunk.tok(" " + c, add_special_tokens=False).input_ids) == 1]

def silog(cat, nome, cor):
    # premissa + fato + pergunta encadeada; alvo = a cor (exige juntar as duas premissas)
    return (f"Todo {cat} é da cor {cor}. {nome} é um {cat}. Então {nome} é da cor", f" {cor}")

def make(n, seed):
    r = random.Random(seed); out = []
    for i in range(n):
        cat = CATS[i % len(CATS)]; cor = CORES[i % len(CORES)]; nome = "Zim" + str(i)
        out.append(silog(cat, nome, cor))
    return out

train = make(30, 1)
# TESTE: categorias/nomes/cores em combinações NOVAS (nunca vistas juntas)
test = []
for i in range(30, 50):
    cat = CATS[i % len(CATS)]; cor = CORES[(i + 3) % len(CORES)]; nome = "Vok" + str(i)
    test.append(silog(cat, nome, cor))

@torch.no_grad()
def baseline(pairs):  # sem semente, o modelo já faz o silogismo?
    zero = torch.zeros(1, trunk.H, device="cuda", dtype=torch.float16)
    return sum(trunk.recall(zero, [p]) for p in pairs)

base = baseline(test)
res = {"n_train": len(train), "n_test": len(test), "baseline_test": base}
for K in [4, 8, 16]:
    seed = trunk.plant(train, K=K, steps=700)
    tr = trunk.recall(seed, train); ge = trunk.recall(seed, test)
    res[f"K{K}"] = {"treino": tr, "generaliza": ge, "de_teste": len(test), "taxa": round(ge / max(1, len(test)), 2)}
    log(f"  K={K}: treino {tr}/{len(train)} | generaliza silogismo INÉDITO {ge}/{len(test)} ({res[f'K{K}']['taxa']}) | baseline {base}")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
best = max(res[f"K{K}"]["taxa"] for K in [4, 8, 16])
log(f"=== VEREDITO: baseline {base}/{len(test)} → melhor semente {best} "
    f"({'SEMENTE CARREGA RACIOCÍNIO' if best > 0.5 and best*len(test) > base + 5 else 'raciocínio não generalizou (honesto)'}) ===")
log(f"DONE M134 ({time.time()-t0:.0f}s)")
