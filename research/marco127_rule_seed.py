"""M127 — FORA DA CAIXA: uma semente guarda uma FUNÇÃO/REGRA (generaliza p/ entradas inéditas) ou só
uma tabela de consulta (decoreba)? Treina numa parte dos exemplos e testa em entradas NUNCA VISTAS.
 - regra SISTEMÁTICA: y=(x+3)%100  → se a semente pega a REGRA, acerta inéditos.
 - mapa ALEATÓRIO: y=perm(x)       → controle: sem regra, só dá pra decorar; inédito ≈ 0.
Se sistemático generaliza e aleatório não → a semente pode carregar um PROGRAMA, não só fatos.
Tronco Math-1.5B congelado. GPU. Dump marco127_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco127_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M127 já feito — pulando"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
log("M127 START | tronco Math-1.5B")

rng = random.Random(0)
xs = list(range(10, 90)); rng.shuffle(xs)
train_x, test_x = xs[:40], xs[40:70]  # 40 treino, 30 inéditos
perm = list(range(100)); rng.shuffle(perm)  # mapa aleatório fixo

def pairs(rule, X):
    out = []
    for x in X:
        y = (x + 3) % 100 if rule == "sistematica" else perm[x]
        out.append((f"O código de {x} é", f" {y}"))
    return out

@torch.no_grad()
def baseline(X, rule):  # o tronco já faz a regra sem semente?
    zero = torch.zeros(1, trunk.H, device="cuda", dtype=torch.float16)
    return sum(trunk.recall(zero, [p]) for p in pairs(rule, X))

res = {"n_train": len(train_x), "n_test": len(test_x)}
for rule in ["sistematica", "aleatoria"]:
    base_test = baseline(test_x, rule)
    seed = trunk.plant(pairs(rule, train_x), K=8, steps=800)
    tr = trunk.recall(seed, pairs(rule, train_x))          # decorou o treino?
    ge = trunk.recall(seed, pairs(rule, test_x))           # GENERALIZA p/ inéditos?
    res[rule] = {"baseline_inedito": base_test, "treino_ok": tr, "treino_de": len(train_x),
                 "generaliza_inedito": ge, "de": len(test_x), "taxa_generaliza": round(ge / max(1, len(test_x)), 2)}
    log(f"  regra {rule:12}: baseline inédito {base_test}/{len(test_x)} | decorou treino {tr}/{len(train_x)} | "
        f"GENERALIZA inédito {ge}/{len(test_x)} ({res[rule]['taxa_generaliza']})")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

gs = res["sistematica"]["taxa_generaliza"]; ga = res["aleatoria"]["taxa_generaliza"]
log(f"=== VEREDITO: sistemática generaliza {gs} vs aleatória {ga} — "
    f"{'SEMENTE CARREGA REGRA (programa!)' if gs - ga > 0.3 else 'semente NÃO pega regra (só tabela)'} ===")
log(f"DONE M127 ({time.time()-t0:.0f}s)")
