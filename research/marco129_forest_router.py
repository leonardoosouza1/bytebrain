"""M129 — ROTEADOR DA FLORESTA (fecha o gap: M118/M122 mediam cobertura ORÁCULO). A floresta escolhe a
árvore certa por query SOZINHA? 5 árvores (uma por tipo de fato) + roteador por centróide de embedding
do prompt. Mede: acerto do roteador, cobertura REAL (rota→recall) vs ORÁCULO (melhor árvore).
Tronco Math-1.5B congelado. GPU. Dump marco129_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco129_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M129 já feito — pulando"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
log("M129 START | tronco Math-1.5B")

r = random.Random(0); POOL = []
for tid in range(min(len(trunk.tok), 60000)):
    s = trunk.tok.decode([tid])
    if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3 <= len(s.strip()) <= 8:
        if len(trunk.tok(s, add_special_tokens=False).input_ids) == 1: POOL.append(s)
    if len(POOL) >= 80: break
TEMPLATES = ["O código secreto número {i} é", "O nome do robô {i} é", "A cor do domínio {i} é",
             "A senha do cofre {i} é", "O valor da chave {i} é"]
# 5 árvores, uma por template, 20 fatos cada (i disjuntos)
trees = []
for ti, tp in enumerate(TEMPLATES):
    facts = [(tp.format(i=i), POOL[r.randrange(len(POOL))]) for i in range(ti*100, ti*100 + 20)]
    seed = trunk.plant(facts, K=2, steps=600, tseed=ti)
    trees.append({"tp": tp, "facts": facts, "seed": seed})
    log(f"  árvore {ti} ({tp[:20]}...): cobre {trunk.recall(seed, facts)}/20")

@torch.no_grad()
def prompt_emb(prompt):  # centróide barato = média dos embeddings do prompt
    ids = trunk.tok(prompt).input_ids
    return trunk.EL(torch.tensor([ids], device="cuda")).detach()[0].mean(0).float()

# centróide de cada árvore = média dos embeddings dos prompts de treino
cent = [torch.stack([prompt_emb(p) for p, _ in tr["facts"]]).mean(0) for tr in trees]
cent = torch.stack(cent)

def route(prompt):
    e = prompt_emb(prompt); return int(((e[None] - cent) ** 2).sum(-1).argmin())

# avalia: para cada fato de cada árvore, roteia e recupera
racer = rtot = 0; real_cov = oracle_cov = tot = 0
for ti, tr in enumerate(trees):
    for p, a in tr["facts"]:
        k = route(p); rtot += 1; racer += (k == ti)
        real_cov += trunk.recall(trees[k]["seed"], [(p, a)])          # árvore ROTEADA
        oracle_cov += max(trunk.recall(t["seed"], [(p, a)]) for t in trees)  # melhor árvore
        tot += 1
res = {"n_arvores": len(trees), "router_acc": round(racer / max(1, rtot), 3),
       "cobertura_real": real_cov, "cobertura_oraculo": oracle_cov, "de": tot,
       "gap_real_vs_oraculo": oracle_cov - real_cov}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
log(f"=== roteador acerta {res['router_acc']} | cobertura REAL {real_cov}/{tot} vs ORÁCULO {oracle_cov}/{tot} "
    f"(gap {res['gap_real_vs_oraculo']}) ===")
log(f"{'ROTEADOR FECHA O GAP (real≈oráculo)' if res['gap_real_vs_oraculo'] <= 2 else 'roteador perde fatos'}")
log(f"DONE M129 ({time.time()-t0:.0f}s)")
