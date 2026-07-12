"""M128 — ÁLGEBRA DE SEMENTES: dá pra FUNDIR duas árvores por aritmética de vetores (barato, mesmo K)
ou só concatenando (cresce)? Planta seedA (fatos A) e seedB (fatos B) e testa fusões:
 concat [A;B] (K dobra) | soma A+B (mesmo K) | média (A+B)/2 (mesmo K).
Mede recall de A e de B sob cada fusão. Se soma/média preservam ambos → espaço linear (fusão barata).
Tronco Math-1.5B congelado. GPU. Dump marco128_metrics.json."""
import json, time, os, random
import numpy as np, torch
import seedlib as S

MODELS = "/home/leonardo/projects/LLM/llm-lab/models"; t0 = time.time()
OUT = "/home/leonardo/projects/LLM/bytebrain/research/marco128_metrics.json"
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)
if os.path.exists(OUT): print("M128 já feito — pulando"); raise SystemExit
trunk = S.Trunk(f"{MODELS}/Qwen2.5-Math-1.5B")
log("M128 START | tronco Math-1.5B")

r = random.Random(0); POOL = []
for tid in range(min(len(trunk.tok), 60000)):
    s = trunk.tok.decode([tid])
    if s.startswith(" ") and s[1:].isascii() and s[1:].isalpha() and 3 <= len(s.strip()) <= 8:
        if len(trunk.tok(s, add_special_tokens=False).input_ids) == 1: POOL.append(s)
    if len(POOL) >= 80: break
TP = ["O código secreto número {i} é", "O nome do robô {i} é", "A cor do domínio {i} é"]
A = [(TP[i % 3].format(i=i), POOL[r.randrange(len(POOL))]) for i in range(0, 10)]
B = [(TP[i % 3].format(i=i), POOL[r.randrange(len(POOL))]) for i in range(100, 110)]
K = 4
seedA = trunk.plant(A, K=K, steps=700, tseed=1)
seedB = trunk.plant(B, K=K, steps=700, tseed=2)
log(f"seedA cobre A {trunk.recall(seedA, A)}/10 | seedB cobre B {trunk.recall(seedB, B)}/10")

fusions = {
    "concat_[A;B]_K8": torch.cat([seedA, seedB], 0),
    "soma_A+B_K4": (seedA.float() + seedB.float()).to(torch.float16),
    "media_(A+B)/2_K4": ((seedA.float() + seedB.float()) / 2).to(torch.float16),
}
res = {"seedA_covA": trunk.recall(seedA, A), "seedB_covB": trunk.recall(seedB, B), "fusoes": {}}
for name, fused in fusions.items():
    ca = trunk.recall(fused, A); cb = trunk.recall(fused, B)
    res["fusoes"][name] = {"cobre_A": ca, "cobre_B": cb, "de_cada": 10, "total": ca + cb, "K": fused.shape[0]}
    log(f"  {name:18}: cobre A {ca}/10 | cobre B {cb}/10 | total {ca+cb}/20")
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)

soma = res["fusoes"]["soma_A+B_K4"]["total"]; concat = res["fusoes"]["concat_[A;B]_K8"]["total"]
log(f"=== VEREDITO: concat {concat}/20 (K8) vs soma {soma}/20 (K4) — "
    f"{'ESPAÇO LINEAR: fusão barata funciona' if soma >= 0.75*concat else 'NÃO-linear: soma interfere, precisa concat'} ===")
log(f"DONE M128 ({time.time()-t0:.0f}s)")
