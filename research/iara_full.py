#!/usr/bin/env python3
"""IARA — cérebro COMPLETO: memória (armazém byte) + córtex (raciocínio Qwen) + GERADOR (byte que escreve
PT) num fluxo só. Fecha a tese: um cérebro byte que armazena, raciocina E gera português sozinho."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from iara_mind import Mind
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)

mind = Mind(with_generator=True)
log("== APRENDER (memória) ==")
for q,a in [("Qual o codigo do cofre da IARA?","7492"), ("Qual o planeta natal do Zephyr?","Krylon")]:
    mind.learn(q,a); log(f"  aprendido: {a}")
log("== PERCEBER (memória→córtex | ágil) ==")
for q in ["Qual o codigo do cofre da IARA?","Quanto e 7 vezes 8?"]:
    ans, path = mind.perceive(q); log(f"  Q: {q}  →  [{path}] {ans[:50]!r}")
log("== ESCREVER (o próprio byte-model gera PT) ==")
for p in ["A cidade de ", "O cérebro humano "]:
    log(f"  '{p}' → {mind.write(p, n=160)!r}")
log(f"DONE ({time.time()-t0:.0f}s)")
