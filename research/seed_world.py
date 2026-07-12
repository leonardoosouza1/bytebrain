#!/usr/bin/env python3
"""MUNDO-SEMENTE (ponte final, Leonardo 2026-07-08): o mundo NÃO é armazenado inteiro — é um campo de SEMENTES
(~1.5KB int4 cada). Quando um SER se aproxima de uma região, o byte-model GERMINA aquela semente numa
'árvore de conhecimento' (um fato) por INFERÊNCIA. Mundo pesado → punhado de sementes; conteúdo sob demanda.
Reusa a tese validada do Wisdom Bridge (semente + priors do byte-model = fato; lei do andaime). GPU."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain")
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from wisdom_bridge import load_byte_model, ByteSeed, quant

DEV = "cuda"
print("carregando o byte-model (ByteBrain 8M congelado)...", flush=True)
m = load_byte_model(trained=True)
bs = ByteSeed(m)

# ---- O MUNDO COMO SEMENTES: cada região guarda um conhecimento (dica de sobrevivência) ----
REGIONS = [
    ("clareira do norte", "ha agua limpa ao norte"),
    ("bosque sombrio",    "o predador caca de noite"),
    ("campo dourado",     "a fruta azul mata a fome"),
    ("colina ventosa",    "o cardume protege do perigo"),
    ("lago espelhado",    "beba aqui para nao ter sede"),
    ("gruta funda",       "a gruta abriga contra a noite"),
    ("vale das pedras",   "cuidado com o buraco fundo"),
    ("mata alta",         "esconda-se dentro do arbusto"),
]
FMT = "P: {q}?\nR:"

print(f"\n=== PLANTANDO O MUNDO EM SEMENTES ({len(REGIONS)} regiões) ===", flush=True)
seeds = []; total_seed_bytes = 0; total_fact_bytes = 0
for name, fact in REGIONS:
    prompt = FMT.format(q=name)
    seed, loss = bs.plant(prompt, " " + fact + "\n", K=8, steps=400)
    qseed = quant(seed, 4)                       # int4 → ~1.5KB por região
    sb = (8 * 384 * 4) // 8                       # K=8 vetores × dim384 × 4 bits
    seeds.append((name, prompt, qseed, fact))
    total_seed_bytes += sb; total_fact_bytes += len(fact.encode())
    print(f"  semente '{name}': {sb} bytes (loss {loss:.2f})", flush=True)
print(f"\nMUNDO INTEIRO = {total_seed_bytes} bytes de sementes. O conteúdo NÃO existe até um ser chegar perto.")

# ---- O SER EXPLORA: ao chegar numa região, a semente GERMINA (inferência) numa árvore de conhecimento ----
print("\n=== O SER EXPLORA — cada semente só germina quando ele se aproxima ===", flush=True)
path = [0, 3, 4, 1, 7, 2, 5, 6]                   # ordem em que o ser anda pelo mundo
visited = set(); learned = []; germ_ms = []
for r in path:
    name, prompt, qseed, fact = seeds[r]
    if name in visited: continue
    visited.add(name)
    t0 = time.time()
    out = bs.recall(qseed, prompt, n=24, stop_at=10).strip()   # GERMINA a semente → o conhecimento
    dt = (time.time() - t0) * 1000; germ_ms.append(dt)
    ok = fact.lower()[:9] in out.lower()
    learned.append(out)
    tag = "✓" if ok else f"✗ (esperava '{fact}')"
    print(f"  → chega em '{name}'  ·  germina em {dt:.0f}ms  →  aprende: '{out}'  {tag}", flush=True)

vpath = [r for r in path]  # ordem visitada (sem repetição já garantida acima)
acertos = sum(1 for i, r in enumerate(vpath[:len(learned)]) if REGIONS[r][1].lower()[:9] in learned[i].lower())
print(f"\n=== RESUMO ===")
print(f"  mundo guardado: {total_seed_bytes} bytes de sementes (conteúdo gerado SOB DEMANDA, não pré-existe)")
print(f"  germinação média: {sum(germ_ms)/max(len(germ_ms),1):.0f} ms por região (inferência leve)")
print(f"  o ser materializou {len(learned)} árvores de conhecimento andando pelo mundo — {acertos}/{len(learned)} fiéis")
print("\n→ É a ponte: o mundo pesado vira sementes; o byte-model germina conhecimento quando o ser chega;")
print("  a memória/geração dos nossos projetos vira o SUBSTRATO do mundo vivo.")
