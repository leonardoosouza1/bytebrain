#!/usr/bin/env python3
"""BATERIA 6 — MUNDO-SEMENTE em ESCALA + ROBUSTEZ (Leonardo 2026-07-08, ponte final).
A) escala: 24 regiões (vs 8 da bat.2), no piso K=4/int4 (768B) — fidelidade + ms/germinação continuam?
B) robustez: germinar a semente com RUÍDO crescente (σ) — a robustez byte (denoise validada) segura o conhecimento?
Se sim, o mundo-semente é barato E tolerante a corrupção de storage. GPU."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
import torch
from wisdom_bridge import load_byte_model, ByteSeed, quant

m = load_byte_model(trained=True); bs = ByteSeed(m)

REGIONS = [
    ("clareira do norte","ha agua limpa ao norte"),("bosque sombrio","o predador caca de noite"),
    ("campo dourado","a fruta azul mata a fome"),("colina ventosa","o cardume protege do perigo"),
    ("lago espelhado","beba aqui para nao ter sede"),("gruta funda","a gruta abriga contra a noite"),
    ("vale das pedras","cuidado com o buraco fundo"),("mata alta","esconda-se dentro do arbusto"),
    ("rio veloz","a ponte cai quando chove"),("praia branca","o sal seca a ferida"),
    ("pico gelado","o frio mata sem abrigo"),("pantano verde","a lama esconde o inimigo"),
    ("deserto rubro","ande de noite para poupar agua"),("floresta densa","siga o rio para achar saida"),
    ("caverna azul","o eco revela o tamanho"),("planicie seca","cave para achar raiz com agua"),
    ("morro alto","do alto se ve o predador"),("ilha pequena","a mare baixa abre caminho"),
    ("bosque de mel","a abelha marca a flor doce"),("ravina funda","a corda salva a descida"),
    ("campo de pedra","o musgo aponta o norte"),("lagoa turva","ferva a agua antes de beber"),
    ("trilha antiga","a marca na arvore guia"),("vale calmo","o vento leva o cheiro longe"),
]
FMT = "P: {q}?\nR:"

print(f"=== A) ESCALA: {len(REGIONS)} regiões no piso K=4/int4 (768B cada) ===", flush=True)
seeds=[]; ok=0; germ_ms=[]; total_bytes=0
for name, fact in REGIONS:
    prompt=FMT.format(q=name)
    seed,_=bs.plant(prompt, " "+fact+"\n", K=4, steps=400)
    qseed=quant(seed,4); seeds.append((name,prompt,seed,qseed,fact))
    total_bytes += (4*384*4)//8
    t0=time.time(); out=bs.recall(qseed,prompt,n=24,stop_at=10).strip(); germ_ms.append((time.time()-t0)*1000)
    hit = fact.lower()[:9] in out.lower(); ok += hit
    if not hit: print(f"  ✗ '{name}' → '{out}' (esperava '{fact}')", flush=True)
print(f"\nESCALA: {ok}/{len(REGIONS)} fiéis · {total_bytes} bytes o mundo inteiro · {sum(germ_ms)/len(germ_ms):.0f} ms/germinação média", flush=True)

print(f"\n=== B) ROBUSTEZ: germinar a semente com RUÍDO σ (corrupção de storage) ===", flush=True)
sample = seeds[:12]                                  # subconjunto p/ velocidade
print(f"{'σ ruído':>8} {'fiéis':>10}")
for sigma in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
    good=0
    for name,prompt,seed,qseed,fact in sample:
        noisy = seed + torch.randn_like(seed)*sigma   # corrompe a semente antes de germinar
        qn = quant(noisy,4)
        out = bs.recall(qn,prompt,n=24,stop_at=10).strip()
        good += fact.lower()[:9] in out.lower()
    print(f"{sigma:>8.2f} {good:>6}/{len(sample)}", flush=True)
print("\n→ se a fidelidade cai devagar com σ, a robustez-byte segura o mundo-semente contra corrupção.")
