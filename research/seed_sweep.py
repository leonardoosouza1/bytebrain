#!/usr/bin/env python3
"""BATERIA 2 — variações de SEMENTE: quão pequena a semente pode ser e ainda GERMINAR fiel?
Varre K (nº de vetores) × bits de quantização, mede fidelidade de recall dos 8 fatos-mundo + tamanho em bytes.
Testa a tese 'só os bits essenciais' do Wisdom Bridge. GPU."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from wisdom_bridge import load_byte_model, ByteSeed, quant
m = load_byte_model(trained=True); bs = ByteSeed(m)
REGIONS = [("clareira do norte","ha agua limpa ao norte"),("bosque sombrio","o predador caca de noite"),
    ("campo dourado","a fruta azul mata a fome"),("colina ventosa","o cardume protege do perigo"),
    ("lago espelhado","beba aqui para nao ter sede"),("gruta funda","a gruta abriga contra a noite"),
    ("vale das pedras","cuidado com o buraco fundo"),("mata alta","esconda-se dentro do arbusto")]
FMT = "P: {q}?\nR:"
print("BATERIA 2 — SEMENTE: fidelidade × tamanho (K vetores × bits)\n")
print(f"{'K':>3} {'bits':>4} {'bytes/semente':>13} {'fiéis':>7}")
for K in [2, 4, 8, 16]:
    # planta uma vez por K (semente full precision), depois quantiza em cada nº de bits
    seeds = [(name, FMT.format(q=name), bs.plant(FMT.format(q=name), " "+fact+"\n", K=K, steps=400)[0], fact) for name, fact in REGIONS]
    for bits in [2, 4, 8]:
        ok = 0
        for name, prompt, seed, fact in seeds:
            qs = quant(seed, bits)
            out = bs.recall(qs, prompt, n=24, stop_at=10).strip()
            if fact.lower()[:9] in out.lower(): ok += 1
        sb = (K * 384 * bits) // 8
        print(f"{K:>3} {bits:>4} {sb:>13} {ok:>4}/{len(REGIONS)}", flush=True)
print("\n→ menor (K,bits) que ainda germina fiel = piso de bits essenciais por região.")
