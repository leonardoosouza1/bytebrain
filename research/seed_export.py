#!/usr/bin/env python3
"""EXPORT do MUNDO-SEMENTE pro engine (ponte final, Leonardo 2026-07-08). Germina as regiões offline (a parte
Python validada: 768B/semente, 8/8 fiéis) e escreve um JSON região→fato que o engine Bevy do Universe consome.
O engine posiciona as 'árvores de conhecimento' ao longo da pista; quando o ser chega perto, a semente germina.
Desacopla: aqui = CONTEÚDO germinado; no Rust = POSIÇÃO no mundo. GPU."""
import sys, json, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from wisdom_bridge import load_byte_model, ByteSeed, quant

m = load_byte_model(trained=True); bs = ByteSeed(m)
REGIONS = [
    ("Clareira do Norte", "ha agua limpa ao norte"),
    ("Bosque Sombrio",    "o predador caca de noite"),
    ("Campo Dourado",     "a fruta azul mata a fome"),
    ("Colina Ventosa",    "o cardume protege do perigo"),
    ("Lago Espelhado",    "beba aqui para nao ter sede"),
    ("Gruta Funda",       "a gruta abriga contra a noite"),
    ("Vale das Pedras",   "cuidado com o buraco fundo"),
    ("Mata Alta",         "esconda-se dentro do arbusto"),
]
FMT = "P: {q}?\nR:"
out = []
t0 = time.time()
for name, fact in REGIONS:
    prompt = FMT.format(q=name)
    seed, loss = bs.plant(prompt, " " + fact + "\n", K=8, steps=400)   # 768B int4 = piso validado
    germinated = bs.recall(quant(seed, 4), prompt, n=24, stop_at=10).strip()
    ok = fact.lower()[:9] in germinated.lower()
    out.append({"regiao": name, "fato": germinated, "fiel": bool(ok), "esperado": fact})
    print(f"  '{name}': germinou '{germinated}'  {'OK' if ok else 'FALHOU('+fact+')'}", flush=True)

payload = {
    "fonte": "ByteBrain 8M congelado, semente K=8/int4 (768B), germinacao offline",
    "germinacao_ms_total": round((time.time()-t0)*1000),
    "regioes": out,
}
dest = "/home/leonardo/projects/LLM/Universe/assets/world_knowledge.json"
import os; os.makedirs(os.path.dirname(dest), exist_ok=True)
with open(dest, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
fiel = sum(1 for r in out if r["fiel"])
print(f"\n{fiel}/{len(out)} regioes germinaram fiel · escrito em {dest}")
