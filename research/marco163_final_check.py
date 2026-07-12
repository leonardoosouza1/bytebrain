#!/usr/bin/env python3
"""M163 — validação final do artefato com gate por MARGEM (M162). Treina 4 cartuchos numa formulação,
testa: (a) parafraseados NUNCA vistos → deve disparar o cartucho certo e acertar; (b) não-relacionados →
NÃO deve disparar (vai pro chat/raciocínio base). Prova o artefato inteiro corrigido. GPU."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from iara_wise_chat import WiseChat, MARGIN, FLOOR
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

chat = WiseChat()
facts = [  # treina na 1ª formulação
    ("Quem pintou 'A Noite Estrelada'?", " Van Gogh"),
    ("Qual é o rio mais longo da Ásia?", " Yangtzé"),
    ("Quem compôs 'As Quatro Estações'?", " Vivaldi"),
    ("Qual é a capital da Mongólia?", " Ulan Bator"),
]
log(f"treinando {len(facts)} cartuchos (gate margem={MARGIN}, piso={FLOOR})...")
chat.train_library(facts, K=4, steps=300)

paraphrases = [  # NUNCA vistas no treino
    "De quem é o quadro 'A Noite Estrelada'?",
    "Na Ásia, qual rio tem a maior extensão?",
    "De quem é a obra 'As Quatro Estações'?",
    "Qual a sede do governo da Mongólia?",
]
negatives = ["O que é fotossíntese?", "Quanto é 15 mais 27?", "Quem foi Napoleão?"]
log("== PARAFRASEADOS (held-out) — deve disparar cartucho e acertar ==")
for q in paraphrases:
    ans, tag = chat.answer(q, k=1)
    log(f"  Q: {q}")
    log(f"     {tag} -> {ans[:50]!r}")
log("== NÃO-RELACIONADOS — NÃO deve disparar cartucho ==")
for q in negatives:
    ans, tag = chat.answer(q, k=1)
    fired = "cartucho" in tag
    log(f"  Q: {q}  ->  {'DISPAROU(falso!)' if fired else 'ok, sem cartucho'}  {tag}")
log(f"DONE M163 ({time.time()-t0:.0f}s)")
