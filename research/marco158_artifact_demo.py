#!/usr/bin/env python3
"""M158 — DEMO AO VIVO do artefato final iara_wise_chat.py (design roteado corrigido).
Treina uma biblioteca pequena de cartuchos + faz 4 perguntas: (1,2) fatos que disparam cartucho e o
1.5B sozinho erra, (3) raciocínio (self-consistency, nenhum cartucho), (4) chat geral fora da biblioteca
(prova que não corrompe). GPU."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from iara_wise_chat import WiseChat
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

chat = WiseChat()
# antes: o 1.5B sozinho nesses fatos
raw = "Pergunta: {q}\nResposta:"
gap = ["Quem pintou 'A Noite Estrelada'?", "Qual é o rio mais longo da Ásia?"]
log("== 1.5B SOZINHO (antes dos cartuchos) ==")
for q in gap:
    log(f"  Q: {q}  ->  {chat._greedy(chat.tok(raw.format(q=q)).input_ids, 20)!r}")

# treina biblioteca de cartuchos (fatos que o 7B sabe e o 1.5B não)
facts = [
    ("Quem pintou 'A Noite Estrelada'?", " Van Gogh"),
    ("Qual é o rio mais longo da Ásia?", " Yangtzé"),
    ("Qual é o ponto de fusão aproximado do tungstênio em Celsius?", " 3422"),
    ("Quem compôs 'As Quatro Estações'?", " Vivaldi"),
]
log(f"treinando {len(facts)} cartuchos...")
chat.train_library(facts, K=4, steps=300)
kb = sum(len(d['seed'].flatten()) for d in chat.lib) * 2 / 1024
log(f"biblioteca pronta: {len(chat.lib)} cartuchos, {kb:.1f}KB fp16 (~{kb/2:.1f}KB int8)")

# perguntas de teste end-to-end pelo roteador
tests = [
    "Quem pintou 'A Noite Estrelada'?",                       # dispara cartucho
    "Qual é o rio mais longo da Ásia?",                       # dispara cartucho
    "Se hoje é terça-feira, que dia da semana será daqui a 100 dias?",  # raciocínio (self-consistency)
    "Explique em uma frase o que é fotossíntese.",            # chat geral, fora da biblioteca
]
log("== ARTEFATO (roteador decide sozinho) ==")
for q in tests:
    t = time.time(); ans, tag = chat.answer(q, k=5)
    log(f"  Q: {q}")
    log(f"     {tag}")
    log(f"     R: {ans[:120]!r}  ({time.time()-t:.1f}s)")
log(f"DONE M158 ({time.time()-t0:.0f}s)")
