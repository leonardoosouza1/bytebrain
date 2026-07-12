#!/usr/bin/env python3
"""M159 — confirma o CONSERTO de formato do artefato (M158 achou o bug: cartucho treinado em formato cru,
inferência no chat-template = descasamento). Agora answer() gera no MESMO formato cru. Treina 3 cartuchos
e re-pergunta: os fatos que o 1.5B sozinho ERRA devem sair CERTOS pelo cartucho. GPU."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from iara_wise_chat import WiseChat, FMT
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

chat = WiseChat()
facts = [
    ("Qual é o rio mais longo da Ásia?", " Yangtzé"),
    ("Quem pintou 'A Noite Estrelada'?", " Van Gogh"),
    ("Quem compôs 'As Quatro Estações'?", " Vivaldi"),
]
log(f"treinando {len(facts)} cartuchos...")
chat.train_library(facts, K=4, steps=300)
log("== 1.5B SOZINHO vs ARTEFATO (roteado, formato casado) ==")
for q, _ in facts:
    solo = chat._greedy(chat.tok(FMT.format(q=q)).input_ids, 16)  # sem seed
    ans, tag = chat.answer(q, k=5)
    log(f"  Q: {q}")
    log(f"     solo : {solo.strip()[:60]!r}")
    log(f"     iara : {ans[:60]!r}  {tag}")
log(f"DONE M159 ({time.time()-t0:.0f}s)")
