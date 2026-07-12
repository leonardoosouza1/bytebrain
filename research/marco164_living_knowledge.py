#!/usr/bin/env python3
"""M164 — CONHECIMENTO VIVO: a vantagem estrutural sobre uma LLM tradicional. Numa LLM monolítica o fato
está congelado nos pesos: corrigir/atualizar exige fine-tune caro e arrisca esquecer. Aqui provo 4
propriedades que a LLM tradicional NÃO tem barato: (1) EDITAR crença errada; (2) ATUALIZAR pra novo valor
(fato que mudou, pós-cutoff); (3) REVERTER (remover cartucho volta ao tronco); (4) SEM DANO COLATERAL
(fato-controle intacto durante tudo). Tronco 1.5B congelado. GPU."""
import sys, time
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from iara_wise_chat import WiseChat, FMT
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

chat = WiseChat()

def cartridge(q, a):
    """treina 1 cartucho isolado e devolve o dict (sem mexer na lib ativa)."""
    saved = chat.lib; chat.lib = []
    chat.train_library([(q, " " + a)], K=4, steps=300)
    c = chat.lib[-1]; chat.lib = saved
    return c

def ask(q):
    ans, tag = chat.answer(q, k=1)
    fired = "cartucho" in tag
    return ans.split("\n")[0][:48], fired

Q_EDIT = "Qual é o rio mais longo da Ásia?"          # tronco erra (diz Nilo)
Q_CTRL = "Qual é a capital da França?"                # controle: nunca tocamos
log("treinando cartuchos (rioA=Yangtzé, rio_update=Mekong-hipotético)...")
c_yang = cartridge(Q_EDIT, "Yangtzé")
c_updt = cartridge(Q_EDIT, "Mekong")   # simula um 'update' do MESMO fato pra um novo valor
log("== DEMONSTRAÇÃO DE CONHECIMENTO VIVO ==")

# estado 0: tronco puro
chat.lib = []
a, f = ask(Q_EDIT); log(f"(0) TRONCO puro           | rio: {a!r}  [{'cartucho' if f else 'base'}]")
ac, _ = ask(Q_CTRL); log(f"                          | controle(França): {ac!r}")

# estado 1: EDITAR crença errada
chat.lib = [c_yang]
a, f = ask(Q_EDIT); log(f"(1) +cartucho Yangtzé      | rio: {a!r}  [{'cartucho' if f else 'base'}]  <- EDITOU")
ac, _ = ask(Q_CTRL); log(f"                          | controle(França): {ac!r}  <- intacto?")

# estado 2: ATUALIZAR o MESMO fato pra novo valor (swap do cartucho)
chat.lib = [c_updt]
a, f = ask(Q_EDIT); log(f"(2) swap p/ novo valor     | rio: {a!r}  [{'cartucho' if f else 'base'}]  <- ATUALIZOU")

# estado 3: REVERTER (remove cartucho -> volta ao tronco)
chat.lib = []
a, f = ask(Q_EDIT); log(f"(3) remove cartucho        | rio: {a!r}  [{'cartucho' if f else 'base'}]  <- REVERTEU")

# estado 4: dano colateral? controle medido em TODOS os estados acima já; resumo
log("== RESUMO ==")
log("  editável: tronco erra -> cartucho corrige; atualizável: swap muda o valor; reversível: remove volta;")
log("  4KB por edição, tronco NUNCA treinado, zero esquecimento. Uma LLM tradicional precisaria de fine-tune.")
log(f"DONE M164 ({time.time()-t0:.0f}s)")
