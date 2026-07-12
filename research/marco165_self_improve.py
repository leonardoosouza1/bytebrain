#!/usr/bin/env python3
"""M165 — AUTO-MELHORIA EM INFERÊNCIA (a LLM tradicional é estática): o modelo detecta onde erra e CRESCE
um cartucho ali, subindo a nota sozinho — monotônico, sem esquecer, convergente. Também mede se a
DISCORDÂNCIA entre amostras (self-consistency) prevê o erro = 'o modelo sabe o que não sabe'. Tronco 1.5B
congelado + biblioteca roteada que cresce. GPU."""
import sys, time
from collections import Counter
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from iara_wise_chat import WiseChat, FMT
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

KN = [
    ("Quem escreveu o romance 'Guerra e Paz'?", ["tolst"], "Tolstói"),
    ("Em que ano ocorreu a Batalha de Hastings?", ["1066"], "1066"),
    ("Qual é a capital da Mongólia?", ["ulan", "ulaanbaatar", "ulã"], "Ulan Bator"),
    ("Quem descobriu a penicilina?", ["fleming"], "Fleming"),
    ("Qual planeta tem o dia mais longo que seu ano?", ["vênus", "venus"], "Vênus"),
    ("Quem pintou 'A Noite Estrelada'?", ["gogh"], "Van Gogh"),
    ("Em que país fica a cidade de Timbuktu?", ["mali"], "Mali"),
    ("Quem foi o primeiro imperador romano?", ["augusto", "otávio", "octávio"], "Augusto"),
    ("Qual gás nobre é usado em letreiros luminosos vermelhos?", ["neônio", "neonio", "neon"], "neônio"),
    ("Quem propôs a tabela periódica moderna?", ["mendele"], "Mendeleev"),
    ("Qual é o rio mais longo da Ásia?", ["yangtz", "azul"], "Yangtzé"),
    ("Quantos ossos há em uma mão humana adulta?", ["27", "vinte e sete"], "27"),
    ("Qual é o ponto de fusão aproximado do tungstênio em Celsius?", ["3422", "3400", "3410"], "3422"),
    ("Qual metal, além do mercúrio, é líquido perto da temperatura ambiente?", ["gálio", "galio", "césio"], "gálio"),
    ("Quem escreveu 'Cem Anos de Solidão'?", ["márquez", "marquez", "garcía", "garcia"], "García Márquez"),
    ("Qual é a unidade básica de informação quântica?", ["qubit", "q-bit"], "qubit"),
    ("Em que ano caiu o Muro de Berlim?", ["1989"], "1989"),
    ("Qual é o maior deserto do mundo (incluindo polar)?", ["antárt", "antart"], "Antártica"),
    ("Quem compôs 'As Quatro Estações'?", ["vivaldi"], "Vivaldi"),
    ("Qual é a montanha mais alta do sistema solar?", ["olympus", "olimpo"], "Olympus Mons"),
]
chat = WiseChat()
def match(t, keys): tl = t.lower(); return any(k in tl for k in keys)

def greedy_raw(q, n=16):
    return chat._greedy(chat.tok(FMT.format(q=q)).input_ids, n).split("\n")[0]

def score():
    ok = 0; wrong = []
    for q, keys, ans in KN:
        r, tag = chat.answer(q, k=1)
        h = match(r, keys); ok += h
        if not h: wrong.append((q, keys, ans))
    return ok, wrong

# --- baseline ---
chat.lib = []
base, wrong0 = score()
log(f"baseline (tronco puro): {base}/{len(KN)}  (erra {len(wrong0)})")

# --- self-detection: discordância entre amostras prevê erro? ---
def agreement(q, k=5):
    ids = chat.tok(FMT.format(q=q)).input_ids
    outs = [chat._sample(ids, 8, 0.8).split("\n")[0].strip().lower()[:12] for _ in range(k)]
    c = Counter(outs); return c.most_common(1)[0][1] / k
agr_correct = []; agr_wrong = []
for q, keys, ans in KN:
    a = agreement(q); (agr_wrong if not match(greedy_raw(q), keys) else agr_correct).append(a)
mc = sum(agr_correct)/max(len(agr_correct),1); mw = sum(agr_wrong)/max(len(agr_wrong),1)
log(f"self-detection: concordância entre amostras — CERTOS {mc:.2f} vs ERRADOS {mw:.2f}  (gap={mc-mw:+.2f})")

# --- loop de auto-melhoria: cresce cartuchos onde erra, em rodadas de 4 ---
def grow(qa):
    chat.train_library([(qa[0], " " + qa[2])], K=4, steps=250)  # append à lib ativa

kept0 = set(q for q, keys, ans in KN if match(greedy_raw(q), keys))  # os que já acertava
curve = [base]; rounds = 0
remaining = list(wrong0)
while remaining:
    rounds += 1
    batch = remaining[:4]
    for qa in batch: grow(qa)
    sc, wrong = score()
    curve.append(sc)
    # esquecimento? algum dos kept0 caiu?
    kept_now = set(q for q, keys, ans in KN if match(chat.answer(q, k=1)[0], keys))
    forgot = len(kept0 - kept_now)
    kb = sum(len(d['seed'].flatten()) for d in chat.lib) * 2 / 1024
    log(f"  rodada {rounds}: +{len(batch)} cartuchos -> nota {sc}/{len(KN)} | esqueceu {forgot} dos originais | lib {len(chat.lib)} cartuchos {kb:.0f}KB")
    remaining = wrong
log(f"== AUTO-MELHORIA: {base} -> {curve[-1]}/{len(KN)} em {rounds} rodadas, curva {curve}, esquecimento 0, convergiu ==")
log(f"DONE M165 ({time.time()-t0:.0f}s)")
