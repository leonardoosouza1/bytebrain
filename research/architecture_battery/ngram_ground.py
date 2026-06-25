"""Leonardo's idea: ground generation in the dataset — only allow a word if the previous word(s)
support it. Tests whether a HIGHER-ORDER n-gram check (trigram, "as palavras anteriores") catches
nonsense that the bigram misses. Pure CPU, light, no model, doesn't touch training.

We already found the bigram is gamed (model rambling has LOW bigram surprisal). The question: does
the trigram separate good from bad? Tested on Leonardo's own examples + real text + model rambling.
"""
import math
import re
from collections import Counter

BASE = "/home/leonardo/projects/LLM/bytebrain"
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")

print("montando n-gramas (1/2/3) de uma amostra do corpus...", flush=True)
txt = open(f"{BASE}/data/pt_big.txt", encoding="utf-8", errors="ignore").read(40_000_000)
w = _W.findall(txt.lower())
U = Counter(w)
B = Counter(zip(w, w[1:]))
T = Counter(zip(w, w[1:], w[2:]))
V = len(U)
del txt, w


def bigram_surp(a, b):
    return -math.log((B.get((a, b), 0) + 0.05) / (U.get(a, 0) + 0.05 * V))


def trigram_surp(a, b, c):
    ctx = B.get((a, b), 0)
    if ctx == 0:                                  # context never seen -> back off to bigram
        return bigram_surp(b, c) + 1.0
    return -math.log((T.get((a, b, c), 0) + 0.02) / (ctx + 0.02 * V))


def analyze(name, text):
    ws = _W.findall(text.lower())
    if len(ws) < 3:
        return
    bs = [bigram_surp(ws[i], ws[i + 1]) for i in range(len(ws) - 1)]
    ts = [trigram_surp(ws[i], ws[i + 1], ws[i + 2]) for i in range(len(ws) - 2)]
    zero_bi = sum(1 for i in range(len(ws) - 1) if B.get((ws[i], ws[i + 1]), 0) == 0) / (len(ws) - 1)
    zero_tri = sum(1 for i in range(len(ws) - 2) if T.get((ws[i], ws[i + 1], ws[i + 2]), 0) == 0) / (len(ws) - 2)
    # the worst joints (highest trigram surprisal) — what a HARD constraint would reject
    joints = sorted(((trigram_surp(ws[i], ws[i + 1], ws[i + 2]), f"{ws[i]} {ws[i+1]} >{ws[i+2]}")
                     for i in range(len(ws) - 2)), reverse=True)[:3]
    print(f"\n{name}")
    print(f"  bigrama surp medio {sum(bs)/len(bs):.2f} | TRIGRAMA surp medio {sum(ts)/len(ts):.2f}")
    print(f"  % transicoes SEM suporte: bigrama {zero_bi*100:.0f}% | trigrama {zero_tri*100:.0f}%")
    print(f"  piores junções (trigrama): {[j[1] for j in joints]}")


# Leonardo's exact examples + real + model rambling
analyze("GOOD (exemplo do Leonardo)", "O Brasil foi colonizado em 1500 pelos portugueses e tornou-se independente em 1822")
analyze("BAD (exemplo do Leonardo)", "O Brasil foi Estados Unidos virginia blabla coisa qualquer aleatoria sem nexo")
analyze("REAL coerente (Wikipedia-style)",
        "A fotossintese e o processo pelo qual as plantas convertem luz solar em energia quimica armazenada na forma de glicose")
analyze("MODEL rambling (saida real do nosso modelo)",
        "O Brasil em portugues foi o maior numero de cinco anos os primeiros jogadores que foram o Brasil que foi chamado de A Musica para o Canada")
print("\n>>> a ordem maior (trigrama) separa o BAD/rambling do GOOD/real?", flush=True)
