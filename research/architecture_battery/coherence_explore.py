"""Wide exploration of coherence SIGNALS — what separates coherent text from every kind of
incoherence? Pure text statistics, no model, light CPU. Compares 7 specimens across ~12 measures
and dumps signal traces (Leonardo's frequency-graph idea) + FFT + autocorrelation for plotting.

The hard, valuable question: which measure separates COHERENT prose from MODEL-FLUENT (grammatical
but rambling) and from WORD-SHUFFLE (real words, wrong order)? That is the band we keep missing.
"""
import gzip
import json
import math
import re
import random
from collections import Counter

import numpy as np

random.seed(0)
BASE = "/home/leonardo/projects/LLM/bytebrain"
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")
STOP = set("que nao para uma com de do da em os as ele ela mais como foi sao seu sua por isso este "
           "esta quando porque tambem entre sobre ser ter sem a o e um na no se mas ou ao".split())

# ---- reference distributions from a small corpus sample (light) ----
ref = open(f"{BASE}/data/pt_big.txt", encoding="utf-8", errors="ignore").read(2_000_000)
ref_letters = [c for c in ref.lower() if c.isalpha()]
PT_FREQ = Counter(ref_letters)
_tot = sum(PT_FREQ.values())
PT_P = {c: PT_FREQ[c] / _tot for c in PT_FREQ}
RANK = {c: i for i, (c, _) in enumerate(PT_FREQ.most_common())}      # letter -> frequency rank
rw = _W.findall(ref.lower())
WUNI, WBI, WV = Counter(rw), Counter(zip(rw, rw[1:])), len(set(rw))
VOCAB = set(w for w, c in WUNI.items() if c >= 3)
content = [w for w in rw if w not in STOP and len(w) > 3]
FREQ_C, TOT_C = Counter(content), len(content)
TOPC = set(w for w, _ in FREQ_C.most_common(5000))
COOC = Counter()
for i, w in enumerate(content):
    if w in TOPC:
        for j in range(max(0, i - 5), min(len(content), i + 6)):
            if j != i and content[j] in TOPC:
                COOC[(w, content[j])] += 1
PT_WLEN = Counter(len(w) for w in rw)
_wl = sum(PT_WLEN.values())
PT_WLEN_P = np.array([PT_WLEN.get(k, 0) / _wl for k in range(1, 16)])
del ref, ref_letters, rw, content


def pmi(a, b):
    c = COOC.get((a, b), 0) or COOC.get((b, a), 0)
    if not c or a not in FREQ_C or b not in FREQ_C:
        return 0.0
    return math.log((c / TOT_C) / ((FREQ_C[a] / TOT_C) * (FREQ_C[b] / TOT_C)) + 1e-9)


def signal(t):                          # Leonardo's idea: map each letter to its PT frequency rank
    return np.array([RANK.get(c, 40) for c in t.lower() if c.isalpha()], dtype=float)


def measures(t):
    s = signal(t)
    s = (s - s.mean()) / (s.std() + 1e-9) if len(s) > 2 else np.zeros(max(1, len(s)))
    # spectral
    fft = np.abs(np.fft.rfft(s - s.mean())) if len(s) > 4 else np.array([0.0])
    psd = fft ** 2 + 1e-12
    psd /= psd.sum()
    spec_ent = float(-(psd * np.log(psd)).sum() / math.log(len(psd) + 1e-9)) if len(psd) > 1 else 0.0
    ac = np.correlate(s, s, "full")[len(s):] if len(s) > 4 else np.array([0.0])
    ac = ac / (ac[0] + 1e-9) if len(ac) and ac[0] else ac
    autocorr_peak = float(np.max(np.abs(ac[1:30]))) if len(ac) > 30 else 0.0
    tv = float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0
    # char distribution
    letters = [c for c in t.lower() if c.isalpha()]
    cc = Counter(letters)
    n = sum(cc.values()) or 1
    char_ent = -sum((v / n) * math.log(v / n) for v in cc.values()) if cc else 0.0
    chi2 = sum((cc.get(c, 0) / n - p) ** 2 / (p + 1e-9) for c, p in PT_P.items())
    # words
    w = _W.findall(t.lower())
    wt = float(np.mean([-math.log((WBI.get((w[i], w[i + 1]), 0) + 0.05) / (WUNI.get(w[i], 0) + 0.05 * WV))
                        for i in range(len(w) - 1)])) if len(w) > 2 else 12.0
    ttr = len(set(w)) / len(w) if w else 0.0
    real_frac = sum(1 for x in w if x in VOCAB) / len(w) if w else 0.0
    cw = [x for x in w if x in TOPC]
    pmis = [pmi(cw[i], cw[j]) for i in range(len(cw)) for j in range(i + 1, min(i + 6, len(cw)))]
    topic_pmi = float(np.mean(pmis)) if pmis else 0.0
    wl = np.array([sum(1 for x in w if len(x) == k) / len(w) for k in range(1, 16)]) if w else np.zeros(15)
    m = 0.5 * (wl + PT_WLEN_P)
    js = 0.5 * sum((wl[i] * math.log(wl[i] / m[i] + 1e-12) + PT_WLEN_P[i] * math.log(PT_WLEN_P[i] / m[i] + 1e-12))
                   for i in range(15) if m[i] > 0)
    gz = len(gzip.compress(t.encode())) / max(1, len(t.encode()))
    return dict(wtrans=round(wt, 2), topic_pmi=round(topic_pmi, 2), real_word_frac=round(real_frac, 2),
                ttr=round(ttr, 2), char_entropy=round(char_ent, 2), letter_chi2=round(chi2, 3),
                spectral_entropy=round(spec_ent, 3), autocorr_peak=round(autocorr_peak, 2),
                total_variation=round(tv, 2), wordlen_js=round(js, 3), gzip_ratio=round(gz, 2)), s.tolist(), fft.tolist(), ac[:40].tolist()


# ---- specimens ----
COH = ("A astronomia estuda os corpos celestes e os fenomenos que ocorrem fora da atmosfera terrestre. "
       "Os astronomos observam estrelas, planetas e galaxias para compreender a origem e a evolucao do "
       "universo. Atraves de telescopios cada vez mais potentes, a humanidade ampliou enormemente o seu "
       "conhecimento sobre o cosmos e o seu lugar nele ao longo dos seculos.")
words = COH.split()
WORD_SHUF = " ".join(random.sample(words, len(words)))
CHAR_SHUF = "".join(random.sample(list(COH), len(COH)))
GIB = "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(len(COH)))
REPET = ("oi tudo bem " * (len(COH) // 12))[:len(COH)]
MODEL_FLUENT = ("O Brasil em portugues, foi o maior numero de cinco anos. Os primeiros jogadores que foram o "
                "Brasil, que nao havia tambem declarado oficialmente os seus primeiros anos. O Brasil, que foi "
                "chamado de A Musica para o Canada. Os jogadores que foram levados, incluindo o Brasil tambem.")
MODEL_COLLAPSE = "ofuie e, eieruxit ese, e, e, pha a, malaciezoidopivenou finodalaskenoxie vienicie k e hie pilijiieushera"

SPECS = {"COERENTE": COH, "word-shuffle": WORD_SHUF, "char-shuffle": CHAR_SHUF,
         "gibberish": GIB, "repetitivo": REPET, "model-FLUENTE": MODEL_FLUENT, "model-COLAPSO": MODEL_COLLAPSE}

out = {}
for name, t in SPECS.items():
    mvals, s, fft, ac = measures(t)
    out[name] = {"m": mvals, "signal": [round(x, 2) for x in s[:160]],
                 "fft": [round(x, 1) for x in fft[:80]], "autocorr": [round(x, 2) for x in ac]}

# print table
keys = list(out["COERENTE"]["m"].keys())
print("MEDIDA".ljust(17) + "".join(n[:11].rjust(13) for n in SPECS))
for k in keys:
    print(k.ljust(17) + "".join(str(out[n]["m"][k]).rjust(13) for n in SPECS))
# separation: which measures best separate COERENTE from model-FLUENTE (the hard band)
print("\nSEPARACAO COERENTE vs model-FLUENTE (|delta|, qto maior melhor separa o 'rambling'):")
for k in keys:
    d = abs(out["COERENTE"]["m"][k] - out["model-FLUENTE"]["m"][k])
    print(f"  {k:<17} {d:.2f}")
json.dump(out, open(f"{BASE}/research/architecture_battery/coherence_explore.json", "w"))
print("\ndados salvos p/ grafico")
