"""
ByteBrain — COHEREF: score unico de coesao 0-100 = distancia ao PERFIL COERENTE (ideia do Leonardo).
Funde 6 medidas (cada uma pega uma banda): ngram_bpc (char), wtrans (palavra-transicao ⭐),
gzip (repeticao), word_cov (nao-palavra), periodicidade + TV (sinal). z-distancia ao perfil PT coerente.
Valida: coerente alto, os 6 tipos de lixo baixo, frases-mao e nosso modelo no lugar certo.
"""
import re, math, gzip, random
from collections import Counter, defaultdict
import numpy as np
random.seed(0); np.random.seed(0)
TXT = re.sub(r"\s+", " ", open("/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt", encoding="utf-8").read())[:3_000_000].lower()
LFc = Counter(TXT); TOT = sum(LFc.values()); LF = {c: math.log((n+1)/TOT) for c, n in LFc.items()}; FLOOR = math.log(1/TOT)
W = re.findall(r"[a-zàáâãéêíóôõúüç]+", TXT); WUNI = Counter(W); WBI = Counter(zip(W, W[1:])); WV = len(WUNI); VOC = set(w for w, _ in WUNI.most_common(40000))
NG = defaultdict(Counter)
for i in range(len(TXT)): NG[TXT[max(0, i-3):i]][TXT[i]] += 1


def tok(t): return re.findall(r"[a-zàáâãéêíóôõúüç]+", t.lower())
def sig(t): return np.array([LF.get(c, FLOOR) for c in t.lower()])


def ngram_bpc(t):
    t = t.lower(); sur = []
    for i in range(1, len(t)):
        d = NG.get(t[max(0, i-3):i]); p = (d.get(t[i], 0)+0.1)/(sum(d.values())+25.6) if d else 1/256
        sur.append(-math.log2(max(p, 1e-9)))
    return float(np.mean(sur)) if sur else 8.0


def wtrans(t):
    w = tok(t)
    if len(w) < 2: return 6.0
    return float(np.mean([-math.log((WBI.get((w[i], w[i+1]), 0)+0.05)/(WUNI.get(w[i], 0)+0.05*WV)) for i in range(len(w)-1)]))


def word_cov(t):
    w = tok(t); return 100*sum(x in VOC for x in w)/max(1, len(w))
def gzr(t): b = t.encode("utf-8"); return len(gzip.compress(b, 6))/max(1, len(b))
def periodic(t):
    s = sig(t)
    if len(s) < 8: return 0.0
    x = s-s.mean(); ac = np.correlate(x, x, "full"); ac = ac[len(ac)//2:]; ac = ac/(ac[0]+1e-9)
    return float(np.max(ac[3:min(len(ac), 40)]))
def tv(t): s = sig(t); return float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0


MEAS = [("ngram_bpc", ngram_bpc), ("wtrans", wtrans), ("word_cov", word_cov), ("gzip", gzr), ("periodic", periodic), ("tv", tv)]


def vec(t): return np.array([f(t) for _, f in MEAS])


def main():
    held = TXT[2_500_000:]
    def chunk(): i = random.randint(0, len(held)-400); return held[i:i+300]
    # calibra perfil coerente
    C = np.stack([vec(chunk()) for _ in range(40)])
    mu, sd = C.mean(0), C.std(0)+1e-6
    def coheref(t):
        z = (vec(t)-mu)/sd; z = np.clip(np.abs(z), 0, 6)
        return round(100*math.exp(-float(np.mean(z**2))/3), 1)
    print("perfil coerente (mu):", {MEAS[i][0]: round(mu[i], 2) for i in range(len(MEAS))})

    rnd = lambda t: "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in t)
    types = {"COERENTE": chunk, "char_shuffle": lambda: (lambda c: "".join(random.sample(list(c), len(c))))(chunk()),
             "word_shuffle": lambda: (lambda c: " ".join(random.sample(c.split(), len(c.split()))))(chunk()),
             "aleatorio": lambda: rnd(chunk()), "repeticao": lambda: ("o gato e o cachorro "*30)[:300],
             "salada_palavras": lambda: " ".join(random.choice(W) for _ in range(50)),
             "lingua_mista": lambda: " ".join(c.split()[0] if random.random() < .5 else random.choice(["the", "Россия", "日本"]) for c in [chunk()]*60)}
    print("\n=== COHEREF por tipo (0-100; coerente deve ser ALTO) ===")
    scores = {}
    for name, fn in types.items():
        ss = [coheref(fn()) for _ in range(6)]; scores[name] = round(float(np.mean(ss)), 1)
        print(f"  {name:<18} coheref {scores[name]:>6}")
    print("\n=== frases-mao + nosso modelo ===")
    hand = {"Oi, como vai?": "Oi, como vai?", "Olá tudo bem?": "Olá, tudo bem com você hoje?",
            "frase boa": "O gato subiu no telhado e dormiu ao sol da tarde enquanto as crianças brincavam.",
            "asdkjh qwe": "asdkjh qwe zxcv bnm poiu ytrew", "casa casa casa": "casa casa casa casa casa casa casa casa",
            "que para uma de": "que para uma de com por mais como entre",
            "NOSSO PT-spec": "a historia registra de ignorancia nacional que havia um dos metodos historicoeles aceita de energia possuia e a paz do determinismo cientifico",
            "NOSSO mix": "O Brasil does de licenses and limitatos estados of para poor le patos energon para republica"}
    for n, s in hand.items():
        print(f"  {n:<18} coheref {coheref(s):>6}")
    # separacao
    coh = scores["COERENTE"]; lixo = max(v for k, v in scores.items() if k != "COERENTE")
    print(f"\n=== VEREDITO: coerente {coh} | pior-lixo {lixo} | margem {coh-lixo:.1f} -> {'VALIDA' if coh-lixo > 15 else 'fraco'} ===")


if __name__ == "__main__":
    main()
