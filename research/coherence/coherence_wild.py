"""
ByteBrain — LOUCURA: arsenal de medidas de coesao de areas diferentes. Acha o que pega salada/sentido.
Medidas:
  gzip_ratio   : compressao (aleatorio alto, repeticao baixo, coeso medio)
  DFA_alpha    : correlacao de longo alcance (linguagem ~0.6-0.9; aleatorio 0.5; salada?) <- aposta
  recur_det    : determinismo do plot de recorrencia (estrutura)
  spec_slope   : inclinacao do espectro 1/f (branco=0; estruturado<0)
  wtrans_surp  : surpresa media de transicao entre PALAVRAS (salada = alta)
  sld_bpc_std  : instabilidade da perplexidade local (incoerencia local pula)
Testa em coerente vs degradado e ve quem separa o que. CPU.
"""
import re, math, gzip, random
from collections import Counter, defaultdict
import numpy as np
random.seed(0); np.random.seed(0)
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt"
RAW = open(CORPUS, encoding="utf-8").read()
TXT = re.sub(r"\s+", " ", RAW)[:3_000_000].lower()
LFc = Counter(TXT); TOT = sum(LFc.values())
LF = {c: math.log((n+1)/TOT) for c, n in LFc.items()}; FLOOR = math.log(1/TOT)
WORDS = re.findall(r"[a-zàáâãéêíóôõúüç]+", TXT)
WUNI = Counter(WORDS); WBI = Counter(zip(WORDS, WORDS[1:])); WV = len(WUNI)
# char ngram (ordem 4) p/ bpc local
NG = defaultdict(Counter)
for i in range(len(TXT)):
    NG[TXT[max(0, i-3):i]][TXT[i]] += 1


def sig(t): return np.array([LF.get(c, FLOOR) for c in t.lower()])
def tok(t): return re.findall(r"[a-zàáâãéêíóôõúüç]+", t.lower())


def gzip_ratio(t):
    b = t.encode("utf-8"); return len(gzip.compress(b, 6))/max(1, len(b))


def dfa(x):
    if len(x) < 32: return 0.5
    y = np.cumsum(x - x.mean()); ns = [4, 8, 16, 32, 64]; Fs = []
    for n in ns:
        if n >= len(y): break
        segs = len(y)//n; rms = []
        for s in range(segs):
            seg = y[s*n:(s+1)*n]; tt = np.arange(n); fit = np.polyval(np.polyfit(tt, seg, 1), tt)
            rms.append(math.sqrt(np.mean((seg-fit)**2)+1e-12))
        Fs.append(np.mean(rms))
    if len(Fs) < 3: return 0.5
    return float(np.polyfit(np.log(ns[:len(Fs)]), np.log(Fs), 1)[0])


def recur_det(x, thr=0.3):
    n = min(len(x), 140); x = x[:n]
    if n < 10: return 0.0
    R = (np.abs(x[:, None]-x[None, :]) < thr*(x.std()+1e-9)).astype(int)
    # determinismo: fracao de pontos recorrentes com vizinho na MESMA diagonal
    det = 0; rec = R.sum()
    for i in range(n-1):
        for j in range(n-1):
            if R[i, j] and R[i+1, j+1]: det += 1
    return round(det/max(1, rec), 3)


def spec_slope(x):
    if len(x) < 16: return 0.0
    F = np.abs(np.fft.rfft(x-x.mean()))[1:]+1e-9; fr = np.arange(1, len(F)+1)
    return float(np.polyfit(np.log(fr), np.log(F), 1)[0])


def wtrans_surp(t):
    w = tok(t)
    if len(w) < 3: return 0.0
    ll = [-math.log((WBI.get((w[i], w[i+1]), 0)+0.05)/(WUNI.get(w[i], 0)+0.05*WV)) for i in range(len(w)-1)]
    return float(np.mean(ll))


def sld_bpc_std(t, win=30):
    if len(t) < win*2: return 0.0
    bpcs = []
    for k in range(0, len(t)-win, win//2):
        seg = t[k:k+win]; sur = []
        for i in range(1, len(seg)):
            d = NG.get(seg[max(0, i-3):i].lower())
            p = (d.get(seg[i].lower(), 0)+0.1)/(sum(d.values())+0.1*256) if d else 1/256
            sur.append(-math.log2(max(p, 1e-9)))
        if sur: bpcs.append(np.mean(sur))
    return round(float(np.std(bpcs)), 2) if bpcs else 0.0


def allm(t):
    s = sig(t)
    return {"gzip": round(gzip_ratio(t), 2), "DFA": round(dfa(s), 2), "recur_det": recur_det(s),
            "spec_slope": round(spec_slope(s), 2), "wtrans": round(wtrans_surp(t), 2), "sld_std": sld_bpc_std(t)}


def main():
    held = TXT[2_500_000:]
    def chunk(): i = random.randint(0, len(held)-700); return held[i:i+600]
    rnd = lambda t: "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in t)
    wsh = lambda t: " ".join(random.sample(t.split(), len(t.split())))
    rep = lambda t: ("o gato e o cachorro "*40)[:len(t)]
    salad = lambda t: " ".join(random.choice(WORDS) for _ in range(len(t.split())))
    csh = lambda t: "".join(random.sample(list(t), len(t)))
    types = {"COERENTE": chunk, "char_shuffle": lambda: csh(chunk()), "word_shuffle": lambda: wsh(chunk()),
             "aleatorio": lambda: rnd(chunk()), "repeticao": lambda: rep(chunk()),
             "salada_palavras": lambda: salad(chunk()), "NOSSO_modelo": lambda: "a historia registra de ignorancia nacional que havia um dos metodos historicoeles aceita de energia possuia e a paz do determinismo cientifico em mil novecentos"}
    print(f"{'tipo':<17}{'gzip':>6}{'DFA':>6}{'recurDet':>9}{'specSlp':>8}{'wtrans':>8}{'sldStd':>7}")
    res = {}
    for name, fn in types.items():
        ms = [allm(fn()) for _ in range(6 if name != "NOSSO_modelo" else 1)]
        agg = {k: round(float(np.mean([m[k] for m in ms])), 3) for k in ms[0]}; res[name] = agg
        print(f"{name:<17}{agg['gzip']:>6}{agg['DFA']:>6}{agg['recur_det']:>9}{agg['spec_slope']:>8}{agg['wtrans']:>8}{agg['sld_std']:>7}")
    print("\n=== quem separa o quê (vs COERENTE) ===")
    coh = res["COERENTE"]
    for tp in types:
        if tp == "COERENTE": continue
        diffs = {k: round(res[tp][k]-coh[k], 2) for k in coh}
        flag = max(diffs, key=lambda k: abs(diffs[k]))
        print(f"  {tp:<16} mais distinto por: {flag} (coh {coh[flag]} vs {res[tp][flag]})")
    print("\nCOERENTE ref:", coh)
    print("aposta DFA: linguagem deve ter alpha alto (~0.7+); aleatorio ~0.5; salada baixa = pega o sentido?")


if __name__ == "__main__":
    main()
