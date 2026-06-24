"""
ByteBrain — BATERIA: existe uma "formula de coesao" pra texto? (alem de frequencia)

Ideia do Leonardo: por que "Oi, como vai?" faz sentido? da pra CALCULAR/aproximar coesao?
Testa varias MEDIDAS contra texto coerente vs degradado, e ve qual separa melhor:
  1. ngram-bpb  : modelo de char n-grama (frequencia pura, nao-neural) -> bits/char
  2. word_cov   : % de palavras que existem em PT (cobertura de lexico)
  3. word_bigLL : log-prob de bigrama de palavras (ordem natural)
  4. neural_bpb : char-GRU treinado em PT (perplexidade neural)
  5. surpr_rhythm: desvio-padrao da surpresa por char (ritmo; coerente tem ritmo)
  6. zipf       : aderencia a lei de Zipf (freq de palavra ~ 1/rank)
  7. composto   : combinacao normalizada
Texto-teste: PT coerente (held-out) + degradacoes (shuffle char/palavra, aleatorio, repeticao,
salada de palavras, lingua-mista, reverso) + frases-mao ("Oi, como vai?" etc) + AMOSTRAS DO NOSSO MODELO.
Mede SEPARACAO (coerente fica acima do lixo?) e onde cada coisa cai. CPU/GPU leve.
"""
import re, math, random, gzip, glob
from collections import Counter, defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CORPUS = "/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt"


def load_pt():
    t = open(CORPUS, encoding="utf-8").read()
    t = re.sub(r"\s+", " ", t)
    return t


# ---------- MEDIDA 1+5: char n-grama interpolado (frequencia) ----------
class CharNgram:
    def __init__(s, order=5):
        s.order = order; s.cnt = [defaultdict(Counter) for _ in range(order)]
    def fit(s, text):
        for i in range(len(text)):
            for o in range(s.order):
                if i-o-1 >= -1:
                    ctx = text[max(0, i-o-1):i]; s.cnt[o][ctx][text[i]] += 1
    def logp(s, ctx, c):
        # interpolacao simples ordens altas->baixas, add-k
        ws = [0.5, 0.25, 0.15, 0.07, 0.03][:s.order]; p = 0.0
        for o in range(s.order):
            sub = ctx[-(o+1):] if o+1 <= len(ctx) else ctx
            d = s.cnt[o].get(sub)
            if d:
                tot = sum(d.values()); p += ws[o]*( (d.get(c, 0)+0.1)/(tot+0.1*256) )
            else:
                p += ws[o]*(1/256)
        return math.log2(max(p, 1e-9))
    def bpc_and_rhythm(s, text):
        sur = []
        for i in range(1, len(text)):
            sur.append(-s.logp(text[max(0, i-s.order):i], text[i]))
        if not sur: return 8.0, 0.0
        return float(np.mean(sur)), float(np.std(sur))


# ---------- MEDIDA 4: char-GRU neural ----------
class GRU(nn.Module):
    def __init__(s, d=128):
        super().__init__(); s.e = nn.Embedding(256, d); s.g = nn.GRU(d, d, 2, batch_first=True); s.o = nn.Linear(d, 256)
    def forward(s, x, h=None):
        y, h = s.g(s.e(x), h); return s.o(y), h


def train_gru(text, steps=1500):
    b = np.frombuffer(text.encode("utf-8"), np.uint8); L = 192
    m = GRU().to(DEV); opt = torch.optim.Adam(m.parameters(), 2e-3); m.train()
    for _ in range(steps):
        ix = np.random.randint(0, len(b)-L-1, 32)
        x = torch.from_numpy(np.stack([b[i:i+L] for i in ix]).astype(np.int64)).to(DEV)
        lo, _ = m(x[:, :-1]); loss = F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m


def gru_bpb(m, s):
    b = list(s.encode("utf-8"))[:240]
    if len(b) < 4: return 8.0
    x = torch.tensor([b], device=DEV)
    with torch.no_grad():
        lo, _ = m(x[:, :-1]); return F.cross_entropy(lo.reshape(-1, 256), x[:, 1:].reshape(-1).to(DEV)).item()/math.log(2)


# ---------- MEDIDAS lexicais ----------
def tok(s): return re.findall(r"[a-zàáâãéêíóôõúüç]+", s.lower())


def main():
    pt = load_pt(); n = len(pt)
    train = pt[:int(n*0.8)]; held = pt[int(n*0.8):]
    print(f"PT corpus {n/1e6:.1f}MB", flush=True)
    # fit medidas
    print("fitando char n-grama...", flush=True); cn = CharNgram(5); cn.fit(train[:3_000_000])
    print("vocab + bigrama de palavra...", flush=True)
    words = tok(train[:3_000_000]); WV = set(w for w, _ in Counter(words).most_common(40000))
    bigc = Counter(zip(words, words[1:])); unic = Counter(words)
    def word_cov(s):
        w = tok(s); return 100*sum(x in WV for x in w)/max(1, len(w))
    def word_bigLL(s):
        w = tok(s)
        if len(w) < 2: return 0.0
        ll = [math.log((bigc.get((w[i], w[i+1]), 0)+0.1)/(unic.get(w[i], 0)+0.1*len(WV))) for i in range(len(w)-1)]
        return float(np.mean(ll))
    def zipf(s):
        w = tok(s); c = Counter(w).most_common(40)
        if len(c) < 8: return 0.0
        r = np.log(np.arange(1, len(c)+1)); f = np.log([x[1] for x in c])
        return float(np.polyfit(r, f, 1)[0])   # ~ -1 ideal
    print("treinando char-GRU neural (PT)...", flush=True); gru = train_gru(train[:2_000_000])

    # degradacoes
    def chunk(): i = random.randint(0, len(held)-600); return held[i:i+500]
    def char_shuf(t): l = list(t); random.shuffle(l); return "".join(l)
    def word_shuf(t): w = t.split(); random.shuffle(w); return " ".join(w)
    def rnd(t): return "".join(random.choice("abcdefghijklmnopqrstuvwxyz ,.") for _ in t)
    def rep(t): return ("casa o de "*60)[:len(t)]
    def salad(t): return " ".join(random.choice(list(WV)) for _ in range(len(t.split())))
    def mixed(t): w = t.split(); return " ".join(w[i] if i%2 else random.choice(["the","Россия","日本","function"]) for i in range(len(w)))

    types = {"COERENTE": [chunk() for _ in range(8)],
             "char_shuffle": [char_shuf(chunk()) for _ in range(6)],
             "word_shuffle": [word_shuf(chunk()) for _ in range(6)],
             "aleatorio": [rnd(chunk()) for _ in range(6)],
             "repeticao": [rep(chunk()) for _ in range(6)],
             "salada_palavras": [salad(chunk()) for _ in range(6)],
             "lingua_mista": [mixed(chunk()) for _ in range(6)]}
    hand = {"Oi, como vai?": "Oi, como vai?", "Ola tudo bem?": "Olá, tudo bem com você?",
            "frase_boa": "O gato subiu no telhado e dormiu ao sol da tarde.",
            "lixo_grafia": "asdkjh qwe zxcv bnm poiu",
            "repeticao_pura": "casa casa casa casa casa casa",
            "ordem_errada": "vai como você Oi tudo?",
            "stopwords_soltas": "que para uma de com por mais como",
            "NOSSO_PT_specialist": "A história registra de ignorância nacional que havia um dos métodos historicoeles, aceita de energia possuía e a da paz do determinismo científico.",
            "NOSSO_generalista_mix": "O Brasil does de licenses and limitatos estados of para poor le patos energon para resp"}

    def measure(s):
        bpc, rhy = cn.bpc_and_rhythm(s)
        return {"ngram_bpc": round(bpc, 2), "word_cov%": round(word_cov(s), 0), "word_bigLL": round(word_bigLL(s), 2),
                "neural_bpb": round(gru_bpb(gru, s), 2), "rhythm_std": round(rhy, 2), "zipf": round(zipf(s), 2)}

    print("\n=== MEDIDAS por tipo (media) — coerente vs degradado ===")
    print(f"{'tipo':<18}{'ngramBPC':>9}{'wordcov':>8}{'wbigLL':>8}{'neurBPB':>8}{'rhythm':>7}")
    base = {}
    for tp, texts in types.items():
        ms = [measure(t) for t in texts]
        agg = {k: round(float(np.mean([m[k] for m in ms])), 2) for k in ms[0]}
        base[tp] = agg
        print(f"{tp:<18}{agg['ngram_bpc']:>9}{agg['word_cov%']:>8.0f}{agg['word_bigLL']:>8}{agg['neural_bpb']:>8}{agg['rhythm_std']:>7}")

    print("\n=== FRASES-MAO + NOSSO MODELO (coeso fica perto de COERENTE?) ===")
    coh = base["COERENTE"]
    for name, s in hand.items():
        m = measure(s)
        print(f"  {name:<22} ngramBPC {m['ngram_bpc']:>5} | wordcov {m['word_cov%']:>3.0f}% | wbigLL {m['word_bigLL']:>6} | neurBPB {m['neural_bpb']:>5}")

    # SEPARACAO: ngram_bpc e neural_bpb do COERENTE vs media degradado
    deg = [base[t] for t in types if t != "COERENTE"]
    print("\n=== SEPARACAO (coerente vs lixo) ===")
    for k in ["ngram_bpc", "neural_bpb", "word_cov%", "word_bigLL"]:
        cval = coh[k]; dval = float(np.mean([d[k] for d in deg]))
        print(f"  {k:<12}: coerente {cval:>6} | lixo medio {dval:>6} | gap {abs(cval-dval):>6.2f}")
    print("\n-> menor ngram_bpc/neural_bpb = mais coeso; maior word_cov/word_bigLL = mais coeso.")


if __name__ == "__main__":
    main()
