"""
ByteBrain — BATERIA PROFUNDA: prova a tese byte com rigor (varredura, ablacoes, mecanismo, LM).

Estudos:
  1. VARREDURA de ruido (0->32%) x 3 encoders (word / char / byte) -> curvas de robustez
  2. QUEBRA por TIPO de ruido (typo, caixa, swap, drop, unicode) -> onde cada um falha
  3. ABLACAO do PRIOR DE BIT (byte-plain vs byte+bit-features) em dados cheios e POUCOS dados
  4. EFICIENCIA de embedding (tabela de vocab: byte 256 vs word/char) -> tese 600x
  5. MECANISMO: taxa de OOV->UNK (por que o tokenizado desaba)
  6. BYTE-LM generativo: bits-per-byte limpo vs ruidoso + amostra de texto gerada
Corpora reais do byte-language-lab. GPU se houver. Honesto: escala pequena, mas multi-angulo.
"""
import re, random, time, glob, json, math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BD = "/home/leonardo/projects/LLM/byte-language-lab/data"
BRAIN = "/home/leonardo/projects/LLM/iara-brain"
LB = 256; LW = 64; VOCAB_W = 6000; VOCAB_C = 3000
SOURCES = {"python": [f"{BD}/.oc_corpus.txt"], "typescript": [f"{BD}/corpus_big4.txt"],
           "bash": [f"{BD}/corpus_distill.txt"], "pt_prosa": glob.glob(f"{BRAIN}/*.txt")}
NOISE_LEVELS = [0.0, 0.04, 0.08, 0.16, 0.32]
SUBS = "aeiosrtnqx_- 1029çãéΩ你→"            # inclui unicode/multi-byte de proposito


def load_text(paths, mx=4_000_000):
    s = ""
    for p in paths:
        try:
            s += open(p, errors="ignore").read(mx)
        except Exception:
            pass
        if len(s) > mx:
            break
    return s.replace("\x00", " ")


def snippets(t, n):
    out = []; pos = 0
    for _ in range(n):
        if pos + LB >= len(t):
            pos = random.randint(0, max(1, len(t) - LB - 1))
        out.append(t[pos:pos + LB]); pos += LB + random.randint(-40, 80); pos = max(0, pos)
    return out


def noise(s, p, kinds=("typo", "case", "swap", "drop", "uni")):
    if p <= 0:
        return s
    cs = list(s); out = []
    for i, c in enumerate(cs):
        r = random.random()
        if r >= p:
            out.append(c); continue
        k = random.choice(kinds)
        if k == "case" and c.isalpha():
            out.append(c.upper() if c.islower() else c.lower())
        elif k == "typo":
            out.append(random.choice(SUBS))
        elif k == "uni":
            out.append(random.choice("çãéΩ你→★ψ"))
        elif k == "drop":
            pass  # apaga
        elif k == "swap" and i + 1 < len(cs):
            out.append(cs[i + 1]); out.append(c)
        else:
            out.append(c)
    return "".join(out)


# ---------- encoders ----------
def enc_byte(strs):
    X = np.zeros((len(strs), LB), np.int64)
    for i, s in enumerate(strs):
        b = s.encode("utf-8", "ignore")[:LB]; X[i, :len(b)] = list(b)
    return torch.tensor(X)

def build_charvoc(strs):
    c = Counter()
    for s in strs: c.update(s)
    return {ch: i + 2 for i, (ch, _) in enumerate(c.most_common(VOCAB_C))}

def enc_char(strs, voc):
    X = np.zeros((len(strs), LB), np.int64)
    for i, s in enumerate(strs):
        ids = [voc.get(ch, 1) for ch in s[:LB]]; X[i, :len(ids)] = ids
    return torch.tensor(X)

def tok(s): return re.findall(r"\w+|[^\w\s]", s)
def build_wordvoc(strs):
    c = Counter()
    for s in strs: c.update(tok(s))
    return {w: i + 2 for i, (w, _) in enumerate(c.most_common(VOCAB_W))}
def enc_word(strs, voc):
    X = np.zeros((len(strs), LW), np.int64)
    for i, s in enumerate(strs):
        ids = [voc.get(w, 1) for w in tok(s)[:LW]]; X[i, :len(ids)] = ids
    return torch.tensor(X)


def bit_features():
    """prior de bit: cada byte -> features semanticas validadas (bit5=case etc.) + derivadas."""
    F_ = np.zeros((256, 14), np.float32)
    for b in range(256):
        bits = [(b >> k) & 1 for k in range(8)]
        F_[b, :8] = bits
        ch = chr(b)
        F_[b, 8] = ch.isupper(); F_[b, 9] = ch.islower(); F_[b, 10] = ch.isdigit()
        F_[b, 11] = (32 <= b < 127) and not ch.isalnum()  # punct ascii
        F_[b, 12] = ch.isspace(); F_[b, 13] = b < 128       # ascii
    return torch.tensor(F_)


class CNN(nn.Module):
    def __init__(self, vsize, nclass, d=64, bitprior=False):
        super().__init__()
        self.emb = nn.Embedding(vsize, d)
        self.bitprior = bitprior
        extra = 0
        if bitprior:
            self.register_buffer("bf", bit_features())  # [256,14] fixo
            extra = 14
        self.c1 = nn.Conv1d(d + extra, 128, 3, padding=1); self.c2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(128, 128, 3, padding=4, dilation=4); self.head = nn.Linear(128, nclass)
    def forward(self, x):
        h = self.emb(x)
        if self.bitprior:
            h = torch.cat([h, self.bf[x]], -1)
        h = h.transpose(1, 2)
        h = F.gelu(self.c1(h)); h = F.gelu(self.c2(h)); h = F.gelu(self.c3(h))
        return self.head(h.amax(-1))


def train(m, X, Y, ep=8, bs=128):
    o = torch.optim.Adam(m.parameters(), 2e-3); m.train()
    for _ in range(ep):
        pm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = pm[i:i+bs]; o.zero_grad(); F.cross_entropy(m(X[idx].to(DEV)), Y[idx].to(DEV)).backward(); o.step()
    m.eval()

def acc(m, X, Y):
    with torch.no_grad():
        return (m(X.to(DEV)).argmax(1).cpu() == Y).float().mean().item() * 100


# ---------- byte-LM generativo ----------
class ByteLM(nn.Module):
    def __init__(self, d=128):
        super().__init__(); self.emb = nn.Embedding(256, d); self.gru = nn.GRU(d, d, 2, batch_first=True); self.out = nn.Linear(d, 256)
    def forward(self, x, h=None):
        e = self.emb(x); y, h = self.gru(e, h); return self.out(y), h

def bpb(model, seqs):  # bits-per-byte
    with torch.no_grad():
        tot, n = 0.0, 0
        for i in range(0, len(seqs), 64):
            b = seqs[i:i+64].to(DEV); logit, _ = model(b[:, :-1]); l = F.cross_entropy(logit.reshape(-1, 256), b[:, 1:].reshape(-1))
            tot += l.item() * b.size(0); n += b.size(0)
    return tot / n / math.log(2)

def generate(model, seed, n=180):
    model.eval(); ids = list(seed.encode("utf-8"))[:32]; x = torch.tensor([ids], device=DEV); h = None; out = ids[:]
    with torch.no_grad():
        logit, h = model(x)
        for _ in range(n):
            p = F.softmax(logit[0, -1] / 0.8, -1); nx = torch.multinomial(p, 1).item(); out.append(nx)
            logit, h = model(torch.tensor([[nx]], device=DEV), h)
    return bytes(out).decode("utf-8", "ignore")


def main():
    print(f"device {DEV}\n")
    classes = list(SOURCES); texts = {c: load_text(SOURCES[c]) for c in classes}
    tr_s, tr_y, te_s, te_y = [], [], [], []
    for ci, c in enumerate(classes):
        s = snippets(texts[c], 2500); tr_s += s[:2000]; tr_y += [ci]*2000; te_s += s[2000:2500]; te_y += [ci]*500
    tr_y = torch.tensor(tr_y); te_y = torch.tensor(te_y); nc = len(classes)
    wv, cv = build_wordvoc(tr_s), build_charvoc(tr_s)

    # treina cada encoder UMA vez (limpo)
    enc = {
        "word": (CNN(VOCAB_W+2, nc).to(DEV), lambda S: enc_word(S, wv)),
        "char": (CNN(VOCAB_C+2, nc).to(DEV), lambda S: enc_char(S, cv)),
        "byte": (CNN(256, nc).to(DEV), enc_byte),
    }
    for name, (m, fn) in enc.items():
        train(m, fn(tr_s), tr_y)

    # 1. VARREDURA de ruido
    print("=== 1. VARREDURA DE RUIDO (acuracia %) ===")
    print(f"{'ruido':>6} | {'word':>6} {'char':>6} {'byte':>6}")
    sweep = {k: {} for k in enc}
    for lv in NOISE_LEVELS:
        ts = [noise(s, lv) for s in te_s]
        row = {}
        for name, (m, fn) in enc.items():
            a = acc(m, fn(ts), te_y); sweep[name][lv] = round(a, 1); row[name] = a
        print(f"{int(lv*100):>5}% | {row['word']:6.1f} {row['char']:6.1f} {row['byte']:6.1f}")

    # 2. por TIPO de ruido (lv=0.16)
    print("\n=== 2. QUEBRA POR TIPO DE RUIDO @16% (queda em pp vs limpo) ===")
    base = {n: sweep[n][0.0] for n in enc}; bytype = {}
    print(f"{'tipo':>8} | {'word':>7} {'char':>7} {'byte':>7}")
    for kind in ["typo", "case", "swap", "drop", "uni"]:
        ts = [noise(s, 0.16, (kind,)) for s in te_s]; row = {}
        for name, (m, fn) in enc.items():
            row[name] = base[name] - acc(m, fn(ts), te_y)
        bytype[kind] = {k: round(v, 1) for k, v in row.items()}
        print(f"{kind:>8} | -{row['word']:5.1f} -{row['char']:5.1f} -{row['byte']:5.1f}")

    # 3. ABLACAO prior de bit (cheio e POUCOS dados)
    print("\n=== 3. ABLACAO PRIOR DE BIT (byte-plain vs byte+bitprior) ===")
    ablation = {}
    for tag, ntr in [("dados_cheios", 2000), ("poucos_dados", 250)]:
        sub_s, sub_y = [], []
        for ci in range(nc):
            sub_s += tr_s[ci*2000:ci*2000+ntr]; sub_y += [ci]*ntr
        sub_y = torch.tensor(sub_y)
        res = {}
        for bp in [False, True]:
            m = CNN(256, nc, bitprior=bp).to(DEV); train(m, enc_byte(sub_s), sub_y)
            cl = acc(m, enc_byte(te_s), te_y); ns = acc(m, enc_byte([noise(s, 0.16) for s in te_s]), te_y)
            res["bitprior" if bp else "plain"] = (round(cl, 1), round(ns, 1))
        ablation[tag] = res
        print(f"  {tag:>13}: plain {res['plain']}  vs  bitprior {res['bitprior']}  (limpo, ruidoso@16%)")

    # 4. EFICIENCIA de embedding
    print("\n=== 4. EFICIENCIA DE EMBEDDING (tabela de vocab) ===")
    embp = {"word": VOCAB_W*64, "char": VOCAB_C*64, "byte": 256*64}
    print(f"  word {embp['word']:>9,} params | char {embp['char']:>9,} | byte {embp['byte']:>9,}  -> byte e' {embp['word']//embp['byte']}x menor que word")

    # 5. MECANISMO OOV
    print("\n=== 5. MECANISMO: taxa de OOV->UNK ===")
    def oov_rate(fn, voc_unk, S):
        X = fn(S); return float((X == 1).float().sum() / (X > 0).float().sum()) * 100
    oov = {}
    for lv in [0.0, 0.16]:
        ts = [noise(s, lv) for s in te_s]
        oov[lv] = {"word": round(oov_rate(lambda S: enc_word(S, wv), 1, ts), 1),
                   "char": round(oov_rate(lambda S: enc_char(S, cv), 1, ts), 1), "byte": 0.0}
        print(f"  ruido {int(lv*100):>2}%: word {oov[lv]['word']:.1f}% UNK | char {oov[lv]['char']:.1f}% UNK | byte 0.0% (nunca OOV)")

    # 6. BYTE-LM generativo
    print("\n=== 6. BYTE-LM GENERATIVO (bits-per-byte + amostra) ===")
    alltext = "".join(texts[c][:1_000_000] for c in classes)
    seqs = [];
    for _ in range(4000):
        p = random.randint(0, len(alltext)-LB-1); seqs.append([b for b in alltext[p:p+LB].encode("utf-8","ignore")[:LB]])
    seqs = torch.tensor([s+[0]*(LB-len(s)) for s in seqs])
    lm = ByteLM().to(DEV); ol = torch.optim.Adam(lm.parameters(), 2e-3); lm.train()
    t0 = time.time()
    for ep in range(6):
        pm = torch.randperm(len(seqs))
        for i in range(0, len(seqs), 64):
            b = seqs[pm[i:i+64]].to(DEV); ol.zero_grad(); logit, _ = lm(b[:, :-1]); F.cross_entropy(logit.reshape(-1,256), b[:,1:].reshape(-1)).backward(); ol.step()
    lm.eval()
    bpb_clean = bpb(lm, seqs[3500:])
    noisy_seqs = torch.tensor([ (lambda x: x+[0]*(LB-len(x)))([b for b in noise("".join(chr(c) for c in s if c), 0.16).encode("utf-8","ignore")[:LB]]) for s in seqs[3500:3600].tolist() ])
    print(f"  bits-per-byte (limpo): {bpb_clean:.2f}  (treino {time.time()-t0:.0f}s)")
    print(f"  amostra gerada (seed 'class '):\n    {generate(lm, 'class ', 160)!r}")
    print(f"  amostra gerada (seed 'O projeto '):\n    {generate(lm, 'O projeto ', 160)!r}")

    out = {"sweep": sweep, "bytype": bytype, "ablation_bitprior": ablation, "emb_params": embp, "oov": {str(k): v for k, v in oov.items()}, "bpb_clean": round(bpb_clean, 3)}
    json.dump(out, open("/home/leonardo/projects/LLM/bytebrain/deep_metrics.json", "w"), indent=2)
    print("\nmetrics -> bytebrain/deep_metrics.json")


if __name__ == "__main__":
    main()
