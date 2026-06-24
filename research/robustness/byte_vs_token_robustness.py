"""
ByteBrain — teste-BASE: byte-level e' ROBUSTO a ruido/OOV onde o tokenizado QUEBRA.

A tese-mae do ByteBrain: modelo SEM tokenizer (256 bytes) entende qualquer script/codigo/lingua
e aguenta texto corrompido (typo, caixa, simbolo) — enquanto o word/BPE vira OOV->UNK e desaba.
Aqui validamos isso LOCAL e BARATO antes de gastar credito AMD: tarefa = classificar snippet em
4 classes (Python / TypeScript / bash / prosa PT-BR) a partir dos corpora reais do byte-language-lab.
Treina LIMPO; avalia LIMPO vs RUIDOSO. Mesma cabeca CNN nos dois; muda so a entrada (byte vs word).
Se o byte degrada pouco e o word desaba -> a tese-base esta validada -> vale escalar na AMD.
"""
import re, random, time, glob, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BD = "/home/leonardo/projects/LLM/byte-language-lab/data"
BRAIN = "/home/leonardo/projects/LLM/iara-brain"
LB = 256          # bytes por snippet
LW = 64           # tokens por snippet (word-level)
VOCAB = 6000      # vocab word-level (top-K) ; resto -> UNK

SOURCES = {  # classe -> arquivos (codigo e prosa reais e heterogeneos)
    "python": [f"{BD}/.oc_corpus.txt"],
    "typescript": [f"{BD}/corpus_big4.txt"],
    "bash": [f"{BD}/corpus_distill.txt"],
    "pt_prosa": glob.glob(f"{BRAIN}/*.txt"),
}


def load_text(paths, maxbytes=4_000_000):
    s = ""
    for p in paths:
        try:
            with open(p, "r", errors="ignore") as f:
                s += f.read(maxbytes)
        except Exception:
            pass
        if len(s) > maxbytes:
            break
    return s


def snippets(text, n, stride_jitter=True):
    out = []; pos = 0; step = LB
    txt = text.replace("\x00", " ")
    for _ in range(n):
        if pos + LB >= len(txt):
            pos = random.randint(0, max(1, len(txt) - LB - 1))
        out.append(txt[pos:pos + LB])
        pos += step + (random.randint(-40, 80) if stride_jitter else 0)
        pos = max(0, pos)
    return out


def add_noise(s, p=0.08):
    """ruido realista: troca de char, caixa, swap, simbolo — o que typo/OCR/usuario faz."""
    cs = list(s)
    for i in range(len(cs)):
        r = random.random()
        if r < p * 0.4 and cs[i].isalpha():
            cs[i] = cs[i].upper() if cs[i].islower() else cs[i].lower()       # caixa
        elif r < p * 0.7:
            cs[i] = random.choice("aeiosrtnలqx_- 1029")                       # substituicao
        elif r < p and i + 1 < len(cs):
            cs[i], cs[i+1] = cs[i+1], cs[i]                                    # swap vizinho
    return "".join(cs)


# ---------- encoders ----------
def enc_bytes(strs):
    X = np.zeros((len(strs), LB), np.int64)
    for i, s in enumerate(strs):
        b = s.encode("utf-8", "ignore")[:LB]; X[i, :len(b)] = list(b)
    return torch.tensor(X)


def tok(s):
    return re.findall(r"\w+|[^\w\s]", s)


def build_vocab(strs):
    from collections import Counter
    c = Counter()
    for s in strs:
        c.update(tok(s))
    voc = {w: i + 2 for i, (w, _) in enumerate(c.most_common(VOCAB))}   # 0=PAD 1=UNK
    return voc


def enc_words(strs, voc):
    X = np.ones((len(strs), LW), np.int64)  # PAD=0? usamos 1=UNK pad-> 0
    X[:] = 0
    for i, s in enumerate(strs):
        ids = [voc.get(w, 1) for w in tok(s)[:LW]]
        X[i, :len(ids)] = ids
    return torch.tensor(X)


class CNN(nn.Module):
    def __init__(self, vsize, nclass, d=64):
        super().__init__()
        self.emb = nn.Embedding(vsize, d)
        self.c1 = nn.Conv1d(d, 128, 3, padding=1); self.c2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.c3 = nn.Conv1d(128, 128, 3, padding=4, dilation=4); self.head = nn.Linear(128, nclass)
    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        h = F.gelu(self.c1(h)); h = F.gelu(self.c2(h)); h = F.gelu(self.c3(h))
        return self.head(h.amax(-1))


def train(model, X, Y, epochs=8, bs=128):
    opt = torch.optim.Adam(model.parameters(), 2e-3); model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i+bs]
            opt.zero_grad(); F.cross_entropy(model(X[idx].to(DEV)), Y[idx].to(DEV)).backward(); opt.step()
    model.eval()


def acc(model, X, Y):
    with torch.no_grad():
        pred = model(X.to(DEV)).argmax(1).cpu()
    return (pred == Y).float().mean().item() * 100


def main():
    print(f"device: {DEV}")
    classes = list(SOURCES); texts = {c: load_text(SOURCES[c]) for c in classes}
    for c in classes:
        print(f"  {c}: {len(texts[c]):,} chars")
    tr_s, tr_y, te_s, te_y = [], [], [], []
    for ci, c in enumerate(classes):
        s = snippets(texts[c], 2500)
        tr_s += s[:2000]; tr_y += [ci] * 2000; te_s += s[2000:2500]; te_y += [ci] * 500
    tr_y = torch.tensor(tr_y); te_y = torch.tensor(te_y)
    te_noisy = [add_noise(s) for s in te_s]

    # BYTE
    Xb_tr = enc_bytes(tr_s); Xb_te = enc_bytes(te_s); Xb_ten = enc_bytes(te_noisy)
    byte = CNN(256, len(classes)).to(DEV)
    t0 = time.time(); train(byte, Xb_tr, tr_y); tb = time.time() - t0
    b_clean, b_noisy = acc(byte, Xb_te, te_y), acc(byte, Xb_ten, te_y)

    # WORD (BPE-like fragil: OOV->UNK)
    voc = build_vocab(tr_s)
    Xw_tr = enc_words(tr_s, voc); Xw_te = enc_words(te_s, voc); Xw_ten = enc_words(te_noisy, voc)
    word = CNN(VOCAB + 2, len(classes)).to(DEV)
    train(word, Xw_tr, tr_y); w_clean, w_noisy = acc(word, Xw_te, te_y), acc(word, Xw_ten, te_y)
    # taxa de OOV no ruidoso (porque o word desaba)
    oov = float(np.mean([(Xw_ten == 1).float().sum().item()]) / max(1, (Xw_ten > 0).float().sum().item())) * 100

    print("\n=== ByteBrain TESTE-BASE: robustez byte vs word (4 classes lang/codigo) ===")
    print(f"params byte {sum(p.numel() for p in byte.parameters()):,} | word {sum(p.numel() for p in word.parameters()):,}")
    print(f"\n              LIMPO    RUIDOSO   QUEDA")
    print(f"  WORD/BPE :  {w_clean:5.1f}%   {w_noisy:5.1f}%   -{w_clean-w_noisy:.1f}pp")
    print(f"  BYTE     :  {b_clean:5.1f}%   {b_noisy:5.1f}%   -{b_clean-b_noisy:.1f}pp")
    print(f"\n  vantagem do BYTE sob ruido: +{b_noisy-w_noisy:.1f}pp")
    print(f"  (OOV->UNK no ruidoso, word-level: ~{oov:.0f}% dos tokens viram UNK)")
    verdict = "VALIDADO: byte e' o substrato robusto" if (b_noisy - w_noisy) > 5 else "inconclusivo"
    print(f"  >>> {verdict}  | treino byte {tb:.0f}s")


if __name__ == "__main__":
    main()
