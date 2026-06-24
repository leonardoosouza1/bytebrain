"""
ByteBrain — modelo LEVE de coesao (ideia do Leonardo): classificador byte-level que APRENDE
coesao (coerente=1 vs degradado=0) em comprimentos VARIADOS -> funciona em texto curto E longo,
onde a formula-na-mao falhava. Testa se acerta "Oi, como vai?" (curto) e o lixo curto.
"""
import re, random, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TXT = re.sub(r"\s+", " ", open("/home/leonardo/projects/LLM/bytebrain/data/pt_overnight.txt", encoding="utf-8").read())
W = re.findall(r"[a-zàáâãéêíóôõúüç]+", TXT.lower())
L = 320


def degrade(s):
    k = random.choice(["cshuf", "wshuf", "rnd", "rep", "salad", "mixed"])
    w = s.split()
    if k == "cshuf": return "".join(random.sample(list(s), len(s)))
    if k == "wshuf" and len(w) > 1: return " ".join(random.sample(w, len(w)))
    if k == "rnd": return "".join(random.choice("abcdefghijklmnopqrstuvwxyz ,.") for _ in s)
    if k == "rep": return ((random.choice(W)+" ")*40)[:len(s)]
    if k == "salad": return " ".join(random.choice(W) for _ in range(max(2, len(w))))
    if k == "mixed": return " ".join(x if random.random() < .5 else random.choice(["the", "Россия", "日本", "code", "der"]) for x in (w or ["a"]))
    return "".join(random.sample(list(s), len(s)))


def enc(s):
    b = s.encode("utf-8")[:L]; a = np.zeros(L, np.int64); a[:len(b)] = list(b); return a


def sample(n):
    X, Y = [], []
    for _ in range(n):
        ln = random.randint(15, 300); i = random.randint(0, len(TXT)-ln-1); c = TXT[i:i+ln]
        X.append(enc(c)); Y.append(1)
        X.append(enc(degrade(c))); Y.append(0)
    return torch.tensor(np.stack(X)), torch.tensor(Y, dtype=torch.float32)


class CohModel(nn.Module):
    def __init__(s, d=48):
        super().__init__(); s.e = nn.Embedding(256, d)
        s.c1 = nn.Conv1d(d, 96, 3, padding=1); s.c2 = nn.Conv1d(96, 96, 3, padding=2, dilation=2)
        s.c3 = nn.Conv1d(96, 96, 3, padding=4, dilation=4); s.h = nn.Linear(96, 1)
    def forward(s, x):
        h = s.e(x).transpose(1, 2); h = F.gelu(s.c1(h)); h = F.gelu(s.c2(h)); h = F.gelu(s.c3(h)); return s.h(h.amax(-1)).squeeze(-1)


def main():
    Xtr, Ytr = sample(5000); Xte, Yte = sample(1200)
    m = CohModel().to(DEV); opt = torch.optim.Adam(m.parameters(), 2e-3)
    print(f"params {sum(p.numel() for p in m.parameters()):,} | treino...", flush=True)
    m.train()
    for ep in range(14):
        pm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = pm[i:i+64]; opt.zero_grad(); F.binary_cross_entropy_with_logits(m(Xtr[idx].to(DEV)), Ytr[idx].to(DEV)).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = ((torch.sigmoid(m(Xte.to(DEV))) > .5).float().cpu() == Yte).float().mean().item()
    print(f"acuracia coerente-vs-lixo (held-out, varios tamanhos): {acc*100:.1f}%")

    def score(s):
        with torch.no_grad():
            return round(torch.sigmoid(m(torch.tensor([enc(s)]).to(DEV))).item()*100, 1)
    print("\n=== TEXTO CURTO (onde a formula-na-mao falhava) ===")
    hand = {"Oi, como vai?": "Oi, como vai?", "Olá tudo bem?": "Olá, tudo bem com você hoje?",
            "frase boa curta": "O gato subiu no telhado.", "bom dia": "Bom dia, espero que tenha um ótimo dia!",
            "asdkjh qwe (lixo)": "asdkjh qwe zxcv bnm", "casa casa (rep)": "casa casa casa casa casa",
            "que para uma (stop)": "que para uma de com por", "ordem errada": "vai como você oi tudo",
            "salada curta": "telhado energia que político uma cachorro determinismo"}
    for n, s in hand.items():
        print(f"  {n:<22} coesao {score(s):>6}%")
    print("\n=== NOSSO MODELO ===")
    for n, s in {"PT-spec": "a historia registra de ignorancia nacional que havia um dos metodos historicoeles aceita de energia",
                 "generalista mix": "O Brasil does de licenses and limitatos estados of para poor le patos energon para republica"}.items():
        print(f"  {n:<22} coesao {score(s):>6}%")


if __name__ == "__main__":
    main()
