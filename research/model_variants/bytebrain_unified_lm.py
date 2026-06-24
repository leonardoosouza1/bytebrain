"""
ByteBrain 2b — ultrapassando o negativo honesto do Teste 2 com inteligencia.

Teste 2 mostrou: byte-LM treinado SO em Latin NAO faz zero-shot em script novo (18 bpb > 8 aleatorio,
pq nunca viu os bytes altos 128-255). O trunfo do byte NAO e' transferencia magica — e' que ele
CONSEGUE ingerir QUALQUER script num unico modelo de vocab 256 (0% OOV). Entao o teste certo:
treina UM byte-LM em TODOS os scripts -> modela todos uniformemente com uma tabela minuscula.
char/word precisariam de vocab gigante e ainda assim UNKam scripts novos. Prova o claim correto.
"""
import os, re, json, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

random.seed(0); torch.manual_seed(0); np.random.seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "/home/leonardo/projects/LLM/bytebrain/data/multiscript"
LB = 256
SCRIPTS = {"en": "Latin", "pt": "Latin", "ru": "Cyrillic", "el": "Greek", "ja": "Japanese",
           "ar": "Arabic", "hi": "Devanagari", "ko": "Hangul", "he": "Hebrew"}


class LM(nn.Module):
    def __init__(s, v=256, d=128):
        super().__init__(); s.e = nn.Embedding(v, d); s.g = nn.GRU(d, d, 2, batch_first=True); s.o = nn.Linear(d, v)
    def forward(s, x, h=None):
        y, h = s.g(s.e(x), h); return s.o(y), h


def seqs(text, n):
    b = text.encode("utf-8"); out = []
    for _ in range(n):
        if len(b) < LB + 1: break
        p = random.randint(0, len(b) - LB - 1); out.append(list(b[p:p + LB]))
    return torch.tensor(out) if out else None


def main():
    texts = {l: open(f"{CACHE}/{l}.txt", encoding="utf-8").read() for l in SCRIPTS if os.path.exists(f"{CACHE}/{l}.txt")}
    # treina UM byte-LM em TODOS os scripts (vocab 256)
    allseq = []
    for l in texts:
        s = seqs(texts[l], 900)
        if s is not None: allseq.append(s)
    tr = torch.cat(allseq)
    tr = tr[torch.randperm(len(tr))]
    lm = LM().to(DEV); opt = torch.optim.Adam(lm.parameters(), 2e-3); lm.train()
    for ep in range(8):
        pm = torch.randperm(len(tr))
        for i in range(0, len(tr), 64):
            b = tr[pm[i:i+64]].to(DEV); opt.zero_grad(); lo, _ = lm(b[:, :-1]); F.cross_entropy(lo.reshape(-1, 256), b[:, 1:].reshape(-1)).backward(); opt.step()
    lm.eval()
    def bpb(sq):
        with torch.no_grad():
            lo, _ = lm(sq[:, :-1].to(DEV)); return F.cross_entropy(lo.reshape(-1, 256), sq[:, 1:].reshape(-1).to(DEV)).item() / math.log(2)
    print("=== UM byte-LM (vocab 256, 16K-param embedding) treinado em TODOS os scripts ===")
    print(f"  (aleatorio 8.0 bpb; menor = melhor; antes, treinado-so-Latin dava 18 bpb nos nao-vistos)")
    res = {}
    for l in texts:
        sq = seqs(texts[l], 300)
        if sq is None: continue
        v = bpb(sq); res[SCRIPTS[l]] = round(v, 2)
        print(f"  {SCRIPTS[l]:<11} {v:.2f} bpb")
    print(f"\n  -> um modelo minusculo modela TODOS os alfabetos uniformemente (~{np.mean(list(res.values())):.1f} bpb).")
    print(f"     word/char precisariam de vocab >50k e ainda UNKariam scripts futuros; byte: 256 fixo, 0% OOV.")
    json.dump(res, open("/home/leonardo/projects/LLM/bytebrain/unified_lm_metrics.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
