#!/usr/bin/env python3
"""IARA byte-router — a TINY learned byte-level classifier that reads a query and
routes it to a domain specialist, with a CONFIDENCE score. Low confidence = "ninguém
sabe" → escalate to the Qwen generalist (the flywheel trigger). Trains on synthetic
labeled queries. This is the 'R' (Roteamento Adaptativo) of IARA, learned not heuristic."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random, sys

DOMAINS = ["fato", "codigo", "definicao", "math", "geral"]
DI = {d: i for i, d in enumerate(DOMAINS)}
random.seed(0)

PAISES = ["Brasil", "França", "Itália", "Japão", "Portugal", "Argentina", "Espanha", "Alemanha", "Rússia",
          "China", "Egito", "México", "Canadá", "Chile", "Peru", "Colômbia", "Grécia", "Índia", "Austrália"]
TERMOS = ["recursão", "variável", "função", "array", "objeto", "herança", "loop", "algoritmo", "compilador",
          "polimorfismo", "ponteiro", "thread", "cache", "API", "banco de dados", "rede neural"]
COISAS = ["fatorial", "fibonacci", "uma pilha", "uma fila", "uma classe Pessoa", "um contador",
          "uma função que soma uma lista", "um verificador de primos", "uma calculadora"]
CIENCIA = ["Do que é feita a água?", "O que causa as marés?", "Por que o céu é azul?",
           "Qual a velocidade da luz?", "Como funciona a fotossíntese?", "Do que são feitos os átomos?"]
HIST = ["Quando terminou a Segunda Guerra Mundial?", "Quem descobriu o Brasil?", "Em que ano o homem foi à Lua?",
        "Quando começou a Revolução Francesa?", "Quem pintou a Mona Lisa?"]
GERAL = ["Oi, tudo bem?", "Como você está hoje?", "Me conte uma piada.", "O que você acha do meu projeto?",
         "Me ajude a decidir o que fazer.", "Estou cansado hoje.", "Qual sua opinião sobre IA?",
         "Bom dia!", "Você pode me dar um conselho?", "O que devo estudar depois?", "Me explique sua ideia.",
         "Acho que vou desistir, o que acha?", "Vamos conversar um pouco?", "Você lembra do que falei?",
         # raciocínio aberto / meta / usa contexto do usuário (o que os micros NÃO fazem)
         "Considerando o que você sabe de mim, que projeto combinaria comigo?",
         "Com base no que conversamos, o que eu deveria fazer agora?", "Pense e me diga sua opinião.",
         "Analise minha situação e recomende um caminho.", "O que você recomendaria pra mim?",
         "Levando em conta meus objetivos, qual o próximo passo?", "Me ajude a planejar meu projeto.",
         "Compare as duas opções e me diga qual é melhor.", "Resuma o que discutimos até agora.",
         "Dado tudo que falei, o que faz mais sentido?", "Qual estratégia você sugere e por quê?",
         "Reflita sobre isso e me dê um conselho pensado.", "O que você acha que combina com meu perfil?",
         "Me dê uma ideia criativa para o meu trabalho.", "Argumente a favor e contra essa decisão."]

def make_data():
    d = []
    for p in PAISES:
        for t in [f"Qual é a capital d{'o' if p in ('Brasil','Japão','Egito','México','Canadá','Chile','Peru') else 'a'} {p}?",
                  f"capital de {p}", f"me diga a capital de {p}", f"qual a capital {p}"]:
            d.append((t, "fato"))
    for c in CIENCIA + HIST: d.append((c, "fato")); d.append((c.lower(), "fato"))
    for c in COISAS:
        for t in [f"escreva {c}", f"crie {c}", f"implemente {c}", f"me dá o código de {c}", f"faça uma função para {c}"]:
            d.append((t, "codigo"))
    for t in TERMOS:
        for f_ in [f"o que é {t}?", f"defina {t}", f"explique o conceito de {t}", f"em programação, o que é {t}", f"o que significa {t}"]:
            d.append((f_, "definicao"))
    ops = ["+", "-", "x", "*", "/", "vezes", "mais", "menos", "dividido por"]
    for _ in range(120):
        a, b = random.randint(1, 99), random.randint(1, 99); op = random.choice(ops)
        d.append((random.choice([f"quanto é {a} {op} {b}?", f"calcule {a} {op} {b}", f"{a} {op} {b} = ?",
                                 f"resolva {a} {op} {b}", f"quanto dá {a} {op} {b}"]), "math"))
    for g in GERAL:
        d.append((g, "geral"))
        d.append((g.lower(), "geral"))
    random.shuffle(d)
    return d

class ByteRouter(nn.Module):
    def __init__(self, d=96, nlayers=2, nheads=4, nclass=5, maxlen=96):
        super().__init__()
        self.emb = nn.Embedding(256, d); self.pos = nn.Embedding(maxlen, d); self.maxlen = maxlen
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, nheads, d * 4, batch_first=True, dropout=0.1) for _ in range(nlayers)])
        self.head = nn.Linear(d, nclass)

    def forward(self, x, mask):
        h = self.emb(x) + self.pos(torch.arange(x.size(1), device=x.device))[None]
        for b in self.blocks: h = b(h, src_key_padding_mask=~mask)
        h = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(h)

def encode(qs, maxlen=96, dev="cpu"):
    B = [list(q.encode("utf-8"))[:maxlen] for q in qs]
    L = max(len(b) for b in B)
    x = torch.zeros(len(B), L, dtype=torch.long); m = torch.zeros(len(B), L, dtype=torch.bool)
    for i, b in enumerate(B):
        x[i, :len(b)] = torch.tensor(b); m[i, :len(b)] = True
    return x.to(dev), m.to(dev)

if __name__ == "__main__":
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    data = make_data()
    n_val = len(data) // 6; val, tr = data[:n_val], data[n_val:]
    print(f"dados: {len(tr)} treino + {len(val)} val | {len(DOMAINS)} domínios", flush=True)
    m = ByteRouter().to(DEV); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=0.01)
    for step in range(600):
        batch = random.sample(tr, 64)
        x, msk = encode([q for q, _ in batch], dev=DEV)
        y = torch.tensor([DI[l] for _, l in batch], device=DEV)
        loss = F.cross_entropy(m(x, msk), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0: print(f"  step {step} loss {loss.item():.3f}", flush=True)
    # eval
    m.eval()
    with torch.no_grad():
        x, msk = encode([q for q, _ in val], dev=DEV)
        pred = m(x, msk).argmax(1).cpu()
    acc = (pred == torch.tensor([DI[l] for _, l in val])).float().mean().item()
    print(f"acurácia val: {acc:.1%}", flush=True)
    torch.save({"model": m.state_dict(), "domains": DOMAINS}, "ckpt_router.pt")
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"salvo ckpt_router.pt ({n:.2f}M params)", flush=True)
    # demo routing + confiança
    m_cpu = m.to("cpu")
    tests = ["Qual é a capital da Suécia?", "escreva uma função que ordena uma lista", "o que é polimorfismo?",
             "quanto é 17 x 23?", "estou meio triste hoje, e você?", "me explica a teoria da relatividade"]
    with torch.no_grad():
        for q in tests:
            x, msk = encode([q]); p = F.softmax(m_cpu(x, msk)[0], -1)
            conf, idx = p.max(0)
            tag = DOMAINS[idx] if conf > 0.55 else "geral→QWEN (baixa confiança)"
            print(f"  [{conf:.2f}] {tag:32} ← {q!r}", flush=True)
    print("DONE byte_router", flush=True)
