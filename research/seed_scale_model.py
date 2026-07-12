#!/usr/bin/env python3
"""BATERIA 13 — ESCALA DO ÓRGÃO estende o piso da SEMENTE? (lote 4). A semente é PLANTADA por otimização, então
a fidelidade de germinação mede a CAPACIDADE do modelo de reconstruir um alvo a partir de poucos bits (lei do
andaime). Na bat.2 o int2 SEMPRE falhou (0/8) no 8M. Pergunta honesta: os priors mais ricos do 40M RESGATAM a
germinação nos orçamentos apertados onde o 8M falhou? Se sim, escala do modelo = semente menor. GPU."""
import sys, torch
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain"); sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research")
from src.model import ByteGPT
from safetensors.torch import load_file
from wisdom_bridge import ByteSeed, quant, DEV, enc

REGIONS = [("clareira do norte","ha agua limpa ao norte"),("bosque sombrio","o predador caca de noite"),
    ("campo dourado","a fruta azul mata a fome"),("colina ventosa","o cardume protege do perigo"),
    ("lago espelhado","beba aqui para nao ter sede"),("gruta funda","a gruta abriga contra a noite"),
    ("vale das pedras","cuidado com o buraco fundo"),("mata alta","esconda-se dentro do arbusto")]
FMT = "P: {q}?\nR:"

def load_8m():
    m = ByteGPT(dim=384, n_layers=6, n_heads=6).to(DEV)
    sd = load_file("/home/leonardo/projects/LLM/bytebrain/export_bytebrain_8m/model.safetensors")
    sd = {k.replace(".mlp.2.", ".mlp.3."): v for k, v in sd.items()}
    m.load_state_dict(sd); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

def load_40m():
    ck = torch.load("/home/leonardo/projects/LLM/bytebrain/research/ckpt_gen/ckpt_best.pt", map_location=DEV, weights_only=False)
    c = ck["config"]; m = ByteGPT(dim=c["dim"], n_layers=c["layers"], n_heads=c["heads"], context=c["ctx"]).to(DEV)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    print(f"  [40M carregado: {sum(p.numel() for p in m.parameters())/1e6:.1f}M · missing {len(missing)} unexpected {len(unexpected)}]", flush=True)
    return m

# orçamentos: (K, bits) — foco onde o 8M falhou/foi marginal na bat.2
BUDGETS = [(2,8),(2,4),(4,2),(2,2)]

def run(m, label):
    bs = ByteSeed(m); res = {}
    for K in sorted(set(k for k,_ in BUDGETS)):
        seeds = [(name, FMT.format(q=name), bs.plant(FMT.format(q=name), " "+fact+"\n", K=K, steps=400)[0], fact) for name, fact in REGIONS]
        for (KK, bits) in [b for b in BUDGETS if b[0]==K]:
            ok = 0
            for name, prompt, seed, fact in seeds:
                out = bs.recall(quant(seed, bits), prompt, n=24, stop_at=10).strip()
                ok += fact.lower()[:9] in out.lower()
            res[(K,bits)] = ok
    print(f"\n[{label}] fidelidade por orçamento:")
    for (K,bits) in BUDGETS:
        sb = (K*384*bits)//8
        print(f"   K={K} bits={bits}  ({sb:>4}B)  →  {res[(K,bits)]}/8", flush=True)
    return res

print("=== 8M (baseline, = bat.2) ===", flush=True)
r8 = run(load_8m(), "8M")
torch.cuda.empty_cache()
print("\n=== 40M (priors mais ricos) ===", flush=True)
r40 = run(load_40m(), "40M")

print("\n=== 8M vs 40M (o resgate acontece?) ===")
for (K,bits) in BUDGETS:
    d = r40[(K,bits)] - r8[(K,bits)]
    tag = "40M RESGATA" if d>1 else ("~igual" if abs(d)<=1 else "40M pior")
    print(f"   K={K} bits={bits}:  8M {r8[(K,bits)]}/8  vs  40M {r40[(K,bits)]}/8   → {tag}")
print("\n→ se o 40M germina onde o 8M falha (esp. bits=2), escala do modelo estende o piso da semente (andaime mais forte).")
