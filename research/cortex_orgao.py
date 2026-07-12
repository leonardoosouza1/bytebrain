#!/usr/bin/env python3
"""CÓRTEX (ByteBrain real, congelado) + ÓRGÃO plástico hormonal EM CIMA.

O modelo de texto de verdade é o córtex (língua, treinado por backprop). Por
cima, um órgão de correção rápida que ADAPTA online — com o design hormonal
validado (gate LOCAL ∈[0,1] + esquecer SELETIVO). Testa a tese central:
num mundo que MUDA (stream PT → código), o córtex+órgão adapta melhor que:
  (a) córtex sozinho (congelado, genérico)
  (b) córtex + ajuste online NAIVE (lr fixo, sem hormônio)

Órgão = tabela de correção de logits indexada pelo byte anterior (memória rápida
tipo neural-cache), somada aos logits do córtex. Aprende online no próximo byte.
Métrica: bpb ao longo do stream, especialmente a adaptação após a troca.
"""
import torch, math, sys, os
sys.path.insert(0, "research")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
from train_graph import GraphByteGPT

ck = torch.load("research/ckpt_gen/ckpt.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]; sd = ck["model"]
cortex = GraphByteGPT(cfg["dim"], cfg["layers"], cfg["heads"], cfg["ctx"])
cortex.load_state_dict(sd, strict=False)
cortex = cortex.to(DEV).eval()
for p in cortex.parameters(): p.requires_grad_(False)
ctx = cfg["ctx"]

def load(p, n, off):
    with open(p, "rb") as f: f.seek(off); return f.read(n)
PT   = load("data/literature_pt.txt", 40000, 600000)
CODE = load("data/instruct_corpus.txt", 40000, 300000)
stream = PT + CODE           # troca de domínio no meio (byte 40000)
SWITCH = len(PT)

@torch.no_grad()
def cortex_logits(window):
    x = torch.tensor([list(window)], dtype=torch.long, device=DEV)
    out = cortex(x)
    logits = out[0] if isinstance(out, tuple) else out
    return logits[0, -1]     # logits do próximo byte (256,)

def run(mode):
    # órgão: correção de logits por byte-anterior (256×256), começa em zero
    F = torch.zeros(256, 256, device=DEV)
    lr = 0.5
    ema_ctx = torch.ones(256, device=DEV) * 3.0
    ema_glob = 3.0
    losses = []; trace = []
    win = list(stream[:ctx])
    for t in range(ctx, len(stream)):
        prev = stream[t-1]; cur = stream[t]
        base = cortex_logits(win[-ctx:])
        logits = base + (F[prev] if mode != "cortex" else 0.0)
        p = torch.softmax(logits, 0)
        loss = -math.log(max(p[cur].item(), 1e-9)) / math.log(2)
        losses.append(loss)
        if len(losses) > 800: losses.pop(0)
        if (t-ctx) % 800 == 0: trace.append(sum(losses)/len(losses))
        # atualiza o órgão (não o córtex)
        if mode != "cortex":
            g = p.clone(); g[cur] -= 1.0
            if mode == "naive":
                F[prev] -= lr * g
            elif mode == "hormonal":
                gate = min(1.0, ema_ctx[prev].item() / max(ema_glob, 0.3))
                ema_ctx[prev] = 0.98*ema_ctx[prev] + 0.02*loss
                ema_glob = 0.999*ema_glob + 0.001*loss
                F[prev] -= lr * gate * g
                F[prev] *= (1.0 - 0.01*(1.0 - gate))   # esquece seletivo
        win.append(cur)
    tr = trace
    n_pre = SWITCH // 800
    pre = sum(tr[max(0,n_pre-6):n_pre]) / max(1,len(tr[max(0,n_pre-6):n_pre]))
    pos = sum(tr[n_pre:n_pre+6]) / max(1,len(tr[n_pre:n_pre+6]))
    return sum(tr)/len(tr), pre, pos

print(f"# CÓRTEX ByteBrain {cfg['dim']}d congelado + ÓRGÃO plástico · {DEV}")
print(f"# stream: PT ({len(PT)}B) → código ({len(CODE)}B), troca no meio\n")
print(f"{'abordagem':>22} {'bpb médio':>10} {'PT (pré)':>10} {'código (pós)':>13}")
r={}
for nome, m in [("córtex só (congelado)","cortex"),("córtex+órgão naive","naive"),("córtex+órgão HORMONAL","hormonal")]:
    a,pre,pos = run(m)
    r[m]=(a,pre,pos)
    print(f"{nome:>22} {a:>10.3f} {pre:>10.3f} {pos:>13.3f}")
print("\nVEREDITO:")
c=r["cortex"]; n=r["naive"]; h=r["hormonal"]
print(f"  órgão ajuda no geral? naive {n[0]:.3f} · hormonal {h[0]:.3f} vs córtex {c[0]:.3f}")
print(f"  no CÓDIGO (domínio novo, pós-troca): córtex {c[2]:.3f} · naive {n[2]:.3f} · hormonal {h[2]:.3f}")
best = min(c[2],n[2],h[2])
who = "córtex só" if best==c[2] else ("naive" if best==n[2] else "HORMONAL")
print(f"  → melhor adaptação ao código: {who}")
