#!/usr/bin/env python3
"""PODA METABÓLICA → PERPLEXIDADE REAL do LM (não reconstrução de camada).

A Lei do Garimpo (2σ²·ln W) aplicada à poda dos neurônios ocultos das MLPs do
ByteBrain 40M. Compara 3 métodos de SELEÇÃO no MESMO nível de esparsidade,
medindo a PERPLEXIDADE/bpb real em bytes held-out (o que importa de verdade):

  - metabólica : mantém os neurônios que mais reduzem o resíduo da saída da
                 camada (critério greedy-resíduo = a nossa lei), com ativações reais.
  - Wanda      : |peso_saída| × ‖ativação‖ (o SOTA de poda sem retreino, 2023).
  - magnitude  : ‖peso_saída‖ × ‖peso_entrada‖ (o baseline clássico).

Poda estruturada: zera neurônios ocultos (colunas de W_in / linhas de W_out)
por bloco, no MESMO orçamento k. Sem retreino (one-shot). Métrica = bpb.

Honesto: usa ativações de um lote real de bytes pra guiar metabólica e Wanda.
Perplexidade medida em bytes held-out NÃO vistos na calibração.
"""
import torch, math, sys, os
sys.path.insert(0, "research")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

from train_graph import GraphByteGPT

ck = torch.load("research/ckpt_gen/ckpt.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]
sd = ck["model"]
print(f"# ByteBrain {cfg} · device {DEV}")

def build():
    m = GraphByteGPT(cfg["dim"], cfg["layers"], cfg["heads"], cfg["ctx"])
    m.load_state_dict(sd, strict=False)
    return m.to(DEV).eval()

# ── dados held-out ────────────────────────────────────────────────────
def load_bytes(path, n):
    with open(path, "rb") as f:
        data = f.read()
    return data[:n]

# calibração (pra ativações) e held-out (pra bpb) — trechos DISJUNTOS
src = "data/literature_pt.txt"
raw = load_bytes(src, 4_000_000)
calib = raw[1_000_000:1_200_000]   # 200 KB pra ativações
held = raw[3_000_000:3_100_000]    # 100 KB held-out DISJUNTO pra bpb
ctx = cfg["ctx"]

def batches(data, bs, n_batches, seed=0):
    import random
    r = random.Random(seed)
    out = []
    for _ in range(n_batches):
        xs = []
        for _ in range(bs):
            i = r.randint(0, len(data) - ctx - 1)
            xs.append(torch.tensor([b for b in data[i:i + ctx + 1]], dtype=torch.long))
        b = torch.stack(xs)
        out.append((b[:, :-1].to(DEV), b[:, 1:].to(DEV)))
    return out

@torch.no_grad()
def bpb(model, data, n_batches=40, bs=16):
    tot_loss, tot = 0.0, 0
    for x, y in batches(data, bs, n_batches, seed=123):
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum")
        tot_loss += loss.item()
        tot += y.numel()
    return tot_loss / tot / math.log(2)  # bits por byte

# ── ativações reais das MLPs (entrada do GELU) por bloco ──────────────
@torch.no_grad()
def collect_activations(model):
    acts = {}
    hooks = []
    for i, blk in enumerate(model.blocks):
        lin1 = blk.mlp[0]  # Linear(dim, 4*dim)
        def mk(i):
            def hook(mod, inp, out):
                # out = pre-GELU (dim 2560); guarda ‖·‖ por neurônio e a média |·|
                h = torch.nn.functional.gelu(out).reshape(-1, out.shape[-1])
                if i not in acts:
                    acts[i] = torch.zeros(out.shape[-1], device=DEV)
                acts[i] += (h * h).sum(0)  # energia de ativação por neurônio
            return hook
        hooks.append(lin1.register_forward_hook(mk(i)))
    for x, _ in batches(calib, 16, 12, seed=7):
        model(x)
    for h in hooks:
        h.remove()
    return {i: v.sqrt() for i, v in acts.items()}  # ‖ativação‖ por neurônio

# ── seleção: quais neurônios MANTER (k de 2560) por bloco ─────────────
def select_keep(model, acts, method, keep_k):
    keep = {}
    for i, blk in enumerate(model.blocks):
        w_in = blk.mlp[0].weight.data   # (2560, 640)
        w_out = blk.mlp[3].weight.data  # (640, 2560)
        W = w_in.shape[0]
        act = acts[i]                   # (2560,)
        if method == "magnitude":
            score = w_out.norm(dim=0) * w_in.norm(dim=1)
        elif method == "wanda":
            score = w_out.norm(dim=0) * act
        elif method == "metabolic":
            # importância ~ energia que o neurônio entrega à saída = ‖ativação‖·‖w_out coluna‖
            # (a nossa lei: quanto o neurônio explica da saída; o custo 2σ²lnW é o
            #  MESMO pra todos → a ordem é por energia entregue, = greedy-resíduo 1ª ordem)
            score = (w_out.norm(dim=0) * act)
            # refina: penaliza redundância (neurônios com w_out muito correlacionados)
            # — o que distingue metabólica de Wanda: cobrir direções DISTINTAS.
            wn = w_out / (w_out.norm(dim=0, keepdim=True) + 1e-9)
            # greedy: escolhe por energia mas desconta projeção nos já escolhidos
            chosen = []
            avail = score.clone()
            cov = torch.zeros_like(w_out[:, 0])
            for _ in range(keep_k):
                j = int(torch.argmax(avail))
                if avail[j] <= -1e30:
                    break
                chosen.append(j)
                avail[j] = -1e30
                # desconta a direção de w_out[:,j] das energias restantes
                dj = wn[:, j]
                proj = (wn * dj[:, None]).sum(0).abs()  # correlação com j
                avail = avail - score * proj * 0.5
            keep[i] = torch.tensor(chosen, device=DEV)
            continue
        keep[i] = torch.topk(score, keep_k).indices
    return keep

def apply_prune(model, keep, refit=False, ref=None):
    for i, blk in enumerate(model.blocks):
        idx = keep[i]
        mask = torch.zeros(blk.mlp[0].weight.shape[0], device=DEV)
        mask[idx] = 1.0
        blk.mlp[0].weight.data *= mask[:, None]
        blk.mlp[0].bias.data *= mask
        blk.mlp[3].weight.data *= mask[None, :]
    if refit and ref is not None:
        _refit_outputs(model, keep, ref)

@torch.no_grad()
def _refit_outputs(model, keep, ref):
    """re-ajusta w_out de cada MLP (só colunas mantidas) por least-squares pra
    reproduzir a saída ORIGINAL da MLP nas ativações de calibração. É o mecanismo
    da lei: dado o subconjunto, resolve as amplitudes ótimas (mín. resíduo)."""
    # coleta H (ativações pós-GELU) e Y (saída da MLP original) por bloco
    Hs, Ys = {}, {}
    hooks=[]
    for i, blk in enumerate(model.blocks):
        def mk(i, blk):
            def hook(mod, inp, out):
                h = torch.nn.functional.gelu(out).reshape(-1, out.shape[-1])
                Hs.setdefault(i, []).append(h.detach())
            return hook
        hooks.append(blk.mlp[0].register_forward_hook(mk(i, blk)))
    # roda calib pra pegar H; Y reconstruímos de H com o w_out ATUAL (já mascarado é ok:
    # queremos a saída-alvo = H_full·W_out_orig; mas o modelo já foi podado. Usamos o
    # w_out ORIGINAL guardado em ref).
    for x,_ in batches(calib, 16, 8, seed=99):
        model(x)
    for h in hooks: h.remove()
    for i, blk in enumerate(model.blocks):
        H = torch.cat(Hs[i], 0)                 # (N, 2560) — já mascarado (zeros fora)
        idx = keep[i]
        Hk = H[:, idx]                          # (N, k) só mantidos
        Wout_orig = ref[i]                      # (640, 2560) original
        Y = H @ Wout_orig.T                     # alvo: saída que a MLP mascarada daria...
        # na verdade queremos reproduzir a saída do modelo CHEIO. H aqui é pós-poda,
        # mas os neurônios mantidos têm a MESMA ativação (a máscara não muda w_in dos
        # mantidos). Então H[:,idx] é a ativação real dos mantidos. Alvo = saída cheia:
        # aproximamos por H_full·Wout_orig, mas H_full≈H exceto zeros nos podados.
        # least-squares: Wnew (k×640) tal que Hk·Wnew ≈ Y
        # resolve normal eqs (Hk^T Hk) Wnew = Hk^T Y
        A = Hk.T @ Hk + 1e-3*torch.eye(Hk.shape[1], device=DEV)
        B = Hk.T @ Y
        Wnew = torch.linalg.solve(A, B)         # (k, 640)
        newout = torch.zeros_like(blk.mlp[3].weight.data)
        newout[:, idx] = Wnew.T
        blk.mlp[3].weight.data = newout

# ── experimento ───────────────────────────────────────────────────────
base = build()
ref_wout = {i: blk.mlp[3].weight.data.clone() for i, blk in enumerate(base.blocks)}
bpb_full = bpb(base, held)
W = cfg["dim"] * 4
print(f"\nbpb do modelo CHEIO (held-out): {bpb_full:.4f}  (W={W} neurônios/MLP)\n")
acts = collect_activations(base)

print(f"{'esparsidade':>12} {'método':>12} {'bpb':>10} {'Δ vs cheio':>12}")
results = {}
for frac in [0.3, 0.5, 0.7]:  # fração PODADA
    keep_k = int(W * (1 - frac))
    for method in ["wanda", "metabolic", "wanda+refit", "metabolic+refit"]:
        m = build()
        base_method = method.replace("+refit","")
        keep = select_keep(m, acts, base_method, keep_k)
        apply_prune(m, keep, refit=("refit" in method), ref=ref_wout)
        b = bpb(m, held)
        results[(frac, method)] = b
        star = " ◄" if "metabolic" in method else ""
        print(f"{int(frac*100):>10}% {method:>16} {b:>10.4f} {b-bpb_full:>+12.4f}{star}")
    print()

# veredito
print("VEREDITO (bpb — menor é melhor):")
for frac in [0.3, 0.5, 0.7]:
    r = {k: results[(frac,k)] for k in ["wanda","metabolic","wanda+refit","metabolic+refit"]}
    best = min(r, key=r.get)
    print(f"  {int(frac*100)}% podado → melhor: {best} ({r[best]:.4f})")
    print(f"     {' · '.join(f'{k} {v:.3f}' for k,v in r.items())}")

import json
json.dump({f"{k[0]}_{k[1]}": v for k, v in results.items()} | {"full": bpb_full},
          open("research/poda_perplexidade.json", "w"), indent=1)
print("\n→ research/poda_perplexidade.json")
