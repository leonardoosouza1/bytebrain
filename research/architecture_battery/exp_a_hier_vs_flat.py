"""Architecture battery — Experiment A: Hierarchical (MEGABYTE-lite) vs Flat byte-LM.

Hypothesis: a flat byte-LM degrades after ~30 words because 256 bytes of context is only ~40
words and there is no long-range structure. A hierarchical model runs the big transformer over
*patches* (so 128 patches x 8 bytes = 1024 bytes ~ 170 words of cheap context) and decodes bytes
locally. If the hypothesis holds, the hierarchical model should generate a LONGER coherent span
(lower wtrans on long samples) at the SAME parameter and token budget.

Fair fight: same corpus, same tokens/step, same number of steps. We report val bits/byte and the
word-transition surprisal (wtrans) of long generations for both.
"""
import math
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
BASE = "/home/leonardo/projects/LLM/bytebrain"

# ---- shared hyperparameters (matched budgets) ----
TIME_BUDGET_S = 600          # FAIR FIGHT: equal wall-clock per model (hierarchy is ~5x cheaper/step)
TOK_PER_STEP = 24576          # both models see the same bytes/step
PATCH = 4                     # bytes per patch (smaller = less local-decoder bottleneck)
T_HIER = 1024                 # hierarchical sequence length in bytes (-> 256 patches)
CTX_FLAT = 256               # flat context in bytes
DROP = 0.1


# ---------------- data ----------------
TXT = open(f"{BASE}/data/pt_clean.txt", encoding="utf-8").read()
B = np.frombuffer(TXT.encode("utf-8"), np.uint8)
CUT = int(len(B) * 0.97)
TR, VA = B[:CUT], B[CUT:]
_W = re.compile(r"[a-zàáâãéêíóôõúüç]+")
_words = _W.findall(TXT.lower())
WUNI, WBI, WV = Counter(_words), Counter(zip(_words, _words[1:])), len(set(_words))


def wtrans(t):
    w = _W.findall(t.lower())
    if len(w) < 3:
        return 12.0
    return float(np.mean([
        -math.log((WBI.get((w[i], w[i + 1]), 0) + 0.05) / (WUNI.get(w[i], 0) + 0.05 * WV))
        for i in range(len(w) - 1)
    ]))


def batch(arr, bs, L):
    ix = np.random.randint(0, len(arr) - L - 1, bs)
    x = np.stack([arr[i:i + L + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(DEV)


# ---------------- shared transformer block ----------------
class Block(nn.Module):
    def __init__(self, dim, heads, drop=DROP):
        super().__init__()
        self.h = heads
        self.l1 = nn.LayerNorm(dim)
        self.l2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.pr = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * dim, dim))
        self.do = nn.Dropout(drop)

    def forward(self, x):
        Bn, L, D = x.shape
        h = self.l1(x)
        q, k, v = self.qkv(h).view(Bn, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        p = DROP if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        x = x + self.do(self.pr(a.transpose(1, 2).reshape(Bn, L, D)))
        return x + self.do(self.mlp(self.l2(x)))


class Stack(nn.Module):
    def __init__(self, dim, layers, heads):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


# ---------------- flat baseline ----------------
class FlatGPT(nn.Module):
    def __init__(self, dim=384, layers=6, heads=6, ctx=CTX_FLAT):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(256, dim)
        self.pos = nn.Embedding(ctx, dim)
        self.body = Stack(dim, layers, heads)
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 256)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))[None]
        return self.head(self.lnf(self.body(h)))


# ---------------- hierarchical (MEGABYTE-lite) ----------------
class HierGPT(nn.Module):
    def __init__(self, gdim=384, glayers=6, gheads=6, ldim=192, llayers=4, lheads=6, patch=PATCH):
        super().__init__()
        self.P = patch
        self.tok = nn.Embedding(256, ldim)
        self.patch_proj = nn.Linear(patch * ldim, gdim)
        self.gstart = nn.Parameter(torch.zeros(1, 1, gdim))
        self.gpos = nn.Embedding(4096, gdim)
        self.global_ = Stack(gdim, glayers, gheads)
        self.g2l = nn.Linear(gdim, ldim)
        self.lpos = nn.Embedding(patch, ldim)
        self.local = Stack(ldim, llayers, lheads)
        self.lnf = nn.LayerNorm(ldim)
        self.head = nn.Linear(ldim, 256)
        self.gdim, self.ldim = gdim, ldim

    def _global(self, x):
        """x: (Bn, T) bytes, T multiple of P. Returns gctx (Bn, K, ldim): context for each patch."""
        Bn, Tt = x.shape
        K = Tt // self.P
        be = self.tok(x).view(Bn, K, self.P * self.ldim)
        patch_emb = self.patch_proj(be)                                  # (Bn, K, gdim)
        gin = torch.cat([self.gstart.expand(Bn, 1, self.gdim), patch_emb[:, :-1]], 1)  # shift: patch k sees 0..k-1
        gin = gin + self.gpos(torch.arange(K, device=x.device))[None]
        G = self.global_(gin)                                            # (Bn, K, gdim)
        return self.g2l(G)                                               # (Bn, K, ldim)

    def forward(self, x):
        Bn, Tt = x.shape
        K = Tt // self.P
        gctx = self._global(x)                                          # (Bn, K, ldim)
        be = self.tok(x).view(Bn, K, self.P, self.ldim)                 # (Bn, K, P, ldim)
        lin = torch.cat([gctx.unsqueeze(2), be[:, :, :-1]], 2)          # (Bn, K, P, ldim): [gctx, b0..b_{P-2}]
        lin = lin + self.lpos(torch.arange(self.P, device=x.device))[None, None]
        lin = lin.reshape(Bn * K, self.P, self.ldim)
        out = self.lnf(self.local(lin))
        return self.head(out).view(Bn, Tt, 256)

    @torch.no_grad()
    def generate(self, n_bytes=400, seed="O ", temp=0.6, top_p=0.85, rep=1.4, device=DEV):
        self.eval()
        out = list(seed.encode())
        n_patches = (n_bytes + self.P - 1) // self.P
        for _ in range(n_patches):
            complete = (len(out) // self.P) * self.P
            if complete == 0:
                gin = self.gstart.expand(1, 1, self.gdim)
                Kk = 1
            else:
                x = torch.tensor([out[:complete]], device=device)
                K = complete // self.P
                be = self.tok(x).view(1, K, self.P * self.ldim)
                pe = self.patch_proj(be)
                gin = torch.cat([self.gstart.expand(1, 1, self.gdim), pe], 1)  # context for next patch
                Kk = K + 1
            gin = gin + self.gpos(torch.arange(Kk, device=device))[None]
            gctx = self.g2l(self.global_(gin))[0, -1]                   # (ldim,) context for next patch
            seq = [gctx]
            for j in range(self.P):
                h = torch.stack(seq, 0)[None] + self.lpos(torch.arange(len(seq), device=device))[None]
                logit = self.head(self.lnf(self.local(h)))[0, -1].clone()
                for b in set(out[-48:]):
                    logit[b] /= rep
                pr = F.softmax(logit / temp, -1)
                sp, si = torch.sort(pr, descending=True)
                keep = torch.cumsum(sp, 0) <= top_p
                keep[0] = True
                sp = (sp * keep)
                sp /= sp.sum()
                nb = int(si[torch.multinomial(sp, 1)])
                out.append(nb)
                seq.append(self.tok(torch.tensor([nb], device=device))[0])
        return bytes(out[:len(seed.encode()) + n_bytes]).decode("utf-8", "ignore")


# ---------------- flat generation ----------------
@torch.no_grad()
def flat_generate(g, n_bytes=400, seed="O ", temp=0.6, top_p=0.85, rep=1.4):
    g.eval()
    x = torch.tensor([list(seed.encode())], device=DEV)
    for _ in range(n_bytes):
        logit = g(x[:, -g.ctx:])[0, -1].clone()
        for b in set(x[0, -48:].tolist()):
            logit[b] /= rep
        pr = F.softmax(logit / temp, -1)
        sp, si = torch.sort(pr, descending=True)
        keep = torch.cumsum(sp, 0) <= top_p
        keep[0] = True
        sp = sp * keep
        sp /= sp.sum()
        x = torch.cat([x, si[torch.multinomial(sp, 1)].view(1, 1)], 1)
    return bytes(x[0].tolist()).decode("utf-8", "ignore")


# ---------------- train / eval ----------------
def train(model, L, bs, tag):
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=0.05)
    model.train()
    t0 = time.time()
    step = 0
    while time.time() - t0 < TIME_BUDGET_S:
        step += 1
        xb = batch(TR, bs, L)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"  [{tag}] step {step} bpb {loss.item()/math.log(2):.2f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [{tag}] FIM: {step} steps em {time.time()-t0:.0f}s", flush=True)
    return model


@torch.no_grad()
def val_bpb(model, L, bs):
    model.eval()
    tot = 0.0
    for _ in range(15):
        xb = batch(VA, bs, L)
        logits = model(xb[:, :-1])
        tot += F.cross_entropy(logits.reshape(-1, 256), xb[:, 1:].reshape(-1)).item()
    return tot / 15 / math.log(2)


def main():
    print(f"corpus {len(B)/1e6:.2f}MB | {TIME_BUDGET_S}s/modelo (tempo igual) | tokens/step {TOK_PER_STEP}\n", flush=True)

    flat = FlatGPT().to(DEV)
    hier = HierGPT().to(DEV)
    print(f"FLAT params {sum(p.numel() for p in flat.parameters())/1e6:.1f}M | "
          f"HIER params {sum(p.numel() for p in hier.parameters())/1e6:.1f}M\n", flush=True)

    bs_flat = TOK_PER_STEP // CTX_FLAT      # 96
    bs_hier = TOK_PER_STEP // T_HIER        # 24

    print("=== treinando FLAT (baseline) ===", flush=True)
    train(flat, CTX_FLAT, bs_flat, "flat")
    print("=== treinando HIER (MEGABYTE-lite) ===", flush=True)
    train(hier, T_HIER, bs_hier, "hier")

    vf = val_bpb(flat, CTX_FLAT, bs_flat)
    vh = val_bpb(hier, T_HIER, bs_hier)

    fs = [flat_generate(flat, 400) for _ in range(6)]
    hs = [hier.generate(400) for _ in range(6)]
    wf = float(np.mean([wtrans(s) for s in fs]))
    wh = float(np.mean([wtrans(s) for s in hs]))

    print("\n================ RESULTADO ================", flush=True)
    print(f"FLAT  | val_bpb {vf:.3f} | wtrans(400B) {wf:.2f}", flush=True)
    print(f"HIER  | val_bpb {vh:.3f} | wtrans(400B) {wh:.2f}", flush=True)
    print(f"\nFLAT amostra:\n{fs[0][:380]!r}", flush=True)
    print(f"\nHIER amostra:\n{hs[0][:380]!r}", flush=True)
    win = "HIER VENCE (span mais coerente)" if wh < wf - 0.2 else ("EMPATE" if abs(wh-wf) <= 0.2 else "FLAT vence")
    print(f"\n=> {win} | delta wtrans {wf-wh:+.2f}", flush=True)


if __name__ == "__main__":
    main()
