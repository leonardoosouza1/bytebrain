#!/usr/bin/env python3
"""Self-contained (isolated from src/model.py + train.py so it never disturbs a
running sweep) trainer for Leonardo's "neuron graph" ideas:

  --topk K     : top-K SPARSE causal attention — each position connects to only
                 its K most-relevant others ("de um neurônio, quais K ativam").
                 K=0 = dense (baseline). This is the '16 connections / synapse'
                 idea, learned + differentiable, O(N*K) instead of O(N^2).

  --vq-codes N : a discrete N-way 'relation code' per position (Gumbel-softmax,
                 hard) that conditions the next-byte prediction — the '4 bits =
                 relation type' angle. 0 = off.

Run on GPU (after the sweep frees it):
  PY=.../.venv-rocm/bin/python; HSA_OVERRIDE_GFX_VERSION=10.3.0 $PY research/train_graph.py \
     --corpus data/pt_big.txt --ckpt-dir ckpt_graph_top16 --topk 16 --amp --seed 0 ...
"""
import argparse, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


_AQ_BITS = 0


def set_act_quant(bits):
    global _AQ_BITS
    _AQ_BITS = int(bits)


def _fake_quant_act(x):
    """4-bit (nibble) per-channel activation quant, STE — combine the graph with
    the validated light neuron."""
    b = _AQ_BITS
    if b <= 0 or b >= 16:
        return x
    qmax = (1 << b) - 1
    dims = tuple(range(x.dim() - 1))
    lo = x.amin(dims, keepdim=True).detach(); hi = x.amax(dims, keepdim=True).detach()
    rng = (hi - lo).clamp_min(1e-9)
    xq = torch.round(((x - lo) / rng) * qmax) / qmax * rng + lo
    return x + (xq - x).detach()


def topk_causal_attn(q, k, v, topk):
    B, H, L, d = q.shape
    scores = (q @ k.transpose(-2, -1)) * (d ** -0.5)
    causal = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(causal, float("-inf"))
    if topk and topk < L:                                  # keep only top-K keys per query
        thr = scores.topk(min(topk, L), dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < thr, float("-inf"))
    return scores.softmax(-1) @ v


class GBlock(nn.Module):
    def __init__(self, dim, n_heads, topk, dropout=0.15):
        super().__init__()
        self.n_heads, self.topk = n_heads, topk
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim); self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        if self.topk and self.topk < L:
            a = topk_causal_attn(q, k, v, self.topk)
        else:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.1 if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, L, D)))
        return _fake_quant_act(x + self.drop(self.mlp(self.ln2(x))))   # nibble if quant on


def mtp_loss(h, xb, heads):
    """Multi-byte prediction: extra heads predict bytes t+2..t+1+len(heads) from
    the SAME hidden state — uses the representation's spare capacity to plan ahead."""
    ctx = h.size(1); total = 0.0; cnt = 0
    for j, head in enumerate(heads):
        o = j + 2; n = ctx - o + 1
        if n <= 0: break
        total = total + F.cross_entropy(head(h[:, :n]).reshape(-1, 256), xb[:, o:o + n].reshape(-1))
        cnt += 1
    return total / max(1, cnt)


def topic_loss_fn(h, xb, model, H=16, W=48):
    """Topic-lookahead loss: from h_t, predict the GIST (mean input-embedding) of a
    FUTURE window [t+H, t+H+W]. Forces the representation to anticipate where the
    passage is going = topic pressure (which next-byte CE lacks). Exact future bytes
    are unpredictable, but the gist is — so this is learnable and targets drift."""
    x = xb[:, :-1]
    emb = model.tok(x).detach().float()                 # [B,L,D] target embeddings (fp32: cumsum safe)
    B, L, D = emb.shape
    n = L - (H + W)
    if n <= 0:
        return h.new_zeros(())
    cs = emb.cumsum(1)
    target = (cs[:, H + W - 1: H + W - 1 + n] - cs[:, H - 1: H - 1 + n]) / W   # [B,n,D]
    pred = model.topic_head(h[:, :n]).float()
    return F.mse_loss(pred, target)


class HebbMemory(nn.Module):
    """Persistent associative memory (Hebbian / fast-weights / linear-attention).
    Accumulates outer-product associations Σ φ(k)⊗v CAUSALLY over the sequence — a
    CONSOLIDATED graph that PERSISTS across the whole passage (unlike attention,
    recomputed+forgotten each token). Gated retrieval. Targets topic-drift = the
    gap the coherence metric measured (ancoragem ~0)."""
    def __init__(self, dim, m=32):
        super().__init__()
        self.q = nn.Linear(dim, m); self.k = nn.Linear(dim, m); self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim); self.g = nn.Linear(dim, dim)

    def forward(self, h):                                # h [B,L,D]
        dt = h.dtype
        # fp32: the causal cumsum accumulates over the whole sequence and OVERFLOWS
        # fp16 (autocast) -> NaN. Compute the memory in fp32, return original dtype.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            qf = F.elu(self.q(h)) + 1                     # [B,L,m] positive feature
            kf = F.elu(self.k(h)) + 1
            v = self.v(h)
            kv = (kf.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(1)   # [B,L,m,D] consolidated memory
            kz = kf.cumsum(1)                            # [B,L,m] normalizer
            num = torch.einsum('blm,blmd->bld', qf, kv)
            den = torch.einsum('blm,blm->bl', qf, kz).clamp_min(1e-6).unsqueeze(-1)
            out = torch.sigmoid(self.g(h)) * self.out(num / den)   # gated retrieval
        return out.to(dt)


class GraphByteGPT(nn.Module):
    def __init__(self, dim, n_layers, n_heads, context, topk=0, vq_codes=0, mtp=0, mem=0, topic=0, grad_ckpt=0):
        super().__init__()
        self.context = context; self.grad_ckpt = grad_ckpt
        self.tok = nn.Embedding(256, dim); self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList([GBlock(dim, n_heads, topk) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(dim); self.head = nn.Linear(dim, 256)
        self.code_head = nn.Linear(dim, vq_codes) if vq_codes else None
        self.code_emb = nn.Parameter(torch.randn(vq_codes, dim) * 0.02) if vq_codes else None
        self.mtp_heads = nn.ModuleList([nn.Linear(dim, 256) for _ in range(mtp)]) if mtp else None
        self.mem = HebbMemory(dim) if mem else None
        self.topic_head = nn.Linear(dim, dim) if topic else None    # topic-lookahead (anti-drift objective)

    def encode(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        for b in self.blocks:
            if self.grad_ckpt and self.training:        # trade compute for VRAM (fit bigger/longer)
                h = torch.utils.checkpoint.checkpoint(b, h, use_reentrant=False)
            else:
                h = b(h)
        if self.mem is not None:                           # persistent associative memory (anti-drift)
            h = h + self.mem(h)
        h = self.lnf(h)
        if self.code_head is not None:                     # discrete relation code (synapse-type)
            code = F.gumbel_softmax(self.code_head(h), tau=1.0, hard=True)   # [B,L,N] one-hot, STE
            h = h + code @ self.code_emb
        return h

    def forward(self, x):
        return self.head(self.encode(x))                   # main = next-byte (used for bpb/gen)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def get_batch(data, bs, L, device):
    ix = np.random.randint(0, len(data) - L - 1, bs)
    x = np.stack([np.asarray(data[i:i + L + 1]) for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt-dir", default="ckpt_graph")
    ap.add_argument("--dim", type=int, default=320); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=5); ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=96); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--max-steps", type=int, default=15000)
    ap.add_argument("--warmup", type=int, default=600); ap.add_argument("--decay-steps", type=int, default=15000)
    ap.add_argument("--val-every", type=int, default=1000); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--topk", type=int, default=0, help="top-K sparse attention; 0 = dense")
    ap.add_argument("--vq-codes", type=int, default=0, help="N-way discrete relation code; 0 = off")
    ap.add_argument("--mtp", type=int, default=0, help="multi-byte prediction: extra heads for t+2..t+1+MTP (spare-capacity idea)")
    ap.add_argument("--mtp-weight", type=float, default=0.3, help="weight of the MTP aux loss")
    ap.add_argument("--quant-bits", type=int, default=0, help="N-bit nibble activation quant (combine w/ graph); 0 = off")
    ap.add_argument("--mem", type=int, default=0, help="Hebbian persistent associative memory (anti-drift); 0 = off")
    ap.add_argument("--topic-loss", type=float, default=0.0, help="weight of topic-lookahead loss (predict gist of future window); 0 = off")
    ap.add_argument("--grad-ckpt", type=int, default=0, help="gradient checkpointing (trade compute for VRAM, fit bigger/longer); 0 = off")
    ap.add_argument("--init-from", default="", help="init model weights from this checkpoint (fine-tune/distill); optimizer+step start fresh")
    ap.add_argument("--curriculum", type=int, default=0, help="grow sequence length short→long during training (Leonardo's curriculum idea); 0=off")
    ap.add_argument("--cur-start", type=int, default=64, help="curriculum starting seq length")
    ap.add_argument("--cur-steps", type=int, default=0, help="steps to grow cur-start→ctx (0 = 60%% of max-steps)")
    a = ap.parse_args()
    set_act_quant(a.quant_bits)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    if a.seed >= 0:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
    DEV = a.device
    data = np.memmap(a.corpus, dtype=np.uint8, mode="r")
    cut = int(len(data) * 0.999); TR, VA = data[:cut], data[cut:]
    model = GraphByteGPT(a.dim, a.layers, a.heads, a.ctx, topk=a.topk, vq_codes=a.vq_codes, mtp=a.mtp, mem=a.mem, topic=(1 if a.topic_loss > 0 else 0), grad_ckpt=a.grad_ckpt).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    print(f"GRAPH | {model.n_params/1e6:.1f}M params | topk={a.topk} vq={a.vq_codes} mtp={a.mtp} qbits={a.quant_bits} mem={a.mem} topic={a.topic_loss} gckpt={a.grad_ckpt} curric={a.curriculum}({a.cur_start}→{a.ctx}) | ctx={a.ctx} batch={a.batch} | seed={a.seed}", flush=True)

    cur_grow = a.cur_steps if a.cur_steps > 0 else int(a.max_steps * 0.6)
    def cur_len(s):                                  # curriculum: short→long sequence length
        if not a.curriculum: return a.ctx
        p = min(1.0, s / max(1, cur_grow))
        return max(8, min(a.ctx, int(a.cur_start + p * (a.ctx - a.cur_start))))

    def lr_at(s):
        if s < a.warmup: return a.lr * (s + 1) / a.warmup
        if s >= a.decay_steps: return a.lr * 0.1
        p = (s - a.warmup) / max(1, a.decay_steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    @torch.no_grad()
    def validate():
        model.eval(); tot = 0.0
        for _ in range(20):
            xb = get_batch(VA, 32, a.ctx, DEV)
            tot += F.cross_entropy(model(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1)).item()
        model.train(); return tot / 20 / math.log(2)

    cfg = {"dim": a.dim, "layers": a.layers, "heads": a.heads, "ctx": a.ctx,
           "topk": a.topk, "vq_codes": a.vq_codes, "mtp": a.mtp, "quant_bits": a.quant_bits, "mem": a.mem,
           "topic": (1 if a.topic_loss > 0 else 0)}

    def save(tag, step, best):                      # atomic full checkpoint (crash-safe)
        tmp = os.path.join(a.ckpt_dir, tag + ".pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "best": best, "config": cfg,
                    "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()}, tmp)
        os.replace(tmp, os.path.join(a.ckpt_dir, tag + ".pt"))

    start, best = 0, 1e9
    ckp = os.path.join(a.ckpt_dir, "ckpt.pt")
    if a.init_from and not os.path.exists(ckp):     # FINE-TUNE: load base weights, fresh optimizer
        s = torch.load(a.init_from, map_location=DEV, weights_only=False)
        miss = model.load_state_dict(s["model"], strict=False)
        print(f"INIT-FROM {a.init_from} (missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)})", flush=True)
    if os.path.exists(ckp):                         # RESUME from last checkpoint
        s = torch.load(ckp, map_location=DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        start, best = s["step"], s.get("best", 1e9)
        try:
            torch.set_rng_state(s["torch_rng"].cpu()); np.random.set_state(s["np_rng"])
        except Exception:
            pass
        print(f"RESUME from step {start} (best {best:.3f})", flush=True)

    t0 = time.time(); model.train()
    for step in range(start + 1, a.max_steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step)
        xb = get_batch(TR, a.batch, cur_len(step), DEV)   # curriculum length (or fixed ctx)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
            if a.mtp or a.topic_loss > 0:
                h = model.encode(xb[:, :-1])
                loss = F.cross_entropy(model.head(h).reshape(-1, 256), xb[:, 1:].reshape(-1))
                if a.mtp:
                    loss = loss + a.mtp_weight * mtp_loss(h, xb, model.mtp_heads)
                if a.topic_loss > 0:
                    loss = loss + a.topic_loss * topic_loss_fn(h, xb, model)
            else:
                loss = F.cross_entropy(model(xb[:, :-1]).reshape(-1, 256), xb[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 100 == 0:
            print(f"step {step} | bpb {loss.item()/math.log(2):.3f} | {(time.time()-t0)/3600:.2f}h", flush=True)
        if step % a.val_every == 0:
            v = validate()
            if v < best:
                best = v; save("ckpt_best", step, best)
            save("ckpt", step, best)                # resumable snapshot every val
            print(f"  val_bpb {v:.3f} (best {best:.3f})", flush=True)
    save("ckpt", a.max_steps, best)
    print(f"DONE {a.ckpt_dir} best={best:.3f}", flush=True)


if __name__ == "__main__":
    main()
