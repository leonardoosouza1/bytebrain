"""Symbol salience: which in-scope local symbol is used next?

Each local symbol of a function gets its own small recurrent state, updated token by
token from (a) a shared embedding of the current token category and (b) whether this
symbol occurs here (load/store). At every load of a local symbol, the model scores
all symbols born so far and must pick the one actually used.

Arms:
  recency    last-mentioned symbol wins (no learning)
  frequency  most-mentioned symbol wins, ties by recency (no learning)
  feat       MLP over hand features: log gap since last use, log count, is_param, query context
  tau1       liquid cell, every unit starts at the same time constant (tau=8)
  taubank    liquid cell, units start at tau in {1, 4, 16, 64}
  gru        GRU cell, same width
  taubank_bytes  taubank + byte embedding of the symbol name

Usage: train.py --data DATA.pt --out DIR --arms a,b --seeds 0,1,2
"""

import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

D, E, Q_CTX = 32, 16, 3


def to_dev(split, device, limit=None):
    n = split["cats"].shape[0] if limit is None else min(limit, split["cats"].shape[0])
    out = {k: split[k][:n].to(device).long() for k in ("cats", "sids", "store", "is_param", "len")}
    names = split["names"][:n]
    S = split["is_param"].shape[1]
    nb = torch.zeros((n, S, 16), dtype=torch.long)
    for i, nm in enumerate(names):
        for s, name in enumerate(nm):
            b = list(name.encode()[:16])
            nb[i, s, :len(b)] = torch.tensor(b) + 1
    out["name_bytes"] = nb.to(device)
    out["n_sym"] = torch.tensor([len(nm) for nm in names], device=device)
    return out


def batch_tensors(b):
    """Occurrence one-hots, born mask, query mask and targets for a batch."""
    sids, store, cats = b["sids"], b["store"], b["cats"]
    N, T = sids.shape
    S = b["is_param"].shape[1]
    sym = torch.arange(S, device=sids.device)
    occ = (sids.unsqueeze(1) == sym.view(1, S, 1))                    # [N,S,T]
    occ_store = occ & (store.unsqueeze(1) == 1)
    occ_load = occ & (store.unsqueeze(1) == 0)
    born_after = occ_store.long().cummax(dim=2).values.bool()          # born at or before t
    born_before = torch.zeros_like(born_after)
    born_before[:, :, 1:] = born_after[:, :, :-1]                      # born strictly before t
    tgt = sids.clamp(min=0)
    is_load = (sids >= 0) & (store == 0)
    tgt_born = born_before.gather(1, tgt.unsqueeze(1)).squeeze(1)
    n_cand = born_before.sum(1)
    query = is_load & tgt_born & (n_cand >= 2)
    return occ_load, occ_store, born_before, query, tgt, n_cand


class SalienceNet(nn.Module):
    def __init__(self, arm, n_cat):
        super().__init__()
        self.arm = arm
        self.emb = nn.Embedding(n_cat, E)
        self.byte_emb = nn.Embedding(257, 8, padding_idx=0) if arm == "taubank_bytes" else None
        nin = E + 3 + (8 if self.byte_emb is not None else 0)
        if arm == "gru":
            self.cell = nn.GRUCell(nin, D)
        elif arm in ("tau1", "taubank", "taubank_bytes"):
            self.inp = nn.Linear(nin + D, D)
            self.gate = nn.Linear(nin, D)
            if arm == "tau1":
                tau = torch.full((D,), 8.0)
            else:
                tau = torch.tensor([1.0, 4.0, 16.0, 64.0]).repeat_interleave(D // 4)
            self.log_tau = nn.Parameter(tau.log())
        sdim = 3 if arm == "feat" else D
        self.head = nn.Sequential(nn.Linear(sdim + Q_CTX * E, 64), nn.ReLU(), nn.Linear(64, 1))

    def step(self, h, x):
        if self.arm == "gru":
            return self.cell(x, h)
        # liquid update: input-dependent effective time constant, tau_eff = tau / (1 + softplus(gate))
        rate = (1.0 + F.softplus(self.gate(x))) / self.log_tau.exp()
        a = torch.exp(-rate)
        g = torch.tanh(self.inp(torch.cat([x, h], -1)))
        return a * h + (1 - a) * g

    def forward(self, b, tensors):
        occ_load, occ_store, born_before, query, tgt, n_cand = tensors
        cats = b["cats"]
        N, T = cats.shape
        S = b["is_param"].shape[1]
        ce = self.emb(cats)                                                # [N,T,E]
        # query context: embeddings of the Q_CTX previous tokens
        pads = [F.pad(ce, (0, 0, k, 0))[:, :T] for k in range(1, Q_CTX + 1)]
        qctx = torch.cat(pads, -1)                                         # [N,T,Q*E]
        is_param = b["is_param"].float().unsqueeze(-1)                     # [N,S,1]
        if self.arm == "feat":
            occ_any = (occ_load | occ_store).float()
            count = occ_any.cumsum(2)
            t_idx = torch.arange(T, device=cats.device).float().view(1, 1, T)
            last = torch.where(occ_any.bool(), t_idx, torch.full_like(t_idx, -1e9).expand_as(occ_any))
            last = last.cummax(2).values
            # features at time t must use info strictly before t
            count_b = F.pad(count, (1, 0))[:, :, :T]
            last_b = F.pad(last, (1, 0), value=-1e9)[:, :, :T]
            gap = (t_idx - last_b).clamp(min=1, max=1e4)
            feats = torch.stack([gap.log(), (count_b + 1).log(), is_param.expand(-1, -1, T).squeeze(-1)], -1)
            states = feats                                                 # [N,S,T,3]
        else:
            extra = None
            if self.byte_emb is not None:
                nb = self.byte_emb(b["name_bytes"])                       # [N,S,16,8]
                m = (b["name_bytes"] > 0).float().unsqueeze(-1)
                extra = (nb * m).sum(2) / m.sum(2).clamp(min=1)            # [N,S,8]
            h = torch.zeros(N * S, D, device=cats.device)
            hs = []
            for t in range(T):
                hs.append(h.view(N, S, D))                                 # state BEFORE token t
                parts = [ce[:, t].unsqueeze(1).expand(N, S, E), occ_load[:, :, t].float().unsqueeze(-1),
                         occ_store[:, :, t].float().unsqueeze(-1), is_param]
                if extra is not None:
                    parts.append(extra)
                x = torch.cat(parts, -1).view(N * S, -1)
                h = self.step(h, x)
            states = torch.stack(hs, 2)                                    # [N,S,T,D]
        qn, qt = query.nonzero(as_tuple=True)
        st = states[qn, :, qt]                                             # [Q,S,sdim]
        qc = qctx[qn, qt].unsqueeze(1).expand(-1, S, -1)
        logits = self.head(torch.cat([st, qc], -1)).squeeze(-1)            # [Q,S]
        cand = born_before[qn, :, qt]
        logits = logits.masked_fill(~cand, -1e9)
        return logits, tgt[qn, qt], cats[qn, (qt - 1).clamp(min=0)], n_cand[qn, qt]


def heuristic(b, tensors, kind):
    occ_load, occ_store, born_before, query, tgt, n_cand = tensors
    T = b["cats"].shape[1]
    occ_any = (occ_load | occ_store).float()
    t_idx = torch.arange(T, device=occ_any.device).float().view(1, 1, T)
    last = torch.where(occ_any.bool(), t_idx, torch.full_like(t_idx, -1.0).expand_as(occ_any)).cummax(2).values
    last_b = F.pad(last, (1, 0), value=-1.0)[:, :, :T]
    count_b = F.pad(occ_any.cumsum(2), (1, 0))[:, :, :T]
    score = last_b if kind == "recency" else count_b * 1e4 + last_b
    qn, qt = query.nonzero(as_tuple=True)
    s = score[qn, :, qt].masked_fill(~born_before[qn, :, qt], -1e18)
    return s, tgt[qn, qt], b["cats"][qn, (qt - 1).clamp(min=0)], n_cand[qn, qt]


def batches(data, bs, perm=None):
    n = data["cats"].shape[0]
    idx = torch.arange(n, device=data["cats"].device) if perm is None else perm
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        b = {k: v[j] for k, v in data.items()}
        L = int(b["len"].max())
        S = int(b["n_sym"].max())
        b = {k: (v[:, :L] if k in ("cats", "sids", "store") else v) for k, v in b.items()}
        b["is_param"] = b["is_param"][:, :S]
        b["name_bytes"] = b["name_bytes"][:, :S]
        yield b


def evaluate(model, data, arm, bs, return_id):
    hit = tot = hit_ret = tot_ret = hit3 = tot3 = 0
    with torch.no_grad():
        for b in batches(data, bs):
            tens = batch_tensors(b)
            if arm in ("recency", "frequency"):
                s, y, prev, nc = heuristic(b, tens, arm)
            else:
                s, y, prev, nc = model(b, tens)
            ok = s.argmax(-1) == y
            hit += int(ok.sum()); tot += ok.numel()
            r = prev == return_id
            hit_ret += int(ok[r].sum()); tot_ret += int(r.sum())
            m3 = nc >= 3
            hit3 += int(ok[m3].sum()); tot3 += int(m3.sum())
    return {"acc": hit / max(tot, 1), "n": tot, "acc_return": hit_ret / max(tot_ret, 1), "n_return": tot_ret,
            "acc_3plus": hit3 / max(tot3, 1), "n_3plus": tot3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="recency,frequency,feat,tau1,taubank,gru,taubank_bytes")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out-limit", type=int, default=4000)
    args = ap.parse_args()
    dev = "cuda"
    raw = torch.load(args.data, weights_only=False)
    vocab = raw["vocab"]
    return_id = vocab.index("kw:return")
    tr, va, ou = to_dev(raw["train"], dev), to_dev(raw["val"], dev), to_dev(raw["out"], dev, args.out_limit)
    results = []
    for arm in args.arms.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            if arm in ("recency", "frequency") and seed > 0:
                continue
            torch.manual_seed(seed)
            t0 = time.time()
            model = None
            if arm not in ("recency", "frequency"):
                model = SalienceNet(arm, len(vocab)).to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
                n = tr["cats"].shape[0]
                total = args.epochs * math.ceil(n / args.bs)
                sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total)
                for ep in range(args.epochs):
                    perm = torch.randperm(n, device=dev)
                    lsum = lcnt = 0
                    for b in batches(tr, args.bs, perm):
                        logits, y, _, _ = model(b, batch_tensors(b))
                        loss = F.cross_entropy(logits, y)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step(); sched.step()
                        lsum += float(loss.detach()); lcnt += 1
                    print(f"{arm} s{seed} ep{ep} loss {lsum / lcnt:.4f} ({time.time() - t0:.0f}s)", flush=True)
                model.eval()
            res = {"arm": arm, "seed": seed, "val": evaluate(model, va, arm, args.bs, return_id),
                   "out": evaluate(model, ou, arm, args.bs, return_id),
                   "params": sum(p.numel() for p in model.parameters()) if model else 0,
                   "train_s": round(time.time() - t0, 1)}
            if model is not None and hasattr(model, "log_tau"):
                res["tau_sorted"] = sorted(round(x, 2) for x in model.log_tau.exp().tolist())
            print(json.dumps(res), flush=True)
            results.append(res)
            with open(f"{args.out}/results.json", "w") as fh:
                json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
