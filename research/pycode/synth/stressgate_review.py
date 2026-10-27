"""Independent review of ledger #8 (stressgate.py), task 3: causal isolation of the surprise gate.
Does NOT modify stressgate.py / bridge_cpu.py. Retrains cfc_stressgate (same train() path, same lr / steps / seed,
gate-lr-mult 1, 2 threads) per seed, saves the net, checks the positive control (test_a must equal the run8 json),
then evaluates eval-time interventions on the holder groups (wait_rest rate < 1/64; claimed 2-7):

  none            : unchanged (positive control)
  h_rate1         : holders rate 1 at every step (always overwrite)
  h_rate0_after   : holders rate 0 from the first PAD on (pure hold after the name; gate only on the name)
  h_rate0_wait    : holders rate 0 on the wait only (first PAD .. last PAD), gate elsewhere
  h_const_mean    : holders pinned to their mean rate over all real val positions (constant, no surprise)
  h_const_name    : holders pinned to their mean name rate (constant write rate, no hold)
  z_spanmean      : every group's z replaced by the val mean z of its span kind (name / first PAD / other PAD / SEP /
                    plan): keeps WHICH span we are in, drops all surprise variation inside a span
  z_globalmean    : z replaced by one global val mean (gate becomes a constant for every group)
  h_z_shuffle     : holders get the z of a random other position of the same row (same marginal, wrong timing)

Span kind is computed from x itself: n = index of the first PAD (name tokens are never PAD), K = data.delay.
"""

import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import bridge_cpu as bc  # noqa: E402
import stressgate as sg  # noqa: E402


class AblStressCfC(sg.StressCfC):
    """StressCfC.forward copied verbatim (stressgate mode) + intervention hooks; with abl=None it is identical."""

    abl = None            # dict(kind=..., groups=[...], value=tensor[g] or None, zspan=tensor[5], zglob=float, K=int)

    def kinds(self, x):
        K = self.abl["K"]
        B, L = x.shape
        n = (x == bc.PAD).float().argmax(1)                  # first PAD = first wait step
        t = torch.arange(L)[None]
        k = torch.full((B, L), 4)                            # plan
        k[t < n[:, None]] = 0                                # name
        k[t == n[:, None]] = 1                               # first PAD
        k[(t > n[:, None]) & (t < (n + K)[:, None])] = 2     # other PADs
        k[t == (n + K)[:, None]] = 3                         # SEP
        return k

    def forward(self, x):
        if self.abl is None or self.abl["kind"] == "none":
            return super().forward(x)
        A = self.abl
        e = self.emb(x)
        B, L = x.shape
        kind = self.kinds(x)
        hm = torch.zeros(self.g, dtype=torch.bool)
        hm[list(A["groups"])] = True
        s = torch.zeros(B, self.h)
        out = []
        zrow = []
        if A["kind"] == "h_z_shuffle":                       # first pass: record true z per position
            zs_saved = self.zs
            self.abl, self.zs = None, []
            super().forward(x)
            ztrue = torch.stack(self.zs, 1)                  # [B, L]
            self.abl, self.zs = A, zs_saved
            g = torch.Generator().manual_seed(11)
            perm = torch.stack([torch.randperm(L, generator=g) for _ in range(B)])
            zshuf = torch.gather(ztrue, 1, perm)
        for t in range(L):
            b = F.silu(self.bb(torch.cat([e[:, t], s], -1)))
            gate = torch.sigmoid(self.ta(b) + self.tb(b))
            c = torch.tanh(self.f1(b)) * (1 - gate) + gate * torch.tanh(self.f2(b))
            y = e[:, t].detach() @ self.R
            mse = ((self.pred(s) - y) ** 2).mean(-1)
            z = torch.log(mse.detach() + 1e-3)[:, None]
            if A["kind"] == "z_spanmean":
                z = A["zspan"][kind[:, t]][:, None]
            elif A["kind"] == "z_tokmean":                   # val mean z per (span kind, input token id)
                z = A["ztok"][kind[:, t], x[:, t]][:, None]
            elif A["kind"] == "z_globalmean":
                z = torch.full_like(z, A["zglob"])
            rate = torch.sigmoid(self.wg(z))                 # [B, g]
            if A["kind"] == "h_z_shuffle":
                rs = torch.sigmoid(self.wg(zshuf[:, t:t + 1]))
                rate = torch.where(hm[None], rs, rate)
            elif A["kind"] == "h_rate1":
                rate = torch.where(hm[None], torch.ones_like(rate), rate)
            elif A["kind"] in ("h_const_mean", "h_const_name"):
                rate = torch.where(hm[None], A["value"][None].expand_as(rate), rate)
            elif A["kind"] == "h_rate0_after":
                pin = hm[None] & (kind[:, t] >= 1)[:, None]
                rate = torch.where(pin, torch.zeros_like(rate), rate)
            elif A["kind"] == "h_rate0_wait":
                pin = hm[None] & ((kind[:, t] == 1) | (kind[:, t] == 2))[:, None]
                rate = torch.where(pin, torch.zeros_like(rate), rate)
            if self.zs is not None:
                self.zs.append(z.mean(-1))
            if self.rates is not None:
                self.rates.append(rate)
            rate = rate.repeat_interleave(self.h // self.g, -1)
            s = s + (c - s) * rate
            out.append(s)
        self.aux_loss = None
        return self.head(torch.stack(out, 1))


_sg_make = sg.make


def make(arm, V, delay=0):
    if arm == "cfc_stressgate":
        return AblStressCfC(V, [2.0 ** i for i in range(8)], mode="stressgate")
    return _sg_make(arm, V, delay)


@torch.no_grad()
def val_stats(net, data):
    """per-kind mean z and per-group mean rate (all real positions, name positions), from the unmodified net."""
    tasks = data.sets["val"]
    seqs = [data.seq(t) for t in tasks]
    x = bc.batchify(seqs)[:, :-1]
    net.abl = {"kind": "none", "K": data.delay}
    kind = net.kinds(x)
    real = torch.zeros(x.shape, dtype=torch.bool)
    for r, q in enumerate(seqs):
        real[r, :len(q) - 1] = True
    net.rates, net.zs = [], []
    net.eval()
    net(x)
    rr, zz = torch.stack(net.rates, 1), torch.stack(net.zs, 1)
    net.rates, net.zs = None, None
    zspan = torch.stack([zz[real & (kind == k)].mean() for k in range(5)])
    V = int(net.emb.num_embeddings)
    ztok = zspan[:, None].repeat(1, V).clone()               # fallback: span mean for (kind, token) unseen on val
    for k in range(5):
        for v in x[real & (kind == k)].unique().tolist():
            m = real & (kind == k) & (x == v)
            ztok[k, v] = zz[m].mean()
    return {"zspan": zspan, "ztok": ztok, "zglob": zz[real].mean().item(),
            "rate_all": rr[real].mean(0), "rate_name": rr[real & (kind == 0)].mean(0),
            "rate_wait_rest": rr[real & (kind == 2)].mean(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", required=True, help="run8 arm_cfc_stressgate_d48.json (positive control)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--conds", default="", help="comma list; default = all original conditions")
    ap.add_argument("--tag", default="", help="suffix of the result json")
    a = ap.parse_args()
    bc.GATE_LR_MULT = 1.0
    torch.set_num_threads(a.threads)
    sg.make = make                                           # sg.train calls make() from its own module globals
    bc.make, bc.train = make, sg.train
    data = bc.Data(48)
    os.makedirs(a.out, exist_ok=True)
    pt = f"{a.out}/net_cfc_stressgate_d48_s{a.seed}.pt"
    net = make("cfc_stressgate", data.vocab, 48)
    if os.path.exists(pt):
        net.load_state_dict(torch.load(pt))
    else:
        net, info = sg.train("cfc_stressgate", data, a.seed, a.lr, a.steps)
        torch.save(net.state_dict(), pt)
    st = val_stats(net, data)
    holders = [g for g, v in enumerate(st["rate_wait_rest"].tolist()) if v < 1 / 64]
    ref = json.load(open(a.ref))["runs"][a.seed]["eval"]
    res = {"seed": a.seed, "lr": a.lr, "steps": a.steps, "holders": holders,
           "zspan": [round(v, 3) for v in st["zspan"].tolist()], "zglob": round(st["zglob"], 3),
           "rate_all": [round(v, 4) for v in st["rate_all"].tolist()],
           "rate_name": [round(v, 4) for v in st["rate_name"].tolist()],
           "rate_wait_rest": [round(v, 4) for v in st["rate_wait_rest"].tolist()],
           "ref_test_a": ref["test_a"]["pass1"], "ref_train": ref["train"]["pass1"], "eval": {}}
    conds = ["none", "h_rate1", "h_rate0_after", "h_rate0_wait", "h_const_mean", "h_const_name",
             "z_spanmean", "z_globalmean", "h_z_shuffle"]
    if a.conds:
        conds = a.conds.split(",")
    for c in conds:
        net.abl = {"kind": c, "groups": holders, "K": 48, "zspan": st["zspan"], "zglob": st["zglob"], "ztok": st["ztok"],
                   "value": st["rate_all"] if c == "h_const_mean" else st["rate_name"]}
        ev = bc.evaluate(net, data, ["train", "test_a"])
        res["eval"][c] = {k: {"pass1": v["pass1"], "block_exact": v["block_exact"], "name_info": v.get("name_info")}
                          for k, v in ev.items()}
        print(json.dumps({"seed": a.seed, "cond": c, "train": ev["train"]["pass1"], "test_a": ev["test_a"]["pass1"],
                          "name_info_test_a": ev["test_a"]["name_info"]}), flush=True)
        json.dump(res, open(f"{a.out}/ablate_s{a.seed}{a.tag}.json", "w"), indent=1)
    net.abl = None


if __name__ == "__main__":
    main()
