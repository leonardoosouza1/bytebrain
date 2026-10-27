"""Ledger #8 (and #45): "T tao grande que o dado fica la e so muda se houver um 'estresse'... o caso da def de soma
que so trocou um sinal". A tau bank whose groups HOLD (tau -> infinity) and are rewritten only when a SURPRISE
(stress) signal fires. Registered 2026-10-27 09:47 (clock from `date`), before any run.

Own-bug check of bridge_cpu.py cfc_stress (rule 7), read from the code and log_cfc_stress_d48.txt:
  1. Floor 1/tau (structural, same bug class as cfc_gate in adapt 2/3): rate = 1/tau + (1 - 1/tau) * sigmoid(.), so
     a group can only leak FASTER than its tau, never hold longer. Over 48 wait steps only tau 64/128 keep > 45 %.
     The ablation of cfc_hold showed the name crosses the wait only through groups at rate 0.000; cfc_stress could
     never produce such a group. This alone caps the arm near cfc_bank (.013).
  2. The 'stress' signal was not a surprise signal: mis = |c - s|, c being a FRESH candidate every step. On the 48
     PAD inputs c(PAD, s) != s, so mismatch is large exactly when nothing new comes in; opening on it overwrites the
     memory. The optimizer answered by shutting the slow groups (sb -> -10..-15, sa 10..16, logged final values).
  3. Init/lr: sigmoid(4*mis - 8) starts at ~0.001 (gradient ~1e-3) but with 10x lr the scalars moved a lot (sa
     4 -> 16), so saturation did not freeze them; not the cause. Padding / ctx / grader are shared with gru (fits).
  Verdict: own bugs 1 (floor) and 2 (wrong signal). Fixed arm cfc_stressfix = same mismatch signal |c - s|, NO
  floor, init at the bank's rates (like cfc_hold), gate lr x1 (like cfc_hold). It tests bug 1 alone.

New arm cfc_stressgate (his idea with a real surprise signal), identical to cfc_hold except the group rate:
  y_t   = R e_t.detach()  (fixed random projection 128 -> 16 of the input embedding, a buffer, not trained)
  yhat  = P s_{t-1}       (P = Linear(128, 16): predicts the NEXT input from the held state)
  z_t   = log(mean((yhat - y_t)^2) + 1e-3).detach()    one scalar surprise per sequence and step
  rate_g = sigmoid(a_g * z_t + b_g)                     no floor; a_g init 1, b_g init logit(1/tau_g)
  learned threshold per group: z* = -b_g / a_g. The gate sees ONLY the surprise (detached), not the input content,
  so it cannot become cfc_hold's input gate. P is trained only by an auxiliary loss 0.1 * mean(mse) over the real
  input positions (the next-input prediction that defines 'surprise'). +2064 (P) +16 (a, b) params vs cfc_hold's
  +1032 (wg): non-embedding 101024 vs 99976 (+1.05 %, within 5 %; cfc_stressfix 98960, -1.0 %; gru 99072; measured 09:48 by `stressgate.py params`). Same K=48, data, 1200 steps, batch, schedule,
  eval, greedy decoding, lr grid rule. Expected spans: name tokens unpredictable -> z high -> write; PAD after the
  first one perfectly predictable -> z low -> hold; plan symbols become predictable as the net learns the plan.

Budget (rule 8, cut declared): gru and cfc_hold run at their already-tuned K=48 lrs (gru 2e-2 picked from a 6-lr
grid, cfc_hold 1e-2 from a 5-lr grid, adapt 3), --no-extend; cfc_stressgate and cfc_stressfix get a 3-lr grid
6e-3,1e-2,2e-2 with the top-edge extension. So the new arms get LESS lr search than the baselines (conservative).
1200 steps (adapt-3 budget, where cfc_hold = .743, gru = .942). 3 seeds each.
Positive control: cfc_hold seed 0 must give test_a ~ .800 (adapt 3), gru seed 0 ~ .972 (bridge_cpu45 K=48);
threads 2 instead of 1, so equal within ~3 pp, not bit-exact.

Prediction (09:47): cfc_stressgate K=48 test_a mean .55 (range .3-.8), below cfc_hold .743 and far below gru .942.
Mechanism: z is one scalar shared by all groups, so the write/hold decision is a 1-D threshold per group; name
tokens are partly predictable BPE pieces (low z on some -> under-written name), and SEP / first PAD / plan symbols
are also surprising early in training, which rewrites the holders. Gate-open prediction: mean rate of the holder
groups on the wait (PAD 2..48) < .02 and on the name > 10x that (the gate opens on surprise); first PAD rate high.
cfc_stressfix: .10 (range .0-.4): without the floor it can hold, but |c - s| is large on PAD so the learned
threshold must shut the holders to rate ~0 on every input, which also blocks the write.
KILL of 'surprise-only gate holds the name': cfc_stressgate <= cfc_bank K=48 (.013) + 10 pp.
Success for his #8: cfc_stressgate >= cfc_hold - 5 pp with holder rate on wait < .02 and name/wait ratio > 10.

Adaptation --wait cycle (2026-10-27 11:27 from `date`, registered before any code or run). Review 10-27 found the
detached gate (.723/.728/.741, lr 6e-3) acting as "freeze while the input repeats (PAD after PAD, z ~ -4.7), write on
any new token": a state-free z lookup per (span, token id) kept .696. Test of his formulation with repetition removed:
  --wait pad    (default) K wait inputs = PAD id 0, bit-identical to the code above.
  --wait cycle  K wait inputs = w0 w1 w2 w3 w0 w1 ..., 4 dedicated ids appended after the name tokens (vocab + 4; never
                targets, inputs only; name ids and plan ids unchanged). Start phase per example = crc32(true name) % 4
                (random-looking, fixed per task, so the true/swapped name_info pair shares it). Consecutive inputs always
                differ: "input repeated" is no longer a cue. A real surprise gate can learn the cycle in P (z low ->
                holders freeze); the first wait token is surprising (unknown phase); a repeat detector writes every
                step and erases the name. Shared Data path, so gru / cfc_hold / cfc_stressgate are comparable.
Runs: gru, cfc_hold, cfc_stressgate --pred-detach; --wait cycle --delay 48 --steps 1200 --seeds 3; lr grid
3e-3,6e-3,1e-2,2e-2 for every arm (best at the edge -> extend). Positive control first: --wait pad, gru, seed 0,
lr 2e-2 must give test_a .9719 (review fair run); else stop.
Pre-registered verdict (pad reference cfc_stressgate .731):
  PASS (his idea works without repetition): cfc_stressgate cycle test_a mean within 5 pp of .731 (>= .681) AND the
       holders' (groups 2-7) mean rate on wait inputs 2..K < .02.
  FAIL as a surprise gate: the holders' wait rate rises toward their name rate and test_a collapses.
  In between (one of the two PASS conditions only): PARTIAL, reported with the mechanism.
  Baselines gru and cfc_hold under cycle; all comparisons paired by seed (mean diff + per seed).
Predictions (11:27): gru cycle ~.93 (.88-.97; a content-reading update gate ignores the cycle). cfc_hold ~.72 (.55-.85;
its input gate reads e_t, so 4 wait ids are as easy to hold on as 1). cfc_stressgate ~.62 (.30-.76), holder wait-rest
rate .005-.03 (pad: .002-.0055), wait-rest z -3.5 .. -2 (pad -4.7: a linear P on s must map 4 phase states to 4
targets, less exact than a fixed point). P(PASS) ~45 %. Watch-out declared now: the wait embeddings are trained by the
CE loss and decayed by AdamW, so the net can make w0..w3 nearly equal (or ~0); then z is low because the inputs became
a repeat in embedding space, not because P learned the cycle. Diagnostic reported: pairwise spread of R e_w (and e_w)
over the 4 wait ids vs over name tokens, and z on wait-rest vs pad.
Result (11:58; out .../stressgate8/cycle). Positive control exact: pad gru s0 lr 2e-2 test_a .9719 (same line as review).
Cycle test_a (pad paired by seed): gru lr 2e-2 .877/.897/.831 = .868 (pad .942, diff -.073; 4e-2 extension val .96
< .993, so 2e-2 is interior). cfc_hold lr 6e-3 .000/.892/.472 = .455 (pad .743, diff -.289; seed 0 val 0 at all 4
lrs, the lr was picked on nats). cfc_stressgate lr 1e-2 (interior) .309/.505/.327 = .380 (pad .731, diff -.351; train
.67/.90/.67). Holders 2-7 wait-rest rate .152/.143/.187 (pad .006/.004/.002) vs name .240/.221/.220. Groups 2-3 lost
their gate (a -> ~0, constant rate .5-.8). Groups 4-7 (3-7 in seed 1) still freeze on the non-repeating wait: rate
.004-.009 vs name ~.10. Wait-rest z -3.0/-3.5/-2.5 (pad -4.7; name -.2), so P partly learned the cycle. R e spread
ratio wait/name .83/.58/.69: the wait embeddings were pulled together but not collapsed (gru's emb ratio was .53-.75 too).
Verdict: PASS NO (test_a -35 pp, holders 2-7 > .02). FAIL-as-surprise-gate only half: test_a collapsed, but the slow
groups that kept a gate did not rise to the name rate. A pure repeat detector would write on every cycle step, and
these groups do not. The loss: fewer holder groups (4-5 of 8 vs 6) and lower name_info (26-34 vs ~45 nats).
The cycle also cost gru -7 pp and cfc_hold -29 pp, so part of the drop is the task, not the gate.

Usage: stressgate.py arm --arm cfc_stressgate --delay 48 --out DIR [--lrs ... --seeds 3 --steps 1200 --wait cycle]
"""

import argparse
import json
import math
import os
import random
import sys
import time
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import bridge_cpu as bc  # noqa: E402

NEW_ARMS = ["cfc_stressgate", "cfc_stressfix"]
AUX_W = 0.1
PRED_DETACH = False   # review 10-27: True = aux loss trains only P (docstring claim); False = original run8
N_WAIT = 4            # --wait cycle: number of distinct wait ids in the cycle


class WaitData(bc.Data):
    """bc.Data with a choice of wait inputs. pad: bc.Data unchanged. cycle: w0 w1 w2 w3 w0 ... from phase
    crc32(true name) % N_WAIT; the wait ids are appended after the name tokens (never targets)."""

    def __init__(self, delay=0, wait="pad"):
        super().__init__(delay)
        self.wait = wait
        self.wait_ids = []
        if wait == "cycle":
            self.wait_ids = list(range(self.vocab, self.vocab + N_WAIT))
            self.vocab += N_WAIT

    def phase(self, task):
        return zlib.crc32(bc.name_of(*task).encode()) % N_WAIT

    def prefix(self, task, name=None):
        if self.wait == "pad":
            return super().prefix(task, name)
        p = self.phase(task)
        return (name if name is not None else self.name_ids(task)) + \
            [self.wait_ids[(p + i) % N_WAIT] for i in range(self.delay)]


class GroupAffine(nn.Module):
    """per-group scalar weight and bias, named wg.weight / wg.bias so bridge_cpu's optimizer groups treat them
    exactly as cfc_hold's gate (no weight decay, lr x GATE_LR_MULT)."""

    def __init__(self, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones_like(bias))
        self.bias = nn.Parameter(bias.clone())

    def forward(self, z):                                   # z [B, g] or [B, 1]
        return z * self.weight + self.bias


class StressCfC(bc.CfC):
    def __init__(self, V, taus, h=128, mode="stressgate", proj=16):
        super().__init__(V, taus, h, mode="fixed")
        self.mode = mode
        b0 = torch.logit(torch.tensor(taus, dtype=torch.float).reciprocal().clamp(max=0.99))
        self.wg = GroupAffine(b0)
        if mode == "stressgate":
            self.pred = nn.Linear(h, proj)
            g = torch.Generator().manual_seed(1234)
            self.register_buffer("R", torch.randn(h, proj, generator=g) / math.sqrt(h))
        self.zs = None                                      # diagnostic: surprise per step
        self.aux_len = None                                 # train: true input lengths -> aux loss mask
        self.aux_loss = None

    def forward(self, x):
        e = self.emb(x)
        B, L = x.shape
        s = torch.zeros(B, self.h)
        out, aux = [], []
        for t in range(L):
            b = F.silu(self.bb(torch.cat([e[:, t], s], -1)))
            gate = torch.sigmoid(self.ta(b) + self.tb(b))
            c = torch.tanh(self.f1(b)) * (1 - gate) + gate * torch.tanh(self.f2(b))
            if self.mode == "stressgate":
                y = e[:, t].detach() @ self.R
                mse = ((self.pred(s.detach() if PRED_DETACH else s) - y) ** 2).mean(-1)    # [B] prediction error of this input from s_{t-1}
                aux.append(mse)
                z = torch.log(mse.detach() + 1e-3)[:, None]
                rate = torch.sigmoid(self.wg(z))            # [B, g]
            else:                                           # stressfix: mismatch |c - s| per group, no floor
                z = (c - s).abs().view(B, self.g, -1).mean(-1)
                rate = torch.sigmoid(self.wg(z))
            if self.zs is not None:
                self.zs.append(z.mean(-1))
            rate = rate.repeat_interleave(self.h // self.g, -1)
            if self.force is not None:
                gs, v = self.force
                m = torch.zeros(self.g, dtype=torch.bool)
                m[list(gs)] = True
                rate = torch.where(m.repeat_interleave(self.h // self.g), torch.full_like(rate, v), rate)
            if self.rates is not None:
                self.rates.append(rate.view(-1, self.g, self.h // self.g).mean(-1))
            s = s + (c - s) * rate
            out.append(s)
        if aux and self.aux_len is not None:
            a = torch.stack(aux, 1)
            m = (torch.arange(L)[None] < self.aux_len[:, None]).float()
            self.aux_loss = (a * m).sum() / m.sum()
        else:
            self.aux_loss = None
        return self.head(torch.stack(out, 1))


_orig_make, _orig_train, _orig_rbi = bc.make, bc.train, bc.rate_by_input


def make(arm, V, delay=0):
    if arm in NEW_ARMS:
        return StressCfC(V, [2.0 ** i for i in range(8)], mode=arm[4:])
    return _orig_make(arm, V, delay)


def train(arm, data, seed, lr, steps, batch=48):
    """bridge_cpu.train unchanged for the old arms (positive control through the real path); for the new arms the
    same loop plus the auxiliary next-input loss (stressgate only)."""
    if arm not in NEW_ARMS:
        net, info = _orig_train(arm, data, seed, lr, steps, batch)
        if isinstance(net, bc.CfC):                         # extra diagnostics only; the net is unchanged
            info["span_stats"] = span_stats(net, data, data.sets["val"])
        if getattr(data, "wait_ids", None):
            info["wait_spread"] = wait_spread(net, data)
        return net, info
    torch.manual_seed(seed)
    net = make(arm, data.vocab, data.delay)
    nd = {"wg.weight", "wg.bias"}
    groups = [{"params": [p for n, p in net.named_parameters() if n not in nd]},
              {"params": [p for n, p in net.named_parameters() if n in nd], "weight_decay": 0.0,
               "lr_mult": bc.GATE_LR_MULT}]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01)
    r = random.Random(seed + 100)
    tr = data.sets["train"]
    t0, losses, auxs = time.perf_counter(), [], []
    for s in range(steps):
        lr_s = lr * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        for gp in opt.param_groups:
            gp["lr"] = lr_s * gp.get("lr_mult", 1.0)
        ts = [tr[r.randrange(len(tr))] for _ in range(batch)]
        seqs = [data.seq(t) for t in ts]
        x = bc.batchify(seqs)
        net.aux_len = torch.tensor([len(q) - 1 for q in seqs])
        logits = net(x[:, :-1])
        net.aux_len = None
        ce = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
        w = torch.zeros_like(ce)
        for row, t in enumerate(ts):
            n = len(data.prefix(t))
            w[row, n:n + len(data.target_ids(t))] = 1.0
        loss = (ce * w).sum() / w.sum()
        losses.append(loss.item())
        if net.aux_loss is not None:
            auxs.append(net.aux_loss.item())
            loss = loss + AUX_W * net.aux_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    info = {"train_loss_last100": round(sum(losses[-100:]) / 100, 4), "train_wall_s": round(time.perf_counter() - t0, 1)}
    if auxs:
        info["aux_mse_last100"] = round(sum(auxs[-100:]) / 100, 4)
    info["gate_a"] = [round(v, 3) for v in net.wg.weight.tolist()]
    info["gate_b"] = [round(v, 3) for v in net.wg.bias.tolist()]
    info["rate_by_input"] = bc.rate_by_input(net, data, data.sets["val"])
    info["span_stats"] = span_stats(net, data, data.sets["val"])
    if getattr(data, "wait_ids", None):
        info["wait_spread"] = wait_spread(net, data)
    return net, info


@torch.no_grad()
def wait_spread(net, data):
    """mean pairwise squared distance per dim among the wait-id embeddings vs among all name-token embeddings;
    for stressgate also in the predicted space R e. Low wait/name ratio = the net made the wait ids alike
    (a repeat in embedding space)."""
    E = net.emb.weight if hasattr(net, "emb") else None
    if E is None:
        return None
    names = sorted(data.to_c.values())
    def spread(M):
        d = ((M[:, None] - M[None]) ** 2).mean(-1)
        n = M.shape[0]
        return d.sum().item() / (n * (n - 1))
    out = {}
    spaces = [("emb", lambda M: M)] + ([("Re", lambda M: M @ net.R)] if hasattr(net, "R") else [])
    for tag, f in spaces:
        w, nm = spread(f(E[data.wait_ids])), spread(f(E[names]))
        out[tag] = {"wait": round(w, 4), "name": round(nm, 4), "ratio": round(w / nm, 4)}
    return out


@torch.no_grad()
def span_stats(net, data, tasks):
    """gate-open rate per group and mean surprise z on: name tokens (changed), first wait PAD (changed: name -> wait),
    wait PAD 2..K (stable), SEP, plan inputs."""
    seqs = [data.seq(t) for t in tasks]
    x = bc.batchify(seqs)[:, :-1]
    net.rates, net.zs = [], []
    net.eval()
    net(x)
    net.train()
    r = torch.stack(net.rates, 1)
    z = torch.stack(net.zs, 1) if net.zs else None          # old CfC arms record no surprise
    net.rates, net.zs = None, None
    kind = torch.full(x.shape, -1)
    for row, t in enumerate(tasks):
        n, K = len(data.name_ids(t)), data.delay
        kind[row, :n] = 0
        if K:
            kind[row, n] = 1
            kind[row, n + 1:n + K] = 2
        kind[row, n + K] = 3
        kind[row, n + K + 1:len(seqs[row]) - 1] = 4
    out = {}
    for k, name in enumerate(("name", "wait_first", "wait_rest", "sep", "plan")):
        m = kind == k
        if bool(m.any()):
            out[name] = {"rate": [round(v, 4) for v in r[m].mean(0).tolist()],
                         "z": round(z[m].mean().item(), 3) if z is not None else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arm", "params"])
    ap.add_argument("--arm", choices=bc.ARMS + NEW_ARMS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--delay", type=int, default=48)
    ap.add_argument("--lrs", default="6e-3,1e-2,2e-2")
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gate-lr-mult", type=float, default=1.0)
    ap.add_argument("--pred-detach", action="store_true")
    ap.add_argument("--wait", choices=["pad", "cycle"], default="pad", help="wait inputs: PAD (old) or a 4-id cycle")
    a = ap.parse_args()
    bc.GATE_LR_MULT = a.gate_lr_mult
    global PRED_DETACH
    PRED_DETACH = a.pred_detach
    torch.set_num_threads(a.threads)
    bc.make, bc.train = make, train
    bc.Data = lambda delay=0: WaitData(delay, a.wait)       # shared data path for every arm
    if a.cmd == "params":
        data = bc.Data(a.delay)
        print(json.dumps({arm: bc.nonemb(make(arm, data.vocab, a.delay)) for arm in ["gru", "cfc_hold"] + NEW_ARMS}))
        return
    bc.cmd_arm(a)


if __name__ == "__main__":
    main()
