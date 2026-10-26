"""Program search by derivative (ledger #24 + #26, CPU). v2 after the 2026-10-26 review (GUESS-AND-DERIVE.md sec 4, 6).

Input is the PROBLEM, not the name: n_ex input/output examples of a hidden synth2 task (units + final).
Each program slot is a softmax over all units (+ identity). A differentiable executor runs every unit on a soft
list state (values v [N, L], keep mask m [N, L]); the next state is the probability-weighted mix of all unit
outputs. Loss vs the target outputs; Adam moves the slot logits. Argmax -> discrete program -> exact check on the
examples (restart if wrong) -> bwrap grade on the 40 hidden synth2 INPUTS. The final (sum/len/max/...) is given.

Arms (all see the same problem; random gets the derive_only unit-application budget):
  derive_only   random init (+ Gumbel), Adam steps                          (the old "grad" arm)
  rand          random discrete programs, same budget as derive_only; also run to the full budget to count fits
  guess_only    argmax of the guess, 0 steps                                 (only if --guess-source != none)
  guess_derive  Adam steps starting from the guess logits                    (only if --guess-source != none)
  guess_sample  programs sampled per slot from softmax(guess logits), same unit-application budget as guess_derive:
                the control for "the derivative points the way" vs "searching near the guess" (repair = resampling,
                51.75 = 51.75 on 10-22)                                       (only if --guess-source != none)
  guess_derive_soft  as guess_derive but sharpness annealed t0 -> t1 like derive_only (secondary arm, declared before the run)
  guess_derive runs at constant sharpness t1 (warm start); every derive checks its start before the first step.
  guess_only, guess_derive and guess_sample all see the SAME single guess per problem; derive restarts > 0 add Gumbel
  noise to it (fixed 10-26 before the first run: restarts used to draw a fresh oracle guess, i.e. more oracle information).
Guess sources: none (random init), oracle_noisy (true program one-hot + noise: a stand-in for the planner,
NOT a learned guess), planner (TODO, not implemented).

Derive improvements, each behind a flag, default ON (--legacy or the --no-* flags reproduce the old numbers):
  anneal sharpness T 2 -> 30 (--no-anneal: fixed 20) | loss / (1 + mean target^2) (--no-norm)
  Gumbel noise on the init logits per restart (--no-gumbel) | reverse compacts kept items to the front (--no-compact-reverse)
  --stop-dloss eps: stop a restart when |loss_t - loss_{t-25}| < eps (idea #26); default 0 = off. Trace always recorded.

Metrics per length and arm: <arm>_fit8 (fits the examples), <arm>_hidden40_io (MAIN: output equals the true program on the
40 hidden inputs, strict=False), <arm>_hidden40_strict (secondary: list state before the final must also match),
<arm>_fit8_not_io(_given_fit) (fits the examples but is wrong on hidden io). Ambiguity per problem: number of distinct
programs fitting the examples (exact enumeration for len <= --exact-max, else distinct fits found by random search run to
its full budget, capped), and how many of them are io-equivalent on the 40 hidden inputs.
Results JSON is rewritten after every problem (partial) and every length.

PREDICTION (registered 10-26 before the run; --guess-source oracle_noisy --guess-noise 1.5, n 40, lens 2,3,4, defaults):
  Monte Carlo of the oracle guess at noise 1.5 over 47 logits: right slot 0.66, right program .44/.29/.19 (len2/3/4).
  P1 guess_only fit8 >= .44/.29/.19 (other programs may also fit the 8 examples).
  P2 (#26 core) guess_derive fit8 >= guess_only + 30 pp at len3 and len4; hidden40_io >= guess_only + 15 pp.
  P3 guess_sample at the same budget: hidden40_io >= guess_derive (sampling near the guess is the stronger baseline,
     as resampling tied repair before). If guess_derive wins by > 5 pp, the direction of the derivative is doing work.
  P4 derive_only with the 4 fixes: len2 fit8 >= 0.9 (old 0.67), still below rand fit8 at len2 and len3.
  P5 ambiguity, len2 exact enumeration: mean number of programs fitting the 8 examples >= 3.
  KILL (#26 core): guess_derive hidden40_io <= guess_only + 5 pp at len3 AND len4.
  The guess is an ORACLE stand-in, not the planner: a pass says "derive can correct a noisy guess", nothing about a
  learned guess.

Usage: gradsearch.py --out DIR [--n 40 --lens 2,3,4 --steps 300 --restarts 8 --n-ex 8 --guess-source oracle_noisy --guess-noise 1.0]
"""

import argparse
import functools
import itertools
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import synth2  # noqa: E402
from synth2 import FINAL, INPUTS, UNITS, grade, nargs  # noqa: E402

U = list(UNITS)                       # 46 units; slot choice index len(U) = identity
L, BIG, T_LEGACY = 12, 1e4, 20.0      # padded list length, masked sentinel, old fixed sharpness of soft comparisons


@functools.lru_cache(maxsize=200_000)
def _fn(units, final):
    """compiled composition of OUR unit code under a neutral name (a function named `abs` would shadow the builtin).
    k defaults to 0 like the hidden grade's found-program wrapper."""
    g = {}
    exec("def _f(xs, k=0):\n" + "".join(f"    {l}\n" for u in units for l in UNITS[u]) + f"    {FINAL[final]}\n", g)
    return g["_f"]


def py_run(units, final, xs, k):
    """trusted in-process run of a composition of OUR unit code (search scoring only; the grade uses bwrap)."""
    f = _fn(tuple(units), final)
    try:
        return f(list(xs), k) if nargs(units) == 2 else f(list(xs))
    except Exception:
        return "!"


def soft_unit(u, v, m, k, T, crev=True):
    sg = torch.sigmoid
    if u in ("sort",):
        key = torch.where(m > 0.5, v, torch.full_like(v, BIG))
        idx = torch.sort(key, dim=1, stable=True).indices
        return v.gather(1, idx), m.gather(1, idx)
    if u == "reverse":
        if not crev:                                   # legacy: kept items end up at the END of the padded row
            return v.flip(1), m.flip(1)
        pos = torch.arange(L).float()[None].expand_as(v)   # kept items reversed AND compacted to the front (like sort)
        idx = torch.sort(torch.where(m > 0.5, -pos, torch.full_like(v, BIG)), dim=1, stable=True).indices
        return v.gather(1, idx), m.gather(1, idx)
    if u == "negate":
        return -v, m
    if u == "abs":
        return v.abs(), m
    if u == "drop_first":
        return v, m * sg(T * (torch.cumsum(m, 1) - 1.5))
    if u == "cumsum":
        return torch.cumsum(v * m, 1), m
    if u == "running_max":
        return torch.cummax(torch.where(m > 0.5, v, torch.full_like(v, -BIG)), 1).values, m
    if u in ("unique", "dedup_adjacent"):
        eq = torch.exp(-(v[:, :, None] - v[:, None, :]) ** 2 * 25)         # [N, i, j] soft v_i == v_j
        prior = torch.tril(torch.ones(L, L), -1)[None]                      # j < i
        if u == "dedup_adjacent":                                          # only the last kept j < i counts
            kept = (m > 0.5).float()
            last = torch.cummax(torch.where(kept > 0, torch.arange(L).float()[None], torch.full_like(v, -1)), 1).values
            prev = torch.cat([torch.full_like(last[:, :1], -1), last[:, :-1]], 1)
            prior = (torch.arange(L).float()[None, None] == prev[:, :, None]).float()
        dup = (eq * prior * m[:, None, :]).clamp(max=1)
        return v, m * torch.prod(1 - dup, 2)
    op, _, n = u.rpartition("_")
    n = k if n == "k" else float(n)
    if op in ("add",):
        return v + n, m
    if op == "times":
        return v * n, m
    if op == "keep_gt":
        return v, m * sg(T * (v - n - 0.5))
    if op == "keep_lt":
        return v, m * sg(T * (n - 0.5 - v))
    if op == "clip":
        return torch.minimum(v, n * torch.ones_like(v)), m
    if op == "mod":
        return torch.remainder(v, n), m
    if op == "take":
        return v, m * sg(T * (n + 0.5 - torch.cumsum(m, 1)))
    raise KeyError(u)


def soft_final(final, v, m, T):
    if final == "sum":
        return (v * m).sum(1)
    if final == "len":
        return m.sum(1)
    if final == "max":                                                      # sharpness T/5 (= 4 at the legacy T=20)
        w = torch.softmax(torch.where(m > 0.5, v, torch.full_like(v, -BIG)) * (T / 5), 1)
        return (w * v).sum(1) * (m.sum(1) > 0.5)
    if final == "first":
        w = m * torch.cumprod(torch.cat([torch.ones_like(m[:, :1]), 1 - m[:, :-1]], 1), 1)
        return (w * v).sum(1)
    if final.startswith("count_gt_"):
        return (m * torch.sigmoid(T * (v - int(final.rsplit("_", 1)[1]) - 0.5))).sum(1)
    raise KeyError(final)


def list_loss(v, m, targets, scale=1.0):
    """kept element of rank r must equal target[r]; length must match. The value term is divided by `scale`."""
    r = torch.cumsum(m, 1) - 1
    loss = 0
    for i, t in enumerate(targets):
        tt = torch.tensor(t + [0.0] * (L - len(t)), dtype=torch.float32)
        ri = r[i].clamp(0, L - 1)
        lo = ri.floor().long()
        frac = ri - lo
        tv = tt[lo] * (1 - frac) + tt[(lo + 1).clamp(max=L - 1)] * frac
        loss = loss + (m[i] * (v[i] - tv) ** 2).sum() / (1 + len(t)) / scale + (m[i].sum() - len(t)) ** 2
    return loss / len(targets)


class Problem:
    def __init__(self, units, final, n_ex, seed):
        r = random.Random(seed)
        self.units, self.final = units, final
        self.ex = [(xs, k) for xs, k in r.sample(INPUTS, n_ex)]
        self.out = [py_run(units, final, xs, k) for xs, k in self.ex]
        self.v = torch.tensor([xs + [0] * (L - len(xs)) for xs, _ in self.ex], dtype=torch.float32)
        self.m = torch.tensor([[1.0] * len(xs) + [0.0] * (L - len(xs)) for xs, _ in self.ex])
        self.k = torch.tensor([float(k) for _, k in self.ex])[:, None]
        self.tgt = None if final == "" else torch.tensor([float(o) if o != "!" else 0.0 for o in self.out])
        vals = [float(x) for o in self.out if o != "!" for x in (o if final == "" else [o])]
        self.scale = 1.0 + (sum(x * x for x in vals) / len(vals) if vals else 0.0)   # 1 + mean(target^2)
        self.hid = None                                                    # truth outputs on the 40 hidden inputs

    def check(self, units):
        return all(py_run(units, self.final, xs, k) == o for (xs, k), o in zip(self.ex, self.out))

    def io_equiv(self, units):
        """output equals the true program on the 40 hidden inputs, called like the hidden grade (truth's arity, k=0 default)."""
        two = nargs(self.units) == 2

        def run(us, xs, k):
            try:
                f = _fn(tuple(us), self.final)
                return f(list(xs), k) if two else f(list(xs))
            except Exception:
                return "!"
        if self.hid is None:
            self.hid = [run(self.units, xs, k) for xs, k in INPUTS]
        return all(run(units, xs, k) == h for (xs, k), h in zip(INPUTS, self.hid))


def soft_loss(p, logits, T=T_LEGACY, norm=False, crev=False):
    """defaults = the legacy executor; main() passes the annealed T / norm / crev."""
    v, m = p.v, p.m
    for lg in logits:
        w = torch.softmax(lg, 0)
        vs, ms = [v], [m]                                   # index len(U) = identity
        for u in U:
            a, b = soft_unit(u, v, m, p.k, T, crev)
            vs.append(a)
            ms.append(b)
        order = [len(U)] + list(range(len(U)))              # stacked [identity, U...] -> weights reordered
        w = w[order]
        v = (w[:, None, None] * torch.stack(vs)).sum(0)
        m = (w[:, None, None] * torch.stack(ms)).sum(0)
        v = v.clamp(-BIG, BIG)
    sc = p.scale if norm else 1.0
    if p.final == "":
        return list_loss(v, m, p.out, sc)
    return ((soft_final(p.final, v, m, T) - p.tgt) ** 2).mean() / sc


def decode(logits):
    return [U[i] for i in (int(lg.argmax()) for lg in logits) if i < len(U)]


def temp_at(a, s):
    """sharpness at step s: geometric t0 -> t1 over the steps (legacy: fixed 20)."""
    return a.t0 * (a.t1 / a.t0) ** (s / max(1, a.steps - 1)) if a.anneal else T_LEGACY


def guess_logits(p, n_slots, source, g, noise=0.0, gumbel=0.0, strength=4.0):
    """-> n_slots leaf logit tensors [len(U)+1] (last = identity). Pluggable source:
    none          random init N(0, 0.5) (+ Gumbel * `gumbel`): the derive_only start
    oracle_noisy  true program one-hot (`strength`) + N(0, noise) per logit. ORACLE stand-in for the planner, NOT a learned guess.
    planner       TODO: blockaddr planner reading the NAME (never the examples); needs a per-slot distribution
                  [n_slots, len(U)+1] over U from the name, e.g. softmax of its unit-token logits, plugged in here."""
    out = []
    for j in range(n_slots):
        if source == "none":
            lg = torch.randn(len(U) + 1, generator=g) * 0.5
            if gumbel > 0:
                u = torch.rand(len(U) + 1, generator=g).clamp(1e-9, 1 - 1e-9)
                lg = lg + gumbel * (-torch.log(-torch.log(u)))
        elif source == "oracle_noisy":
            lg = torch.randn(len(U) + 1, generator=g) * noise
            lg[U.index(p.units[j])] += strength
        elif source == "planner":
            raise NotImplementedError("TODO planner: blockaddr planner reading the name -> [n_slots, len(U)+1] logits")
        else:
            raise ValueError(source)
        out.append(lg.requires_grad_())
    return out


def derive(p, n_slots, a, init, sharp=False):
    """Adam on slot logits. init(r) -> logits of restart r. Returns the program, unit applications, restart index,
    steps run per restart, and the loss / |loss_t - loss_{t-25}| trace of the last restart.
    The start is checked before the first step (a fitting guess is kept, 0 steps). sharp=True: constant sharpness t1
    (a warm start from a guess; annealing from a soft t0 moved a fitting guess away from the fit in the smoke)."""
    apps, stops = 0, []
    for r in range(a.restarts):
        logits = init(r)
        hist, dl = [], []
        if p.check(decode(logits)):
            stops.append(0)
            return decode(logits), apps, r, stops, hist, dl
        opt = torch.optim.Adam(logits, lr=0.3)
        for s in range(a.steps):
            loss = soft_loss(p, logits, a.t1 if sharp else temp_at(a, s), a.norm, a.crev)
            opt.zero_grad()
            loss.backward()
            opt.step()
            apps += n_slots * len(U)
            hist.append(loss.item())
            if s >= 25:
                dl.append(abs(hist[s] - hist[s - 25]))
            if s % 25 == 24 and p.check(decode(logits)):
                stops.append(s + 1)
                return decode(logits), apps, r, stops, hist, dl
            if a.stop_dloss > 0 and s >= 25 and dl[-1] < a.stop_dloss:
                break
        stops.append(s + 1)
        if p.check(decode(logits)):
            return decode(logits), apps, r, stops, hist, dl
    return decode(logits), apps, a.restarts, stops, hist, dl


def random_search(p, n_slots, budget, seed, cap=50):
    """-> (first fit or last tried, apps at first fit or total, distinct fits found). Runs to the full budget or `cap` fits."""
    r, apps, first, apps_first, fits, units = random.Random(seed), 0, None, 0, set(), []
    while apps < budget and len(fits) < cap:
        units = [r.choice(U) for _ in range(r.randint(1, n_slots))]
        apps += len(units)
        if p.check(units):
            if first is None:
                first, apps_first = units, apps
            fits.add(tuple(units))
    return (first, apps_first, fits) if first is not None else (units, apps, fits)


def guess_sample(p, logits, budget, seed):
    """sample programs slot-wise from softmax(guess logits) until one fits the examples or `budget` unit applications
    are spent (same accounting as random_search). -> (first fit or last tried, apps used)"""
    if p.check(decode(logits)):                          # the start is checked for free, as in derive
        return decode(logits), 0
    g = torch.Generator().manual_seed(seed)
    probs = torch.stack([lg.detach().softmax(-1) for lg in logits])
    apps, units = 0, []
    while apps < max(1, budget):
        units = [U[i] for i in torch.multinomial(probs, 1, generator=g)[:, 0].tolist() if i < len(U)]
        apps += max(1, len(units))
        if p.check(units):
            break
    return units, apps


def enumerate_fits(p, n_slots):
    return [list(us) for n in range(1, n_slots + 1) for us in itertools.product(U, repeat=n) if p.check(list(us))]


def hidden(found, truth, strict):
    """found/truth [(units, final)] -> [bool] output (and, if strict, list state) equals the truth on all 40 hidden inputs."""
    # found body under the true name; `k=0` default so a found program of other arity still runs
    srcs = [[f"def {synth2.name_of(*t)}(xs, k=0):\n" + "".join(f"    {l}\n" for u in fu for l in UNITS[u])
             + f"    {FINAL[ff]}\n"] for (fu, ff), t in zip(found, truth)]
    return [s[0] == 1.0 for s in grade(truth, srcs, strict)]


def dump(res, path):
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(path + ".tmp", path)


def smoke(a):
    """one-hot slots must reproduce the python run: legacy executor and the annealed path at final T (known number: 0)."""
    r = random.Random(3)
    fixed = [(["keep_gt_2", "reverse", "take_3"], f) for f in ("", "first", "sum")] + \
            [(["reverse", "drop_first"], "first"), (["unique", "reverse", "dedup_adjacent"], "len"),
             (["sort", "reverse", "running_max"], "max"), (["abs"], ""), (["add_k", "keep_gt_k", "take_k"], "sum")]
    cases = [(fixed_u, f) for fixed_u, f in fixed]
    for _ in range(200):
        cases.append(([r.choice(U) for _ in range(r.randint(1, 4))], r.choice(["sum", "len", "first", "max", "count_gt_3", ""])))
    modes = {"legacy": (T_LEGACY, False, False), "final_T": (a.t1, True, True)}
    bad = {k: [] for k in modes}
    for units, fin in cases:
        p = Problem(units, fin, 8, r.randint(0, 99))
        if any(o == "!" for o in p.out):
            continue
        hard = [torch.full((len(U) + 1,), -1e4).index_fill_(0, torch.tensor([U.index(u)]), 1e4) for u in units]
        for k, (T, norm, crev) in modes.items():
            loss = soft_loss(p, hard, T, norm, crev).item()
            if loss > (5e-2 if fin == "max" else 1e-3):             # soft max is only ~exact (softmax over values)
                bad[k].append([units, fin, round(loss, 4)])
    print(json.dumps({"smoke_onehot_cases": len(cases), **{f"smoke_mismatch_{k}": len(v) for k, v in bad.items()},
                      "smoke_examples": {k: v[:3] for k, v in bad.items()}}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lens", default="2,3,4")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--n-ex", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--legacy", action="store_true", help="all four derive improvements off")
    ap.add_argument("--no-anneal", dest="anneal", action="store_false")
    ap.add_argument("--no-norm", dest="norm", action="store_false")
    ap.add_argument("--no-gumbel", dest="gumbel_on", action="store_false")
    ap.add_argument("--no-compact-reverse", dest="crev", action="store_false")
    ap.add_argument("--t0", type=float, default=2.0)
    ap.add_argument("--t1", type=float, default=30.0)
    ap.add_argument("--gumbel-scale", type=float, default=0.5)
    ap.add_argument("--stop-dloss", type=float, default=0.0, help="stop a restart when |loss_t - loss_{t-25}| < eps; 0 = off")
    ap.add_argument("--guess-source", default="none", choices=["none", "oracle_noisy", "planner"])
    ap.add_argument("--guess-noise", type=float, default=1.0)
    ap.add_argument("--guess-logit", type=float, default=4.0, help="one-hot strength of the oracle guess")
    ap.add_argument("--exact-max", type=int, default=2, help="enumerate all fitting programs exactly up to this length")
    ap.add_argument("--fit-cap", type=int, default=50)
    a = ap.parse_args()
    if a.legacy:
        a.anneal = a.norm = a.gumbel_on = a.crev = False
    a.gumbel = a.gumbel_scale if a.gumbel_on else 0.0
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    smoke(a)
    if a.smoke:
        return
    path = f"{a.out}/gradsearch.json"
    res = {"config": vars(a), "guess_is_oracle": a.guess_source == "oracle_noisy", "eval": {}}
    arms = ["derive_only", "rand"] + (["guess_only", "guess_derive", "guess_sample", "guess_derive_soft"] if a.guess_source != "none" else [])
    g = random.Random(17)
    for n_slots in map(int, a.lens.split(",")):
        rows = []
        for i in range(a.n):
            while True:
                units = [g.choice(U) for _ in range(n_slots)]
                fin = g.choice(list(FINAL))
                p = Problem(units, fin, a.n_ex, g.randint(0, 10**6))
                if "!" not in p.out:
                    break
            row = {"true": units, "final": fin, "arms": {}}
            gen = torch.Generator().manual_seed(i)             # one stream across restarts (legacy order)
            t0 = time.perf_counter()
            gu, gapps, gr, stops, hist, dl = derive(p, n_slots, a, lambda r: guess_logits(p, n_slots, "none", gen, gumbel=a.gumbel))
            row["arms"]["derive_only"] = {"units": gu, "apps": gapps, "restarts": gr, "steps_per_restart": stops,
                                          "s": round(time.perf_counter() - t0, 2), "fit": p.check(gu),
                                          "loss_trace": [round(x, 5) for x in hist], "dloss25_trace": [round(x, 5) for x in dl]}
            t0 = time.perf_counter()
            ru, rapps, fits = random_search(p, n_slots, gapps, i, a.fit_cap)
            row["arms"]["rand"] = {"units": ru, "apps": rapps, "s": round(time.perf_counter() - t0, 2), "fit": p.check(ru)}
            amb = {"n_fit_rand": len(fits), "n_fit_io_rand": sum(p.io_equiv(list(f)) for f in fits),
                   "rand_cap_hit": len(fits) >= a.fit_cap}
            if n_slots <= a.exact_max:
                ex = enumerate_fits(p, n_slots)
                amb.update({"n_fit_exact": len(ex), "n_fit_io_exact": sum(p.io_equiv(f) for f in ex)})
            row["ambiguity"] = amb
            if a.guess_source != "none":
                kw = dict(noise=a.guess_noise, strength=a.guess_logit)
                gl = guess_logits(p, n_slots, a.guess_source, torch.Generator().manual_seed(i), **kw)
                row["arms"]["guess_only"] = {"units": decode(gl), "apps": 0, "fit": p.check(decode(gl))}
                g1 = torch.Generator().manual_seed(10**6 + i)

                def from_guess(r):                             # ONE guess per problem (a planner gives one); restarts
                    out = []                                   # add Gumbel exploration, never a fresh oracle draw
                    for lg in gl:
                        x = lg.detach().clone()
                        if r > 0 and a.gumbel > 0:
                            x = x + a.gumbel * (-torch.log(-torch.log(torch.rand(x.shape, generator=g1).clamp(1e-9, 1 - 1e-9))))
                        out.append(x.requires_grad_())
                    return out
                t0 = time.perf_counter()
                du, dapps, dr, dstops, dh, ddl = derive(p, n_slots, a, from_guess, sharp=True)
                row["arms"]["guess_derive"] = {"units": du, "apps": dapps, "restarts": dr, "steps_per_restart": dstops,
                                               "s": round(time.perf_counter() - t0, 2), "fit": p.check(du),
                                               "loss_trace": [round(x, 5) for x in dh], "dloss25_trace": [round(x, 5) for x in ddl]}
                t0 = time.perf_counter()
                su, sapps = guess_sample(p, gl, dapps, i)
                row["arms"]["guess_sample"] = {"units": su, "apps": sapps, "s": round(time.perf_counter() - t0, 2),
                                               "fit": p.check(su)}
                g1 = torch.Generator().manual_seed(10**6 + i)        # same Gumbel stream, annealed from t0
                t0 = time.perf_counter()
                fu, fapps, fr, fstops, _, _ = derive(p, n_slots, a, from_guess)
                row["arms"]["guess_derive_soft"] = {"units": fu, "apps": fapps, "restarts": fr, "steps_per_restart": fstops,
                                                    "s": round(time.perf_counter() - t0, 2), "fit": p.check(fu)}
            rows.append(row)
            res["partial"] = {"len": n_slots, "rows": rows}
            dump(res, path)
        truth = [(r_["true"], r_["final"]) for r_ in rows]
        out = {"n": a.n, "n_ex": a.n_ex}
        for arm in arms:
            found = [(r_["arms"][arm]["units"], r_["final"]) for r_ in rows]
            io, st = hidden(found, truth, False), hidden(found, truth, True)
            fit = [r_["arms"][arm]["fit"] for r_ in rows]
            for r_, x, y in zip(rows, io, st):
                r_["arms"][arm].update({"hidden_io": x, "hidden_strict": y})
            nf = sum(fit)
            fni = sum(f and not x for f, x in zip(fit, io))
            out.update({f"{arm}_fit8": nf / a.n, f"{arm}_hidden40_io": sum(io) / a.n, f"{arm}_hidden40_strict": sum(st) / a.n,
                        f"{arm}_fit8_not_io": fni / a.n, f"{arm}_fit8_not_io_given_fit": fni / max(1, nf),
                        f"{arm}_mean_apps": sum(r_["arms"][arm]["apps"] for r_ in rows) / a.n,
                        f"{arm}_s": round(sum(r_["arms"][arm].get("s", 0) for r_ in rows), 1)})
        ex_key = "n_fit_exact" if n_slots <= a.exact_max else "n_fit_rand"
        nfit = [r_["ambiguity"][ex_key] for r_ in rows]
        nio = [r_["ambiguity"][ex_key.replace("fit", "fit_io", 1)] for r_ in rows]
        out.update({"ambiguity_source": "exact" if ex_key == "n_fit_exact" else f"random_search_capped_{a.fit_cap}",
                    "amb_n_fit_mean": round(sum(nfit) / a.n, 2), "amb_n_fit_median": statistics.median(nfit),
                    "amb_unique_frac": sum(x == 1 for x in nfit) / a.n, "amb_n_fit_io_mean": round(sum(nio) / a.n, 2),
                    "amb_rand_cap_hit_frac": sum(r_["ambiguity"]["rand_cap_hit"] for r_ in rows) / a.n,
                    "derive_mean_steps_per_restart": round(sum(sum(r_["arms"]["derive_only"]["steps_per_restart"]) /
                                                              len(r_["arms"]["derive_only"]["steps_per_restart"]) for r_ in rows) / a.n, 1),
                    "rows": rows})
        res["eval"][f"len{n_slots}"] = out
        res.pop("partial", None)
        dump(res, path)
        print(json.dumps({f"len{n_slots}": {k: v for k, v in out.items() if k != "rows"}}), flush=True)


if __name__ == "__main__":
    main()
