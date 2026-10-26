"""Guess + derive v3 (ledger #25 + #26, CPU): the derivative where it is native, on real constants.

v2 (gradsearch.py, 10-26) showed that over DISCRETE op slots, sampling near the guess beats following the gradient
(len3 io .825 vs .675 at ~1/1000 of the work). His idea is "chuta e deriva até |Δ| quase não muda". This file puts
each half where it fits: discrete choices (which op, which argument) are sampled from the guess; real constants are
fitted by the derivative until |dloss| ~ 0.

Program = straight-line steps in the A1 spec language, s_k = op_k(x, y), with x, y chosen from
{a, b, s_1..s_k-1, c_k}; c_k is a real constant owned by step k; the output is the last step.
Problem = 8 (a, b) -> y examples of a hidden program that uses >= 1 constant; grade = 40 hidden inputs, output within
relative error 1e-4 (io). Inputs a, b ~ U[1, 10]. Hidden programs use every step (no dangling) and stay finite.

Guess = noisy oracle over the discrete slots (true one-hot x4 + N(0, sigma)), a stand-in for the planner, as in v2.
Constants are not guessed: every arm starts them at 1.0.

Arms (same budget of program executions on the 8 examples; a forward counts 1, a backward 2):
  guess_only      argmax program, constants 1.0                                      (floor)
  sample_derive   programs sampled from the guess (first = argmax), constants by the derivative (Levenberg-Marquardt:
                  Jacobian steps) until |dloss| ~ 0 or the per-program cap; next program if it does not fit
  sample_es       same programs, constants by a (1+1)-evolution strategy, 1/5 rule (derivative-free local search,
                  the control for "the gradient direction does the work")
  sample_random   same programs, constants drawn log-uniform in +-[0.01, 100] (floor of constant search)
  soft_derive     everything soft (op and argument softmax + constants), Adam from the sharp guess, Gumbel restarts
                  (v2's guess_derive with constants added)
Baselines (gate 1): return a; mean of the 8 example outputs (constant answer).

PREDICTION (registered before the first run, n 40, lens 1,2,3, sigma 1.5):
  P1 sample_derive io >= sample_random io + 30 pp at every length (continuous constants: random draws cannot hit 1e-4).
  P2 sample_derive io >= sample_es io + 10 pp at len2 and len3, with fewer executions (median) -> the gradient
     direction does work on constants. If sample_es ties (within 5 pp), the gain is local search, not the gradient.
  P3 sample_derive io >= soft_derive io + 10 pp at len3 (discrete by sampling, continuous by derivative wins over
     all-soft, as in v2).
  P4 guess_only io <= .10 (constants at 1.0 rarely right).
  KILL (derivative on constants): sample_derive - sample_es <= 5 pp at len2 AND len3.
Limit: the guess is an oracle stand-in; the problems are synthetic formulas; nothing here says a learned planner
gives this guess.

v3.1 (2026-10-26 18:50, after the Sonnet review of v3; registered before any v3.1 run). Review findings verified in the
code: (1) every fitter stopped at the grade tolerance 1e-4, so solutions just under it failed on hidden inputs (ES hit
hardest); (2) LM gave a wrong program up after 5 rejected steps while ES always spent the 900 cap, so ES tried 5-14x
fewer programs: P2 mixed "gradient" with "early stopping"; (3) ES had no recovery from sigma collapse and accepted on
max error while LM minimised L2; (4) sampler and fitter shared one generator, so arms saw different program sequences;
(5) soft_derive: lr 0.3 on the logits threw away a right guess, polish restarted the constants at 1.0, the step was
charged 3|OPS| at every S; (6) the generator made constant-output problems (avg(c, c)) and constants not pinned by the 8
visible examples; LM ran reverse-mode Jacobians (N_EX passes) charged as S columns. Fixes (default; --legacy restores v3
and reproduces its 40 len1 rows exactly: derive 36/38 apps 5680, ES 37/39 apps 22996): fitters iterate to 1e-7 and
accept at 1e-4; separate generators; acceptance check charged 1; ES accepts on L2, restarts once on sigma < 1e-9 and
gives the program up on the second collapse; LM forward-mode Jacobian, stall only on 5 accepted steps gaining < 0.1%;
soft logits lr 0.03, constants lr 0.3, cost 3 S |OPS| per step, polish from the soft constants; the constant must move
the 8 visible outputs; constant-output problems rejected; baseline return b added. Each arm's per-program cap is chosen
on a dev set (`tune`, seeds disjoint from the test), grid 100..1800.
PREDICTIONS v3.1 (n 200, lens 1,2,3, same noise and budget, tuned caps): Q1 derive >= random + 30 pp at every length.
Q2 derive >= ES + 10 pp at len3; at len2 the review's paired rerun (better ES: 38/40 vs 39/40) says the gap is < 5 pp, so
the len2 part of v3's P2 is expected to FAIL. Q3 derive >= soft + 10 pp at len3. Q4 guess_only <= .10 and base_mean <=
.02 (constant-output problems removed). Q5 cost: on problems both solve, derive's median executions <= 1/5 of ES's at
every length. KILL (as v3): derive - ES <= 5 pp at len2 AND len3.

Usage: gradconst.py --out DIR [--n 40 --lens 1,2,3 --noise 1.5 --budget 6000 --threads 1] | gradconst.py smoke --out DIR
"""

import argparse
import json
import math
import os
import random
import statistics
import time

import torch
from torch.func import jacfwd

OPS = ("add", "sub", "mul", "div", "avg", "hyp", "maxx", "pct")
N_EX, N_HID, TOL = 8, 40, 1e-4
TOL_FIT = 1e-7          # v3.1: fitters iterate past the grade tolerance (review 10-26: stopping at TOL lost hidden io)
LEGACY = False          # --legacy: the v3 behaviour, kept for the reproduction control


def op_t(name, x, y):
    if name == "add":
        return x + y
    if name == "sub":
        return x - y
    if name == "mul":
        return x * y
    if name == "div":
        return x / torch.where(y.abs() < 1e-3, torch.full_like(y, 1e-3) * torch.sign(y + 1e-12), y)
    if name == "avg":
        return (x + y) / 2
    if name == "hyp":
        return torch.sqrt(x * x + y * y + 1e-12)
    if name == "maxx":
        return torch.maximum(x, y)
    if name == "pct":
        return x * y / 100
    raise KeyError(name)


def n_src(k):
    """argument sources of step k (0-based): a, b, s_0..s_{k-1}, c_k"""
    return 3 + k


def run(prog, consts, a, b):
    """prog = [(op, i, j)] with i, j source indices; consts [S]; a, b [N] -> output [N] (exact, discrete)."""
    s = []
    for k, (o, i, j) in enumerate(prog):
        src = [a, b] + s + [consts[k].expand_as(a)]
        s.append(op_t(OPS[o], src[i], src[j]))
    return s[-1]


def uses_all(prog):
    """every step feeds a later one (the last is the output) and at least one constant is used"""
    S = len(prog)
    used = {S - 1}
    for k in range(S - 1, -1, -1):
        if k in used:
            for x in prog[k][1:]:
                if 2 <= x < 2 + k:
                    used.add(x - 2)
    const = any(x == 2 + k for k, (_, i, j) in enumerate(prog) for x in (i, j))
    return len(used) == S and const


def text(prog, consts):
    out = []
    for k, (o, i, j) in enumerate(prog):
        names = ["a", "b"] + [f"s{t + 1}" for t in range(k)] + [f"{float(consts[k]):.4g}"]
        out.append(f"s{k + 1} = {OPS[o]}({names[i]}, {names[j]})")
    return out


class Problem:
    def __init__(self, S, seed):
        r = random.Random(seed)
        g = torch.Generator().manual_seed(seed)
        while True:
            prog = [(r.randrange(len(OPS)), r.randrange(n_src(k)), r.randrange(n_src(k))) for k in range(S)]
            if not uses_all(prog):
                continue
            consts = torch.tensor([round(math.exp(r.uniform(math.log(0.2), math.log(50))) * r.choice((1, -1)), 2)
                                   for _ in range(S)], dtype=torch.float64)
            ab = torch.rand(2, N_EX + N_HID, generator=g, dtype=torch.float64) * 9 + 1
            y = run(prog, consts, ab[0], ab[1])
            ok = torch.isfinite(y).all() and y.abs().max() < 1e4 and y.abs().min() > 1e-3
            # the constant must matter: changing it by 10% moves the output (v3.1: on the 8 visible examples, so the
            # examples can pin it; v3 checked all 48 points)
            v = slice(None) if LEGACY else slice(0, N_EX)
            moved = all((run(prog, consts * torch.where(torch.arange(S) == k, 1.1, 1.0), ab[0], ab[1]) - y)[v].abs().max()
                        > 1e-3 * (1 + y[v].abs().max()) for k in range(S) if any(x == 2 + k for x in prog[k][1:]))
            flat = float(y.max() - y.min()) <= 1e-6 * (1 + float(y.abs().max()))     # constant answer (avg(c, c)...)
            if ok and moved and (LEGACY or not flat):
                break
        self.S, self.prog, self.consts = S, prog, consts
        self.a, self.b, self.y = ab[0, :N_EX], ab[1, :N_EX], y[:N_EX]
        self.ha, self.hb, self.hy = ab[0, N_EX:], ab[1, N_EX:], y[N_EX:]

    def err(self, prog, consts, hidden=False):
        a, b, y = (self.ha, self.hb, self.hy) if hidden else (self.a, self.b, self.y)
        with torch.no_grad():
            p = run(prog, consts.detach(), a, b)
        e = ((p - y).abs() / (1 + y.abs())).max()
        return float(e) if torch.isfinite(e) else float("inf")

    def fits(self, prog, consts):
        return self.err(prog, consts) < TOL


def rel_loss(p, y):
    return (((p - y) / (1 + y.abs())) ** 2).mean()


def guess_logits(pb, noise, g, strength=4.0):
    """per step: op logits [len(OPS)], arg logits [n_src] x 2; true one-hot x strength + N(0, noise)"""
    out = []
    for k, (o, i, j) in enumerate(pb.prog):
        lo = torch.randn(len(OPS), generator=g, dtype=torch.float64) * noise
        lo[o] += strength
        li = torch.randn(n_src(k), generator=g, dtype=torch.float64) * noise
        li[i] += strength
        lj = torch.randn(n_src(k), generator=g, dtype=torch.float64) * noise
        lj[j] += strength
        out.append((lo, li, lj))
    return out


def argmax_prog(lg):
    return [(int(lo.argmax()), int(li.argmax()), int(lj.argmax())) for lo, li, lj in lg]


def sample_prog(lg, g):
    return [tuple(int(torch.multinomial(torch.softmax(x, 0), 1, generator=g)) for x in t) for t in lg]


# ---- constant fitters: each returns (consts, apps) ----

def fit_derive(pb, prog, cap, eps=1e-14, c0=None):
    """Levenberg-Marquardt on the constants: follow the derivative (Jacobian of the residuals) until |dloss| ~ 0.
    Adam with a fixed lr cannot travel from 1.0 to a constant of 50 inside the cap (checked 10-26 before the run),
    so the step uses the curvature too. Cost per iteration: S forward-mode Jacobian columns (counted 2 each) + 1
    trial forward (+1 kept from v3). v3.1: forward-mode Jacobian (v3 ran reverse mode, N_EX passes, charged as S), iterate
    to TOL_FIT, stall only on accepted steps that gain < 0.1% of the loss (v3 also counted rejected steps)."""
    c = torch.ones(pb.S, dtype=torch.float64) if c0 is None else c0.detach().clone()
    scale = 1 + pb.y.abs()

    def resid(cc):
        return (run(prog, cc, pb.a, pb.b) - pb.y) / scale

    r = resid(c)
    apps = 1
    if not torch.isfinite(r).all():
        return c, apps
    loss, lam, hist = float((r ** 2).mean()), 1e-2, []
    tol = TOL if LEGACY else TOL_FIT
    while apps + 2 + 2 * pb.S <= cap and float(((r.abs()).max())) >= tol:
        J = torch.autograd.functional.jacobian(resid, c) if LEGACY else jacfwd(resid)(c)     # [N_EX, S]
        A = J.T @ J
        # a constant with zero derivative (unused by the sampled program, or a flat maxx) gives a zero column of J:
        # Marquardt's lam * diag(A) alone is then singular, so the damping has a floor (crash found 10-26 at n=200)
        damp = lam * (A.diagonal() + 1e-9 * (1.0 + float(A.diagonal().max())))
        step = -torch.linalg.lstsq(A + torch.diag(damp), (J.T @ r)[:, None]).solution[:, 0]
        cand = c + step
        rc = resid(cand)
        apps += 2 * pb.S + 2
        lc = float((rc ** 2).mean()) if torch.isfinite(rc).all() else float("inf")
        if lc < loss:
            c, r, lam = cand, rc, max(lam / 3, 1e-12)
            hist.append(loss - lc if LEGACY else (loss - lc) / max(loss, 1e-300))
            loss = lc
        else:
            lam *= 4
            if LEGACY:
                hist.append(0.0)
        if lam > 1e8 or (len(hist) >= 5 and max(hist[-5:]) < (eps if LEGACY else 1e-3)):   # his rule: |dloss| ~ 0
            break
    return c, apps


def fit_es(pb, prog, cap, g):
    if not LEGACY:
        return fit_es31(pb, prog, cap, g)
    c = torch.ones(pb.S, dtype=torch.float64)
    best, apps, sig = pb.err(prog, c), 1, 1.0
    win = []
    while apps < cap and best >= TOL:
        cand = c + sig * torch.randn(pb.S, generator=g, dtype=torch.float64) * (1 + c.abs())
        e = pb.err(prog, cand)
        apps += 1
        ok = e < best
        if ok:
            c, best = cand, e
        win.append(ok)
        if len(win) == 10:                  # 1/5 success rule
            sig *= 1.5 if sum(win) > 2 else 0.6
            win = []
    return c, apps


def fit_es31(pb, prog, cap, g):
    """v3.1 (review 10-26): accept on the same L2 loss LM minimises, iterate to TOL_FIT, and on sigma collapse restart
    once from the best point with sigma 1; a second collapse gives the program up (ES's own stall rule, like LM's)."""
    scale = 1 + pb.y.abs()

    def l2_mx(cc):
        with torch.no_grad():
            rr = (run(prog, cc, pb.a, pb.b) - pb.y) / scale
        if not torch.isfinite(rr).all():
            return float("inf"), float("inf")
        return float((rr ** 2).mean()), float(rr.abs().max())

    c = torch.ones(pb.S, dtype=torch.float64)
    (best, mx), apps, sig, win, restarts = l2_mx(c), 1, 1.0, [], 0
    while apps < cap and mx >= TOL_FIT:
        cand = c + sig * torch.randn(pb.S, generator=g, dtype=torch.float64) * (1 + c.abs())
        lc, mc = l2_mx(cand)
        apps += 1
        ok = lc < best
        if ok:
            c, best, mx = cand, lc, mc
        win.append(ok)
        if len(win) == 10:                  # 1/5 success rule
            sig *= 1.5 if sum(win) > 2 else 0.6
            win = []
        if sig < 1e-9:
            restarts += 1
            if restarts > 1:
                break
            sig = 1.0
    return c, apps


def fit_random(pb, prog, cap, g):
    c = torch.ones(pb.S, dtype=torch.float64)
    apps = 1
    if pb.fits(prog, c):
        return c, apps
    while apps < cap:
        u = torch.rand(pb.S, generator=g, dtype=torch.float64)
        sgn = torch.where(torch.rand(pb.S, generator=g, dtype=torch.float64) < 0.5, -1.0, 1.0)
        c = sgn * torch.exp(math.log(0.01) + u * (math.log(100) - math.log(0.01)))
        apps += 1
        if pb.fits(prog, c):
            break
    return c, apps


def sample_search(pb, lg, budget, per_prog, fitter, seed):
    """programs from the guess (argmax first, then samples, no repeats); constants by `fitter`; until fit or budget.
    v3.1: the sampler and the fitter draw from separate generators (every arm sees the same program sequence), the
    acceptance check is charged 1, and 2000 draws in a row without a new program end the search."""
    g = torch.Generator().manual_seed(seed)
    gf = g if LEGACY else torch.Generator().manual_seed(seed + 7919)
    check = 0 if LEGACY else 1
    seen, apps, tries, stale = set(), 0, 0, 0
    prog = argmax_prog(lg)
    while apps + check < budget and tries < 10 * budget:
        tries += 1
        key = tuple(prog)
        if key not in seen:
            stale = 0
            seen.add(key)
            c, used = fitter(pb, prog, min(per_prog, budget - apps - check), gf)
            apps += used + check
            if pb.fits(prog, c):
                return prog, c, apps
        elif not LEGACY:
            stale += 1
            if stale > 2000:
                break
        prog = sample_prog(lg, g)
    return None, None, apps


def soft_derive(pb, lg, budget, steps, seed, gumbel=1.0, lr=0.3, T=1.0, lr_logit=0.03):
    """all-soft executor: op mix and argument mix per step + constants; Adam from the guess; argmax + refit check."""
    g = torch.Generator().manual_seed(seed)
    apps, r = 0, 0
    while apps < budget:
        par = []
        for lo, li, lj in lg:
            noise = [(-torch.log(-torch.log(torch.rand(x.shape, generator=g, dtype=torch.float64).clamp(1e-9, 1 - 1e-9))))
                     * (gumbel if r > 0 else 0.0) for x in (lo, li, lj)]
            par += [(x + n).clone().requires_grad_(True) for x, n in zip((lo, li, lj), noise)]
        c = torch.ones(pb.S, dtype=torch.float64, requires_grad=True)
        # v3.1 (review 10-26): lr 0.3 on the logits threw away a right guess in most problems; logits get lr_logit, the
        # constants keep lr; a step costs 3 * S * |OPS| (v3 charged 3 * |OPS| at every S) and must fit the budget
        opt = torch.optim.Adam(par + [c], lr=lr) if LEGACY else \
            torch.optim.Adam([{"params": par, "lr": lr_logit}, {"params": [c], "lr": lr}])
        cost = 3 * len(OPS) if LEGACY else 3 * len(OPS) * pb.S
        for _ in range(steps):
            if apps + (3 if LEGACY else cost) > budget:
                break
            s = []
            for k in range(pb.S):
                lo, li, lj = par[3 * k:3 * k + 3]
                src = torch.stack([pb.a, pb.b] + s + [c[k].expand_as(pb.a)])          # [n_src, N]
                x = (torch.softmax(li / T, 0)[:, None] * src).sum(0)
                y = (torch.softmax(lj / T, 0)[:, None] * src).sum(0)
                outs = torch.stack([op_t(o, x, y) for o in OPS])                       # [ops, N]
                s.append((torch.softmax(lo / T, 0)[:, None] * outs).sum(0))
            loss = rel_loss(s[-1], pb.y)
            apps += cost                      # every op runs on every step: a soft step costs |OPS| executions
            if not torch.isfinite(loss):
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        prog = [(int(par[3 * k].argmax()), int(par[3 * k + 1].argmax()), int(par[3 * k + 2].argmax()))
                for k in range(pb.S)]
        cfit, used = fit_derive(pb, prog, min(300, max(0, budget - apps)),           # polish the hard program
                                c0=None if LEGACY else c)                            # v3.1: from the soft constants
        apps += used
        if pb.fits(prog, cfit):
            return prog, cfit, apps
        r += 1
    return None, None, apps


def io_ok(pb, prog, c):
    return prog is not None and pb.err(prog, c, hidden=True) < TOL


def summarize(rows, arms):
    out = {}
    for arm in arms:
        rs = [r[arm] for r in rows]
        out[arm] = {"fit8": round(statistics.mean(x["fit"] for x in rs), 4),
                    "io40": round(statistics.mean(x["io"] for x in rs), 4),
                    "apps_median": statistics.median(x["apps"] for x in rs),
                    "apps_mean": round(statistics.mean(x["apps"] for x in rs), 1)}
    return out


def solve_all(pb, lg, a, seed):
    res = {}
    prog = argmax_prog(lg)
    c1 = torch.ones(pb.S, dtype=torch.float64)
    res["guess_only"] = {"fit": pb.fits(prog, c1), "io": io_ok(pb, prog, c1), "apps": 1}
    fitters = {"sample_derive": lambda p, q, cap, _g: fit_derive(p, q, cap),
               "sample_es": fit_es, "sample_random": fit_random}
    caps = {"sample_derive": a.per_prog_derive or a.per_prog, "sample_es": a.per_prog_es or a.per_prog,
            "sample_random": a.per_prog}
    for name, f in fitters.items():
        t0 = time.perf_counter()
        q, c, apps = sample_search(pb, lg, a.budget, caps[name], f, seed)
        res[name] = {"fit": q is not None, "io": io_ok(pb, q, c), "apps": apps,
                     "prog": text(q, c) if q else None, "wall_s": round(time.perf_counter() - t0, 3)}
    t0 = time.perf_counter()
    q, c, apps = soft_derive(pb, lg, a.budget, a.soft_steps, seed)
    res["soft_derive"] = {"fit": q is not None, "io": io_ok(pb, q, c), "apps": apps,
                          "prog": text(q, c) if q else None, "wall_s": round(time.perf_counter() - t0, 3)}
    # gate-1 baselines
    res["base_return_a"] = {"fit": float(((pb.a - pb.y).abs() / (1 + pb.y.abs())).max()) < TOL,
                            "io": float(((pb.ha - pb.hy).abs() / (1 + pb.hy.abs())).max()) < TOL, "apps": 0}
    res["base_return_b"] = {"fit": float(((pb.b - pb.y).abs() / (1 + pb.y.abs())).max()) < TOL,
                            "io": float(((pb.hb - pb.hy).abs() / (1 + pb.hy.abs())).max()) < TOL, "apps": 0}
    m = pb.y.mean()
    res["base_mean"] = {"fit": float(((m - pb.y).abs() / (1 + pb.y.abs())).max()) < TOL,
                        "io": float(((m - pb.hy).abs() / (1 + pb.hy.abs())).max()) < TOL, "apps": 0}
    return res


ARMS = ("guess_only", "sample_derive", "sample_es", "sample_random", "soft_derive", "base_return_a", "base_return_b",
        "base_mean")


def smoke(a):
    """positive controls: (1) the true program with the true constants fits and is io-right; (2) fit_derive recovers
    the true constants from 1.0 on the TRUE program for most len1 problems; (3) the guess at noise 0 is the truth."""
    rep = {}
    pbs = [Problem(1, 1000 + i) for i in range(20)]
    rep["truth_fits"] = sum(pb.fits(pb.prog, pb.consts) and io_ok(pb, pb.prog, pb.consts) for pb in pbs)
    rep["derive_true_prog_len1"] = sum(pb.fits(pb.prog, fit_derive(pb, pb.prog, 3000)[0]) for pb in pbs)
    g = torch.Generator().manual_seed(0)
    rep["noise0_guess_is_truth"] = sum(argmax_prog(guess_logits(pb, 0.0, g)) == pb.prog for pb in pbs)
    pb = Problem(2, 7)
    rep["example_len2"] = text(pb.prog, pb.consts)
    print(json.dumps(rep), flush=True)
    assert rep["truth_fits"] == 20 and rep["noise0_guess_is_truth"] == 20, rep
    return rep


def tune(a):
    """v3.1 (gate 4/8): pick each sample arm's per-program cap on a DEV set (problem seeds 50_000 * S + i, guess seeds
    777_000 + i: disjoint from the test seeds 10_000 * S + i and i); the cap with the best mean io40 over the lengths
    wins; a best cap at the top edge of the grid is reported (extend before using it)."""
    caps = [int(x) for x in a.caps.split(",")]
    lens = [int(x) for x in a.lens.split(",")]
    pbs = {S: [(Problem(S, 50_000 * S + i), i) for i in range(a.n)] for S in lens}
    out = {}
    for name, f in (("sample_derive", lambda p, q, cap, _g: fit_derive(p, q, cap)), ("sample_es", fit_es)):
        out[name] = {}
        for cap in caps:
            io = {}
            for S in lens:
                hits = 0
                for pb, i in pbs[S]:
                    lg = guess_logits(pb, a.noise, torch.Generator().manual_seed(777_000 + i))
                    q, c, _ = sample_search(pb, lg, a.budget, cap, f, 777_000 + i)
                    hits += io_ok(pb, q, c)
                io[f"len{S}"] = hits / len(pbs[S])
            out[name][cap] = {"io": io, "mean": round(statistics.mean(io.values()), 4)}
            print(name, cap, json.dumps(out[name][cap]), flush=True)
        best = max(caps, key=lambda k: (out[name][k]["mean"], -k))
        out[name]["best"] = best
        out[name]["best_at_top_edge"] = best == max(caps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "smoke", "tune"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lens", default="1,2,3")
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--per-prog", type=int, default=900, help="constant-fit cap per sampled program (executions)")
    ap.add_argument("--per-prog-derive", type=int, default=0, help="v3.1: own cap for sample_derive (0 = --per-prog)")
    ap.add_argument("--per-prog-es", type=int, default=0, help="v3.1: own cap for sample_es (0 = --per-prog)")
    ap.add_argument("--caps", default="100,200,300,600,900,1800", help="tune: per-program caps tried per arm")
    ap.add_argument("--legacy", action="store_true", help="v3 behaviour (reproduction control)")
    ap.add_argument("--soft-steps", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    global LEGACY
    LEGACY = a.legacy
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "smoke":
        json.dump(smoke(a), open(os.path.join(a.out, "smoke.json"), "w"), indent=1)
        return
    if a.cmd == "tune":
        json.dump(tune(a), open(os.path.join(a.out, "tune.json"), "w"), indent=1)
        return
    res = {"args": vars(a), "lens": {}}
    for S in map(int, a.lens.split(",")):
        rows = []
        mc = 0
        for i in range(a.n):
            pb = Problem(S, 10_000 * S + i)
            lg = guess_logits(pb, a.noise, torch.Generator().manual_seed(i))
            mc += argmax_prog(lg) == pb.prog
            rows.append({"id": i, "truth": text(pb.prog, pb.consts), **solve_all(pb, lg, a, i)})
            res["lens"][f"len{S}"] = {"n_done": len(rows), "guess_argmax_is_truth": round(mc / len(rows), 4),
                                      "summary": summarize(rows, ARMS), "rows": rows}
            json.dump(res, open(os.path.join(a.out, "gradconst.json"), "w"), indent=1)
        print(f"len{S}", json.dumps(res["lens"][f"len{S}"]["summary"]), "guess_right", mc / a.n, flush=True)


if __name__ == "__main__":
    main()
