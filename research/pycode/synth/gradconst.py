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

OPS = ("add", "sub", "mul", "div", "avg", "hyp", "maxx", "pct")
N_EX, N_HID, TOL = 8, 40, 1e-4


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
            # the constant must matter: changing it by 10% moves the output
            moved = all((run(prog, consts * torch.where(torch.arange(S) == k, 1.1, 1.0), ab[0], ab[1]) - y).abs().max()
                        > 1e-3 * (1 + y.abs().max()) for k in range(S) if any(x == 2 + k for x in prog[k][1:]))
            if ok and moved:
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

def fit_derive(pb, prog, cap, eps=1e-14):
    """Levenberg-Marquardt on the constants: follow the derivative (Jacobian of the residuals) until |dloss| ~ 0.
    Adam with a fixed lr cannot travel from 1.0 to a constant of 50 inside the cap (checked 10-26 before the run),
    so the step uses the curvature too. Cost per iteration: 1 forward + S backward-equivalents (Jacobian rows,
    counted 2 each) + 1 trial forward."""
    c = torch.ones(pb.S, dtype=torch.float64)
    scale = 1 + pb.y.abs()

    def resid(cc):
        return (run(prog, cc, pb.a, pb.b) - pb.y) / scale

    r = resid(c)
    apps = 1
    if not torch.isfinite(r).all():
        return c, apps
    loss, lam, hist = float((r ** 2).mean()), 1e-2, []
    while apps + 2 + 2 * pb.S <= cap and float(((r.abs()).max())) >= TOL:
        J = torch.autograd.functional.jacobian(resid, c)                     # [N_EX, S]
        A = J.T @ J
        step = -torch.linalg.solve(A + lam * torch.diag(A.diagonal() + 1e-12), J.T @ r)
        cand = c + step
        rc = resid(cand)
        apps += 2 * pb.S + 2
        lc = float((rc ** 2).mean()) if torch.isfinite(rc).all() else float("inf")
        if lc < loss:
            c, r, lam = cand, rc, lam / 3
            hist.append(loss - lc)
            loss = lc
        else:
            lam *= 4
            hist.append(0.0)
        if lam > 1e8 or (len(hist) >= 5 and max(hist[-5:]) < eps):        # his stop rule: |dloss| ~ 0
            break
    return c, apps


def fit_es(pb, prog, cap, g):
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
    """programs from the guess (argmax first, then samples, no repeats); constants by `fitter`; until fit or budget."""
    g = torch.Generator().manual_seed(seed)
    seen, apps, tries = set(), 0, 0
    prog = argmax_prog(lg)
    while apps < budget and tries < 10 * budget:
        tries += 1
        key = tuple(prog)
        if key not in seen:
            seen.add(key)
            c, used = fitter(pb, prog, min(per_prog, budget - apps), g)
            apps += used
            if pb.fits(prog, c):
                return prog, c, apps
        prog = sample_prog(lg, g)
    return None, None, apps


def soft_derive(pb, lg, budget, steps, seed, gumbel=1.0, lr=0.3, T=1.0):
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
        opt = torch.optim.Adam(par + [c], lr=lr)
        for _ in range(steps):
            if apps + 3 > budget:
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
            apps += 3 * len(OPS)              # every op runs on every step: a soft step costs |OPS| executions
            if not torch.isfinite(loss):
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        prog = [(int(par[3 * k].argmax()), int(par[3 * k + 1].argmax()), int(par[3 * k + 2].argmax()))
                for k in range(pb.S)]
        cfit, used = fit_derive(pb, prog, min(300, max(0, budget - apps)))           # polish the hard program
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
    for name, f in fitters.items():
        t0 = time.perf_counter()
        q, c, apps = sample_search(pb, lg, a.budget, a.per_prog, f, seed)
        res[name] = {"fit": q is not None, "io": io_ok(pb, q, c), "apps": apps,
                     "prog": text(q, c) if q else None, "wall_s": round(time.perf_counter() - t0, 3)}
    t0 = time.perf_counter()
    q, c, apps = soft_derive(pb, lg, a.budget, a.soft_steps, seed)
    res["soft_derive"] = {"fit": q is not None, "io": io_ok(pb, q, c), "apps": apps,
                          "prog": text(q, c) if q else None, "wall_s": round(time.perf_counter() - t0, 3)}
    # gate-1 baselines
    res["base_return_a"] = {"fit": float(((pb.a - pb.y).abs() / (1 + pb.y.abs())).max()) < TOL,
                            "io": float(((pb.ha - pb.hy).abs() / (1 + pb.hy.abs())).max()) < TOL, "apps": 0}
    m = pb.y.mean()
    res["base_mean"] = {"fit": float(((m - pb.y).abs() / (1 + pb.y.abs())).max()) < TOL,
                        "io": float(((m - pb.hy).abs() / (1 + pb.hy.abs())).max()) < TOL, "apps": 0}
    return res


ARMS = ("guess_only", "sample_derive", "sample_es", "sample_random", "soft_derive", "base_return_a", "base_mean")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "smoke"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lens", default="1,2,3")
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--per-prog", type=int, default=900, help="constant-fit cap per sampled program (executions)")
    ap.add_argument("--soft-steps", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "smoke":
        json.dump(smoke(a), open(os.path.join(a.out, "smoke.json"), "w"), indent=1)
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
