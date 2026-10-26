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

v2.1 (10-26 evening, FAILED-ADAPT of his "chuta e deriva até |Δ| quase não mudar", Opus review of v2 A-D + gradconst E-G):
  A  the argmax is checked every --check-every steps (default 1; v2 checked every 25, a fit could pass unseen) and only
     when it changed; each check costs its unit applications like a random-search try (v2: free).
  B  problems whose 8 example outputs are all equal are dropped (--keep-const keeps them). v2 had 11/13/17 of 40 such
     problems at len 2/3/4: any constant answer fit them, so every v2 rate was inflated.
  C  the example check calls a program with the TRUE program's arity, like the hidden grade (v2: own arity, so a k-unit
     program could fit the examples and then run with k=0 on the hidden inputs).
  D  <arm>_unseen32_io: io on the hidden inputs NOT among the 8 examples (hidden40 includes them);
     <arm>_hidden40_not_unseen must be 0 (sanity: hidden40 io implies unseen io).
  gate 1 baselines: arm `identity` (no unit, the final on the input) and `const` (most frequent example output
     everywhere; fit8 = 0 by construction once B drops constant problems).
  his stop rule ON (--stop-eps > 0): arms derive_only_stop / guess_derive_stop end a restart when
     |loss_t - loss_{t-25}| < eps and restart (up to 64 restarts) until they have spent the no-stop arm's MAXIMUM
     budget (restarts x steps x n_slots x len(U)); restart r starts from the same init as the no-stop arm's restart r.
     <arm>_start_wins: fits found at a restart's start, before any Adam step (restart 0 of guess_derive = the guess).
  --v2 reproduces the v2 runs (check every 25 free, own arity, constants kept, problem seed 17): 5/5 len2 rows of
     gradsearch_v2/main_l23 reproduced exactly (units, apps, fit, hidden_io, restarts, ambiguity), 10-26 20:2x.
  eps chosen on the v2 DEV traces (seed 17), never on the test set (problem seed 18): at 3e-5 the rule would have cut
     4/26 derive_only and 1/9 guess_derive fitting restarts before their fit, and saved 23% / 20% of the steps of
     failing restarts (1e-4: 6/26, 2/9 cut; 1e-5: 2/26, 0/9 cut, 13% / 9% saved).

PREDICTION v2.1 (registered before the run; test set problem seed 18, n 80 per length, lens 2,3,4, oracle_noisy 1.5,
--stop-eps 3e-5, all other defaults). Dev reference = v2 recomputed on its NON-constant problems only (n 29/27/23),
hidden40_io: derive_only .38/.19/.13, rand .76/.37/.44, guess_only .38/.22/.35, guess_derive .86/.52/.61,
guess_sample .97/.74/.87, guess_derive_soft .83/.48/.70.
  Q0 sanity: <arm>_hidden40_not_unseen = 0 for every arm; guess_derive_start_wins >= n x guess_only_fit8.
  Q1 gate 1: identity and const unseen32_io <= .05 at every length; every other arm above them.
  Q2 guess_derive unseen32_io in .80-.95 / .50-.75 / .55-.80; >= guess_only + 25 pp at every length.
  Q3 (the #26 core) guess_sample >= guess_derive at every length (pooled paired sign test over 240 problems: sampling
     near the guess wins more discordant pairs, p < .05). A guess_derive win there would reverse v2.
  Q4 derive_only < rand at every length (no guess, the derivative alone loses to random programs at equal budget).
  Q5 his stop rule (paired by problem, pooled 240, two-sided sign test on unseen32_io discordant pairs):
     guess_derive_stop vs guess_derive NOT ESTABLISHED (|b - c| small; the rule mostly trades a few cut fits for
     ~1.25x more restarts); derive_only_stop <= derive_only (a stopped restart decodes before the sharpness anneal ends).
  Verdict for his stop rule at eps 3e-5: PASS if a stop arm wins the sign test (p < .05) against its no-stop twin;
  HARM if it loses (p < .05); else NOT ESTABLISHED. Compute: median apps on problems both arms solve, reported.
  Wall: v2 took derive_only 121/225/303 s and guess_derive 31/116/87 s per 40; the stop arms spend up to the full
  budget on every unsolved problem -> ~25-40 min per length at 2 threads, 3 lengths in parallel.

v2.2 (10-26 ~20:40, Opus pre-run review of v2.1; NO v2.1 test run happened, so v2.1's Q0-Q5 are replaced by v2.2's below):
  1  every check is charged max(1, len): a restart's start (v2.1: free; the *_stop arms run up to 64 restarts, so up to
     56 free near-guess samples: on dev, free starts at restart 8..63 fit 4/4, 7/13, 6/9 of guess_derive's failures)
     and guess_sample's / guess_only's start. Per derive arm: start_win (fit at a restart's start), start_win_late
     (start win at restart >= --restarts: a restart the no-stop twin never had), adam_win (fit after >= 1 Adam step).
  3  rand and guess_sample get the MAXIMUM budget B = restarts x steps x n_slots x len(U) (v2.1: the reference arm's
     realized apps, 0 when it fit at restart 0: 4/2/0 of 80 dev rows) and return their first fit, so success at any
     budget b is `fit and unseen io and apps <= b`: <arm>_solved_at_B/64 .. B/1 (success-vs-budget curve). The no-stop
     derive arms can exceed B by their check charges (<= 1/47 of B); the stop arms are capped at B.
  derive returns the number of restarts run (v2.1 returned --restarts on a budget-out).
  --v2 control after these changes: 5/5 len2 rows of gradsearch_v2/main_l23 identical in every arm and field (0 diffs,
  gradsearch_v22/ctrl_v2, 10-26 20:3x). Smoke (dev seed 17, n 3, lens 2,3, --stop-eps 3e-5): runs, every field present.
  The verdict is computed by gradsearch_verdict.py, written before the test results.

PREDICTION v2.2 (registered before the run; test = problem seed 18, n 80 per length, lens 2,3,4, oracle_noisy 1.5,
--stop-eps 3e-5; SOLVED = fits the 8 examples and io-equal on the 32 unseen hidden inputs; pooled = 240 problems).
Dev reference (v2 on its non-constant problems, realized-budget challengers): unseen io derive_only .38/.19/.13,
rand .76/.37/.44, guess_only .38/.22/.35, guess_derive .86/.52/.61, guess_sample .97/.74/.87; identity .09/.06/.04.
  Q1 gate 1: every non-baseline arm's solved rate > max(identity, const) at every length (const is 0 by construction).
  Q2 guess_derive unseen32_io in .80-.95 / .50-.75 / .55-.80 and >= guess_only + 25 pp at every length.
  Q3 (#26 core) guess_sample vs guess_derive at the same maximum budget B: sampling solves more (pooled sign test,
     p < .05) and its curve is >= derive's at every b in B/64..B. guess_derive winning with p < .05 reverses v2.
  Q4 rand > derive_only at every length (solved at B) and at every b of the curve.
  Q5 his stop rule, 4 pooled sign tests, Bonferroni .05/4: {guess_derive_stop vs guess_derive, derive_only_stop vs
     derive_only} x {all wins, Adam-only wins}. Expected: all wins, the stop arms win (more restarts inside B, i.e.
     more starts = more samples), mostly through start_win_late; Adam-only NOT ESTABLISHED for guess_derive_stop and
     derive_only_stop <= derive_only (a stopped restart decodes before the anneal ends).
  Verdict for his rule 'stop when |dloss| barely changes' at eps 3e-5: PASS only if guess_derive_stop wins BOTH its
  all-wins and its Adam-only test; a win carried by start_win_late alone credits restarting (sampling), not the rule;
  HARM if it loses either with p < .0125; else NOT ESTABLISHED. Median apps on both-solved reported per pair.
  Wall (smoke): an unsolved derive problem ~12 s (len2) at 2 threads; ~1-1.5 h per length, 3 lengths in parallel.
  Amendment 10-26 ~20:50 (second Opus review, before any test result; run code unchanged): the verdict script's
  A1 (Q5 rule verdict coded, HARM via a strict Adam-only test that does not count a problem the stop arm solved at
  a later restart's start), A2 (B uses 47 units, the curves must equal the fields above), A3 (every arm capped at B;
  the curve conditions are part of Q3 / Q4; 0-0 curve ties skipped). See gradsearch_verdict.py.

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

U = list(UNITS)                       # 47 units; slot choice index len(U) = identity
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
    def __init__(self, units, final, n_ex, seed, truth_arity=True):
        r = random.Random(seed)
        self.units, self.final = units, final
        self.two, self.truth_arity = nargs(units) == 2, truth_arity
        ix = r.sample(range(len(INPUTS)), n_ex)              # the same draw as r.sample(INPUTS, n_ex)
        self.ex = [INPUTS[i] for i in ix]
        self.unseen = [INPUTS[i] for i in range(len(INPUTS)) if i not in ix]   # review D: hidden minus the examples
        self.out = [py_run(units, final, xs, k) for xs, k in self.ex]
        self.v = torch.tensor([xs + [0] * (L - len(xs)) for xs, _ in self.ex], dtype=torch.float32)
        self.m = torch.tensor([[1.0] * len(xs) + [0.0] * (L - len(xs)) for xs, _ in self.ex])
        self.k = torch.tensor([float(k) for _, k in self.ex])[:, None]
        self.tgt = None if final == "" else torch.tensor([float(o) if o != "!" else 0.0 for o in self.out])
        vals = [float(x) for o in self.out if o != "!" for x in (o if final == "" else [o])]
        self.scale = 1.0 + (sum(x * x for x in vals) / len(vals) if vals else 0.0)   # 1 + mean(target^2)
        self.hid = None                                                    # truth outputs on the 40 hidden inputs
        self.hid_u = None                                                  # ... on the inputs not among the examples

    def run(self, us, xs, k):
        """called like the hidden grade: the truth's arity, k=0 default."""
        try:
            f = _fn(tuple(us), self.final)
            return f(list(xs), k) if self.two else f(list(xs))
        except Exception:
            return "!"

    def check(self, units):
        """fits the examples. v2.1 calls the program with the truth's arity, like the hidden grade (review C);
        v2 (truth_arity=False) called it with its own arity, so a k-unit could fit the examples and then run with k=0."""
        if not self.truth_arity:
            return all(py_run(units, self.final, xs, k) == o for (xs, k), o in zip(self.ex, self.out))
        return all(self.run(units, xs, k) == o for (xs, k), o in zip(self.ex, self.out))

    def io_equiv(self, units, unseen=False):
        """output equals the true program on the 40 hidden inputs (unseen=True: the inputs not among the examples),
        called like the hidden grade (truth's arity, k=0 default)."""
        if unseen:
            if self.hid_u is None:
                self.hid_u = [self.run(self.units, xs, k) for xs, k in self.unseen]
            return all(self.run(units, xs, k) == h for (xs, k), h in zip(self.unseen, self.hid_u))
        if self.hid is None:
            self.hid = [self.run(self.units, xs, k) for xs, k in INPUTS]
        return all(self.run(units, xs, k) == h for (xs, k), h in zip(INPUTS, self.hid))

    def const_answer(self):
        """gate-1 baseline: the most frequent example output answered for every input -> (fit8, unseen io)."""
        reps = [repr(o) for o in self.out]
        best = max(set(reps), key=reps.count)
        val = self.out[reps.index(best)]
        if self.hid_u is None:
            self.hid_u = [self.run(self.units, xs, k) for xs, k in self.unseen]
        return all(o == val for o in self.out), all(h == val for h in self.hid_u)


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


def derive(p, n_slots, a, init, sharp=False, stop=None, restarts=None, budget=None):
    """Adam on slot logits. init(r) -> logits of restart r. Returns the program, unit applications, restart index,
    steps run per restart, and the loss / |loss_t - loss_{t-25}| trace of the last restart.
    The start is checked before the first step (a fitting guess is kept, 0 steps). sharp=True: constant sharpness t1
    (a warm start from a guess; annealing from a soft t0 moved a fitting guess away from the fit in the smoke).
    The argmax program is checked every a.check_every steps (v2: 25, review A: a fit could pass unseen for 24 steps);
    a check runs only when the argmax changed and, unless a.v2, costs its unit applications like a random-search try.
    stop: end a restart when |loss_t - loss_{t-25}| < stop (his '|Δ| quase não muda'; default a.stop_dloss);
    restarts / budget: the stop arm restarts until it has spent the no-stop arm's maximum budget."""
    stop = a.stop_dloss if stop is None else stop
    R = a.restarts if restarts is None else restarts
    apps, stops = 0, []
    for r in range(R):
        if budget is not None and apps >= budget:
            break
        logits = init(r)
        hist, dl = [], []
        last = decode(logits)
        apps += 0 if a.v2 else max(1, len(last))           # v2.2: the start check costs like any other check
        if p.check(last):
            stops.append(0)
            return last, apps, r, stops, hist, dl
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
            if s % a.check_every == a.check_every - 1:
                cur = decode(logits)
                if cur != last:
                    last = cur
                    apps += 0 if a.v2 else max(1, len(cur))
                    if p.check(cur):
                        stops.append(s + 1)
                        return cur, apps, r, stops, hist, dl
            if stop > 0 and s >= 25 and dl[-1] < stop:
                break
            if budget is not None and apps >= budget:
                break
        stops.append(s + 1)
        cur = decode(logits)
        if cur != last and not a.v2:                       # v2.2: a new argmax costs its check (equal to `last` when
            apps += max(1, len(cur))                       # every step is checked; then it already failed)
        if p.check(cur):
            return cur, apps, r, stops, hist, dl
    return decode(logits), apps, len(stops), stops, hist, dl


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


def guess_sample(p, logits, budget, seed, charge_start=False):
    """sample programs slot-wise from softmax(guess logits) until one fits the examples or `budget` unit applications
    are spent (same accounting as random_search). -> (first fit or last tried, apps used)
    The start (argmax of the guess) is checked first: free in v2, charged like any check from v2.2 (charge_start)."""
    apps = max(1, len(decode(logits))) if charge_start else 0
    if p.check(decode(logits)):
        return decode(logits), apps
    g = torch.Generator().manual_seed(seed)
    probs = torch.stack([lg.detach().softmax(-1) for lg in logits])
    units = []
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
    for sd in (0, 7, 123456):                                   # Problem's index draw == the v2 draw of INPUTS
        assert [INPUTS[i] for i in random.Random(sd).sample(range(len(INPUTS)), 8)] == random.Random(sd).sample(INPUTS, 8)
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
    ap.add_argument("--check-every", type=int, default=1, help="check the argmax program every N Adam steps (v2: 25)")
    ap.add_argument("--keep-const", action="store_true", help="keep problems whose example outputs are all equal (v2)")
    ap.add_argument("--problem-seed", type=int, default=18, help="problem generator seed (v2 runs: 17; v2.1 test: 18)")
    ap.add_argument("--stop-eps", type=float, default=0.0, help="> 0 adds the *_stop arms: his stop rule ON with this eps, "
                    "restarts until the no-stop arm's maximum budget is spent")
    ap.add_argument("--v2", action="store_true", help="reproduce the v2 runs: check every 25, keep constant problems, "
                    "program's own arity in the example check, checks free, problem seed 17")
    a = ap.parse_args()
    if a.v2:
        a.check_every, a.keep_const, a.problem_seed = 25, True, 17
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
    if not a.v2:
        arms += ["identity"] + (["derive_only_stop"] + (["guess_derive_stop"] if a.guess_source != "none" else [])
                                if a.stop_eps > 0 else [])
    g = random.Random(a.problem_seed)
    for n_slots in map(int, a.lens.split(",")):
        rows = []
        for i in range(a.n):
            while True:
                units = [g.choice(U) for _ in range(n_slots)]
                fin = g.choice(list(FINAL))
                p = Problem(units, fin, a.n_ex, g.randint(0, 10**6), truth_arity=not a.v2)
                if "!" not in p.out and (a.keep_const or len({repr(o) for o in p.out}) > 1):   # review B
                    break
            row = {"true": units, "final": fin, "arms": {}}
            B = a.restarts * a.steps * n_slots * len(U)       # v2.2: every challenger gets the max budget of the no-stop
                                                                # derive arms; success at budget b <=> fit with apps <= b
            gen = torch.Generator().manual_seed(i)             # one stream across restarts (legacy order)
            t0 = time.perf_counter()
            gu, gapps, gr, stops, hist, dl = derive(p, n_slots, a, lambda r: guess_logits(p, n_slots, "none", gen, gumbel=a.gumbel))
            row["arms"]["derive_only"] = {"units": gu, "apps": gapps, "restarts": gr, "steps_per_restart": stops,
                                          "s": round(time.perf_counter() - t0, 2), "fit": p.check(gu),
                                          "loss_trace": [round(x, 5) for x in hist], "dloss25_trace": [round(x, 5) for x in dl]}
            t0 = time.perf_counter()
            ru, rapps, fits = random_search(p, n_slots, gapps if a.v2 else B, i, a.fit_cap)
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
                row["arms"]["guess_only"] = {"units": decode(gl), "apps": 0 if a.v2 else max(1, len(decode(gl))),
                                             "fit": p.check(decode(gl))}
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
                su, sapps = guess_sample(p, gl, dapps if a.v2 else B, i, charge_start=not a.v2)
                row["arms"]["guess_sample"] = {"units": su, "apps": sapps, "s": round(time.perf_counter() - t0, 2),
                                               "fit": p.check(su)}
                g1 = torch.Generator().manual_seed(10**6 + i)        # same Gumbel stream, annealed from t0
                t0 = time.perf_counter()
                fu, fapps, fr, fstops, _, _ = derive(p, n_slots, a, from_guess)
                row["arms"]["guess_derive_soft"] = {"units": fu, "apps": fapps, "restarts": fr, "steps_per_restart": fstops,
                                                    "s": round(time.perf_counter() - t0, 2), "fit": p.check(fu)}
                if "guess_derive_stop" in arms:                      # his stop rule, same guess and Gumbel stream
                    g1 = torch.Generator().manual_seed(10**6 + i)
                    t0 = time.perf_counter()
                    xu, xapps, xr, xstops, _, _ = derive(p, n_slots, a, from_guess, sharp=True, stop=a.stop_eps,
                                                         restarts=64, budget=B)
                    row["arms"]["guess_derive_stop"] = {"units": xu, "apps": xapps, "restarts": xr, "steps_per_restart": xstops,
                                                        "s": round(time.perf_counter() - t0, 2), "fit": p.check(xu)}
            if "derive_only_stop" in arms:                           # same init stream as derive_only (restart 0 paired)
                gen2 = torch.Generator().manual_seed(i)
                t0 = time.perf_counter()
                xu, xapps, xr, xstops, _, _ = derive(p, n_slots, a, lambda r: guess_logits(p, n_slots, "none", gen2, gumbel=a.gumbel),
                                                     stop=a.stop_eps, restarts=64, budget=B)
                row["arms"]["derive_only_stop"] = {"units": xu, "apps": xapps, "restarts": xr, "steps_per_restart": xstops,
                                                   "s": round(time.perf_counter() - t0, 2), "fit": p.check(xu)}
            if "identity" in arms:                                   # gate-1 baseline: no unit, the final on the input
                row["arms"]["identity"] = {"units": [], "apps": 0, "fit": p.check([])}
            if not a.v2:
                cf, cu = p.const_answer()
                row["baseline_const"] = {"fit": cf, "unseen_io": cu}
                for arm_, x in row["arms"].items():                  # review D: io on the inputs not among the examples
                    x["unseen_io"] = p.io_equiv(x["units"], unseen=True)
                    if "steps_per_restart" in x:
                        x["start_win"] = x["fit"] and x["steps_per_restart"][-1] == 0
                        x["start_win_late"] = x["start_win"] and x["restarts"] >= a.restarts      # v2.2 (review 1)
                        x["adam_win"] = x["fit"] and not x["start_win"]
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
                        f"{arm}_median_apps": statistics.median(r_["arms"][arm]["apps"] for r_ in rows),
                        f"{arm}_s": round(sum(r_["arms"][arm].get("s", 0) for r_ in rows), 1)})
            if not a.v2:
                uio = [r_["arms"][arm]["unseen_io"] for r_ in rows]
                out.update({f"{arm}_unseen32_io": sum(uio) / a.n,
                            f"{arm}_hidden40_not_unseen": sum(x and not y for x, y in zip(io, uio)),   # sanity: must be 0
                            f"{arm}_start_wins": sum(r_["arms"][arm].get("start_win", False) for r_ in rows),
                            f"{arm}_start_wins_late": sum(r_["arms"][arm].get("start_win_late", False) for r_ in rows)})
                Bmax = a.restarts * a.steps * n_slots * len(U)
                for d in (64, 16, 4, 1):                             # v2.2: success-vs-budget (anytime: first fit)
                    out[f"{arm}_solved_at_B/{d}"] = sum(r_["arms"][arm]["fit"] and r_["arms"][arm]["unseen_io"]
                                                         and r_["arms"][arm]["apps"] <= Bmax / d for r_ in rows) / a.n
        if not a.v2:
            out.update({"const_fit8": sum(r_["baseline_const"]["fit"] for r_ in rows) / a.n,
                        "const_unseen32_io": sum(r_["baseline_const"]["unseen_io"] for r_ in rows) / a.n})
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
