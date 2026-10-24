"""Program search by derivative (ledger #24, registered 2026-10-23, CPU).

Input is the PROBLEM, not the name: 8 input/output examples of a hidden synth2 task (units + final).
Each program slot is a softmax over all units (+ identity). A differentiable executor runs every unit on a soft
list state (values v [N, L], keep mask m [N, L]); the next state is the probability-weighted mix of all unit
outputs. Loss vs the target outputs; Adam moves the slot logits. Argmax -> discrete program -> exact check on the
8 examples (restart if wrong) -> strict bwrap grade on the 40 hidden synth2 INPUTS.
Control: random search over discrete programs with the same budget of unit applications.
The final (sum/len/max/...) is given, as the task head is given to the planner.
Prediction: len2 >= 80% both arms (space 47^2 is small); len4 gradient >> random (47^4 = 4.9M programs).

Usage: gradsearch.py --out DIR [--n 40 --lens 2,3,4 --steps 300 --restarts 8]
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import synth2  # noqa: E402
from synth2 import FINAL, INPUTS, UNITS, grade, nargs, source  # noqa: E402

U = list(UNITS)                       # 46 units; slot choice index len(U) = identity
L, BIG, T = 12, 1e4, 20.0             # padded list length, masked sentinel, sharpness of soft comparisons


def py_run(units, final, xs, k):
    """trusted in-process run of a composition of OUR unit code (search scoring only; the grade uses bwrap)."""
    g = {}
    exec(source(units, final), g)
    f = g[synth2.name_of(units, final)]
    try:
        return f(list(xs), k) if nargs(units) == 2 else f(list(xs))
    except Exception:
        return "!"


def soft_unit(u, v, m, k):
    sg = torch.sigmoid
    if u in ("sort",):
        key = torch.where(m > 0.5, v, torch.full_like(v, BIG))
        idx = torch.sort(key, dim=1, stable=True).indices
        return v.gather(1, idx), m.gather(1, idx)
    if u == "reverse":
        # reverse kept order: kept elements move to the front reversed; holes do not matter for later ops
        return v.flip(1), m.flip(1)
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


def soft_final(final, v, m):
    if final == "sum":
        return (v * m).sum(1)
    if final == "len":
        return m.sum(1)
    if final == "max":
        w = torch.softmax(torch.where(m > 0.5, v, torch.full_like(v, -BIG)) * 4, 1)
        return (w * v).sum(1) * (m.sum(1) > 0.5)
    if final == "first":
        w = m * torch.cumprod(torch.cat([torch.ones_like(m[:, :1]), 1 - m[:, :-1]], 1), 1)
        return (w * v).sum(1)
    if final.startswith("count_gt_"):
        return (m * torch.sigmoid(T * (v - int(final[-1]) - 0.5))).sum(1)
    raise KeyError(final)


def list_loss(v, m, targets):
    """kept element of rank r must equal target[r]; length must match."""
    r = torch.cumsum(m, 1) - 1
    loss = 0
    for i, t in enumerate(targets):
        tt = torch.tensor(t + [0.0] * (L - len(t)), dtype=torch.float32)
        ri = r[i].clamp(0, L - 1)
        lo = ri.floor().long()
        frac = ri - lo
        tv = tt[lo] * (1 - frac) + tt[(lo + 1).clamp(max=L - 1)] * frac
        loss = loss + (m[i] * (v[i] - tv) ** 2).sum() / (1 + len(t)) + (m[i].sum() - len(t)) ** 2
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

    def check(self, units):
        return all(py_run(units, self.final, xs, k) == o for (xs, k), o in zip(self.ex, self.out))


def soft_loss(p, logits):
    v, m = p.v, p.m
    for lg in logits:
        w = torch.softmax(lg, 0)
        vs, ms = [v], [m]                                   # index len(U) = identity
        for u in U:
            a, b = soft_unit(u, v, m, p.k)
            vs.append(a)
            ms.append(b)
        order = [len(U)] + list(range(len(U)))              # stacked [identity, U...] -> weights reordered
        w = w[order]
        v = (w[:, None, None] * torch.stack(vs)).sum(0)
        m = (w[:, None, None] * torch.stack(ms)).sum(0)
        v = v.clamp(-BIG, BIG)
    if p.final == "":
        return list_loss(v, m, p.out)
    tgt = torch.tensor([float(o) if o != "!" else 0.0 for o in p.out])
    return ((soft_final(p.final, v, m) - tgt) ** 2).mean()


def decode(logits):
    return [U[i] for i in (int(lg.argmax()) for lg in logits) if i < len(U)]


def grad_search(p, n_slots, steps, restarts, seed):
    g = torch.Generator().manual_seed(seed)
    apps = 0
    for r in range(restarts):
        logits = [(torch.randn(len(U) + 1, generator=g) * 0.5).requires_grad_() for _ in range(n_slots)]
        opt = torch.optim.Adam(logits, lr=0.3)
        for s in range(steps):
            loss = soft_loss(p, logits)
            opt.zero_grad()
            loss.backward()
            opt.step()
            apps += n_slots * len(U)
            if s % 25 == 24 and p.check(decode(logits)):
                return decode(logits), apps, r
        if p.check(decode(logits)):
            return decode(logits), apps, r
    return decode(logits), apps, restarts


def random_search(p, n_slots, budget, seed):
    r, apps = random.Random(seed), 0
    while apps < budget:
        units = [r.choice(U) for _ in range(r.randint(1, n_slots))]
        apps += len(units)
        if p.check(units):
            return units, apps
    return units, apps


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
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    # smoke 1: soft executor with one-hot slots must equal the python run on every example (known number: 0 loss)
    r = random.Random(3)
    bad = 0
    for _ in range(200):
        units = [r.choice(U) for _ in range(r.randint(1, 4))]
        fin = r.choice(["sum", "len", "first", "count_gt_3", ""])
        p = Problem(units, fin, 8, r.randint(0, 99))
        if any(o == "!" for o in p.out):
            continue
        hard = [torch.full((len(U) + 1,), -1e4).index_fill_(0, torch.tensor([U.index(u)]), 1e4) for u in units]
        bad += soft_loss(p, hard).item() > 1e-3
    print(json.dumps({"smoke_onehot_mismatch_of_200": int(bad)}), flush=True)
    if a.smoke:
        return
    res = {"n_ex": a.n_ex, "steps": a.steps, "restarts": a.restarts, "eval": {}}
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
            t0 = time.perf_counter()
            gu, gapps, gr = grad_search(p, n_slots, a.steps, a.restarts, i)
            g_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            ru, rapps = random_search(p, n_slots, gapps, i)
            r_s = time.perf_counter() - t0
            rows.append({"true": units, "final": fin, "grad": gu, "grad_apps": gapps, "grad_restarts": gr,
                         "grad_s": round(g_s, 2), "rand": ru, "rand_apps": rapps, "rand_s": round(r_s, 2),
                         "grad_fit": p.check(gu), "rand_fit": p.check(ru)})
        tasks_g = [(r_["grad"], r_["final"]) for r_ in rows]
        tasks_r = [(r_["rand"], r_["final"]) for r_ in rows]
        truth = [(r_["true"], r_["final"]) for r_ in rows]
        # strict bwrap grade: found program vs the true one on the 40 hidden inputs (output + list state)
        def hidden(found):
            # found body under the true name; `k=0` default so a found program of other arity still runs
            srcs = [[f"def {synth2.name_of(*t)}(xs, k=0):\n" + "".join(f"    {l}\n" for u in fu for l in UNITS[u])
                     + f"    {FINAL[ff]}\n"] for (fu, ff), t in zip(found, truth)]
            sc = grade(truth, srcs)
            return sum(s[0] == 1.0 for s in sc) / len(found)
        res["eval"][f"len{n_slots}"] = {
            "n": a.n, "grad_fit8": sum(r_["grad_fit"] for r_ in rows) / a.n,
            "rand_fit8": sum(r_["rand_fit"] for r_ in rows) / a.n,
            "grad_hidden40": hidden(tasks_g), "rand_hidden40": hidden(tasks_r),
            "mean_apps": sum(r_["grad_apps"] for r_ in rows) / a.n,
            "grad_s": round(sum(r_["grad_s"] for r_ in rows), 1), "rand_s": round(sum(r_["rand_s"] for r_ in rows), 1),
            "rows": rows}
        print(json.dumps({f"len{n_slots}": {k: v for k, v in res["eval"][f"len{n_slots}"].items() if k != "rows"}}),
              flush=True)
    json.dump(res, open(f"{a.out}/gradsearch.json", "w"), indent=1)


if __name__ == "__main__":
    main()
