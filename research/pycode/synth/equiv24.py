"""Helpers for guess_derive24 (review fixes, 2026-10-27): a symbolic canonical form of a skeleton's FUNCTION FAMILY
(constants are free parameters) and a numeric functional-equivalence check.

Canonical form (used for the held-out split hash). Every constant subtree is the anonymous free constant C (coarse on
purpose: merging families only moves more skeletons into the same split bucket, it can never leak). Rewrites:
  sub(x, C) = add(x, C)            add / avg offsets are pulled out of sums (affine node aff(t, slope, offset))
  mul(x, C) = div(x, C) = pct(x, C) = pct(C, x) = C * x         div(C, x) = C * inv(x)      div(x, y) = x * inv(y)
  add / mul are flattened and sorted (associativity + commutativity), scalings pulled out of products,
  avg(x, y) = .5 * (x + y), sub(x, y) = x + (-1) * y, op(C, C) = C, x op x special cases.
  Steps are expanded into one expression tree, so independent steps in a different order give the same key.
The form is NOT complete (maxx / hyp with constants, like-term collection, domain facts such as a, b >= 1 are not
modelled). The residual is MEASURED by func_equiv: fit a candidate skeleton's constants (batched Levenberg-Marquardt,
several starts) to the target function on fresh random inputs (not the hidden test inputs) and call it equivalent
when the max relative error |p - y| / (1 + |y|) < 1e-9 on every fresh input.
"""

import functools
import itertools
import math

import torch

OPS = ("add", "sub", "mul", "div", "avg", "hyp", "maxx", "pct")
NOP = len(OPS)
C = ("C",)
F = "F"                                           # free (constant) slope


def _isc(t):
    return t == C


def _sl(x, y):
    """slope product; F absorbs"""
    return F if F in (x, y) else round(x * y, 9)


def mk_aff(t, s, off):
    """s * t (+ C if off)"""
    if _isc(t):
        return C
    if t[0] == "aff":
        s, off = _sl(s, t[2]), off or t[3]
        t = t[1]
    if s != F:
        s = round(float(s), 9) + 0.0                   # one numeric spelling (1 == 1.0, no -0.0)
    if s == 1 and not off:
        return t
    return ("aff", t, s, off)


def _split(t):
    """-> (core, slope, offset)"""
    return (t[1], t[2], t[3]) if t[0] == "aff" else (t, 1, False)


def mk_add(x, y):
    off, terms = False, []
    for t in (x, y):
        if _isc(t):
            off = True
            continue
        core, s, o = _split(t)
        off |= o
        t = mk_aff(core, s, False)
        terms.extend(t[1] if t[0] == "add" else (t,))
    if not terms:
        return C
    neg = [mk_aff(t, -1, False) for t in terms]
    terms, neg = sorted(terms, key=repr), sorted(neg, key=repr)
    sg = 1
    if repr(neg) < repr(terms):                    # sign orientation: x - y and y - x differ by the factor -1 only
        terms, sg = neg, -1
    node = terms[0] if len(terms) == 1 else ("add", tuple(terms))
    return mk_aff(node, sg, off)


def _pos(t):
    """offset term s * x + C = s * (x + C') (C free), for multiplicative contexts -> (unit-slope form, s)"""
    if t[0] == "aff" and t[3]:
        return ("aff", t[1], 1, True), t[2]
    return t, 1


def mk_inv(t):
    if _isc(t):
        return C
    core, s, o = _split(t)
    if o:
        t, sg = _pos(t)
        return mk_aff(("inv", t), F if sg == F else round(1 / sg, 9), False)
    if core[0] == "inv":
        return mk_aff(core[1], F if s == F else round(1 / s, 9), False)
    if core[0] == "mul":
        return mk_aff(("mul", tuple(sorted((mk_inv(f) for f in core[1]), key=repr))), F if s == F else round(1 / s, 9),
                      False)
    return mk_aff(("inv", core), F if s == F else round(1 / s, 9), False)


def mk_mul(x, y):
    sc, fac = 1, []
    for t in (x, y):
        if _isc(t):
            sc = _sl(sc, F)
            continue
        core, s, o = _split(t)
        if o:
            t, sg = _pos(t)
            sc = _sl(sc, sg)
            fac.append(t)
            continue
        sc = _sl(sc, s)
        fac.extend(core[1] if core[0] == "mul" else (core,))
    if not fac:
        return C
    node = fac[0] if len(fac) == 1 else ("mul", tuple(sorted(fac, key=repr)))
    return mk_aff(node, sc, False)


def _flip(t):
    """-t when t has a negative numeric slope (s x (+ C) -> |s| x (+ C'))"""
    core, s, o = _split(t)
    return t if s == F or s >= 0 else mk_aff(core, -s, o)


def norm(op, x, y):
    name = OPS[op]
    if _isc(x) and _isc(y):
        return C
    if x == y:
        if name in ("sub", "div"):
            return C
        if name in ("avg", "maxx"):
            return x
    if name == "add":
        return mk_add(x, y)
    if name == "sub":
        return mk_add(x, mk_aff(y, -1, False))
    if name == "avg":
        return mk_aff(mk_add(x, y), 0.5, False)
    if name == "mul":
        return mk_mul(x, y)
    if name == "pct":
        return mk_aff(mk_mul(x, y), 0.01, False)
    if name == "div":
        return mk_mul(x, mk_inv(y))
    # maxx / hyp: commutative, no algebra; hyp(s * x, C) = |s| hyp(x, C) when no offset
    if name == "hyp" and (_isc(x) or _isc(y)):
        t = y if _isc(x) else x
        core, s, o = _split(t)
        if not o:
            return mk_aff(("hypc", core), F if s == F else abs(s), False)
        t, sg = _pos(t)
        return mk_aff(("hypc", t), F if sg == F else abs(sg), False)
    if name == "maxx" and (_isc(x) or _isc(y)):
        return ("maxc", y if _isc(x) else x)
    if name == "hyp":                              # hyp(-x, y) = hyp(x, y)
        x, y = (_flip(t) for t in (x, y))
    return (name,) + tuple(sorted((x, y), key=repr))


@functools.lru_cache(maxsize=None)
def canon_key(prog):
    """prog: tuple of (op, i, j) -> repr of the normalized expression tree (function family, free constants)"""
    s = []
    for k, (o, i, j) in enumerate(prog):
        src = [("a",), ("b",)] + s + [C]
        s.append(norm(o, src[i], src[j]))
    return repr(s[-1])


def uses_all(prog):
    """= gradconst.uses_all"""
    S = len(prog)
    used = {S - 1}
    for k in range(S - 1, -1, -1):
        if k in used:
            for x in prog[k][1:]:
                if 2 <= x < 2 + k:
                    used.add(x - 2)
    return len(used) == S and any(x == 2 + k for k, (_, i, j) in enumerate(prog) for x in (i, j))


def all_skeletons(S):
    steps = [[(o, i, j) for o in range(NOP) for i in range(3 + k) for j in range(3 + k)] for k in range(S)]
    return [p for p in itertools.product(*steps) if uses_all(p)]


@functools.lru_cache(maxsize=None)
def shorter_keys(S):
    """keys reachable by any valid skeleton shorter than S (a family that collapses to a shorter program)"""
    out = set()
    for L in range(1, S):
        out |= {canon_key(p) for p in all_skeletons(L)}
    return frozenset(out)


# ---------------- numeric functional equivalence ----------------

def _all_ops(x, y):
    d = torch.where(y.abs() < 1e-3, torch.full_like(y, 1e-3) * torch.sign(y + 1e-12), y)
    return torch.stack([x + y, x - y, x * y, x / d, (x + y) / 2, torch.sqrt(x * x + y * y + 1e-12),
                        torch.maximum(x, y), x * y / 100], -2)


def _run(ops, I, J, Cc, a, b):
    """ops/I/J [R, S], Cc [R, S], a/b [M] (shared inputs) -> [R, M]"""
    R, S = ops.shape
    M = a.shape[0]
    A, B = a.expand(R, M), b.expand(R, M)
    s = []
    for k in range(S):
        src = torch.stack([A, B] + s + [Cc[:, k:k + 1].expand(R, M)], 1)
        x = src.gather(1, I[:, k].view(R, 1, 1).expand(R, 1, M))[:, 0]
        y = src.gather(1, J[:, k].view(R, 1, 1).expand(R, 1, M))[:, 0]
        s.append(_all_ops(x, y).gather(1, ops[:, k].view(R, 1, 1).expand(R, 1, M))[:, 0])
    return s[-1]


def fresh_inputs(m=96, seed=24_2024):
    g = torch.Generator().manual_seed(seed)
    ab = torch.rand(2, m, generator=g, dtype=torch.float64) * 9 + 1
    return ab[0], ab[1]


def _lm(ops, I, J, c0, a, b, y, iters, tol):
    """plain batched LM (no budget accounting), rows independent; -> best constants, best max rel err"""
    R, S = ops.shape
    sc = 1 + y.abs()

    def resid(c):
        return (_run(ops, I, J, c, a, b) - y) / sc

    c = c0.clone()
    r = resid(c)
    fin = torch.isfinite(r).all(1)
    loss = torch.where(fin, (r ** 2).mean(1), torch.full((R,), float("inf"), dtype=c.dtype))
    lam = torch.full((R,), 1e-2, dtype=c.dtype)
    live = fin.clone()
    eye = torch.eye(S, dtype=c.dtype)
    for _ in range(iters):
        mx = torch.nan_to_num(r.abs(), nan=float("inf")).amax(1)
        live &= (mx >= tol) & (lam < 1e7)
        if not bool(live.any()):
            break
        rows = live.nonzero()[:, 0]
        o_, i_, j_, cr = ops[rows], I[rows], J[rows], c[rows]
        yr, sr = y[rows], sc[rows]
        f = lambda cc: (_run(o_, i_, j_, cc, a, b) - yr) / sr                      # noqa: E731
        Jm = torch.stack([torch.func.jvp(f, (cr,), (eye[k].expand_as(cr).contiguous(),))[1] for k in range(S)], 2)
        Jm = torch.nan_to_num(Jm, 0.0, 0.0, 0.0)
        A = Jm.transpose(1, 2) @ Jm
        dg = A.diagonal(dim1=1, dim2=2)
        Ad = A + torch.diag_embed(lam[rows, None] * (dg + 1e-9 * (1.0 + dg.amax(1, keepdim=True))))
        rhs = Jm.transpose(1, 2) @ torch.nan_to_num(r[rows], 0.0, 0.0, 0.0)[:, :, None]
        try:
            stp = -torch.linalg.solve(Ad, rhs)[:, :, 0]
        except RuntimeError:
            stp = -torch.linalg.lstsq(Ad, rhs).solution[:, :, 0]
        cand = cr + torch.nan_to_num(stp, 0.0, 0.0, 0.0)
        rc = f(cand)
        lc = torch.where(torch.isfinite(rc).all(1), (rc ** 2).mean(1), torch.full_like(loss[rows], float("inf")))
        acc = lc < loss[rows]
        acc_rows = rows[acc]
        c[acc_rows], r[acc_rows], loss[acc_rows] = cand[acc], rc[acc], lc[acc]
        lam[rows] = torch.where(acc, (lam[rows] / 3).clamp(min=1e-15), lam[rows] * 4)
    mx = torch.nan_to_num(r.abs(), nan=float("inf")).amax(1)
    return c, mx


def func_equiv(pairs, tol=1e-9, iters=60, n_rand=2, n_fit=24, chunk=60_000, seed=7):
    """pairs [(cand_prog, target_prog, target_consts)] (cand and target the same length) -> [bool]:
    the candidate skeleton, with constants fitted (batched LM, starts 1, -1, c, -c, 1/c, 100c, c/100 + n_rand random)
    on n_fit fresh inputs, reproduces the target within tol on ALL 96 fresh inputs (none of them a hidden test input)."""
    if not pairs:
        return []
    a, b = fresh_inputs()
    af, bf = a[:n_fit], b[:n_fit]
    g = torch.Generator().manual_seed(seed)
    out = [False] * len(pairs)
    by_S = {}
    for q, (cp, tp, tc) in enumerate(pairs):
        by_S.setdefault((len(cp), len(tp)), []).append(q)                  # candidate and target length may differ
    for (S, St), qs in by_S.items():
        rows = []                                                           # (pair index, start constants [S])
        for q in qs:
            tc = pairs[q][2].double()
            tc = tc[:S] if St >= S else torch.cat([tc, torch.ones(S - St, dtype=torch.float64)])
            sts = [torch.ones(S), -torch.ones(S), tc, -tc, 1 / tc, 100 * tc, tc / 100]
            for _ in range(n_rand):
                mag = torch.exp(torch.rand(S, generator=g, dtype=torch.float64) * (math.log(50) - math.log(0.2))
                                + math.log(0.2))
                sts.append(mag * torch.where(torch.rand(S, generator=g) < 0.5, 1.0, -1.0).double())
            rows.extend((q, st.double()) for st in sts)
        for s0 in range(0, len(rows), chunk):
            part = rows[s0:s0 + chunk]
            idx = [q for q, _ in part]
            cp = torch.tensor([pairs[q][0] for q in idx], dtype=torch.long)
            tp = torch.tensor([pairs[q][1] for q in idx], dtype=torch.long)
            tcs = torch.stack([pairs[q][2].double() for q in idx])
            y = _run(tp[:, :, 0], tp[:, :, 1], tp[:, :, 2], tcs, af, bf)
            c0 = torch.stack([st for _, st in part])
            cf, mx = _lm(cp[:, :, 0], cp[:, :, 1], cp[:, :, 2], c0, af, bf, y, iters, tol * 1e-2)
            good = ((mx < tol) & torch.isfinite(y).all(1)).nonzero()[:, 0]
            if len(good):                                                   # verify on all 96 fresh inputs
                yv = _run(tp[good, :, 0], tp[good, :, 1], tp[good, :, 2], tcs[good], a, b)
                pv = _run(cp[good, :, 0], cp[good, :, 1], cp[good, :, 2], cf[good], a, b)
                e = ((pv - yv).abs() / (1 + yv.abs())).amax(1)
                for r_, v in zip(good.tolist(), (torch.isfinite(e) & (e < tol)).tolist()):
                    out[idx[r_]] = out[idx[r_]] or v
    return out


def neighbourhood(prog):
    """same-length candidates around prog: per step every (op', i', j') that still uses the step's non-constant
    sources (product over steps), plus any single step replaced by any step. Raw skeletons, uses_all only."""
    S = len(prog)
    loc = []
    for k, (o, i, j) in enumerate(prog):
        c = 2 + k
        N = {x for x in (i, j) if x != c}
        pool = sorted(N | {c})
        loc.append([(o_, i_, j_) for o_ in range(NOP) for i_ in pool for j_ in pool if N <= {i_, j_}])
    out = set(itertools.product(*loc))
    for k in range(S):
        for o_ in range(NOP):
            for i_ in range(3 + k):
                for j_ in range(3 + k):
                    p = list(prog)
                    p[k] = (o_, i_, j_)
                    out.add(tuple(p))
    seen, res = set(), []
    for p in out:
        r = tuple((o, min(i, j), max(i, j)) if OPS[o] in COMM_OPS else (o, i, j) for o, i, j in p)
        if r not in seen and uses_all(r):
            seen.add(r)
            res.append(r)
    return res


COMM_OPS = ("add", "mul", "avg", "hyp", "maxx", "pct")
