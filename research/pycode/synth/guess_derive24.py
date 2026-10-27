"""Guess and derive with a REAL guesser (ledger #24, #25, #26). Written 2026-10-27 10:00 (time from `date`).

His words. #24 "isso pode virar um modelo 'matemático' algo que dependendo do problema ele resolve ou encontra o
melhor caminho... podemos usar derivada". #25 mixture of experts / modules. #26 "se ele chuta e deriva, a derivada vai
apontar a direção, basta o modelo seguir e derivar, até chegar em algum momento que a distância da derivação quase não
muda" - a loop in nano / micro seconds.

What was missing (audit 10-27): gradsearch.py / gradconst.py start from an ORACLE guess plus noise; no model read the
problem. Here the "chuta" is a learned net that READS the 8 (a, b) -> y examples and outputs a soft program over the
gradconst.py language (straight-line s_k = op_k(x, y), 8 calc ops = the experts, x / y from {a, b, s_1..s_k-1, c_k},
c_k a real constant), plus a guess of every constant. The "deriva" is gradient descent on the I/O loss of the relaxed
(soft) program, started from the guesser's soft output, stopped by HIS rule (relative |loss_t - loss_t-1| < eps for k
steps in a row), then snapped to a discrete program (constants polished by Levenberg-Marquardt, also the derivative),
then judged on hidden inputs. Restarts add Gumbel noise to the guesser's logits until the budget is spent. The derive
loop runs MANY problems at once as tensors (GPU-batched; one row per problem, per-row optimizer state and stop rule).

Own-bug check of the earlier code (10-27, before writing this file; probes in the session scratchpad gd24/):
  B1 Adam is not "follow the derivative": its first step is lr * sign(g) for every coordinate (measured: grads 1e-6 /
     1e-3 / 10 -> moves -.297 / -.300 / -.300). gradsearch (lr .3, 48 logits per slot) and gradconst.soft_derive moved
     every logit by the same amount whatever its derivative. Here: arm g_derive_gd = normalized gradient descent
     (step = lr * g / ||g|| per parameter group: the derivative's DIRECTION kept), next to Adam (g_derive).
  B2 gradconst.soft_derive (v3.3): fixed 100 steps x 3 S |OPS| = 72 executions per step at len3 -> ONE attempt per
     problem at len3 (dev probe 30 problems: 30 restarts total; len1 39, len2 39); his stop rule was not in that loop.
     The soft phase changed the argmax program in 14/39, 18/39, 14/30 attempts; the len3 wins were 15, of which only 4
     had a changed argmax (the rest = guess argmax right + LM polish). Here: his stop rule ends an attempt early, so
     flat attempts free budget for restarts.
  B3 gradconst lr on logits .03 (v3.1 fix) caps a logit's move at ~2.4 in 79 steps (measured mean max move 1.4-1.8),
     while the oracle guess margin is 4 +- 1.5: the derivative could rarely overturn the guess. Here lr is tuned on
     the dev set (grid, edge => extend) for both optimizers.
  B4 gradsearch guess_derive ran at sharpness T = 30: d sigmoid(30 (v - n - .5)) at integer v is <= 30 e^-15 ~ 1e-5, so
     masking units pass ~no derivative back to earlier slots (math, not measured on runs). Not in this language
     (no masks), noted for gradsearch.
  B5 his stop rule in the LM (gradconst, 5 accepted steps gaining < 0.1%): at the tuned cap 100, v3.3 sample_derive
     rows differ between stall and no-stall in 40 / 28 / 33 of 200 problems (apps), io identical in every row: the
     rule saved executions, never changed an outcome (lam > 1e8 needs >= 17 rejections, len3 has <= 12 iterations).
  B6 budget parity in gradsearch: a derive step is charged its forward only (backward free); sampling still won, so
     the v2.2 sampling win is not a budget artefact (the accounting favoured derive). gradconst charges 3 S |OPS| per
     soft step (forward + backward) - kept here.
  B7 every earlier "io" was graded in-process (torch). Here the produced program is printed as Python and executed in
     bwrap on the 40 hidden inputs (main metric); in-process io is kept and the disagreement count is reported.

Arms (all at the same budget B = 6000 program executions on the 8 examples, gradconst accounting: a discrete run 1,
an LM iteration 2 S + 2, a soft step 3 S |OPS|, a fit check 1):
  g_only           guesser argmax program + guessed constants, no search                         (floor of the guess)
  g_derive         HIS loop: soft program from the guesser, Adam on logits + constants, stop rule, snap, LM polish,
                   Gumbel restarts                                                               (batched on device)
  g_derive_gd      same, normalized gradient descent (follows the derivative's direction)      (batched on device)
  g_sample_es      programs sampled from the guesser's softmax, constants from the guess by (1+1)-ES: no derivative
                   anywhere - the "sample" control at the same number of executions
  g_sample_lm      programs sampled from the guesser, constants by LM from the guess (v3.3's winner, guesser instead
                   of oracle): derivative only where it is native
  prior_sample_lm  programs sampled from the TRAINING PRIOR (per-step frequencies, does not read the examples), LM from
                   1.0: gate 1 for "the guesser reads the problem"
  o_derive         oracle-noisy guess (gradconst.guess_logits, sigma 1.5, the old setting) + the same batched derive
  o_sample_lm      v3.3 sample_derive exactly (oracle guess, LM cap 100): POSITIVE CONTROL, rows must equal
                   gradconst33/stall/run_len*/gradconst.json on the v33 set
  r_derive         random start (Gumbel logits) + the same batched derive                        (no guess at all)
  base_a / base_b / base_mean   constant-answer baselines (gate 1)
Tuning (dev seeds only, before the test): lr per guess source (guesser / oracle / random) and optimizer, and eps of his
stop rule, grid; a best value at a grid edge blocks `eval` (gate 8). Smoke 10-27 showed why per source: on 20 dev
problems the oracle start wants Adam lr 0.01 (.75) and the guesser start 0.3; one shared lr made o_derive a straw man.
Attribution fields (rule 11): wins_soft_changed_argmax (the soft phase changed the program in the winning attempt, vs
start + LM polish), attempts_stopped_by_rule_frac (how often HIS rule, not the step cap, ended an attempt).
Metric: io_bwrap = the produced Python, run in bwrap on the 40 hidden inputs, within relative error 1e-4 of the truth
(output equivalence on held-out I/O). Also fit8, executions used, wall, canonical-skeleton match.

Data and splits. Training problems are generated on the fly on the device by a vectorized copy of gradconst.Problem's
rejection rule (same law: skeleton, constants sign * exp U[log .2, log 50] rounded to .01, a, b ~ U[1, 10], the same
uses-all / constant-moves / finite / not-flat filters). Canonical skeleton = (op, i, j) per step with (i, j) sorted for
the commutative ops (add mul avg hyp maxx pct). HELD-OUT COMPOSITIONS: for len >= 2, a canonical skeleton whose hash
falls in bucket 0 of 5 is never generated for training (len1 has ~40 skeletons, all are training atoms; len1 test
problems are new instances only). Test sets: `heldout` = gradconst.Problem(S, 90_000 S + i) scanning i until n
held-out-skeleton problems (len >= 2); `v33` = the exact v3.3 test problems (10_000 S + i), mixed seen / held-out
skeletons, reported per subset; `dev` = 50_000 S + i (gradconst's dev seeds) for lr / eps / temperature tuning only.
Leak not closed: two different canonical skeletons can compute the same function (sub(a, c) vs add(a, -c)).

PREDICTION (registered 2026-10-27 10:00, before any run of this file; GPU run, >= 3 guesser seeds, n 200 per length,
`heldout` set for len2/3, len1 on `v33`; numbers = mean over seeds of io_bwrap):
  P1 guesser reads the problem: g_only canonical-skeleton top-1 >= prior top-1 + 20 pp at every length; expected
     len1 .50-.85, len2 .20-.50, len3 .05-.30 (held-out compositions are harder than the oracle's .735/.59/.42).
  P2 g_sample_lm io > prior_sample_lm io + 15 pp at len2 and len3 (the guess, not the search, carries the gain).
  P3 (#26 core, HIS loop) g_derive (or g_derive_gd, whichever is better on DEV) io >= g_only io + 20 pp at every
     length (the derivative from the guess does work; g_only io is ~0 since constants must be right to 1e-4).
  P4 g_derive >= r_derive + 20 pp at len2 and len3 (the guess matters to the derivative).
  P5 against the idea, as v2.2 / v3.3: g_sample_lm >= g_derive + 10 pp at len3 (discrete by sampling, continuous by
     derivative beats all-soft derive). If g_derive wins by > 5 pp (paired CI > 0), the relaxed derivative beats
     sampling with a learned guess - that would reverse v2.2.
  P6 g_sample_lm >= g_sample_es + 5 pp at len3 (derivative on constants, v3.3 +8.5 pp with the oracle).
  P7 normalized GD (B1) vs Adam: |g_derive_gd - g_derive| <= 5 pp (no strong prior; reported, not judged).
  P8 speed: batched derive at P = 200 problems on the RX 6750 XT: <= 50 us per problem per soft step (CPU smoke: ms).
  CONTROL C1: o_sample_lm on v33 equals the recorded v3.3 sample_derive rows (fit, io, apps) in >= 99% of rows.
  KILL for the learned guess: P1 fails at len2 AND len3 (then the guesser does not read the problem; fix the encoder
  before any derive verdict). Verdict on HIS loop = P3 and P4 (paired bootstrap CI over problems, pooled seeds).

AMENDMENT (review fixes, written 2026-10-27 11:17, time from `date`; before ANY GPU run of this file; the 10:00 predictions above
are unchanged, this block only adds arms / subsets / gates the independent review asked for):
  F1 crash: guess_batch divided the masked arg logits (-1e9) by Tcal, derive_batch rebuilt the masks as value > NEG/2,
     so at Tcal 3 / 4 invalid slots became valid. BEFORE (review/probe_review.py): r_derive at Tcal 3.0 "CRASH:
     RuntimeError index 4 is out of bounds for dimension 1 with size 3"; 12 invalid slots unmasked at Tcal 3 and 4.
     NOW masks are built by index (arg_mask) in guess_batch and derive_batch. AFTER (smoke_fixes/smoke.json F1):
     Tcal 1.0 / 1.5 / 3.0 / 4.0 -> no crash, 0 programs with an out-of-range source.
  F2 held-out leak: the split hashed (op, i, j) with commutative sort only, so sub(x,c) / add(x,c'), mul / div / pct
     by a constant, associativity etc. put one function family on both sides. BEFORE (probe P3, single-step constant
     rewrites only = a lower bound): 101/200 len2 and 91/200 len3 held-out test problems had a trained equivalent.
     NOW the split hashes family_key = equiv24.canon_key (constants free; sub=add, mul=div=pct by a constant, add/mul
     flattened + sorted, affine offsets and scalings pulled out, sign orientation, step order via one expression
     tree); a family reachable by a len1 skeleton is never held out; a held-out family is excluded at EVERY length.
     Held-out share of test-seed problems: len2 .19, len3 .197 (was len3 .21). dev = train-side families only
     (smoke F2_dev_heldout_rows 0). The canonical form is NOT complete (maxx / hyp with constants, domain facts a,b>=1),
     so the residual is MEASURED: `equiv` -> heldout_equiv.json flags every held-out test problem for which a TRAINED
     skeleton (same-length neighbourhood + all len1) reproduces the function (constants fitted on 96 fresh inputs, not
     the hidden ones; max rel err < 1e-9); len2 calibrated against the exhaustive candidate set on 30 problems.
     verdict24.py: PRIMARY for heldout len2/3 = the no-equivalent subset (n < 200; all-held-out = secondary).
     MEASURED (2026-10-27 12:20, time from `date`; equiv24.py md5 b72269ff, heldout_equiv.json, 3664 s CPU): held-out test
     problems WITH a trained functional equivalent: len2 27/200 = .135, len3 119/200 = .595 (all 119 re-confirmed on
     a second input draw, 119/119; none of the candidates is a held-out skeleton). Primary subsets: len2 n=173,
     len3 n=81. len2 calibration (30 problems, exhaustive 1120 candidates): 3 found, neighbourhood also 3, missed 0.
     len3 is a lower bound (neighbourhood + len1 only); the old canon-hash split leaked most len3 test problems.
     P1 / KILL now use the FUNCTIONAL top-1 (guess_func_right: argmax skeleton + fitted constants equals the truth on
     fresh inputs); exact-skeleton top-1 kept as secondary.
  F3 run_gpu24.sh: no CPU job runs beside the GPU process (GPU part for all seeds first, then CPU sample arms strictly
     serially, 1 thread, HIP/CUDA_VISIBLE_DEVICES empty); per-step rc logged; step 0 = `gpucheck` (n 20 len3: cuda
     rows == cpu rows on fit and apps for lm_batch true / wrong programs and derive_batch oracle-Adam / random-GD /
     polish; exit 1 on any mismatch); torch.cuda.set_per_process_memory_fraction(0.3) on every cuda path.
  F4 tune: eps at a grid edge is flagged too (edge_eps_*, blocks eval like lr); ties go to the interior; eps grid
     1e-5 .. 1.0 (at 1.0 HIS rule fires after the minimum k_stop = 5 steps; CPU plumbing n=4: rule ended 6% / 52% /
     94% of attempts at eps 1e-4 / 1e-2 / .3, so the old grid <= 3e-2 rarely let it fire); grid_rule_frac recorded.
     New arms: g_polish = the same loop and budget with the soft phase removed (guesser argmax + Gumbel restarts + LM
     polish from the guessed constants); r_derive_gd = random start with normalized GD (tune source random|gd).
     New registered prediction P3b: dev-chosen derive - g_polish > 0 with paired CI low > 0 at each length (the
     derivative over the PROGRAM, not only on the constants, does work). P4 now uses the SAME optimizer per seed
     (dev choice Adam -> g_derive vs r_derive, GD -> g_derive_gd vs r_derive_gd); the Adam pair as written = secondary.
  Smoke after the fixes (CPU, smoke_fixes/smoke.json): C1 positive control len1 20/20 identical rows (io .95 = .95
     recorded), len3 20/20 (io .90 = .90); executors bit-identical; lm_batch vs fit_derive 60/60 fit + apps; ES copy
     20/20; bwrap truth 30/30, perturbed 0/30; S6 o_derive len3 fit 4/20 (unchanged); g_polish oracle len3 fit 18/20
     (apps <= 5988); r_derive_gd len3 fit 3/20. verdict24.py synthetic cases (review/test_verdict24_fixes.py): all
     recovered; a gain that exists only on leaked problems is refused on the primary subset.
  New usage:
  guess_derive24.py gpucheck --out DIR [--n-ctrl 20]              (exit 1 when cuda rows != cpu rows)
  guess_derive24.py equiv    --out DIR [--n 200 --n-cal 30]       (CPU; writes DIR/heldout_equiv.json)

AMENDMENT C (independent review B = NO-GO, lead decisions; written 2026-10-27 12:40, time from `date`; before ANY GPU run; the
10:00 predictions and thresholds stay):
  B2 budget units (lead decision). PRIMARY: one soft step = one forward + backward of the relaxed program ~ 3 NOP
     program runs -> charged 3 NOP (was 3 NOP S, header line B6 above). Every arm gets the same total budget 6000 in
     these units; P1-P5 are judged on it. SECONDARY: every soft-phase arm is RE-RUN with the old charge 3 NOP S as
     arm + "_oldc" (g_derive, g_derive_gd, o_derive, r_derive, r_derive_gd; g_polish has no soft step), same tuned
     lr / eps / k_stop; verdict24.py prints P3 / P3b / P4 / P5 on them (reported, never judged). Post-hoc re-charging
     of the primary trajectory was tried and rejected (smoke B2: oracle len3, real old-charge run 4/20 fit, post hoc
     0/20). Rows also carry apps_old (the primary trajectory re-charged, informational).
  B1 grid edges. Adam lr grid up to 30 (adds 10, 30); new k_stop axis (first {1, 2, 5, 10}; HIS rule: k_stop flat steps in
     a row) replaces "eps past 1.0": eps = 1.0 = "the rule always fires" and k_stop = 1 are natural ends, not edges;
     every other end value that is the only best is an edge. run_gpu24.sh: the first edge stops the whole run (exit
     3); s0 train + tune runs first and must have no edge before s1, s2 and any eval run.
     Re-review (2026-10-27 13:25, `date`): 1344 cells did not fit (reviewer: 33 s mean per CPU cell -> ~11 h per seed). Grid used
     by run_gpu24.sh now: Adam lr 0.003, 0.03, 0.3, 3, 30; GD lr 0.01, 0.1, 1, 10; eps 1e-5, 1e-3, 1e-2, 0.1, 1.0;
     k_stop 1, 3, 10 -> 345 cells. tune saves tune_sS.partial.json after every cell and resumes from it (same
     checkpoint md5 + grid); gpucheck times two real tune cells on the GPU and run_gpu24.sh sizes the tune / eval
     timeouts from the slower (exit 6 if one seed's tune would pass 8 h).
  B3 cross-length leak: equiv adds one skeleton per TRAINED len2 family (336) as candidates for every len3 target
     (reviewer ids 900045, 900297, 900979 = mul-family len2 equivalents; checked flagged before the re-run).
  B4 guards: run_gpu24.sh refuses to start (exit 5) while any other process holds the GPU (/sys/class/kfd/kfd/proc,
     also before every GPU step); the equiv file must hold len2, len3 (+ len2-family field) and the calibration;
     gpucheck uses DEV problems (no test problem), passes with <= 1 differing (fit, apps) row per check and the same
     total fit, and also checks the old-charge path; --gpucheck-dry runs it cpu vs cpu (exit 2, plumbing only).

Usage:
  guess_derive24.py smoke  --out DIR [--n-ctrl 20]                (CPU controls, positive control C1)
  guess_derive24.py train  --out DIR --seed S [--steps 20000 --batch 2048]
  guess_derive24.py tune   --out DIR --ckpt DIR/guesser_sS.pt [--n 60]  (dev: temperature, lr, eps, k_stop)
  guess_derive24.py eval   --out DIR --ckpt ... --set heldout|v33 --lens 1,2,3 --n 200 [--tuned DIR/tune_sS.json]
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import equiv24 as E  # noqa: E402
import gradconst as G  # noqa: E402

OPS = G.OPS                                   # add sub mul div avg hyp maxx pct
NOP, SMAX = len(OPS), 3
NSRC = 3 + SMAX - 1                           # padded argument slots (a, b, s_1..s_k-1, c_k)
COMM = torch.tensor([o in ("add", "mul", "avg", "hyp", "maxx", "pct") for o in OPS])
NEG = -1e9
B_DEF, CAP_LM, CAP_ES = 6000, 100, 300       # budget and v3.3 stall-mode tuned caps (gradconst33/tune_stall)
R_RUNS = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
V33 = R_RUNS + "/gradconst33/stall/run_len{S}/gradconst.json"


def dev_of(a):
    return torch.device("cuda" if (getattr(a, "device", "auto") == "auto" and torch.cuda.is_available())
                        else ("cpu" if getattr(a, "device", "auto") in ("auto", "cpu") else a.device))


# ---------------- skeleton canonical form and the held-out split ----------------

def canon(prog):
    """exact skeleton up to argument order of commutative ops (reporting only; the split uses family_key)"""
    return tuple((o, min(i, j), max(i, j)) if bool(COMM[o]) else (o, i, j) for o, i, j in prog)


def family_key(prog):
    """review fix 2: canonical FUNCTION FAMILY (constants free): sub(x,c)=add(x,c'), mul=div=pct by a constant,
    add/mul associativity + commutativity, affine offsets, sign, step order (equiv24.canon_key)"""
    return E.canon_key(tuple((int(o), int(i), int(j)) for o, i, j in prog))


def heldout(prog):
    """len >= 2: family key in hash bucket 0 of 5 is never trained on, at ANY length (a family reachable by a len1
    skeleton is always a training family: len1 has no held-out split)."""
    if len(prog) < 2:
        return False
    key = family_key(prog)
    if key in E.shorter_keys(2):
        return False
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 5 == 0


def arg_mask(S, dev="cpu"):
    """[S, NSRC] bool: argument slot valid at step k (a, b, s_1..s_k-1, c_k). Built by INDEX, never from the logit
    values (review fix 1: values divided by Tcal >= 3 climbed above NEG / 2 and unmasked invalid slots)."""
    return torch.stack([torch.arange(NSRC, device=dev) < 3 + k for k in range(S)])


# ---------------- batched exact executor (discrete) ----------------

def all_ops(x, y):
    """[..., M] x2 -> [..., NOP, M], the same formulas as gradconst.op_t"""
    d = torch.where(y.abs() < 1e-3, torch.full_like(y, 1e-3) * torch.sign(y + 1e-12), y)
    return torch.stack([x + y, x - y, x * y, x / d, (x + y) / 2, torch.sqrt(x * x + y * y + 1e-12),
                        torch.maximum(x, y), x * y / 100], -2)


def run_b(ops, I, J, C, a, b):
    """ops/I/J [P, S] long, C [P, S], a/b [P, M] -> [P, M]"""
    P, S = ops.shape
    M = a.shape[1]
    s = []
    for k in range(S):
        src = torch.stack([a, b] + s + [C[:, k:k + 1].expand_as(a)], 1)              # [P, 3+k, M]
        x = src.gather(1, I[:, k].view(P, 1, 1).expand(P, 1, M))[:, 0]
        y = src.gather(1, J[:, k].view(P, 1, 1).expand(P, 1, M))[:, 0]
        s.append(all_ops(x, y).gather(1, ops[:, k].view(P, 1, 1).expand(P, 1, M))[:, 0])
    return s[-1]


def soft_run(Lo, Li, Lj, C, a, b, T=1.0):
    """relaxed program: Lo [P, S, NOP], Li/Lj [P, S, NSRC] (invalid slots = NEG), C [P, S] -> [P, M]"""
    P, S, _ = Lo.shape
    s = []
    for k in range(S):
        src = torch.stack([a, b] + s + [C[:, k:k + 1].expand_as(a)], 1)              # [P, 3+k, M]
        x = (torch.softmax(Li[:, k, :3 + k] / T, -1)[:, :, None] * src).sum(1)
        y = (torch.softmax(Lj[:, k, :3 + k] / T, -1)[:, :, None] * src).sum(1)
        s.append((torch.softmax(Lo[:, k] / T, -1)[:, :, None] * all_ops(x, y)).sum(1))
    return s[-1]


def max_rel_err(p, y):
    e = ((p - y).abs() / (1 + y.abs())).amax(1)
    return torch.where(torch.isfinite(e), e, torch.full_like(e, float("inf")))


# ---------------- vectorized generator (same law as gradconst.Problem) ----------------

def gen_problems(S, n, g, dev, split="train", m_hid=G.N_HID):
    """-> dict ops, I, J, C [n, S], a, b, y [n, 8 + m_hid] (float64). split train: held-out skeletons excluded;
    'all': no split filter. Rejection = gradconst.Problem's rules."""
    got, keep = 0, []
    M = G.N_EX + m_hid
    while got < n:
        m = max(4 * (n - got), 256)
        ops = torch.randint(0, NOP, (m, S), generator=g)
        I = torch.stack([torch.randint(0, 3 + k, (m,), generator=g) for k in range(S)], 1)
        J = torch.stack([torch.randint(0, 3 + k, (m,), generator=g) for k in range(S)], 1)
        used = [torch.zeros(m, dtype=torch.bool) for _ in range(S)]
        used[S - 1][:] = True
        for k in range(S - 1, -1, -1):
            for arg in (I[:, k], J[:, k]):
                for t in range(k):
                    used[t] |= used[k] & (arg == 2 + t)
        cu = torch.stack([(I[:, k] == 2 + k) | (J[:, k] == 2 + k) for k in range(S)], 1)     # constant k used
        ok = torch.stack(used, 1).all(1) & cu.any(1)
        mag = torch.exp(torch.rand(m, S, generator=g, dtype=torch.float64) * (math.log(50) - math.log(0.2)) + math.log(0.2))
        sgn = torch.where(torch.rand(m, S, generator=g) < 0.5, 1.0, -1.0).double()
        C = torch.round(mag * sgn * 100) / 100
        ab = torch.rand(2, m, M, generator=g, dtype=torch.float64) * 9 + 1
        o_, I_, J_, C_, a_, b_ = ops.to(dev), I.to(dev), J.to(dev), C.to(dev), ab[0].to(dev), ab[1].to(dev)
        y = run_b(o_, I_, J_, C_, a_, b_)
        fin = torch.isfinite(y).all(1)
        ya = torch.where(torch.isfinite(y), y.abs(), torch.full_like(y, float("inf")))
        ok_d = ok.to(dev) & fin & (ya.amax(1) < 1e4) & (ya.amin(1) > 1e-3)
        yv = y[:, :G.N_EX]
        for k in range(S):
            Ck = C_ * torch.where(torch.arange(S, device=dev) == k, 1.1, 1.0)
            dv = (run_b(o_, I_, J_, Ck, a_, b_)[:, :G.N_EX] - yv).abs().amax(1)
            mv = dv > 1e-3 * (1 + yv.abs().amax(1))
            ok_d &= mv | ~cu[:, k].to(dev)
        flat = (y.amax(1) - y.amin(1)) <= 1e-6 * (1 + ya.amax(1))
        ok_d &= ~flat
        idx = ok_d.nonzero()[:, 0].cpu()
        if split == "train" and S >= 2 and len(idx):
            idx = torch.tensor([int(t) for t in idx if not heldout([(int(ops[t, k]), int(I[t, k]), int(J[t, k]))
                                                                       for k in range(S)])], dtype=torch.long)
        if len(idx):
            idx = idx[: n - got]
            keep.append((ops[idx], I[idx], J[idx], C[idx], ab[0][idx], ab[1][idx], y[idx.to(dev)].cpu()))
            got += len(idx)
    cat = [torch.cat(x) for x in zip(*keep)]
    return dict(zip(("ops", "I", "J", "C", "a", "b", "y"), cat))


# ---------------- the guesser ----------------

def slog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def feats(a, b, y, S):
    """examples sorted by a (order-invariant), 10 features each, + one-hot length -> [P, 8*10 + SMAX] float32"""
    o = a.argsort(1)
    a, b, y = a.gather(1, o), b.gather(1, o), y.gather(1, o)
    f = torch.stack([a / 10, b / 10, slog(y), slog(y - a), slog(y - b), slog(y / a), slog(y / b), slog(y - a - b),
                     slog(y - a * b), slog(y - torch.maximum(a, b))], -1).float()
    f = torch.nan_to_num(f, 0.0, 20.0, -20.0).clamp(-20, 20)
    sh = F.one_hot(torch.full((a.shape[0],), S - 1, device=a.device, dtype=torch.long), SMAX).float()
    return torch.cat([f.flatten(1), sh], 1)


class Guesser(nn.Module):
    """reads 8 examples -> per step: op logits [NOP], arg logits [NSRC] x 2 (invalid = NEG), constant sign + log|c|"""
    def __init__(self, H=512, depth=3):
        super().__init__()
        self.inp = nn.Linear(8 * 10 + SMAX, H)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H), nn.GELU()) for _ in range(depth))
        self.head = nn.Linear(H, SMAX * (NOP + 2 * NSRC + 2))

    def forward(self, x, S):
        h = F.gelu(self.inp(x))
        for blk in self.blocks:
            h = h + blk(h)
        o = self.head(h).view(-1, SMAX, NOP + 2 * NSRC + 2)[:, :S]
        mask = torch.stack([torch.arange(NSRC, device=x.device) < 3 + k for k in range(S)])        # [S, NSRC]
        Lo = o[..., :NOP]
        Li = o[..., NOP:NOP + NSRC].masked_fill(~mask, NEG)
        Lj = o[..., NOP + NSRC:NOP + 2 * NSRC].masked_fill(~mask, NEG)
        return Lo, Li, Lj, o[..., -2], o[..., -1]                  # sign logit, log|c|


def guess_loss(net, d, S, dev):
    x = feats(d["a"][:, :G.N_EX].to(dev), d["b"][:, :G.N_EX].to(dev), d["y"][:, :G.N_EX].to(dev), S)
    Lo, Li, Lj, sl, lm = net(x, S)
    ops, I, J, C = d["ops"].to(dev), d["I"].to(dev), d["J"].to(dev), d["C"].to(dev).float()
    lo = F.cross_entropy(Lo.reshape(-1, NOP), ops.reshape(-1))
    pi, pj = Li.log_softmax(-1), Lj.log_softmax(-1)
    g = lambda t, ix: t.gather(-1, ix[..., None])[..., 0]                                         # noqa: E731
    direct = g(pi, I) + g(pj, J)
    swap = g(pi, J) + g(pj, I)
    comm = COMM.to(dev)[ops] & (I != J)
    la = -torch.where(comm, torch.logaddexp(direct, swap), direct).mean()
    cu = torch.stack([(I[:, k] == 2 + k) | (J[:, k] == 2 + k) for k in range(S)], 1).float()
    lc = ((F.binary_cross_entropy_with_logits(sl, (C > 0).float(), reduction="none")
           + (lm - C.abs().log()) ** 2) * cu).sum() / cu.sum().clamp(min=1)
    return lo + la + lc, (Lo.argmax(-1) == ops).float().mean().item()


def train(a):
    dev = dev_of(a)
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(1_000_003 * (a.seed + 1))
    net = Guesser(a.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    log, t0 = [], time.time()
    for step in range(a.steps):
        S = 1 + step % SMAX
        d = gen_problems(S, a.batch, g, dev, "train")
        loss, acc = guess_loss(net, d, S, dev)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, a.steps // 50) == 0 or step == a.steps - 1:
            log.append({"step": step, "S": S, "loss": round(loss.item(), 4), "op_acc": round(acc, 4),
                        "s": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    ck = {"state": net.state_dict(), "hidden": a.hidden, "seed": a.seed, "steps": a.steps, "batch": a.batch,
          "prior": prior_logits(g, dev), "log": log}
    torch.save(ck, f"{a.out}/guesser_s{a.seed}.pt")
    return ck


def prior_logits(g, dev, n=20000):
    """training-prior per-step frequencies (does not read the examples) -> {S: (Lo, Li, Lj)} log-probs, cpu"""
    out = {}
    for S in range(1, SMAX + 1):
        d = gen_problems(S, n, g, dev, "train")
        lo = torch.stack([torch.bincount(d["ops"][:, k], minlength=NOP) for k in range(S)]).double() + 1
        li = torch.stack([F.pad(torch.bincount(d["I"][:, k], minlength=3 + k), (0, NSRC - 3 - k)) for k in range(S)]).double() + 1
        lj = torch.stack([F.pad(torch.bincount(d["J"][:, k], minlength=3 + k), (0, NSRC - 3 - k)) for k in range(S)]).double() + 1
        mask = torch.stack([torch.arange(NSRC) < 3 + k for k in range(S)])
        out[S] = (lo.log() - lo.sum(-1, keepdim=True).log(), li.log().masked_fill(~mask, NEG),
                  lj.log().masked_fill(~mask, NEG))
    return out


def load_guesser(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Guesser(ck["hidden"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


@torch.no_grad()
def guess_batch(net, pbs, S, dev, Tcal=1.0):
    """gradconst.Problem list -> padded logits [P, S, *] (cpu float64, divided by Tcal) and guessed constants [P, S]"""
    a = torch.stack([pb.a for pb in pbs]).to(dev)
    b = torch.stack([pb.b for pb in pbs]).to(dev)
    y = torch.stack([pb.y for pb in pbs]).to(dev)
    Lo, Li, Lj, sl, lm = net(feats(a, b, y, S), S)
    c0 = torch.where(sl > 0, 1.0, -1.0) * lm.clamp(-5, 6).exp()
    f = lambda t: t.double().cpu()                                                                # noqa: E731
    bad = ~arg_mask(S)                                                     # fix 1: re-mask by index after / Tcal
    return f(Lo) / Tcal, (f(Li) / Tcal).masked_fill(bad, NEG), (f(Lj) / Tcal).masked_fill(bad, NEG), f(c0)


def to_tuples(Lo, Li, Lj, p):
    """padded row p -> gradconst's per-step tuples (lo [NOP], li [3+k], lj [3+k])"""
    return [(Lo[p, k].clone(), Li[p, k, :3 + k].clone(), Lj[p, k, :3 + k].clone()) for k in range(Lo.shape[1])]


def pad_tuples(lgs):
    """list (per problem) of gradconst guess tuples -> padded Lo, Li, Lj"""
    S = len(lgs[0])
    Lo = torch.stack([torch.stack([t[0] for t in lg]) for lg in lgs])
    Li = torch.full((len(lgs), S, NSRC), NEG, dtype=torch.float64)
    Lj = Li.clone()
    for p, lg in enumerate(lgs):
        for k, (_, li, lj) in enumerate(lg):
            Li[p, k, :3 + k], Lj[p, k, :3 + k] = li, lj
    return Lo, Li, Lj


# ---------------- batched Levenberg-Marquardt (= gradconst.fit_derive, stall ON, per row) ----------------

def lm_batch(ops, I, J, C0, a, b, y, cap):
    """rows independent; same rule and accounting as gradconst.fit_derive (v3.3, STALL on). cap [n] long."""
    n, S = ops.shape
    scale = 1 + y.abs()
    resid = lambda c: (run_b(ops, I, J, c, a, b) - y) / scale                                     # noqa: E731
    c = C0.clone()
    r = resid(c)
    apps = torch.ones(n, dtype=torch.long, device=c.device)
    fin = torch.isfinite(r).all(1)
    loss = torch.where(fin, (r ** 2).mean(1), torch.full_like(r[:, 0], float("inf")))
    lam = torch.full((n,), 1e-2, dtype=c.dtype, device=c.device)
    hist = torch.full((n, 5), float("inf"), dtype=c.dtype, device=c.device)                      # last 5 accepted gains
    nacc = torch.zeros(n, dtype=torch.long, device=c.device)
    live = fin.clone()
    eye = torch.eye(S, dtype=c.dtype, device=c.device)
    while True:
        live &= (apps + 2 + 2 * S <= cap) & (torch.nan_to_num(r.abs(), nan=float("inf")).amax(1) >= G.TOL_FIT)
        if not bool(live.any()):
            break
        cols = []
        for k in range(S):
            _, jv = torch.func.jvp(resid, (c,), (eye[k].expand_as(c).contiguous(),))
            cols.append(jv)
        Jm = torch.stack(cols, 2)                                                                 # [n, 8, S]
        A = Jm.transpose(1, 2) @ Jm
        dg = A.diagonal(dim1=1, dim2=2)
        damp = lam[:, None] * (dg + 1e-9 * (1.0 + dg.amax(1, keepdim=True)))
        rhs = (Jm.transpose(1, 2) @ r[:, :, None])
        Ad = A + torch.diag_embed(damp)
        Ad = torch.where(live[:, None, None], Ad, eye.expand_as(Ad))
        step = -torch.linalg.solve(Ad, torch.where(live[:, None, None], rhs, torch.zeros_like(rhs)))[:, :, 0]
        cand = c + step
        rc = resid(cand)
        apps = apps + live.long() * (2 * S + 2)
        lc = torch.where(torch.isfinite(rc).all(1), (rc ** 2).mean(1), torch.full_like(loss, float("inf")))
        acc = live & (lc < loss)
        gain = (loss - lc) / loss.clamp(min=1e-300)
        c = torch.where(acc[:, None], cand, c)
        r = torch.where(acc[:, None], rc, r)
        hist = torch.where(acc[:, None], torch.cat([hist[:, 1:], gain[:, None]], 1), hist)
        nacc = nacc + acc.long()
        loss = torch.where(acc, lc, loss)
        lam = torch.where(acc, (lam / 3).clamp(min=1e-12), torch.where(live, lam * 4, lam))
        stall = (lam > 1e8) | ((nacc >= 5) & (hist.amax(1) < 1e-3))
        live &= ~stall
    return c, apps


# ---------------- ES with a start point (copy of gradconst.fit_es31, v3.3 stall rules; c0=None == original) ----------

def fit_es_c0(pb, prog, cap, g, c0=None):
    scale = 1 + pb.y.abs()

    def l2_mx(cc):
        with torch.no_grad():
            rr = (G.run(prog, cc, pb.a, pb.b) - pb.y) / scale
        if not torch.isfinite(rr).all():
            return float("inf"), float("inf")
        return float((rr ** 2).mean()), float(rr.abs().max())

    c = torch.ones(pb.S, dtype=torch.float64) if c0 is None else c0.clone()
    (best, mx), apps, sig, win, restarts = l2_mx(c), 1, 1.0, [], 0
    w_best, flat = best, 0
    while apps < cap and mx >= G.TOL_FIT:
        cand = c + sig * torch.randn(pb.S, generator=g, dtype=torch.float64) * (1 + c.abs())
        lc, mc = l2_mx(cand)
        apps += 1
        ok = lc < best
        if ok:
            c, best, mx = cand, lc, mc
        win.append(ok)
        if len(win) == 10:
            acc = sum(win)
            sig *= 1.5 if acc > 2 else 0.6
            win = []
            if acc > 0:
                flat = flat + 1 if (w_best - best) < 1e-3 * max(w_best, 1e-300) else 0
            w_best = best
            if G.STALL and flat >= 5:
                break
        if sig < 1e-9:
            restarts += 1
            if restarts > 1 and G.STALL:
                break
            sig = 1.0
    return c, apps


# ---------------- HIS loop, batched: guess -> derive until |dloss| ~ 0 -> snap -> polish -> judge ----------------

def derive_batch(pbs, Lo0, Li0, Lj0, c0, dev, opt="adam", lr=0.3, lr_c=0.3, T=1.0, budget=B_DEF, gumbel=1.0,
                 random_start=False, eps=1e-3, k_stop=5, steps_cap=100, polish=CAP_LM, seed=0, no_soft=False,
                 charge="real"):
    """all rows = problems of one length. Per row: soft attempt (Adam or normalized GD on logits + constants) until his
    stop rule (relative |loss_t - loss_t-1| < eps for k_stop steps in a row) or steps_cap or non-finite loss; snap the
    argmax; LM polish (cap `polish`) from the soft constants; fit check (1). Not fit and budget left -> restart from the
    guess + Gumbel noise. Every row stops at its own time; finished rows are frozen (no update, no charge).
    no_soft (arm g_polish, review fix 4): every attempt = argmax of the (Gumbel-perturbed) start, LM polish from the
    start constants, fit check - the same loop and budget with the soft phase removed.
    charge (AMENDMENT C): 'real' (PRIMARY) = 3 NOP program runs per soft step (one forward + backward of the relaxed
    program ~ 3 NOP runs); 'old' = 3 NOP S per soft step (the gradconst charge). apps_old in the output = the same
    trajectory re-charged at 3 NOP S per soft step (SECONDARY, post hoc: not a re-run)."""
    P, S = Lo0.shape[:2]
    a = torch.stack([pb.a for pb in pbs]).to(dev)
    b = torch.stack([pb.b for pb in pbs]).to(dev)
    y = torch.stack([pb.y for pb in pbs]).to(dev)
    base = [Lo0.to(dev), Li0.to(dev), Lj0.to(dev)]
    C0 = c0.to(dev)
    gen = torch.Generator().manual_seed(seed)
    cost_old = 3 * NOP * S
    cost = {"real": 3 * NOP, "old": cost_old}[charge]
    apps = torch.zeros(P, dtype=torch.long, device=dev)
    restart = torch.zeros(P, dtype=torch.long, device=dev)
    done = torch.zeros(P, dtype=torch.bool, device=dev)
    solved = torch.zeros(P, dtype=torch.bool, device=dev)
    res_prog = [None] * P
    res_c = [None] * P
    soft_steps = [[] for _ in range(P)]
    win_restart = [-1] * P
    steps_in = torch.zeros(P, dtype=torch.long, device=dev)
    flat = torch.zeros(P, dtype=torch.long, device=dev)
    prev = torch.full((P,), float("inf"), dtype=torch.float64, device=dev)
    ncap = torch.zeros(P, dtype=torch.long, device=dev)
    X = [t.clone() for t in base] + [C0.clone()]
    m1 = [torch.zeros_like(t) for t in X]
    m2 = [torch.zeros_like(t) for t in X]
    tt = torch.zeros(P, dtype=torch.long, device=dev)
    vm = arg_mask(S, dev).expand(P, S, NSRC)                            # fix 1: by index, not by value
    masks = [None, vm, vm, None]
    start_prog = torch.zeros(P, 3, S, dtype=torch.long, device=dev)      # argmax at the attempt's start
    changed = [[] for _ in range(P)]                                     # per attempt: soft phase changed the argmax
    by_rule = [0] * P                                                    # attempts ended by HIS stop rule

    def init_rows(rows):
        nonlocal X
        for q, t in enumerate(base + [C0]):
            if q < 3:
                u = torch.rand((len(rows),) + t.shape[1:], generator=gen, dtype=torch.float64).clamp(1e-9, 1 - 1e-9)
                gn = (-torch.log(-torch.log(u))).to(dev)
                use = (restart[rows] > 0) | random_start
                start = (0 * t[rows] if random_start else t[rows]) + gumbel * gn * use.view(-1, 1, 1).double()
                if masks[q] is not None:
                    start = start.masked_fill(~masks[q][rows], NEG)
            else:
                start = t[rows]
            X[q][rows] = start
            m1[q][rows] = 0
            m2[q][rows] = 0
        tt[rows] = 0
        start_prog[rows] = torch.stack([X[0][rows].argmax(-1), X[1][rows].argmax(-1), X[2][rows].argmax(-1)], 1)
        steps_in[rows] = 0
        flat[rows] = 0
        prev[rows] = float("inf")
        if no_soft:                                 # one attempt needs only the polish reserve
            ncap[rows] = torch.where(budget - apps[rows] - polish - 1 >= 0, 0, -1)
        else:
            ncap[rows] = torch.clamp((budget - apps[rows] - polish - 1) // cost, max=steps_cap)

    init_rows(torch.arange(P, device=dev))
    done |= ncap < (0 if no_soft else 1)
    t_steps, n_row_steps, t0 = 0.0, 0, time.perf_counter()
    while not bool(done.all()):
        act = ~done
        if no_soft:                                 # g_polish: the attempt ends at once (no soft step, no charge)
            rows = act.nonzero()[:, 0]
        else:
            ts = time.perf_counter()
            params = [t.detach().requires_grad_(True) for t in X]
            out = soft_run(params[0], params[1], params[2], params[3], a, b, T)
            L = (((out - y) / (1 + y.abs())) ** 2).mean(1)
            finite = torch.isfinite(L)
            grads = torch.autograd.grad(torch.where(finite & act, L, torch.zeros_like(L)).sum(), params)
            grads = [torch.nan_to_num(g_, 0.0, 0.0, 0.0) for g_ in grads]
            upd = act & finite
            tt = tt + upd.long()
            for q in range(4):
                g_ = grads[q]
                sh = (-1,) + (1,) * (g_.dim() - 1)
                if opt == "adam":
                    m1[q] = torch.where(upd.view(sh), 0.9 * m1[q] + 0.1 * g_, m1[q])
                    m2[q] = torch.where(upd.view(sh), 0.999 * m2[q] + 0.001 * g_ * g_, m2[q])
                    tq = tt.clamp(min=1).double().view(sh)
                    stp = (lr if q < 3 else lr_c) * (m1[q] / (1 - 0.9 ** tq)) / ((m2[q] / (1 - 0.999 ** tq)).sqrt() + 1e-8)
                else:                                   # normalized gradient descent: the derivative's direction
                    if q < 3:
                        stp = lr * g_ / (g_.flatten(1).norm(dim=1).view(sh) + 1e-12)
                    else:                               # constants: per-row normalized, scaled like Adam's step
                        stp = lr_c * g_ / (g_.norm(dim=1, keepdim=True) + 1e-12)
                new = X[q] - stp
                if masks[q] is not None:
                    new = new.masked_fill(~masks[q], NEG)
                X[q] = torch.where(upd.view(sh), new, X[q])
            apps = apps + act.long() * cost
            steps_in = steps_in + act.long()
            rel = (prev - L).abs() / prev.abs().clamp(min=1e-300)
            flat = torch.where(act & finite & (rel < eps), flat + 1, torch.zeros_like(flat))
            prev = torch.where(act, L.detach(), prev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_steps += time.perf_counter() - ts
            n_row_steps += int(act.sum())
            end = act & ((flat >= k_stop) | (steps_in >= ncap) | ~finite | (L.detach() < 1e-20))
            rows = end.nonzero()[:, 0]
        if len(rows) == 0:
            continue
        ops = X[0][rows].argmax(-1)
        I = X[1][rows].argmax(-1)
        Jx = X[2][rows].argmax(-1)
        cst = torch.where(torch.isfinite(X[3][rows]), X[3][rows], C0[rows])
        capv = torch.clamp(budget - apps[rows] - 1, min=0, max=polish)
        cf, used = lm_batch(ops, I, Jx, cst, a[rows], b[rows], y[rows], capv)
        apps[rows] += used + 1
        ok = max_rel_err(run_b(ops, I, Jx, cf, a[rows], b[rows]), y[rows]) < G.TOL
        for q_, p in enumerate(rows.tolist()):
            soft_steps[p].append(int(steps_in[p]))
            changed[p].append(bool((start_prog[p] != torch.stack([ops[q_], I[q_], Jx[q_]])).any()))
            by_rule[p] += int(flat[p] >= k_stop)
            res_prog[p] = [(int(ops[q_, k]), int(I[q_, k]), int(Jx[q_, k])) for k in range(S)]
            res_c[p] = cf[q_].detach().cpu()
            if bool(ok[q_]):
                solved[p] = True
                done[p] = True
                win_restart[p] = int(restart[p])
        rest = rows[~ok]
        if len(rest):
            restart[rest] += 1
            init_rows(rest)
            done[rest] |= ncap[rest] < (0 if no_soft else 1)
    wall = time.perf_counter() - t0
    out = []
    for p in range(P):
        out.append({"fit": bool(solved[p]), "prog": res_prog[p], "c": res_c[p], "apps": int(apps[p]),
                    "apps_old": int(apps[p]) + (cost_old - cost) * sum(soft_steps[p]),
                    "restarts": int(restart[p]), "soft_steps": soft_steps[p],
                    "win_restart": win_restart[p], "changed": changed[p],
                    "win_changed": bool(solved[p]) and changed[p][-1],
                    "stopped_by_rule": by_rule[p]})
    return out, {"wall_s": round(wall, 3), "step_wall_s": round(t_steps, 3), "row_steps": n_row_steps,
                 "us_per_problem_step": round(1e6 * t_steps / max(1, n_row_steps), 2)}


# ---------------- bwrap judge ----------------

PRELUDE = """import math
def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
def div(x, y):
    if abs(y) < 1e-3:
        s = y + 1e-12
        y = 1e-3 * (1.0 if s > 0 else (-1.0 if s < 0 else 0.0))
    return x / y
def avg(x, y): return (x + y) / 2
def hyp(x, y): return math.sqrt(x * x + y * y + 1e-12)
def maxx(x, y): return max(x, y)
def pct(x, y): return x * y / 100
"""

RUNNER = r"""
import json, math, sys
items = json.load(sys.stdin)
out = []
for src, pts in items:
    g = {}
    try:
        exec(src, g)
        row = []
        for a, b in pts:
            try:
                v = float(g["main"](a, b))
                row.append(v if math.isfinite(v) else None)
            except Exception:
                row.append(None)
        out.append(row)
    except Exception:
        out.append(None)
print(json.dumps(out))
"""


def py_source(prog, c):
    lines = []
    for k, (o, i, j) in enumerate(prog):
        names = ["a", "b"] + [f"s{t + 1}" for t in range(k)] + [repr(float(c[k]))]
        lines.append(f"    s{k + 1} = {OPS[o]}({names[i]}, {names[j]})")
    return PRELUDE + "def main(a, b):\n" + "\n".join(lines) + f"\n    return s{len(prog)}\n"


def bwrap_run(items):
    """items [(src, [(a, b)])] -> [[float|None] | None]; the same sandbox flags as multifn.run_many"""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    try:
        p = subprocess.run(cmd, input=json.dumps(items), capture_output=True, text=True, timeout=60 + len(items))
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        if len(items) == 1:
            return [None]
        h = len(items) // 2
        return bwrap_run(items[:h]) + bwrap_run(items[h:])


def judge_bwrap(pbs, found):
    """found [(prog, c) | (None, None)] -> [bool] io on the 40 hidden inputs, executed in bwrap"""
    items, where = [], []
    for q, (pb, (prog, c)) in enumerate(zip(pbs, found)):
        if prog is None:
            continue
        items.append((py_source(prog, c), [(float(x), float(z)) for x, z in zip(pb.ha, pb.hb)]))
        where.append(q)
    res = [False] * len(pbs)
    outs = bwrap_run(items) if items else []
    for q, o in zip(where, outs):
        if o is None or any(v is None for v in o):
            continue
        hy = pbs[q].hy.tolist()
        res[q] = max(abs(v - t) / (1 + abs(t)) for v, t in zip(o, hy)) < G.TOL
    return res


# ---------------- test sets ----------------

def problems(set_, S, n):
    if set_ == "v33":
        return [G.Problem(S, 10_000 * S + i) for i in range(n)], list(range(n))
    if set_ == "dev":                         # fix 2: dev = train-side families only (never a held-out one)
        out, ids, i = [], [], 0
        while len(out) < n:
            pb = G.Problem(S, 50_000 * S + i)
            if not heldout(pb.prog):
                out.append(pb)
                ids.append(777_000 + i)
            i += 1
        return out, ids
    out, ids, i = [], [], 0
    while len(out) < n:                       # heldout: len >= 2 held-out skeletons only; len1 = fresh instances
        pb = G.Problem(S, 90_000 * S + i)
        if S == 1 or heldout(pb.prog):
            out.append(pb)
            ids.append(900_000 + i)
        i += 1
    return out, ids


def evaluate(a, net, ck, dev, set_, S, n, cfg, arms):
    pbs, seeds = problems(set_, S, n)
    rows = [{"id": sd, "truth": G.text(pb.prog, pb.consts), "heldout_skel": heldout(pb.prog), "arms": {}} for pb, sd in
            zip(pbs, seeds)]
    found = {}
    Tc = cfg.get("Tcal", 1.0)
    t0 = time.perf_counter()
    Lo, Li, Lj, c0 = guess_batch(net, pbs, S, torch.device("cpu"), Tc)      # net is a CPU copy (see main)
    t_guess = time.perf_counter() - t0
    glg = [to_tuples(Lo, Li, Lj, p) for p in range(len(pbs))]
    pr = ck["prior"][S] if isinstance(ck["prior"], dict) and S in ck["prior"] else ck["prior"][str(S)]
    plg = [[(pr[0][k].clone(), pr[1][k, :3 + k].clone(), pr[2][k, :3 + k].clone()) for k in range(S)] for _ in pbs]
    olg = [G.guess_logits(pb, 1.5, torch.Generator().manual_seed(sd)) for pb, sd in zip(pbs, seeds)]
    pairs = []
    for r, pb, lg, pl, ol in zip(rows, pbs, glg, plg, olg):
        r["guess_canon_right"] = canon(G.argmax_prog(lg)) == canon(pb.prog)
        r["prior_canon_right"] = canon(G.argmax_prog(pl)) == canon(pb.prog)
        r["oracle_canon_right"] = canon(G.argmax_prog(ol)) == canon(pb.prog)
        for lgx in (lg, pl, ol):
            pairs.append((tuple(G.argmax_prog(lgx)), tuple(pb.prog), pb.consts))
    # fix 2: top-1 by FUNCTION: the argmax skeleton, constants fitted on fresh inputs (no hidden test input), equals
    # the truth within 1e-9 relative (KILL test: a functionally equivalent program counts as right)
    feq = E.func_equiv(pairs)
    for q, r in enumerate(rows):
        r["guess_func_right"], r["prior_func_right"], r["oracle_func_right"] = feq[3 * q:3 * q + 3]
    stats = {"guess_wall_s": round(t_guess, 4)}
    if "g_only" in arms:
        found["g_only"] = []
        for r, pb, lg, cc in zip(rows, pbs, glg, c0):
            prog = G.argmax_prog(lg)
            fit = pb.fits(prog, cc)
            r["arms"]["g_only"] = {"fit": fit, "apps": 1, "apps_old": 1, "io_inproc": G.io_ok(pb, prog, cc)}
            found["g_only"].append((prog, cc) if fit else (None, None))
    sample_arms = {
        "g_sample_lm": (glg, lambda c_: (lambda p, q, cap, _g: G.fit_derive(p, q, cap, c0=c_)), CAP_LM, True),
        "g_sample_es": (glg, lambda c_: (lambda p, q, cap, g_: fit_es_c0(p, q, cap, g_, c0=c_)), CAP_ES, True),
        "prior_sample_lm": (plg, lambda c_: (lambda p, q, cap, _g: G.fit_derive(p, q, cap)), CAP_LM, False),
        "o_sample_lm": (olg, lambda c_: (lambda p, q, cap, _g: G.fit_derive(p, q, cap)), CAP_LM, False),
    }
    for arm, (lgs, mk, cap, use_c0) in sample_arms.items():
        if arm not in arms:
            continue
        found[arm] = []
        for q, (r, pb, sd) in enumerate(zip(rows, pbs, seeds)):
            t1 = time.perf_counter()
            prog, c, apps = G.sample_search(pb, lgs[q], a.budget, cap, mk(c0[q] if use_c0 else None), sd)
            r["arms"][arm] = {"fit": prog is not None, "apps": apps, "apps_old": apps, "io_inproc": G.io_ok(pb, prog, c),
                              "wall_s": round(time.perf_counter() - t1, 4),
                              "canon_right": prog is not None and canon(prog) == canon(pb.prog)}
            found[arm].append((prog, c))
    ones = torch.ones(len(pbs), S, dtype=torch.float64)
    der = {"g_derive": (Lo, Li, Lj, c0, False, "adam", "guesser", False),
           "g_derive_gd": (Lo, Li, Lj, c0, False, "gd", "guesser", False),
           "g_polish": (Lo, Li, Lj, c0, False, "adam", "guesser", True),      # fix 4: no soft phase, same loop/budget
           "o_derive": (*pad_tuples(olg), ones, False, "adam", "oracle", False),
           "r_derive": (Lo, Li, Lj, ones, True, "adam", "random", False),
           "r_derive_gd": (Lo, Li, Lj, ones, True, "gd", "random", False)}     # fix 4: P4 with the same optimizer
    # AMENDMENT C: the PRIMARY arms are charged 3 NOP per soft step; arm + "_oldc" = the same arm RE-RUN with the old
    # charge 3 NOP S per soft step (SECONDARY; same tuned lr / eps / k_stop). g_polish has no soft step (no _oldc).
    for k_ in [k_ for k_ in der if k_ != "g_polish"]:
        der[k_ + "_oldc"] = der[k_]
    for arm, (A, Bx, Cx, cc, rnd, opt, src, nos) in der.items():
        chg = "old" if arm.endswith("_oldc") else "real"
        if arm not in arms:
            continue
        lr = cfg.get(f"lr_{src}_{opt}", 0.3)          # lr tuned per guess source on dev (fairness: no straw man)
        eps = cfg.get(f"eps_{src}_{opt}", 1e-3)
        ks = cfg.get(f"kstop_{src}_{opt}", 5)            # review B fix 1: k_stop tuned with lr and eps
        out, st = derive_batch(pbs, A, Bx, Cx, cc, dev, opt=opt, lr=lr, T=1.0, budget=a.budget, gumbel=1.0,
                               random_start=rnd, eps=eps, k_stop=ks,
                               steps_cap=a.steps_cap, polish=CAP_LM, seed=12345 + S, no_soft=nos, charge=chg)
        stats[arm] = {**st, "lr": None if nos else lr, "eps": None if nos else eps, "k_stop": None if nos else ks,
                      "charge": {"real": "3 NOP per soft step", "old": "3 NOP S per soft step"}[chg]}
        found[arm] = []
        for r, pb, o in zip(rows, pbs, out):
            prog, c = (o["prog"], o["c"]) if o["fit"] else (None, None)
            r["arms"][arm] = {"fit": o["fit"], "apps": o["apps"], "apps_old": o["apps_old"],
                              "io_inproc": G.io_ok(pb, prog, c),
                              "restarts": o["restarts"], "soft_steps": o["soft_steps"], "win_restart": o["win_restart"],
                              "win_changed": o["win_changed"], "attempts_changed": sum(o["changed"]),
                              "attempts": len(o["changed"]), "stopped_by_rule": o["stopped_by_rule"],
                              "canon_right": prog is not None and canon(prog) == canon(pb.prog)}
            found[arm].append((prog, c))
    for arm, fl in found.items():
        jb = judge_bwrap(pbs, fl)
        for r, x in zip(rows, jb):
            r["arms"][arm]["io_bwrap"] = x
    for r, pb in zip(rows, pbs):                                    # gate-1 constant answers (in-process)
        for nm, v in (("base_a", pb.ha), ("base_b", pb.hb), ("base_mean", pb.y.mean().expand_as(pb.hy))):
            r["arms"][nm] = {"io_bwrap": float(((v - pb.hy).abs() / (1 + pb.hy.abs())).max()) < G.TOL}
    summ = {"n": len(rows), "n_heldout_skel": sum(r["heldout_skel"] for r in rows),
            "guess_canon_top1": statistics.mean(r["guess_canon_right"] for r in rows),
            "prior_canon_top1": statistics.mean(r["prior_canon_right"] for r in rows),
            "oracle_canon_top1": statistics.mean(r["oracle_canon_right"] for r in rows),
            "guess_func_top1": statistics.mean(r["guess_func_right"] for r in rows),
            "prior_func_top1": statistics.mean(r["prior_func_right"] for r in rows),
            "oracle_func_top1": statistics.mean(r["oracle_func_right"] for r in rows)}
    for arm in list(found) + ["base_a", "base_b", "base_mean"]:
        xs = [r["arms"][arm] for r in rows]
        s_ = {"io_bwrap": statistics.mean(x["io_bwrap"] for x in xs)}
        if "fit" in xs[0]:
            s_.update({"fit8": statistics.mean(x["fit"] for x in xs),
                       "io_inproc_disagree": sum(x["io_bwrap"] != x["io_inproc"] for x in xs),
                       "apps_median": statistics.median(x["apps"] for x in xs),
                       "apps_old_median": statistics.median(x["apps_old"] for x in xs),
                       "io_bwrap_old_charge": statistics.mean(x["io_bwrap"] and x["apps_old"] <= a.budget for x in xs)})
        if any(r["heldout_skel"] for r in rows) and not all(r["heldout_skel"] for r in rows):
            s_["io_bwrap_heldout_skel"] = statistics.mean(r["arms"][arm]["io_bwrap"] for r in rows if r["heldout_skel"])
            s_["io_bwrap_seen_skel"] = statistics.mean(r["arms"][arm]["io_bwrap"] for r in rows if not r["heldout_skel"])
        if "win_changed" in xs[0]:      # rule 11: wins where the derivative changed the program vs start + polish
            s_["wins_soft_changed_argmax"] = sum(x["win_changed"] for x in xs)
            s_["attempts_mean"] = statistics.mean(x["attempts"] for x in xs)
            s_["attempts_stopped_by_rule_frac"] = sum(x["stopped_by_rule"] for x in xs) / max(1, sum(x["attempts"] for x in xs))
            s_["attempts_changed_frac"] = sum(x["attempts_changed"] for x in xs) / max(1, sum(x["attempts"] for x in xs))
        summ[arm] = s_
    return {"summary": summ, "stats": stats, "rows": rows, "pbs_seeds": seeds}


def control_v33(res, S):
    """C1: o_sample_lm rows vs the recorded v3.3 stall sample_derive rows (fit, io, apps)"""
    rec = json.load(open(V33.format(S=S)))["lens"][f"len{S}"]["rows"]
    rows = res["rows"]
    same = sum(r["arms"]["o_sample_lm"]["fit"] == q["sample_derive"]["fit"] and
               r["arms"]["o_sample_lm"]["io_inproc"] == q["sample_derive"]["io"] and
               r["arms"]["o_sample_lm"]["apps"] == q["sample_derive"]["apps"] for r, q in zip(rows, rec))
    return {"rows": len(rows), "identical": same,
            "io_ours": statistics.mean(r["arms"]["o_sample_lm"]["io_inproc"] for r in rows),
            "io_recorded": statistics.mean(q["sample_derive"]["io"] for q in rec[:len(rows)])}


# ---------------- tune (dev only) ----------------

@torch.no_grad()
def calib_T(net, dev, n=300):
    """temperature scaling of the guesser's op/arg logits on dev problems (NLL of the true canonical choices)"""
    best = None
    for T in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        nll, cnt = 0.0, 0
        for S in (1, 2, 3):
            pbs, _ = problems("dev", S, n)
            Lo, Li, Lj, _ = guess_batch(net, pbs, S, dev, T)
            for p, pb in enumerate(pbs):
                for k, (o, i, j) in enumerate(pb.prog):
                    nll -= float(Lo[p, k].log_softmax(0)[o] + Li[p, k, :3 + k].log_softmax(0)[i]
                                 + Lj[p, k, :3 + k].log_softmax(0)[j])
                    cnt += 1
        if best is None or nll / cnt < best[1]:
            best = (T, nll / cnt)
    return best[0]


def tune(a):
    """dev seeds only (train-side families). Per guess source (guesser / oracle / random) and optimizer: lr x eps x
    k_stop grid, best mean io over lens 1-3. Edge rule (review fixes 4 + B1): a best value at a grid edge is flagged
    (run_gpu24.sh stops the whole run). 'edge' = the best score is reached only at that axis' edge (ties go to the
    interior, then the smallest lr / the middle eps / the middle k_stop). Natural ends are NOT edges: eps = 1.0 means
    'HIS rule always fires' (relative change < 1 after k_stop steps), so going further along eps is done by the k_stop
    axis; k_stop = 1 (stop after one flat step) cannot go lower."""
    dev = dev_of(a)
    net, ck = load_guesser(a.ckpt, torch.device("cpu"))
    Tc = calib_T(net, torch.device("cpu"), a.n)
    out = {"Tcal": Tc, "grid": {}, "grid_rule_frac": {}, "rule_count": {}, "charge": "real (3 NOP per soft step)"}
    grids = {"adam": [float(x) for x in a.lr_adam.split(",")], "gd": [float(x) for x in a.lr_gd.split(",")]}
    egrid = [float(x) for x in a.eps_grid.split(",")]
    kgrid = [int(x) for x in a.kstop_grid.split(",")]
    combos = [("guesser", "adam"), ("guesser", "gd"), ("oracle", "adam"), ("random", "adam"), ("random", "gd")]
    # review B2 fix 1: the grid is saved after EVERY cell; a re-run with the same checkpoint and grid resumes
    meta = {"ckpt_md5": hashlib.md5(open(a.ckpt, "rb").read()).hexdigest(), "lr_adam": a.lr_adam, "lr_gd": a.lr_gd,
            "eps_grid": a.eps_grid, "kstop_grid": a.kstop_grid, "n": a.n, "budget": a.budget,
            "steps_cap": a.steps_cap, "Tcal": Tc}
    part = f"{a.out}/tune_s{ck['seed']}.partial.json"
    out["meta"] = meta
    if os.path.exists(part):
        old = json.load(open(part))
        if old.get("meta") == meta:
            out["grid"], out["grid_rule_frac"], out["rule_count"] = old["grid"], old["grid_rule_frac"], old["rule_count"]
            print(f"resume: {len(out['grid'])} cells from {part}", flush=True)
        else:
            print(f"{part} was made with another checkpoint / grid: ignored", flush=True)
    cache = {}
    for src, opt in combos:
        grid = grids[opt]
        t_c = time.perf_counter()
        for lr in grid:
            for eps in egrid:
                never = None                    # exact shortcut: the rule never fired at a smaller k_stop ->
                for ks in sorted(kgrid):        # flat >= ks' > ks never holds either -> identical trajectories
                    key = f"{src}|{opt}|{lr}|{eps}|{ks}"
                    if key in out["grid"]:          # resumed cell
                        if out["rule_count"][key] == 0 and never is None:
                            never = key
                        continue
                    if never is not None:
                        out["grid"][key], out["grid_rule_frac"][key] = out["grid"][never], 0.0
                        out["rule_count"][key] = 0
                        print(src, opt, lr, eps, ks, out["grid"][key], "copied from", never, flush=True)
                        continue
                    n_rule, n_att = 0, 0
                    hits, tot = 0, 0
                    for S in (1, 2, 3):
                        if (src, S) not in cache:
                            pbs, seeds = problems("dev", S, a.n)
                            Lo, Li, Lj, c0 = guess_batch(net, pbs, S, torch.device("cpu"), Tc)
                            if src == "oracle":
                                Lo, Li, Lj = pad_tuples([G.guess_logits(pb, 1.5, torch.Generator().manual_seed(sd))
                                                         for pb, sd in zip(pbs, seeds)])
                            if src != "guesser":
                                c0 = torch.ones(len(pbs), S, dtype=torch.float64)
                            cache[(src, S)] = (pbs, Lo, Li, Lj, c0)
                        pbs, Lo, Li, Lj, c0 = cache[(src, S)]
                        res, _ = derive_batch(pbs, Lo, Li, Lj, c0, dev, opt=opt, lr=lr, budget=a.budget, eps=eps,
                                              k_stop=ks, random_start=src == "random", steps_cap=a.steps_cap,
                                              seed=777 + S)
                        hits += sum(G.io_ok(pb, r["prog"], r["c"]) if r["fit"] else False for pb, r in zip(pbs, res))
                        tot += len(pbs)
                        n_rule += sum(r["stopped_by_rule"] for r in res)
                        n_att += sum(len(r["changed"]) for r in res)
                    out["grid"][key] = hits / tot
                    out["rule_count"][key] = n_rule
                    if n_rule == 0:
                        never = key
                    out["grid_rule_frac"][key] = round(n_rule / max(1, n_att), 4)
                    print(src, opt, lr, eps, ks, hits / tot, "stopped_by_rule", round(n_rule / max(1, n_att), 4),
                          f"{time.perf_counter() - t_c:.0f}s", flush=True)
                    tmp = part + ".tmp"
                    json.dump(out, open(tmp, "w"))
                    os.replace(tmp, part)
        sc = {(lr, eps, ks): out["grid"][f"{src}|{opt}|{lr}|{eps}|{ks}"] for lr in grid for eps in egrid for ks in kgrid}
        (blr, beps, bks), (e_lr, e_eps, e_ks) = pick_interior(sc, [grid, egrid, kgrid],
                                                              natural=[(), (1.0,), (1,)])
        out[f"lr_{src}_{opt}"] = blr
        out[f"eps_{src}_{opt}"] = beps
        out[f"kstop_{src}_{opt}"] = bks
        out[f"edge_{src}_{opt}"] = e_lr
        out[f"edge_eps_{src}_{opt}"] = e_eps
        out[f"edge_kstop_{src}_{opt}"] = e_ks
        out[f"wall_s_{src}_{opt}"] = round(time.perf_counter() - t_c, 1)
    json.dump(out, open(f"{a.out}/tune_s{ck['seed']}.json", "w"), indent=1)
    return out


def pick_interior(sc, grids, natural=None):
    """sc {(v_axis0, v_axis1, ...): score}, grids [values per axis], natural [per axis: end values that are natural
    limits, not edges] -> (best point, per-axis edge flags). Ties: points interior on more axes first, then axis 0
    smallest (lr), the other axes closest to their grid middle (log scale). Edge flag per axis: the best score is
    reached only at a non-natural end of that axis (no top-tied point has an interior or natural value there)."""
    natural = natural or [()] * len(grids)
    best = max(sc.values())
    top = [k for k, v in sc.items() if v == best]
    ok = lambda v, ax: len(grids[ax]) < 2 or v not in (min(grids[ax]), max(grids[ax])) or v in natural[ax]  # noqa: E731
    mids = [sorted(g)[len(g) // 2] for g in grids]
    key = lambda k: (sum(not ok(v, ax) for ax, v in enumerate(k)), k[0],                           # noqa: E731
                     *[abs(math.log(k[ax] / mids[ax])) for ax in range(1, len(k))])
    bp = min(top, key=key)
    edges = tuple(len(grids[ax]) > 1 and not any(ok(k[ax], ax) for k in top) for ax in range(len(grids)))
    return bp, edges


# ---------------- smoke (CPU controls) ----------------

def smoke(a):
    dev = dev_of(a)
    rep = {}
    # S1 batched exact executor == gradconst.run (bitwise) on 60 problems, 40 hidden inputs each
    diffs = 0
    for S in (1, 2, 3):
        pbs = [G.Problem(S, 50_000 * S + i) for i in range(20)]
        ops = torch.tensor([[o for o, _, _ in pb.prog] for pb in pbs])
        I = torch.tensor([[i for _, i, _ in pb.prog] for pb in pbs])
        J = torch.tensor([[j for _, _, j in pb.prog] for pb in pbs])
        C = torch.stack([pb.consts for pb in pbs])
        yb = run_b(ops, I, J, C, torch.stack([pb.ha for pb in pbs]), torch.stack([pb.hb for pb in pbs]))
        diffs += int((yb != torch.stack([pb.hy for pb in pbs])).sum())
        # soft executor with hard one-hot logits == exact
        Lo = F.one_hot(ops, NOP).double() * 1e4
        Li = torch.where(F.one_hot(I, NSRC) > 0, 1e4, 0.0).double()
        Lj = torch.where(F.one_hot(J, NSRC) > 0, 1e4, 0.0).double()
        for k in range(S):
            Li[:, k, 3 + k:] = NEG
            Lj[:, k, 3 + k:] = NEG
        ys = soft_run(Lo, Li, Lj, C, torch.stack([pb.a for pb in pbs]), torch.stack([pb.b for pb in pbs]))
        rep[f"soft_onehot_maxrel_len{S}"] = float(max_rel_err(ys, torch.stack([pb.y for pb in pbs])).max())
    rep["exact_executor_bit_diffs"] = diffs
    # S2 generator: every row passes gradconst's own filters; held-out skeletons absent; op histogram
    g = torch.Generator().manual_seed(5)
    bad, held = 0, 0
    for S in (1, 2, 3):
        d = gen_problems(S, 300, g, dev, "train")
        for t in range(300):
            prog = [(int(d["ops"][t, k]), int(d["I"][t, k]), int(d["J"][t, k])) for k in range(S)]
            y = G.run(prog, d["C"][t], d["a"][t], d["b"][t])
            bad += (not G.uses_all(prog)) or bool((y - d["y"][t]).abs().max() > 0)
            held += heldout(prog)
    rep["gen_rows_failing_gradconst_rules"] = bad
    rep["gen_heldout_skeletons_in_train"] = held
    rep["heldout_frac_len3_problems"] = round(statistics.mean(heldout(G.Problem(3, 90_000 * 3 + i).prog)
                                                              for i in range(100)), 3)
    # S3 batched LM == gradconst.fit_derive (stall ON) on the TRUE programs from 1.0, dev seeds
    agree, apps_eq, n_ = 0, 0, 0
    for S in (1, 2, 3):
        pbs = [G.Problem(S, 50_000 * S + i) for i in range(a.n_ctrl)]
        ops = torch.tensor([[o for o, _, _ in pb.prog] for pb in pbs])
        I = torch.tensor([[i for _, i, _ in pb.prog] for pb in pbs])
        J = torch.tensor([[j for _, _, j in pb.prog] for pb in pbs])
        cb, ub = lm_batch(ops, I, J, torch.ones(len(pbs), S, dtype=torch.float64), torch.stack([pb.a for pb in pbs]),
                          torch.stack([pb.b for pb in pbs]), torch.stack([pb.y for pb in pbs]),
                          torch.full((len(pbs),), CAP_LM))
        for q, pb in enumerate(pbs):
            cr, ur = G.fit_derive(pb, pb.prog, CAP_LM)
            agree += pb.fits(pb.prog, cr) == pb.fits(pb.prog, cb[q])
            apps_eq += ur == int(ub[q])
            n_ += 1
    rep["lm_batch_vs_fit_derive_fit_agree"] = f"{agree}/{n_}"
    rep["lm_batch_vs_fit_derive_apps_equal"] = f"{apps_eq}/{n_}"
    # S4 ES copy with c0=None reproduces gradconst.fit_es31 exactly
    eq = 0
    for i in range(a.n_ctrl):
        pb = G.Problem(2, 50_000 * 2 + i)
        c1, u1 = G.fit_es31(pb, pb.prog, CAP_ES, torch.Generator().manual_seed(i))
        c2, u2 = fit_es_c0(pb, pb.prog, CAP_ES, torch.Generator().manual_seed(i))
        eq += u1 == u2 and bool((c1 == c2).all())
    rep["es_copy_identical"] = f"{eq}/{a.n_ctrl}"
    # S5 bwrap judge: truth passes, constant +1% fails
    pbs = [G.Problem(S, 10_000 * S + i) for S in (1, 2, 3) for i in range(10)]
    rep["bwrap_truth_pass"] = sum(judge_bwrap(pbs, [(pb.prog, pb.consts) for pb in pbs]))
    rep["bwrap_perturbed_pass"] = sum(judge_bwrap(pbs, [(pb.prog, pb.consts * 1.01) for pb in pbs]))
    rep["bwrap_vs_inproc_truth"] = sum(G.io_ok(pb, pb.prog, pb.consts) for pb in pbs)
    # C1 positive control through the eval path: o_sample_lm == recorded v3.3 rows (len1 and len3)
    ck = {"prior": prior_logits(torch.Generator().manual_seed(9), dev, 2000), "seed": -1}
    net = Guesser(64).to(dev).eval()                                   # untrained: only the o_* arm is checked here
    for S in (1, 3):
        res = evaluate(a, net, ck, dev, "v33", S, a.n_ctrl, {}, ["o_sample_lm"])
        rep[f"C1_len{S}"] = control_v33(res, S)
    # S6 batched derive runs end to end (oracle guess) and its timing
    pbs, sd = problems("v33", 3, a.n_ctrl)
    olg = [G.guess_logits(pb, 1.5, torch.Generator().manual_seed(s)) for pb, s in zip(pbs, sd)]
    out, st = derive_batch(pbs, *pad_tuples(olg), torch.ones(len(pbs), 3, dtype=torch.float64), dev, budget=a.budget)
    rep["S6_o_derive_len3"] = {"fit": sum(o["fit"] for o in out), "n": len(out),
                               "io": sum(G.io_ok(pb, o["prog"], o["c"]) if o["fit"] else False for pb, o in zip(pbs, out)),
                               "restarts_mean": statistics.mean(o["restarts"] + 1 for o in out), **st}
    # ---- review fixes (10-27) ----
    # F1 Tcal masks: random-start derive from Tcal-scaled logits must not crash nor pick an out-of-range source
    f1 = {}
    for Tc in (1.0, 1.5, 3.0, 4.0):
        pbs3, sd3 = problems("dev", 3, 8)
        Lo, Li, Lj = pad_tuples([G.guess_logits(pb, 1.5, torch.Generator().manual_seed(x)) for pb, x in zip(pbs3, sd3)])
        bad = ~arg_mask(3)
        Lo, Li, Lj = Lo / Tc, (Li / Tc).masked_fill(bad, NEG), (Lj / Tc).masked_fill(bad, NEG)
        try:
            o3, _ = derive_batch(pbs3, Lo, Li, Lj, torch.ones(8, 3, dtype=torch.float64), dev, random_start=True,
                                 budget=1500)
            f1[str(Tc)] = {"crash": False, "out_of_range_progs": sum(
                any(i >= 3 + k or j >= 3 + k for k, (_, i, j) in enumerate(o["prog"])) for o in o3 if o["prog"])}
        except Exception as e:                                                       # noqa: BLE001
            f1[str(Tc)] = {"crash": True, "err": f"{type(e).__name__}: {str(e)[:120]}"}
    rep["F1_tcal_mask"] = f1
    # F2 split: canonical-family identities, dev is train-side only, held-out share of test problems
    A_, SU_, MU_, DI_, AV_, PC_ = (OPS.index(o) for o in ("add", "sub", "mul", "div", "avg", "pct"))
    rep["F2_canon_identities"] = {
        "sub(a,c)=add(a,c)": family_key([(SU_, 0, 2)]) == family_key([(A_, 0, 2)]),
        "mul=div=pct(a,c)": len({family_key([(o, 0, 2)]) for o in (MU_, DI_, PC_)}) == 1,
        "assoc add chain = len1 family": family_key([(A_, 0, 2), (A_, 2, 3)]) in E.shorter_keys(2),
        "sign F(b-a)=F(a-b)": family_key([(SU_, 1, 0), (PC_, 3, 2)]) == family_key([(SU_, 0, 1), (MU_, 2, 3)]),
        "avg(a,c) in product": family_key([(SU_, 0, 2), (PC_, 0, 2), (PC_, 3, 4)]) ==
                                family_key([(AV_, 0, 2), (PC_, 0, 2), (DI_, 3, 4)]),
        "add(a,c) != mul(a,c)": family_key([(A_, 0, 2)]) != family_key([(MU_, 0, 2)])}
    rep["F2_dev_heldout_rows"] = sum(heldout(pb.prog) for S in (2, 3) for pb in problems("dev", S, 60)[0])
    rep["F2_heldout_frac_problems"] = {S: round(statistics.mean(heldout(G.Problem(S, 90_000 * S + i).prog)
                                                                for i in range(300)), 3) for S in (2, 3)}
    # F2b the KILL top-1 accepts a functionally equivalent skeleton (sub(a,c) for add(a,c')), rejects a wrong one
    pb1 = G.Problem(1, 10_000 + 3)
    rep["F2_func_equiv_ctrl"] = E.func_equiv([(((SU_, 0, 2),), ((A_, 0, 2),), torch.tensor([3.7], dtype=torch.float64)),
                                              (((MU_, 0, 2),), ((A_, 0, 2),), torch.tensor([3.7], dtype=torch.float64)),
                                              (tuple(pb1.prog), tuple(pb1.prog), pb1.consts)])
    # F4 new arms run end to end: g_polish (no soft phase) and r_derive_gd, budget respected
    o_p, _ = derive_batch(pbs, *pad_tuples(olg), torch.ones(len(pbs), 3, dtype=torch.float64), dev, budget=a.budget,
                          no_soft=True)
    o_r, _ = derive_batch(pbs, *pad_tuples(olg), torch.ones(len(pbs), 3, dtype=torch.float64), dev, budget=a.budget,
                          opt="gd", random_start=True)
    rep["F4_g_polish_oracle_len3"] = {"fit": sum(o["fit"] for o in o_p), "n": len(o_p),
                                      "max_apps": max(o["apps"] for o in o_p),
                                      "soft_steps_total": sum(sum(o["soft_steps"]) for o in o_p),
                                      "attempts_mean": statistics.mean(len(o["changed"]) for o in o_p)}
    rep["F4_r_derive_gd_len3"] = {"fit": sum(o["fit"] for o in o_r), "n": len(o_r),
                                  "max_apps": max(o["apps"] for o in o_r)}
    gl = [[0.1, 0.3, 1.0, 3.0], [1e-3, 1e-2, 0.1, 1.0], [1, 2, 5, 10]]
    nat = [(), (1.0,), (1,)]
    pi = {}
    for name, peak in (("flat", None), ("eps_1.0_natural", (0.3, 1.0, 2)), ("eps_low_edge", (0.3, 1e-3, 2)),
                       ("kstop_1_natural", (0.3, 0.1, 1)), ("kstop_10_edge", (0.3, 0.1, 10)),
                       ("lr_top_edge", (3.0, 0.1, 2)), ("tie_edge_and_interior", (0.1, 0.1, 2))):
        sc = {(lr, e, k): 0.5 for lr in gl[0] for e in gl[1] for k in gl[2]}
        if peak:
            sc[peak] = 0.9
            if name == "tie_edge_and_interior":
                sc[(0.3, 0.1, 2)] = 0.9
        pi[name] = pick_interior(sc, gl, nat)
    rep["F4_pick_interior_lr_eps_kstop"] = pi
    # review B fix 2: real charge (3 NOP / soft step, PRIMARY) vs old charge (3 NOP S); same oracle starts, len3
    ch = {}
    for chg in ("real", "old"):
        o_c, _ = derive_batch(pbs, *pad_tuples(olg), torch.ones(len(pbs), 3, dtype=torch.float64), dev,
                              budget=a.budget, charge=chg)
        ch[chg] = {"fit": sum(o["fit"] for o in o_c), "n": len(o_c), "max_apps": max(o["apps"] for o in o_c),
                   "soft_steps_total": sum(sum(o["soft_steps"]) for o in o_c),
                   "apps_old_eq_apps": all(o["apps_old"] == o["apps"] for o in o_c) if chg == "old" else
                   all(o["apps_old"] == o["apps"] + (3 * NOP * 3 - 3 * NOP) * sum(o["soft_steps"]) for o in o_c),
                   "fit_within_old_budget": sum(o["fit"] and o["apps_old"] <= a.budget for o in o_c)}
    rep["B2_charge_o_derive_len3"] = ch
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(f"{a.out}/smoke.json", "w"), indent=1)
    return rep


# ---------------- GPU smoke (review fix 3): device rows must equal CPU rows ----------------

def gpucheck(a):
    """n_ctrl len3 DEV problems (review B fix 4: no test problem is touched): lm_batch (true and random wrong programs,
    float64 linalg.solve + torch.func.jvp) and derive_batch (oracle Adam, random GD, g_polish, oracle Adam with the
    old charge) on cuda vs cpu. Pass: per check <= 1 row whose (fit, apps) differs AND the same total fit."""
    dry = a.gpucheck_dry                           # CPU-only plumbing test of this function: cpu vs cpu, never a pass
    rep = {"cuda": torch.cuda.is_available(), "dry_cpu_vs_cpu": dry}
    if not rep["cuda"] and not dry:
        print(json.dumps(rep), flush=True)
        return 1
    if not dry:
        torch.cuda.set_per_process_memory_fraction(0.3)
        rep["device"] = torch.cuda.get_device_name(0)
    pbs, sd = problems("dev", 3, a.n_ctrl)
    rnd = random.Random(1)
    wrong = [[(rnd.randrange(NOP), rnd.randrange(3 + k), rnd.randrange(3 + k)) for k in range(3)] for _ in pbs]
    olg = [G.guess_logits(pb, 1.5, torch.Generator().manual_seed(x)) for pb, x in zip(pbs, sd)]
    res = {}
    for name, dv in (("cpu", torch.device("cpu")), ("cuda", torch.device("cpu" if dry else "cuda"))):
        r = {}
        for nm, progs in (("lm_true", [pb.prog for pb in pbs]), ("lm_wrong", wrong)):
            T = lambda f: torch.tensor([[f(st) for st in p] for p in progs]).to(dv)                # noqa: E731
            st = lambda f: torch.stack([f(pb) for pb in pbs]).to(dv)                             # noqa: E731
            cb, ub = lm_batch(T(lambda x: x[0]), T(lambda x: x[1]), T(lambda x: x[2]),
                              torch.ones(len(pbs), 3, dtype=torch.float64, device=dv), st(lambda p: p.a),
                              st(lambda p: p.b), st(lambda p: p.y), torch.full((len(pbs),), CAP_LM, device=dv))
            r[nm] = [(pb.fits(pg, cb[q].cpu()), int(ub[q])) for q, (pb, pg) in enumerate(zip(pbs, progs))]
        ones = torch.ones(len(pbs), 3, dtype=torch.float64)
        for nm, kw in (("derive_oracle_adam", dict(opt="adam", lr=0.01)), ("derive_random_gd", dict(opt="gd",
                       random_start=True)), ("polish", dict(no_soft=True)),
                       ("derive_oracle_adam_oldc", dict(opt="adam", lr=0.01, charge="old"))):
            out, stt = derive_batch(pbs, *pad_tuples(olg), ones, dv, budget=a.budget, **kw)
            r[nm] = [(o["fit"], o["apps"]) for o in out]
            r[nm + "_us_per_problem_step"] = stt["us_per_problem_step"]
        res[name] = r
    mism = {k: sum(x != y for x, y in zip(res["cpu"][k], res["cuda"][k])) for k in res["cpu"]
            if not k.endswith("_step")}
    rep.update({"mismatch_rows": mism, "fit_cpu": {k: sum(f for f, _ in v) for k, v in res["cpu"].items()
                                                   if not k.endswith("_step")},
                "us_per_problem_step": {d: {k: v for k, v in res[d].items() if k.endswith("_step")} for d in res}})
    # review B2 fix 1: wall time of REAL tune cells (n_cell dev problems x lens 1-3, the tune code path) on the GPU;
    # run_gpu24.sh sizes the tune / eval timeouts from the slower one (random start = rows use the whole budget)
    dvg = torch.device("cpu" if dry else "cuda")
    cell = {}
    for nm, src, opt in (("oracle_adam", "oracle", "adam"), ("random_gd", "random", "gd")):
        t1 = time.perf_counter()
        for S in (1, 2, 3):
            pc, sc_ = problems("dev", S, a.n_cell)
            if src == "oracle":
                Lo_, Li_, Lj_ = pad_tuples([G.guess_logits(pb, 1.5, torch.Generator().manual_seed(x))
                                            for pb, x in zip(pc, sc_)])
            else:
                Lo_ = torch.zeros(len(pc), S, NOP, dtype=torch.float64)
                Li_ = torch.zeros(len(pc), S, NSRC, dtype=torch.float64).masked_fill(~arg_mask(S), NEG)
                Lj_ = Li_.clone()
            res_, _ = derive_batch(pc, Lo_, Li_, Lj_, torch.ones(len(pc), S, dtype=torch.float64), dvg, opt=opt,
                                   lr=0.1, eps=1e-3, k_stop=5, budget=a.budget, random_start=src == "random",
                                   steps_cap=a.steps_cap, seed=777 + S)
            sum(G.io_ok(pb, r["prog"], r["c"]) if r["fit"] else False for pb, r in zip(pc, res_))
        cell[nm] = round(time.perf_counter() - t1, 2)
    rep["tune_cell_s"] = cell
    rep["tune_cell_s_max"] = max(cell.values())
    rep["n_cell"] = a.n_cell
    fit_eq = {k: sum(f for f, _ in res["cpu"][k]) == sum(f for f, _ in res["cuda"][k]) for k in mism}
    rep["fit_total_equal"] = fit_eq
    rep["ok"] = all(v <= 1 for v in mism.values()) and all(fit_eq.values())
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(f"{a.out}/gpucheck{'_dry' if dry else ''}.json", "w"), indent=1)
    return 2 if dry else (0 if rep["ok"] else 1)


# ---------------- held-out leak measurement (review fix 2) ----------------

def equiv_cmd(a):
    """per held-out test problem (len2 / len3, the eval ids): does a TRAINED skeleton compute the same function?
    Candidates: same-length neighbourhood (equiv24.neighbourhood) + every len1 skeleton (always trained) + for len3
    targets one skeleton per TRAINED len2 family (review B fix 3: cross-length leak), trained =
    uses_all and not heldout; equal = constants fitted on fresh inputs, max rel err < 1e-9 on 96 fresh inputs.
    Calibration: on the first n_cal len2 problems also the EXHAUSTIVE len2 candidate set (how much the
    neighbourhood misses). The verdict reports the no-equivalent subset as primary."""
    md5 = hashlib.md5(open(E.__file__, "rb").read()).hexdigest()
    lens = [int(x) for x in a.equiv_lens.split(",")]
    fn = f"{a.out}/heldout_equiv.json"
    rep = {"equiv24_md5": md5, "n": a.n, "tol": 1e-9}
    if lens != [2, 3]:                              # partial re-run: keep the other length and the calibration
        old = json.load(open(fn))
        if old["equiv24_md5"] != md5 or old["n"] != a.n:
            print("heldout_equiv.json was made by another equiv24.py or n - re-run all lengths", flush=True)
            return 1
        rep = old
    len1 = [p for p in E.all_skeletons(1)]
    len2_fam, seen2 = [], set()                     # review B fix 3: one skeleton per trained len2 family
    for p in E.all_skeletons(2):
        if not heldout(p):
            k_ = E.canon_key(p)
            if k_ not in seen2:
                seen2.add(k_)
                len2_fam.append(p)
    rep["n_trained_len2_families"] = len(len2_fam)
    t0 = time.time()
    for S in lens:
        pbs, ids = problems("heldout", S, a.n)
        flags, kinds = [], []
        for q, pb in enumerate(pbs):
            cands = [p for p in E.neighbourhood(tuple(pb.prog)) if not heldout(p)]
            f_same = E.func_equiv([(p, tuple(pb.prog), pb.consts) for p in cands])
            f_l1 = E.func_equiv([(p, tuple(pb.prog), pb.consts) for p in len1])
            f_l2 = E.func_equiv([(p, tuple(pb.prog), pb.consts) for p in len2_fam]) if S == 3 else []
            ex = [list(p) for p, v in zip(cands, f_same) if v][:3]
            ex2 = [list(p) for p, v in zip(len2_fam, f_l2) if v][:3]
            flags.append(any(f_same) or any(f_l1) or any(f_l2))
            kinds.append({"same_len": sum(f_same), "len1": sum(f_l1), "len2_family": sum(f_l2), "n_cand": len(cands),
                          "examples": ex, "examples_len2": ex2})
            if q % 20 == 0:
                print(f"len{S} {q}/{len(pbs)} equiv so far {sum(flags)} {time.time() - t0:.0f}s", flush=True)
        rep[f"len{S}"] = {"ids": ids, "has_trained_equiv": flags, "frac": round(statistics.mean(flags), 4),
                          "detail": kinds}
        print(f"len{S} frac with a trained equivalent: {rep[f'len{S}']['frac']}", flush=True)
        if S == 3:
            rep["len3"]["only_cross_len2"] = sum(f and not (k["same_len"] or k["len1"]) for f, k in zip(flags, kinds))
        json.dump(rep, open(fn, "w"), indent=1)
    if 2 not in lens:
        rep["wall_s_len3_rerun"] = round(time.time() - t0, 1)
        json.dump(rep, open(fn, "w"), indent=1)
        return 0
    pbs, _ = problems("heldout", 2, a.n_cal)
    seen, ex2 = set(), []
    for p in E.all_skeletons(2):
        r_ = canon(p)
        if r_ not in seen and not heldout(p):
            seen.add(r_)
            ex2.append(tuple(r_))
    ex_flags = [any(E.func_equiv([(p, tuple(pb.prog), pb.consts) for p in ex2])) for pb in pbs]
    neigh = rep["len2"]["has_trained_equiv"][:a.n_cal]
    rep["calib_len2_exhaustive"] = {"n": a.n_cal, "n_cand": len(ex2), "exhaustive": sum(ex_flags),
                                    "neighbourhood": sum(neigh),
                                    "missed_by_neighbourhood": sum(e and not n_ for e, n_ in zip(ex_flags, neigh))}
    rep["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("len")}), flush=True)
    json.dump(rep, open(f"{a.out}/heldout_equiv.json", "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "train", "tune", "eval", "gpucheck", "equiv"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ckpt")
    ap.add_argument("--tuned")
    ap.add_argument("--set", default="heldout", choices=["heldout", "v33", "dev"])
    ap.add_argument("--lens", default="1,2,3")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-ctrl", type=int, default=20)
    ap.add_argument("--n-cal", type=int, default=30, help="equiv: len2 problems also checked exhaustively")
    ap.add_argument("--n-cell", type=int, default=60, help="gpucheck: dev problems per length in the timed tune cell")
    ap.add_argument("--gpucheck-dry", action="store_true", help="gpucheck plumbing on CPU only (exit 2, never a pass)")
    ap.add_argument("--equiv-lens", default="2,3", help="equiv: '3' re-runs len3 only, keeps len2 + calibration")
    ap.add_argument("--budget", type=int, default=B_DEF)
    ap.add_argument("--steps-cap", type=int, default=100, help="max soft steps per attempt (gradconst soft_steps)")
    ap.add_argument("--lr-adam", default="0.003,0.03,0.3,3,30")
    ap.add_argument("--lr-gd", default="0.01,0.1,1,10")
    ap.add_argument("--tag", default="", help="output file suffix (e.g. _derive on GPU, _sample on CPU)")
    ap.add_argument("--allow-edge", action="store_true", help="smoke only: run with an edge-of-grid lr")
    ap.add_argument("--eps-grid", default="1e-5,1e-3,1e-2,0.1,1.0")
    ap.add_argument("--kstop-grid", default="1,3,10", help="HIS stop rule: flat steps in a row (review B fix 1)")
    ap.add_argument("--arms", default="g_only,g_derive,g_derive_gd,g_polish,g_sample_es,g_sample_lm,prior_sample_lm,"
                                      "o_derive,o_sample_lm,r_derive,r_derive_gd,g_derive_oldc,g_derive_gd_oldc,"
                                      "o_derive_oldc,r_derive_oldc,r_derive_gd_oldc")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd != "gpucheck" and dev_of(a).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.3)                    # review fix 3: VRAM cap on the cuda path
    if a.cmd == "gpucheck":
        sys.exit(gpucheck(a))
    if a.cmd == "equiv":
        sys.exit(equiv_cmd(a))
    if a.cmd == "smoke":
        smoke(a)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "tune":
        tune(a)
    else:
        dev = dev_of(a)
        net, ck = load_guesser(a.ckpt, torch.device("cpu"))       # guesses on CPU: identical in every process
        cfg = json.load(open(a.tuned)) if a.tuned else {}
        edges = [k for k, v in cfg.items() if k.startswith("edge_") and v]
        assert not edges or a.allow_edge, f"tuned lr at a grid edge {edges}: extend the grid (gate 8)"
        res = {"args": vars(a), "cfg": cfg, "guesser_seed": ck["seed"], "lens": {}}
        for S in map(int, a.lens.split(",")):
            r = evaluate(a, net, ck, dev, a.set, S, a.n, cfg, a.arms.split(","))
            if a.set == "v33" and "o_sample_lm" in a.arms:
                r["control_C1"] = control_v33(r, S)
            res["lens"][f"len{S}"] = r
            print(f"len{S}", json.dumps({"summary": r["summary"], "stats": r["stats"],
                                         "C1": r.get("control_C1")}), flush=True)
            json.dump(res, open(f"{a.out}/eval_{a.set}_s{ck['seed']}{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
