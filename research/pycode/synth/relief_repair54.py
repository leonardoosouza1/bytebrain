"""Relief as a repair signal inside the generate -> check loop (ledger #54(B)). CPU only.

Written 2026-10-27 12:14 (from `date`), BEFORE any score of this file.

HIS IDEA (#54): the components must work as an ECOSYSTEM inside the pipeline loop, not as isolated tests. Relief
(#53) localizes defective lines in isolation (top-1: wrong_indent .58, foreign_token .82, delete_line .53, swap .53;
MIX = relief + 2 x gap pond adds +1.7 pp). Here relief / MIX are put inside generate -> check -> repair and we
measure what they buy END TO END.

TESTBED (choice justified by recorded counts, read before writing this header):
  multifn belt (multifn_s{0,1,2}.json): len4 pass1 .41/.38/.29 with plan_exact .25/.19/.16 and text_exact ~=
    plan_exact; len5..len8 and full ~0 with plan_exact 0. Its failures are PLAN-level (missing/extra defs): a
    line-local repair of one unit cannot fix them. Not used.
  writer7seed (planner -> byte writer -> exec grade; summary.json): d32 gold-plan writer fails test_num (.86/.58/.72
    over writer seeds = writer-level, held (op, digit) pairs); e2e len5 .88 / len6 .76 = plan-level (planner seeds
    0, 5, 6). d32_ter writer-alone fails everywhere (.49-.87) = writer-level. d16_ter never learned (0): skipped.
  -> writer7seed has BOTH line-local (writer) and plan-level failures; the census below counts them per program.

PIPELINE (unchanged imports): planner_s{0..6}.pt (saved by writer7seed) -> plan -> writer (retrained with
  bytewriter.train_writer, same seeds 0,1,2, arms d32 and d32_ter) writes each block (op, digit) -> source assembled
  exactly as writer7seed.score -> bwrap (synth2.run_many, the existing runner).
CHECK (acceptance) vs GRADE (gate 3):
  ACCEPT = non-strict output equality on 6 NEW acceptance inputs (random.Random(5400), same generator form as
    synth2.INPUTS, none equal to a grading input). This is the analog of a user's examples. Run through the same
    bwrap runner by swapping synth2.INPUTS for the duration of the call.
  GRADE  = the recorded strict grade: 40 synth2.INPUTS, output AND pre-final list state equal (synth2.grade).
  Acceptance never sees grading inputs; grade never chooses a repair. Disagreement is counted per arm.
LOOP (pre-registered): repair runs ONLY when acceptance fails (acc < 1). Budget R = 8 candidate programs per task
  (= 8 extra acceptance runs) for every arm. Sequential arms: pick a unit, resample it from the writer at temperature
  tau = 1.0 (rest kept), accept the candidate if acc(candidate) >= acc(current), stop when acc == 1. Units already
  tried on the current version are skipped until the version changes (then the ranking restarts).
  The line ranking maps line -> unit (a block's lines); empty units attach to the next line (a gap) and are
  resampled together with it. The head line is never a target.
ARMS (equal budget R):
  none     greedy, no repair (= recorded pipeline).
  random   uniform random body line -> its unit.
  relief   top line by relief (relief53: z byte n-gram + token junction + line junction, row max) -> its unit.
  mix      relief + 2.0 x W2 gap pond (frozen.json: kr 3, kc 1, g 0.5, lam 2.0).
  mf       multi-frequency: mix + unit scale (mean S over the unit's cells, z-free sum). The function scale (mean S
           over all body cells) is constant within a program so it cannot rank lines; it is reported as a DETECTOR
           (AUROC greedy fail vs pass).
  conf     generator's own uncertainty: unit with the lowest min per-char log-prob under the writer (internal
           baseline: does an external frequency relief beat what the writer already knows?).
  oracle   unit whose text differs from the planned block's reference text (gold used for DIAGNOSIS only;
           ceiling for line-local repair; random unit if none differ).
  full     full resample: R whole-program samples (every unit at tau 1.0, plan fixed), keep the first with the
           highest acceptance score (greedy included as candidate 0).
  Scoring text for relief: the head row is replaced by `def f(xs, k):` / `def f(xs):` (len6 names exceed the 100-col
  grid; the head is never a target); rows truncated to 100 bytes and 40 rows (grid limits of relief53).
METRIC: end-to-end pass = strict grade == 1.0 on the 40 grading inputs, pooled over the 6 splits (2690 tasks),
  per (writer arm, writer seed, planner seed); mean over 7 planner x 3 writer seeds. Paired CIs: (a) t-interval
  over the 3 writer seeds of the planner-mean paired difference, (b) bootstrap over tasks (1000) of the pooled
  paired difference. Curves at r = 0, 1, 2, 4, 8.
  Identical plans across planner seeds give identical programs: arms run once per unique (writer, split, task,
  plan) and are mapped back (the pipeline output is a function of the plan; declared).

PREDICTIONS (written before any score):
  C0 positive control: arm `none` reproduces the recorded writer7seed e2e pass1 for EVERY (arm, ws, ps, split);
     pooled means: d32 ~.907, d32_ter ~.647.
  C1 census of greedy failures: d32: test_num failures ~100% writer-level (plan exact); len5/len6 ~100%
     plan-level; pooled plan-level share 55-75%. d32_ter: writer-level >= 80% of failures.
  P1 d32_ter at R=8 (pooled): full +15..+25 pp; oracle +15..+25; relief +8..+18; random +8..+18;
     relief - random in [-3, +4] pp (CI includes 0, p ~0.6): relief comes from general Python statistics and the
     writer's typical error (wrong digit / op in a well-formed line) is ordinary Python; garbled ternary bytes are
     the case relief can see.
  P2 mix - relief within +-2 pp; mf - relief within +-3 pp.
  P3 conf - relief >= +3 pp (the writer knows where it is unsure; p ~0.6).
  P4 full >= relief at equal R (p ~0.65): the confident units barely change at tau 1, so a whole-program resample
     behaves like resampling the uncertain unit and needs no localization.
  P5 d32 at R=8: test_num +5..+15 pp for every repair arm; len5/len6 < +3 pp (plan-level); pooled +1..+4 pp;
     arm differences < 2 pp.
  P6 acceptance/grade disagreement: false accept (acc 1, grade fail) <= 3% of accepted greedy programs; rises to
     2-6% after repair (search against the acceptance set overfits it); false reject ~0.
  P7 function-level relief as a detector: AUROC d32_ter 0.55-0.70, d32 ~0.5.

AMENDMENT 12:20 (after the 2-config smoke R=2 on d32_ter ws0, ps 4/6, test_num/len6; BEFORE the full run;
  predictions unchanged):
  (1) OWN BUG fixed: the sequential rule "accept if acc' >= acc" let a sideways candidate REPLACE a correct line when
      acc was already 0 (smoke trace: relief picked the correct `xs = [x % 3 ...]` line of a program broken by a
      syntax error elsewhere and accepted `xs = [x x in xs]`). Now strict: accept only if acc' > acc (same rule as
      `full`). Smoke numbers were produced with the buggy rule and are discarded (smoke_*.json overwritten).
  (2) Writer ceiling seen in the smoke: d32_ter ws0 at tau 1.0 samples the right text for take_3 in 3/20 draws,
      keep_lt_2 3/20, running_max / dedup_adjacent 0/20 (samples are noise). Primary stays tau 1.0 (registered);
      a SECONDARY run at tau 0.5 (--tau 0.5, out dir tau0.5/) is added, same arms/budget, declared as adaptation.
  Smoke positive control: none == recorded pass1 on all 4 (ps, split) cells (0.8086, 0.8086, 0.665, 0.0825).

ERRATUM 12:33 (after the v1 full run at tau 1.0, results kept in this directory as v1; v2 rerun in v2/):
  v1 oracle (gold ceiling) scored BELOW relief (d32_ter R=8 .6599 vs .6663). Own bug in the arm logic: after each bad
  unit was tried once, oracle fell back to random CORRECT units and never retried the bad ones; relief/mix/mf/conf,
  once every unit was tried, kept re-picking only their top-1 (no cycle). Fix (v2): oracle cycles over the bad
  units only; every ranked arm restarts its ranking (clears `tried`) once all units were tried. Added arm
  random_nr (random line among untried units, same no-repeat rule as the ranked arms; review point: v1 relief -
  random mixed localization with no-repeat). Paired CIs: v1 task bootstrap treated identical programs across
  planner seeds as independent (too narrow, review); v2 adds a cluster bootstrap over (writer seed, split, task).
  Predictions unchanged; v1 numbers stay reported next to v2.
  12:44 (second review, before v2 started): with 2+ bad units a single-unit fix rarely raises acc under the strict
  rule, so the v2 oracle resamples ALL bad units (vs the PLANNED block text) in one candidate = "plan-faithful
  oracle", a ceiling for writer-level failures only; v2 reports the fixed rate per census class (writer / plan /
  noplan) for every arm. Free control: v2 random@R and full@R must equal v1 (same seeds, unchanged path).

Usage (CPU only; one process; check /proc/loadavg < 12 first):
  cd /home/leonardo/projects/LLM/bytebrain/research/pycode/synth && HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 timeout 7200 /home/leonardo/projects/LLM/.venv-rocm/bin/python -u \
  relief_repair54.py --out /media/llm-data/universe-runs/bytebrain-pycode-20261022/relief_repair54 [--smoke]
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics as st
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))
import blockaddr as BA  # noqa: E402
import bytewriter as BW  # noqa: E402
import synth2  # noqa: E402
import writer7seed as W7  # noqa: E402
from bridge_cpu import Data, blocks_of  # noqa: E402
from synth2 import head, name_of, nargs  # noqa: E402
import relief53 as R53  # noqa: E402
import relief53_adapt as RA  # noqa: E402

RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
PLANNERS = f"{RUN}/writer7seed"
RELIEF = f"{RUN}/relief53"
SPLITS = W7.SPLITS
ARMS_W = {"d32": (32, False), "d32_ter": (32, True)}
REPAIR_ARMS = ["random", "random_nr", "relief", "mix", "mf", "conf", "oracle", "full"]
CURVE = (0, 1, 2, 4, 8)
TAU = 1.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ acceptance inputs (disjoint from grading)
def make_acc_inputs(n=6, seed=5400):
    r = random.Random(seed)
    grade_set = {json.dumps(x) for x in synth2.INPUTS}
    out = []
    while len(out) < n:
        x = [[r.randint(-9, 19) for _ in range(r.randint(0, 9))], r.randint(0, 6)]
        if json.dumps(x) not in grade_set and json.dumps(x) not in {json.dumps(y) for y in out}:
            out.append(x)
    return out


ACC_INPUTS = make_acc_inputs()


@contextlib.contextmanager
def inputs(xs):
    old = synth2.INPUTS
    synth2.INPUTS = xs
    try:
        yield
    finally:
        synth2.INPUTS = old


ACC_CACHE, REF_ACC, GRADE_CACHE = {}, {}, {}


def acc_scores(items):
    """items [(task, src)] -> fraction of acceptance inputs with output == reference (non-strict)."""
    new = [(t, s) for t, s in items if s not in ACC_CACHE]
    uniq = {}
    for t, s in new:
        uniq[s] = t
    if uniq:
        srcs = list(uniq)
        with inputs(ACC_INPUTS):
            outs = []
            for i in range(0, len(srcs), 256):
                outs += synth2.run_many([(s, name_of(*uniq[s]), nargs(uniq[s][0])) for s in srcs[i:i + 256]])
            for s in srcs:
                k = name_of(*uniq[s])
                if k not in REF_ACC:
                    REF_ACC[k] = synth2.reference(*uniq[s])
        for s, o in zip(srcs, outs):
            ref = REF_ACC[name_of(*uniq[s])]
            ACC_CACHE[s] = 0.0 if o is None else sum(a == b for a, b in zip(o, ref)) / len(ref)
    return [ACC_CACHE[s] for _, s in items]


def grade_srcs(items):
    """items [(task, src)] -> strict grade pass (bool) on the 40 grading inputs, cached by source."""
    uniq = {}
    for t, s in items:
        if s not in GRADE_CACHE:
            uniq[s] = t
    if uniq:
        srcs = list(uniq)
        sc = synth2.grade([uniq[s] for s in srcs], [[s] for s in srcs])
        for s, g in zip(srcs, sc):
            GRADE_CACHE[s] = g[0] == 1.0
    return [GRADE_CACHE[s] for _, s in items]


# ------------------------------------------------------------------ writer: greedy / sample / score
@torch.no_grad()
def write(writer, keys, tau=0.0, gen=None, max_len=160):
    """same loop as bytewriter.Writer.write; tau 0 = argmax (must equal writer.write)."""
    if not keys:
        return []
    op = torch.tensor([k[0] for k in keys])
    dg = torch.tensor([k[1] for k in keys])
    cond = writer.op(op) + writer.dig(dg)
    B = op.shape[0]
    h, prev = torch.zeros(B, writer.cell.hidden_size), torch.full((B,), len(BW.CH) + 1)
    out = []
    for _ in range(max_len):
        h = writer._cell(torch.cat([cond, writer.ch(prev)], 1), h)
        lo = writer.logits(h)
        if tau <= 0:
            prev = lo.argmax(-1)
        else:
            prev = torch.multinomial(F.softmax(lo / tau, -1), 1, generator=gen)[:, 0]
        out.append(prev)
    res = []
    for row in torch.stack(out, 1).tolist():
        s = row[:row.index(BW.END)] if BW.END in row else row
        res.append("".join(BW.CH[i] for i in s))
    return res


@torch.no_grad()
def min_logprob(writer, keys, texts):
    """teacher-forced min per-char log-prob (incl. END if the text ended) of each text under the writer."""
    if not keys:
        return []
    ys = [[BW.C2I[c] for c in t if c in BW.C2I] + [BW.END] for t in texts]
    T = max(map(len, ys))
    y = torch.tensor([v + [-100] * (T - len(v)) for v in ys])
    y_in = torch.cat([torch.full((len(ys), 1), len(BW.CH) + 1), y[:, :-1].clamp(min=0)], 1)
    lo = writer(torch.tensor([k[0] for k in keys]), torch.tensor([k[1] for k in keys]), y_in)
    lp = F.log_softmax(lo, -1).gather(2, y.clamp(min=0)[..., None])[..., 0]
    lp = torch.where(y >= 0, lp, torch.zeros_like(lp))
    return lp.min(1).values.tolist()


# ------------------------------------------------------------------ program assembly
def assemble(task, blocks, texts):
    """-> src (exactly writer7seed.score), unit_of_row {row: [unit indices]} for body rows."""
    if not blocks:
        return head(*task) + "    return None\n", {}
    body, rows, pending, r = [], {}, [], 1
    for i, w in enumerate(texts):
        ls = [l for l in w.split("\n") if l]
        if not ls:
            pending.append(i)
            continue
        for j, l in enumerate(ls):
            body.append(f"    {l}\n")
            rows[r] = (pending + [i]) if j == 0 else [i]
            pending = [] if j == 0 else pending
            r += 1
    if pending:
        if rows:
            rows[r - 1] = rows[r - 1] + pending
        else:
            rows[1] = pending        # whole body empty: one virtual row (score 0), resampling all units
    return head(*task) + "".join(body), rows


def scoring_text(task, src):
    rows = src.rstrip("\n").split("\n")
    rows[0] = "def f(xs, k):" if nargs(task[0]) == 2 else "def f(xs):"
    rows = [r.encode()[:R53.CMAX].decode("ascii", "ignore") for r in rows[:R53.RMAX]]
    if len(rows) < 2:
        rows.append("    pass")
    return "\n".join(rows)


# ------------------------------------------------------------------ relief scorers
class Relief:
    def __init__(self):
        self.bng = R53.ByteNgram(None, os.path.join(RELIEF, "byte_ngram.npz"))
        self.jn = R53.Junctions(None, os.path.join(RELIEF, "junctions.json"))
        self.jn.setup()
        self.stats = json.load(open(os.path.join(RELIEF, "results.json")))["config"]["calib_stats"]
        fr = json.load(open(os.path.join(RELIEF, "adapt", "frozen.json")))
        self.w2 = fr["W2"]
        self.lam = fr["MIX"]["lam"]
        assert fr["MIX"]["water"] == "W2", fr["MIX"]
        self.w2name = RA.name("W2", self.w2)
        self.G, self.S = {}, {}

    def scores(self, texts):
        """texts (scoring texts) -> {text: {relief, mix, unitS (row-> (sum S, n cells)), fun}}."""
        new = [t for t in dict.fromkeys(texts) if t not in self.S]
        for t in new:
            self.G[t] = RA.relief_grids(t, self.bng, self.jn, self.stats)
        if new:
            w2 = RA.w2_scores(self.G, new, torch.device("cpu"), [self.w2])
            for t in new:
                g = self.G[t]
                rel = R53.line_scores(g["S"], g["mask"], "max")
                mix = rel + self.lam * w2[t][self.w2name]
                Ssum = np.where(g["mask"], g["S"], 0.0).sum(1)
                ncell = g["mask"].sum(1)
                body = slice(1, None)
                fun = float(Ssum[body].sum() / max(ncell[body].sum(), 1))
                self.S[t] = {"relief": rel, "mix": mix, "Ssum": Ssum, "n": ncell, "fun": fun}
                del self.G[t]
        return [self.S[t] for t in texts]


# ------------------------------------------------------------------ repair loop over unique programs
def unit_choice(arm, prog, sc, tried, rng, writer_lp):
    """-> list of unit indices to resample (one candidate), or None."""
    rows = prog["rows"]
    if not rows:
        return None
    nunits = len(prog["texts"])
    if arm == "random":
        r = rng.choice(sorted(rows))
        return rows[r]
    if arm == "random_nr":                 # v2: random line among units not yet tried (same no-repeat as relief)
        cand = [r for r in sorted(rows) if not set(rows[r]) <= tried]
        if not cand:
            tried.clear()
            cand = sorted(rows)
        return rows[rng.choice(cand)]
    if arm == "oracle":                    # v2: resample ALL bad units jointly (plan-faithful oracle)
        allbad = [i for i in range(nunits) if prog["texts"][i] != prog["ref"][i]]
        if allbad:
            return allbad
        rest = [i for i in range(nunits) if i not in tried]
        if not rest:
            tried.clear()
            rest = list(range(nunits))
        return [rng.choice(rest)]
    if arm == "conf":
        order = sorted(range(nunits), key=lambda i: (writer_lp[i], rng.random()))
        for i in order:
            if i not in tried:
                return [i]
        tried.clear()                      # v2: all tried -> restart the ranking (cycle)
        return [order[0]]
    # line scorers
    nr = len(sc["relief"])
    def row_score(r):
        if r >= nr:
            return -1e9
        if arm == "relief":
            return sc["relief"][r]
        if arm == "mix":
            return sc["mix"][r]
        if arm == "mf":
            us = rows[r]
            cells = [rr for rr, u in rows.items() if rr < nr and set(u) & set(us)]
            unit = sum(sc["Ssum"][rr] for rr in cells) / max(sum(sc["n"][rr] for rr in cells), 1)
            return sc["mix"][r] + unit
        raise ValueError(arm)
    order = sorted(rows, key=lambda r: (-row_score(r), rng.random()))
    for r in order:
        if not set(rows[r]) <= tried:
            return rows[r]
    tried.clear()                          # v2: all tried -> restart the ranking (cycle)
    return rows[order[0]]


def run_arm(arm, progs, writer, relief, R, seed):
    """progs: list of dicts (task, blocks, keys, ref, texts, src, rows, acc). Returns per prog srcs at CURVE steps
    and the number of writer unit samples used."""
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    cur = [dict(p) for p in progs]
    for c in cur:
        c["tried"] = set()
        c["hist"] = {0: c["src"]}
        c["best_acc"] = c["acc"]
        c["samples"] = 0
        c["cands"] = 0
    for step in range(1, R + 1):
        act = [c for c in cur if c["best_acc"] < 1.0 and c["rows"]]
        if act:
            if arm == "full":
                keys = [k for c in act for k in c["keys"]]
                texts = write(writer, keys, TAU, gen)
                j, cands = 0, []
                for c in act:
                    n = len(c["keys"])
                    tx = texts[j:j + n]
                    j += n
                    c["samples"] += n
                    cands.append((c, tx) + assemble(c["task"], c["blocks"], tx))
            else:
                lp = {}
                if arm == "conf":
                    flat = [(id(c), i, c["keys"][i], c["texts"][i]) for c in act for i in range(len(c["keys"]))]
                    vals = min_logprob(writer, [f[2] for f in flat], [f[3] for f in flat])
                    for f, v in zip(flat, vals):
                        lp.setdefault(f[0], {})[f[1]] = v
                scs = relief.scores([scoring_text(c["task"], c["src"]) for c in act]) \
                    if arm in ("relief", "mix", "mf") else [None] * len(act)
                picks = []
                for c, sc in zip(act, scs):
                    u = unit_choice(arm, c, sc, c["tried"], rng, lp.get(id(c)))
                    picks.append(u)
                keys = [c["keys"][i] for c, u in zip(act, picks) for i in u]
                texts = write(writer, keys, TAU, gen)
                j, cands = 0, []
                for c, u in zip(act, picks):
                    tx = list(c["texts"])
                    for i in u:
                        tx[i] = texts[j]
                        j += 1
                    c["samples"] += len(u)
                    c["tried"] |= set(u)
                    cands.append((c, tx) + assemble(c["task"], c["blocks"], tx))
            accs = acc_scores([(c["task"], s) for c, _, s, _ in cands])
            for (c, tx, s, rows), a in zip(cands, accs):
                c["cands"] += 1
                take = a > c["best_acc"]            # strict for every arm (amendment 12:20)
                if take and s != c["src"]:
                    c.update(texts=tx, src=s, rows=rows, best_acc=a)
                    c["tried"] = set()
                elif take:
                    c["best_acc"] = a
        for c in cur:
            c["hist"][step] = c["src"]
    return cur


# ------------------------------------------------------------------ stats
def auroc(pos, neg):
    if not pos or not neg:
        return None
    a = np.array(pos)[:, None]
    b = np.array(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (len(pos) * len(neg)))


def t_ci(xs):
    if len(xs) < 2:
        return (round(xs[0], 4), None, None)
    m, s = st.mean(xs), st.stdev(xs)
    tq = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}.get(len(xs), 2.0)
    h = tq * s / math.sqrt(len(xs))
    return (round(m, 4), round(m - h, 4), round(m + h, 4))


def boot_ci(diff, n=1000, seed=0):
    d = np.asarray(diff, float)
    r = np.random.default_rng(seed)
    bs = [d[r.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return (round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="d32_ter,d32")
    ap.add_argument("--writer-seeds", default="0,1,2")
    ap.add_argument("--planner-seeds", default="0,1,2,3,4,5,6")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    a = ap.parse_args()
    global TAU
    TAU = a.tau
    torch.set_num_threads(a.threads)
    if a.smoke:
        a.arms, a.writer_seeds, a.planner_seeds, a.splits, a.R = "d32_ter", "0", "4,6", "test_num,len6", 2
    os.makedirs(a.out, exist_ok=True)
    tag = "smoke_" if a.smoke else ""
    T0 = time.perf_counter()
    wseeds = [int(s) for s in a.writer_seeds.split(",")]
    pseeds = [int(s) for s in a.planner_seeds.split(",")]
    splits = a.splits.split(",")
    R = a.R
    curve = [c for c in CURVE if c <= R]
    log(f"acceptance inputs: {ACC_INPUTS}")
    data = Data()
    sets = {k: data.sets[k] for k in splits}
    plans = {}
    for ps in pseeds:
        pl = BA.Net("addr", gate_bias=-4.0)
        pl.load_state_dict(torch.load(f"{PLANNERS}/planner_s{ps}.pt"))
        plans[ps] = {k: W7.planner_plans(pl, v) for k, v in sets.items()}
    log(f"planners loaded {pseeds} ({time.perf_counter() - T0:.0f}s)")
    relief = Relief()
    log("relief loaded")
    results = {"header_time": "2026-10-27 12:14", "amendment": "12:20 strict accept", "R": R, "tau": TAU, "acc_inputs": ACC_INPUTS, "rows": []}
    rows_out = results["rows"]
    task_rec = open(f"{a.out}/{tag}tasks.jsonl", "w")

    for arm_w in a.arms.split(","):
        d, ter = ARMS_W[arm_w]
        rec = json.load(open(f"{PLANNERS}/{arm_w}_w0.json")) if not a.smoke else None
        for ws in wseeds:
            t0 = time.perf_counter()
            writer, _, wloss = BW.train_writer(data.sets["train"], d, ter, ws)
            writer.eval()
            recorded = json.load(open(f"{PLANNERS}/{arm_w}_w{ws}.json"))
            log(f"{arm_w} ws{ws} writer trained loss {wloss} (recorded {recorded['writer_loss']}) "
                f"{time.perf_counter() - t0:.0f}s")
            # unique programs per split
            for k in splits:
                tasks = sets[k]
                uniq = {}                       # (task idx, plan symbols) -> prog
                where = {}                      # (ps, idx) -> uniq key
                for ps in pseeds:
                    for i, (t, p) in enumerate(zip(tasks, plans[ps][k])):
                        key = (i, None if p is None else tuple(blocks_of(*p)))
                        where[(ps, i)] = key
                        if key not in uniq:
                            bl = BW.blocks_of_task(p) if p is not None else []
                            uniq[key] = {"task": t, "blocks": bl, "keys": [BW.block_key(*b) for b in bl],
                                         "ref": [BW.block_text(*b) for b in bl],
                                         "plan_exact": p is not None and blocks_of(*p) == blocks_of(*t)}
                ul = list(uniq.values())
                allk = sorted({kk for u in ul for kk in u["keys"]})
                gt = dict(zip(allk, write(writer, allk)))
                if k == splits[0]:
                    ref_w = dict(zip(allk, writer.write(torch.tensor([x[0] for x in allk]),
                                                        torch.tensor([x[1] for x in allk]))))
                    assert all(gt[x] == ref_w[x] for x in allk), "greedy write != Writer.write"
                for u in ul:
                    u["texts"] = [gt[kk] for kk in u["keys"]]
                    u["src"], u["rows"] = assemble(u["task"], u["blocks"], u["texts"])
                    u["text_exact"] = bool(u["blocks"]) and u["texts"] == u["ref"]
                g0 = grade_srcs([(u["task"], u["src"]) for u in ul])
                a0 = acc_scores([(u["task"], u["src"]) for u in ul])
                for u, g, ac in zip(ul, g0, a0):
                    u["grade0"], u["acc"] = g, ac
                # positive control: greedy pass1 == recorded per planner seed
                ctrl = {}
                for ps in pseeds:
                    p1 = round(sum(uniq[where[(ps, i)]]["grade0"] for i in range(len(tasks))) / len(tasks), 4)
                    ctrl[ps] = (p1, recorded["e2e"][str(ps)][k]["pass1"])
                ok = all(x == y for x, y in ctrl.values())
                log(f"  {arm_w} ws{ws} {k}: {len(ul)} unique programs, greedy fail {sum(not u['grade0'] for u in ul)}, "
                    f"acc fail {sum(u['acc'] < 1 for u in ul)}; control {'OK' if ok else 'MISMATCH ' + str(ctrl)}")
                # function-level relief detector on greedy programs
                fsc = relief.scores([scoring_text(u["task"], u["src"]) for u in ul])
                for u, s in zip(ul, fsc):
                    u["fun"] = s["fun"]
                # repair arms on acceptance-failing programs
                fail = [u for u in ul if u["acc"] < 1.0]
                arm_res = {}
                for ra in REPAIR_ARMS:
                    t1 = time.perf_counter()
                    seed = int(hashlib.sha1(f"{arm_w}|{ws}|{k}|{ra}".encode()).hexdigest()[:8], 16)
                    arm_res[ra] = run_arm(ra, fail, writer, relief, R, seed)
                    log(f"    arm {ra}: {len(fail)} programs, {time.perf_counter() - t1:.0f}s")
                # grade every curve point
                fidx = {id(u): j for j, u in enumerate(fail)}
                need = [(u["task"], c["hist"][r]) for ra in REPAIR_ARMS for u, c in zip(fail, arm_res[ra])
                        for r in curve]
                grade_srcs(need)
                acc_scores(need)
                # expand to (ps, task)
                for ps in pseeds:
                    row = {"w_arm": arm_w, "ws": ws, "ps": ps, "split": k, "n": len(tasks),
                           "recorded_pass1": recorded["e2e"][str(ps)][k]["pass1"], "pass": {}, "acc_pass": {},
                           "false_accept": {}, "false_reject": {}, "samples": {}, "cands": {}}
                    per_task = {"none": [], **{ra: {r: [] for r in curve} for ra in REPAIR_ARMS}}
                    census = {"pass": 0, "fail_writer": 0, "fail_plan": 0, "fail_both": 0, "fail_noplan": 0,
                              "fail_plan_ok_text_ok": 0}
                    cls = []
                    for i in range(len(tasks)):
                        u = uniq[where[(ps, i)]]
                        per_task["none"].append((u["grade0"], u["acc"] == 1.0))
                        cls.append("pass" if u["grade0"] else "noplan" if not u["blocks"] else
                                   ("writer" if u["plan_exact"] else "plan"))
                        if u["grade0"]:
                            census["pass"] += 1
                        elif not u["blocks"]:
                            census["fail_noplan"] += 1
                        else:
                            te = u["texts"] == u["ref"]
                            if u["plan_exact"] and not te:
                                census["fail_writer"] += 1
                            elif not u["plan_exact"] and te:
                                census["fail_plan"] += 1
                            elif not u["plan_exact"]:
                                census["fail_both"] += 1
                            else:
                                census["fail_plan_ok_text_ok"] += 1
                        j = fidx.get(id(u))
                        for ra in REPAIR_ARMS:
                            for r in curve:
                                if j is None:
                                    per_task[ra][r].append((u["grade0"], u["acc"] == 1.0))
                                else:
                                    s = arm_res[ra][j]["hist"][r]
                                    per_task[ra][r].append((GRADE_CACHE[s], ACC_CACHE[s] == 1.0))
                    row["census"] = census
                    row["cls"] = cls
                    n = len(tasks)

                    def summ(lst):
                        g = [x for x, _ in lst]
                        acc = [y for _, y in lst]
                        return (round(sum(g) / n, 4), round(sum(acc) / n, 4),
                                sum(y and not x for x, y in lst), sum(x and not y for x, y in lst))
                    row["pass"]["none"], row["acc_pass"]["none"], row["false_accept"]["none"], \
                        row["false_reject"]["none"] = summ(per_task["none"])
                    for ra in REPAIR_ARMS:
                        for r in curve:
                            kk = f"{ra}@{r}"
                            row["pass"][kk], row["acc_pass"][kk], row["false_accept"][kk], \
                                row["false_reject"][kk] = summ(per_task[ra][r])
                        js = [fidx[id(uniq[where[(ps, i)]])] for i in range(n) if id(uniq[where[(ps, i)]]) in fidx]
                        row["samples"][ra] = sum(arm_res[ra][j]["samples"] for j in js)
                        row["cands"][ra] = sum(arm_res[ra][j]["cands"] for j in js)
                    row["task_grades"] = {"none": [int(x) for x, _ in per_task["none"]],
                                          **{ra: [int(x) for x, _ in per_task[ra][R]] for ra in REPAIR_ARMS}}
                    row["fun_relief"] = {"fail": [uniq[where[(ps, i)]]["fun"] for i in range(n)
                                                  if not uniq[where[(ps, i)]]["grade0"]],
                                         "pass": [uniq[where[(ps, i)]]["fun"] for i in range(n)
                                                  if uniq[where[(ps, i)]]["grade0"]]}
                    rows_out.append(row)
                    log(f"    ps{ps} {k}: none {row['pass']['none']} (rec {row['recorded_pass1']}) | " +
                        " ".join(f"{ra} {row['pass'][f'{ra}@{R}']}" for ra in REPAIR_ARMS))
                # a few example traces
                for u, ci in list(zip(fail, arm_res["relief"]))[:3]:
                    task_rec.write(json.dumps({"w_arm": arm_w, "ws": ws, "split": k, "greedy": u["src"],
                                               "ref": u["ref"], "texts": u["texts"], "relief_final": ci["src"],
                                               "full_final": arm_res["full"][fail.index(u)]["src"]}) + "\n")
                task_rec.flush()
            json.dump(results, open(f"{a.out}/{tag}rows.json", "w"))

    # ------------------------------------------------------------------ summary
    summ = {"R": R, "curve": curve, "wall_s": round(time.perf_counter() - T0, 1),
            "graded_sources": len(GRADE_CACHE), "acc_sources": len(ACC_CACHE)}
    arms_all = ["none"] + [f"{ra}@{r}" for ra in REPAIR_ARMS for r in curve]
    for arm_w in a.arms.split(","):
        rs = [r for r in rows_out if r["w_arm"] == arm_w]
        if not rs:
            continue
        S = {}
        # pooled over splits per (ws, ps)
        def pooled(ws, ps, key, fld="pass"):
            rr = [r for r in rs if r["ws"] == ws and r["ps"] == ps]
            return sum(r[fld][key] * r["n"] for r in rr) / sum(r["n"] for r in rr)
        S["control_all_match"] = all(r["pass"]["none"] == r["recorded_pass1"] for r in rs)
        S["control_mismatches"] = [(r["ws"], r["ps"], r["split"], r["pass"]["none"], r["recorded_pass1"])
                                   for r in rs if r["pass"]["none"] != r["recorded_pass1"]]
        S["pooled_pass_mean"] = {kk: round(st.mean(pooled(ws, ps, kk) for ws in wseeds for ps in pseeds), 4)
                                 for kk in arms_all}
        S["pooled_pass_by_ws"] = {kk: {ws: round(st.mean(pooled(ws, ps, kk) for ps in pseeds), 4) for ws in wseeds}
                                  for kk in arms_all}
        S["pooled_pass_by_ws_ps"] = {f"w{ws}p{ps}": {kk: round(pooled(ws, ps, kk), 4) for kk in
                                                    ["none"] + [f"{ra}@{R}" for ra in REPAIR_ARMS]}
                                     for ws in wseeds for ps in pseeds}
        S["split_pass_mean"] = {k: {kk: round(st.mean(r["pass"][kk] for r in rs if r["split"] == k), 4)
                                    for kk in ["none"] + [f"{ra}@{R}" for ra in REPAIR_ARMS]} for k in splits}
        cen = {}
        for r in rs:
            for kk, v in r["census"].items():
                cen.setdefault(r["split"], {}).setdefault(kk, 0)
                cen[r["split"]][kk] += v
        S["census_by_split_sum_over_configs"] = cen
        tot = {kk: sum(c[kk] for c in cen.values()) for kk in next(iter(cen.values()))}
        S["census_total"] = tot
        # paired differences
        pairs = [("relief", "random_nr"), ("mix", "random_nr"), ("random_nr", "random"), ("oracle", "full"),
                 ("relief", "random"), ("relief", "full"), ("mix", "relief"), ("mf", "relief"), ("conf", "relief"),
                 ("oracle", "relief"), ("full", "random"), ("mix", "random"), ("relief", "none"), ("random", "none"),
                 ("full", "none"), ("mix", "full"), ("conf", "full")]
        S["paired"] = {}
        for x, y in pairs:
            kx = "none" if x == "none" else f"{x}@{R}"
            ky = "none" if y == "none" else f"{y}@{R}"
            per_ws = [st.mean(pooled(ws, ps, kx) - pooled(ws, ps, ky) for ps in pseeds) for ws in wseeds]
            diff, clus = [], {}
            for r in rs:
                for i, (p, q) in enumerate(zip(r["task_grades"][x], r["task_grades"][y])):
                    diff.append(p - q)
                    clus.setdefault((r["ws"], r["split"], i), []).append(p - q)
            cl = list(clus.values())
            rg = np.random.default_rng(0)
            sums = np.array([sum(c) for c in cl], float)
            cnts = np.array([len(c) for c in cl], float)
            bs = []
            for _ in range(1000):
                ix = rg.integers(0, len(cl), len(cl))
                bs.append(sums[ix].sum() / cnts[ix].sum())
            S["paired"][f"{x}-{y}"] = {"t_over_writer_seeds": t_ci(per_ws), "per_ws": [round(v, 4) for v in per_ws],
                                       "boot_tasks_naive": boot_ci(diff),
                                       "boot_cluster_ws_split_task": (round(float(sums.sum() / cnts.sum()), 4),
                                                                      round(float(np.percentile(bs, 2.5)), 4),
                                                                      round(float(np.percentile(bs, 97.5)), 4))}
        # acceptance / grade disagreement
        dis = {}
        for kk in ["none"] + [f"{ra}@{R}" for ra in REPAIR_ARMS]:
            fa = sum(r["false_accept"][kk] for r in rs)
            fr = sum(r["false_reject"][kk] for r in rs)
            nacc = sum(round(r["acc_pass"][kk] * r["n"]) for r in rs)
            ntot = sum(r["n"] for r in rs)
            dis[kk] = {"false_accept": fa, "false_reject": fr, "accepted": nacc, "n": ntot,
                       "false_accept_of_accepted": round(fa / max(nacc, 1), 4),
                       "disagree_rate": round((fa + fr) / ntot, 4)}
        S["acc_grade_disagreement"] = dis
        S["fixed_by_class"] = {}
        for ra in REPAIR_ARMS:
            d_ = {}
            for r in rs:
                for c_, g in zip(r["cls"], r["task_grades"][ra]):
                    if c_ != "pass":
                        d_.setdefault(c_, [0, 0])
                        d_[c_][0] += g
                        d_[c_][1] += 1
            S["fixed_by_class"][ra] = {c_: (v[0], v[1], round(v[0] / v[1], 4)) for c_, v in d_.items()}
        S["samples_total"] = {ra: sum(r["samples"][ra] for r in rs) for ra in REPAIR_ARMS}
        S["cands_total"] = {ra: sum(r["cands"][ra] for r in rs) for ra in REPAIR_ARMS}
        S["fun_relief_auroc_fail_vs_pass"] = auroc([v for r in rs for v in r["fun_relief"]["fail"]],
                                                   [v for r in rs for v in r["fun_relief"]["pass"]])
        summ[arm_w] = S
    json.dump(summ, open(f"{a.out}/{tag}summary.json", "w"), indent=1)
    log(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()
                                                           if kk in ("control_all_match", "pooled_pass_mean",
                                                                     "census_total", "paired",
                                                                     "fun_relief_auroc_fail_vs_pass")})
                    for k, v in summ.items()}))


if __name__ == "__main__":
    main()
