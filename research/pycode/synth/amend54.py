"""Ledger #54(A), world `amend` for liquid13.py (Amendment E, registered 2026-10-27; predictions in liquid13.py header).

His words (#54A): "banco de T... T tao grande que o dado fica la e so muda se houver um 'estresse'... o caso da def de
soma que so trocou um sinal". A held fact must be REWRITTEN mid-sequence when a correction arrives, and then held for a
long gap before it is used.

Input (bytes a-z 0-9 _, items joined by "_"; one decoder step reads one item through blockaddr's address table):
  f<v>   fact, always item 0: the operator the function is about (v in VALS, e.g. "fadd" = "def sum")
  x<v>   AMEND: the operator is corrected to v (v != every earlier value; with p .5 it is the sign partner of the
         current one: add<->sub, mul<->div, min<->max, and<->or = "only one sign changed")
  n<d>   filler: output d:<d> (a copy: keeps the address read sharp, so the read cannot blur back to the fact)
  ret    USE: output op:<current value> (the value from the most recent f/x item)
Output per item: f/x -> op:<v> (echo), n<d> -> d:<d>, ret -> op:<current>; then <eos>.
Example (train-like): fadd_n3_n8_ret_n1_xsub_n0_n4_n9_n2_n7_n5_n6_n1_n3_n8_ret_n2
         -> op:add d:3 d:8 op:add d:1 op:sub d:0 d:4 d:9 d:2 d:7 d:5 d:6 d:1 d:3 d:8 op:sub d:2 <eos>
  the first `ret` must say add (old value), the second must say sub, 11 items after the correction.
Nothing but the held state can answer a `ret`: the address read r_t at a `ret` step reads "ret"; the conv encoder
sees +-4 bytes; prev symbol is a digit (uses never follow a value item directly unless D = 1).

Distance D of a use = items since the last value-bearing item (f, x or an earlier ret: each of those puts the value into
the decoder's input streams). Only the FIRST use after a fact/amend has D = distance to that event; the generator gives
every segment exactly one use at D ~ U[lo, hi] (+ 0-4 filler after it).

Splits (one fixed generator, random.Random(54); every input string unique across all splits):
  train      20000  n_amend 0/1/2 (p .25/.45/.30), D 1-16, amends only at item index p % 4 != 0
  val          500  same distribution as train
  test_iid    1000  same distribution as train
  test_pos     500  n_amend 1-2, D 1-16, amends ONLY at p % 4 == 0 (held-out amend positions)
  test_long    500  n_amend 1: pre-segment D 1-8, then amend, then D 17-32 (held-out longer gap after the amend)
  test_multi   500  n_amend 3-4 (held-out count), D 1-16: held-out longer sequences (up to ~120 items, train <= ~70)
  test_hold    500  control (a): n_amend 0, D 1-32 (pure hold; 17-32 held-out)
Controls: (a) test_hold + n_amend 0 in iid, (b) D buckets 1-4 / 5-9 / 10-16 / 17-32, (c) n_amend 2 (iid) and 3-4 (multi).

Scorer (same for nets and baselines; greedy output truncated at the first <eos>, position-wise vs gold):
  seq_exact, filler_acc, echo_acc, use_acc, later_use_acc (uses that are not the first after an event),
  pre_amend_use_acc (first-segment uses in sequences that have an amend: must say the OLD value),
  first_use[hold|amend][bucket] = acc and n, amend_first_use (all D), amend_long (D >= 10), hold_long (D >= 10),
  amend_first_use_addr_ok (only sequences whose filler copies are all right: address health conditioned).
PRIMARY (Amendment E): test_long amend_first_use (D 17-32).
"""

import json
import os
import random
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import blockaddr as BA  # noqa: E402

VALS = ["add", "sub", "mul", "div", "min", "max", "and", "or"]
PARTNER = {"add": "sub", "sub": "add", "mul": "div", "div": "mul", "min": "max", "max": "min", "and": "or", "or": "and"}
SYM = ["<pad>", "<sep>", "<eos>"] + [f"op:{v}" for v in VALS] + [f"d:{i}" for i in range(10)]
START = len(SYM)
EOS = SYM.index("<eos>")
STEPS = 2000                                  # training steps for every arm (batch 48, liquid13.train schedule)
NAMES = ["val", "test_iid", "test_pos", "test_long", "test_multi", "test_hold", "test_xlong", "test_xhold"]
BUCKETS = [(1, 4), (5, 9), (10, 16), (17, 32), (33, 63), (64, 160)]
LONG_D = 10


def item_str(it):
    k, v = it
    return {"f": "f", "x": "x", "n": "n"}[k] + str(v) if k != "r" else "ret"


def name_of(task):
    return "_".join(item_str(it) for it in task)


def gold_syms(task):
    out, cur = [], None
    for k, v in task:
        if k in "fx":
            cur = v
            out.append(f"op:{v}")
        elif k == "n":
            out.append(f"d:{v}")
        else:
            out.append(f"op:{cur}")
    return out + ["<eos>"]


def gen_task(r, n_amend, d_rng, pos="train", d0_rng=None, tail=(0, 4)):
    v0 = r.choice(VALS)
    items, cur, used = [("f", v0)], v0, {v0}
    fill = lambda: ("n", r.randrange(10))
    for k in range(n_amend + 1):
        lo, hi = d0_rng if (k == 0 and d0_rng) else d_rng
        D = r.randint(lo, hi)
        items += [fill() for _ in range(D - 1)] + [("r", None)]
        if k == n_amend:
            items += [fill() for _ in range(r.randint(*tail))]
            break
        items += [fill() for _ in range(r.randint(*tail))]
        ok = (lambda p: p % 4 != 0) if pos == "train" else (lambda p: p % 4 == 0) if pos == "heldout" else (lambda p: True)
        while not ok(len(items)):
            items.append(fill())
        free = [v for v in VALS if v not in used]
        new = PARTNER[cur] if (PARTNER[cur] in free and r.random() < 0.5) else r.choice(free)
        items.append(("x", new))
        cur = new
        used.add(new)
    return tuple(items)


def _n_amend_train(r):
    u = r.random()
    return 0 if u < .25 else 1 if u < .70 else 2


SPECS = {  # name: (count, sampler(r) -> task)
    "val": (500, lambda r: gen_task(r, _n_amend_train(r), (1, 16))),
    "test_iid": (1000, lambda r: gen_task(r, _n_amend_train(r), (1, 16))),
    "test_pos": (500, lambda r: gen_task(r, r.choice([1, 2]), (1, 16), pos="heldout")),
    "test_long": (500, lambda r: gen_task(r, 1, (17, 32), d0_rng=(1, 8))),
    "test_multi": (500, lambda r: gen_task(r, r.choice([3, 4]), (1, 16))),
    "test_hold": (500, lambda r: gen_task(r, 0, (1, 32))),
    "train": (20000, lambda r: gen_task(r, _n_amend_train(r), (1, 16))),
    # Amendment E1 (added after the CPU smoke showed gru/cfc at 1.0 on test_long; generated AFTER train, so every set
    # above, train included, is byte-identical to the E registration): D 64-160 = 4-10x the longest train gap.
    "test_xlong": (500, lambda r: gen_task(r, 1, (64, 160), d0_rng=(1, 8))),
    "test_xhold": (500, lambda r: gen_task(r, 0, (64, 160))),
}


class Data:
    def __init__(self):
        r = random.Random(54)
        self.sets, seen = {}, set()
        for k, (n, fn) in SPECS.items():           # eval sets first, train last: train never contains an eval string
            out = []
            while len(out) < n:
                t = fn(r)
                s = name_of(t)
                if s not in seen:
                    seen.add(s)
                    out.append(t)
            self.sets[k] = out
        self.max_out = max(len(t) + 1 for v in self.sets.values() for t in v) + 4


def name_bytes(tasks):
    ns = [[BA.B2I[b] for b in name_of(t).encode()] for t in tasks]
    L = max(map(len, ns))
    return torch.tensor([n + [0] * (L - len(n)) for n in ns])


def targets(tasks):
    ys = [[SYM.index(s) for s in gold_syms(t)] for t in tasks]
    T = max(map(len, ys))
    y = torch.tensor([v + [0] * (T - len(v)) for v in ys])
    y_in = torch.cat([torch.full((len(ys), 1), START), y[:, :-1]], 1)
    return y_in, y


# ------------------------------------------------------------------------------------------------- scorer
def use_meta(task):
    """per item t: kind, and for uses (D, event kind 'hold'|'amend', first-after-event, in first segment, seq has amend)."""
    meta, ev_t, ev_kind, last_val_t, used_since = [], 0, "hold", 0, False
    has_amend = any(k == "x" for k, _ in task)
    seen_amend = False
    for t, (k, _) in enumerate(task):
        if k == "f":
            meta.append({"kind": "fact"})
        elif k == "x":
            ev_t, ev_kind, last_val_t, used_since, seen_amend = t, "amend", t, False, True
            meta.append({"kind": "amend"})
        elif k == "n":
            meta.append({"kind": "filler"})
        else:
            meta.append({"kind": "use", "D": t - last_val_t, "event": ev_kind, "first": not used_since,
                         "pre_amend": has_amend and not seen_amend})
            used_since, last_val_t = True, t
    return meta


def _bucket(D):
    for lo, hi in BUCKETS:
        if lo <= D <= hi:
            return f"{lo}-{hi}"
    return f">{BUCKETS[-1][1]}"


def score(tasks, gens):
    """gens: list of token-id lists (raw greedy rows). -> metrics dict (see module docstring)."""
    c = {}

    def add(key, ok):
        a = c.setdefault(key, [0, 0])
        a[0] += int(ok)
        a[1] += 1
    for task, row in zip(tasks, gens):
        g = [SYM[j] if j < len(SYM) else "?" for j in row]
        g = g[:g.index("<eos>") + 1] if "<eos>" in g else g
        gold = gold_syms(task)
        add("seq_exact", g == gold)
        g = g + ["<pad>"] * max(0, len(gold) - len(g))
        meta = use_meta(task)
        fill_ok = all(g[t] == gold[t] for t, m in enumerate(meta) if m["kind"] == "filler")
        add("seq_filler_all_ok", fill_ok)
        for t, m in enumerate(meta):
            ok = g[t] == gold[t]
            if m["kind"] == "filler":
                add("filler_acc", ok)
            elif m["kind"] in ("fact", "amend"):
                add("echo_acc", ok)
            else:
                add("use_acc", ok)
                if m["pre_amend"]:
                    add("pre_amend_use_acc", ok)
                if not m["first"]:
                    add("later_use_acc", ok)
                    continue
                add(f"first_use/{m['event']}/{_bucket(m['D'])}", ok)
                add(f"{m['event']}_first_use", ok)
                if m["D"] >= LONG_D:
                    add(f"{m['event']}_long", ok)
                if m["event"] == "amend" and fill_ok:
                    add("amend_first_use_addr_ok", ok)
        add("eos_ok", len(g) > len(gold) - 1 and g[len(gold) - 1] == "<eos>")
    res = {k: round(v[0] / v[1], 4) for k, v in sorted(c.items())} | {"n/" + k: v[1] for k, v in sorted(c.items())}
    res["pass1"] = res["seq_exact"]                   # liquid13 run-log field (whole output exact)
    return res


@torch.no_grad()
def greedy_rows(net, tasks, wrap=None):
    """net.greedy over chunks of 256; wrap(chunk_tasks) -> context manager installed around each chunk (ablations)."""
    out = []
    for i in range(0, len(tasks), 256):
        ch = tasks[i:i + 256]
        if wrap is None:
            out += net.greedy(name_bytes(ch)).tolist()
        else:
            with wrap(ch):
                out += net.greedy(name_bytes(ch)).tolist()
    return out


def evaluate(net, data, names=NAMES):
    net.eval()
    res = {k: score(data.sets[k], greedy_rows(net, data.sets[k])) for k in names}
    net.train()
    return res


# ------------------------------------------------------------------------------------------------- baselines (gate 1)
def baseline_rows(tasks, rule, const_val):
    """oracle on every non-use position; the rule answers the uses."""
    rows = []
    for task in tasks:
        g = gold_syms(task)
        vals = []                                     # (t, value) of value-bearing items, incl. uses (gold value)
        first = task[0][1]
        for t, (k, v) in enumerate(task):
            if k == "r":
                if rule == "oracle":
                    ans = g[t][3:]
                elif rule == "const":
                    ans = const_val
                elif rule == "never_update":
                    ans = first
                else:                                 # recentW_else_first: a fast memory of W items + birth snapshot
                    W = int(rule[len("recent"):rule.index("_")])
                    tl, vl = vals[-1]
                    ans = vl if t - tl <= W else first
                g[t] = f"op:{ans}"
                vals.append((t, ans))                 # a use re-emits its OWN answer into the stream
            elif k in "fx":
                vals.append((t, v))
        rows.append([SYM.index(s) for s in g])
    return rows


RULES = ["const", "never_update", "recent4_else_first", "recent8_else_first", "oracle"]


def baselines(data):
    from collections import Counter
    const_val = Counter(s[3:] for t in data.sets["train"] for s, (k, _) in zip(gold_syms(t), t) if k == "r").most_common(1)[0][0]
    out = {"const_val": const_val}
    for rule in RULES:
        out[rule] = {k: score(data.sets[k], baseline_rows(data.sets[k], rule, const_val)) for k in NAMES}
    return out


# ------------------------------------------------------------------------------------------------- stress2/3 instruments
class _Wrap:
    """eval-only ablations that the cell flags cannot express, installed by shadowing cell.forward (cell code untouched):
    freeze_after_t0  slow half copied from x_{t-1} for t >= 1 (rate exactly 0: no 1/tau leak, unlike closed_after_t0)
    oracle_gate      slow groups forced open at t0, at an amend step and the step after it; closed (leak 1/tau) else
                     (stress4, Amendment F: closed = no leak, rate 0). Amendment G: also open at every birth item.
    closed_after_births  Amendment G (review m1): the gate acts normally on the birth items (t < n_birth), closed after."""

    def __init__(self, cell, mode, tasks, W=None):
        W = W or sys.modules[__name__]
        self.cell, self.mode, self.t = cell, mode, 0
        T = max(len(t) for t in tasks) + 2
        m = torch.zeros(len(tasks), T + BA.MAX_OUT, dtype=torch.bool)
        nb = torch.zeros(len(tasks), T + BA.MAX_OUT, dtype=torch.bool)
        for i, task in enumerate(tasks):
            m[i, :W.n_birth(task)] = True
            nb[i, :W.n_birth(task)] = True
            for t, (k, _) in enumerate(task):
                if k == "x":
                    m[i, t] = m[i, t + 1] = True
        self.mask = nb if mode == "closed_after_births" else m

    def __enter__(self):
        cell, orig = self.cell, self.cell.forward

        def fwd(streams, x, surprise, start=None, dt=1.0):
            t = self.t
            self.t += 1
            if self.mode == "freeze_after_t0":
                out = orig(streams, x, surprise, start, dt)
                return out if t == 0 else torch.cat([out[:, :cell.Hf], x[:, cell.Hf:]], 1)
            snap = (cell._sum, cell._n) if hasattr(cell, "w3") else None   # stress3 (tensors are reassigned, not mutated)
            cell.force_open = self.mode == "oracle_gate"
            try:
                a = orig(streams, x, surprise, start, dt)
            finally:
                cell.force_open = False
            if t == 0:                                # gate open at t0: the closed branch is not used
                return a
            post = (cell._sum, cell._n) if snap is not None else None
            if snap is not None:                      # stress3 keeps a running input-surprise baseline: count once
                cell._sum, cell._n = snap[0], snap[1]
            cell.closed_after_t0 = True
            try:
                b = orig(streams, x, surprise, torch.zeros_like(start), dt)
            finally:
                cell.closed_after_t0 = False
            if post is not None:                      # keep the state of the first (real-start) call
                cell._sum, cell._n = post
            op = self.mask[:, t]
            return torch.where(op.to(x.device)[:, None], a, b)
        cell.forward = fwd
        return self

    def __exit__(self, *exc):
        del self.cell.forward
        return False


ABL_SPLITS = ["test_iid", "test_long", "test_hold", "test_xlong", "test_xhold"]
ABL_KEYS = ["amend_first_use", "amend_long", "hold_first_use", "hold_long", "pre_amend_use_acc", "use_acc", "filler_acc",
            "seq_exact"]


@torch.no_grad()
def cell_stats(net, data, nsamp=512, W=None):
    """Amendment E instruments for stress2 / stress3 / stress4 (None for other arms). W = world module (amend54 or
    amendk54, Amendment F): its name_bytes / greedy_rows / score / ABL_SPLITS are used."""
    W = W or sys.modules[__name__]
    if net.kind not in ("stress2", "stress3", "stress4"):
        return None
    net.eval()
    cell = net.cell
    out = {"what": f"{net.kind} slow-group hard open rate around AMEND steps (t0 excluded from 'elsewhere')"}
    for sp in ("test_iid", "test_long", "test_xlong"):
        tasks = data.sets[sp][:nsamp]
        cell.stat = []
        gen = net.greedy(W.name_bytes(tasks))
        q = cell.stat
        cell.stat = None
        hard = torch.stack([s[0] for s in q], 1).float().cpu()          # [B, T, G]
        surp = torch.stack([s[2] for s in q], 1).float().cpu()          # [B, T]
        dis = torch.stack([s[3] for s in q], 1).float().cpu().mean(-1)  # [B, T]
        zin = torch.stack([s[4] for s in q], 1).float().cpu() if len(q[0]) > 4 else None
        B, T, _ = hard.shape
        alive = ~(torch.cumsum((gen == EOS).long(), 1) - (gen == EOS).long() > 0)
        cls = torch.full((B, T), 5, dtype=torch.long)                   # 0 fact 1 amend 2 post_amend 3 use 4 filler 5 off
        for i, task in enumerate(tasks):
            for t, (k, _) in enumerate(task):
                if t < T:
                    cls[i, t] = {"f": 0, "x": 1, "r": 3, "n": 4}[k]
            for t, (k, _) in enumerate(task):
                if k == "x" and t + 1 < T and cls[i, t + 1] != 1:
                    cls[i, t + 1] = 2
        of = hard.mean(-1)
        anyo = (hard.max(-1).values > 0)
        valid = alive & (cls != 5)
        names = ["fact_t0", "amend", "post_amend", "use", "filler"]
        row = {}
        for ci, nm in enumerate(names):
            m = valid & (cls == ci)
            row[nm] = {"open_rate": round(of[m].mean().item(), 4) if m.any() else None,
                       "any_open": round(anyo[m].float().mean().item(), 4) if m.any() else None, "n": int(m.sum()),
                       "surprise": round(surp[m].mean().item(), 3) if m.any() else None,
                       "dis": round(dis[m].mean().item(), 4) if m.any() else None,
                       **({"zin": round(zin[m].mean().item(), 3) if m.any() else None} if zin is not None else {})}
        win = valid & ((cls == 1) | (cls == 2))
        else_ = valid & (cls >= 3) & (torch.arange(T)[None] >= 1)
        t1 = valid & (torch.arange(T)[None] >= 1)
        a_rate, e_rate = of[win].mean().item() if win.any() else float("nan"), of[else_].mean().item()
        n_events = int(anyo[t1].sum())
        row["amend_window_open_rate"] = round(a_rate, 4)
        row["elsewhere_open_rate"] = round(e_rate, 4)
        row["ratio_amend_vs_elsewhere"] = round(a_rate / e_rate, 2) if e_rate > 0 else (None if a_rate == 0 else "inf")
        row["t>=1_open_rate"] = round(of[t1].mean().item(), 4)
        row["t>=1_open_events"] = n_events
        tb = valid & (torch.arange(T)[None] >= torch.tensor([W.n_birth(t_) for t_ in tasks])[:, None])
        row["t>=birth_open_rate"] = round(of[tb].mean().item(), 4)   # Amendment G (review m1): after ALL birth items
        row["t>=birth_open_events"] = int(anyo[tb].sum())
        row["amend_event_hit"] = None                  # fraction of amend events with any group open at p or p+1
        hits = []
        for i, task in enumerate(tasks):
            for t, (k, _) in enumerate(task):
                if k == "x" and t + 1 < T and alive[i, t]:
                    hits.append(bool(anyo[i, t] or anyo[i, t + 1]))
        row["amend_event_hit"] = round(sum(hits) / len(hits), 4) if hits else None
        row["n_amend_events"] = len(hits)
        if n_events >= 20:
            x, y = of[t1].float(), win[t1].float()
            x, y = x - x.mean(), y - y.mean()
            d = (x.pow(2).sum() * y.pow(2).sum()).sqrt().item()
            row["corr_open_amend_window"] = round((x * y).sum().item() / d, 4) if d > 0 else None
        else:
            row["corr_open_amend_window"] = f"not computed: {n_events} open events at t>=1 (< 20)"
        prof = {}
        for off in range(-2, 4):
            vs = []
            for i, task in enumerate(tasks):
                for t, (k, _) in enumerate(task):
                    if k == "x" and 0 <= t + off < T and alive[i, t + off]:
                        vs.append(of[i, t + off].item())
            prof[str(off)] = round(sum(vs) / len(vs), 4) if vs else None
        row["open_profile_around_amend"] = prof
        out[sp] = row
    out["w1"] = [round(v, 3) for v in torch.nn.functional.softplus(cell.w1).tolist()]
    out["w2"] = [round(v, 3) for v in torch.nn.functional.softplus(cell.w2).tolist()]
    out["gb"] = [round(v, 3) for v in cell.gb.tolist()]
    if hasattr(cell, "w3"):
        out["w3"] = [round(v, 3) for v in torch.nn.functional.softplus(cell.w3).tolist()]
    abl = {}
    for sp in W.ABL_SPLITS:
        tasks = data.sets[sp]
        row = {}
        modes = ("normal", "zero_slow", "force_open", "closed_after_t0", "freeze_after_t0", "oracle_gate")
        if W.MECH["ablation"] not in modes:
            modes = modes + (W.MECH["ablation"],)
        for mode in modes:
            wrap = None
            if mode in ("freeze_after_t0", "oracle_gate", "closed_after_births"):
                wrap = lambda ch, mode=mode: _Wrap(cell, mode, ch, W)
            else:
                cell.zero_slow, cell.force_open, cell.closed_after_t0 = (mode == "zero_slow", mode == "force_open",
                                                                         mode == "closed_after_t0")
            try:
                sc = W.score(tasks, W.greedy_rows(net, tasks, wrap))
            finally:
                cell.zero_slow = cell.force_open = cell.closed_after_t0 = False
            row[mode] = {k: sc.get(k) for k in W.ABL_KEYS}
        abl[sp] = row
    out["ablation"] = abl
    net.eval()
    return out


# ------------------------------------------------------------------------------------------------- report
ARMS = ["gru", "cfc", "ltc", "lstm", "stress", "stress2", "stress3", "stress4"]
VARIANTS = ("stress2", "stress3", "stress4")
WORLD = "amend"
REPORT = "report54e.json"
PRIMARY = ("test_long", "amend_first_use")
PRIMARY_E1 = ("test_xlong", "amend_first_use")   # Amendment E1: used for the verdict only if E is at ceiling
FALLBACK = PRIMARY_E1
LABEL = "E"                                    # verdict label (amendk54: "G")
MECH = {"rate": "t>=1_open_rate", "ablation": "closed_after_t0", "metric": "amend_first_use"}


def n_birth(task):
    return 1
SECONDARY = [("test_xlong", "amend_first_use_addr_ok"), ("test_xlong", "eos_ok"), ("test_xlong", "amend_first_use"), ("test_xhold", "hold_long"), ("test_xlong", "filler_acc"), ("test_iid", "amend_long"), ("test_multi", "amend_first_use"), ("test_pos", "amend_first_use"),
             ("test_hold", "hold_long"), ("test_long", "amend_first_use_addr_ok"), ("test_long", "filler_acc")]
T2 = 2.969                                    # t_.9875,6: Bonferroni over 2 comparisons (vs gru, vs cfc) per variant
TQ = 3.863                                    # Amendment G (review m2): family-wise, 3 variants x 2 refs, t_{1-.05/12,6}


def _paired(x, g, tq, margin=0.10):
    d = [x[s] - g[s] for s in sorted(g)]
    m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
    lo, hi = m - tq * se, m + tq * se
    call = "BETTER" if lo > 0 else "PASS" if lo > -margin else "LOSES" if hi < -margin else "INCONCLUSIVE"
    return {"call": call, "mean_diff": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
            "per_seed_diff": [round(v, 4) for v in d]}


def report(out_dir, primary_lr=3e-3, W=None, world=None):
    """W = world module (Amendment F: amendk54 reuses this report with its own PRIMARY, no fallback)."""
    W = W or sys.modules[__name__]
    world = world or W.WORLD
    PRIMARY, FALLBACK, SECONDARY, ARMS = W.PRIMARY, W.FALLBACK, W.SECONDARY, W.ARMS
    d = os.path.join(out_dir, world)
    rows = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            r = json.load(open(os.path.join(d, fn)))
            rows.setdefault((r["arm"], r["lr"]), {})[r["seed"]] = r
    get = lambda r, sp, k: r["eval"][sp].get(k)
    rep = {"what": f"Amendment E/F (#54A world {world})", "primary": list(PRIMARY), "primary_lr": primary_lr, "arms": {}}
    lrs = sorted({l for (_, l) in rows})
    for arm in ARMS:
        a = {}
        for lr in lrs:
            rs = rows.get((arm, lr), {})
            if not rs:
                continue
            e = {}
            for sp, k in [PRIMARY] + SECONDARY + [("val", "amend_long")]:
                v = [get(rs[s], sp, k) for s in sorted(rs)]
                v = [x for x in v if x is not None]
                if v:
                    e[f"{sp}/{k}"] = {"mean": round(st.mean(v), 4), "sd": round(st.stdev(v), 4) if len(v) > 1 else None,
                                      "min": min(v), "n": len(v), "seeds": v}
            a[f"lr{lr:g}"] = e
        if a:
            have = [lr for lr in lrs if f"lr{lr:g}" in a]
            means = {lr: a[f"lr{lr:g}"].get("val/amend_long", {}).get("mean", -1) for lr in have}
            a["best_lr_by_val"] = bv = max(means, key=means.get)
            a["best_lr_by_val_at_grid_edge"] = len(have) > 1 and bv in (min(have), max(have))
            for tag, prim in (("", PRIMARY),) + ((("_e1", FALLBACK),) if FALLBACK else ()):
                pm = {lr: a[f"lr{lr:g}"].get("/".join(prim), {}).get("mean", -1) for lr in have}
                a[f"best_lr_by_primary{tag}"] = bp = max(pm, key=pm.get)
                a[f"best_lr_by_primary{tag}_at_grid_edge"] = len(have) > 1 and bp in (min(have), max(have))
            rep["arms"][arm] = a
    lr_key = f"lr{primary_lr:g}"

    def decide(var, prim, tag, n_ne):
        pv = lambda arm, lr: {s: get(r, *prim) for s, r in rows.get((arm, lr), {}).items()}
        comps = {}
        for lr in lrs:
            x = pv(var, lr)
            for ref in ("gru", "cfc"):
                g = pv(ref, lr)
                g = {s_: (0.0 if v_ is None else v_) for s_, v_ in g.items()}   # ref undefined (no healthy event) = 0
                none = sorted(s_ for s_ in x if x[s_] is None)                  # variant undefined -> no verdict
                if len(x) == 7 and sorted(x) == sorted(g) and none:   # e.g. co-primary with no address-healthy event
                    comps[f"lr{lr:g}/vs_{ref}"] = {"call": f"NO VERDICT: metric undefined on seeds {none}"}
                elif len(x) == 7 and sorted(x) == sorted(g):
                    comps[f"lr{lr:g}/vs_{ref}"] = _paired(x, g, W.TQ)
                else:
                    comps[f"lr{lr:g}/vs_{ref}"] = {"call": f"NO VERDICT: seeds {sorted(x)} vs {ref} {sorted(g)}"}
        cg = comps.get(f"{lr_key}/vs_gru", {}).get("call", "")
        cc = comps.get(f"{lr_key}/vs_cfc", {}).get("call", "")
        ceiling = all(rep["arms"].get(r_, {}).get(lr_key, {}).get("/".join(prim), {}).get("mean", 0) >= .95
                      for r_ in ("gru", "cfc"))
        edge = rep["arms"].get(var, {}).get(f"best_lr_by_primary{tag}_at_grid_edge")
        if cg.startswith("NO VERDICT") or cc.startswith("NO VERDICT"):
            verdict = f"NO VERDICT ({cg} | {cc})"
        elif cg == "BETTER" and cc == "BETTER" and getattr(W, "FLOOR", None) is not None and \
                rep["arms"].get(var, {}).get(lr_key, {}).get("/".join(prim), {}).get("mean", 0) <= W.FLOOR:
            verdict = f"NOT SHOWN: better than gru and cfc but itself <= {W.FLOOR} (slot-blind floor); no kill"
        elif cg == "BETTER" and cc == "BETTER":
            verdict = "SHOWN: better than gru AND cfc on long-distance amend items"
        elif ceiling:
            verdict = "WORLD AT CEILING: gru and cfc >= .95; not discriminative, no kill"
        elif getattr(W, "FLOOR", None) is not None and all(
                rep["arms"].get(r_, {}).get(lr_key, {}).get("/".join(prim), {}).get("mean", 1) <= W.FLOOR
                for r_ in ("gru", "cfc")):
            verdict = f"WORLD AT FLOOR: gru and cfc <= {W.FLOOR} (slot-blind baseline + .05); not discriminative, no kill"
        elif n_ne >= 4:
            verdict = f"MECHANISM NOT ENGAGED ({n_ne}/7 seeds): no kill; own-bug check + adapt (rule 7)"
        else:
            verdict = f"NOT SHOWN on this world (kill rule {W.LABEL})" + (" [best lr at grid edge: extend grid]" if edge else "")
        return {"primary": list(prim), "comparisons": comps, "ceiling": ceiling, "best_lr_at_grid_edge": edge,
                "verdict": f"{verdict} | vs gru {cg}, vs cfc {cc}"}

    for var in W.VARIANTS:
        v = {}
        # mechanism tags at the primary lr (Amendment E / review M2)
        mech, n_ne = {}, 0
        for s, r in sorted(rows.get((var, primary_lr), {}).items()):
            m = r.get("sat") or {}
            if "test_long" not in m:
                mech[s] = "no stats"
                continue
            ab = m["ablation"]
            MK = W.MECH                                # amend: t>=1 / closed_after_t0; amendg: after births (review m1)
            t1 = m["test_iid"][MK["rate"]]
            drop_c = max(abs((ab[sp]["normal"][MK["metric"]] or 0) - (ab[sp][MK["ablation"]][MK["metric"]] or 0))
                         for sp in ("test_iid", "test_long"))
            ne = t1 < 0.03 and drop_c <= 0.01
            n_ne += ne
            dr = lambda sp, k, mode: round((ab[sp]["normal"][k] or 0) - (ab[sp][mode][k] or 0), 4)
            mech[s] = {"mech_rule": MK, "open_rate_iid": t1, "ratio_iid": m["test_iid"]["ratio_amend_vs_elsewhere"],
                       "ratio_long": m["test_long"]["ratio_amend_vs_elsewhere"],
                       "amend_window_open_long": m["test_long"]["amend_window_open_rate"],
                       "elsewhere_open_long": m["test_long"]["elsewhere_open_rate"],
                       "amend_event_hit_long": m["test_long"]["amend_event_hit"],
                       "open_events_t>=1_iid": m["test_iid"]["t>=1_open_events"],
                       "freeze_drop_amend_long": dr("test_long", "amend_first_use", "freeze_after_t0"),
                       "freeze_drop_hold_long": dr("test_hold", "hold_long", "freeze_after_t0"),
                       "freeze_drop_amend_xlong": dr("test_xlong", "amend_first_use", "freeze_after_t0"),
                       "zero_slow_drop_amend_long": dr("test_long", "amend_first_use", "zero_slow"),
                       "force_open_drop_hold_long": dr("test_hold", "hold_long", "force_open"),
                       "oracle_gate_drop_amend_long": dr("test_long", "amend_first_use", "oracle_gate"),
                       "closed_max_abs_drop": round(drop_c, 4), "not_engaged": ne}
        v["mechanism"] = {"per_seed": mech, "n_not_engaged": n_ne}
        v["E"] = decide(var, PRIMARY, "", n_ne)
        if getattr(W, "CO_PRIMARY", None):          # Amendment G: SHOWN also needs the co-primary not LOSES vs either
            v["co_primary"] = decide(var, W.CO_PRIMARY, "_co", n_ne)
            calls = [c.get("call", "") for k_, c in v["co_primary"]["comparisons"].items() if k_.startswith(lr_key)]
            if v["E"]["verdict"].startswith("SHOWN") and any(c == "LOSES" or c.startswith("NO VERDICT") for c in calls):
                v["E"]["verdict"] = ("NOT SHOWN: primary better but co-primary (address-healthy) LOSES or undefined | "
                                     + v["E"]["verdict"])
        if FALLBACK:
            v["E1"] = decide(var, FALLBACK, "_e1", n_ne)
        # registered order: E decides; only if E is at ceiling does E1 (test_xlong, same rule) decide.
        # Amendment F (amendk): no fallback, the primary (test_xlong) decides alone.
        v["verdict"] = ((W.LABEL + ": ") + v["E"]["verdict"]) if not (FALLBACK and v["E"]["ceiling"]) else (
            "E at ceiling -> E1 (test_xlong): " + v["E1"]["verdict"])
        rep[f"verdict_{var}"] = v
    json.dump(rep, open(os.path.join(out_dir, W.REPORT.format(world=world)), "w"), indent=1)
    sec = FALLBACK or SECONDARY[0]
    for arm, a in rep["arms"].items():
        print(arm, {lr: (e.get("/".join(PRIMARY), {}).get("mean"), e.get("/".join(sec), {}).get("mean"),
                         e.get("test_iid/amend_long", {}).get("mean"))
                    for lr, e in a.items() if lr.startswith("lr")}, "best_lr", a["best_lr_by_primary"],
              "edge" if a["best_lr_by_primary_at_grid_edge"] else "", "best_lr_e1", a.get("best_lr_by_primary_e1"),
              "edge" if a.get("best_lr_by_primary_e1_at_grid_edge") else "")
    for var in W.VARIANTS:
        print(var, rep[f"verdict_{var}"]["verdict"])
    return rep


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("baselines")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p = sub.add_parser("show")
    a = ap.parse_args()
    if a.cmd == "show":
        data = Data()
        for k, v in data.sets.items():
            ls = [len(t) for t in v]
            print(k, len(v), "len min/mean/max", min(ls), round(sum(ls) / len(ls), 1), max(ls),
                  "n_amend", dict(sorted(__import__("collections").Counter(sum(k_ == "x" for k_, _ in t) for t in v).items())))
        print("max_out", data.max_out)
        t = data.sets["train"][0]
        print(name_of(t))
        print(" ".join(gold_syms(t)))
        return
    if a.cmd == "baselines":
        out = baselines(Data())
        os.makedirs(a.out, exist_ok=True)
        json.dump(out, open(os.path.join(a.out, "baselines54e.json"), "w"), indent=1)
        for rule in RULES:
            print(rule, {sp: (out[rule][sp].get("amend_first_use"), out[rule][sp].get("hold_long"),
                              out[rule][sp].get("seq_exact")) for sp in NAMES})
        return
    report(a.out)


if __name__ == "__main__":
    main()
