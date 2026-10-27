"""Ledger #54(A), worlds `amendg` / `amendg8` for liquid13.py (Amendment G, registered 2026-10-27; predictions and
verdict rule in the liquid13.py header). Supersedes the F world `amendk` (review B1: the first use after an amend
ALWAYS asked the amended slot, so the slot-blind rule `last_x_else_birth` scored 1.0 on amend_first_use).

Why: on world `amend` (one held operator) gru and cfc sat at the ceiling. His case "a def de soma que so trocou um
sinal" is a SIGNATURE where ONE sign of several changes and the rest must stay: K held slots, a correction rewrites
one slot, a use asks one named slot.

Input (bytes a-z 0-9 _, items joined by "_"; one decoder step reads one item):
  <s><v>    birth, items 0..K-1 in slot order a, b, c, ...: slot s holds operator v (v in VALS, repeats allowed)
  x<s><v>   AMEND: slot s is corrected to v (v != current and != birth value of s; with p .5 the sign partner of the
            current value: add<->sub, mul<->div, min<->max, and<->or = "only one sign changed")
  n<d>      filler: output d:<d>
  r<s>      USE: output op:<current value of slot s>
Output per item: birth/amend -> op:<v> (echo), n<d> -> d:<d>, r<s> -> op:<current s>; then <eos>.
Distance D of a use of slot s = items since the last item that put s's value into the stream (birth, amend, or an
earlier use of s). Segment 0 asks a random slot at D ~ U[d0] from its birth (then, with p `other`, a different slot).
Every amend segment (G): amend of slot a; after a gap of D ~ U[d] items the segment asks a PAIR (first, second) =
(a, o) or (o, a) with p .5 each, o != a uniform, 0-2 fillers between them. Inside each gap `distract` uses of slots
other than a and `first` (so neither the last value in the stream nor the last x item is the answer).
Primary metric post_amend_all = per amend event, ALL uses in its segment correct (the amended slot AND the other
slot): slot-blind rules cannot pass it (last_x_else_birth .014, last_seen_any .014 on test_xlong).

Splits (random.Random(56); every input string unique across splits; eval sets first, train last):
  train 20000 / val 500 / test_iid 1000: K ~ U[K range], n_amend 0/1/2/3 (p .20/.35/.30/.15) for 75% of sequences,
             4-6 for 25% (so train lengths cover the primary test lengths: review M1), D 1-16, other p .5,
             distract 0-2 per gap
  test_long   500  n_amend 1, d0 1-8, D 17-32 after the amend, other p 1, distract 2
  test_xlong  500  n_amend 1, d0 1-8, D 33-64 after the amend, other p 1, distract 2   <- PRIMARY G (held-out GAP,
                   lengths inside the train range: 0% of post-amend uses beyond train length p95)
  test_hold   500  n_amend 0, D 1-32 from birth, distract 2     test_xhold 500  n_amend 0, D 33-64, distract 2
  test_xxlong 500  as test_xlong with D 64-160 (SECONDARY: gap AND length extrapolation, 24% of uses beyond train max)
  test_xxhold 500  as test_xhold with D 64-160 (secondary)
Scorer = amend54's keys (seq_exact, filler_acc, echo_acc, use_acc, ..., amend_first_use, amend_first_use_addr_ok,
eos_ok) + hold_after_amend + post_amend_use/{amended|other} (every use after the first amend, split by whether it asks
the slot amended in its segment) + post_amend_all + post_amend_all_addr_ok (post_amend_all on events whose segment
fillers are all right: address healthy).
"""

import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import amend54 as AM  # noqa: E402
import blockaddr as BA  # noqa: E402

VALS, PARTNER, SYM, START, EOS = AM.VALS, AM.PARTNER, AM.SYM, AM.START, AM.EOS
SLOTS = "abcdefgh"
STEPS = AM.STEPS
CONFIGS = {  # world name -> K range (Amendment G; the F world `amendk` is superseded: B1 slot-blind shortcut)
    "amendg": (4, 6),
    "amendg8": (6, 8),
}
NAMES = ["val", "test_iid", "test_long", "test_xlong", "test_hold", "test_xhold", "test_xxlong", "test_xxhold"]
ABL_SPLITS = ["test_iid", "test_long", "test_hold", "test_xlong", "test_xhold"]
ABL_KEYS = AM.ABL_KEYS + ["post_amend_all", "post_amend_all_addr_ok", "post_amend_use/amended", "post_amend_use/other"]
BUCKETS, LONG_D = AM.BUCKETS, AM.LONG_D
WORLD = "amendg"


def item_str(it):
    k, v = it
    if k == "f":
        return v[0] + v[1]
    if k == "x":
        return "x" + v[0] + v[1]
    if k == "r":
        return "r" + v
    return "n" + str(v)


def name_of(task):
    return "_".join(item_str(it) for it in task)


def gold_syms(task):
    out, cur = [], {}
    for k, v in task:
        if k in "fx":
            cur[v[0]] = v[1]
            out.append(f"op:{v[1]}")
        elif k == "n":
            out.append(f"d:{v}")
        else:
            out.append(f"op:{cur[v]}")
    return out + ["<eos>"]


def gen_task(r, K, n_amend, d_rng, d0_rng=None, other=0.5, tail=(0, 4), distract=(0, 2)):
    """Amendment G: after an amend of slot a, the FIRST query at D ~ U[d] asks a with p .5, else another slot; the
    other one of the pair (a, or a slot != a) is asked right after (0-2 filler). Distractor uses inside every gap
    never ask the slot whose distance is being measured (and never the amended slot in an amend segment)."""
    slots = SLOTS[:K]
    birth = {s: r.choice(VALS) for s in slots}
    items = [("f", (s, birth[s])) for s in slots]
    cur = dict(birth)
    last = {s: i for i, s in enumerate(slots)}             # index of the last value-bearing item per slot
    fill = lambda: ("n", r.randrange(10))

    def use(s):
        items.append(("r", s))
        last[s] = len(items) - 1

    def gap_to(t_use, avoid):
        gap = t_use - len(items)
        m = min(r.randint(*distract), gap)                 # uses of OTHER slots inside the gap: the last value in
        at = set(r.sample(range(gap), m))                  # the stream is then not the answer
        for j in range(gap):
            if j in at:
                use(r.choice([s for s in slots if s not in avoid]))
            else:
                items.append(fill())
    q = r.choice(slots)
    lo, hi = d0_rng or d_rng
    gap_to(max(last[q] + r.randint(lo, hi), len(items)), {q})
    use(q)
    items += [fill() for _ in range(r.randint(*tail))]
    if r.random() < other:
        use(r.choice([s for s in slots if s != q]))
        items += [fill() for _ in range(r.randint(0, 2))]
    for k in range(n_amend):
        a = r.choice(slots)
        free = [v for v in VALS if v not in (cur[a], birth[a])]
        new = PARTNER[cur[a]] if (PARTNER[cur[a]] in free and r.random() < 0.5) else r.choice(free)
        items.append(("x", (a, new)))
        cur[a] = new
        last[a] = len(items) - 1
        o = r.choice([s for s in slots if s != a])
        first, second = (a, o) if r.random() < 0.5 else (o, a)
        gap_to(len(items) - 1 + r.randint(*d_rng), {a, first})
        use(first)
        items += [fill() for _ in range(r.randint(0, 2))]
        use(second)
        items += [fill() for _ in range(r.randint(*tail))]
    return tuple(items)


def _n_amend_train(r):
    """75% short (0-3 amends), 25% long (4-6 amends): train lengths cover the primary test lengths (review M1)."""
    if r.random() < .25:
        return r.randint(4, 6)
    u = r.random()
    return 0 if u < .20 else 1 if u < .55 else 2 if u < .85 else 3


def specs(kr):
    K = lambda r: r.randint(*kr)
    return {
        "val": (500, lambda r: gen_task(r, K(r), _n_amend_train(r), (1, 16))),
        "test_iid": (1000, lambda r: gen_task(r, K(r), _n_amend_train(r), (1, 16))),
        "test_long": (500, lambda r: gen_task(r, K(r), 1, (17, 32), d0_rng=(1, 8), other=1.0, distract=(2, 2))),
        "test_xlong": (500, lambda r: gen_task(r, K(r), 1, (33, 64), d0_rng=(1, 8), other=1.0, distract=(2, 2))),
        "test_hold": (500, lambda r: gen_task(r, K(r), 0, (1, 32), distract=(2, 2))),
        "test_xhold": (500, lambda r: gen_task(r, K(r), 0, (33, 64), distract=(2, 2))),
        # secondary: gap 64-160 = LENGTH extrapolation too (inputs longer than train), reported apart (review M1)
        "test_xxlong": (500, lambda r: gen_task(r, K(r), 1, (64, 160), d0_rng=(1, 8), other=1.0, distract=(2, 2))),
        "test_xxhold": (500, lambda r: gen_task(r, K(r), 0, (64, 160), distract=(2, 2))),
        "train": (20000, lambda r: gen_task(r, K(r), _n_amend_train(r), (1, 16))),
    }


class Data:
    def __init__(self, world="amendg"):
        r = random.Random(56)
        self.world, self.k_range = world, CONFIGS[world]
        self.sets, seen = {}, set()
        for k, (n, fn) in specs(self.k_range).items():
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


def use_meta(task):
    amended_later = {}
    for t, (k, v) in enumerate(task):
        if k == "x":
            amended_later.setdefault(v[0], t)
    meta, last, ev, used, any_amend_t, seg, seg_slot = [], {}, {}, {}, None, -1, None
    for t, (k, v) in enumerate(task):
        if k == "f":
            last[v[0]], ev[v[0]], used[v[0]] = t, "hold", False
            meta.append({"kind": "fact"})
        elif k == "x":
            last[v[0]], ev[v[0]], used[v[0]] = t, "amend", False
            any_amend_t = t if any_amend_t is None else any_amend_t
            seg, seg_slot = seg + 1, v[0]                     # amend segment id (uses until the next amend)
            meta.append({"kind": "amend"})
        elif k == "n":
            meta.append({"kind": "filler"})
        else:
            s = v
            meta.append({"kind": "use", "D": t - last[s], "event": ev[s], "first": not used[s],
                         "pre_amend": s in amended_later and t < amended_later[s],
                         "after_other_amend": ev[s] == "hold" and any_amend_t is not None and any_amend_t < t,
                         "seg": seg, "asked_amended": seg >= 0 and s == seg_slot})
            used[s], last[s] = True, t
    return meta


def _bucket(D):
    return AM._bucket(D)


def score(tasks, gens):
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
        segs = {}                                     # Amendment G: per amend event, ALL uses of its segment right
        for t, m in enumerate(meta):
            if m["kind"] == "use" and m["seg"] >= 0:
                segs.setdefault(m["seg"], []).append(g[t] == gold[t])
                add("post_amend_use/" + ("amended" if m["asked_amended"] else "other"), g[t] == gold[t])
        for v in segs.values():
            add("post_amend_all", all(v))
            if fill_ok:
                add("post_amend_all_addr_ok", all(v))
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
                if m["after_other_amend"]:
                    add("hold_after_amend", ok)
                if m["event"] == "amend" and fill_ok:
                    add("amend_first_use_addr_ok", ok)
        add("eos_ok", len(g) > len(gold) - 1 and g[len(gold) - 1] == "<eos>")
    res = {k: round(v[0] / v[1], 4) for k, v in sorted(c.items())} | {"n/" + k: v[1] for k, v in sorted(c.items())}
    res["pass1"] = res["seq_exact"]
    return res


@torch.no_grad()
def greedy_rows(net, tasks, wrap=None):
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


def cell_stats(net, data, nsamp=512):
    return AM.cell_stats(net, data, nsamp, W=sys.modules[__name__])


# ------------------------------------------------------------------------------------------------- baselines
def baseline_rows(tasks, rule, const_val):
    """oracle on every non-use position; the rule answers the uses (a use re-emits its OWN answer into the stream)."""
    rows = []
    for task in tasks:
        g = gold_syms(task)
        birth, slot_last, any_last, last_x = {}, {}, None, None   # birth; slot -> (t, value); last value; last x value
        for t, (k, v) in enumerate(task):
            if k == "r":
                if rule == "oracle":
                    ans = g[t][3:]
                elif rule == "const":
                    ans = const_val
                elif rule == "never_update":
                    ans = birth[v]
                elif rule == "last_seen_any":         # one-slot memory: whatever value came last
                    ans = any_last
                elif rule == "last_x_else_birth":     # slot-blind amend memory: value of the last x item, else birth
                    ans = last_x if last_x is not None else birth[v]
                else:                                 # recentW_else_birth: per-slot fast memory of W items + birth
                    W = int(rule[len("recent"):rule.index("_")])
                    tl, vl = slot_last[v]
                    ans = vl if t - tl <= W else birth[v]
                gold_v = g[t][3:]
                g[t] = f"op:{ans}"
                # last_seen_any reads a CORRECT stream (gold value at every use): with its own answers re-emitted it
                # would echo the amended value through the distractor uses and score 1.0 on amend uses while wrong
                # on the distractors, a policy no net trained on every position can take.
                slot_last[v], any_last = (t, ans), gold_v
            elif k in "fx":
                if k == "f":
                    birth[v[0]] = v[1]
                else:
                    last_x = v[1]
                slot_last[v[0]], any_last = (t, v[1]), v[1]
        rows.append([SYM.index(s) for s in g])
    return rows


RULES = ["const", "never_update", "last_seen_any", "last_x_else_birth", "recent4_else_birth", "recent8_else_birth",
         "oracle"]


def baselines(data):
    from collections import Counter
    const_val = Counter(s[3:] for t in data.sets["train"] for s, (k, _) in zip(gold_syms(t), t) if k == "r").most_common(1)[0][0]
    out = {"const_val": const_val, "world": data.world}
    for rule in RULES:
        out[rule] = {k: score(data.sets[k], baseline_rows(data.sets[k], rule, const_val)) for k in NAMES}
    return out


# ------------------------------------------------------------------------------------------------- report (Amendment G)
ARMS = AM.ARMS
VARIANTS = ("stress2", "stress3", "stress4")
PRIMARY = ("test_xlong", "post_amend_all")       # Amendment G: per amend event, the amended-slot use AND the other-slot
CO_PRIMARY = ("test_xlong", "post_amend_all_addr_ok")   # use(s) of its segment all right; co-primary: address healthy
FALLBACK = None                               # the primary decides alone
LABEL = "G"
FLOOR = .064                                  # Amendment G note: best slot-blind baseline .014 + .05
SECONDARY = [CO_PRIMARY, ("test_xlong", "post_amend_use/amended"), ("test_xlong", "post_amend_use/other"),
             ("test_xlong", "amend_first_use"), ("test_xlong", "filler_acc"), ("test_xlong", "eos_ok"),
             ("test_long", "post_amend_all"), ("test_xhold", "hold_long"), ("test_iid", "post_amend_all"),
             ("test_hold", "hold_long"), ("test_xxlong", "post_amend_all"), ("test_xxlong", "filler_acc"),
             ("test_xxhold", "hold_long")]
REPORT = "report54g_{world}.json"
TQ = 3.863                                    # t_{1-.05/12, 6}: family-wise over 3 variants x 2 refs (review m2)
MECH = {"rate": "t>=birth_open_rate", "ablation": "closed_after_births", "metric": "post_amend_all"}   # review m1


def n_birth(task):
    return sum(k == "f" for k, _ in task)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "baselines", "report"])
    ap.add_argument("--world", default="amendg", choices=list(CONFIGS))
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "show":
        data = Data(a.world)
        from collections import Counter
        for k, v in data.sets.items():
            ls = [len(t) for t in v]
            print(k, len(v), "len min/mean/max", min(ls), round(sum(ls) / len(ls), 1), max(ls),
                  "n_amend", dict(sorted(Counter(sum(k_ == "x" for k_, _ in t) for t in v).items())))
        print("max_out", data.max_out)
        tr = sorted(len(t) for t in data.sets["train"])
        print("train len p50/p90/p95/p99/max", [tr[int(q * (len(tr) - 1))] for q in (.5, .9, .95, .99, 1)])
        for sp in ("test_long", "test_xlong", "test_xxlong"):
            pos = [t for task in data.sets[sp] for t, m in enumerate(use_meta(task)) if m["kind"] == "use" and m["seg"] >= 0]
            print(sp, "post-amend use positions > train p95:", round(sum(p > tr[int(.95 * (len(tr) - 1))] for p in pos) / len(pos), 3),
                  "> train max:", round(sum(p > tr[-1] for p in pos) / len(pos), 3))
        for t in data.sets["train"][:2] + data.sets["test_xlong"][:1]:
            print(name_of(t))
            print(" ".join(gold_syms(t)))
        return
    if a.cmd == "baselines":
        data = Data(a.world)
        out = baselines(data)
        os.makedirs(a.out, exist_ok=True)
        json.dump(out, open(os.path.join(a.out, f"baselines54g_{a.world}.json"), "w"), indent=1)
        for rule in RULES:
            print(rule, {sp: (out[rule][sp].get("post_amend_all"), out[rule][sp].get("amend_first_use"),
                              out[rule][sp].get("post_amend_use/other"), out[rule][sp].get("hold_long")) for sp in NAMES})
        return
    AM.report(a.out, W=sys.modules[__name__], world=a.world)


if __name__ == "__main__":
    main()
