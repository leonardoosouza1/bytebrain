"""S1: stress-gated learning where the WORLD is the judge (Qwen3-0.6B + LoRA).

The model predicts the printed output of small generated Python snippets.
Ground truth comes from really executing each snippet in a sandbox
(bwrap: no network, no writable filesystem, whitelisted builtins, 1 s alarm),
not from a curated answer key. Stress S = mean NLL of the executor's output
tokens; the training rule is the 10/13 one (lr ~ S, up to 3 steps, replay 3).

Snippets come from template families:
  train families   - appear in the stream (new numbers / words every time)
  new-structure    - compositions never shown in the stream (loop with a
                     product, slice then sum, reverse then upper, while, ...)
Eval sets: fresh snippets of the train families ("in_family") and of the
new-structure families ("new_struct"); none of their code strings occurs in
the stream.

Arms (same stream, fresh LoRA each): never, always, stress, random
(random replays the stress schedule - event count, steps, lrs - at random
stream positions).
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

from fast_learning import Engine
from stress_learning import MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("never", "always", "stress", "random")

SYL = ["ba", "ce", "di", "fo", "gu", "ka", "le", "mi", "no", "pu", "ra", "se", "ti", "vo"]


def word(r):
    return "".join(r.choice(SYL) for _ in range(r.randint(2, 3)))


def ints(r, n, lo=1, hi=20):
    return [r.randint(lo, hi) for _ in range(n)]


TRAIN = {
    "add_mul": lambda r: f"a = {r.randint(1, 20)}\nb = {r.randint(1, 9)}\nprint(a + b * {r.randint(2, 9)})",
    "loop_sum": lambda r: f"s = {r.randint(0, 30)}\nfor i in range({r.randint(2, 12)}):\n    s += i\nprint(s)",
    "list_index": lambda r: f"xs = {ints(r, 5)}\nprint(xs[{r.randint(0, 4)}])",
    "str_reverse": lambda r: f'print("{word(r)}"[::-1])',
    "str_len": lambda r: f'print(len("{word(r)}"))',
    "list_sum": lambda r: f"xs = {ints(r, 4)}\nprint(sum(xs))",
    "max_of": lambda r: f"print(max({r.randint(1, 50)}, {r.randint(1, 50)}, {r.randint(1, 50)}))",
    "str_upper": lambda r: f'print("{word(r)}".upper())',
    "div_mod": lambda r: (lambda a, b: f"print({a} // {b}, {a} % {b})")(r.randint(10, 60), r.randint(2, 9)),
    "if_else": lambda r: (lambda a, b: f"x = {a}\nif x > {b}:\n    print(x - {b})\nelse:\n    print({b} - x)")(
        r.randint(1, 30), r.randint(1, 30)),
}

NEW_STRUCT = {
    "loop_sum_mul": lambda r: f"s = {r.randint(0, 30)}\nfor i in range({r.randint(2, 8)}):\n    s += i * {r.randint(2, 5)}\nprint(s)",
    "slice_sum": lambda r: f"xs = {ints(r, 4)}\nprint(sum(xs[1:]))",
    "reverse_upper": lambda r: f'print("{word(r)}"[::-1].upper())',
    "len_times": lambda r: f'print(len("{word(r)}") * {r.randint(2, 9)})',
    "while_count": lambda r: (lambda n, k: f"n = {n}\nc = 0\nwhile n > 0:\n    n -= {k}\n    c += 1\nprint(c)")(
        r.randint(5, 30), r.randint(2, 6)),
    "sorted_index": lambda r: f"xs = {ints(r, 5)}\nprint(sorted(xs)[{r.randint(0, 4)}])",
    "argmax": lambda r: f"xs = {r.sample(range(1, 40), 5)}\nprint(xs.index(max(xs)))",
    "count_if": lambda r: (lambda xs, t: f"c = 0\nfor x in {xs}:\n    if x > {t}:\n        c += 1\nprint(c)")(
        ints(r, 5), r.randint(5, 15)),
}


def run_sandboxed(codes):
    """Execute snippets with no network, read-only /usr, no home, no /media."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
           "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
           "--proc", "/proc", "--dev", "/dev",
           "--ro-bind", os.path.join(HERE, "sandbox_worker.py"), "/w.py",
           "/usr/bin/python3", "-I", "-S", "/w.py"]
    p = subprocess.run(cmd, input=json.dumps(codes), capture_output=True, text=True,
                       timeout=30 + 2 * len(codes))
    if p.returncode != 0:
        raise RuntimeError(f"sandbox failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def gen_unique(r, families, n_per_family, taken):
    out = []
    for fam, fn in families.items():
        k = 0
        while k < n_per_family:
            c = fn(r)
            if c in taken:
                continue
            taken.add(c)
            out.append({"family": fam, "code": c})
            k += 1
    return out


def make_setup(seed, n_stream, n_in_family, n_new):
    r = random.Random(seed)
    taken = set()
    in_family = gen_unique(r, TRAIN, n_in_family, taken)
    new_struct = gen_unique(r, NEW_STRUCT, n_new, taken)
    stream, fams = [], list(TRAIN)
    while len(stream) < n_stream:
        fam = r.choice(fams)
        c = TRAIN[fam](r)
        if c in taken:
            continue  # stream may repeat itself, never the eval sets
        stream.append({"family": fam, "code": c})
    rows = stream + in_family + new_struct
    outs = run_sandboxed([x["code"] for x in rows])
    for x, o in zip(rows, outs):
        x["out"] = o
    bad = [x for x in rows if x["out"].startswith("ERROR") or not x["out"]]
    assert not bad, bad[:3]
    return stream, in_family, new_struct


class CodeTable:
    """Rows (code, output) tokenized once, right-padded, on the GPU (same layout as PairTable)."""

    def __init__(self, tok, rows, device, prefix=""):
        enc = []
        for x in rows:
            p = tok(f"{prefix}```python\n{x['code']}\n```\nOutput:", add_special_tokens=False)["input_ids"]
            t = tok(f" {x['out']}\n", add_special_tokens=False)["input_ids"]
            enc.append((p, t))
        L = max(len(p) + len(t) for p, t in enc)
        T = max(len(t) for _, t in enc)
        N = len(enc)
        ids = torch.full((N, L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((N, L), dtype=torch.long)
        pos = torch.zeros((N, T), dtype=torch.long)
        tgt = torch.zeros((N, T), dtype=torch.long)
        valid = torch.zeros((N, T), dtype=torch.bool)
        for i, (p, t) in enumerate(enc):
            n = len(p) + len(t)
            ids[i, :n] = torch.tensor(p + t)
            att[i, :n] = 1
            pos[i, :len(t)] = torch.arange(len(p) - 1, n - 1)
            tgt[i, :len(t)] = torch.tensor(t)
            valid[i, :len(t)] = True
        self.ids, self.att, self.pos = ids.to(device), att.to(device), pos.to(device)
        self.tgt, self.valid = tgt.to(device), valid.to(device)
        self.n_tok = self.valid.sum(1).float()
        self.device = device
        # The answer ends at a newline; the model may close it with any newline-led
        # token ("\n\n", "\n```"...), so the last position accepts all of them.
        last = torch.zeros((N, T), dtype=torch.bool)
        for i, (_, t) in enumerate(enc):
            last[i, len(t) - 1] = True
        self.last = last.to(device)
        nl = [i for i in range(len(tok)) if tok.decode([i]).startswith("\n")]
        self.nl_ids = torch.tensor(nl, device=device)

    def index(self, rows):
        return torch.tensor(rows, device=self.device)


@torch.no_grad()
def evaluate(eng, tab, splits, rows):
    out = {}
    for name, (lo, hi) in splits.items():
        s, hit = eng.score(tab, list(range(lo, hi)))
        hit, s = hit.tolist(), s.tolist()
        fam = {}
        for i, h in zip(range(lo, hi), hit):
            fam.setdefault(rows[i]["family"], []).append(h)
        out[name] = {"acc": sum(hit) / len(hit), "mean_s": sum(s) / len(s),
                     "per_family": {f: sum(v) / len(v) for f, v in fam.items()}}
    return out


class CodeEngine(Engine):
    """Engine whose exact-match hit accepts any newline-led token as the answer terminator."""

    def _answer_stats(self, tab, idx):
        ctx = torch.autocast("cuda", torch.float16) if self.fp16 else torch.autocast("cuda", enabled=False)
        with ctx:
            h = self.body(input_ids=tab.ids[idx], attention_mask=tab.att[idx]).last_hidden_state
        pos, tgt, valid, last = tab.pos[idx], tab.tgt[idx], tab.valid[idx], tab.last[idx]
        hs = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        logits = torch.nn.functional.linear(hs.float(), self.head_w.float())
        logp = torch.log_softmax(logits, -1)
        tok_nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * valid
        pred = logits.argmax(-1)
        tok_hit = torch.where(last, torch.isin(pred, tab.nl_ids), pred == tgt) | ~valid
        n = tab.n_tok[idx]
        return tok_nll.sum(1) / n, tok_hit.all(1), tok_nll.sum(), n.sum()


class Throttled:
    """Steady GPU brake, same rule as iara-semente's Freio: busy <= duty of wall time.

    Paid in short slices (one score batch or one train step, a few ms each), so the
    card sits at a steady average load instead of multi-second 99%/idle swings that
    cycle the temperature. Default 0.75 keeps the RX 6750 XT near 145 W of 200.
    hwmon power1_average is logged only to check the brake, never to steer it.
    """

    def __init__(self, eng, duty, bs):
        self.eng, self.duty, self.bs = eng, duty, bs
        self.sensor = next(iter(glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average")), None)
        self.watts = []

    def _pace(self, t0):
        torch.cuda.synchronize()
        busy = time.time() - t0
        if self.duty < 1.0:
            time.sleep(busy * (1.0 - self.duty) / self.duty)
        if self.sensor and len(self.watts) < 100000:
            with open(self.sensor) as fh:
                self.watts.append(int(fh.read()) / 1e6)

    def score(self, tab, idx):
        out_s, out_h = [], []
        for i in range(0, len(idx), self.bs):
            t0 = time.time()
            s, h = self.eng.score(tab, idx[i:i + self.bs], bs=self.bs)
            self._pace(t0)
            out_s.append(s)
            out_h.append(h)
        return torch.cat(out_s), torch.cat(out_h)

    def train_step(self, tab, idx, lr):
        t0 = time.time()
        out = self.eng.train_step(tab, idx, lr)
        self._pace(t0)
        return out

    def power_report(self):
        w = sorted(self.watts)
        if not w:
            return {}
        return {"power_mean_w": round(sum(w) / len(w), 1), "power_p95_w": w[int(0.95 * (len(w) - 1))],
                "power_max_w": w[-1]}


def run_arm(arm, args, rows, splits, tab, schedule=None):
    eng = Throttled(CodeEngine(args, "cuda"), args.duty, args.score_bs)
    rng = random.Random(args.seed * 1000 + 7)
    n = splits["stream"][1]
    buffer, events, curve, stream_log = [], [], [], []
    counters = {"train_steps": 0, "item_grads": 0}
    random_plan = {}
    if arm == "random":
        times = sorted(rng.sample(range(n), len(schedule)))
        random_plan = dict(zip(times, schedule))

    def train(i, lr):
        items = [i] + ([rng.choice(buffer) for _ in range(args.replay)] if buffer else [])
        out = eng.train_step(tab, tab.index(items), lr)
        counters["train_steps"] += 1
        counters["item_grads"] += len(items)
        return out

    def remember(i):
        if i not in buffer:
            buffer.append(i)

    def do_eval(t):
        ev = evaluate(eng, tab, {k: v for k, v in splits.items() if k != "stream"}, rows)
        row = {"t": t, **counters, "events": len(events),
               "in_family": ev["in_family"]["acc"], "new_struct": ev["new_struct"]["acc"],
               "in_family_s": ev["in_family"]["mean_s"], "new_struct_s": ev["new_struct"]["mean_s"],
               "elapsed_s": round(time.time() - t0, 1)}
        curve.append({**row, "per_family": {k: ev[k]["per_family"] for k in ev}})
        print(json.dumps({"arm": arm, "seed": args.seed, **row}), flush=True)

    t0 = time.time()
    do_eval(0)
    t = 0
    while t < n:
        end = min(t + args.window, (t // args.eval_every + 1) * args.eval_every, n)
        if arm == "always":
            end = t + 1
        s_win, h_win = eng.score(tab, list(range(t, end)))
        s_win, h_win = s_win.tolist(), h_win.tolist()
        cut = None
        for k in range(end - t):
            q = t + k
            s, hit = s_win[k], bool(h_win[k])
            stream_log.append((round(s, 4), hit))
            if arm == "always":
                remember(q)
                train(q, args.always_lr)
            elif arm == "stress" and s >= args.theta:
                remember(q)
                lrs, s_now = [], s
                for _ in range(args.max_steps):
                    lr = args.lr * min(s_now / args.theta, args.lr_gain_cap)
                    train(q, lr)
                    lrs.append(lr)
                    s_now = eng.score(tab, [q])[0].item()
                    if s_now < args.theta_stop:
                        break
                events.append({"t": q, "s_before": s, "s_after": s_now, "lrs": lrs,
                               "family": rows[q]["family"]})
                cut = k
            elif arm == "random" and q in random_plan:
                remember(q)
                for lr in random_plan[q]["lrs"]:
                    train(q, lr)
                events.append({"t": q, "lrs": random_plan[q]["lrs"], "family": rows[q]["family"]})
                cut = k
            if (q + 1) % args.eval_every == 0 or q + 1 == n:
                do_eval(q + 1)
            if cut is not None:
                break
        t = end if cut is None else t + cut + 1

    res = {"arm": arm, "curve": curve, "events": events, **counters,
           "stream_nll_hit": stream_log,
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "wall_s": round(time.time() - t0, 1)}
    del eng
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return res


def fewshot_prefix(n, taken):
    """n solved snippets, one per train family, disjoint from every stream/eval code."""
    if n == 0:
        return ""
    r = random.Random(987654)
    codes = []
    for fam in list(TRAIN)[:n]:
        c = TRAIN[fam](r)
        while c in taken:
            c = TRAIN[fam](r)
        codes.append(c)
    outs = run_sandboxed(codes)
    assert all(o and not o.startswith("ERROR") for o in outs), outs
    return "".join(f"```python\n{c}\n```\nOutput: {o}\n\n" for c, o in zip(codes, outs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stream", type=int, default=2000)
    ap.add_argument("--in-family", type=int, default=30, help="eval snippets per train family")
    ap.add_argument("--new-struct", type=int, default=40, help="eval snippets per new-structure family")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-stop", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--always-lr", type=float, default=3e-4)
    ap.add_argument("--lr-gain-cap", type=float, default=3.0)
    ap.add_argument("--replay", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--duty", type=float, default=1.0, help="max fraction of wall time the GPU is busy (1.0 = no brake; owner asked for continuous load, 2026-10-22)")
    ap.add_argument("--score-bs", type=int, default=32)
    ap.add_argument("--fewshot", type=int, default=0, help="solved train-family examples prepended to every prompt")
    ap.add_argument("--arms", default="never,stress,random,always")
    ap.add_argument("--setup-only", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stream, in_family, new_struct = make_setup(args.seed, args.stream, args.in_family, args.new_struct)
    rows = stream + in_family + new_struct
    a, b = len(stream), len(stream) + len(in_family)
    splits = {"stream": (0, a), "in_family": (a, b), "new_struct": (b, len(rows))}
    with open(os.path.join(args.out, f"seed{args.seed}_setup.json"), "w") as fh:
        json.dump({"args": vars(args), "rows": rows, "splits": splits}, fh)
    if args.setup_only:
        print(json.dumps({"rows": len(rows), "splits": splits, "examples": rows[:3] + new_struct[:3]}, indent=1))
        return
    prefix = fewshot_prefix(args.fewshot, {x["code"] for x in rows})
    if prefix:
        print(json.dumps({"fewshot_prefix": prefix}))
    tab = CodeTable(AutoTokenizer.from_pretrained(MODEL), rows, "cuda", prefix)

    schedule = None
    for arm in args.arms.split(","):
        assert arm in ARMS, arm
        if arm == "random":
            with open(os.path.join(args.out, f"seed{args.seed}_stress.json")) as fh:
                schedule = [{"lrs": e["lrs"]} for e in json.load(fh)["events"]]
        res = run_arm(arm, args, rows, splits, tab, schedule)
        res["args"] = vars(args)
        with open(os.path.join(args.out, f"seed{args.seed}_{arm}.json"), "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    sys.exit(main())
