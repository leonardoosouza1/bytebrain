"""Lane engine: many independent learners (arm x seed) share one GPU forward.

Every lane is an independent LoRA adapter on the same frozen Qwen3-0.6B. The
arm logic of each lane is a Python generator that mirrors fast_learning.run_arm
step by step and *yields* requests instead of calling the model:

    ("score", items)       -> (S list, hit list) with the lane's current weights
    ("train", items, lr)   -> pre-update (S, hit); one AdamW step of that lane

Each tick the scheduler takes the single pending request of every lane, runs
all score rows in one no-grad pass and all train rows in one forward/backward,
then steps only the lanes that trained. One host read per tick returns the
results. Lanes never see each other: each row multiplies only its own lane's
LoRA block (one-hot mask), and the optimizer touches only active lanes.

Per-lane semantics are those of the sequential loop; only float rounding of
batched kernels differs (gate: lanes_gate.py).
"""

import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fast_learning import PairTable
from stress_learning import MODEL, make_facts, make_stream, train_paraphrases

ARMS = ("never", "always", "wake", "random", "wake_replay", "sleep_uniform", "sleep_fading")
GATED = ("wake", "wake_replay", "sleep_uniform", "sleep_fading")
TARGETS = (("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj"),
           ("self_attn", "o_proj"), ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"))


class Ctx:
    onehot = None  # [rows, lanes] float, set before every forward


class LaneLoRA(torch.nn.Module):
    """y = W x + scale * B_lane(A_lane x), all lanes in two GEMMs."""

    def __init__(self, base, n_lanes, r, alpha):
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.scale = alpha / r
        self.A = torch.nn.Parameter(torch.zeros(n_lanes, base.in_features, r, device=base.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(n_lanes, r, base.out_features, device=base.weight.device))

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        oh = Ctx.onehot
        if oh is None:
            return y
        nl, i, r = self.A.shape
        h = x @ self.A.permute(1, 0, 2).reshape(i, nl * r)                      # [b, L, nl*r]
        h = (h.view(*h.shape[:-1], nl, r) * oh[:, None, :, None]).view(h.shape)  # keep own lane only
        return y + self.scale * (h @ self.B.reshape(nl * r, -1))


class LaneEngine:
    def __init__(self, n_lanes, device="cuda", r=16, alpha=32, max_rows=16, score_rows=128):
        self.n, self.device, self.r, self.max_rows = n_lanes, device, r, max_rows
        self.score_rows = score_rows
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
        model.requires_grad_(False)
        self.mods = []
        for layer in model.model.layers:
            for parent, name in TARGETS:
                owner = getattr(layer, parent)
                m = LaneLoRA(getattr(owner, name), n_lanes, r, alpha)
                setattr(owner, name, m)
                self.mods.append(m)
        model.eval()
        self.model, self.body = model, model.model
        self.head_w = model.get_output_embeddings().weight
        self.params = [p for m in self.mods for p in (m.A, m.B)]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.steps = [0] * n_lanes

    @torch.no_grad()
    def init_lane(self, k, seed):
        """Fresh adapter: A ~ kaiming_uniform(a=sqrt(5)) like peft, B = 0; fresh optimizer."""
        g = torch.Generator().manual_seed(seed)
        for m in self.mods:
            i = m.A.shape[1]
            bound = 1.0 / math.sqrt(i)
            a = torch.empty(self.r, i).uniform_(-bound, bound, generator=g)
            m.A[k].copy_(a.t())
            m.B[k].zero_()
        for t in self.m + self.v:
            t[k].zero_()
        self.steps[k] = 0

    def _stats(self, tab, idx, lanes, Lb):
        Ctx.onehot = F.one_hot(lanes, self.n).float()
        try:
            h = self.body(input_ids=tab.ids[idx, :Lb], attention_mask=tab.att[idx, :Lb],
                          use_cache=False).last_hidden_state
        finally:
            Ctx.onehot = None
        pos, tgt, valid = tab.pos[idx], tab.tgt[idx], tab.valid[idx]
        hs = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        logits = F.linear(hs, self.head_w)
        tok_nll = (torch.logsumexp(logits, -1) - logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)) * valid
        hit = ((logits.argmax(-1) == tgt) | ~valid).all(1)
        nll_sum = tok_nll.sum(1)
        return nll_sum, nll_sum / tab.n_tok[idx], hit

    def run_tick(self, tab, lens, score_rows, train_rows, lane_lr):
        """score_rows / train_rows: lists of (pair_index, lane). Returns flat S, hit
        for score rows then train rows (train values are pre-update)."""
        outs_s, outs_h = [], []
        with torch.no_grad():
            for i in range(0, len(score_rows), self.score_rows):
                chunk = score_rows[i:i + self.score_rows]
                idx = torch.tensor([p for p, _ in chunk], device=self.device)
                ln = torch.tensor([k for _, k in chunk], device=self.device)
                _, s, h = self._stats(tab, idx, ln, max(lens[p] for p, _ in chunk))
                outs_s.append(s)
                outs_h.append(h)
        if train_rows:
            ntok = {}
            for p, k in train_rows:
                ntok[k] = ntok.get(k, 0) + tab.n_tok_host[p]
            for i in range(0, len(train_rows), self.max_rows):
                chunk = train_rows[i:i + self.max_rows]
                idx = torch.tensor([p for p, _ in chunk], device=self.device)
                ln = torch.tensor([k for _, k in chunk], device=self.device)
                w = torch.tensor([1.0 / ntok[k] for _, k in chunk], device=self.device)
                nll_sum, s, h = self._stats(tab, idx, ln, max(lens[p] for p, _ in chunk))
                (nll_sum * w).sum().backward()  # = sum over lanes of each lane's token-mean loss
                outs_s.append(s.detach())
                outs_h.append(h)
            self._adamw(lane_lr)
        if not outs_s:
            return [], []
        s = torch.cat(outs_s)
        h = torch.cat(outs_h).float()
        both = torch.stack([s, h]).tolist()  # the one host read of the tick
        return both[0], [bool(x) for x in both[1]]

    @torch.no_grad()
    def _adamw(self, lane_lr, b1=0.9, b2=0.999, eps=1e-8):
        """torch AdamW (foreach, wd=0) + clip_grad_norm_(1.0), applied per active lane."""
        for k, lr in lane_lr.items():
            g = [p.grad[k] for p in self.params]
            norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(g, 2.0)), 2.0)
            torch._foreach_mul_(g, torch.clamp(1.0 / (norm + 1e-6), max=1.0))
            m = [t[k] for t in self.m]
            v = [t[k] for t in self.v]
            p = [t.data[k] for t in self.params]
            self.steps[k] += 1
            bc1 = 1 - b1 ** self.steps[k]
            bc2 = 1 - b2 ** self.steps[k]
            torch._foreach_lerp_(m, g, 1 - b1)
            torch._foreach_mul_(v, b2)
            torch._foreach_addcmul_(v, g, g, 1 - b2)
            denom = torch._foreach_sqrt(v)
            torch._foreach_div_(denom, math.sqrt(bc2))
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(p, m, denom, -lr / bc1)
        for p in self.params:
            p.grad = None


def arm_proc(arm, args, facts, stream, schedule=None):
    """Generator with the exact logic of fast_learning.run_arm, yielding requests."""
    rng = random.Random(args.seed * 1000 + 7)
    seen = [0] * len(facts)
    buffer, buffer_set = [], set()
    events, sleeps, curve, nll_log = [], [], [], []
    c = {"wake_steps": 0, "sleep_steps": 0, "item_grads": 0, "scored": 0,
         "consults": 0, "served_ok": 0, "forwards": 0}
    st = {"last_eval": None}
    replay_n = args.matched_replay if arm == "wake_replay" else args.replay
    t0 = time.time()

    random_plan = {}
    if arm == "random":
        times = rng.sample(range(len(stream)), len(schedule))
        for t, ev in zip(sorted(times), schedule):
            random_plan[t] = ev

    def remember(fid):
        if fid not in buffer_set:
            buffer_set.add(fid)
            buffer.append(fid)

    def replay_items(fid, para):
        items = [(fid, para)]
        if buffer:
            for _ in range(replay_n):
                rf = rng.choice(buffer)
                items.append((rf, rng.choice(facts[rf]["train_para"])))
        return items

    def train(items, lr, kind):
        res = yield ("train", items, lr)
        c[kind] += 1
        c["item_grads"] += len(items)
        return res

    def stress_event(t, fid, para, s):
        remember(fid)
        lrs, s_now = [], s
        for _ in range(args.max_steps):
            lr = args.lr * min(s_now / args.theta, args.lr_gain_cap)
            yield from train(replay_items(fid, para), lr, "wake_steps")
            lrs.append(lr)
            (s_now,), _ = yield ("score", [(fid, para)])
            c["forwards"] += 1
            if s_now < args.theta_stop:
                break
        events.append({"t": t, "fid": fid, "s_before": s, "s_after": s_now, "lrs": lrs})

    def sleep_phase(t):
        if not buffer:
            return
        if arm == "sleep_uniform":
            for _ in range(args.sleep_steps):
                fids = [rng.choice(buffer) for _ in range(args.sleep_batch)]
                yield from train([(f, rng.choice(facts[f]["train_para"])) for f in fids],
                                 args.sleep_lr, "sleep_steps")
            sleeps.append({"t": t, "steps": args.sleep_steps})
            return
        items = [(f, rng.choice(facts[f]["train_para"])) for f in buffer]
        s_all, _ = yield ("score", items)
        c["scored"] += len(items)
        fading = sorted([x for x in zip(items, s_all) if x[1] > args.fade_theta], key=lambda x: -x[1])
        n_steps = min(args.sleep_steps, -(-len(fading) // args.sleep_batch))
        queue = [it for it, _ in fading]
        for k in range(n_steps):
            batch = queue[k * args.sleep_batch:(k + 1) * args.sleep_batch]
            while len(batch) < args.sleep_batch:
                f = rng.choice(buffer)
                batch.append((f, rng.choice(facts[f]["train_para"])))
            yield from train(batch, args.sleep_lr, "sleep_steps")
        sleeps.append({"t": t, "buffer": len(buffer), "fading": len(fading), "steps": n_steps,
                       "mean_fading_s": (sum(s for _, s in fading) / len(fading)) if fading else 0.0})

    def evaluate(full):
        seen_f = [f for f in range(len(facts)) if seen[f] > 0]
        splits = ("train_para", "heldout_para") if full else ("heldout_para",)
        items = [(f, p, sp) for sp in splits for f in seen_f for p in facts[f][sp]]
        _, hits = yield ("score", [(f, p) for f, p, _ in items])
        out = {"n_seen": len(seen_f)}
        total = sum(seen[f] for f in seen_f)
        for sp in splits:
            per = {}
            for (f, _, s2), h in zip(items, hits):
                if s2 == sp:
                    per.setdefault(f, []).append(h)
            acc = {f: sum(v) / len(v) for f, v in per.items()}
            out[sp] = {"acc": sum(acc.values()) / len(acc),
                       "acc_freq_weighted": sum(acc[f] * seen[f] for f in seen_f) / total,
                       "per_fact": acc}
        return out

    def boundary(t):
        n = t + 1
        last = n == len(stream)
        if (n % args.eval_every == 0 and arm != "never") or last:
            ev = yield from evaluate(full=last)
            relearned = len(events) - len({e["fid"] for e in events})
            row = {"t": n, **{k: v for k, v in c.items() if k != "served_ok"},
                   "relearn_events": relearned, "served_acc": c["served_ok"] / n,
                   "n_seen": ev["n_seen"], "acc_train": ev.get("train_para", {}).get("acc"),
                   "acc_heldout": ev["heldout_para"]["acc"],
                   "acc_heldout_fw": ev["heldout_para"]["acc_freq_weighted"],
                   "elapsed_s": round(time.time() - t0, 1)}
            curve.append(row)
            print(json.dumps({"arm": arm, "seed": args.seed, **row}), flush=True)
            st["last_eval"] = ev
        if arm.startswith("sleep") and n % args.sleep_every == 0 and n < len(stream):
            yield from sleep_phase(n)

    period = math.gcd(args.eval_every, args.sleep_every)
    t = 0
    window = args.window
    while t < len(stream):
        if arm == "always":
            fid, para = stream[t]
            seen[fid] += 1
            remember(fid)
            s, hit = yield from train(replay_items(fid, para), args.always_lr, "wake_steps")
            nll_log.append((round(s[0], 4), hit[0]))
            c["served_ok"] += int(hit[0])
            yield from boundary(t)
            t += 1
            continue
        end = min(t + window, (t // period + 1) * period, len(stream))
        s_win, h_win = yield ("score", stream[t:end])
        c["forwards"] += 1
        cut = None
        for i in range(end - t):
            q = t + i
            fid, para = stream[q]
            s, hit = s_win[i], h_win[i]
            seen[fid] += 1
            nll_log.append((round(s, 4), hit))
            knows = s < args.theta
            if arm == "never" or not knows:
                c["consults"] += 1
                c["served_ok"] += 1
            else:
                c["served_ok"] += int(hit)
            if arm in GATED and not knows:
                yield from stress_event(q, fid, para, s)
                cut = i
            elif arm == "random" and q in random_plan:
                remember(fid)
                for lr in random_plan[q]["lrs"]:
                    yield from train(replay_items(fid, para), lr, "wake_steps")
                events.append({"t": q, "fid": fid, "lrs": random_plan[q]["lrs"]})
                cut = i
            yield from boundary(q)
            if cut is not None:
                break
        if cut is None:
            t = end
            window = min(args.window, window * 2)
        else:
            t = t + cut + 1
            window = max(args.window_min, min(args.window, 2 * (cut + 1)))

    stressed = {e["fid"] for e in events}
    counts = {}
    for e in events:
        counts[e["fid"]] = counts.get(e["fid"], 0) + 1
    first_t = {}
    for e in events:
        first_t.setdefault(e["fid"], e["t"])
    early = [f for f, ft in first_t.items() if ft < len(stream) // 2]
    final_heldout = st["last_eval"]["heldout_para"]["per_fact"]
    return {
        "arm": arm, "seed": args.seed, "curve": curve, "events": events, "sleeps": sleeps,
        **{k: v for k, v in c.items() if k != "served_ok"},
        "served_acc": c["served_ok"] / len(stream), "stressed_facts": len(stressed),
        "relearned_fraction": sum(1 for v in counts.values() if v > 1) / max(1, len(stressed)),
        "retention_early_heldout": (sum(final_heldout.get(f, 0.0) for f in early) / len(early)) if early else None,
        "stream_nll_hit": nll_log, "final_per_fact_heldout": final_heldout,
        "final_per_fact_train": st["last_eval"]["train_para"]["per_fact"],
        "lane_wall_s": round(time.time() - t0, 1),
    }


def schedule_lanes(eng, tab, lens, lanes, log_every=200):
    """lanes: list of (lane_index, generator). Runs all to completion, returns results."""
    pending, results = {}, {}
    for k, gen in lanes:
        pending[k] = (gen, next(gen))
    ticks, rows_total, t0 = 0, 0, time.time()
    while pending:
        score_rows, train_rows, lane_lr, spans = [], [], {}, {}
        for k, (gen, req) in pending.items():
            if req[0] == "score":
                spans[k] = ("s", len(score_rows), len(req[1]))
                score_rows += [(tab.pair_index(f, p), k) for f, p in req[1]]
            else:
                spans[k] = ("t", len(train_rows), len(req[1]))
                train_rows += [(tab.pair_index(f, p), k) for f, p in req[1]]
                lane_lr[k] = req[2]
        s, h = eng.run_tick(tab, lens, score_rows, train_rows, lane_lr)
        ns = len(score_rows)
        for k in list(pending):
            gen, _ = pending[k]
            kind, a, n = spans[k]
            off = a if kind == "s" else ns + a
            try:
                pending[k] = (gen, gen.send((s[off:off + n], h[off:off + n])))
            except StopIteration as stop:
                results[k] = stop.value
                del pending[k]
        ticks += 1
        rows_total += len(score_rows) + len(train_rows)
        if ticks % log_every == 0:
            print(json.dumps({"tick": ticks, "lanes_alive": len(pending), "rows_per_tick":
                              round(rows_total / ticks, 1), "elapsed_s": round(time.time() - t0, 1),
                              "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}),
                  flush=True)
    return results, {"ticks": ticks, "rows_total": rows_total, "wall_s": round(time.time() - t0, 1),
                     "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def add_host_tables(tab):
    tab.n_tok_host = tab.n_tok.tolist()
    tab.pair_index = lambda f, p: f * tab.n_para + p
    return [int(x) for x in tab.att.sum(1).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--facts", type=int, default=200)
    ap.add_argument("--questions", type=int, default=2000)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-stop", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--always-lr", type=float, default=3e-4)
    ap.add_argument("--lr-gain-cap", type=float, default=3.0)
    ap.add_argument("--replay", type=int, default=3)
    ap.add_argument("--matched-replay", type=int, default=7)
    ap.add_argument("--sleep-every", type=int, default=100)
    ap.add_argument("--sleep-steps", type=int, default=8)
    ap.add_argument("--sleep-batch", type=int, default=16)
    ap.add_argument("--sleep-lr", type=float, default=1e-4)
    ap.add_argument("--fade-theta", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--window-min", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=16)
    ap.add_argument("--arms", default="wake,always,wake_replay,sleep_uniform,sleep_fading")
    ap.add_argument("--random", action="store_true", help="second phase: random arm per seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = [int(x) for x in args.seeds.split(",")]
    arms = args.arms.split(",")
    for a in arms:
        assert a in ARMS and a != "random", a
    tok = AutoTokenizer.from_pretrained(MODEL)
    setups = {}
    for seed in seeds:
        rng = random.Random(seed)
        facts = make_facts(rng, args.facts)
        stream = train_paraphrases(facts, make_stream(rng, args.facts, args.questions, args.zipf))
        setups[seed] = (facts, stream)
        with open(os.path.join(args.out, f"seed{seed}_setup.json"), "w") as fh:
            json.dump({"args": vars(args), "facts": facts, "stream": stream}, fh)

    combos = [(seed, arm) for seed in seeds for arm in arms]
    eng = LaneEngine(len(combos), max_rows=args.max_rows)
    # one PairTable per seed (facts differ); lanes of a seed share it
    tabs = {s: PairTable(tok, setups[s][0], "cuda") for s in seeds}
    lens = {s: add_host_tables(tabs[s]) for s in seeds}
    summary = {}

    def run_phase(phase_combos, name, schedules=None):
        by_seed = {}
        for k, (seed, arm) in enumerate(phase_combos):
            eng.init_lane(k, seed)
            a = argparse.Namespace(**{**vars(args), "seed": seed})
            facts, stream = setups[seed]
            gen = arm_proc(arm, a, facts, stream, schedules.get(seed) if schedules else None)
            by_seed.setdefault(seed, []).append((k, gen))
        results, stats = schedule_multi(eng, tabs, lens, by_seed)
        for k, (seed, arm) in enumerate(phase_combos):
            res = results[k]
            res["args"] = {**vars(args), "seed": seed}
            with open(os.path.join(args.out, f"seed{seed}_{arm}.json"), "w") as fh:
                json.dump(res, fh)
        summary[name] = stats
        print(json.dumps({"phase": name, **stats}), flush=True)

    run_phase(combos, "main")
    if args.random:
        schedules = {}
        for seed in seeds:
            with open(os.path.join(args.out, f"seed{seed}_wake.json")) as fh:
                schedules[seed] = [{"lrs": e["lrs"]} for e in json.load(fh)["events"]]
        run_phase([(s, "random") for s in seeds], "random", schedules)
    with open(os.path.join(args.out, "timing.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


class MergedTable:
    """Concatenate per-seed PairTables so lanes of different seeds share a tick."""

    def __init__(self, tabs):
        self.offsets, off = {}, 0
        for s, t in tabs.items():
            self.offsets[s] = off
            off += t.ids.shape[0]
        def cat(name):
            ts = [getattr(t, name) for t in tabs.values()]
            if ts[0].dim() == 1:
                return torch.cat(ts)
            w = max(t.shape[1] for t in ts)  # seeds differ in max length; pad right
            return torch.cat([F.pad(t, (0, w - t.shape[1])) for t in ts])
        self.ids, self.att, self.pos = cat("ids"), cat("att"), cat("pos")
        self.tgt, self.valid, self.n_tok = cat("tgt"), cat("valid"), cat("n_tok")
        self.n_tok_host = self.n_tok.tolist()
        self.n_para = next(iter(tabs.values())).n_para


def schedule_multi(eng, tabs, lens, by_seed):
    mt = MergedTable(tabs)
    mlens = [x for s in tabs for x in lens[s]]
    lanes = []
    for seed, lst in by_seed.items():
        off = mt.offsets[seed]
        for k, gen in lst:
            lanes.append((k, _offset_gen(gen, off, mt.n_para)))
    return schedule_lanes(eng, _TabView(mt), mlens, lanes)


class _TabView:
    """Scheduler sees pair indices already offset by seed (items arrive as (row, -1))."""

    def __init__(self, mt):
        self.mt = mt
        for k in ("ids", "att", "pos", "tgt", "valid", "n_tok", "n_tok_host", "n_para"):
            setattr(self, k, getattr(mt, k))

    def pair_index(self, row, _):
        return row


def _offset_gen(gen, off, n_para):
    """Translate (fid, para) items of one seed into merged-table rows."""
    req = next(gen)
    while True:
        tr = [(off + f * n_para + p, -1) for f, p in req[1]]
        out = (req[0], tr) if req[0] == "score" else (req[0], tr, req[2])
        res = yield out
        try:
            req = gen.send(res)
        except StopIteration as stop:
            return stop.value


if __name__ == "__main__":
    main()
