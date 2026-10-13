"""Sleep consolidation on top of stress-gated learning (Qwen3-0.6B + LoRA).

The stress battery (stress_learning.py, commit 38dbb3a) showed that stress
selects what to learn far better than chance, but 43-48% of stressed facts
were forgotten and relearned. This battery tests whether an offline "sleep"
phase reduces that forgetting.

Wake is identical in every arm: stress-gated learning with the calibrated
rule from the stress battery (lr = 1e-4 * min(S/0.5, 3), max 3 steps, stop
when S < 0.2, 3 replay items per step).

Arms:
  wake          - wake only (baseline, re-run with this code)
  wake_replay   - wake only, with more replay per wake step (--matched-replay),
                  sized to spend about as many item-gradients as sleep does.
                  Separates "sleep" from "simply more replay".
  sleep_uniform - wake + every --sleep-every questions, --sleep-steps batched
                  steps on facts drawn uniformly from the learned buffer.
  sleep_fading  - wake + every --sleep-every questions, score every learned
                  fact (stress on one train paraphrase) and train only on the
                  ones that are fading (S > --fade-theta), weakest first.
                  If nothing fades, the sleep is skipped (cost adapts).

Facts and stream are generated exactly as in stress_learning.py, so seeds
0-2 reuse the same 500 facts and 5000 questions; the always/random/never
results of the stress battery remain valid references.
"""

import argparse
import json
import os
import random
import time

import torch
from transformers import AutoTokenizer

from stress_learning import (MODEL, Encoder, answer_stats, evaluate, load_model,
                             make_facts, make_stream, train_step)

ARMS = ("wake", "wake_replay", "sleep_uniform", "sleep_fading")


@torch.no_grad()
def score_buffer(model, enc, facts, buffer, rng, bs=64):
    """Stress of every learned fact on one random train paraphrase."""
    items = [(f, rng.choice(facts[f]["train_para"])) for f in buffer]
    scores = []
    for i in range(0, len(items), bs):
        scores += answer_stats(model, enc, items[i:i + bs])[0]
    return list(zip(items, scores))


def run_arm(arm, args, facts, stream, device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = Encoder(tok, facts, device)
    model, opt = load_model(args.seed, device)
    rng = random.Random(args.seed * 1000 + 7)
    seen = [0] * len(facts)
    buffer, buffer_set = [], set()
    events, sleeps, curve = [], [], []
    wake_steps = sleep_steps = item_grads = scored = 0
    consults = served_ok = 0
    replay_n = args.matched_replay if arm == "wake_replay" else args.replay

    def replay_batch(fid, para):
        items = [(fid, para)]
        if buffer:
            for _ in range(replay_n):
                rf = rng.choice(buffer)
                items.append((rf, rng.choice(facts[rf]["train_para"])))
        return items

    def sleep_phase(t):
        nonlocal sleep_steps, item_grads, scored
        if not buffer:
            return
        if arm == "sleep_uniform":
            for _ in range(args.sleep_steps):
                batch = [(f, rng.choice(facts[f]["train_para"]))
                         for f in (rng.choice(buffer) for _ in range(args.sleep_batch))]
                train_step(model, opt, enc, batch, args.sleep_lr)
                sleep_steps += 1
                item_grads += len(batch)
            sleeps.append({"t": t, "steps": args.sleep_steps})
            return
        # sleep_fading
        scored_items = score_buffer(model, enc, facts, buffer, rng)
        scored += len(scored_items)
        fading = sorted([x for x in scored_items if x[1] > args.fade_theta],
                        key=lambda x: -x[1])
        n_steps = min(args.sleep_steps, -(-len(fading) // args.sleep_batch))
        queue = [item for item, _ in fading]
        for k in range(n_steps):
            batch = queue[k * args.sleep_batch:(k + 1) * args.sleep_batch]
            while len(batch) < args.sleep_batch:  # pad with uniform rehearsal
                f = rng.choice(buffer)
                batch.append((f, rng.choice(facts[f]["train_para"])))
            train_step(model, opt, enc, batch, args.sleep_lr)
            sleep_steps += 1
            item_grads += len(batch)
        sleeps.append({"t": t, "buffer": len(buffer), "fading": len(fading),
                       "steps": n_steps,
                       "mean_fading_s": (sum(s for _, s in fading) / len(fading)) if fading else 0.0})

    t0 = time.time()
    for t, (fid, para) in enumerate(stream):
        seen[fid] += 1
        with torch.no_grad():
            (s,), (hit,) = answer_stats(model, enc, [(fid, para)])
        knows = s < args.theta
        if knows:
            served_ok += int(hit)
        else:
            consults += 1
            served_ok += 1  # bank answer is correct by construction
            if fid not in buffer_set:
                buffer_set.add(fid)
                buffer.append(fid)
            lrs, s_now = [], s
            for _ in range(args.max_steps):
                lr = args.lr * min(s_now / args.theta, args.lr_gain_cap)
                batch = replay_batch(fid, para)
                train_step(model, opt, enc, batch, lr)
                lrs.append(lr)
                wake_steps += 1
                item_grads += len(batch)
                with torch.no_grad():
                    (s_now,), _ = answer_stats(model, enc, [(fid, para)])
                if s_now < args.theta_stop:
                    break
            events.append({"t": t, "fid": fid, "s_before": s, "s_after": s_now, "lrs": lrs})

        if (t + 1) % args.eval_every == 0 or t + 1 == len(stream):
            ev = evaluate(model, enc, facts, seen)
            relearned = len(events) - len({e["fid"] for e in events})
            row = {"t": t + 1, "wake_steps": wake_steps, "sleep_steps": sleep_steps,
                   "item_grads": item_grads, "scored": scored, "consults": consults,
                   "relearn_events": relearned,
                   "served_acc": served_ok / (t + 1), "n_seen": ev["n_seen"],
                   "acc_train": ev["train_para"]["acc"],
                   "acc_heldout": ev["heldout_para"]["acc"],
                   "acc_heldout_fw": ev["heldout_para"]["acc_freq_weighted"],
                   "elapsed_s": round(time.time() - t0, 1)}
            curve.append(row)
            print(json.dumps({"arm": arm, "seed": args.seed, **row}), flush=True)
            last_eval = ev

        # Sleep after the eval, so every arm is measured at the end of a wake period.
        if arm.startswith("sleep") and (t + 1) % args.sleep_every == 0 and t + 1 < len(stream):
            sleep_phase(t + 1)

    stressed = {e["fid"] for e in events}
    counts = {}
    for e in events:
        counts[e["fid"]] = counts.get(e["fid"], 0) + 1
    result = {
        "arm": arm, "curve": curve, "events": events, "sleeps": sleeps,
        "wake_steps": wake_steps, "sleep_steps": sleep_steps,
        "item_grads": item_grads, "scored": scored, "consults": consults,
        "stressed_facts": len(stressed),
        "relearned_fraction": sum(1 for v in counts.values() if v > 1) / max(1, len(stressed)),
        "final_per_fact_heldout": last_eval["heldout_para"]["per_fact"],
        "final_per_fact_train": last_eval["train_para"]["per_fact"],
    }
    del model, opt
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts", type=int, default=500)
    ap.add_argument("--questions", type=int, default=5000)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-stop", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-gain-cap", type=float, default=3.0)
    ap.add_argument("--replay", type=int, default=3)
    ap.add_argument("--matched-replay", type=int, default=7)
    ap.add_argument("--sleep-every", type=int, default=250)
    ap.add_argument("--sleep-steps", type=int, default=20)
    ap.add_argument("--sleep-batch", type=int, default=16)
    ap.add_argument("--sleep-lr", type=float, default=1e-4)
    ap.add_argument("--fade-theta", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--arms", default="sleep_fading,sleep_uniform,wake_replay,wake")
    ap.add_argument("--reference-setup", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    facts = make_facts(rng, args.facts)
    stream = make_stream(rng, args.facts, args.questions, args.zipf)
    if args.reference_setup:
        with open(args.reference_setup) as fh:
            ref = json.load(fh)
        assert ref["facts"] == facts and [tuple(x) for x in ref["stream"]] == stream, \
            "facts/stream differ from the stress battery reference"
    with open(os.path.join(args.out, f"seed{args.seed}_setup.json"), "w") as fh:
        json.dump({"args": vars(args), "facts": facts, "stream": stream}, fh)

    for arm in args.arms.split(","):
        assert arm in ARMS, arm
        res = run_arm(arm, args, facts, stream, "cuda")
        res["args"] = vars(args)
        with open(os.path.join(args.out, f"seed{args.seed}_{arm}.json"), "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main()
