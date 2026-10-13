"""Gate before the fast battery: does fast_learning reproduce the old loop?

Runs the old sequential `stress` arm (stress_learning.run_arm) and the fast
`wake` arm on the same facts/stream prefix and compares, per question, the
stress S and the event list. Also reports wall time of each (the speedup).
With --dtype fp16 it compares fast fp16 against fast fp32 instead.
"""

import argparse
import json
import os
import random
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import fast_learning as fast
import stress_learning as old

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--facts", type=int, default=200)
ap.add_argument("--questions", type=int, default=300)
ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rng = random.Random(a.seed)
facts = old.make_facts(rng, a.facts)
stream = old.train_paraphrases(facts, old.make_stream(rng, a.facts, a.questions, 1.0))
common = dict(seed=a.seed, theta=0.5, theta_stop=0.2, max_steps=3, lr=1e-4, always_lr=3e-4,
              lr_gain_cap=3.0, replay=3, eval_every=10**9, debug=False)
fast_args = SimpleNamespace(**common, matched_replay=7, sleep_every=10**9, sleep_steps=0,
                            sleep_batch=16, sleep_lr=1e-4, fade_theta=0.3, window=64,
                            window_min=4, dtype="fp32")
tab = fast.PairTable(AutoTokenizer.from_pretrained(old.MODEL), facts, "cuda")

if a.dtype == "fp32":
    ref = old.run_arm("stress", SimpleNamespace(**common), facts, stream, "cuda")
    ref_name, ref_wall = "old_seq_fp32", ref["curve"][-1]["elapsed_s"]
else:
    ref = fast.run_arm("wake", fast_args, facts, stream, tab)
    ref_name, ref_wall = "fast_fp32", ref["wall_s"]
fast_args.dtype = a.dtype
new = fast.run_arm("wake", fast_args, facts, stream, tab)

s_ref = [x[0] for x in ref["stream_nll_hit"]]
s_new = [x[0] for x in new["stream_nll_hit"]]
ev_ref = [(e["t"], e["fid"], len(e["lrs"])) for e in ref["events"]]
ev_new = [(e["t"], e["fid"], len(e["lrs"])) for e in new["events"]]
first_div = next((i for i, (x, y) in enumerate(zip(ev_ref, ev_new)) if x != y), None)
# compare S only up to the first diverging event (after it, weights differ legitimately)
upto = ev_ref[first_div][0] if first_div is not None else len(stream)
d = [abs(x - y) for x, y in zip(s_ref[:upto], s_new[:upto])]
flips = sum(1 for x, y in zip(s_ref[:upto], s_new[:upto]) if (x < 0.5) != (y < 0.5))
report = {
    "reference": ref_name, "candidate": f"fast_{a.dtype}", "questions": len(stream),
    "events_ref": len(ev_ref), "events_new": len(ev_new),
    "identical_event_prefix": first_div if first_div is not None else min(len(ev_ref), len(ev_new)),
    "compared_questions": upto, "max_abs_dS": max(d) if d else None,
    "mean_abs_dS": sum(d) / len(d) if d else None, "theta_flips": flips,
    "heldout_ref": ref["curve"][-1]["acc_heldout"], "heldout_new": new["curve"][-1]["acc_heldout"],
    "wall_ref_s": ref_wall, "wall_new_s": new["wall_s"],
    "speedup": round(ref_wall / max(new["wall_s"], 1e-9), 2),
    "peak_mem_gb_new": new["peak_mem_gb"],
}
print(json.dumps(report, indent=1))
with open(os.path.join(a.out, f"equivalence_{a.dtype}_seed{a.seed}.json"), "w") as fh:
    json.dump(report, fh, indent=1)
