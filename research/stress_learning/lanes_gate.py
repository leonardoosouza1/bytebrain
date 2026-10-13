"""Gate for the lane engine (run before any lane battery).

1. lane 0 initialised with the peft adapter of fast_learning.Engine (seed 0):
   score 64 items, 6 identical AdamW steps, rescore -> differences vs peft.
2. isolation: training lane 0 leaves every other lane's params and Adam state
   bit-identical.
3. concurrency: lane 2 trained together with lane 1 in the same tick equals
   lane 0 trained alone on the same batch.
4. throughput: one tick (train 4 rows + score 1 row per lane) with 1 lane vs
   all lanes.
"""

import argparse
import json
import random
import time
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import fast_learning as fast
import lanes_learning as ln
import stress_learning as old

ap = argparse.ArgumentParser()
ap.add_argument("--lanes", type=int, default=15)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rng = random.Random(0)
facts = old.make_facts(rng, 200)
stream = old.train_paraphrases(facts, old.make_stream(rng, 200, 300, 1.0))
tok = AutoTokenizer.from_pretrained(old.MODEL)
tab = fast.PairTable(tok, facts, "cuda")
lens = ln.add_host_tables(tab)
ref = fast.Engine(SimpleNamespace(seed=0, dtype="fp32"), "cuda")
eng = ln.LaneEngine(a.lanes)
rep = {}


@torch.no_grad()
def copy_peft_to_lane(k):
    layers = ref.model.get_base_model().model.layers
    i = 0
    for layer in layers:
        for parent, name in ln.TARGETS:
            pm = getattr(getattr(layer, parent), name)
            eng.mods[i].A[k].copy_(pm.lora_A["default"].weight.t())
            eng.mods[i].B[k].copy_(pm.lora_B["default"].weight.t())
            i += 1
    for t in eng.m + eng.v:
        t[k].zero_()
    eng.steps[k] = 0


def lane_vs_ref_max_param_diff(k):
    layers = ref.model.get_base_model().model.layers
    d, i = 0.0, 0
    for layer in layers:
        for parent, name in ln.TARGETS:
            pm = getattr(getattr(layer, parent), name)
            d = max(d, (eng.mods[i].A[k] - pm.lora_A["default"].weight.t()).abs().max().item(),
                    (eng.mods[i].B[k] - pm.lora_B["default"].weight.t()).abs().max().item())
            i += 1
    return d


def tick(score=(), train=(), lrs=None):
    sr = [(tab.pair_index(f, p), k) for f, p, k in score]
    tr = [(tab.pair_index(f, p), k) for f, p, k in train]
    return eng.run_tick(tab, lens, sr, tr, lrs or {})


for k in range(a.lanes):
    eng.init_lane(k, 100 + k)
copy_peft_to_lane(0)
items = stream[:64]
s_ref = ref.score(tab, tab.index(items))[0].tolist()
s_lane, _ = tick(score=[(f, p, 0) for f, p in items])
rep["score64_max_dS"] = max(abs(x - y) for x, y in zip(s_ref, s_lane))

batch = stream[:4]
_, _, l_ref = ref.train_step(tab, tab.index(batch), 3e-4)
s_pre, _ = tick(train=[(f, p, 0) for f, p in batch], lrs={0: 3e-4})
rep["step1_param_max_diff"] = lane_vs_ref_max_param_diff(0)
for _ in range(5):
    ref.train_step(tab, tab.index(batch), 3e-4)
    tick(train=[(f, p, 0) for f, p in batch], lrs={0: 3e-4})
rep["step6_param_max_diff"] = lane_vs_ref_max_param_diff(0)
s_ref = ref.score(tab, tab.index(items[:16]))[0].tolist()
s_lane, _ = tick(score=[(f, p, 0) for f, p in items[:16]])
rep["after6_score16_max_dS"] = max(abs(x - y) for x, y in zip(s_ref, s_lane))
rep["after6_scores_ref_lane"] = [[round(x, 4) for x in s_ref[:4]], [round(x, 4) for x in s_lane[:4]]]

# 2. isolation
snap = [(p[1:].clone(), m[1:].clone(), v[1:].clone()) for p, m, v in zip(eng.params, eng.m, eng.v)]
tick(train=[(f, p, 0) for f, p in stream[4:8]], lrs={0: 1e-4})
rep["isolation_other_lanes_identical"] = all(
    torch.equal(p[1:], a0) and torch.equal(m[1:], b0) and torch.equal(v[1:], c0)
    for (p, m, v), (a0, b0, c0) in zip(zip(eng.params, eng.m, eng.v), snap))

# 3. concurrency: lane 0 alone vs lane 2 together with lane 1
for k in (0, 2):
    copy_peft_to_lane(k)
X = [(f, p) for f, p in stream[10:14]]
Y = [(f, p) for f, p in stream[20:28]]
tick(train=[(f, p, 0) for f, p in X], lrs={0: 2e-4})
tick(train=[(f, p, 2) for f, p in X] + [(f, p, 1) for f, p in Y], lrs={2: 2e-4, 1: 1e-4})
rep["concurrency_lane0_vs_lane2_max_diff"] = max(
    (p[0] - p[2]).abs().max().item() for p in eng.params)

print(json.dumps(rep, indent=1), flush=True)
del ref
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


# 4. throughput
def timed(n_lanes, reps=5):
    tr = [(f, p, k) for k in range(n_lanes) for f, p in stream[30:34]]
    sc = [(f, p, k) for k in range(n_lanes) for f, p in stream[40:41]]
    lrs = {k: 1e-4 for k in range(n_lanes)}
    tick(score=sc, train=tr, lrs=lrs)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        tick(score=sc, train=tr, lrs=lrs)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps


t1 = timed(1)
tn = timed(a.lanes)
rep["tick_1lane_s"] = round(t1, 3)
rep[f"tick_{a.lanes}lanes_s"] = round(tn, 3)
rep["per_lane_speedup"] = round(t1 * a.lanes / tn, 2)
rep["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
print(json.dumps(rep, indent=1))
with open(f"{a.out}/lanes_gate.json", "w") as fh:
    json.dump(rep, fh, indent=1)
