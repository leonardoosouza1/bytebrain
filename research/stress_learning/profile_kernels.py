"""List the GPU kernels and host syncs of one fast-engine cycle:
score a window of 64 questions, one train step (4 items), one rescore (1 item)."""

import argparse
import collections
import json
import random
from types import SimpleNamespace

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

import fast_learning as fast
import stress_learning as old

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rng = random.Random(0)
facts = old.make_facts(rng, 200)
stream = old.train_paraphrases(facts, old.make_stream(rng, 200, 2000, 1.0))
tab = fast.PairTable(AutoTokenizer.from_pretrained(old.MODEL), facts, "cuda")
eng = fast.Engine(SimpleNamespace(seed=0, dtype=a.dtype), "cuda")
win = tab.index(stream[:64])
batch = tab.index(stream[:4])
one = tab.index(stream[:1])


def cycle():
    s, h = eng.score(tab, win)
    s.tolist()
    eng.train_step(tab, batch, 1e-4)
    eng.score(tab, one)[0].item()


for _ in range(3):  # warm-up
    cycle()
torch.cuda.synchronize()
phases = {}
for name, fn in (("score_window64", lambda: eng.score(tab, win)[0].tolist()),
                 ("train_step4", lambda: eng.train_step(tab, batch, 1e-4)),
                 ("rescore1", lambda: eng.score(tab, one)[0].item())):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kern = collections.Counter()
    ktime = collections.Counter()
    syncs = 0
    for e in prof.events():
        dev = getattr(e, "device_type", None)
        if dev is not None and dev.name in ("CUDA", "PrivateUse1"):
            kern[e.name[:90]] += 1
            ktime[e.name[:90]] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if "Memcpy" in e.name and ("DtoH" in e.name or "D2H" in e.name) or e.name in ("hipStreamSynchronize", "cudaStreamSynchronize", "hipMemcpyWithStream"):
            syncs += 1
    cpu_total = sum(e.cpu_time_total for e in prof.events() if e.cpu_parent is None) / 1000
    gpu_total = sum(ktime.values()) / 1000
    phases[name] = {"kernel_launches": sum(kern.values()), "distinct_kernels": len(kern),
                    "gpu_busy_ms": round(gpu_total, 2), "host_wall_ms": round(cpu_total, 2),
                    "dtoh_or_sync_events": syncs,
                    "top_by_time": [(k, kern[k], round(ktime[k] / 1000, 3)) for k, _ in ktime.most_common(12)]}
print(json.dumps(phases, indent=1))
with open(f"{a.out}/kernels_{a.dtype}.json", "w") as fh:
    json.dump(phases, fh, indent=1)
