"""Ledger #49: the s5 student understander on a bigger Qwen (Coder-1.5B / Coder-3B), same recipe as s5.

His words (10-27): "Cara pq não usa um modelo qwen maior do que 0.5B?"
Thin wrapper over student_understander.py. That file is imported by a running GPU queue (q7c), so it is NOT edited:
this module imports it, sets its global MODEL (read inside main() at call time, student_understander.py:167/169, so
the override takes effect) and calls its main(). Nothing else of the recipe changes.

Recipe = s5 (e2e/student_s5_s1, run_student_s5.log): --arm s5, 523 steps (= 1 epoch of 16755 pairs / batch 32),
--batch 32, --lr 2e-4, --rank 16 (alpha 32, all 7 linear modules), --data-seed 1 (same 16755 pairs for every run;
--seed only moves LoRA init + batch order, like arm G2 of #7). s5 ran with seed 1 = data seed 1 and no --steps;
the explicit --steps 523 --data-seed 1 gives the same pairs and the same step count.
What the wrapper changes, and why it does not change the math:
  - MODEL = Qwen2.5-Coder-{0.5B,1.5B,3B}-Instruct (the variable under test).
  - output name: student_s5_q{05,15,30}_s{seed} (s5 is student_s5_s1; no collision). The module builds the tag from
    f"student_{a.arm}..."; a.arm is replaced after parsing by a str that EQUALS "s5" (train_pairs' `arm == "s5"`,
    result.json "arm": "s5") but formats as "s5_q15". Eval runs (--lora) take the tag from the adapter dir, as before.
  - --mem-frac / --micro defaults per size (VRAM below). --micro only splits the 32-row batch for gradient
    accumulation (loss weighted by rows/32, student_understander.py:220): same gradient up to fp summation order.
  - from_pretrained(..., device_map={"": 0}) on GPU: weights go straight to VRAM (the 3B would otherwise sit
    ~6.2 GB in host RAM on top of ~3.5 GB process baseline, 15 GB box); then the module's .to("cuda") is a no-op.
  - the load is logged: {"q49_model": name, "params", "hidden_size", "layers"} = proof of which model ran.
--smoke with no visible GPU (CPU smoke only, never a number): torch.cuda.* calls become no-ops, "cuda" -> "cpu",
fp32 weights (bf16 matmul on Zen3 is slow). Training/eval code is the module's own.

VRAM (RX 6750 XT, 12.87e9 B; 80% rule = 10.3 GB incl. ~0.5 GB display + HIP context; training peak <= 9.6 GB).
Measured 0.5B s5: peak allocated 2.38 GB at micro 4 (result.json vram_peak_gb). Split:
  weights 0.494e9 x 2 B = 0.99 GB; LoRA 8.8e6 fp32 x 16 B (param+grad+2 Adam) = 0.14 GB;
  rest 1.25 GB = activations ~1.0 GB + target-only logits ~0.25 GB (loss_of: lm_head only on target positions).
  Activations scale per token per layer ~ 4*I + 14*H elements x layers (Qwen2 MLP gate/up/act/prod + norms/attn):
  0.5B 4*4864+14*896=32.0k x24 = 0.77M; 1.5B 4*8960+14*1536=57.3k x28 = 1.61M (x2.1); 3B 4*11008+14*2048=72.7k
  x36 = 2.62M (x3.4).
  1.5B micro 4: 1.544e9 x 2 = 3.09 + LoRA 18.5e6 x 16 = 0.30 + act 2.1 + logits 0.25 = 5.7 GB (+10% frag 6.3)
           -> --mem-frac 0.60 (cap 7.7 GB) -> <= 8.2 GB with display = 64%.
  3B   micro 4: 3.086e9 x 2 = 6.17 + LoRA 29.9e6 x 16 = 0.48 + act 3.4 + 0.25 = 10.3 GB -> does NOT fit.
  3B   micro 2: 6.17 + 0.48 + 1.7 + 0.13 = 8.5 GB (+5% frag 8.9) <= 9.6 -> --mem-frac 0.72 (cap 9.27 GB, + 0.5 GB
           = 9.8 GB = 76%). The cap makes an overrun an OOM in this process, not a spill: the queue script then
           retries the 3B at micro 1 (6.17 + 0.48 + 0.85 + 0.07 = 7.6 GB), same batch 32, same 523 steps.
  0.5B micro 4, --mem-frac 0.35 (= s5).
No gradient checkpointing (the module has none; adding it would change the code path, not needed under the caps).

PRE-REGISTERED PREDICTION (written 10-27 12:23, `date`), belt pass rate, mean of seeds 1-3, grader v1 (recorded):
  size   A (179, templated)   B (174, para/ rewrites)   v2pool (593, Sonnet rewrites)
  0.5B   97.8 (s5 seed 1)     85.6                      74.4        <- baseline, recorded
  1.5B   97.5-98.5            88-91                     78-82
  3B     97.5-98.5            89-93                     80-85
Why: A is saturated (s5 fails 3 of 4 `open` + 1 calls5; ceiling ~98.3 without the open split): no size can show
there. B and v2pool failures are free-phrasing reads (the same trained recipe loses 14 pts from A to B), which is
pretrained language competence, the thing that grows with size; the formula-tree format is learned equally by all
three in 523 steps (s5 A exact 97.8). Expected seed spread ~1-2 pts (G vs G2 in #7), so a size effect counts only
if it beats the 0.5B seed range on all 3 seeds (paired by seed). Refutation: 3B within the 0.5B seed range on B and
v2pool -> size is not the bottleneck at this recipe (then: steps/rank scale with size, or data).
Strict metrics reported next to v1: understander exact (raw JSON == gold), grader v2 pass_nonconst (drops the
constant-gold requests such as x119 = div(maxx(b,b), sub(b,b)) that any answer passes, #34 caveat 1) and
pass_exact_set_dispatch (no extra op dispatched, #34 caveat 3).

`summary` subcommand: table of every q49 run + s5 baseline (A/B/v2pool, v1/v2/exact, mean and paired deltas).
"""

import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_understander as SU  # noqa: E402

SIZES = {  # size: (model, tag, default micro, default mem-frac)
    "0.5": ("Qwen/Qwen2.5-Coder-0.5B-Instruct", "q05", 4, 0.35),
    "1.5": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "q15", 4, 0.60),
    "3": ("Qwen/Qwen2.5-Coder-3B-Instruct", "q30", 2, 0.72),
}
R = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
E2E = f"{R}/e2e"


class ArmTag(str):
    """Equal to the module's arm ("s5": data, result.json) but printed with the size in the run tag."""

    def __new__(cls, arm, label):
        s = super().__new__(cls, arm)
        s.label = label
        return s

    def __format__(self, spec):
        return format(self.label, spec)


def cpu_shim():
    """CPU smoke only: the module hardcodes dev = "cuda" and calls torch.cuda.*."""
    torch.cuda.set_per_process_memory_fraction = lambda *a, **k: None
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0


def cpu_device_shim():
    """CPU smoke only, installed AFTER from_pretrained (review 10-27: installed before the load, these patches made the
    loaded 0.5B score step-0 loss 6.11 instead of 2.6078 on the same batch; after the load the loss is unchanged)."""
    fix = (lambda d: "cpu" if (d == "cuda" or isinstance(d, torch.device) and d.type == "cuda") else d)

    def wrap(f):
        def g(self, *a, **k):
            a = tuple(fix(x) if isinstance(x, (str, torch.device)) else x for x in a)
            if "device" in k:
                k["device"] = fix(k["device"])
            return f(self, *a, **k)
        return g
    torch.Tensor.to = wrap(torch.Tensor.to)
    torch.nn.Module.to = wrap(torch.nn.Module.to)
    tt = torch.tensor
    torch.tensor = lambda *a, **k: tt(*a, **{**k, **({"device": fix(k["device"])} if "device" in k else {})})


def hook_loader(cpu):
    from transformers import AutoModelForCausalLM
    orig = AutoModelForCausalLM.from_pretrained

    def load(name, *a, **k):
        if cpu:
            k["dtype"] = torch.float32
        else:
            k.setdefault("device_map", {"": 0})
        m = orig(name, *a, **k)
        if cpu:
            cpu_device_shim()
        c = m.config
        print(json.dumps({"q49_model": name, "params": sum(p.numel() for p in m.parameters()),
                          "hidden_size": c.hidden_size, "layers": c.num_hidden_layers,
                          "dtype": str(next(m.parameters()).dtype),
                          "device": str(next(m.parameters()).device)}), flush=True)
        return m
    AutoModelForCausalLM.from_pretrained = load


def config_proof():
    """No forward pass: config + exact param count of every size on the meta device (no RAM)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    for size, (name, tag, micro, frac) in SIZES.items():
        c = AutoConfig.from_pretrained(name, cache_dir=SU.CACHE, local_files_only=True)
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(c)
        print(json.dumps({"size": size, "model": name, "tag": tag, "params": sum(p.numel() for p in m.parameters()),
                          "hidden_size": c.hidden_size, "layers": c.num_hidden_layers,
                          "intermediate": c.intermediate_size, "micro": micro, "mem_frac": frac}), flush=True)


def run(argv):
    size = argv[argv.index("--size") + 1]
    i = argv.index("--size")
    argv = argv[:i] + argv[i + 2:]
    model, tag, micro, frac = SIZES[size]
    SU.MODEL = model                                     # read by SU.main() at call time
    cpu = not torch.cuda.is_available()
    assert not cpu or "--smoke" in argv, "no GPU visible: only --smoke may run on CPU"
    if cpu:
        cpu_shim()
    if "--mem-frac" not in argv:
        argv += ["--mem-frac", str(frac)]
    if "--micro" not in argv:
        argv += ["--micro", str(micro)]
    assert "--extra" not in argv and "--arm" not in argv, "the s5 recipe: arm s5, no extra data"
    argv += ["--arm", "s5"]
    hook_loader(cpu)
    import argparse
    parse = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        ns = parse(self, *a, **k)
        ns.arm = ArmTag(ns.arm, f"{ns.arm}_{tag}")
        return ns
    argparse.ArgumentParser.parse_args = parse_args
    print(json.dumps({"q49_override": SU.MODEL, "argv": argv}), flush=True)
    sys.argv = [SU.__file__] + argv
    SU.main()


def summary(out):
    """Every q49 run dir + the s5 baseline (seed 1 = e2e/student_s5_s1: v1 = recorded belt tag student_s5,
    v2 = student_s5_rerun49_v2g from the same raws)."""
    sets = {"A": E2E, "B": f"{E2E}/para", "v2pool": f"{E2E}/v2pool"}

    def get(d, tag, v2tag=None):
        row = {}
        for k, sd in sets.items():
            try:
                row[k] = round(100 * json.load(open(f"{sd}/summary_{tag}.json"))["pass"], 1)
            except (OSError, KeyError):
                row[k] = None
            try:
                v2 = json.load(open(f"{sd}/summary_{v2tag or tag}_v2g.json"))
                row[k + "_v2_nonconst"] = round(100 * v2["pass_nonconst"], 1)
                row[k + "_v2_exactset"] = (round(100 * v2["pass_exact_set_dispatch"], 1)
                                           if v2.get("pass_exact_set_dispatch") is not None else None)
            except (OSError, KeyError, TypeError):
                row[k + "_v2_nonconst"] = row[k + "_v2_exactset"] = None
            try:
                row[k + "_exact"] = round(100 * json.load(open(f"{sd}/{d}/result.json"))["exact"], 1)
            except (OSError, KeyError):
                row[k + "_exact"] = None
        return row
    rows = {"q05": {1: get("student_s5_s1", "student_s5", "student_s5_rerun49")}}
    rows["q05"][1]["source"] = "recorded s5"
    for size, (model, tag, _, _) in SIZES.items():
        for d in sorted(glob.glob(f"{E2E}/student_s5_{tag}_s[0-9]")):
            sd = int(d.rsplit("_s", 1)[1])
            t = "q05_ctl" if (tag, sd) == ("q05", 1) else tag   # wrapper rerun of s5 = positive control
            rows.setdefault(t, {})[sd] = get(os.path.basename(d), f"student_s5_{tag}_s{sd}")
            try:
                r = json.load(open(f"{d}/result.json"))
                rows[t][sd].update(train_s=r["train_s"], vram_peak_gb=r["vram_peak_gb"])
            except (OSError, KeyError):
                pass
    keys = [k for k in rows["q05"][1] if k != "source"]
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None
    mean = {t: {k: avg(v.get(k) for v in by.values()) for k in keys} | {"n_seeds": len(by)}
            for t, by in rows.items()}
    paired = {t: {k: [round(by[s][k] - rows["q05"][s][k], 1) for s in sorted(by)
                      if s in rows["q05"] and by[s].get(k) is not None and rows["q05"][s].get(k) is not None]
                  for k in keys} for t, by in rows.items() if t != "q05"}
    # review 10-27: also pair against the 0.5B run of the SAME wrapper path (seed 1 = q05_ctl, seeds 2/3 = q05), so a
    # wrapper-path shift (device_map load, GPU non-determinism) cannot masquerade as a size effect on seed 1
    wrap = {s: v for s, v in rows["q05"].items() if s != 1} | ({1: rows["q05_ctl"][1]} if 1 in rows.get("q05_ctl", {})
                                                                 else {})
    paired_w = {t: {k: [round(by[s][k] - wrap[s][k], 1) for s in sorted(by)
                        if s in wrap and by[s].get(k) is not None and wrap[s].get(k) is not None]
                    for k in keys} for t, by in rows.items() if t not in ("q05", "q05_ctl")}
    res = {"runs": {t: {str(s): v for s, v in by.items()} for t, by in rows.items()}, "mean": mean,
           "paired_delta_vs_q05": paired, "paired_delta_vs_q05_wrapper": paired_w,
           "recorded_s5": {"A": 97.8, "B": 85.6, "v2pool": 74.4}}
    json.dump(res, open(out, "w"), indent=1)
    for t, by in rows.items():
        for s, v in sorted(by.items()):
            print(t, s, json.dumps(v))
    print("MEAN", json.dumps(mean))
    print("PAIRED", json.dumps(paired))
    print("PAIRED_WRAPPER", json.dumps(paired_w))


if __name__ == "__main__":
    if sys.argv[1:2] == ["config"]:
        config_proof()
    elif sys.argv[1:2] == ["summary"]:
        summary(sys.argv[2] if len(sys.argv) > 2 else f"{E2E}/q49_summary.json")
    else:
        run(sys.argv[1:])
