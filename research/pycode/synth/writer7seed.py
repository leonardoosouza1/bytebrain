"""7-planner-seed rerun of the byte writer battery (ledger #17/#18 rule-8 fix, audit #50). CPU only.

Written 2026-10-27 09:46 (from `date`), BEFORE any run.

Why: every bytewriter*/ result JSON used planner_seed 4 (one of two perfect planners out of 7); the plain seq2seq
baseline (plain_bytelm.py) used unselected seeds 0-2. Not comparable. Here: planner seeds 0..6, no selection,
writer seeds 0,1,2 (as the original), same recipe and steps (planner 1500, writer 1500, bytewriter.py code imported
unchanged). Reported per (arm, writer seed, planner seed):
  e2e   = planner(name) -> plan -> writer(op, digit) -> text -> bwrap exec + strict grade   (the real pipeline)
  gold  = writer-alone given the GOLD plan (planner replaced by the reference chain)       (writer failure only)
  plan_exact per planner seed                                                             (planner failure only)
Grading is cached by source string (same source -> same bwrap verdict), so only new sources are executed.
Equivalence gate: for planner 4 + d32 s0 the cached path must equal bytewriter.evaluate() exactly, and both must
reproduce bytewriter18/bytewriter_d32_s0.json (test_num pass1 0.86, all other splits 1.0) = positive control.

Baseline rows (gate 1): constant/`return arg0` bodies are not meaningful here (strict grade on composed list ops);
the reference floor is plain_bytelm (no address), len6 pass1 0.005/0.02/0.035 on seeds 0-2. Seeds 3-6 of it are
run separately with the unchanged CLI (plain_bytelm.py --seed 3..6) so that both families have 7 unselected seeds.

PREDICTION (mechanism: the writer is keyed by (op, digit) only, so it is independent of the planner; a wrong plan
almost never yields a strictly-correct function; failures multiply: e2e ~= plan_exact x writer_alone).
  plan_exact per planner seed = blockaddr gb-4 block_exact on disk (same recipe, same seeds):
    len6: .565 / 1 / .93 / .975 / 1 / .5125 / .0375  -> mean .717 (sd ~.37)
    len5 mean .863, len4 mean .982, len3 ~1.0, test_a 1.0, test_num 1.0
  gold (writer alone) = the old planner-4 numbers, since planner 4 is perfect:
    d32 fp:  1.0 on test_a, len3-6; test_num .86/.58/.72 (writer seeds 0/1/2)
    d32 ter: len6 .665/.60/.49 (mean ~.58), test_a ~.78
    d16 ter: 0 everywhere (writer loss ~.59, never learned)
  e2e mean over 7 planner seeds x 3 writer seeds:
    d32 fp:  test_a 1.0, test_num ~.72, len3 ~1.0, len4 ~.98, len5 ~.86, len6 ~.72 (+-.37 across planner seeds)
    d32 ter: len6 ~.42 (= .58 x .72), test_a ~.78
    d16 ter: 0
  => headline "100% on len3-6" does NOT survive as a mean; it survives only as "best planner seed"; the address
     still beats plain seq2seq on length (len6 ~.72 vs ~.02), because even planner seed 6 (.0375) is >= plain.
  Planner sd dominates on len5/len6; writer seed dominates on test_num (digit copy).

Usage (CPU only):
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 5 timeout 3600 \
    .venv-rocm/bin/python -u writer7seed.py --out OUT [--smoke]
"""

import argparse
import json
import os
import statistics as st
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import blockaddr as BA  # noqa: E402
import bytewriter as BW  # noqa: E402
from bridge_cpu import MAX_OUT, SYM, Data, blocks_of, parse  # noqa: E402
from synth2 import grade, head, name_of, nargs, run_many  # noqa: E402

SPLITS = ["test_a", "test_num", "len3", "len4", "len5", "len6"]
ARMS = {"d32": (32, False), "d32_ter": (32, True), "d16_ter": (16, True)}
CACHE = {}          # src -> (exec_ok, pass_ok)


def planner_plans(planner, tasks):
    planner.eval()
    plans = []
    for i in range(0, len(tasks), 256):
        for row in planner.greedy(BA.name_bytes(tasks[i:i + 256])).tolist():
            s = [SYM[j] for j in row]
            plans.append(parse(s[:s.index("<eos>") + 1] if "<eos>" in s else s))
    return plans


def gold_plans(tasks):
    return [parse(blocks_of(*t)) for t in tasks]


class GoldPlanner:
    """Planner stand-in for bytewriter.evaluate(): returns the reference chain for each name."""
    def __init__(self, data):
        self.m = {}
        for k in SPLITS:
            for t in data.sets[k]:
                self.m[name_of(*t)] = [SYM.index(s) for s in blocks_of(*t)]

    def eval(self):
        return self

    def greedy(self, x):
        rows = []
        for r in x.tolist():
            n = bytes(BA.BYTES[i - 1] for i in r if i > 0).decode()
            v = self.m[n]
            rows.append(v + [0] * (MAX_OUT - len(v)))
        return torch.tensor(rows)


def writer_texts(writer, keys):
    writer.eval()
    keys = sorted(keys)
    out = writer.write(torch.tensor([k[0] for k in keys]), torch.tensor([k[1] for k in keys])) if keys else []
    return dict(zip(keys, out))


def score(tasks, plans, texts):
    """same source construction as bytewriter.evaluate, grading cached by source."""
    srcs, text_ok = [], []
    for t, p in zip(tasks, plans):
        bl = BW.blocks_of_task(p) if p is not None else []
        written = [texts[BW.block_key(*b)] for b in bl]
        body = "".join("".join(f"    {l}\n" for l in w.split("\n") if l) for w in written)
        srcs.append(head(*t) + (body if bl else "    return None\n"))
        text_ok.append(bool(bl) and all(w == BW.block_text(*b) for w, b in zip(written, bl)))
    new = [(s, t) for s, t in zip(srcs, tasks) if s not in CACHE]
    if new:
        runs = run_many([(s, name_of(*t), nargs(t[0])) for s, t in new])
        sc = grade([t for _, t in new], [[s] for s, _ in new])
        for (s, _), o, g in zip(new, runs, sc):
            CACHE[s] = (o is not None and "!" not in o, g[0] == 1.0)
    n = len(tasks)
    return {"n": n, "exec": round(sum(CACHE[s][0] for s in srcs) / n, 4),
            "pass1": round(sum(CACHE[s][1] for s in srcs) / n, 4),
            "plan_exact": round(sum(p is not None and blocks_of(*p) == blocks_of(*t)
                                    for p, t in zip(plans, tasks)) / n, 4),
            "text_exact": round(sum(text_ok) / n, 4)}


def needed_keys(all_plans):
    ks = set()
    for plans in all_plans:
        for p in plans:
            if p is not None:
                ks |= {BW.block_key(*b) for b in BW.blocks_of_task(p)}
    return ks


def msd(xs):
    return (round(st.mean(xs), 4), round(st.stdev(xs), 4) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--planner-seeds", default="0,1,2,3,4,5,6")
    ap.add_argument("--writer-seeds", default="0,1,2")
    ap.add_argument("--arms", default="d32,d32_ter,d16_ter")
    ap.add_argument("--smoke", action="store_true", help="planner 4 + d32 s0 only, equivalence + positive control")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.smoke:
        a.planner_seeds, a.writer_seeds, a.arms = "4", "0", "d32"
    pseeds = [int(s) for s in a.planner_seeds.split(",")]
    wseeds = [int(s) for s in a.writer_seeds.split(",")]
    arms = a.arms.split(",")
    T0 = time.perf_counter()
    data = Data()
    sets = {k: data.sets[k] for k in SPLITS}
    gold = {k: gold_plans(v) for k, v in sets.items()}

    plans, planner_info, planners = {}, {}, {}
    for ps in pseeds:
        t0 = time.perf_counter()
        pl = BW.train_planner(data, ps)
        planners[ps] = pl
        torch.save(pl.state_dict(), f"{a.out}/planner_s{ps}.pt")
        plans[ps] = {k: planner_plans(pl, v) for k, v in sets.items()}
        planner_info[ps] = {k: round(sum(p is not None and blocks_of(*p) == blocks_of(*t)
                                         for p, t in zip(plans[ps][k], sets[k])) / len(sets[k]), 4) for k in SPLITS}
        print(json.dumps({"planner_seed": ps, "plan_exact": planner_info[ps],
                          "wall_s": round(time.perf_counter() - t0, 1)}), flush=True)
    keys = needed_keys([gold[k] for k in SPLITS] + [plans[ps][k] for ps in pseeds for k in SPLITS])

    rows = []
    for arm in arms:
        d, ter = ARMS[arm]
        for ws in wseeds:
            t0 = time.perf_counter()
            writer, pool, wloss = BW.train_writer(data.sets["train"], d, ter, ws)
            texts = writer_texts(writer, keys)
            tw = round(time.perf_counter() - t0, 1)
            g = {k: score(sets[k], gold[k], texts) for k in SPLITS}
            row = {"arm": arm, "writer_seed": ws, "writer_loss": wloss, "writer_train_s": tw,
                   "writer_params": sum(p.numel() for p in writer.parameters()), "gold": g, "e2e": {}}
            for ps in pseeds:
                row["e2e"][ps] = {k: score(sets[k], plans[ps][k], texts) for k in SPLITS}
            if a.smoke:
                t1 = time.perf_counter()
                ref = BW.evaluate(planners[4], writer, data, SPLITS)
                gref = BW.evaluate(GoldPlanner(data), writer, data, SPLITS)
                rec = json.load(open("/media/llm-data/universe-runs/bytebrain-pycode-20261022/"
                                     "bytewriter18/bytewriter_d32_s0.json"))["eval"]
                ms = ["exec", "pass1", "plan_exact", "text_exact"]
                row["control"] = {
                    "cached_eq_evaluate": all(row["e2e"][4][k][m] == ref[k][m] for k in SPLITS for m in ms),
                    "gold_eq_evaluate": all(g[k][m] == gref[k][m] for k in SPLITS for m in ms),
                    "eq_recorded_json": all(ref[k][m] == rec[k][m] for k in SPLITS for m in ms),
                    "evaluate_pass1": {k: ref[k]["pass1"] for k in SPLITS},
                    "recorded_pass1": {k: rec[k]["pass1"] for k in SPLITS},
                    "evaluate_wall_s": round(time.perf_counter() - t1, 1)}
            row["wall_s"] = round(time.perf_counter() - t0, 1)
            rows.append(row)
            json.dump(row, open(f"{a.out}/{'smoke_' if a.smoke else ''}{arm}_w{ws}.json", "w"), indent=1)
            print(json.dumps({"arm": arm, "ws": ws, "wloss": wloss, "wall_s": row["wall_s"],
                              "gold_pass1": {k: g[k]["pass1"] for k in SPLITS},
                              "e2e_pass1": {ps: {k: row["e2e"][ps][k]["pass1"] for k in SPLITS} for ps in pseeds},
                              **({"control": row["control"]} if a.smoke else {})}), flush=True)

    summ = {"planner_plan_exact": planner_info,
            "planner_plan_exact_mean_sd": {k: msd([planner_info[ps][k] for ps in pseeds]) for k in SPLITS}}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        summ[arm] = {
            "gold_pass1_mean_sd": {k: msd([r["gold"][k]["pass1"] for r in rs]) for k in SPLITS},
            "e2e_pass1_mean_sd": {k: msd([r["e2e"][ps][k]["pass1"] for r in rs for ps in pseeds]) for k in SPLITS},
            "e2e_exec_mean": {k: msd([r["e2e"][ps][k]["exec"] for r in rs for ps in pseeds])[0] for k in SPLITS},
            "e2e_pass1_by_planner": {ps: {k: round(st.mean(r["e2e"][ps][k]["pass1"] for r in rs), 4)
                                          for k in SPLITS} for ps in pseeds}}
    summ["total_wall_s"] = round(time.perf_counter() - T0, 1)
    summ["graded_sources"] = len(CACHE)
    json.dump(summ, open(f"{a.out}/{'smoke_' if a.smoke else ''}summary.json", "w"), indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
