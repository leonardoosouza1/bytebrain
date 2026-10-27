"""Ledger #56 / master queue I2 (#25): a mixture of understanders feeding the real belt.

His words (10-27, #56): "vai misturar modelo, vai misturar expert, vai botar o qwen 0.5 junto com o writer planner...
comece a juntar as coisas". (#25, 10-23: "isso já é uma junção de modules, mistura de experts, então da para encaixar
mais coisas".)

Why the understander: on v2pool all 152 belt failures of s5 have root = understander (ledger #7 row, 10-26).

Belt = the recorded one, imported, not edited: e2e_pipeline.check_spec -> bridge -> Planner("net", e2e/ckpt) ->
Writer("net") -> bwrap exec (E.run_belt), grader v1 (end_pass). Each arm hands the belt ONE raw spec per request.

Experts (priority order = cost, then creation time; fixed here, not tuned):
  0.5B LoRA (Qwen2.5-Coder-0.5B-Instruct, s5 recipe family, cached raws on A, B, v2pool):
    s5 (student_s5_s1), s5st (steps format, A1), sfG, sfTs, sfT, sfO (#7 selftrain arms; sfO = gold labels on pool F)
  bigger, trained (#49, other agent): student_s5_q15_s*/ and student_s5_q30_s*/ under e2e/; picked up by glob when
    their raws/lora exist (priority after the 0.5B experts). Not present at registration.
  q3b: Qwen2.5-Coder-3B-Instruct GGUF q8_0, PROMPTED (few-shot + json_schema, e2e_pipeline.understand_qwen, no
    send-back). Recorded: A = traces_q3b_net raw_spec; B = round-0 raw of traces_q3b_sb2 (hist[0] when a send-back
    fired, else the final raw: llama greedy T=0 is deterministic). v2pool and DEV: GPU job (llama-server).
Mixtures: M6 = the six 0.5B experts (cached on all 3 sets). M7 = M6 + q3b. MQ = s5 + every q15/q30 seed present.
Routers (all gold-free; `valid` = check_spec bridges the spec AND raises no flag; key = (world, sorted ops) for
dispatch, (world, prefix tuple) for expr; a group's raw = its highest-priority member's raw):
  vote      K calls. Largest group of valid keys; tie -> the group holding the highest-priority member; no valid -> the
            first expert's raw.
  agree     (b) sequential in priority order: call experts one by one; accept the first valid spec whose key equals an
            earlier valid key (cross-expert self-consistency). After K calls with no agreement: first valid, else e1.
            calls = experts called.
  conf      (c) K calls; among valid specs argmax mean token logprob of the expert's own greedy tokens (teacher-forced
            on the recorded raw = the generation's own logprobs; parity checked vs saved --token-scores rows). GPU.
  cascade   (d) s5 first; escalate to the bigger expert when s5 is not valid or conf_s5 < theta. If the escalated spec
            is not valid and s5's is, keep s5. d0 = theta -inf (check_spec only, no confidence, runs from cache).
            theta tuned on DEV (below), grid = s5 DEV conf quantiles 0..0.5 step 0.025, max DEV belt pass, ties ->
            lower escalation. calls = 1 + escalated.
Equal-budget baselines: best single expert at 1 call is chosen ON THE TEST SET ITSELF (post-hoc max, optimistic for
  the baseline, so conservative for any MoE claim); s5-SC@K = s5 greedy + (K-1) T=1 samples (top_k 0, top_p 1, no
  penalty), vote with the same key rule (tie -> greedy's group): the same K calls spent on the best-known single model.
  v2pool samples = #46's recorded full_k (saci_code/redo/rows.jsonl); A, B samples = GPU job. Cascade self-baseline:
  same trigger, escalation = s5's first sample (same call count).
oracle@K  a request passes if any expert of the mixture passes: an UPPER BOUND, labelled as such, never a result.
DEV split for theta (gate 4): pool F (selftrain7/pool.json, 587 Sonnet rewrites of s5 training pairs) is disjoint by
  construction from A, B and v2pool (no text, orig_text or gold shared; re-asserted here, overlaps dropped). The sf*
  arms TRAINED on pool F, so DEV is only used for cascades whose experts never saw it (s5, q3b, q15, q30).
Stats: per set, paired by request id, 95% percentile bootstrap of (arm - baseline), 10_000 resamples,
  numpy default_rng(0), the same index matrix for every difference of a set.
Gate 3 (where gold enters): only in `grade` (belt end_pass) and in DEV theta selection; routers read raw text and
  model logprobs only (grep: no `gold` in route_*).
Known before this registration (not blind): every expert's recorded belt pass on A/B/v2pool (A: s5 97.8 sfG 95.5
  sfO 88.8 sfT 88.3 sfTs 88.8 s5st 88.3 q3b 72.6; B: 85.6 89.1 89.1 88.5 86.8 79.9 q3b(sb2) 74.7; v2pool: 74.4 82.6
  93.9 90.9 88.5 77.2). The priority order and rules above are fixed by cost/creation time, not by these numbers.

REGISTRATION 10-27 12:30 (`date`), before any arm of this file was scored:
  Positive control (gate 5): the single-expert s5 arm through this harness = 175/179, 149/174, 441/593 (97.8 / 85.6 /
    74.4), and every expert's per-request end_pass equals its recorded trace (6 experts x 3 sets, 100%).
  P1 vote(M6): A 93-97, B 87-91, v2pool 88-93; vs post-hoc best single: A -1..-5, B -2..+2, v2pool -1..-5. Mechanism:
    5 of 6 experts share the s5 recipe and 4 share pool F labels, so the vote follows the sf* family (weak on A).
  P2 agree(M6): A 96-98, B 85-89, v2pool 76-85, calls/request 2.1-2.6 (s5 and s5st agree on most items, incl. wrong).
  P3 oracle@6 (bound): A 98.3-99.5, B 92-96, v2pool 95-98.
  P4 s5-SC@6 (equal budget): v2pool 74-77 (#7 arith: the sample vote moved ~4 labels).
  P5 conf(M6) (GPU): between vote and best single; v2pool >= vote - 2.
  P6 cascade d0 s5->q3b: A 97.8-98.3, B 85.6-87.5, escalation A <= 2%, B <= 8%. Tuned d(theta) s5->q3b: within +-3
    of s5 on B (q3b is weaker than s5 on its own; escalating valid-but-unsure s5 specs to it often loses).
  P7 headline: no router beats the post-hoc best single by a CI-clean margin on v2pool; the gain of the mixture is
    not needing to know which expert is best per set (vote/agree land near each set's best), not exceeding it.

Usage:
  moe_understander56.py route [--out DIR]          CPU: route from cache + recorded raws, run the belt, write table
  moe_understander56.py gen --phase hf|llama|all   GPU: token confidences, DEV greedy, s5 samples, q3b via llama-server
  --smoke: GPU hidden, tiny limits, CPU (never a number).
"""

import argparse
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e_pipeline as E  # noqa: E402

RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
E2E = f"{RUN}/e2e"
OUT = f"{RUN}/moe56"
CKPT = f"{E2E}/ckpt"
SET_DIRS = {"A": E2E, "B": f"{E2E}/para", "v2pool": f"{E2E}/v2pool"}
SETS = ("A", "B", "v2pool")
LORA = [("s5", "student_s5_s1"), ("s5st", "student_s5st_s1"), ("sfG", "student_s5sfG_s1"),
        ("sfTs", "student_s5sfTs_s1"), ("sfT", "student_s5sfT_s1"), ("sfO", "student_s5sfO_s1")]
BIG_BASE = {"q15": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "q30": "Qwen/Qwen2.5-Coder-3B-Instruct"}
LLAMA_BIN = "/media/llm-data/scratch/llama-hip-build/bin/llama-server"
GGUF_3B = "/media/llm-data/models/gguf/qwen2.5-coder-3b-instruct-q8_0.gguf"
LLAMA_PORT = 8156
K_SAMPLES = 6
NBOOT = 10_000


def jload(p):
    return json.load(open(p))


def jdump(x, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(x, open(tmp, "w"), ensure_ascii=False, indent=0)
    os.replace(tmp, p)

# ------------------------------------------------------------------------------------------------ data / experts


def big_experts():
    """[(name, tag, base model)] for every #49 run whose adapter exists, seed order; q15 before q30."""
    out = []
    for q in ("q15", "q30"):
        for d in sorted(glob.glob(f"{E2E}/student_s5_{q}_s[0-9]")):
            if os.path.isdir(f"{d}/lora"):
                tag = os.path.basename(d)
                out.append((f"{q}s{tag.rsplit('_s', 1)[1]}", tag, BIG_BASE[q]))
    return out


def dev_requests(out):
    """pool F -> belt requests (world, caps, prefix from the gold); drop anything shared with A, B or v2pool."""
    p = f"{out}/dev/requests.json"
    if os.path.exists(p):
        return jload(p)
    pool = jload(f"{RUN}/selftrain7/pool.json")
    others = [q for s in SETS for q in jload(f"{SET_DIRS[s]}/requests.json")]
    bad_t = {q["text"] for q in others} | {q.get("orig_text") for q in others if q.get("orig_text")}
    bad_g = {json.dumps(q["gold"], sort_keys=True) for q in others}
    reqs, dropped = [], {"text": 0, "gold": 0}
    for x in pool:
        if x["text"] in bad_t or x.get("orig_text") in bad_t:
            dropped["text"] += 1
            continue
        if json.dumps(x["gold"], sort_keys=True) in bad_g:
            dropped["gold"] += 1
            continue
        g = x["gold"]
        q = {"id": "f" + x["id"][1:], "world": g["main"], "split": "dev", "text": x["text"], "gold": g,
             "orig_text": x.get("orig_text")}
        if g["main"] == "dispatch":
            q["caps"] = list(g["calls"])
        else:
            prefix, why = E.parse_expr(g["expr"])
            assert prefix is not None, why
            q["caps"], q["prefix"] = ["formula"], list(prefix)
        reqs.append(q)
    jdump(reqs, p)
    jdump({"n_pool": len(pool), "n_dev": len(reqs), "dropped": dropped}, f"{out}/dev/provenance.json")
    return reqs


def requests_of(out, s):
    return dev_requests(out) if s == "DEV" else jload(f"{SET_DIRS[s]}/requests.json")


def recorded_raws(name, tag, s):
    """{id: {raw, ms}} recorded by an earlier eval, or None."""
    if name == "q3b":
        f = {"A": f"{E2E}/traces_q3b_net.jsonl", "B": f"{E2E}/para/traces_q3b_sb2.jsonl"}.get(s)
        if not f:
            return None
        R = {}
        for line in open(f):
            t = json.loads(line)
            raw = t["sendback"][0]["raw"] if t["sendback"] else t["raw_spec"]   # round 0 = plain prompted 3B
            R[t["id"]] = {"raw": raw, "ms": t["lat_ms"]["understander"] if not t["sendback"] else None}
        return R
    if s == "DEV":
        return None
    p = f"{SET_DIRS[s]}/{tag}/raws.json"
    return jload(p) if os.path.exists(p) else None


def recorded_trace(name, s):
    """recorded belt traces {id: end_pass} of an expert (positive control), or None."""
    tags = {"s5": "student_s5", "s5st": "student_s5st", "q3b": "q3b_net" if s == "A" else None}
    t = tags.get(name, f"student_{name.replace('sf', 's5sf', 1)}_s1" if name.startswith("sf") else None)
    if not t or s == "DEV":
        return None
    p = f"{SET_DIRS[s]}/traces_{t}.jsonl"
    if not os.path.exists(p):
        return None
    return {d["id"]: d["end_pass"] for d in map(json.loads, open(p))}


def load_expert(out, name, tag, s):
    """{id: {raw, ms, conf?}}: recorded raw wins; conf from the GPU cache (its raw must equal the recorded one)."""
    rec = recorded_raws(name, tag, s)
    cp = f"{out}/cache/{name}/{s}.json"
    cache = jload(cp) if os.path.exists(cp) else None
    if rec is None:
        return cache
    if cache:
        for i, r in rec.items():
            c = cache.get(i)
            if c and c["raw"] == r["raw"] and c.get("conf") is not None:
                r["conf"] = c["conf"]
            if r.get("ms") is None and c:
                r["ms"] = c.get("ms")
    return rec


def s5_samples(out, s):
    """{id: [K_SAMPLES raws at T=1]} or None."""
    if s == "v2pool":
        rows = [json.loads(l) for l in open(f"{RUN}/saci_code/redo/rows.jsonl")]
        return {r["id"]: [c for c, _ in r["cand"]["full_k"]][:K_SAMPLES] for r in rows}
    p = f"{out}/cache/s5/{s}_samples.json"
    return jload(p) if os.path.exists(p) else None

# ------------------------------------------------------------------------------------------------ routers (gold-free)


def key_of(raw):
    _, b, fl = E.check_spec(raw)
    if b is None or fl:
        return None
    w, t = b
    return (w, tuple(sorted(t))) if w == "dispatch" else (w, tuple(t))


def route_vote(cands):
    """cands: [(name, raw, conf)] in priority order -> raw."""
    groups = {}
    for n, (_, raw, _) in enumerate(cands):
        k = key_of(raw)
        if k is not None:
            groups.setdefault(k, []).append(n)
    if not groups:
        return cands[0][1]
    best = max(groups.values(), key=lambda m: (len(m), -m[0]))
    return cands[best[0]][1]


def route_agree(cands):
    """-> (raw, calls)."""
    seen = {}
    for n, (_, raw, _) in enumerate(cands):
        k = key_of(raw)
        if k is None:
            continue
        if k in seen:
            return cands[seen[k]][1], n + 1
        seen[k] = n
    first = min(seen.values()) if seen else 0
    return cands[first][1], len(cands)


def route_conf(cands):
    valid = [(c[2], -n, c[1]) for n, c in enumerate(cands) if key_of(c[1]) is not None]
    if not valid:
        return cands[0][1]
    assert all(v[0] is not None for v in valid), "conf router needs every expert's confidence"
    return max(valid)[2]


def route_cascade(small, big, theta):
    """small/big: (raw, conf) -> (raw, escalated)."""
    ok = key_of(small[0]) is not None
    if ok and (theta == -math.inf or (small[1] is not None and small[1] >= theta)):
        return small[0], False
    if key_of(big[0]) is None and ok:
        return small[0], True
    return big[0], True

# ------------------------------------------------------------------------------------------------ belt grading


class Grader:
    """end_pass of (set, id, raw) through the real belt; each unique pair is run once."""

    def __init__(self, out):
        self.P, self.W = E.Planner("net", CKPT), E.Writer("net", CKPT)
        self.memo, self.reqs, self.out = {}, {}, out

    def grade(self, s, pairs):
        if s not in self.reqs:
            self.reqs[s] = {q["id"]: q for q in requests_of(self.out, s)}
        todo = sorted({(i, r) for i, r in pairs if (s, i, r) not in self.memo})
        for j in range(0, len(todo), 400):
            chunk = todo[j:j + 400]
            T = E.run_belt([self.reqs[s][i] for i, _ in chunk], [r for _, r in chunk], self.P, self.W)
            for (i, r), t in zip(chunk, T):
                self.memo[(s, i, r)] = bool(t["end_pass"])
        return [self.memo[(s, i, r)] for i, r in pairs]


def boot_ci(diff, idx):
    m = diff[idx].mean(1)
    return [round(100 * float(np.percentile(m, 2.5)), 1), round(100 * float(np.percentile(m, 97.5)), 1)]

# ------------------------------------------------------------------------------------------------ route (CPU)


def tune_theta(out, G, big):
    """theta for s5 -> big on DEV (gold used here only). None when DEV caches are missing."""
    s5 = load_expert(out, "s5", "student_s5_s1", "DEV")
    bg = load_expert(out, *big[:2], "DEV")
    if not s5 or not bg or any(v.get("conf") is None for v in s5.values()):
        return None
    ids = sorted(set(s5) & set(bg))
    confs = np.array([s5[i]["conf"] for i in ids])
    grid = sorted({-math.inf} | {float(np.quantile(confs, q)) for q in np.arange(0.025, 0.501, 0.025)})
    best = None
    for th in grid:
        ch = [route_cascade((s5[i]["raw"], s5[i]["conf"]), (bg[i]["raw"], None), th) for i in ids]
        p = G.grade("DEV", [(i, r) for i, (r, _) in zip(ids, ch)])
        esc = sum(e for _, e in ch)
        cand = (sum(p), -esc, th)
        if best is None or cand > best:
            best = cand
    s5_pass = sum(G.grade("DEV", [(i, s5[i]["raw"]) for i in ids]))
    return {"theta": best[2], "dev_n": len(ids), "dev_pass": best[0], "dev_escalated": -best[1],
            "dev_s5_pass": s5_pass}


def cmd_route(a):
    t0 = time.time()
    G = Grader(a.out)
    big = big_experts()
    q3b = ("q3b", None, None)
    experts = [(n, t, None) for n, t in LORA] + big + [q3b]
    res = {"registered": "10-27 12:30", "experts": [e[0] for e in experts], "sets": {}, "pc": {}, "theta": {}}
    thetas = {}
    for b in big + [q3b]:
        thetas[b[0]] = tune_theta(a.out, G, b)
        res["theta"][b[0]] = thetas[b[0]]
    for s in SETS:
        reqs = requests_of(a.out, s)
        ids = [q["id"] for q in reqs][:a.limit or None]
        X = {e[0]: load_expert(a.out, e[0], e[1], s) for e in experts}
        X = {k: v for k, v in X.items() if v and all(i in v for i in ids)}
        arms, calls, esc = {}, {}, {}
        # singles + positive control (per request vs recorded trace)
        for n in X:
            arms[f"single:{n}"] = [X[n][i]["raw"] for i in ids]
            calls[f"single:{n}"] = 1.0
        mixes = {"M6": [n for n, _ in LORA if n in X]}
        if "q3b" in X:
            mixes["M7"] = mixes["M6"] + ["q3b"]
        bq = [b[0] for b in big if b[0] in X]
        if bq:
            mixes["MQ"] = ["s5"] + bq
            mixes["M6Q"] = mixes["M6"] + bq
        SC = s5_samples(a.out, s)
        for m, names in mixes.items():
            K = len(names)
            C = {i: [(n, X[n][i]["raw"], X[n][i].get("conf")) for n in names] for i in ids}
            arms[f"vote:{m}"] = [route_vote(C[i]) for i in ids]
            calls[f"vote:{m}"] = float(K)
            ag = [route_agree(C[i]) for i in ids]
            arms[f"agree:{m}"] = [r for r, _ in ag]
            calls[f"agree:{m}"] = round(float(np.mean([c for _, c in ag])), 3)
            if all(c[2] is not None for i in ids for c in C[i]):
                arms[f"conf:{m}"] = [route_conf(C[i]) for i in ids]
                calls[f"conf:{m}"] = float(K)
            arms[f"oracle@{K}:{m}"] = None             # filled after grading
            if SC and all(i in SC and len(SC[i]) >= K - 1 for i in ids):
                arms[f"s5-SC@{K}"] = [route_vote([("s5", X["s5"][i]["raw"], None)] +
                                                 [("smp", r, None) for r in SC[i][:K - 1]]) for i in ids]
                calls[f"s5-SC@{K}"] = float(K)
        for b in [x for x in bq] + (["q3b"] if "q3b" in X else []):
            th = (thetas.get(b) or {}).get("theta")
            for tag, theta in (("d0", -math.inf), ("d", th)):
                if theta is None:
                    continue
                if tag == "d" and any(X["s5"][i].get("conf") is None for i in ids):
                    continue
                ch = [route_cascade((X["s5"][i]["raw"], X["s5"][i].get("conf")), (X[b][i]["raw"], None), theta)
                      for i in ids]
                arms[f"cascade-{tag}:s5->{b}"] = [r for r, _ in ch]
                esc[f"cascade-{tag}:s5->{b}"] = round(float(np.mean([e for _, e in ch])), 4)
                calls[f"cascade-{tag}:s5->{b}"] = round(1 + float(np.mean([e for _, e in ch])), 4)
                if SC and all(i in SC and SC[i] for i in ids):     # same trigger, escalation = s5's first sample
                    chs = [route_cascade((X["s5"][i]["raw"], X["s5"][i].get("conf")), (SC[i][0], None), theta)
                           for i in ids]
                    arms[f"cascade-{tag}:s5->s5smp(trigger of {b})"] = [r for r, _ in chs]
                    calls[f"cascade-{tag}:s5->s5smp(trigger of {b})"] = calls[f"cascade-{tag}:s5->{b}"]
        # grade
        P = {}
        for arm, raws in arms.items():
            if raws is not None:
                P[arm] = np.array(G.grade(s, list(zip(ids, raws))), dtype=float)
        for m, names in mixes.items():
            P[f"oracle@{len(names)}:{m}"] = np.max([P[f"single:{n}"] for n in names], 0)
        pc = {}
        for n in X:
            tr = recorded_trace(n, s)
            if tr:
                same = sum(tr[i] == bool(p) for i, p in zip(ids, P[f"single:{n}"]))
                pc[n] = {"same": same, "n": len(ids), "pass_here": int(P[f"single:{n}"].sum()),
                         "pass_recorded": sum(tr[i] for i in ids)}
        res["pc"][s] = pc
        singles = [k for k in P if k.startswith("single:")]
        top = max(P[k].sum() for k in singles)
        bests = [k for k in singles if P[k].sum() == top]     # review 10-27: ties (B: sfG = sfO = 155) -> all of them
        best = bests[0]
        idx = np.random.default_rng(0).integers(0, len(ids), (NBOOT, len(ids)))
        rows = {}
        for arm in P:
            row = {"pass": round(100 * float(P[arm].mean()), 1), "n_pass": int(P[arm].sum()), "n": len(ids),
                   "calls": calls.get(arm), "escalation": esc.get(arm),
                   "upper_bound": arm.startswith("oracle")}
            d = P[arm] - P["single:s5"]
            row["vs_s5"] = {"d": round(100 * float(d.mean()), 1), "ci": boot_ci(d, idx)}
            vb = []                                           # least favourable of the tied best singles = headline
            for b in bests:
                d = P[arm] - P[b]
                vb.append({"d": round(100 * float(d.mean()), 1), "ci": boot_ci(d, idx), "base": b})
            row["vs_best_single"] = min(vb, key=lambda r: (r["ci"][0], r["d"]))
            if len(vb) > 1:
                row["vs_each_tied_best"] = vb
            K = int(calls[arm]) if arm.split(":")[0].startswith(("vote", "conf")) else None
            if K and f"s5-SC@{K}" in P:
                d = P[arm] - P[f"s5-SC@{K}"]
                row["vs_s5_SC_equal_budget"] = {"d": round(100 * float(d.mean()), 1), "ci": boot_ci(d, idx)}
            rows[arm] = row
        res["sets"][s] = {"n": len(ids), "best_single_posthoc": best, "best_single_tied": bests, "arms": rows}
        print(f"=== {s} n={len(ids)} best single (post-hoc) {bests}", flush=True)
        for arm, r in sorted(rows.items(), key=lambda kv: -kv[1]["pass"]):
            extra = f" vsSC {r['vs_s5_SC_equal_budget']}" if "vs_s5_SC_equal_budget" in r else ""
            print(f"  {arm:46s} {r['pass']:5.1f} ({r['n_pass']}/{r['n']}) calls {r['calls']} esc {r['escalation']}"
                  f" vs_s5 {r['vs_s5']} vs_best {r['vs_best_single']}{extra}{'  [UPPER BOUND]' if r['upper_bound'] else ''}",
                  flush=True)
        print(f"  PC per-request parity: {pc}", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    jdump(res, f"{a.out}/route{'_smoke' if a.limit else ''}.json")
    print(json.dumps({"theta": res["theta"], "wall_s": res["wall_s"]}), flush=True)

# ------------------------------------------------------------------------------------------------ gen (GPU)


def cpu_smoke_patch():
    """GPU hidden: make torch.cuda bookkeeping calls no-ops so the module's code path runs on CPU (smoke only)."""
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.set_per_process_memory_fraction = lambda *a, **k: None


class HF:
    def __init__(self, base, lora, dev, smoke):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import student_understander as SU
        self.SU, self.dev = SU, dev
        self.tok = AutoTokenizer.from_pretrained(base, cache_dir=SU.CACHE, local_files_only=True)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        m = AutoModelForCausalLM.from_pretrained(base, cache_dir=SU.CACHE, local_files_only=True,
                                                 dtype=torch.float32 if smoke else torch.bfloat16,
                                                 **({} if dev == "cpu" else {"device_map": {"": 0}}))
        self.model = PeftModel.from_pretrained(m, lora).merge_and_unload().eval()
        fmt = jload(os.path.join(os.path.dirname(os.path.normpath(lora)), "result.json")).get("spec_format", "expr")
        self.max_new = SU.MAX_NEW[fmt]

    @torch.no_grad()
    def score(self, text, raw):
        p = self.tok(self.SU.prompt(text), add_special_tokens=False).input_ids
        g = self.tok(raw, add_special_tokens=False).input_ids + [self.im_end]
        x = torch.tensor([p + g], device=self.dev)
        lp = self.model(input_ids=x).logits[0, len(p) - 1:len(p) + len(g) - 1].float().log_softmax(-1)
        c = lp.gather(-1, torch.tensor(g, device=self.dev)[:, None])[:, 0]
        return round(float(c.mean()), 5), round(float(c.min()), 5), len(g)

    def greedy(self, reqs):
        r = self.SU.evaluate(self.model, self.tok, self.im_end, reqs, self.dev, self.max_new, None, True)
        out = {}
        for i, v in r["raws"].items():
            lps = [t[1] for t in v["tok"]]
            out[i] = {"raw": v["raw"], "ms": v["ms"], "conf": round(float(np.mean(lps)), 5) if lps else None,
                      "minlp": round(float(min(lps)), 5) if lps else None, "n_tok": len(lps), "src": "gen"}
        return out

    @torch.no_grad()
    def samples(self, text, k, seed):
        torch.manual_seed(seed)
        ids = torch.tensor([self.tok(self.SU.prompt(text), add_special_tokens=False).input_ids], device=self.dev)
        o = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new,
                                do_sample=True, num_return_sequences=k, temperature=1.0, top_k=0, top_p=1.0,
                                repetition_penalty=1.0, eos_token_id=self.im_end, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(s[ids.shape[1]:], skip_special_tokens=True).strip() for s in o]


def gen_hf(a, dev):
    sets = list(SETS) + ["DEV"]
    lim = (lambda xs: xs[:a.limit]) if a.limit else (lambda xs: xs)
    todo = [(n, t, None) for n, t in LORA] + big_experts()
    import student_understander as SU
    log = {}
    for name, tag, base in todo:
        base = base or SU.MODEL
        dev_needed = name == "s5" or name[:3] in ("q15", "q30")      # DEV only for experts that never trained on it
        need = []
        for s in sets:
            if s == "DEV" and not dev_needed:
                continue
            cp = f"{a.out}/cache/{name}/{s}.json"
            if os.path.exists(cp) and not a.limit:
                continue
            need.append(s)
        smp_need = [s for s in ("A", "B") if name == "s5" and (a.limit or
                                                                not os.path.exists(f"{a.out}/cache/s5/{s}_samples.json"))]
        if not need and not smp_need:
            continue
        t0 = time.time()
        M = HF(base, f"{E2E}/{tag}/lora", dev, a.smoke)
        for s in need:
            reqs = lim(requests_of(a.out, s))
            rec = recorded_raws(name, tag, s)
            out, parity = {}, None
            if rec is None:                                   # no recorded raw: generate greedy with token scores
                out = M.greedy(reqs)
            else:                                             # teacher-force the recorded raw
                for q in reqs:
                    c, mn, n = M.score(q["text"], rec[q["id"]]["raw"])
                    out[q["id"]] = {"raw": rec[q["id"]]["raw"], "ms": rec[q["id"]]["ms"], "conf": c, "minlp": mn,
                                    "n_tok": n, "src": "forced"}
                chk = M.greedy(reqs[:a.parity_n])            # parity: greedy here == recorded greedy
                parity = {"same": sum(chk[i]["raw"] == rec[i]["raw"] for i in chk), "n": len(chk),
                          "conf_abs_diff_max": max((abs(chk[i]["conf"] - out[i]["conf"]) for i in chk
                                                    if chk[i]["raw"] == rec[i]["raw"]), default=None)}
            jdump(out, f"{a.out}/cache{'_smoke' if a.smoke else ''}/{name}/{s}.json")
            log[f"{name}/{s}"] = {"n": len(out), "parity": parity, "wall_s": round(time.time() - t0, 1)}
            print(json.dumps({f"{name}/{s}": log[f"{name}/{s}"]}), flush=True)
        for s in smp_need:
            reqs = lim(requests_of(a.out, s))
            S = {q["id"]: M.samples(q["text"], K_SAMPLES, 1000 + n) for n, q in enumerate(reqs)}
            jdump(S, f"{a.out}/cache{'_smoke' if a.smoke else ''}/s5/{s}_samples.json")
            print(json.dumps({f"s5/{s}_samples": len(S), "wall_s": round(time.time() - t0, 1)}), flush=True)
        del M
        if dev == "cuda":
            torch.cuda.empty_cache()
    return log


def llama_call(url, text):
    prompt = E.chat(E.U_SYS, [(u, json.dumps(x)) for u, x in E.U_SHOTS], text)
    body = {"prompt": prompt, "n_predict": 160, "temperature": 0, "top_k": 1, "cache_prompt": False,
            "stop": ["<|im_end|>"], "json_schema": E.SPEC_SCHEMA, "n_probs": 1}
    rq = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=600) as f:
        r = json.loads(f.read())
    lps = []
    for p in r.get("completion_probabilities") or []:
        if p.get("logprob") is not None:
            lps.append(float(p["logprob"]))
        elif p.get("prob") is not None:
            lps.append(math.log(max(p["prob"], 1e-12)))
    return r["content"], lps


def gen_llama(a):
    url = f"http://127.0.0.1:{LLAMA_PORT}"
    import socket                                         # review 10-27: a foreign server on the port would answer
    with socket.socket() as sk:
        sk.settimeout(2)
        if sk.connect_ex(("127.0.0.1", LLAMA_PORT)) == 0:
            raise RuntimeError(f"port {LLAMA_PORT} already in use: refusing to start (not our server)")
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.dirname(LLAMA_BIN) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    logf = open(f"{a.out}/llama_server.log", "w")
    srv = subprocess.Popen([LLAMA_BIN, "-m", GGUF_3B, "-ngl", "99", "-c", "8192", "-np", "1", "-t", "2",
                            "--host", "127.0.0.1", "--port", str(LLAMA_PORT)], stdout=logf, stderr=subprocess.STDOUT,
                           env=env, start_new_session=True)
    with open(f"{a.out}/llama_server.pid", "w") as f:     # the shell kills by this PID (process group), never pkill -f
        f.write(str(srv.pid))
    log = {}
    try:
        for _ in range(180):
            if srv.poll() is not None:
                raise RuntimeError(f"llama-server exited rc={srv.returncode} before healthy")
            try:
                with urllib.request.urlopen(url + "/health", timeout=5) as f:
                    if json.loads(f.read()).get("status") == "ok":
                        break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError("llama-server did not become healthy")
        for s in list(SETS) + ["DEV"]:
            cp = f"{a.out}/cache{'_smoke' if a.limit else ''}/q3b/{s}.json"
            if os.path.exists(cp) and not a.limit:
                continue
            reqs = requests_of(a.out, s)[:a.limit or None]
            out, t0 = {}, time.time()
            for q in reqs:
                t1 = time.perf_counter()
                try:
                    raw, lps = llama_call(url, q["text"])
                except OSError as e:
                    raw, lps = f"<error {e}>", []
                out[q["id"]] = {"raw": raw, "ms": round(1000 * (time.perf_counter() - t1), 1),
                                "conf": round(float(np.mean(lps)), 5) if lps else None,
                                "minlp": round(float(min(lps)), 5) if lps else None, "n_tok": len(lps), "src": "gen"}
            rec = recorded_raws("q3b", None, s)
            par = {"same": sum(out[i]["raw"] == rec[i]["raw"] for i in out), "n": len(out)} if rec else None
            jdump(out, cp)
            log[f"q3b/{s}"] = {"n": len(out), "parity_vs_recorded": par, "wall_s": round(time.time() - t0, 1)}
            print(json.dumps({f"q3b/{s}": log[f"q3b/{s}"]}), flush=True)
    finally:
        os.killpg(srv.pid, signal.SIGTERM)
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
    return log


def cmd_gen(a):
    # review 10-27: `timeout` sends SIGTERM; the default handler skips `finally`, orphaning llama-server.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if a.phase == "llama" and not a.smoke:       # HTTP client only: never touch torch.cuda (no HIP/KFD init here)
        log = {"llama": gen_llama(a)}
        jdump(log, f"{a.out}/gen_log_{a.phase}.json")
        return
    dev = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    if dev == "cpu":
        assert a.smoke, "gen without a GPU is a smoke only"
        cpu_smoke_patch()
    elif a.phase != "llama":                     # the llama phase is an HTTP client only: no CUDA context here
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    log = {}
    if a.phase in ("hf", "all"):
        log["hf"] = gen_hf(a, dev)
    if a.phase in ("llama", "all"):
        log["llama"] = gen_llama(a)
    if dev == "cuda" and a.phase != "llama":
        log["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    jdump(log, f"{a.out}/gen_log{'_smoke' if a.smoke else ''}_{a.phase}.json")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("route")
    r.add_argument("--out", default=OUT)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--threads", type=int, default=2)
    g = sp.add_parser("gen")
    g.add_argument("--out", default=OUT)
    g.add_argument("--phase", choices=["hf", "llama", "all"], default="all")
    g.add_argument("--mem-frac", type=float, default=0.75)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--parity-n", type=int, default=12)
    g.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(getattr(a, "threads", 2))
    os.makedirs(a.out, exist_ok=True)
    (cmd_route if a.cmd == "route" else cmd_gen)(a)


if __name__ == "__main__":
    main()
