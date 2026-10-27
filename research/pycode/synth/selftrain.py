"""ERRO as teacher without gold (ledger #7) / the score trains the generator (ledger #2): retrain the understander on
labels that come from its own error signal, not from the generator's gold.

Why the understander: on v2pool all 152 belt failures of student s5 have root = understander; planner and writer fail
0 times, so there is no planner/writer error to learn from there (ledger #7, 10-26 ~19:20).

Pool F (`paraphrase_requests.py --from-train 600 --seed 1 --name pool --exclude v2pool para A`): Sonnet rewrites of
s5's own training pairs (seed 1 = s5's train_pairs), with no text, orig_text or gold shared with v2pool, B (para) or
A (e2e requests). Same provenance as v2pool (rewrites of trained trees), disjoint by construction (gate 4).

Steps (gold enters only in `extras` for arm O and the label report, and in `verdict`; `labels` and `teacher` read the
pool through `texts_of`, which drops every field but id and text):
  labels   GPU. Student s5: greedy + k samples at T=1 on the full distribution (saci_redo.Student.sample, PC4 checked).
           Candidates = greedy + samples. Gold-free signals: bridge validity (check_spec) and the execution signature
           (each candidate spec's own reference module on the probe inputs, saci_redo.signatures).
  teacher  Sonnet (claude -p, serial, no tools, cwd <out>/sonnet_cwd, call cap): flagged items only; it sees the
           request text and the student's distinct valid candidate specs (shuffled, no counts), and answers the spec
           (one of them or its own). His design (10-25 voice note): "Se a do Queen já estiver boa, vai usar aquilo...
           Sonnet vê que errou... ele mesmo vai ajustar e passar para o modelo retreinar". The Qwen-3B first pass is
           left out: on these requests the student is stronger than Qwen-3B (A: s5 .978 vs plain 3B .726), so its
           candidates take the first-judge place.
  extras   label sets, one per arm:
             G  greedy spec, else the first valid sample (plain self-training: no error signal)
             V  execution vote: largest signature group (ties: the group holding greedy, then the earliest); inside
                the group the most frequent raw (ties: greedy, then the earliest)  [gold-free, self only; computed and
                reported, not trained: see REGISTRATION]
             T  flagged -> Sonnet's spec (invalid answer -> V); not flagged -> V             [gold-free, + teacher]
                [superseded by AMENDMENT F2/F3: fallback is G, not V; arm Ts = T labels on kept_self]
             O  generator gold                                                               [ceiling]
           flag = greedy invalid OR greedy outside the vote group OR vote group < 5 of k+1.
           G trains on the texts the student can label itself (kept_self); T and O on every kept text (kept_all): the
           teacher also labels texts with no valid student candidate, and that coverage is part of what T measures.
           A text whose label in ANY arm is dropped by student_understander's --extra filters (gold equal to a test
           or excluded gold, held-frame wording) is dropped from every arm.
  verdict  see REGISTRATION below.

REGISTRATION (10-26 ~22:11, before any pool label, training or eval exists; the only #7 numbers seen are v2pool-arith
on #46's recorded s5 candidates, selftrain7/v2pool_arith.json, no model trained):
  v2pool-arith (gate 7): n 593; G right 449, V right 453, right spec among the 9 candidates 464, G none 67, flagged
  112, flagged & G wrong 98 of 144. The vote moves ~4 labels, so a V arm cannot differ from G by a measurable amount:
  V is computed and reported, not trained (a measured negative for #2's self-only loop: when greedy is wrong, the
  right spec is among the student's own samples in 15 of 144).
  Gate 4: v2pool informed that cut (not the flag rule or any threshold, fixed before the arith ran); B (para, 174) is
  the set nothing here was chosen on and is reported next to v2pool.
  Arms (student_understander --arm s5 --steps 523 --extra extra_X.json --extra-reps 3 --exclude para v2pool
  --tag-suffix sfX; the baseline s5 = the same trainer, seed 1, 523 steps, no extra):
    G   seed 1, extra_G (kept_self)     G2  seed 2, extra_G (training-seed noise floor)
    T   seed 1, extra_T (kept_all)      O   seed 1, extra_O (kept_all, gold: ceiling)
    [AMENDMENT F1/F3: every arm --data-seed 1; arm Ts seed 1, extra_Ts (kept_self)]
  Metric: belt end_pass (e2e_pipeline run --understander file, grader v1) per request, paired by request id; 95% CI =
  percentile bootstrap, 10_000 resamples, numpy default_rng(0), the same indices for every difference of a set.
  Primary set v2pool (593). B (para, 174) and A (e2e requests, 179) reported, not judged.
  Gate 1 (the ceiling exists): O - G CI lower > 0 on v2pool, else NO VERDICT (gate 1).
  VERDICT on 'the teacher's error signal beats plain self-training' (T - G on v2pool):
    PASS             CI lower > 0 and T - G point > |G2 - G| point
    KILL             CI upper < +.02
    NOT ESTABLISHED  otherwise (names the noise floor when the CI alone held)
  Asserted: each arm's result.json has steps 523 and extra_dropped 0, seed-1 arms have train_pairs - 3 * extra equal to
  s5's 16755; each belt summary's raws_file is that arm's raws.json; the s5 belt rerun on v2pool gives 441 (gate 5).
  Predictions (gate 7): O - s5 on v2pool +4 pp (range +1..+8), on B +3.5 (0..+7); G - s5 within +-3 pp; T recovers
  .6-.9 of O - G; A may drop in every arm (s5p's cue 'main' => formula cost A -8.4).

AMENDMENT (10-26 ~22:22, Opus pre-run review; still before any pool label, training or eval exists):
  F1 G2 was not seed noise: train_pairs is seeded by --seed (phrasing + the 3000 s5 extra trees), so seed 2 also
     changed the data. student_understander gets --data-seed; EVERY trained arm runs --data-seed 1 (asserted).
  F2 T on unflagged items = G (was V), and a flagged item whose Sonnet answer is invalid -> G: T differs from G only
     where the teacher spoke (V vs G moved commutative swaps). extras reports unflagged_V_ne_G.
  F3 new arm Ts = T labels on kept_self (G's texts). Ts - G = relabelling alone (mechanism); T - Ts = coverage of the
     texts with no valid student candidate. The registered verdict stays on T - G; the same rule is applied to Ts - G
     and reported as the mechanism verdict. Prediction: Ts - G <= +1 pp (likely NOT ESTABLISHED); most of T - G is
     coverage (~68% of T's fixable labels are no-candidate texts).
  F4 TEACH named sqrt/neg/... as one-argument expr ops; parse_expr accepts in expr only add sub mul div mod maxx avg
     pct hyp with two arguments over a and b. Prompt fixed. Sonnet call timeout 600 -> 280 s, TimeoutExpired retried.
  F5 queue gate: the teacher must return a valid spec for >= 90% of flagged items, else #7 stops before training.
  F6 kept_self = kept texts with a G label (was: G and V); V can only be missing where G has no execution signature.
     Claude 'result' null / non-dict output is retried like a timeout (second Opus review, 22:28, GO, nits only).
  Verdict order stays: gate 1 -> PASS -> KILL -> NOT ESTABLISHED. Power ~50% for a T - G of ~2.7 pp.
  Interpretation limit: Sonnet wrote pool F from the gold spec, so as teacher it relabels text it authored (an
  authorship advantage a real teacher on user text would not have); v2pool's 'main' cue (100% expr / 0% dispatch) is
  the same as the pool's, so v2pool cannot show the cue harm; B and A (18 dispatch texts with 'main' each) can.

usage:
  selftrain.py labels  --pool POOL.json --out DIR [--k 8] [--limit N] [--check-raws RAWS.json]
  selftrain.py teacher --out DIR [--max-calls 12]
  selftrain.py extras  --pool POOL.json --out DIR --exclude REQ.json ...
  selftrain.py v2pool-arith --out FILE      (prediction arithmetic on #46's recorded v2pool candidates; no model)
  selftrain.py verdict --out DIR      (reads the arms' result.json and belt traces; writes DIR/verdict.json)
"""

import argparse
import collections
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_pipeline as E  # noqa: E402

RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
E2E = f"{RUN}/e2e"
HELD = [f.split("{")[0].strip().lower() for f in (E.FRAMES_D[3], E.FRAMES_X[1])] + ["preciso de uma função main",
                                                                                      "escolha entre"]


def texts_of(pool):
    return [{"id": q["id"], "text": q["text"]} for q in pool]


def valid(raw):
    _, b, fl = E.check_spec(raw)
    return b if b and not fl else None


def spec_json(raw):
    b = valid(raw)
    return json.dumps(E.spec_of(b)) if b else None


def signatures(raws):
    import saci_redo as SR                              # bwrap reference runs; gold-free (the spec's own blocks)
    return SR.signatures(raws)


def label_rules(cands, sig, k):
    """cands = [greedy raw] + k sample raws -> {"G", "V", "flag", "top"}; labels are canonical spec JSON or None."""
    g_ok = valid(cands[0]) is not None
    first_valid = next((r for r in cands if valid(r)), None)
    G = spec_json(cands[0]) if g_ok else (spec_json(first_valid) if first_valid else None)
    groups = collections.OrderedDict()
    for n, r in enumerate(cands):
        s = sig.get(r)
        if s is not None:
            groups.setdefault(s, []).append((n, r))
    if not groups:
        return {"G": G, "V": None, "flag": True, "top": 0, "greedy_valid": g_ok}
    best_sig = max(groups, key=lambda s: (len(groups[s]), groups[s][0][0] == 0, -groups[s][0][0]))
    mem = groups[best_sig]
    cnt = collections.Counter(r for _, r in mem)
    first = {}
    for n, r in mem:
        first.setdefault(r, n)
    v_raw = max(cnt, key=lambda r: (cnt[r], -first[r]))
    flag = (not g_ok) or sig.get(cands[0]) != best_sig or len(mem) < (k + 2) // 2
    return {"G": G, "V": spec_json(v_raw), "flag": flag, "top": len(mem), "greedy_valid": g_ok}


def cmd_labels(a):
    import torch
    import saci_redo as SR
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    pool = texts_of(json.load(open(a.pool)))
    if a.limit:
        pool = pool[:a.limit]
    S = SR.Student(a.lora, a.mem_frac)
    t0, rows = time.perf_counter(), []
    for n, q in enumerate(pool):
        p = S.prompt_ids(q["text"])
        g = S.decode(S.greedy(p, S.max_new))
        smp = [S.decode(ids) for ids, _ in S.sample(p, S.max_new, a.k, 5000 + n)]
        rows.append({"id": q["id"], "cands": [g] + smp})
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(pool)} sampled, {time.perf_counter() - t0:.0f} s", flush=True)
    res = {"n": len(rows), "k": a.k, "lora": a.lora, "gen_s": round(time.perf_counter() - t0, 1),
           "pc4": {"equal_calls": S.pc4[0], "calls": S.pc4[1], "pass": S.pc4[0] == S.pc4[1] > 0}}
    if a.check_raws:                                    # gate 5: greedy here == the recorded eval greedy
        rec = json.load(open(a.check_raws))
        chk = [r for r in rows if r["id"] in rec]
        same = sum(r["cands"][0] == rec[r["id"]]["raw"] for r in chk)
        res["pc_greedy"] = {"same": same, "checked": len(chk), "pass": len(chk) > 0 and same >= 0.95 * len(chk)}
    sig = signatures([c for r in rows for c in r["cands"]])
    for r in rows:
        r.update(label_rules(r["cands"], sig, a.k))
        r["sigs"] = [sig.get(c) for c in r["cands"]]
    res["flagged"] = sum(r["flag"] for r in rows)
    res["no_label"] = {x: sum(r[x] is None for r in rows) for x in ("G", "V")}
    with open(os.path.join(a.out, "labels.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(res, open(os.path.join(a.out, "labels_summary.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


TEACH = """You turn Portuguese requests into the JSON spec of a small Python code generator.
Spec formats: {"main": "dispatch", "calls": [op ids in the order the request lists them]} = a calculator / program
that offers those operations; {"main": "expr", "expr": "<nested call>"} = main(a, b) returns that formula, written as
nested calls of op ids over a and b, e.g. "add(mul(a, b), b)". Operation ids: add a+b, sub a-b, mul a*b, div a/b,
mod remainder of a/b, maxx larger of a and b, sqrt square root of a, neg -a, abs |a|, square a*a, pow a**b,
fact a!, hyp hypotenuse with legs a and b, avg average of a and b, pct a percent of b. One-argument ops (sqrt, neg,
abs, square, fact) and pow exist only in "dispatch". In "expr" only add, sub, mul, div, mod, maxx, avg, pct, hyp may
appear, each with exactly two arguments; leaves are only a and b (no numbers). Keep every operation, operand (a or b),
operand order and nesting.
For each item, a smaller model proposed the candidate specs listed (possibly all wrong). If one of them is exactly
right, answer it; otherwise write the right spec yourself.
Answer with one JSON array only: [{"id": "...", "spec": {...}}], same ids, same count.

ITEMS:
"""


def sonnet(prompt, cwd, budget):
    for _ in range(3):
        if budget["left"] <= 0:
            return None
        budget["left"] -= 1
        try:
            p = subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", "", "--output-format", "json"],
                               input=prompt, capture_output=True, text=True, timeout=280, cwd=cwd)
            text = (json.loads(p.stdout) or {}).get("result") or ""
            assert isinstance(text, str)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, AttributeError, AssertionError):
            continue
        if "operation blocked by hook" in text[:300]:
            continue
        i, j = text.find("["), text.rfind("]")
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            continue
    return None


def cmd_teacher(a):
    rows = [json.loads(x) for x in open(os.path.join(a.out, "labels.jsonl"))]
    text = {q["id"]: q["text"] for q in texts_of(json.load(open(a.pool)))}
    flagged = [r for r in rows if r["flag"]]
    cwd = os.path.join(a.out, "sonnet_cwd")
    os.makedirs(cwd, exist_ok=True)
    budget, out, log = {"left": a.max_calls}, {}, []
    rng = random.Random(0)
    for i in range(0, len(flagged), a.batch):
        B = flagged[i:i + a.batch]
        items = []
        for r in B:
            c = sorted({s for s in (spec_json(x) for x in r["cands"]) if s})
            rng.shuffle(c)
            items.append({"id": r["id"], "request": text[r["id"]], "candidates": [json.loads(s) for s in c]})
        ans = sonnet(TEACH + json.dumps(items, ensure_ascii=False), cwd, budget)
        got = {x["id"]: x.get("spec") for x in (ans or []) if isinstance(x, dict) and "id" in x}
        for r in B:
            s = got.get(r["id"])
            sj = spec_json(json.dumps(s)) if isinstance(s, dict) else None
            out[r["id"]] = sj
            log.append({"id": r["id"], "answer": s, "valid": sj is not None})
        print(json.dumps({"batch": i, "answered": sum(v is not None for v in out.values()),
                          "calls_left": budget["left"]}), flush=True)
    res = {"flagged": len(flagged), "valid_answers": sum(v is not None for v in out.values()),
           "calls_used": a.max_calls - budget["left"]}
    json.dump({"teacher": out, "log": log, **res}, open(os.path.join(a.out, "teacher.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(res), flush=True)


def gold_spec(q):
    return json.dumps(E.spec_of(E.gold_bridged(q))) if "world" in q else json.dumps(E.spec_of(valid(json.dumps(q["gold"]))))


def cmd_extras(a):
    """G, Ts (and V, not trained) on the texts the student can label itself (kept_self); T and O on every kept text
    (kept_all: the teacher also labels texts with no valid student candidate). T = Ts labels on more texts."""
    pool = json.load(open(a.pool))
    rows = {r["id"]: r for r in map(json.loads, open(os.path.join(a.out, "labels.jsonl")))}
    T = json.load(open(os.path.join(a.out, "teacher.json")))["teacher"]
    lab = {}
    for q in pool:
        r = rows[q["id"]]
        t = T.get(q["id"]) if r["flag"] else None       # amendment 22:2x: unflagged (and invalid answer) -> G, not V
        lab[q["id"]] = {"G": r["G"], "V": r["V"], "T": t or r["G"], "O": gold_spec(q)}
    # student_understander --extra filters, applied once to the union of arms (a text dropped in one arm is dropped in
    # all), so the trainer's own filter drops nothing (checked in verdict: extra_dropped == 0)
    tests = E.load_requests(E2E)
    allq = tests + [q for f in a.exclude for q in json.load(open(f))]
    bad_t = {q["text"] for q in allq} | {q["orig_text"] for q in allq if "orig_text" in q}
    bad_g = {json.dumps(q["gold"], sort_keys=True) for q in allq if q["split"] != "one"}
    drop = collections.Counter()
    keep_all = []
    for q in pool:
        L = lab[q["id"]]
        why = ("no_teacher_label" if L["T"] is None else
               "text" if q["text"] in bad_t else
               "held_wording" if any(h in q["text"].lower() for h in HELD) else
               "gold_equal_excluded" if any(json.dumps(json.loads(v), sort_keys=True) in bad_g
                                            for v in L.values() if v) else None)
        if why:
            drop[why] += 1
        else:
            keep_all.append(q)
    keep_self = [q for q in keep_all if lab[q["id"]]["G"]]
    sets = {"G": (keep_self, "G"), "V": ([q for q in keep_self if lab[q["id"]]["V"]], "V"), "Ts": (keep_self, "T"),
            "T": (keep_all, "T"),
            "O": (keep_all, "O")}                         # arm -> (texts, label key); V reported, not trained
    rep = {"pool": len(pool), "kept_all": len(keep_all), "kept_self": len(keep_self), "dropped": dict(drop),
           "trained_arms": ["G", "Ts", "T", "O"], "arms": {},
           "unflagged_V_ne_G": sum(not rows[q["id"]]["flag"] and lab[q["id"]]["V"] != lab[q["id"]]["G"]
                                   for q in keep_self)}
    sig = signatures([lab[q["id"]][x] for q in keep_all for x in "GVTO" if lab[q["id"]][x]])

    def right(q, x):
        return sig.get(lab[q["id"]][x]) is not None and sig.get(lab[q["id"]][x]) == sig.get(lab[q["id"]]["O"])
    for name, (ks, x) in sets.items():
        assert all(lab[q["id"]][x] for q in ks), name
        json.dump([{"id": q["id"], "text": q["text"], "gold": json.loads(lab[q["id"]][x])} for q in ks],
                  open(os.path.join(a.out, f"extra_{name}.json"), "w"), ensure_ascii=False, indent=0)
        fl = [q for q in ks if rows[q["id"]]["flag"]]
        rep["arms"][name] = {"n": len(ks),
                             "exact_vs_gold": round(sum(lab[q["id"]][x] == lab[q["id"]]["O"] for q in ks) / len(ks), 4),
                             "same_outputs_as_gold": round(sum(right(q, x) for q in ks) / len(ks), 4),
                             "flagged_right": f"{sum(right(q, x) for q in fl)}/{len(fl)}",
                             "unflagged_right": f"{sum(right(q, x) for q in ks if q not in fl)}/{len(ks) - len(fl)}"}
    rep["no_self_label_in_kept_all"] = len(keep_all) - len(keep_self)
    rep["flag_catches_G_wrong"] = (f"{sum(rows[q['id']]['flag'] and not right(q, 'G') for q in keep_self)}/"
                                   f"{sum(not right(q, 'G') for q in keep_self)}")
    rep["teacher_valid_on_flagged"] = f"{sum(T.get(q['id']) is not None for q in keep_all if rows[q['id']]['flag'])}/" \
                                      f"{sum(rows[q['id']]['flag'] for q in keep_all)}"
    # gate 2: one-class cue words in the LLM-made texts
    cue = {}
    for w in ("main", "calculadora", "função", "retorn", "devolv", "opç", "operaç", "programa"):
        cue[w] = {k: sum(w in q["text"].lower() for q in keep_all if json.loads(lab[q["id"]]["O"])["main"] == k)
                  for k in ("expr", "dispatch")}
    rep["cue_words"] = cue
    rep["worlds"] = dict(collections.Counter(json.loads(lab[q["id"]]["O"])["main"] for q in keep_all))
    json.dump(rep, open(os.path.join(a.out, "extras_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def cmd_v2pool_arith(a):
    """gate 7: the label rules on #46's recorded v2pool candidates (greedy + k=8 full samples from the start)."""
    V = f"{E2E}/v2pool"
    reqs = {q["id"]: q for q in json.load(open(f"{V}/requests.json"))}
    rows = [json.loads(x) for x in open(f"{RUN}/saci_code/redo/rows.jsonl")]
    raws = json.load(open(f"{V}/student_s5_s1/raws.json"))
    cands = {r["id"]: [raws[r["id"]]["raw"]] + [c for c, _ in r["cand"]["full_k"]] for r in rows}
    sig = signatures([c for cs in cands.values() for c in cs] + [gold_spec(q) for q in reqs.values()])
    res = collections.Counter()
    labs = {i: label_rules(cs, sig, len(cs) - 1) for i, cs in cands.items()}
    sig.update(signatures([L[x] for L in labs.values() for x in ("G", "V") if L[x]]))
    for i, cs in cands.items():
        L = labs[i]
        gs = sig.get(gold_spec(reqs[i]))
        g_ok, v_ok = sig.get(L["G"]) == gs, sig.get(L["V"]) == gs
        res["n"] += 1
        res["G_right"] += g_ok
        res["V_right"] += v_ok
        res["flag"] += L["flag"]
        res["flag_and_G_wrong"] += L["flag"] and not g_ok
        res["G_wrong"] += not g_ok
        res["oracle_any_right"] += any(sig.get(c) == gs for c in cs)
        res["flag_and_oracle_right_and_V_wrong"] += L["flag"] and not v_ok and any(sig.get(c) == gs for c in cs)
        res["G_none"] += L["G"] is None
    out = dict(res)
    print(json.dumps(out), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


ARMS = {"s5": ("student_s5_s1", "student_s5_rerun7"), "G": ("student_s5sfG_s1", "student_s5sfG_s1"),
        "G2": ("student_s5sfG_s2", "student_s5sfG_s2"), "Ts": ("student_s5sfTs_s1", "student_s5sfTs_s1"),
        "T": ("student_s5sfT_s1", "student_s5sfT_s1"),
        "O": ("student_s5sfO_s1", "student_s5sfO_s1")}               # arm -> (student dir, belt tag)
EXTRA_OF = {"G": "G", "G2": "G", "Ts": "Ts", "T": "T", "O": "O"}     # arm -> label set it trains on
SETS = {"v2pool": f"{E2E}/v2pool", "B": f"{E2E}/para", "A": E2E}


def cmd_verdict(a):
    import numpy as np
    res = {"checks": {}, "sets": {}, "verdict": None}
    for x, (d, _) in ARMS.items():
        r = json.load(open(f"{E2E}/{d}/result.json"))
        if x != "s5":
            assert r["steps"] == 523 and r["extra_dropped"] == 0 and r["extra_reps"] == 3, (x, r["steps"], r["extra_dropped"])
            assert r.get("data_seed") == 1, (x, r.get("data_seed"))          # amendment: G2 = seed noise, same data
            ef = r["extra_file"][0] if isinstance(r["extra_file"], list) else r["extra_file"]
            assert os.path.normpath(ef) == os.path.normpath(os.path.join(a.out, f"extra_{EXTRA_OF[x]}.json")), (x, ef)
        res["checks"][x] = {k: r.get(k) for k in ("seed", "data_seed", "steps", "train_pairs", "extra", "extra_dropped",
                                                  "train_s")}
    base = res["checks"]["s5"]["train_pairs"]
    for x in EXTRA_OF:
        c = res["checks"][x]
        assert c["train_pairs"] - 3 * c["extra"] == base, (x, c, base)
    pas = {}
    for S, D in SETS.items():
        for x, (d, t) in ARMS.items():
            summ = json.load(open(f"{D}/summary_{t}.json"))
            assert os.path.normpath(summ["config"]["raws_file"]) == os.path.normpath(f"{D}/{d}/raws.json"), (S, x)
            pas[S, x] = {r["id"]: float(r["end_pass"]) for r in map(json.loads, open(f"{D}/traces_{t}.jsonl"))}
        ids = sorted(pas[S, "s5"])
        assert all(sorted(pas[S, x]) == ids for x in ARMS), S
        if S == "v2pool":
            res["checks"]["gate5_s5_rerun_v2pool_pass"] = int(sum(pas[S, "s5"].values()))
            assert res["checks"]["gate5_s5_rerun_v2pool_pass"] == 441, res["checks"]["gate5_s5_rerun_v2pool_pass"]
        idx = np.random.default_rng(0).integers(0, len(ids), (10_000, len(ids)))
        v = {x: np.array([pas[S, x][i] for i in ids]) for x in ARMS}
        out = {"n": len(ids), "pass": {x: round(float(v[x].mean()), 4) for x in ARMS}, "diff": {}}
        for p, q in (("T", "G"), ("Ts", "G"), ("T", "Ts"), ("O", "G"), ("G2", "G"), ("G", "s5"), ("T", "s5"),
                     ("O", "s5"), ("O", "T")):
            dd = v[p] - v[q]
            bs = dd[idx].mean(1)
            out["diff"][f"{p}-{q}"] = {"d": round(float(dd.mean()), 4),
                                       "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        og = out["diff"]["O-G"]["d"]
        out["T_recovers_of_O_minus_G"] = round(out["diff"]["T-G"]["d"] / og, 3) if og else None
        res["sets"][S] = out
    V = res["sets"]["v2pool"]["diff"]
    og, noise = V["O-G"], abs(V["G2-G"]["d"])

    def rule(d):                       # registered order: gate 1 -> PASS -> KILL -> NOT ESTABLISHED
        if og["ci"][0] <= 0:
            return "NO VERDICT (gate 1: O - G CI lower <= 0)"
        if d["ci"][0] > 0 and d["d"] > noise:
            return "PASS"
        if d["ci"][1] < .02:
            return "KILL"
        if d["ci"][0] > 0:
            return f"NOT ESTABLISHED (CI lower > 0 but {d['d']} <= seed noise |G2 - G| {noise})"
        return "NOT ESTABLISHED"
    verdict = rule(V["T-G"])
    res["verdict"] = verdict
    res["verdict_mechanism_Ts_minus_G"] = rule(V["Ts-G"])     # amendment: relabelling alone, same texts as G
    json.dump(res, open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
    print(json.dumps({"verdict": verdict, "mechanism_Ts_minus_G": res["verdict_mechanism_Ts_minus_G"],
                      "v2pool": res["sets"]["v2pool"]["diff"],
                      "pass": {S: res["sets"][S]["pass"] for S in SETS}}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("labels")
    c.add_argument("--pool", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--k", type=int, default=8)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--lora", default=f"{E2E}/student_s5_s1/lora")
    c.add_argument("--mem-frac", type=float, default=0.25)
    c.add_argument("--threads", type=int, default=2)
    c.add_argument("--check-raws", default=None, help="gate 5: recorded greedy raws {id: {raw}} of the same texts")
    c = sub.add_parser("teacher")
    c.add_argument("--pool", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--batch", type=int, default=30)
    c.add_argument("--max-calls", type=int, default=12)
    c = sub.add_parser("extras")
    c.add_argument("--pool", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--exclude", nargs="+", required=True)
    c = sub.add_parser("v2pool-arith")
    c.add_argument("--out", required=True)
    c = sub.add_parser("verdict")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"labels": cmd_labels, "teacher": cmd_teacher, "extras": cmd_extras, "v2pool-arith": cmd_v2pool_arith,
     "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    main()
