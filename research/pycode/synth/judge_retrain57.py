"""Judge-routed retraining of OUR planner/writer (ledger #37 + #57, master queue I7). Registered 2026-10-27 13:19.

HIS FORMULATION (quoted, followed literally):
  #37 "o Qwen vai dizer o quanto que o modelo acertou ou errou, só que o Sonnet também vai fazer isso. Ele vai pegar o
  que o Qwen escreveu e qual foi o resultado, e ele vai comparar... regra para o Sonnet seguir... dar a nota... o quanto
  que o modelo errou do quanto que foi pedido... Ambos podem dar já a justificativa do erro e pedir para melhorar o
  treino... Se a do Qwen já estiver boa, vai usar aquilo e ponto final... se o Sonnet vê que o Qwen errou na
  observação, ele mesmo vai ajustar e passar para o modelo retreinar... roteamento de erro (not backprop)... se o
  planner errou, o writer tem que avisar".
  #57 "EU NÃO PEDI PARA O QWEN APRENDER": Qwen and Sonnet are JUDGES; the error they point at retrains OUR small
  planner/writer, never Qwen.

WORLD (where OUR modules fail naturally; #35: on belt sets A/B/v2pool only the understander fails):
  synth2 list-pipeline functions `u1_then_u2_..._then_final(xs[, k])`, planner = blockaddr address net (gate_bias -4,
  1500 steps, trained on 1-2 unit names), writer = bytewriter byte GRU d32 ternary (1500 steps, 50 train blocks).
  Inputs = len5 / len6 chains (out of the planner's training range). Known: planner seeds 0/5/6 drop to len6
  plan_exact .565/.5125/.0375 (blockaddr_addr_s{0,5,6}_gb-4.json); writer d32_ter text_exact len5 .70/.595/.505 on
  seeds 0/1/2 (bytewriter18). So BOTH stages fail here, and the blamed stage is not constant.
  Seeds: s in {0, 5, 6}; base planner seed s, base writer seed s. Declared selection: 0/5/6 are the 3 of 7 known
  planner seeds that fail at len5/6 (seeds 1-4 are ~perfect there: nothing to route to the planner).
  I6 statement-level writer (interp55_writer.py): NO trained checkpoints exist (gpuq/hold/60_interp55_writer.sh never
  ran, the writer saves no .pt) -> part (b) NOT done; only (a).

PIPELINE per pool item (gold-free; gold only in the report):
  our planner -> our writer -> bwrap facts (outputs on 8 shown inputs, machine flags) ->
  Qwen2.5-Coder-3B (FIRST judge, HF bf16, greedy) {score, stage planner|writer|none, location, justification, fix} ->
  triage (gold-free): Sonnet sees the item if machine flags OR Qwen invalid JSON OR Qwen stage != none OR score < 100,
  plus a 25% hash audit of the rest ->
  Sonnet (FINAL judge, `claude -p --model sonnet`, batched 5 items/call, serial) gets request + Qwen judgment + result
  + RULE below -> {agree_with_qwen, score, stage, location, justification, fix}. agree -> Qwen's verdict ("usa aquilo e
  ponto final"); else Sonnet's. Qwen invalid JSON => qwen_ok=false always (logged); Sonnet "agree" while its own
  stage/location differs => treated as disagree (logged 'agree_inconsistent'); this is the e2e_pipeline x170 defect.
  Fix gate (A4, gold-free): corrected code runs in bwrap on all 40 inputs with no flags (loads, right name, no
  exception) AND the final judge re-passes it (pass=true, score >= 90). Accepted fixes -> rows ROUTED to the blamed
  module at the blamed location: planner row = (name, corrected step chain); writer row = (block key at location,
  corrected block text).

ARMS (paired by seed: same base nets, same original-data draw order, same extra-row draw order; equal budget =
  E extra steps for the planner AND E for the writer in every arm, 25% of each batch from extra rows when the arm has
  rows for that module):
  N  no extra rows (continue on original data only)        J  judge-routed rows (Qwen first, Sonnet final)
  R  n_J random generator rows, len5/6, same planner/writer split as J ("just more data")
  Q  Qwen-only verdicts + Qwen re-pass gate (what Sonnet adds; row count n_Q is its own, reported)
  O  oracle-routed rows from the same pool failures (true stage/location/text from gold; n_J rows; upper bound)
  Held-out: EVAL_SEED generator (len5 300 + len6 300), disjoint by name from pool, R rows, train and every bridge set.
  Original lengths (regression): Data test_a, len3, len4.

PREDICTIONS (written 13:19, before any run; pass@1 strict bwrap grade, mean of 3 seeds):
  P0 positive control: base s0 test_a pass@1 = 0.8703 exactly on CPU (= bytewriter18 d32_ter s0, planner s0 test_a
     plan 1.0); on GPU within +-0.08 of it (ternary GRU, different numerics).
  P1 base len5 pass@1 0.35-0.55, len6 0.15-0.35; failures split planner 35-60% / writer 40-65%.
  P2 N - base: +0..+8 pp on len5/6 (the writer is underfit at 1500 steps).
  P3 R - N: +5..+15 pp on len6 (length data for the planner).
  P4 J - R >= +3 pp on len5/6 mean (targeted writer rows fix the blocks that actually fail; planner part ~equal).
     KILL of "judge routing beats more data": J - R <= 0 with the 95% CI covering 0, or J > R in < 3/3 seeds
     (the CI resamples items within seed only, so seed sign is required too; review S7).
  P5 O - J <= 4 pp (judges near oracle); J - Q >= +3 pp (Sonnet adds; Qwen-3B accepts fewer/wronger fixes).
  P6 judge quality on truly failing items: Sonnet-final stage acc >= 0.90, location acc >= 0.80; Qwen-3B stage acc
     0.50-0.75, location 0.30-0.60; best constant-stage baseline 0.50-0.65. Label precision of J rows 0.85-0.95.
  P7 no regression: every arm within -2 pp of N on test_a / len3 / len4.

REVIEW 10-27 (independent Opus, NO-GO -> fixed): B1 a failed/capped phase now aborts (no silent empty J/R/O);
  S1 flag-rule stage baseline added; S3 planner steps must spell the requested name; S4 O matches J's module split;
  S5 base nets trained on CPU (the known models; GPU only for Qwen and arm training/eval); S6 plan/text exact per arm;
  S7 seed-sign rule. DECLARED (S2): with this RULE (house style, finals) and a name that lists the steps, an accepted
  fix is close to an oracle row once the error is caught; J vs R therefore mostly measures WHICH items are picked
  (judge-found failures vs random) and the judges' catch/routing rate, not deep reasoning. That is his loop as stated.

N-DROP FIX 10-27 14:31 (coordinator: N at 0 rows lost test_a 0.8703 -> 0.7844 after 40 steps at lr 1e-3).
  Root cause, measured (CPU, writer-only probe, base s0 0.8703): the planner never moves (0.8703 at every lr/steps);
  the damage is the ternary d32 writer. test_a is a step function of ~3 marginal train-pool blocks (all 50 test_a
  blocks are in the 50-block train pool; one block = ~4 pp). (a) lr 1e-3 with a FRESH AdamW is 3.3x the base run's
  final lr (3e-3 cosine, 10% floor = 3e-4); Adam's first bias-corrected steps are ~lr per weight, which flips
  ternary thresholds: 40 steps 0.7828 (one block lost), recovers to 0.8688 by 400. (b) Even at lr 1e-4/3e-4,
  ternary flips land at random late steps: raw 400-step s5 0.8281/0.7672, s6 0.8219 (vs 0.8688). (c) Base s6 has a
  block whose TEXT differs from the training block but passes the grade, so a text-exact guard still lost 4.7 pp.
  Fix (same for all arms): lr 1e-4 (1/3 of the base final lr) + Guard57: every 25 steps snapshot if every train
  item the base got right is still right (planner: exact train plan; writer: exact text OR same behaviour on the 40
  inputs); restore the last such snapshot (base if none). Training data only, never eval gold.
  N - base on test_a after the full 400 steps, CPU smoke, seeds 0/5/6: before (lr 1e-3, no guard, s0 40 steps)
  -8.6 pp; after +4.22 / +0.31 / 0.00 pp (no seed loses; s0's +4.2 is one under-fit train block learned, which a
  no-loss guard should keep). Cost: when no checkpoint is a superset the module stays at base (smoke s6 writer:
  kept_step 0 in every arm), so that arm's extra rows have no writer effect for that seed. Logged per arm.
ADDENDUM 10-27 14:31 (predictions above stay as stamped 13:19; this is new information, not a revision): in the smoke,
  N alone (original data, 0 rows) lifts the planner a lot: len5+len6 pass N - base +30 / +10 / +62.5 pp (s0/s5/s6;
  s6 len6 plan_exact 0.05 -> 1.0). The base planner's length generalisation is under-training, not a missing skill,
  so the "failing lengths" shrink under N and J - N / J - R headroom is mostly the writer. Expect P2-P4 to be hard.

SONNET CAP 10-27 14:31: pool 100 items/seed (50 len5 + 50 len6), batch 6, calls/seed = ceil(sent/6) verdict +
  ceil(accepted/6) re-pass, retries (<=1 per call) count. Smoke send rates (Qwen-0.5B, 10 items): s0 8/10, s5 7/10,
  s6 5/10, accepted 7/6/4 -> expected per seed ~ceil(67/6)+ceil(57/6) = 12+10 = 22, 3 seeds ~66; upper bound (every
  item sent and accepted, Qwen-3B flags more) 3 x (17+17) = 102. Cap 150 = 1.47x the upper bound, 2.3x expected.

REVIEW-2 10-27 14:34 (second independent Opus, GO-with-fixes): #1 GPU train path untested -> the run script first runs
  `train --device cuda --smoke --seeds 0` on a copy of smoke3 (aborts the job on failure, before Qwen). #2 arm B =
  base re-evaluated on the train device with 0 steps (GPU positive control; N - B, not N - base, is the damage test).
  #3 Qwen timeout 14400 + one resume retry. #4 DECLARED: the writer is context-free over the same 50 pool blocks, so
  its effective n is ~blocks x seeds; the item bootstrap CI is too narrow for writer effects; read seeds_positive and
  verdict.modules_moved (pool blocks ok per arm/seed) first. #5 DECLARED: Guard57 makes writer regression impossible
  on test_a by construction (P7 is then trivially met for the writer); a seed whose writer kept_step is 0 in J and R
  says nothing about writer routing. #6 not adopted (last, not best, snapshot; same rule for all arms). #7 Sonnet cap
  now counts calls already logged (reruns cannot exceed it). #8/#9 covered by label precision / logged.

Phases (each a separate process; caches in --out):
  base   (CPU) train base nets, build sets, run belt on pool/eval/orig            -> base_s*.pt, base_s*.json, sets.json
  qwen   (GPU) Qwen judge every pool item + Qwen fix gate + Qwen re-pass          -> qwen_s*.jsonl
  sonnet (no GPU) triage, Sonnet judge, final verdict, final fix gate, Sonnet re-pass -> final_s*.jsonl, sonnet_calls.jsonl
  train  (GPU) build rows per arm, continue training, eval                         -> arm_<A>_s*.json, rows_<A>_s*.json
  report (CPU) verdict.json: arm table, paired bootstrap, judge quality, label precision
Usage: judge_retrain57.py <phase> --out DIR [--device cuda|cpu --smoke --seeds 0,5,6 ...]
"""

import argparse
import ast
import hashlib
import json
import copy
import math
import os
import random
import shutil
import subprocess
import sys
import textwrap
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockaddr as BA  # noqa: E402
import bytewriter as BW  # noqa: E402
import synth2 as S2  # noqa: E402
from bridge_cpu import SYM, Data, blocks_of, parse  # noqa: E402
from e2e_pipeline import last_json  # noqa: E402  (reuse: defensive JSON extraction)

OUT = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/judge57"
CACHE = "/media/llm-data/models/hf/hub"
POOL_SEED, EVAL_SEED, RAND_SEED = 5701, 5702, 5703
SEEN = [u for u in S2.UNITS if u not in S2.HELD]
FINALS = list(S2.FINAL)
N_SHOW = 8                         # inputs shown to the judges (outputs only, never expected values)
WRITER_D, WRITER_TER = 32, True
ARMS = ["N", "J", "R", "Q", "O"]

RULE = """RULE (the judges follow this exactly).
A small code generator wrote a Python function from its NAME only. The name lists list-transformation steps joined by
`_then_`, applied in order to the list `xs`; an optional last step is a FINAL that turns the list into the result.
With no final the function returns `xs`. If a step name ends in `_k` the function takes a second argument `k`.
House conventions: one statement block per step, each block reassigns `xs` (e.g. `xs = sorted(xs)`,
`xs = [x for x in xs if x > k]`, `xs = xs[:k]`); loops build `out` and end with `xs = out`. FINALS: `sum` ->
`return sum(xs)`, `max` -> `return max(xs, default=0)`, `len` -> `return len(xs)`, `first` -> `return xs[0] if xs
else 0`, `count_gt_N` -> `return sum(1 for x in xs if x > N)`, none -> `return xs`.
Example (name cumsum_then_keep_gt_k_then_max):
def cumsum_then_keep_gt_k_then_max(xs, k):
    acc, out = 0, []
    for x in xs:
        acc += x
        out.append(acc)
    xs = out
    xs = [x for x in xs if x > k]
    return max(xs, default=0)

The generator has two stages. PLANNER: reads the name and outputs the step chain (the PLAN). WRITER: writes one code
block per planned step. Blame:
- stage "planner" if the PLAN differs from the steps the name asks for (missing, extra, swapped, wrong number, or a
  malformed plan). location = index (0-based, final counted as the last step) of the FIRST wrong step of the plan.
- stage "writer" if the PLAN is right but a block's code does not do its step. location = index of the first wrong
  block (blocks are numbered like the plan).
- stage "none" if the function does what the name asks. Do not guess expected outputs you cannot derive.
score = 0-100, how much of what the name asks the code really does (100 = all of it).
fix (null when stage is none):
- planner: {"steps": [unit step names in order, exactly as in the name, without the final], "final": "<final name or
  empty string>", "code": "<the full corrected function, house style>"}
- writer: {"blocks": [{"location": i, "text": "<corrected code of block i, no indentation, one statement per line>"},
  ...]} one entry per wrong block. Keep the house style of the other blocks."""

Q_ROLE = ('\n\nYou are the FIRST judge. Answer with ONE JSON object only: {"score": int, "stage": "none"|"planner"|'
          '"writer", "location": int or null, "justification": str, "fix": object or null}.')
S_ROLE = ("\n\nYou are the FINAL judge. For each item you get the evidence and the FIRST judge's verdict (Qwen). "
          "Compare the first judge's observation with the code and its execution. If the first verdict is right "
          "(stage, location and fix), set agree_with_qwen true and copy its fields. If it is wrong anywhere, or "
          "invalid, set agree_with_qwen false and give your own corrected score, stage, location, justification and "
          'fix. Answer with ONE JSON object only: {"verdicts": [{"id": str, "agree_with_qwen": bool, "score": int, '
          '"stage": str, "location": int or null, "justification": str, "fix": object or null}, ...]}.')
REPASS = ("You re-check corrected functions against the same RULE. For each item, pass = true only if the code does "
          "everything the name asks.")


# ------------------------------------------------------------------ data / belt

def gen_tasks(seed, ks, n_each, exclude):
    g, out, names = random.Random(seed), [], set()
    for k in ks:
        got = 0
        while got < n_each:
            t = [[g.choice(SEEN) for _ in range(k)], g.choice(FINALS)]
            n = S2.name_of(*t)
            if n not in exclude and n not in names:
                names.add(n)
                out.append(t)
                got += 1
    g.shuffle(out)
    return out


def build_sets(data, smoke):
    known = set()
    for v in json.load(open(f"{S2.D}/splits.json")).values():
        known |= {S2.name_of(*t) for t in v}
    for v in data.sets.values():
        known |= {S2.name_of(*t) for t in v}
    npool, neval, nrand = (5, 20, 60) if smoke else (50, 300, 400)
    pool = gen_tasks(POOL_SEED, (5, 6), npool, known)
    ev = gen_tasks(EVAL_SEED, (5, 6), neval, known | {S2.name_of(*t) for t in pool})
    rnd = gen_tasks(RAND_SEED, (5, 6), nrand, known | {S2.name_of(*t) for t in pool + ev})
    names = [S2.name_of(*t) for t in pool + ev + rnd]
    assert len(names) == len(set(names)) and not (set(names) & known)
    return {"pool": pool, "eval_len5": [t for t in ev if len(t[0]) == 5], "eval_len6": [t for t in ev if len(t[0]) == 6],
            "rand": rnd}


def assemble(task, texts):
    return S2.head(*task) + "".join("".join(f"    {l}\n" for l in w.split("\n") if l) for w in texts)


def run_belt(planner, writer, tasks):
    """mirrors bytewriter.evaluate, per item."""
    planner.eval()
    writer.eval()
    out = []
    for i in range(0, len(tasks), 256):
        ch = tasks[i:i + 256]
        rows = planner.greedy(BA.name_bytes(ch)).tolist()
        syms = []
        for row in rows:
            s = [SYM[j] for j in row]
            syms.append(s[:s.index("<eos>") + 1] if "<eos>" in s else s)
        plans = [parse(s) for s in syms]
        blocks = [BW.blocks_of_task(p) if p is not None else [] for p in plans]
        flat = [b for bl in blocks for b in bl]
        texts = writer.write(torch.tensor([BW.block_key(*b)[0] for b in flat]),
                             torch.tensor([BW.block_key(*b)[1] for b in flat])) if flat else []
        j = 0
        for t, s, p, bl in zip(ch, syms, plans, blocks):
            w = texts[j:j + len(bl)]
            j += len(bl)
            out.append({"task": t, "syms": s, "plan": [p[0], p[1]] if p else None, "blocks": [list(b) for b in bl],
                        "texts": w, "src": assemble(t, w) if bl else S2.head(*t) + "    return None\n"})
    planner.train()
    writer.train()
    return out


def run_srcs(items):
    """[(src, task)] -> raw bwrap outputs on all 40 inputs (gold-free facts)."""
    res = []
    for i in range(0, len(items), 256):
        res += S2.run_many([(s, S2.name_of(*t), S2.nargs(t[0])) for s, t in items[i:i + 256]])
    return res


def flags_of(src, outs, task):
    f = []
    try:
        tree = ast.parse(src)
        if not any(isinstance(n, ast.FunctionDef) and n.name == S2.name_of(*task) for n in tree.body):
            f.append("missing_function")
    except SyntaxError:
        f.append("syntax_error")
    if outs is None:
        f.append("load_fail")
    elif any(o == "!" for o in outs):
        f.append("exception")
    return f


def gold_steps(task):
    return list(task[0]) + [task[1]]


def truth_of(item, score):
    """gold: root stage + location (+ every wrong writer block). REPORT ONLY."""
    t = item["task"]
    if score == 1.0:
        return {"stage": "none", "loc": None, "wrong": []}
    G = gold_steps(t)
    if item["plan"] is None:
        gs = blocks_of(*t)[:-1]
        ps = item["syms"]
        i = next((k for k in range(min(len(gs), len(ps))) if gs[k] != ps[k]), min(len(gs), len(ps)))
        loc = max(0, sum(1 for s in gs[:i + 1] if s.startswith(("op:", "fin:"))) - 1)
        return {"stage": "planner", "loc": loc, "wrong": []}
    P = list(item["plan"][0]) + [item["plan"][1]]
    if P != G:
        loc = next((k for k in range(min(len(P), len(G))) if P[k] != G[k]), min(len(P), len(G)))
        return {"stage": "planner", "loc": loc, "wrong": []}
    wrong = [k for k, (b, w) in enumerate(zip(item["blocks"], item["texts"])) if w != BW.block_text(*b)]
    return {"stage": "writer", "loc": wrong[0] if wrong else None, "wrong": wrong}


def evidence(item, outs, flags, iid):
    t = item["task"]
    L = [f"ID: {iid}", f"REQUEST: write `{S2.head(*t).strip()}`"]
    if item["plan"] is None:
        L.append("PLAN (planner output): MALFORMED chain " + " ".join(item["syms"]))
    else:
        L.append("PLAN (planner output): steps " + json.dumps(item["plan"][0]) + ", final " + json.dumps(item["plan"][1]))
    L.append("BLOCKS (writer output, numbered like the plan):")
    for k, (b, w) in enumerate(zip(item["blocks"], item["texts"])):
        L.append(f"  [{k}] step {'final ' if b[1] else ''}{b[0] or '(none)'}:")
        L += [f"      {l}" for l in w.split("\n") if l] or ["      (empty)"]
    L += ["FULL CODE:", item["src"].rstrip(), "EXECUTION (bwrap; outputs only, no expected values):"]
    for (xs, k), o in zip(S2.INPUTS[:N_SHOW], (outs or ["<module failed to load>"] * N_SHOW)[:N_SHOW]):
        call = f"{S2.name_of(*t)}({xs}, {k})" if S2.nargs(t[0]) == 2 else f"{S2.name_of(*t)}({xs})"
        L.append(f"  {call} -> {'EXCEPTION' if o == '!' else o}")
    L.append("MACHINE FLAGS: " + (", ".join(flags) if flags else "none"))
    return "\n".join(L)


# ------------------------------------------------------------------ verdicts and fixes (gold-free)

def norm_verdict(v, n_blocks):
    """-> cleaned verdict or None (invalid). Never trusts a verdict that is not well formed."""
    if not isinstance(v, dict):
        return None
    st = v.get("stage")
    if st not in ("none", "planner", "writer"):
        return None
    try:
        sc = int(v.get("score"))
    except (TypeError, ValueError):
        return None
    loc = v.get("location")
    if st != "none":
        try:
            loc = int(loc)
        except (TypeError, ValueError):
            return None
    else:
        loc = None
    fx = v.get("fix")
    return {"score": max(0, min(100, sc)), "stage": st, "location": loc,
            "justification": str(v.get("justification", ""))[:600], "fix": fx if isinstance(fx, dict) else None}


def norm_text(s):
    s = textwrap.dedent(str(s).replace("\t", "    ")).strip("\n")
    lines = [l.rstrip() for l in s.split("\n") if l.strip()]
    return "\n".join(lines) + "\n" if lines else ""


def build_fix(item, v):
    """verdict -> {code, rows} or {reject: why}. rows are routed to the blamed module at the blamed location."""
    if v is None or v["stage"] == "none":
        return {"reject": "no_error_claimed"}
    fx, t = v["fix"], item["task"]
    if fx is None:
        return {"reject": "no_fix"}
    name = S2.name_of(*t)
    if v["stage"] == "planner":
        steps, fin = fx.get("steps"), fx.get("final", "")
        if not isinstance(steps, list) or not all(isinstance(u, str) for u in steps) or not isinstance(fin, str):
            return {"reject": "bad_steps"}
        if not steps or any(u not in S2.UNITS for u in steps) or fin not in S2.FINAL or len(steps) > 6:
            return {"reject": "steps_out_of_vocab"}
        code = fx.get("code")
        if not isinstance(code, str) or not code.strip():
            return {"reject": "no_code"}
        code = norm_code(code)
        if S2.name_of(steps, fin) != name:              # gold-free: the row must be the request the gate checked
            return {"reject": "steps_disagree_with_name"}
        syms = blocks_of(steps, fin)
        return {"code": code, "rows": [{"module": "planner", "name": name, "syms": syms}],
                "fix_steps": [steps, fin]}
    blocks = fx.get("blocks")
    if not isinstance(blocks, list) or not blocks or item["plan"] is None:
        return {"reject": "bad_blocks"}
    texts, rows = list(item["texts"]), []
    for e in blocks[:6]:
        if not isinstance(e, dict):
            return {"reject": "bad_blocks"}
        try:
            k = int(e.get("location"))
        except (TypeError, ValueError):
            return {"reject": "bad_location"}
        if not 0 <= k < len(texts):
            return {"reject": "location_out_of_range"}
        txt = norm_text(e.get("text", ""))
        if not txt or any(c not in BW.C2I for c in txt) or len(txt) > 159:
            return {"reject": "unencodable_text"}
        texts[k] = txt
        b = item["blocks"][k]
        op, dg = BW.block_key(*b)
        rows.append({"module": "writer", "op": op, "dig": dg, "text": txt, "block": b, "loc": k})
    return {"code": assemble(t, texts), "rows": rows}


def norm_code(code):
    code = code.strip("\n")
    if code.startswith("```"):
        code = "\n".join(l for l in code.split("\n") if not l.startswith("```"))
    return textwrap.dedent(code).rstrip() + "\n"


def gate_bwrap(fixes):
    """fixes [(item, fix)] -> no-flag check on all 40 inputs. Gold-free."""
    todo = [(i, f) for i, f in enumerate(fixes) if "code" in f[1]]
    outs = run_srcs([(f["code"], it["task"]) for _, (it, f) in todo]) if todo else []
    res = [None] * len(fixes)
    for (i, (it, f)), o in zip(todo, outs):
        fl = flags_of(f["code"], o, it["task"])
        res[i] = {"flags": fl, "outs": o}
    return res


def repass_evidence(item, code, outs, iid):
    t = item["task"]
    L = [f"ID: {iid}", f"REQUEST: write `{S2.head(*t).strip()}`", "CORRECTED CODE:", code.rstrip(),
         "EXECUTION (bwrap; outputs only):"]
    for (xs, k), o in zip(S2.INPUTS[:N_SHOW], (outs or ["<module failed to load>"] * N_SHOW)[:N_SHOW]):
        call = f"{S2.name_of(*t)}({xs}, {k})" if S2.nargs(t[0]) == 2 else f"{S2.name_of(*t)}({xs})"
        L.append(f"  {call} -> {'EXCEPTION' if o == '!' else o}")
    return "\n".join(L)


def repass_ok(v):
    if not isinstance(v, dict):
        return False
    try:
        return v.get("pass") is True and int(v.get("score", 0)) >= 90
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------ judges

class Qwen:
    def __init__(self, model, device, mem_frac):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(mem_frac)
        self.tok = AutoTokenizer.from_pretrained(model, cache_dir=CACHE, local_files_only=True)
        kw = {"device_map": {"": 0}} if device.startswith("cuda") else {}
        self.m = AutoModelForCausalLM.from_pretrained(model, cache_dir=CACHE, local_files_only=True,
                                                      dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                                                      **kw).eval()
        self.dev = device

    @torch.no_grad()
    def ask(self, system, user, max_new):
        ids = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                           add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):                    # transformers 5 returns a BatchEncoding
            ids = ids["input_ids"]
        ids = ids.to(self.dev)
        out = self.m.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def ensure_claude_cfg(path):
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)
    dst = os.path.join(path, ".credentials.json")
    shutil.copy(os.path.expanduser("~/.claude/.credentials.json"), dst)
    os.chmod(dst, 0o600)
    json.dump({"hasCompletedOnboarding": True}, open(os.path.join(path, ".claude.json"), "w"))
    return path


class Sonnet:
    """serial `claude -p`, clean config dir, every call logged, hard cap."""

    def __init__(self, out, cap):
        self.cfg = ensure_claude_cfg(os.path.join(out, "claude_cfg"))
        self.cwd = os.path.join(out, "sonnet_cwd")      # empty dir: no gold next to the judge, and --tools ""
        os.makedirs(self.cwd, exist_ok=True)
        self.log = open(os.path.join(out, "sonnet_calls.jsonl"), "a")
        self.cap = cap                               # review-2 #7: the cap spans reruns (count logged calls)
        self.used = len(jl(os.path.join(out, "sonnet_calls.jsonl")))

    def ask(self, prompt, key):
        env = dict(os.environ, CLAUDE_CONFIG_DIR=self.cfg)
        for attempt in range(2):
            if self.used >= self.cap:
                return None
            self.used += 1
            t0, text = time.time(), ""
            try:
                p = subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", "", "--output-format", "json"],
                                   input=prompt, capture_output=True, text=True, timeout=420, env=env, cwd=self.cwd)
                text = json.loads(p.stdout).get("result", "")
            except (subprocess.TimeoutExpired, json.JSONDecodeError, AttributeError) as e:
                text = f"<error {type(e).__name__}>"
            obj = last_json(text) if "operation blocked by hook" not in text[:300] else None
            ok = isinstance(obj, dict) and isinstance(obj.get("verdicts"), list)
            self.log.write(json.dumps({"call": self.used, "key": key, "attempt": attempt, "ok": ok,
                                       "prompt_md5": hashlib.md5(prompt.encode()).hexdigest(),
                                       "prompt_chars": len(prompt), "s": round(time.time() - t0, 1),
                                       "result_tail": text[-1500:]}) + "\n")
            self.log.flush()
            if ok:
                return obj["verdicts"]
        return None


# ------------------------------------------------------------------ training

def setup_device(dev):
    if dev == "cuda":
        torch.set_default_device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.8)


def name_tensor(names):
    ns = [[BA.B2I[b] for b in n.encode()] for n in names]
    L = max(map(len, ns))
    return torch.tensor([n + [0] * (L - len(n)) for n in ns])


def plan_targets(syms):
    ys = [[SYM.index(s) for s in v] for v in syms]
    T = max(map(len, ys))
    y = torch.tensor([v + [0] * (T - len(v)) for v in ys])
    return torch.cat([torch.full((len(ys), 1), BA.START), y[:, :-1]], 1), y


def lr_at(base, s, steps):
    return base * min(1.0, (s + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))


GUARD_EVERY = 25


def train_plan_ok57(net, train):
    """set of train-task indices whose greedy plan equals the TRAINING plan (training data only, never eval gold)."""
    net.eval()
    ok = set()
    for i in range(0, len(train), 256):
        ch = train[i:i + 256]
        for j, (row, t) in enumerate(zip(net.greedy(BA.name_bytes(ch)).tolist(), ch)):
            sy = [SYM[k] for k in row]
            if (sy[:sy.index("<eos>") + 1] if "<eos>" in sy else sy) == blocks_of(*t):
                ok.add(i + j)
    net.train()
    return frozenset(ok)


def train_block_ok57(net, pool):
    """set of train-pool block indices the writer gets right: exact text, or (bwrap, 40 inputs) the same behaviour as
    the training block. Behaviour, not text: base s6 had a block with different text that still passes; a text-only
    guard dropped it and lost 4.7 pp on test_a (all test_a blocks are in this pool, so a behavioural superset of the
    pool cannot lose test_a). Training data only."""
    net.eval()
    tx = net.write(torch.tensor([BW.block_key(*b)[0] for b in pool]), torch.tensor([BW.block_key(*b)[1] for b in pool]))
    net.train()
    ok = set()
    for i, (x, b) in enumerate(zip(tx, pool)):
        if x == BW.block_text(*b):
            ok.add(i)
        else:
            k = (b, x)
            if k not in _SAME57:
                _SAME57[k] = block_same(x, b)
            if _SAME57[k]:
                ok.add(i)
    return frozenset(ok)


_SAME57 = {}


class Guard57:
    """No-regression guard on ORIGINAL training data, identical for every arm (N/J/R/Q/O). Every GUARD_EVERY steps it
    snapshots the net if every train item the base got exactly right is still exactly right (superset, not count:
    a count rule let s6 swap one block for another and lose 4.7 pp on test_a); at the end it restores the LAST such snapshot (base
    itself if none). Why: the ternary d32 writer's test_a is a step function of ~3 marginal train blocks (one block
    flip = ~4 pp), and continued training flips them at random late steps (N at lr 1e-4, 400 steps, raw: s5 -4.1 pp,
    s6 -4.7 pp; guarded: 0.0 / 0.0). Selection uses training data only, never eval gold."""

    def __init__(self, net, score):
        self.net, self.score = net, score
        self.base = score()
        self.best, self.step, self.last = copy.deepcopy(net.state_dict()), 0, self.base

    def check(self, step):
        if step % GUARD_EVERY == 0:
            v = self.score()
            if v >= self.base:                          # frozenset >= : superset
                self.best, self.step, self.last = copy.deepcopy(self.net.state_dict()), step, v

    def finish(self, loss):
        raw = self.score()
        self.net.load_state_dict(self.best)
        return {"loss": loss, "train_ok_base": len(self.base), "train_ok_raw": len(raw), "raw_lost": len(self.base - raw),
                "train_ok_kept": len(self.last), "kept_step": self.step}


def cont_planner(net, train, rows, seed, steps, mix, lr):
    """E steps; batch 48; the LAST round(48*mix) slots come from rows when rows exist. Draw order is identical across
    arms (both RNGs always consumed), so arms are paired."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    ro, rx, nx = random.Random(seed * 1000 + 11), random.Random(seed * 1000 + 12), round(48 * mix)
    g = Guard57(net, lambda: train_plan_ok57(net, train))
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(lr, s, steps)
        bo = [train[ro.randrange(len(train))] for _ in range(48)]
        ix = [rx.random() for _ in range(nx)]
        names = [S2.name_of(*t) for t in bo]
        syms = [blocks_of(*t) for t in bo]
        if rows:
            for j, u in enumerate(ix):
                r = rows[int(u * len(rows))]
                names[48 - nx + j], syms[48 - nx + j] = r["name"], r["syms"]
        y_in, y = plan_targets(syms)
        loss = F.cross_entropy(net(name_tensor(names), y_in).transpose(1, 2), y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        g.check(s + 1)
    return g.finish(round(loss.item(), 5))


def cont_writer(net, pool, rows, seed, steps, mix, lr):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    ro, rx, nx = random.Random(seed * 1000 + 21), random.Random(seed * 1000 + 22), round(64 * mix)
    g = Guard57(net, lambda: train_block_ok57(net, pool))
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(lr, s, steps)
        bo = [pool[ro.randrange(len(pool))] for _ in range(64)]
        ix = [rx.random() for _ in range(nx)]
        keys = [BW.block_key(*b) for b in bo]
        txt = [BW.block_text(*b) for b in bo]
        if rows:
            for j, u in enumerate(ix):
                r = rows[int(u * len(rows))]
                keys[64 - nx + j], txt[64 - nx + j] = (r["op"], r["dig"]), r["text"]
        ys = [[BW.C2I[c] for c in t] + [BW.END] for t in txt]
        T = max(map(len, ys))
        y = torch.tensor([v + [-100] * (T - len(v)) for v in ys])
        y_in = torch.cat([torch.full((64, 1), len(BW.CH) + 1), y[:, :-1].clamp(min=0)], 1)
        lo = net(torch.tensor([k[0] for k in keys]), torch.tensor([k[1] for k in keys]), y_in)
        loss = F.cross_entropy(lo.transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        g.check(s + 1)
    return g.finish(round(loss.item(), 5))


def eval_sets(planner, writer, sets):
    """-> {set: per-item pass list} (gold grade: eval only)."""
    res = {}
    for k, tasks in sets.items():
        items = run_belt(planner, writer, tasks)
        sc = S2.grade(tasks, [[it["src"]] for it in items])
        res[k] = [int(s[0] == 1.0) for s in sc]
        res[k + ":plan_exact"] = [int(it["plan"] is not None and blocks_of(*it["plan"]) == blocks_of(*t))
                                  for it, t in zip(items, tasks)]
        res[k + ":text_exact"] = [int(it["plan"] is not None and all(w == BW.block_text(*tuple(b)) for w, b in
                                                                     zip(it["texts"], it["blocks"]))) for it in items]
    return res


# ------------------------------------------------------------------ phases

def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def eval_tasks(sets, data, smoke):
    d = {"len5": sets["eval_len5"], "len6": sets["eval_len6"], "test_a": data.sets["test_a"], "len3": data.sets["len3"],
         "len4": data.sets["len4"]}
    if smoke:
        d = {k: v[:20] for k, v in d.items()}
        d["test_a"] = data.sets["test_a"]             # positive control needs the full set
    return d


def cmd_base(a):
    # review S5: base nets are trained on CPU on purpose (the known CPU models: seeds 0/5/6 were chosen there and P0
    # is an exact CPU number); the GPU is used by the Qwen judge and by the arm training/eval.
    data = Data()
    sets = build_sets(data, a.smoke)
    json.dump(sets, open(f"{a.out}/sets.json", "w"))
    pool_writer = sorted({b for t in data.sets["train"] for b in BW.blocks_of_task(t)})
    for s in a.seeds:
        f = f"{a.out}/base_s{s}.json"
        if os.path.exists(f):
            print(f"base s{s} cached", flush=True)
            continue
        t0 = time.time()
        planner = BW.train_planner(data, s, 1500)
        writer, pool, wloss = BW.train_writer(data.sets["train"], WRITER_D, WRITER_TER, s)
        assert pool == pool_writer
        torch.save({"planner": planner.state_dict(), "writer": writer.state_dict()}, f"{a.out}/base_s{s}.pt")
        items = run_belt(planner, writer, sets["pool"])
        outs = run_srcs([(it["src"], it["task"]) for it in items])
        sc = S2.grade([it["task"] for it in items], [[it["src"]] for it in items])
        for i, (it, o, g) in enumerate(zip(items, outs, sc)):
            it["id"] = f"s{s}_p{i}"
            it["outs"], it["flags"] = o, flags_of(it["src"], o, it["task"])
            it["gold"] = {"score": g[0], **truth_of(it, g[0])}           # REPORT ONLY, never read by the judge path
        ev = eval_sets(planner, writer, eval_tasks(sets, data, a.smoke))
        res = {"seed": s, "train_s": round(time.time() - t0, 1), "writer_loss": wloss, "pool": items, "eval": ev,
               "eval_mean": {k: round(sum(v) / len(v), 4) for k, v in ev.items()}}
        json.dump(res, open(f, "w"))
        tr = [it["gold"]["stage"] for it in items]
        print(json.dumps({"base": s, "s": res["train_s"], "eval": res["eval_mean"],
                          "pool_truth": {k: tr.count(k) for k in ("none", "planner", "writer")}}), flush=True)
        if sum(t == "planner" for t in tr) < (2 if a.smoke else 8):
            print(f"ABORT: seed {s} has too few natural planner failures in the pool", flush=True)
            sys.exit(4)
        if s == 0:
            print(f"POSITIVE CONTROL base s0 test_a pass1 {res['eval_mean']['test_a']} (bytewriter18 d32_ter s0 CPU "
                  f"0.8703; device {a.device})", flush=True)


def pool_items(a, s):
    items = json.load(open(f"{a.out}/base_s{s}.json"))["pool"]
    return items[:a.max_items] if a.max_items else items


def judge_view(it):
    """the only fields a judge path may read: no 'gold'."""
    return {k: it[k] for k in ("id", "task", "syms", "plan", "blocks", "texts", "src", "outs", "flags")}


def cmd_qwen(a):
    q = Qwen(a.qwen_model, a.device, a.mem_frac)
    sysq = RULE + Q_ROLE
    for s in a.seeds:
        path = f"{a.out}/qwen_s{s}.jsonl"
        done = {r["id"] for r in jl(path)}
        f = open(path, "a")
        for it in map(judge_view, pool_items(a, s)):
            if it["id"] in done:
                continue
            t0 = time.time()
            raw = q.ask(sysq, evidence(it, it["outs"], it["flags"], it["id"]), 700)
            v = norm_verdict(last_json(raw), len(it["blocks"]))
            row = {"id": it["id"], "qwen": v, "qwen_valid": v is not None, "qwen_raw": raw[-1500:],
                   "qwen_s": round(time.time() - t0, 1)}
            fx = build_fix(it, v)
            row["qfix"] = fx
            if "code" in fx:
                g = gate_bwrap([(it, fx)])[0]
                row["qfix_flags"] = g["flags"]
                if not g["flags"]:
                    rraw = q.ask(RULE + "\n\n" + REPASS + ' Answer with ONE JSON object: {"pass": bool, "score": int}.',
                                 repass_evidence(it, fx["code"], g["outs"], it["id"]), 80)
                    rv = last_json(rraw)
                    row["qrepass"], row["qrepass_raw"] = rv, rraw[-300:]
            row["q_accepted"] = bool("code" in fx and not row.get("qfix_flags") and repass_ok(row.get("qrepass")))
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps({"id": it["id"], "valid": row["qwen_valid"], "stage": (v or {}).get("stage"),
                              "loc": (v or {}).get("location"), "fix": fx.get("reject", "built"),
                              "acc": row["q_accepted"], "s": row["qwen_s"]}), flush=True)


def needs_sonnet(it, qrow):
    v = qrow["qwen"]
    if it["flags"] or v is None or v["stage"] != "none" or v["score"] < 100:
        return "triage"
    return "audit" if int(hashlib.md5(it["id"].encode()).hexdigest(), 16) % 4 == 0 else ""


def final_of(qv, sv_raw, n_blocks):
    """his rule: Sonnet agrees -> Qwen's verdict, else Sonnet's. Defect guards logged in 'note'."""
    sv = norm_verdict(sv_raw, n_blocks) if isinstance(sv_raw, dict) else None
    agree = bool(isinstance(sv_raw, dict) and sv_raw.get("agree_with_qwen") is True)
    if sv_raw is None:
        return None, "sonnet_missing", None
    if sv is None and not (agree and qv is not None):
        return None, "sonnet_invalid", False
    if qv is None:
        return sv, "qwen_invalid" + ("_sonnet_said_agree" if agree else ""), False
    if agree:
        if sv is not None and (sv["stage"] != qv["stage"] or (qv["stage"] != "none" and sv["location"] != qv["location"])):
            return sv, "agree_inconsistent", False
        return qv, "agree" if sv is not None else "agree_unverified", True
    return sv, "disagree", False


def cmd_sonnet(a):
    S = Sonnet(a.out, a.max_sonnet_calls)
    for s in a.seeds:
        items = {it["id"]: judge_view(it) for it in pool_items(a, s)}
        qrows = {r["id"]: r for r in jl(f"{a.out}/qwen_s{s}.jsonl")}
        path = f"{a.out}/final_s{s}.jsonl"
        if os.path.exists(path):
            print(f"final s{s} cached", flush=True)
            continue
        rows = {}
        send = []
        missing = [i for i in items if i not in qrows]
        assert not missing, f"qwen phase incomplete for seed {s}: {len(missing)} items"
        for iid, it in items.items():
            q = qrows[iid]
            why = needs_sonnet(it, q)
            rows[iid] = {"id": iid, "why": why, "qwen_valid": q["qwen_valid"]}
            if why:
                send.append(iid)
            else:
                rows[iid].update(final=q["qwen"], note="qwen_pass_not_sent", qwen_ok=True)
        for b in range(0, len(send), a.batch):
            ids = send[b:b + a.batch]
            body = []
            for iid in ids:
                it, qv = items[iid], qrows[iid]["qwen"]
                first = json.dumps(qv) if qv is not None else (
                    "(INVALID: the first judge produced no valid JSON; agree_with_qwen must be false and you give the "
                    "full verdict)")
                body.append(evidence(it, it["outs"], it["flags"], iid) + "\nFIRST JUDGE (Qwen) VERDICT: " + first)
            prompt = RULE + S_ROLE + "\n\n" + "\n\n=====\n\n".join(body)
            vs = S.ask(prompt, f"judge_s{s}_{b}") or []
            byid = {str(v.get("id")): v for v in vs if isinstance(v, dict)}
            for iid in ids:
                sv_raw = byid.get(iid)
                fin, note, qok = final_of(qrows[iid]["qwen"], sv_raw, len(items[iid]["blocks"]))
                rows[iid].update(sonnet_raw=sv_raw, final=fin, note=note, qwen_ok=qok)
            print(json.dumps({"seed": s, "batch": b, "calls": S.used, "notes": [rows[i]["note"] for i in ids]}),
                  flush=True)
        # final fix gate: bwrap no flags, then the FINAL judge (Sonnet) re-passes
        cand = []
        for iid, r in rows.items():
            fx = build_fix(items[iid], r.get("final"))
            r["fix"] = fx
            if "code" in fx:
                cand.append(iid)
        gates = gate_bwrap([(items[i], rows[i]["fix"]) for i in cand])
        rp = []
        for iid, g in zip(cand, gates):
            rows[iid]["fix_flags"] = g["flags"]
            if not g["flags"]:
                rp.append((iid, g["outs"]))
        for b in range(0, len(rp), a.batch):
            ch = rp[b:b + a.batch]
            prompt = (RULE + "\n\n" + REPASS + ' Answer with ONE JSON object only: {"verdicts": [{"id": str, "pass": '
                      'bool, "score": int}, ...]}.\n\n' + "\n\n=====\n\n".join(
                          repass_evidence(items[i], rows[i]["fix"]["code"], o, i) for i, o in ch))
            vs = S.ask(prompt, f"repass_s{s}_{b}") or []
            byid = {str(v.get("id")): v for v in vs if isinstance(v, dict)}
            for iid, _ in ch:
                rows[iid]["repass"] = byid.get(iid)
        for r in rows.values():
            r["accepted"] = bool("code" in r["fix"] and not r.get("fix_flags") and repass_ok(r.get("repass")))
        with open(path, "w") as f:
            for r in rows.values():
                f.write(json.dumps(r) + "\n")
        print(json.dumps({"seed": s, "sent": len(send), "accepted": sum(r["accepted"] for r in rows.values()),
                          "calls_used": S.used}), flush=True)
        miss = sum(r.get("note") == "sonnet_missing" or (r.get("fix_flags") == [] and r.get("repass") is None)
                   for r in rows.values())
        if miss:                                        # review B1: a capped/failed Sonnet must not look like a result
            os.rename(path, path + ".incomplete")
            print(f"ABORT: seed {s}: {miss} items without a Sonnet verdict/re-pass (cap {S.cap}, used {S.used})",
                  flush=True)
            sys.exit(5)


def rows_for(a, s, sets, base_items):
    """-> {arm: rows}. J/Q from judges (gold-free); R from the generator; O from gold (upper bound)."""
    fin = {r["id"]: r for r in jl(f"{a.out}/final_s{s}.jsonl")}
    qr = {r["id"]: r for r in jl(f"{a.out}/qwen_s{s}.jsonl")}
    ids = {it["id"] for it in base_items}
    assert ids <= set(fin) and ids <= set(qr), f"judge files incomplete for seed {s}"  # review B1
    J = [x for it in base_items if fin.get(it["id"], {}).get("accepted") for x in fin[it["id"]]["fix"]["rows"]]
    Q = [x for it in base_items if qr.get(it["id"], {}).get("q_accepted") for x in qr[it["id"]]["qfix"]["rows"]]
    nP = sum(x["module"] == "planner" for x in J)
    nW = len(J) - nP
    g = random.Random(s * 7 + 3)
    rnd = sets["rand"]
    R = [{"module": "planner", "name": S2.name_of(*t), "syms": blocks_of(*t)} for t in rnd[:nP]]
    blks = [b for t in rnd for b in BW.blocks_of_task(t)]
    for _ in range(nW):
        b = blks[g.randrange(len(blks))]
        op, dg = BW.block_key(*b)
        R.append({"module": "writer", "op": op, "dig": dg, "text": BW.block_text(*b), "block": list(b)})
    O = []
    for it in base_items:                       # gold enters ONLY here (oracle arm) and in the report
        tr = it["gold"]
        if tr["stage"] == "planner":
            O.append({"module": "planner", "name": S2.name_of(*it["task"]), "syms": blocks_of(*it["task"])})
        elif tr["stage"] == "writer":
            for k in tr["wrong"]:
                b = tuple(it["blocks"][k])
                op, dg = BW.block_key(*b)
                O.append({"module": "writer", "op": op, "dig": dg, "text": BW.block_text(*b), "block": list(b)})
    g.shuffle(O)
    O = [x for x in O if x["module"] == "planner"][:nP] + [x for x in O if x["module"] == "writer"][:nW]  # review S4
    return {"N": [], "J": J, "R": R, "Q": Q, "O": O}


def cmd_train(a):
    setup_device(a.device)
    data = Data()
    sets = json.load(open(f"{a.out}/sets.json"))
    pool_w = sorted({b for t in data.sets["train"] for b in BW.blocks_of_task(t)})
    ev_t = eval_tasks(sets, data, a.smoke)
    for s in a.seeds:
        base = json.load(open(f"{a.out}/base_s{s}.json"))
        arm_rows = rows_for(a, s, sets, pool_items(a, s))
        ck = torch.load(f"{a.out}/base_s{s}.pt", map_location=a.device)
        fb = f"{a.out}/arm_B_s{s}.json"             # review-2 #2: base re-evaluated on THIS device, 0 steps
        if not os.path.exists(fb):
            planner = BA.Net("addr", gate_bias=-4.0)
            writer = BW.Writer(WRITER_D, WRITER_TER)
            planner.load_state_dict(ck["planner"])
            writer.load_state_dict(ck["writer"])
            ev = eval_sets(planner, writer, ev_t)
            res = {"arm": "B", "seed": s, "n_rows": 0, "device": a.device, "eval": ev,
                   "eval_mean": {k: round(sum(v) / len(v), 4) for k, v in ev.items()}}
            json.dump(res, open(fb, "w"))
            print(f"POSITIVE CONTROL {a.device} base s{s} test_a pass1 {res['eval_mean']['test_a']} "
                  f"(base phase CPU {base['eval_mean']['test_a']})", flush=True)
        for arm in ARMS:
            f = f"{a.out}/arm_{arm}_s{s}.json"
            if os.path.exists(f):
                continue
            rows = arm_rows[arm]
            json.dump(rows, open(f"{a.out}/rows_{arm}_s{s}.json", "w"))
            planner = BA.Net("addr", gate_bias=-4.0)
            writer = BW.Writer(WRITER_D, WRITER_TER)
            planner.load_state_dict(ck["planner"])
            writer.load_state_dict(ck["writer"])
            t0 = time.time()
            pr = [r for r in rows if r["module"] == "planner"]
            wr = [r for r in rows if r["module"] == "writer"]
            lp = cont_planner(planner, data.sets["train"], pr, s, a.extra_steps, a.mix if pr else 0.0, a.lr)
            lw = cont_writer(writer, pool_w, wr, s, a.extra_steps, a.mix if wr else 0.0, a.lr)
            ev = eval_sets(planner, writer, ev_t)
            res = {"arm": arm, "seed": s, "n_rows": len(rows), "n_planner_rows": len(pr), "n_writer_rows": len(wr),
                   "extra_steps": a.extra_steps, "mix": a.mix, "lr": a.lr, "loss_p": lp, "loss_w": lw,
                   "train_s": round(time.time() - t0, 1), "eval": ev,
                   "eval_mean": {k: round(sum(v) / len(v), 4) for k, v in ev.items()}}
            json.dump(res, open(f, "w"))
            print(json.dumps({k: v for k, v in res.items() if k != "eval"}), flush=True)
        print(json.dumps({"seed": s, "base_eval": base["eval_mean"]}), flush=True)


def boot_diff(A, B, reps=2000, seed=0):
    """A, B: {seed: [0/1 per item]} paired -> mean diff and 95% CI (items resampled within seed)."""
    g = random.Random(seed)
    ss = sorted(A)
    d = [sum(A[s]) / len(A[s]) - sum(B[s]) / len(B[s]) for s in ss]
    m = sum(d) / len(d)
    bs = []
    for _ in range(reps):
        tot = 0.0
        for s in ss:
            n = len(A[s])
            idx = [g.randrange(n) for _ in range(n)]
            tot += sum(A[s][i] - B[s][i] for i in idx) / n
        bs.append(tot / len(ss))
    bs.sort()
    return {"diff": round(m, 4), "ci95": [round(bs[int(0.025 * reps)], 4), round(bs[int(0.975 * reps)], 4)],
            "per_seed": [round(x, 4) for x in d], "seeds_positive": sum(x > 0 for x in d)}


def block_same(text, block):
    """report only: a written block does the same as the gold block on the 40 inputs (style may differ)."""
    def src(t):
        return "def _blk(xs, k=0):\n" + "".join(f"    {l}\n" for l in t.split("\n") if l) + \
            ("" if block[1] else "    return xs\n")
    o = S2.run_many([(src(text), "_blk", 2), (src(BW.block_text(*block)), "_blk", 2)])
    return o[0] is not None and o[0] == o[1]


def cmd_report(a):
    out = {"seeds": a.seeds, "arms": {}, "diffs": {}, "judges": {}, "labels": {}}
    res = {arm: {s: json.load(open(f"{a.out}/arm_{arm}_s{s}.json")) for s in a.seeds
                 if os.path.exists(f"{a.out}/arm_{arm}_s{s}.json")} for arm in ARMS + ["B"]}
    base = {s: json.load(open(f"{a.out}/base_s{s}.json")) for s in a.seeds}
    res["base"] = {s: {"eval": b["eval"], "n_rows": 0} for s, b in base.items()}
    for arm, by in res.items():
        if not by:
            continue
        ks = list(next(iter(by.values()))["eval"])
        out["arms"][arm] = {k: round(sum(sum(r["eval"][k]) / len(r["eval"][k]) for r in by.values()) / len(by), 4)
                            for k in ks}
        out["arms"][arm]["n_rows"] = [r["n_rows"] for r in by.values()]
    for x, y in (("J", "R"), ("J", "N"), ("J", "Q"), ("O", "J"), ("R", "N"), ("N", "B"), ("N", "base"), ("B", "base"),
                 ("Q", "N")):
        if len(res.get(x, {})) == len(a.seeds) and len(res.get(y, {})) == len(a.seeds):
            out["diffs"][f"{x}-{y}"] = {}
            for k in ("len5", "len6", "test_a", "len3", "len4"):
                out["diffs"][f"{x}-{y}"][k] = boot_diff({s: res[x][s]["eval"][k] for s in a.seeds},
                                                        {s: res[y][s]["eval"][k] for s in a.seeds})
            out["diffs"][f"{x}-{y}"]["len56"] = boot_diff(
                {s: res[x][s]["eval"]["len5"] + res[x][s]["eval"]["len6"] for s in a.seeds},
                {s: res[y][s]["eval"]["len5"] + res[y][s]["eval"]["len6"] for s in a.seeds})
    # judge quality vs gold (report only)
    J = {"qwen": [], "sonnet_own": [], "final": []}
    lab = {"J": [], "Q": []}
    for s in a.seeds:
        items = {it["id"]: it for it in pool_items(a, s)}
        qr = {r["id"]: r for r in jl(f"{a.out}/qwen_s{s}.jsonl")}
        fr = {r["id"]: r for r in jl(f"{a.out}/final_s{s}.jsonl")}
        for iid, it in items.items():
            tr = it["gold"]
            so = (fr.get(iid) or {}).get("sonnet_raw")
            whos = [("qwen", (qr.get(iid) or {}).get("qwen")), ("final", (fr.get(iid) or {}).get("final"))]
            if (fr.get(iid) or {}).get("why"):          # Sonnet's own verdict, only on items it was sent
                whos.append(("sonnet_own", norm_verdict(so, len(it["blocks"])) if isinstance(so, dict) else None))
            for who, v in whos:
                J[who].append({"truth": tr["stage"], "loc": tr["loc"], "wrong": tr["wrong"], "flags": it["flags"],
                               "stage": (v or {}).get("stage"), "location": (v or {}).get("location"),
                               "valid": v is not None, "score": (v or {}).get("score"), "gold_score": tr["score"]})
            for arm, ok, fx in (("J", (fr.get(iid) or {}).get("accepted"), (fr.get(iid) or {}).get("fix")),
                                ("Q", (qr.get(iid) or {}).get("q_accepted"), (qr.get(iid) or {}).get("qfix"))):
                if not ok:
                    continue
                for row in fx["rows"]:
                    if row["module"] == "planner":
                        good = row["syms"] == blocks_of(*it["task"])
                        lab[arm].append({"module": "planner", "exact": good})
                    else:
                        b = tuple(row["block"])
                        good = row["text"] == BW.block_text(*b)
                        lab[arm].append({"module": "writer", "exact": good})
                        lab[arm].append({"module": "writer_semantic", "exact": block_same(row["text"], b)})
                sc = S2.grade([it["task"]], [[fx["code"]]])[0][0]
                lab[arm].append({"module": "program", "exact": sc == 1.0})
    rate = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
    for who, rs in J.items():
        fail = [r for r in rs if r["truth"] != "none"]
        out["judges"][who] = {
            "n": len(rs), "valid": rate([r["valid"] for r in rs]),
            "pass_agree": rate([(r["stage"] == "none") == (r["truth"] == "none") for r in rs if r["valid"]]),
            "stage_acc_on_fails": rate([r["stage"] == r["truth"] for r in fail]),
            "loc_acc_on_fails": rate([r["stage"] == r["truth"] and r["location"] == r["loc"] for r in fail]),
            "loc_any_wrong_block": rate([r["stage"] == "writer" and r["location"] in r["wrong"]
                                         for r in fail if r["truth"] == "writer"]),
            "score_mae_01": rate([abs(r["score"] / 100 - r["gold_score"]) for r in rs if r["valid"]]),
            "n_fails": len(fail)}
    fail_tr = [r["truth"] for r in J["qwen"] if r["truth"] != "none"]
    out["judges"]["constant_baseline"] = {k: rate([t == k for t in fail_tr]) for k in ("planner", "writer")}
    # review S1: no-LLM rule "syntax/load flag -> writer, else planner" (location: first block that fails to parse)
    fr_ = [r for r in J["qwen"] if r["truth"] != "none"]
    out["judges"]["flag_rule_baseline"] = {"stage_acc_on_fails": rate([
        ("writer" if {"syntax_error", "load_fail"} & set(r["flags"]) else "planner") == r["truth"] for r in fr_])}
    for arm, xs in lab.items():
        out["labels"][arm] = {m: {"n": len([x for x in xs if x["module"] == m]),
                                  "precision": rate([x["exact"] for x in xs if x["module"] == m])}
                              for m in ("planner", "writer", "writer_semantic", "program")}
    notes = {}
    for s in a.seeds:
        for r in jl(f"{a.out}/final_s{s}.jsonl"):
            notes[r.get("note")] = notes.get(r.get("note"), 0) + 1
    out["final_notes"] = notes
    out["sonnet_calls"] = len(jl(f"{a.out}/sonnet_calls.jsonl"))
    # review-2 #4/#5: the writer is keyed by (op, digit) over the same 50 pool blocks, so its effective n is blocks x
    # seeds, not items; and Guard57 can revert a module to base (then J/R/N share that module for that seed).
    out["modules_moved"] = {arm: {s: {"planner_kept_step": r["loss_p"]["kept_step"], "writer_kept_step":
                                      r["loss_w"]["kept_step"], "writer_pool_blocks_ok": r["loss_w"]["train_ok_kept"],
                                      "planner_train_ok": r["loss_p"]["train_ok_kept"]}
                                  for s, r in by.items()} for arm, by in res.items() if arm in ARMS}
    json.dump(out, open(f"{a.out}/verdict.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["base", "qwen", "sonnet", "train", "report"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,5,6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-items", type=int, default=0, help="judge only the first N pool items per seed (0 = all)")
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--mem-frac", type=float, default=0.8)
    ap.add_argument("--max-sonnet-calls", type=int, default=150)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--extra-steps", type=int, default=400)
    ap.add_argument("--mix", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    a.seeds = [int(x) for x in a.seeds.split(",")]
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"base": cmd_base, "qwen": cmd_qwen, "sonnet": cmd_sonnet, "train": cmd_train, "report": cmd_report}[a.phase](a)


if __name__ == "__main__":
    main()
