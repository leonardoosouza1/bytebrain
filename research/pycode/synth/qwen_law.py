"""Segment-pointer law in frozen Qwen2.5-Coder-0.5B (proposer 1, registered 2026-10-22). Stage g0 only so far.

g0 (gate, CPU, teacher forced): names of n single-line units + a non-empty final (`times_3_then_sort_then_sum`),
prompt `def NAME(xs):\n` + the reference lines before unit j. Margin(j) = logP(line of unit j) - logP(line of the
next name segment: unit j+1, or the final for j = n), summed over the line tokens. Control: the same prefix under
the neutral name `def f(xs):` (what the prefix alone predicts). Rule: G0 passes if margin > 0 on >= 85% of
unit-lines for n = 1..3; n = 4, 5 reported (lengths never seen in synth2 train). Stages pointer/copy/counter
(head re-aim) run only if G0 passes.

Usage: qwen_law.py g0 --out DIR [--per-n 60 --threads 4]
       qwen_law.py smoke --out DIR
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from synth2 import FINAL, HELD, NUMS, PARAM, PLAIN, UNITS, name_of  # noqa: E402

MODEL = ("/media/llm-data/models/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/"
         "ea3f2471cf1b1f0db85067f1ef93848e38e88c25")
SINGLE = [f"{op}_{n}" for op in PARAM for n in NUMS if f"{op}_{n}" not in HELD] + list(PLAIN)
FINALS = [f for f in FINAL if f]


def names(n, k, r):
    """k tasks of n distinct single-line seen units + a non-empty final, distinct unit sets (k capped at the number
    of distinct sets: n = 1 has only len(SINGLE) = 34)."""
    k = min(k, math.comb(len(SINGLE), n))
    out, seen = [], set()
    while len(out) < k:
        u = r.sample(SINGLE, n)
        if frozenset(u) in seen:
            continue
        seen.add(frozenset(u))
        out.append((u, r.choice(FINALS)))
    return out


def line(u):
    return f"    {UNITS[u][0]}\n"


def queries(task, neutral):
    """-> [(prompt, true_line, other_line, j)] one per unit."""
    u, f = task
    head = "def f(xs):\n" if neutral else f"def {name_of(u, f)}(xs):\n"
    seg = [line(x) for x in u] + [f"    {FINAL[f]}\n"]
    return [(head + "".join(seg[:j]), seg[j], seg[j + 1], j) for j in range(len(u))]


@torch.no_grad()
def line_logp(model, tok, pairs, batch=16):
    """pairs [(prompt, continuation)] -> sum log P(continuation tokens | prompt). Right padding: causal positions
    before the pad are unaffected. The continuation starts a new line, so its tokens do not merge with the prompt
    (asserted)."""
    enc = []
    for p, c in pairs:
        a, b = tok(p)["input_ids"], tok(p + c)["input_ids"]
        assert b[:len(a)] == a, (p, c)
        enc.append((b, len(a)))
    out = [0.0] * len(enc)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for s in range(0, len(order), batch):
        ch = order[s:s + batch]
        L = max(len(enc[i][0]) for i in ch)
        ids = torch.full((len(ch), L), pad, dtype=torch.long)
        mask = torch.zeros(len(ch), L, dtype=torch.long)
        for r, i in enumerate(ch):
            ids[r, :len(enc[i][0])] = torch.tensor(enc[i][0])
            mask[r, :len(enc[i][0])] = 1
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
        for r, i in enumerate(ch):
            b, a = enc[i]
            tgt = torch.tensor(b[a:])
            out[i] = lp[r, a - 1:len(b) - 1].gather(1, tgt[:, None]).sum().item()
    return out


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval(), tok


def smoke(a):
    model, tok = load()
    r = random.Random(0)
    ts = names(2, 6, r)
    pairs = [(p, c) for t in ts for p, c, _, _ in queries(t, False)]
    batched = line_logp(model, tok, pairs, batch=8)
    single = line_logp(model, tok, pairs, batch=1)
    with torch.no_grad():                       # HF loss over the continuation tokens == my sum
        p, c = pairs[0]
        a_ids, b_ids = tok(p)["input_ids"], tok(p + c)["input_ids"]
        lab = torch.tensor([[-100] * len(a_ids) + b_ids[len(a_ids):]])
        hf = -model(input_ids=torch.tensor([b_ids]), labels=lab).loss.item() * (len(b_ids) - len(a_ids))
    res = {"batch_vs_single_max_abs": max(abs(x - y) for x, y in zip(batched, single)),
           "hf_loss_vs_manual": abs(hf - single[0]),
           "same_line_margin": line_logp(model, tok, [pairs[0], pairs[0]])}
    res["ok"] = res["batch_vs_single_max_abs"] < 1e-3 and res["hf_loss_vs_manual"] < 1e-3
    print(json.dumps(res))
    json.dump(res, open(f"{a.out}/smoke_g0.json", "w"), indent=1)


def g0(a):
    t0 = time.perf_counter()
    model, tok = load()
    r = random.Random(7)
    res = {"per_n": a.per_n}
    for n in range(1, 6):
        ts = names(n, a.per_n, r)
        row, margins = {}, {}
        for cond in ("named", "neutral"):
            q = [x for t in ts for x in queries(t, cond == "neutral")]
            lt = line_logp(model, tok, [(p, c) for p, c, _, _ in q])
            lo = line_logp(model, tok, [(p, o) for p, _, o, _ in q])
            m = [x - y for x, y in zip(lt, lo)]
            margins[cond] = m
            by_j = {}
            for (_, _, _, j), v in zip(q, m):
                by_j.setdefault(j + 1, []).append(v)
            row[cond] = {"lines": len(m), "margin_pos": round(sum(v > 0 for v in m) / len(m), 3),
                         "mean_margin": round(sum(m) / len(m), 3),
                         "margin_pos_by_unit": {j: round(sum(v > 0 for v in vs) / len(vs), 3)
                                                for j, vs in sorted(by_j.items())}}
        # paired effect of the name (review 2026-10-23): same query, named margin minus neutral margin
        d = [x - y for x, y in zip(margins["named"], margins["neutral"])]
        row["paired_name_effect_pos"] = round(sum(v > 0 for v in d) / len(d), 3)
        row["paired_name_effect_mean"] = round(sum(d) / len(d), 3)
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        res[f"n{n}"] = row
        print(json.dumps({f"n{n}": row}), flush=True)
    pos = [res[f"n{n}"]["named"]["margin_pos"] * res[f"n{n}"]["named"]["lines"] for n in (1, 2, 3)]
    lines = [res[f"n{n}"]["named"]["lines"] for n in (1, 2, 3)]
    res["G0_share_n1_3"] = round(sum(pos) / sum(lines), 3)
    pp = [res[f"n{n}"]["paired_name_effect_pos"] * res[f"n{n}"]["named"]["lines"] for n in (1, 2, 3)]
    res["G0_paired_share_n1_3"] = round(sum(pp) / sum(lines), 3)
    # registered gate (named share >= 0.85) + declared addition after review: the name must beat the neutral prefix
    # on >= 0.85 of the same queries, else a prior/length effect could pass the gate without the name
    res["G0_pass_registered"] = res["G0_share_n1_3"] >= 0.85
    res["G0_pass"] = res["G0_pass_registered"] and res["G0_paired_share_n1_3"] >= 0.85
    print(json.dumps({k: res[k] for k in ("G0_share_n1_3", "G0_paired_share_n1_3", "G0_pass")}), flush=True)
    json.dump(res, open(f"{a.out}/g0.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "g0"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    smoke(a) if a.cmd == "smoke" else g0(a)


if __name__ == "__main__":
    main()
