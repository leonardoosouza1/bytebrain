"""Scorer for every writer arm of ledger #55 / I6 (10-27) + the trivial floors. CPU + bwrap only, no torch.

An arm writes a candidates file {"arm", "seed", "K", "meta": {...}, "cands": {sha1(prompt)[:16]: [body, ...]}}
for the prompts it was asked about (interp55_eval.py prompts --data D -> the list). This scorer then measures, with the
SAME selector and the SAME K for every arm:

  unit mode (func_eval: 176 units whose prompt never occurs in train; also the 170 not functionally leaked):
     greedy = candidate 0 passes all grade cases (pass@1, no selection)
     sel@K  = the first of the first K candidates that passes the spec ACCEPT cases (gold siblings) is graded;
              none accepted -> candidate 0. Gold enters only through the accept examples, which are part of the spec.
     any@K  = some candidate passes grade (oracle selector; diagnostic, NOT comparable with greedy/sel)
  program mode (test_comp = 160 held variants; test_seen = 120 non-held): all units from the arm's candidates, no
     gold code. greedy = all candidate 0; sel@K = two leaf-first passes, each unit takes the first candidate that
     passes ITS accept cases given the current choice of the other units (gold-free); grade = main() grade cases.

Floors (gate 1 of bytebrain/CLAUDE.md): `const` (every body `return None`), `nn` (nearest train prompt of the same
unit name by byte 4-gram cosine; candidates = the K nearest distinct train bodies).

Usage:
  interp55_eval.py prompts --data D/data.json --out D/prompts.json
  interp55_eval.py floors  --data D/data.json --outdir D/floors [--K 8]
  interp55_eval.py score   --data D/data.json --cands FILE --out FILE.score.json [--K 8]
"""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp55_family as F  # noqa: E402
import interp55_grade as G  # noqa: E402

ORDER = ["tokenize", "apply_op", "call_builtin", "Parser.__init__", "Parser.peek", "Parser.advance",
         "parse_atom", "Parser.parse_atom", "parse_unary", "Parser.parse_unary", "parse_power", "Parser.parse_power",
         "parse_term", "Parser.parse_term", "parse_expr", "Parser.parse_expr", "parse_binary", "Parser.parse_binary",
         "evaluate", "run_line", "main"]


def sha(p):
    return hashlib.sha1(p.encode()).hexdigest()[:16]


def needed_prompts(data):
    V = data["variants"]
    ps = {fe["prompt"] for fe in data["func_eval"]}
    for k in data["test_comp"] + data["test_seen"]:
        ps |= {u["prompt"] for u in V[k]["units"]}
    return sorted(ps)


def clean_body(text):
    """model output -> body text at indent 0 (drops ``` fences and a def/class header line if present)."""
    t = text.replace("\r", "")
    t = re.sub(r"^```[a-zA-Z]*\n", "", t.strip("\n"))
    t = t.split("```")[0]
    lines = t.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and re.match(r"\s*(def|class)\s", lines[0]):
        lines.pop(0)
        if lines and re.match(r"\s*def\s", lines[0]):
            lines.pop(0)
    body = textwrap.dedent("\n".join(lines)).strip("\n")
    return (body + "\n") if body.strip() else "return None\n"


# ------------------------------------------------------------------------------------------------------ unit mode
def score_units(data, cands, K, jobs):
    V = data["variants"]
    leak = None
    lp = os.path.join(os.path.dirname(os.path.abspath(ARGS.data)), "leak.json")
    if os.path.exists(lp):
        lj = json.load(open(lp))
        leak = {(r["variant"], r["unit"]): r["n_equiv"] > 0 for r in lj["rows"]}
    items, exps, tags = [], [], []
    for fe in data["func_eval"]:
        row = V[fe["variant"]]
        u = next(x for x in row["units"] if x["prompt"] == fe["prompt"])
        cs = (cands.get(sha(fe["prompt"])) or ["return None\n"])[:K]
        for j, b in enumerate(cs):
            for which in ("accept", "grade"):
                it, ex = G.unit_item(row, u["name"], b, which)
                items.append(it)
                exps.append(ex)
                tags.append((fe["variant"], u["name"], j, which))
    got = G.run_many(items, jobs=jobs)
    R = collections.defaultdict(dict)
    for (v, n, j, w), g, e in zip(tags, got, exps):
        R[(v, n)][(j, w)] = (G.passed(g, e), G.frac(g, e))
    rows = []
    for (v, n), d in R.items():
        k = max(j for j, _ in d) + 1
        acc = [d[(j, "accept")][0] for j in range(k)]
        grd = [d[(j, "grade")][0] for j in range(k)]
        pick = next((j for j in range(k) if acc[j]), 0)
        rows.append({"variant": v, "unit": n, "greedy": grd[0], "sel": grd[pick], "any": any(grd),
                     "accepted": any(acc), "greedy_case_frac": d[(0, "grade")][1],
                     "leaked": bool(leak and leak.get((v, n)))})

    def agg(rs):
        n = len(rs)
        return {"n": n, "greedy": round(sum(r["greedy"] for r in rs) / max(1, n), 4),
                "sel": round(sum(r["sel"] for r in rs) / max(1, n), 4),
                "any": round(sum(r["any"] for r in rs) / max(1, n), 4),
                "accepted": round(sum(r["accepted"] for r in rs) / max(1, n), 4),
                "greedy_case_frac": round(sum(r["greedy_case_frac"] for r in rs) / max(1, n), 4)}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["unit"]].append(r)
    return {"all": agg(rows), "nonleaked": agg([r for r in rows if not r["leaked"]]),
            "by_unit": {k: agg(v) for k, v in sorted(by.items())}, "rows": rows}


# --------------------------------------------------------------------------------------------------- program mode
def score_programs(data, cands, K, split, jobs):
    V = data["variants"]
    ids = data[split]
    pool = {}
    for k in ids:
        for u in V[k]["units"]:
            pool[(k, u["name"])] = (cands.get(sha(u["prompt"])) or ["return None\n"])[:K]
    cur = {k: {u["name"]: pool[(k, u["name"])][0] for u in V[k]["units"]} for k in ids}

    def grade_all(sel):
        items, exps = [], []
        for k in ids:
            it, ex = G.program_item(V[k], sel[k], "grade")
            items.append(it)
            exps.append(ex)
        got = G.run_many(items, jobs=jobs)
        return [G.passed(g, e) for g, e in zip(got, exps)], [G.frac(g, e) for g, e in zip(got, exps)]
    greedy, greedy_frac = grade_all(cur)
    if K > 1:
        for _ in range(2):
            for name in ORDER:
                items, exps, tags = [], [], []
                for k in ids:
                    if name not in cur[k]:
                        continue
                    u = next(x for x in V[k]["units"] if x["name"] == name)
                    for j, b in enumerate(pool[(k, name)]):
                        bodies = dict(cur[k])
                        bodies[name] = b
                        src = F.assemble(V[k]["cfg"], bodies) + F.HARNESS
                        items.append((src, [c for c, _ in u["accept"]]))
                        exps.append([e for _, e in u["accept"]])
                        tags.append((k, j))
                got = G.run_many(items, jobs=jobs)
                best = {}
                for (k, j), g, e in zip(tags, got, exps):
                    if G.passed(g, e) and k not in best:
                        best[k] = j
                for k, j in best.items():
                    cur[k][name] = pool[(k, name)][j]
    sel, sel_frac = grade_all(cur) if K > 1 else (greedy, greedy_frac)
    nov = {k: sum(u["novel"] for u in V[k]["units"]) for k in ids}
    return {"n": len(ids), "greedy": round(sum(greedy) / len(ids), 4), "sel": round(sum(sel) / len(ids), 4),
            "greedy_case_frac": round(sum(greedy_frac) / len(ids), 4), "sel_case_frac": round(sum(sel_frac) / len(ids), 4),
            "sel_with_novel": round(sum(s for k, s in zip(ids, sel) if nov[k]) / max(1, sum(1 for k in ids if nov[k])), 4),
            "n_with_novel": sum(1 for k in ids if nov[k]),
            "rows": [{"variant": k, "greedy": g, "sel": s, "novel_units": nov[k]} for k, g, s in zip(ids, greedy, sel)]}


def score(data, cands, K, jobs, splits=("test_comp", "test_seen")):
    t0 = time.time()
    res = {"K": K, "units": score_units(data, cands, K, jobs)}
    for s in splits:
        res[s] = score_programs(data, cands, K, s, jobs)
    res["score_wall_s"] = round(time.time() - t0, 1)
    return res


def summary_line(arm, res):
    u, c = res["units"], res.get("test_comp", {})
    return (f"{arm}: unit greedy {u['all']['greedy']:.3f} sel@{res['K']} {u['all']['sel']:.3f} "
            f"(nonleaked {u['nonleaked']['sel']:.3f}, any {u['all']['any']:.3f}) | "
            f"prog comp greedy {c.get('greedy', float('nan')):.3f} sel {c.get('sel', float('nan')):.3f} | "
            f"seen sel {res.get('test_seen', {}).get('sel', float('nan')):.3f}")


# ---------------------------------------------------------------------------------------------------------- floors
def ngrams(s, n=4):
    c = collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))
    norm = math.sqrt(sum(v * v for v in c.values())) or 1.0
    return c, norm


def cos(a, b):
    (ca, na), (cb, nb) = a, b
    if len(ca) > len(cb):
        ca, cb = cb, ca
    return sum(v * cb.get(k, 0) for k, v in ca.items()) / (na * nb)


def nn_cands(data, prompts, K):
    tr = collections.defaultdict(list)
    for u in data["train_units"]:
        tr[u["name"]].append((ngrams(u["prompt"]), u["body"]))
    name_of = {}
    for k, row in data["variants"].items():
        for u in row["units"]:
            name_of[u["prompt"]] = u["name"]
    out = {}
    for p in prompts:
        q = ngrams(p)
        sc = sorted(((cos(q, g), b) for g, b in tr[name_of[p]]), key=lambda x: -x[0])
        bodies = []
        for _, b in sc:
            if b not in bodies:
                bodies.append(b)
            if len(bodies) == K:
                break
        out[sha(p)] = bodies
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "floors", "score"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--cands", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--splits", default="test_comp,test_seen")
    a = ARGS = ap.parse_args()
    data = json.load(open(a.data))
    if a.cmd == "prompts":
        ps = needed_prompts(data)
        json.dump(ps, open(a.out, "w"))
        print(len(ps), "prompts")
        return
    if a.cmd == "floors":
        os.makedirs(a.outdir, exist_ok=True)
        ps = needed_prompts(data)
        t0 = time.time()
        gold = {}
        for row in data["variants"].values():
            for u in row["units"]:
                gold[sha(u["prompt"])] = [u["body"]]
        import random
        rr = random.Random(0)
        mut = {k: (G.mutants(v[0], rr, 1) or ["return None\n"]) for k, v in gold.items()}
        # positive control: gold bodies through the same scorer (must be 1.0 everywhere); perturbed: one mutant per
        # unit (must fall well below 1.0; equivalent mutants survive)
        for arm, cands, K in (("gold", gold, 1), ("mutant", mut, 1), ("const", {sha(p): ["return None\n"] for p in ps}, 1),
                              ("nn", nn_cands(data, ps, a.K), a.K)):
            res = score(data, cands, K, a.jobs)
            res["arm"] = arm
            json.dump({"arm": arm, "seed": 0, "K": K, "cands": cands}, open(os.path.join(a.outdir, f"{arm}.cands.json"), "w"))
            json.dump(res, open(os.path.join(a.outdir, f"{arm}.score.json"), "w"), indent=1)
            print(summary_line(arm, res), f"wall {time.time() - t0:.0f}s", flush=True)
        return
    cj = json.load(open(a.cands))
    res = score(data, cj["cands"], min(a.K, cj.get("K", a.K)), a.jobs, a.splits.split(","))
    res.update({"arm": cj.get("arm"), "seed": cj.get("seed"), "meta": cj.get("meta", {})})
    json.dump(res, open(a.out, "w"), indent=1)
    print(summary_line(f"{cj.get('arm')} s{cj.get('seed')}", res), flush=True)


if __name__ == "__main__":
    main()
