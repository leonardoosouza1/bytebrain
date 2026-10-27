"""Verdict for mix5 (ledger #3 confirmatory run), written 10-26 BEFORE the mix5 results exist.

mix4b (seeds 0-3) gave NOT ESTABLISHED on its registered metric (term_parses_novel) and, post hoc, `runs` up for every
poe arm (+4.3..+7.2 pp, uncorrected ci95 > 0). mix5 tests that post-hoc reading on fresh seeds 4-15 with `runs` as the
registered primary metric: share of sampled functions that pass names_ok and, called once in bwrap with fuzz
arguments, raise no NameError / SyntaxError / UnboundLocalError (eval_samples.py).

Arms judged: poe_stress, poe_conf (poe_always left out: it copies 2.4x more 48-byte windows than real code does,
copy48 .314 vs held-out .132; lin24t left out: it lowered parses). Paired (seed, row) differences vs the model rule
(every rule draws the same sampling noise per seed), bootstrap over the 1536 pairs, Bonferroni over the 2 arms
(two-sided 97.5% CI).
  CONFIRMED  some arm's d_runs CI lower bound > 0
  REFUTED    both arms' d_runs CI upper bound < +4 pp (a gain of +4 pp or more ruled out for both)
  else NOT ESTABLISHED
Secondary (registered, outside the verdict): d_runs_multi = the same paired difference counting only functions with
>= 2 statements in the body (ast), 95% CI; d_one_stmt = change in the share of one-statement functions;
d_names_ok, d_term_parses_novel (95% CI), d_copy48 (means), novel.
Reading registered with the rule: CONFIRMED with the d_runs_multi CI covering 0 means "the mirror makes the model
write more short functions whose names exist", not "better longer code".

Amendment 10-26 ~21:00 (Opus pre-run review NO-GO, checked by me on mix4b BEFORE any mix5 number exists): with 4
seeds the SE of d_runs is 1.94 / 1.80 pp, so CONFIRMED needed a point estimate > +4.3 / +4.0 pp while the
predictions were +4 / +3, and REFUTED (upper < +2 pp) was out of reach -> seeds 4-15 (SE ~1.1 pp, CONFIRMED needs
> ~2.5 pp) and REFUTED at < +4 pp. mix4b's effect sat on seed 0 (poe_stress per seed +16.4 / +1.6 / +4.7 / +2.3);
without seed 0 it is +2.9 / +2.6 pp. On mix4b, functions with >= 2 statements carry +0.4 pp (SE 1.2) / +0.2 (SE 1.1):
runs among one-statement functions 47 -> 77 (poe_stress), among the rest 18 -> 20.
Prediction (gate 7, from mix4b seeds 1-3): poe_stress d_runs +3 pp (range 0..+6), poe_conf +2.5 (-0.5..+5.5);
expected verdict CONFIRMED in roughly 2 of 3 replays (reviewer simulation 76% for +2.9 / +2.6), else NOT
ESTABLISHED; REFUTED unlikely (needs both points below ~+1.5). d_runs_multi within +-2 pp with its CI covering 0;
d_one_stmt poe_stress +2..+8 pp (mix4b +4.9); d_term_parses_novel within +-5 pp; copy48 poe_stress .10-.16.

usage: mix5_verdict.py MAIN.json OUT.json
"""
import ast
import json
import sys
import warnings

import numpy as np

ARMS = ("poe_stress", "poe_conf")
SEEDS_EXPECTED = [str(s) for s in range(4, 16)]


def tpn(r):
    return float(r["terminated"] and r["parses"] and r["novel"])


def n_stmt(src):
    """statements in the body of the sampled function (0 when it does not parse)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f = [n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        except (SyntaxError, ValueError):
            return 0
    return len(f[0].body) if f else 0


METRICS = {"runs": lambda r: float(r["runs"]), "names_ok": lambda r: float(r["names_ok"]), "term_parses_novel": tpn,
           "runs_multi": lambda r: float(r["runs"] and n_stmt(r["src"]) >= 2),
           "one_stmt": lambda r: float(n_stmt(r["src"]) == 1)}


def main():
    d = json.load(open(sys.argv[1]))
    base = d["model"]["rows"]
    seeds = sorted(base, key=int)
    assert seeds == SEEDS_EXPECTED, f"confirmatory seeds must be 4-15, got {seeds}"
    rng = np.random.default_rng(0)
    out = {"seeds": seeds, "arms": {}, "verdict": None}
    for arm in ARMS:
        rows = d[arm]["rows"]
        assert sorted(rows, key=int) == seeds and all(len(rows[s]) == len(base[s]) for s in seeds)
        idx = None
        res = {}
        for m, f in METRICS.items():
            x = np.array([f(a) - f(b) for s in seeds for a, b in zip(rows[s], base[s])])
            if idx is None:
                idx = rng.integers(0, len(x), (10_000, len(x)))      # same resamples for every metric of an arm
            bs = x[idx].mean(1)
            lo, hi = (1.25, 98.75) if m == "runs" else (2.5, 97.5)
            res[f"d_{m}"] = round(float(x.mean()), 4)
            res[f"ci_{m}"] = [round(float(np.percentile(bs, lo)), 4), round(float(np.percentile(bs, hi)), 4)]
        res["ci_runs_level"] = "97.5% two-sided (Bonferroni over 2 arms)"
        res["n_pairs"] = len(idx[0])
        res["rate_runs"] = [round(d["model"]["mean"]["runs"], 4), round(d[arm]["mean"]["runs"], 4)]
        res["d_copy48"] = round(d[arm]["mean"]["copy48"] - d["model"]["mean"]["copy48"], 4)
        res["copy48"] = d[arm]["mean"]["copy48"]
        res["novel"] = d[arm]["mean"]["novel"]
        out["arms"][arm] = res
    lows = [out["arms"][a]["ci_runs"][0] for a in ARMS]
    highs = [out["arms"][a]["ci_runs"][1] for a in ARMS]
    out["verdict"] = ("CONFIRMED" if max(lows) > 0 else "REFUTED" if max(highs) < .04 else "NOT ESTABLISHED")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
