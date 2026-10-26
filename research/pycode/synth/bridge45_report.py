"""Verdicts of the #45 tau-bank rerun from the per-arm files (rules and predictions in bridge_cpu.py docstring).

Mean of 3 seeds per arm; a margin counts only if > 3 pp and > 2x the largest seed SD of the two arms compared.
Usage: bridge45_report.py DIR   (reads DIR/arm_<arm>_d<K>.json, writes DIR/report45.json)
"""

import json
import os
import statistics
import sys

ARMS = ("gru", "cfc_bank", "cfc_tau1", "cfc_learn", "cfc_stress")
METRICS = ("test_a", "test_num", "len3", "len4", "len5", "len6")


def load(d, arm, k):
    p = os.path.join(d, f"arm_{arm}_d{k}.json")
    if not os.path.exists(p):
        return None
    runs = json.load(open(p))["runs"]
    out = {"n_seeds": len(runs), "lr": runs[0]["lr"]}
    for m in METRICS:
        v = [r["eval"][m]["pass1"] for r in runs]
        out[m] = (round(statistics.mean(v), 4), round(statistics.pstdev(v), 4))
    if "learned_rate_per_group" in runs[0]:
        out["learned_rate_per_group"] = runs[0]["learned_rate_per_group"]
    for key in ("stress_a", "stress_b"):
        if key in runs[0]:
            out[key] = runs[0][key]
    return out


def margin(x, y, m):
    """x - y on metric m (means); counts if > 0.03 and > 2x the larger seed SD."""
    d = x[m][0] - y[m][0]
    need = max(0.03, 2 * max(x[m][1], y[m][1]))
    return round(d, 4), d > need, round(need, 4)


def main():
    d = sys.argv[1]
    t = {k: {a: load(d, a, k) for a in ARMS} for k in (0, 48)}
    missing = [f"{a}_d{k}" for k in t for a in ARMS if t[k][a] is None or t[k][a]["n_seeds"] < 3]
    v = {"missing_or_short": missing}
    z, f = t[0], t[48]
    if z["cfc_bank"]:
        v["P1_cfc_bank_K0_G0"] = z["cfc_bank"]["test_a"][0] >= 0.9 and z["cfc_bank"]["test_num"][0] >= 0.9
    v["P2_K0_len4_below_15"] = {a: z[a]["len4"][0] < 0.15 for a in ARMS if z[a]}
    if f["cfc_bank"] and f["cfc_tau1"]:
        dm, ok, need = margin(f["cfc_bank"], f["cfc_tau1"], "test_a")
        v["P3_K48_bank_minus_tau1_test_a"] = {"diff": dm, "pass": dm >= 0.20 and ok, "need_noise": need}
        v["KILL_K48_bank_le_tau1_plus_5"] = dm <= 0.05
    if f["cfc_stress"] and f["cfc_bank"]:
        p4 = {}
        for m in ("test_a", "len3"):
            if f["cfc_bank"][m][0] < 0.95:
                dm, ok, need = margin(f["cfc_stress"], f["cfc_bank"], m)
                p4[m] = {"diff": dm, "pass": dm >= 0.05 and ok, "need_noise": need}
            else:
                p4[m] = "not counted (cfc_bank >= 95%)"
        v["P4_K48_stress_minus_bank"] = p4
    taus = [a for a in ("cfc_bank", "cfc_tau1", "cfc_learn", "cfc_stress") if z[a]]
    if z["gru"] and taus:
        best = max(taus, key=lambda a: z[a]["test_a"][0])
        v["P5_K0_gru_within_5_of_best_tau"] = {"best_tau_arm": best,
                                               "diff": round(z["gru"]["test_a"][0] - z[best]["test_a"][0], 4),
                                               "pass": abs(z["gru"]["test_a"][0] - z[best]["test_a"][0]) <= 0.05}
    res = {"table": {f"K{k}": t[k] for k in t}, "verdicts": v}
    json.dump(res, open(os.path.join(d, "report45.json"), "w"), indent=1)
    for k in t:
        print(f"K={k}")
        for a in ARMS:
            r = t[k][a]
            print(f"  {a:11s}", "missing" if r is None else
                  f"lr {r['lr']:.0e} " + " ".join(f"{m} {r[m][0]:.3f}±{r[m][1]:.3f}" for m in METRICS))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
