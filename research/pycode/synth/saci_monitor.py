"""saci-code step 2 (ledger #46): can a monitor reading the understander's token stream tell a failing spec early?

Reads recorded files only (no model): per-token rows [id, logprob, entropy, margin] saved by
`student_understander.py --lora ... --token-scores`, and the belt traces of the same adapter on the same requests.
Label = belt failure. Arms (registered in the ledger before the data existed):
  S0  validator flags at the end (expr_parse_error / oov_op): a final-only check
  S1  peak entropy, total NLL (no fitting)
  S2  every single (signal, tau) EMA, max over tokens; the best one is picked on all data (optimistic for the baseline)
  S3  bank: the 3 signals x tau {1, 4, 16, 64}, max over tokens, logistic, 5-fold CV by request
  P   number of generated tokens
  Q   request text only (length, brackets, parentheses, "main", world), logistic, 5-fold CV
Also AUROC on the silent failures (no validator flag) vs passes, and when S3 fires relative to the first token that
differs from json.dumps(gold) (the training target string).

Usage: saci_monitor.py --raws RAWS_JSON --traces TRACES_JSONL --requests REQUESTS_JSON --out OUT_JSON
"""

import argparse
import json

import numpy as np

TAUS = (1, 4, 16, 64)
SIG = ("nll", "ent", "negmargin")


def auroc(score, y):
    """y=1 positives (failures); ties count half."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return round(float(gt + 0.5 * eq), 4)


def bank(rows):
    """rows [[id, logprob, entropy, margin]] -> [T, 3*len(TAUS)] EMA bank (signal-major)."""
    x = np.array([[-r[1], r[2], -r[3]] for r in rows], float)
    out = np.zeros((len(x), 3 * len(TAUS)))
    for j, tau in enumerate(TAUS):
        e = np.zeros(3)
        for t in range(len(x)):
            e = e + (x[t] - e) / tau
            out[t, [k * len(TAUS) + j for k in range(3)]] = e
    return out


def fit_logistic(X, y, l2=1e-2, steps=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return lambda A: ((A - mu) / sd) @ w + b


def cv_scores(X, y, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
    out, models = np.zeros(len(y)), [None] * len(y)
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        m = fit_logistic(X[tr], y[tr].astype(float))
        out[te] = m(X[te])
        for i in te:
            models[i] = m
    return out, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raws", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raws = json.load(open(a.raws))
    tr = {t["id"]: t for t in map(json.loads, open(a.traces))}
    rq = {q["id"]: q for q in json.load(open(a.requests))}
    ids = [i for i in tr if i in raws and raws[i].get("tok")]
    assert all(raws[i]["raw"] == tr[i]["raw_spec"] for i in ids), "token-score raws differ from the traced raws"
    y = np.array([not tr[i]["end_pass"] for i in ids])
    flag = np.array([any(f["from"] == "validator" for f in tr[i]["flags"]) for i in ids])
    B = {i: bank(raws[i]["tok"]) for i in ids}
    res = {"n": len(ids), "fail": int(y.sum()), "silent_fail": int((y & ~flag).sum())}
    silent = ~flag                                       # silent failures vs all passes (passes carry no flag)
    noflag = np.array([not tr[i]["flags"] for i in ids])  # stricter: no flag of any kind (exec flags too)
    res["noflag_fail"] = int((y & noflag).sum())

    def report(name, s):
        res[name] = {"auroc": auroc(s, y), "auroc_silent": auroc(s[silent], y[silent]),
                     "auroc_noflag": auroc(s[noflag], y[noflag])}

    report("S0_flags", flag.astype(float))
    nll = np.array([B[i][:, 0 * len(TAUS)].sum() for i in ids])          # tau=1 EMA = the raw signal
    report("S1_peak_entropy", np.array([B[i][:, 1 * len(TAUS)].max() for i in ids]))
    report("S1_total_nll", nll)
    singles = {}
    for k, sname in enumerate(SIG):
        for j, tau in enumerate(TAUS):
            s = np.array([B[i][:, k * len(TAUS) + j].max() for i in ids])
            singles[f"{sname}_tau{tau}"] = (auroc(s, y), auroc(s[silent], y[silent]))
    best = max(singles, key=lambda n: singles[n][0])
    res["S2_singles"] = singles
    res["S2_best"] = {"name": best, "auroc": singles[best][0], "auroc_silent": singles[best][1]}
    X3 = np.array([B[i].max(0) for i in ids])
    s3, models = cv_scores(X3, y)
    report("S3_bank_cv", s3)
    report("P_length", np.array([len(raws[i]["tok"]) for i in ids], float))
    # request text only: the trace's "world" is the gold world, so it is not a request feature (review 10-26)
    Xq = np.array([[len(rq[i]["text"]), rq[i]["text"].count("("), rq[i]["text"].count("["),
                    float("main" in rq[i]["text"])] for i in ids])
    sq, _ = cv_scores(Xq, y)
    report("Q_request_cv", sq)

    # when does S3 fire? threshold = 90th percentile of the CV scores of passes (10% false alarms)
    thr = np.quantile(s3[~y], 0.9)
    from transformers import AutoTokenizer                # tokenizer only, offline cache
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B-Instruct", local_files_only=True,
                                        cache_dir="/media/llm-data/models/hf/hub")
    lead = []
    for n, i in enumerate(ids):
        if not y[n]:
            continue
        gold_ids = tok(json.dumps(rq[i]["gold"]), add_special_tokens=False).input_ids
        gen = [r[0] for r in raws[i]["tok"]]
        first_wrong = next((t for t, (g, h) in enumerate(zip(gen, gold_ids)) if g != h), min(len(gen), len(gold_ids)))
        run = np.maximum.accumulate(B[i], 0)
        fire = next((t for t in range(len(run)) if models[n](run[t:t + 1])[0] > thr), None)
        lead.append({"id": i, "silent": bool(not flag[n]), "first_wrong": first_wrong, "fire": fire, "len": len(gen)})
    det = [d for d in lead if d["fire"] is not None]
    res["S3_timing"] = {"threshold_fp_rate": 0.1, "fails": len(lead), "detected": len(det),
                        "detected_silent": sum(d["silent"] for d in det),
                        "fire_at_or_before_first_wrong_plus2": sum(d["fire"] <= d["first_wrong"] + 2 for d in det),
                        "median_tokens_after_first_wrong": float(np.median([d["fire"] - d["first_wrong"] for d in det]))
                        if det else None}
    res["lead_rows"] = lead
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("lead_rows", "S2_singles")}, indent=1))


if __name__ == "__main__":
    main()
