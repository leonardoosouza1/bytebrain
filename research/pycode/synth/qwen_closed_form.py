"""Closed-form readout of frozen Qwen2.5-Coder-0.5B: is the ORDER of the name's units linearly readable from its
state at each block boundary? (proposer 4, registered 2026-10-22 before any run; CPU, no gradients)

Context `def NAME(xs, k):\n` (constant signature: no KARG cue) + the canonical lines of the units written so far.
Boundary j = last token of the text ending at the j-th "\n" of a block (asserted by tokenizer offsets); its state
predicts item j (unit j+1, or the final). Label factored: op (19 units + 6 finals) and N (none, 2..6), so a held
(op, N) binding can be composed. Readout per (layer, PCA rank r): RMS-normalise, z-score (fit set), PCA (fit set),
ridge W* = (P'P + lam I)^-1 P'Y, lam by group 5-fold on fit names. The table writer (unit -> lines) is ORACLE:
this measures name -> ordered units, not code writing.

Sets: fit = all 240 train singles + 760 train pairs; val = 160 test_a (selects layer, rank); eval = 250 other
test_a, 250 test_num, 250 test_b (single shot). Baselines without Qwen: B1 (slot, previous item) counts; B2 bag of
name tokens + bag of items written + slot, ridge. Controls: shuffled label sequences at the chosen config; PCA fit
on the eval states (leak delta); free run (predicted units appended through the table, Qwen re-run) graded strict.

Rule (pre-registered): PASS = a config with layer <= 16 and r <= 64 chosen on val gives test_a function-exact
>= 85% and >= B2 + 30 pp; test_num N accuracy on held bindings >= 85%; test_b function-exact >= 50% and
>= B2_b + 25 pp; free-run strict pass >= 0.8 x teacher-forced function-exact on every eval split.
KILL = the best config on the whole grid (r <= 128) has test_a <= B2 + 10 pp or test_b <= 25%.
Deviations from the proposal (declared): PCA is fit on the fit-set boundary states (not on all positions of 200
functions); no zero-shot restricted-vocab anchor (qwen_teacher.py measures the few-shot writer); no nonsense-word arm.

Usage: qwen_closed_form.py --smoke --out DIR      (known numbers that must reproduce before the run is trusted)
       qwen_closed_form.py --out DIR [--threads 4]
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import synth2  # noqa: E402
from synth2 import FINAL, HELD, KARG, LOOP, NUMS, PARAM, PLAIN, UNITS, grade, name_of  # noqa: E402

MODEL = ("/media/llm-data/models/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/"
         "ea3f2471cf1b1f0db85067f1ef93848e38e88c25")
FINS = ["", "sum", "max", "len", "first", "count_gt"]
OPS = list(PARAM) + list(KARG) + list(LOOP) + list(PLAIN) + [f"fin:{f}" for f in FINS]
LAYERS = list(range(0, 25, 3))
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 896]
LAMS = [10.0 ** e for e in range(-2, 5)]
NO, NN = len(OPS), 1 + len(NUMS)


# ---------------------------------------------------------------- labels, text, table writer

def items(task):
    """[(item, is_final)]: what boundary j must predict."""
    u, f = task
    return [(x, False) for x in u] + [(f, True)]


def label(item, fin):
    if fin:
        if item.startswith("count_gt_"):
            return OPS.index("fin:count_gt"), NUMS.index(int(item.rsplit("_", 1)[1])) + 1
        return OPS.index(f"fin:{item}"), 0
    if item in KARG or item in LOOP or item in PLAIN:
        return OPS.index(item), 0
    op, n = item.rsplit("_", 1)
    return OPS.index(op), NUMS.index(int(n)) + 1


def decode(op_row, n_row):
    """argmax op; N = argmax over 2..6 when the op needs one -> (item, is_final)."""
    op = OPS[int(np.argmax(op_row))]
    n = NUMS[int(np.argmax(n_row[1:]))]
    if op.startswith("fin:"):
        return (f"count_gt_{n}" if op == "fin:count_gt" else op[4:]), True
    return (f"{op}_{n}" if op in PARAM else op), False


def ctx_head(task):
    return f"def {name_of(*task)}(xs, k):\n"


def block(item, fin):
    return "".join(f"    {l}\n" for l in ([FINAL[item]] if fin else UNITS[item]))


def text_and_bounds(task):
    t, bounds = ctx_head(task), []
    for x, fin in items(task):
        bounds.append(len(t))
        t += block(x, fin)
    return t, bounds


def assemble(task, units, final):
    """table writer: the signature has k only if a predicted unit takes k (the task's arity is not copied)."""
    if final is None:
        return f"def {name_of(*task)}(xs):\n    return None\n"
    sig = "xs, k" if any(u in KARG for u in units) else "xs"
    return f"def {name_of(*task)}({sig}):\n" + "".join(block(u, False) for u in units) + block(final, True)


# ---------------------------------------------------------------- Qwen capture

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    return model, tok


def boundary_positions(tok, text, bounds):
    e = tok(text, return_offsets_mapping=True)
    ends = [o[1] for o in e["offset_mapping"]]
    pos = []
    for b in bounds:
        assert text[b - 1] == "\n" and ends.count(b) == 1, (text, b)
        pos.append(ends.index(b))
    return e["input_ids"], pos


@torch.no_grad()
def capture(model, tok, texts_bounds, layers=LAYERS, batch=16, post_norm_last=False):
    """-> (states [n_boundaries, len(layers), 896] float32, func index per boundary, n_tokens). Right padding: a
    causal state at a real position never sees the pad after it (checked in the smoke)."""
    enc = [boundary_positions(tok, t, b) for t, b in texts_bounds]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    out = [None] * len(enc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for c in range(0, len(order), batch):
        ch = order[c:c + batch]
        L = max(len(enc[i][0]) for i in ch)
        ids = torch.full((len(ch), L), pad, dtype=torch.long)
        mask = torch.zeros(len(ch), L, dtype=torch.long)
        for r, i in enumerate(ch):
            ids[r, :len(enc[i][0])] = torch.tensor(enc[i][0])
            mask[r, :len(enc[i][0])] = 1
        hs = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        picked = []
        for l in layers:
            h = hs[l]
            if l == len(hs) - 1 and post_norm_last:
                h = model.model.norm(h)
            picked.append(h)
        H = torch.stack(picked, 2)
        for r, i in enumerate(ch):
            out[i] = H[r, enc[i][1]].float().numpy()
    fid = np.concatenate([np.full(len(enc[i][1]), i) for i in range(len(enc))])
    return np.concatenate(out), fid, sum(len(e[0]) for e in enc)


# ---------------------------------------------------------------- closed-form readout

class Prep:
    """RMS-normalise, z-score and PCA, all fitted on the fit set only."""

    def __init__(self, X):
        Z = X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        Z = (Z - self.mu) / self.sd
        _, _, vt = np.linalg.svd(Z.astype(np.float64), full_matrices=False)
        self.vt = vt

    def __call__(self, X, r):
        Z = X / np.sqrt((X ** 2).mean(1, keepdims=True) + 1e-6)
        return ((Z - self.mu) / self.sd) @ self.vt[:r].T


def onehot(lab):
    Y = np.zeros((len(lab), NO + NN))
    for i, (o, n) in enumerate(lab):
        Y[i, o] = 1
        Y[i, NO + n] = 1
    return Y


def ridge(P, Y, lam):
    """bias column unpenalised."""
    A = np.hstack([P, np.ones((len(P), 1))])
    reg = lam * np.eye(A.shape[1])
    reg[-1, -1] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ Y)


def apply(P, W):
    return np.hstack([P, np.ones((len(P), 1))]) @ W


def correct(S, gold):
    """per boundary: decoded item equals the gold item."""
    return np.array([decode(s[:NO], s[NO:]) == g for s, g in zip(S, gold)])


def fexact(ok, fid):
    per = {}
    for o, f in zip(ok, fid):
        per[f] = per.get(f, True) and bool(o)
    return np.mean(list(per.values()))


def pick_lam(P, Y, gold, fid, folds=5):
    """group k-fold over functions: every boundary of a name sits in one fold."""
    groups = np.array([f % folds for f in fid])
    best = None
    for lam in LAMS:
        acc = 0.0
        for k in range(folds):
            tr, te = groups != k, groups == k
            W = ridge(P[tr], Y[tr], lam)
            acc += correct(apply(P[te], W), [g for g, m in zip(gold, te) if m]).mean() / folds
        if best is None or acc > best[1] + 1e-9:
            best = (lam, acc)
    return best[0]


def wilson(p, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


# ---------------------------------------------------------------- baselines (no Qwen)

def b1(fit, ev):
    """(slot, previous item) -> most frequent op and N, by counting; fallback slot only."""
    def keys(ts):
        out = []
        for t in ts:
            prev = "<s>"
            for j, (x, fin) in enumerate(items(t)):
                out.append(((j, prev), label(x, fin), (x, fin)))
                prev = x
        return out
    cnt_o, cnt_n, slot_o, slot_n = {}, {}, {}, {}
    for (j, p), (o, n), _ in keys(fit):
        for d, k, v, size in ((cnt_o, (j, p), o, NO), (cnt_n, (j, p), n, NN), (slot_o, j, o, NO), (slot_n, j, n, NN)):
            d.setdefault(k, np.zeros(size))[v] += 1
    res = {}
    for name, ts in ev.items():
        kk = keys(ts)
        S = []
        for (j, p), _, _ in kk:
            o = cnt_o.get((j, p), slot_o.get(j, np.zeros(NO)))
            n = cnt_n.get((j, p), slot_n.get(j, np.zeros(NN)))
            S.append(np.concatenate([o, n]))
        ok = correct(np.array(S), [g for _, _, g in kk])
        fid = np.concatenate([np.full(len(items(t)), i) for i, t in enumerate(ts)])
        res[name] = {"item_acc": round(ok.mean(), 4), "fexact": round(fexact(ok, fid), 4)}
    return res, keys


def b1_ridge(fit, ev, keys):
    """same cells as one-hot features, ridge with tiny lam: must agree with counting within 1 pp."""
    cells = sorted({k for k, _, _ in keys(fit)}, key=str)
    ix = {c: i for i, c in enumerate(cells)}

    def feats(ts):
        kk = keys(ts)
        X = np.zeros((len(kk), len(cells)))
        for r, (k, _, _) in enumerate(kk):
            if k in ix:
                X[r, ix[k]] = 1
        return X, kk
    X, kk = feats(fit)
    W = ridge(X, onehot([l for _, l, _ in kk]), 1e-6)
    res = {}
    for name, ts in ev.items():
        Xe, ke = feats(ts)
        ok = correct(apply(Xe, W), [g for _, _, g in ke])
        fid = np.concatenate([np.full(len(items(t)), i) for i, t in enumerate(ts)])
        res[name] = {"item_acc": round(ok.mean(), 4), "fexact": round(fexact(ok, fid), 4)}
    return res


def b2(fit, ev, tok):
    """bag of name tokens + bag of items already written + slot, ridge (order of the name is invisible)."""
    vocab, wvoc = {}, {}

    def rows(ts, grow):
        X, lab, gold, fid = [], [], [], []
        for i, t in enumerate(ts):
            name_ids = tok(name_of(*t))["input_ids"]
            written = []
            for j, (x, fin) in enumerate(items(t)):
                f = {("n", v) for v in name_ids} | {("w", w) for w in written} | {("s", j)}
                if grow:
                    for k in f:
                        vocab.setdefault(k, len(vocab))
                X.append([vocab[k] for k in f if k in vocab])
                lab.append(label(x, fin))
                gold.append((x, fin))
                fid.append(i)
                written.append(x)
        M = np.zeros((len(X), len(vocab)))
        for r, cols in enumerate(X):
            M[r, cols] = 1
        return M, lab, gold, np.array(fid)
    M, lab, gold, fid = rows(fit, True)
    Y = onehot(lab)
    lam = pick_lam(M, Y, gold, fid)
    W = ridge(M, Y, lam)
    res = {"lam": lam}
    for name, ts in ev.items():
        Me, _, ge, fe = rows(ts, False)
        ok = correct(apply(Me, W), ge)
        res[name] = {"item_acc": round(ok.mean(), 4), "fexact": round(fexact(ok, fe), 4)}
    return res


# ---------------------------------------------------------------- sets

def make_sets():
    split = json.load(open(f"{synth2.D}/splits.json"))
    r = random.Random(1)
    singles = [t for t in split["train"] if len(t[0]) == 1]
    pairs = [t for t in split["train"] if len(t[0]) == 2]
    fit = singles + r.sample(pairs, 1000 - len(singles))
    ta = r.sample(split["test_a"], 410)
    sets = {"fit": fit, "val": ta[:160], "test_a": ta[160:], "test_num": r.sample(split["test_num"], 250),
            "test_b": r.sample(split["test_b"], 250)}
    names = {k: {name_of(*t) for t in v} for k, v in sets.items()}
    for a in sets:
        for b in sets:
            assert a == b or not names[a] & names[b], (a, b)
    assert not any(u in HELD for t in fit for u in t[0])
    return sets


def golds(ts):
    lab, gold = [], []
    for t in ts:
        for x, fin in items(t):
            lab.append(label(x, fin))
            gold.append((x, fin))
    return lab, gold


# ---------------------------------------------------------------- smoke

def smoke(a):
    res = {}
    split = json.load(open(f"{synth2.D}/splits.json"))
    r = random.Random(3)
    tasks = [t for v in split.values() for t in r.sample(v, 40)]
    # 1. labels -> decode -> table writer -> strict judge: must be 1.0 on every task (label/table/judge oracle)
    srcs = []
    for t in tasks:
        dec = [decode(np.eye(NO)[o], np.eye(NN)[n]) for o, n in (label(x, fin) for x, fin in items(t))]
        assert dec == items(t), (t, dec)
        srcs.append([assemble(t, [x for x, _ in dec[:-1]], dec[-1][0])])
    res["gold_assembled_strict_1.0"] = sum(s[0] == 1.0 for s in grade(tasks, srcs))
    res["gold_tasks"] = len(tasks)
    bad = [[assemble(t, t[0][:-1] if len(t[0]) > 1 else t[0], t[1])] for t in tasks]
    res["dropped_unit_still_1.0"] = sum(s[0] == 1.0 for s, t in zip(grade(tasks, bad), tasks) if len(t[0]) > 1)
    model, tok = load_model()
    sub = tasks[:20]
    tb = [text_and_bounds(t) for t in sub]
    # 2. last hidden state x tied head == HF logits (decides whether hidden_states[-1] is post-norm)
    with torch.no_grad():
        for post in (False, True):
            X, fid, _ = capture(model, tok, tb, layers=[24], post_norm_last=post)
            mx, agree, k = 0.0, 0, 0
            for i, (t, b) in enumerate(tb):
                ids, pos = boundary_positions(tok, t, b)
                lg = model(input_ids=torch.tensor([ids])).logits[0, pos]
                mine = model.lm_head(torch.tensor(X[fid == i][:, 0]))
                mx = max(mx, (lg - mine).abs().max().item())
                agree += (lg.argmax(-1) == mine.argmax(-1)).sum().item()
                k += len(pos)
            res[f"head_check_post_norm_{post}"] = {"max_abs_diff": round(mx, 6), "argmax_agree": agree / k}
    # 3. batched (right padded) boundary states == unbatched states of the text truncated at the boundary
    X, fid, ntok = capture(model, tok, tb)
    worst, rel = 0.0, 0.0
    for i, (t, b) in enumerate(tb):
        full_ids, _ = boundary_positions(tok, t, b)
        for j, bb in enumerate(b):
            ids, pos = boundary_positions(tok, t[:bb], [bb])
            assert ids == full_ids[:len(ids)], "prefix tokenisation differs"
            single, _, _ = capture(model, tok, [(t[:bb], [bb])], batch=1)
            d = np.abs(single[0] - X[fid == i][j]).max()
            worst, rel = max(worst, d), max(rel, d / (np.abs(single[0]).max() + 1e-9))
    res["truncation_equality"] = {"max_abs": float(worst), "max_rel": float(rel)}
    # 4. throughput
    t0 = time.perf_counter()
    _, _, ntok = capture(model, tok, [text_and_bounds(t) for t in tasks[:64]])
    res["tok_per_s"] = round(ntok / (time.perf_counter() - t0), 1)
    # 5. B1 by counting == B1 by one-hot ridge
    sets = make_sets()
    ev = {k: v for k, v in sets.items() if k != "fit"}
    c, keys = b1(sets["fit"], ev)
    rr = b1_ridge(sets["fit"], ev, keys)
    res["b1_count"], res["b1_ridge"] = c, rr
    res["b1_agree_1pp"] = all(abs(c[k]["fexact"] - rr[k]["fexact"]) <= 0.01 for k in ev)
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{a.out}/smoke.json", "w"), indent=1)


# ---------------------------------------------------------------- main run

def free_run(model, tok, tasks, layer_ix, post, prep, r, W):
    texts = {i: ctx_head(t) for i, t in enumerate(tasks)}
    units = {i: [] for i in range(len(tasks))}
    final = {}
    for _ in range(7):
        act = [i for i in range(len(tasks)) if i not in final]
        if not act:
            break
        X, _, _ = capture(model, tok, [(texts[i], [len(texts[i])]) for i in act], layers=[LAYERS[layer_ix]],
                          post_norm_last=post)
        S = apply(prep(X[:, 0], r), W)
        for i, s in zip(act, S):
            x, fin = decode(s[:NO], s[NO:])
            if fin:
                final[i] = x
            elif len(units[i]) >= 6:
                final[i] = None
            else:
                units[i].append(x)
                texts[i] += block(x, False)
    for i in range(len(tasks)):
        final.setdefault(i, None)
    srcs = [[assemble(t, units[i], final[i])] for i, t in enumerate(tasks)]
    sc = grade(tasks, srcs)
    lab_ok = [units[i] == list(t[0]) and final[i] == t[1] for i, t in enumerate(tasks)]
    return {"n": len(tasks), "strict_pass": round(sum(s[0] == 1.0 for s in sc) / len(tasks), 4),
            "label_exact": round(sum(lab_ok) / len(tasks), 4),
            "examples_wrong": [srcs[i][0] for i in range(len(tasks)) if not lab_ok[i]][:3]}


def run(a):
    t0 = time.perf_counter()
    sets = make_sets()
    ev = {k: v for k, v in sets.items() if k != "fit"}
    smk = json.load(open(f"{a.out}/smoke.json"))
    post = smk["head_check_post_norm_True"]["argmax_agree"] > smk["head_check_post_norm_False"]["argmax_agree"]
    res = {"post_norm_applied_to_layer24": post, "sizes": {k: len(v) for k, v in sets.items()}}
    model, tok = load_model()
    c, keys = b1(sets["fit"], ev)
    res["B1"] = c
    res["B2"] = b2(sets["fit"], ev, tok)
    print(json.dumps({"B1": c, "B2": res["B2"]}), flush=True)
    cap = {}
    for k, ts in sets.items():
        X, fid, ntok = capture(model, tok, [text_and_bounds(t) for t in ts], post_norm_last=post)
        lab, gold = golds(ts)
        cap[k] = (X, fid, lab, gold)
        np.save(f"{a.out}/states_{k}.npy", X.astype(np.float16))
        print(json.dumps({"captured": k, "boundaries": len(X), "tokens": ntok,
                          "wall_s": round(time.perf_counter() - t0, 1)}), flush=True)
    Xf, ff, lf, gf = cap["fit"]
    Yf = onehot(lf)
    grid = {}
    for li, l in enumerate(LAYERS):
        prep = Prep(Xf[:, li])
        for r in RANKS:
            P = prep(Xf[:, li], r)
            lam = pick_lam(P, Yf, gf, ff)
            W = ridge(P, Yf, lam)
            row = {"lam": lam, "fit_item_acc": round(correct(apply(P, W), gf).mean(), 4)}
            for k in ev:
                X, fid, lab, gold = cap[k]
                ok = correct(apply(prep(X[:, li], r), W), gold)
                row[k] = {"item_acc": round(ok.mean(), 4), "fexact": round(fexact(ok, fid), 4)}
                if k == "test_num":
                    held = [g[0] in HELD for g in gold]
                    S = apply(prep(X[:, li], r), W)
                    n_ok = [int(np.argmax(s[NO + 1:])) + 1 == lb[1] for s, lb, h in zip(S, lab, held) if h]
                    row[k]["held_N_acc"] = round(float(np.mean(n_ok)), 4)
            grid[f"{l}/{r}"] = row
        print(json.dumps({"layer": l, "val_fexact_by_rank": {r: grid[f"{l}/{r}"]["val"]["fexact"] for r in RANKS}}),
              flush=True)
    res["grid"] = grid
    # selection on val only: PASS-eligible (layer <= 16, r <= 64) and unrestricted (KILL check uses r <= 128)
    def best(cond):
        ks = [k for k in grid if cond(*map(int, k.split("/")))]
        return max(ks, key=lambda k: (grid[k]["val"]["fexact"], -int(k.split("/")[1]), -int(k.split("/")[0])))
    sel = best(lambda l, r: l <= 16 and r <= 64)
    best_all = best(lambda l, r: r <= 128)
    res["selected"], res["best_r128_by_val"] = sel, best_all
    res["r90_by_layer"] = {}
    for l in LAYERS:
        full = grid[f"{l}/896"]["val"]["fexact"]
        res["r90_by_layer"][l] = next((r for r in RANKS if grid[f"{l}/{r}"]["val"]["fexact"] >= 0.9 * full), None)
    l_sel, r_sel = map(int, sel.split("/"))
    li = LAYERS.index(l_sel)
    prep = Prep(Xf[:, li])
    W = ridge(prep(Xf[:, li], r_sel), Yf, grid[sel]["lam"])
    # controls at the selected config
    rs = random.Random(9)
    by_n = {}
    for i, t in enumerate(sets["fit"]):
        by_n.setdefault(len(t[0]), []).append(i)
    perm = {}
    for n, idx in by_n.items():
        sh = idx[:]
        rs.shuffle(sh)
        perm.update(dict(zip(idx, sh)))
    lab_sh = []
    for i, t in enumerate(sets["fit"]):
        lab_sh += [label(x, fin) for x, fin in items(sets["fit"][perm[i]])]
    W_sh = ridge(prep(Xf[:, li], r_sel), onehot(lab_sh), grid[sel]["lam"])
    res["shuffled_labels"] = {}
    res["leaked_pca_delta"] = {}
    for k in ev:
        X, fid, lab, gold = cap[k]
        res["shuffled_labels"][k] = round(fexact(correct(apply(prep(X[:, li], r_sel), W_sh), gold), fid), 4)
        leak = Prep(X[:, li])                       # PCA fitted on the eval states themselves
        W_l = ridge(leak(Xf[:, li], r_sel), Yf, grid[sel]["lam"])
        res["leaked_pca_delta"][k] = round(fexact(correct(apply(leak(X[:, li], r_sel), W_l), gold), fid)
                                           - grid[sel][k]["fexact"], 4)
        ci = grid[sel][k]["fexact"]
        res.setdefault("selected_ci95", {})[k] = wilson(ci, len(sets[k]))
    print(json.dumps({"selected": sel, "row": grid[sel], "shuffled": res["shuffled_labels"]}), flush=True)
    res["free_run"] = {}
    for k in ("test_a", "test_num", "test_b"):
        res["free_run"][k] = free_run(model, tok, sets[k][:120], li, post, prep, r_sel, W)
        print(json.dumps({"free_run": k, **{x: y for x, y in res["free_run"][k].items() if x != "examples_wrong"}}),
              flush=True)
    g, b2r = grid[sel], res["B2"]
    gb = grid[best_all]
    res["rule"] = {
        "pass": bool(g["test_a"]["fexact"] >= 0.85 and g["test_a"]["fexact"] >= b2r["test_a"]["fexact"] + 0.30
                     and g["test_num"]["held_N_acc"] >= 0.85
                     and g["test_b"]["fexact"] >= 0.50 and g["test_b"]["fexact"] >= b2r["test_b"]["fexact"] + 0.25
                     and all(res["free_run"][k]["strict_pass"] >= 0.8 * g[k]["fexact"]
                             for k in ("test_a", "test_num", "test_b"))),
        "kill": bool(gb["test_a"]["fexact"] <= b2r["test_a"]["fexact"] + 0.10 or gb["test_b"]["fexact"] <= 0.25),
    }
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({"rule": res["rule"], "wall_s": res["wall_s"]}), flush=True)
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    smoke(a) if a.smoke else run(a)


if __name__ == "__main__":
    main()
