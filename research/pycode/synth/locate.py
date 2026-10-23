"""Localized error: WHERE the body went wrong and WHAT that line did instead, by executing it line by line.

For a proposed body of `def p1_then_p2_then_..._then_final(xs):`, line i is checked by running the body up to line i
(bwrap) on the 30 inputs and comparing `xs` with the state the NAME demands after primitive i. The first diverging line
is the error. Its effect is matched against every primitive ("did square, name asks keep_odd"), so the diagnosis names
the mistake without writing the correct line.

Training signal (fine-tune, same optimizer for every arm):
  ce   : exact bodies, plain next-byte loss (what self-training did so far)
  loc  : ce + for wrong bodies: next-byte loss on the correct prefix lines, unlikelihood on the first wrong line
         (push its bytes down), nothing after it.

Usage:
  locate.py diagnose CKPT --out F.json [--n 400]                       (report: where/what errors are)
  locate.py train CKPT --arm {ce,loc} --out-ckpt F.pt [--steps 3000]
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
from synth import FINAL, INPUTS, PRIM, name_of, propose, run_many  # noqa: E402
from eval_samples import ByteLM, CachedGen  # noqa: E402

FINAL_OF = {v: k for k, v in FINAL.items()}


def as_f(prims):
    """the function for a primitive chain, named f (a function named `abs` would shadow the builtin)."""
    g = {}
    exec("def f(xs):\n" + synth.source(prims, "").split("\n", 1)[1], g)
    return g["f"]


def ref_states(prims):
    """[state after primitive i for every input] for i = 0..n (0 = the input itself)."""
    out = [[repr(list(xs)) for xs in INPUTS]]
    for i in range(len(prims)):
        f = as_f(prims[:i + 1])
        out.append([repr(f(list(xs))) for xs in INPUTS])
    return out


PRIM_EFFECT = {}     # effect signature of each primitive on the 30 inputs (to name what a wrong line did)
for _p in PRIM:
    PRIM_EFFECT[tuple(ref_states([_p])[1])] = _p


def body_lines(src):
    return [l for l in src.split("\n")[1:] if l.strip()]


def diagnose(tasks, srcs):
    """-> per sample dict: ok_lines (count of leading correct lines), wrong (index or None), did (primitive or '?'),
    asked (primitive or final), exact (bool)."""
    items, index = [], []
    for t, ((p, f), ss) in enumerate(zip(tasks, srcs)):
        for j, s in enumerate(ss):
            lines = body_lines(s)
            for i in range(len(lines)):
                # prefix up to line i, then return xs (a `return` line is replaced to expose the final value)
                pre = lines[:i + 1]
                if pre[-1].strip().startswith("return"):
                    pre = pre[:-1] + ["    __r = " + pre[-1].strip()[len("return "):], "    return __r"]
                else:
                    pre = pre + ["    return xs"]
                items.append(("def f(xs):\n" + "\n".join(pre) + "\n", "f"))
                index.append((t, j, i))
    outs = []
    for i in range(0, len(items), 512):
        outs += run_many(items[i:i + 512])
    per = {}
    for (t, j, i), o in zip(index, outs):
        per[(t, j, i)] = o
    res = []
    for t, ((p, f), ss) in enumerate(zip(tasks, srcs)):
        ref = ref_states(p)
        fin = synth.reference(p, f)
        row = []
        for j, s in enumerate(ss):
            lines = body_lines(s)
            ok, wrong, did, asked = 0, None, "", ""
            for i, l in enumerate(lines):
                o = per.get((t, j, i))
                is_ret = l.strip().startswith("return")
                want = fin if is_ret else (ref[i + 1] if i + 1 < len(ref) else None)
                early = is_ret and i < len(p)          # return before every primitive was written: wrong line
                if o is not None and want is not None and o == want and not early:
                    ok += 1
                    if is_ret:
                        break
                    continue
                wrong = i
                asked = (f or "list") if is_ret else (p[i] if i < len(p) else "return")
                if is_ret:
                    did = "early_return" if i < len(p) else FINAL_OF.get(l.strip(), "?")
                elif i >= len(p):
                    did = "extra_line"
                elif o is None:
                    did = "crash"
                else:
                    # what did this line do to the correct state before it? compare with every primitive
                    did = "?"
                    prev = [eval(x) for x in ref[i]]
                    for q in PRIM:
                        fq = as_f([q])
                        cand = [repr(fq(list(x))) for x in prev]
                        if cand == o:
                            did = q
                            break
                break
            row.append({"ok_lines": ok, "wrong": wrong, "did": did, "asked": asked,
                        "exact": bool(wrong is None and ok == len(lines) and lines and lines[-1].strip().startswith("return"))})
        res.append(row)
    return res


def ul_batch(model, seqs, dev, ctx, ul=0.1):
    """seqs [(bytes, weights)] with weight +1 (learn), -1 (unlikelihood), 0 (ignore) per target byte."""
    L = min(ctx, max(len(b) for b, _ in seqs))
    x = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
    w = torch.zeros(len(seqs), L - 1, device=dev)
    for i, (b, wt) in enumerate(seqs):
        b, wt = b[:L], wt[:L]
        x[i, :len(b)] = torch.tensor(list(b), device=dev)
        w[i, :len(b) - 1] = torch.tensor(wt[1:len(b)], device=dev, dtype=torch.float)
    logits = model(x[:, :-1]).float()
    logp = F.log_softmax(logits, -1).gather(-1, x[:, 1:, None]).squeeze(-1)
    pos = -(logp * (w > 0)).sum()
    neg = -(torch.log1p(-logp.exp().clamp(max=0.999)) * (w < 0)).sum()
    return (pos + ul * neg) / w.abs().sum().clamp(min=1)


def weights_for(src, head_len, d, arm):
    """per-byte weights of `src` (prompt included, weight 0 there)."""
    b = src.encode()
    wt = [0] * len(b)
    if d["exact"]:
        return b, [0] * head_len + [1] * (len(b) - head_len)
    if arm != "loc" or d["wrong"] is None:
        return None
    lines = src.split("\n")
    pos = len(lines[0]) + 1
    for i, l in enumerate(lines[1:]):
        n = len(l) + 1
        if not l.strip():
            pos += n
            continue
        k = len([x for x in lines[1:i + 1] if x.strip()])     # index among non-blank body lines
        val = 1 if k < d["wrong"] else (-1 if k == d["wrong"] else 0)
        if val == -1:
            st = pos + len(l) - len(l.lstrip())                # skip indentation: the mistake is the content
            for q in range(st, pos + len(l)):
                wt[q] = -1
        elif val == 1:
            for q in range(pos, pos + n):
                wt[q] = 1
        pos += n
    return b, wt


def repair(g, split, a):
    """Fast bank (per task, forgotten afterwards): after a wrong body, keep its correct prefix, remember the wrong line,
    and resample from that line, rejecting continuations that repeat a remembered wrong line. Same sample budget as
    fresh resampling (the control). Budget = a.k samples per task in both arms."""
    tasks = split["test_b"][:]
    random.Random(1).shuffle(tasks)
    tasks = tasks[:a.n]
    heads = [f"def {name_of(p, f)}(xs):\n" for p, f in tasks]
    first = propose(g, tasks, 1, 200, 0.8, law=a.law_mask)
    res = {"fresh": [], "repair": []}
    # control: fresh samples until solved or budget spent (k-1 more after the first)
    more = propose(g, tasks, a.k - 1, 200, 0.8, law=a.law_mask)
    sc0 = synth.grade(tasks, first)
    scm = synth.grade(tasks, more)
    res["fresh"] = [s0[0] == 1 or any(x == 1 for x in sm) for s0, sm in zip(sc0, scm)]
    # repair arm
    cur = [ss[0] for ss in first]
    bad = [set() for _ in tasks]
    solved = [s0[0] == 1 for s0 in sc0]
    used = [1] * len(tasks)
    while True:
        todo = [t for t in range(len(tasks)) if not solved[t] and used[t] < a.k]
        if not todo:
            break
        dg = diagnose([tasks[t] for t in todo], [[cur[t]] for t in todo])
        prompts = []
        for t, row in zip(todo, dg):
            d = row[0]
            lines = body_lines(cur[t])
            w = d["wrong"] if d["wrong"] is not None else len(lines)
            if w < len(lines):
                bad[t].add(lines[w].strip())
            prompts.append((t, heads[t] + "".join(l + "\n" for l in lines[:w])))
        # a.cands candidates per repair (1 = same generation budget as fresh); first one that avoids every remembered wrong line (else the first)
        # batched only with prompts of the same length (left padding shifts absolute position)
        rows = [None] * len(prompts)
        by_len = {}
        for j, (_, pr) in enumerate(prompts):
            by_len.setdefault(len(pr), []).append(j)
        for js in by_len.values():
            ids = torch.tensor([list(b"\n\n" + prompts[j][1].encode()) for j in js for _ in range(a.cands)],
                               device=g.m.device)
            # law mask counts lines from the prompt end: subtract the kept lines already in the prompt
            need = torch.tensor([max(0, len(tasks[prompts[j][0]][0]) - (prompts[j][1].count("\n") - 1))
                                 for j in js for _ in range(a.cands)],
                                device=g.m.device) if a.law_mask else None
            out = synth.generate_rows(g, ids, 200, 0.8, need).cpu().numpy()
            for q, j in enumerate(js):
                rows[j] = out[a.cands * q:a.cands * (q + 1)]
        for j, (t, pr) in enumerate(prompts):
            cands = [synth.cut_function(pr + bytes(r).decode("utf-8", "replace"))[0] for r in rows[j]]
            ok = [c for c in cands if not any(l.strip() in bad[t] for l in body_lines(c))]
            cur[t] = (ok or cands)[0]
            used[t] += 1
        sc = synth.grade([tasks[t] for t, _ in prompts], [[cur[t]] for t, _ in prompts])
        for (t, _), s in zip(prompts, sc):
            solved[t] = solved[t] or s[0] == 1
    res["repair"] = solved
    out = {"n": len(tasks), "budget": a.k, "cands": a.cands, "first_try": round(sum(s[0] == 1 for s in sc0) / len(tasks), 4),
           "fresh_solved": round(sum(res["fresh"]) / len(tasks), 4),
           "repair_solved": round(sum(res["repair"]) / len(tasks), 4)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "train", "repair"])
    ap.add_argument("ckpt")
    ap.add_argument("--out")
    ap.add_argument("--out-ckpt")
    ap.add_argument("--arm", default="loc", choices=["ce", "loc", "final"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6, help="train: propose -> diagnose -> update, repeated")
    ap.add_argument("--steps", type=int, default=250, help="train: optimizer steps per round")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--law-mask", action="store_true")
    ap.add_argument("--ul", type=float, default=0.1, help="weight of the unlikelihood term on the wrong line")
    ap.add_argument("--cands", type=int, default=1, help="repair: candidates generated per repair attempt")
    ap.add_argument("--mem-frac", type=float, default=0.3)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(0)
    split = json.load(open(f"{synth.D}/splits.json"))
    lm = ByteLM(a.ckpt)
    g = CachedGen(lm)
    t0 = time.perf_counter()
    if a.cmd == "repair":
        return repair(g, split, a)
    if a.cmd == "diagnose":
        tasks = split["test_b"][:]
        random.Random(1).shuffle(tasks)
        tasks = tasks[:a.n]
        srcs = propose(g, tasks, a.k, 200, 0.8, law=a.law_mask)
        dg = diagnose(tasks, srcs)
        flat = [d for row in dg for d in row]
        from collections import Counter
        where = Counter(d["wrong"] for d in flat)
        what = Counter((d["asked"], d["did"]) for d in flat if d["wrong"] is not None)
        res = {"n": len(flat), "exact": round(sum(d["exact"] for d in flat) / len(flat), 4),
               "first_wrong_line": {str(k): v for k, v in sorted(where.items(), key=lambda x: str(x[0]))},
               "did_kind": dict(Counter(d["did"] for d in flat if d["wrong"] is not None).most_common(12)),
               "top_confusions": [[f"asked {x} did {y}", c] for (x, y), c in what.most_common(15)],
               "examples": [{"name": name_of(*t), "src": s, **d} for t, ss, row in list(zip(tasks, srcs, dg))[:8]
                            for s, d in list(zip(ss, row))[:1]],
               "wall_s": round(time.perf_counter() - t0, 1)}
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=0))
        return
    # train: rounds of propose (self pool) -> diagnose -> update; replay of train.txt functions always included
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = lm.model
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    replay = [f.strip() + "\n" for f in open(f"{synth.D}/train.txt").read().split("\n\n") if f.strip()]
    pool = split["self"][:]
    rng = random.Random(0)
    log = []
    for r in range(a.rounds):
        model.eval()
        tasks = rng.sample(pool, a.n)
        srcs = propose(g, tasks, a.k, 200, 0.8, law=a.law_mask)
        dg = diagnose(tasks, srcs)
        if a.arm == "final":       # control: truth = final output only (as self3), not the per-line states
            for row, sc in zip(dg, synth.grade(tasks, srcs)):
                for d, x in zip(row, sc):
                    d["exact"] = x == 1.0
        data, negs = [], []
        for (p, f), ss, row in zip(tasks, srcs, dg):
            head = len(f"def {name_of(p, f)}(xs):\n")
            for s, d in zip(ss, row):
                x = weights_for(s.rstrip("\n") + "\n\n", head, d, a.arm)
                if x:
                    (negs if -1 in x[1] else data).append(x)
        st = {"round": r, "exact": round(sum(d["exact"] for row in dg for d in row) / (a.n * a.k), 4),
              "examples": len(data), "neg": len(negs)}
        model.train()
        for s in range(a.steps):
            # half the batch from own samples: exact bodies and localized-error bodies in equal shares when both exist
            own = [data, negs] if data and negs else [data or negs] if (data or negs) else []
            bt = [rng.choice(rng.choice(own)) for _ in range(a.batch // 2)] if own else []
            bt += [(b.encode() + b"\n", [1] * (len(b) + 1)) for b in (rng.choice(replay) for _ in range(a.batch - len(bt)))]
            bt = [(b"\n\n" + b, [0, 0] + w) for b, w in bt]
            loss = ul_batch(model, bt, g.m.device, lm.ctx, a.ul)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        st["loss"] = round(float(loss), 4)
        log.append(st)
        print(json.dumps(st), flush=True)
    ck["model"] = model.state_dict()
    ck["locate_log"] = log
    os.makedirs(os.path.dirname(a.out_ckpt), exist_ok=True)
    torch.save(ck, a.out_ckpt)
    print("saved", a.out_ckpt, round(time.perf_counter() - t0, 1), "s")


if __name__ == "__main__":
    main()
