"""Slow pointer state: WHERE the writer is inside the name, held in the input and changed only at line events.

Diagnosis of the best verified-rounds model (loc_ce, 83% on unseen 3-primitive names): 98% of the remaining errors are
on the MIDDLE line - the model writes primitive 1, then jumps to primitive 3. It loses its place, not the meaning.

Pointer: the body is written one line at a time; before each line the prompt is rebuilt with the name segments
already written in UPPER case (bit 5 of each letter flipped). `def SORT_then_double_then_sum(xs):` = "sort is done,
you are at double". The pointer says only how many segments are done (a count, from the line events), never which
primitive to write or whether the written line was right.

Arms (same per-line example format, same verified on-policy rounds, same optimizer; only the pointer differs):
  pointer : done segments upper-cased in the prompt of each line
  plain   : the name is never changed (control)

Usage:
  pointer.py train CKPT --arm {pointer,plain} --out-ckpt F.pt [--rounds 6 --steps 250]
  pointer.py eval CKPT --arm {pointer,plain} --split {test_a,test_b} --out F.json
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
from synth import FINAL, PRIM, generate_rows, grade, name_of  # noqa: E402
from locate import diagnose, ul_batch  # noqa: E402
from eval_samples import ByteLM, CachedGen  # noqa: E402

LINE_BYTES = 40


def head(prims, final, done, arm):
    segs = prims + ([final] if final else [])
    if arm == "pointer":
        segs = [s.upper() if i < done else s for i, s in enumerate(segs)]
    return f"def {'_then_'.join(segs)}(xs):\n"


def line_examples(prims, final, lines, arm, upto=None):
    """per body line: (prompt bytes + line bytes, weights 1 on the line only)."""
    out = []
    for i, l in enumerate(lines[:upto]):
        pre = ("\n\n" + head(prims, final, min(i, len(prims)), arm) + "".join(x + "\n" for x in lines[:i])).encode()
        tgt = (l + "\n").encode()
        out.append((pre + tgt, [0] * len(pre) + [1] * len(tgt)))
    return out


@torch.no_grad()
def write(g, tasks, k, arm, temp=0.8, batch=512):
    """tasks [(prims, final)] -> per task k sources (real name in the def line), written line by line."""
    rows = [(t, j) for t in range(len(tasks)) for j in range(k)]
    body = [[] for _ in rows]
    live = list(range(len(rows)))
    for step in range(max(len(p) for p, _ in tasks) + 2):       # primitives + return (+1 slack)
        if not live:
            break
        nxt_live = []
        prompts = {}
        for r in live:
            p, f = tasks[rows[r][0]]
            prompts[r] = ("\n\n" + head(p, f, min(len(body[r]), len(p)), arm)
                          + "".join(x + "\n" for x in body[r])).encode()
        # no left padding: the model reads absolute position as a line-count cue (padding 30 bytes turned the
        # second line into `return xs`), so prompts are batched only with prompts of the same length
        by_len = {}
        for r in list(live):
            if len(prompts[r]) + LINE_BYTES > g.ctx:            # would overflow the position table (HSA exception)
                live.remove(r)                                  # stop: the body stays unfinished -> graded wrong
                continue
        for r in live:
            by_len.setdefault(len(prompts[r]), []).append(r)
        for group in by_len.values():
            for c in range(0, len(group), batch):
                ids_rows = group[c:c + batch]
                ids = torch.tensor([list(prompts[r]) for r in ids_rows], device=g.m.device)
                gen = generate_rows(g, ids, LINE_BYTES, temp).cpu().numpy()
                for r, gr in zip(ids_rows, gen):
                    line = bytes(gr).decode("utf-8", "replace").split("\n")[0]
                    if not line.startswith("    ") or not line.strip():
                        continue                                # function ended (or garbage): stop this row
                    body[r].append(line)
                    if not line.strip().startswith("return"):
                        nxt_live.append(r)
        live = nxt_live
    out = [[] for _ in tasks]
    for (t, j), b in zip(rows, body):
        p, f = tasks[t]
        out[t].append(f"def {name_of(p, f)}(xs):\n" + "".join(x + "\n" for x in b))
    return out


def body_of(src):
    return [l for l in src.split("\n")[1:] if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("ckpt")
    ap.add_argument("--arm", required=True, choices=["pointer", "plain"])
    ap.add_argument("--out")
    ap.add_argument("--out-ckpt")
    ap.add_argument("--split", default="test_b")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--warm", type=int, default=500, help="steps on train.txt in the per-line format before round 0")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mem-frac", type=float, default=0.4)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(0)
    split = json.load(open(f"{synth.D}/splits.json"))
    lm = ByteLM(a.ckpt)
    g = CachedGen(lm)
    t0 = time.perf_counter()
    if a.cmd == "eval":
        if a.split.startswith("len"):                          # unseen LENGTH: names of 4 or 5 primitives
            r = random.Random(5)
            tasks = [[r.choices(list(PRIM), k=int(a.split[3:])), r.choice(list(FINAL))] for _ in range(a.n)]
        else:
            tasks = split[a.split][:]
        random.Random(1).shuffle(tasks)                         # same subset as synth.py eval / locate diagnose
        tasks = tasks[:a.n]
        srcs = write(g, tasks, a.k, a.arm)
        res = {"ckpt": a.ckpt, "arm": a.arm, "split": a.split, **synth.summary(grade(tasks, srcs)),
               "wall_s": round(time.perf_counter() - t0, 1),
               "examples": [s[0] for s in srcs[:6]]}
        json.dump(res, open(a.out, "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "examples"}))
        return
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = lm.model
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    replay = []
    for p, f in split["train"]:
        replay += line_examples(p, f, body_of(synth.source(p, f)), a.arm)
    rng = random.Random(0)

    def steps(n, own):
        model.train()
        for _ in range(n):
            bt = [rng.choice(own) for _ in range(a.batch // 2)] if own else []
            bt += [rng.choice(replay) for _ in range(a.batch - len(bt))]
            loss = ul_batch(model, bt, g.m.device, lm.ctx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        return float(loss)

    log = [{"warm": a.warm, "loss": round(steps(a.warm, []), 4)}]
    print(json.dumps(log[-1]), flush=True)
    pool = split["self"][:]
    for r in range(a.rounds):
        tasks = rng.sample(pool, a.n)
        srcs = write(g, tasks, a.k, a.arm)
        dg = diagnose(tasks, srcs)
        own = []
        for (p, f), ss, row in zip(tasks, srcs, dg):
            for s, d in zip(ss, row):
                if d["exact"]:                                  # every line state verified
                    own += line_examples(p, f, body_of(s), a.arm)
        flat = [d for row in dg for d in row]
        st = {"round": r, "exact": round(sum(d["exact"] for d in flat) / len(flat), 4),
              "middle_wrong": round(sum(d["wrong"] == 1 for d in flat) / len(flat), 4), "line_examples": len(own)}
        st["loss"] = round(steps(a.steps, own), 4)
        log.append(st)
        print(json.dumps(st), flush=True)
    ck["model"] = model.state_dict()
    ck["pointer_log"] = log
    os.makedirs(os.path.dirname(a.out_ckpt), exist_ok=True)
    torch.save(ck, a.out_ckpt)
    print("saved", a.out_ckpt, round(time.perf_counter() - t0, 1), "s")


if __name__ == "__main__":
    main()
