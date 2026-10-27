"""Statement-level writer for the mini-interpreter family (ledger #55 / I6, 10-27): the missing piece of the belt.

His words (#55): "quero escrever um projeto pequeno, mas complexo". His architecture idea: planner and writer are
different networks ("fit more things in"). The writer must COMPOSE new bodies (if/for/while/dict/list/str/try/return,
calls to siblings) from a hierarchical spec node (signature + callee signatures + contract), not pick 1 of 15 ops.

Arms (byte-level transformers trained from scratch on the 1050 train units of interp55_family, same data):
  flat     one decoder: prompt bytes SEP body bytes EOS                                   (d256 L8, ~6.4 M params)
  pw       PLANNER (prompt SEP plan EOS; plan = one token per body line = indent depth : AST statement kind,
           e.g. 1:If 2:Return 0:AssignT) + WRITER (prompt SEP plan LSEP then per line [plan token] line bytes '\\n';
           the plan token is FORCED at every line start, so the writer fills statements the planner chose, and
           indentation comes from the plan)                            (planner d192 L4 + writer d256 L6, ~6.5 M)
  pw_gold  diagnostic: the same writer with the GOLD plan (greedy, K=1) -> is the planner the bottleneck?
Budget: params matched within ~5 %, same optimizer steps per network, same batch, same K, same selector
(interp55_eval.py). Sampling: candidate 0 greedy, candidates 1..K-1 temperature --temp; for pw each candidate is one
planner sample + one writer sample (candidate 0 greedy/greedy).

Usage:
  interp55_writer.py --data D/data.json --out DIR --seed 0 [--steps 4000 --K 8 --device cuda --mem-frac 0.75]
  CPU smoke: --device cpu --smoke (tiny dims, 60 steps, 6 prompts)
Writes DIR/{flat,pw,pw_gold}_s{seed}.cands.json (+ train logs), then score with interp55_eval.py score.

PRE-REGISTERED PREDICTIONS (written 2026-10-27 13:06, before any trained-model score; floors already measured:
gold 1.000 everywhere; const 0; mutant unit 0.188; nn unit 0.034 / non-leaked 0.000, prog test_comp 0.081 =
exactly the 13/160 programs without a novel unit, test_seen 1.000). Metric = sel@8, mean over 3 seeds.
  P1 flat byte writer: unit non-leaked 0.03-0.15; test_comp 0.08-0.20; test_seen >= 0.90 (memorisation).
  P2 pw vs flat (his planner/writer split): |pw - flat| <= 0.05 on unit non-leaked (no clear win at 1050 units).
  P3 pw_gold greedy - pw greedy >= 0.05 on units (pw_gold is K=1: greedy vs greedy only) (the planner, not the writer, is the bottleneck on novel compositions).
  P4 Qwen-Coder-0.5B LoRA: unit non-leaked 0.15-0.40, beats flat by >= 0.10; 1.5B LoRA >= 0.5B LoRA.
  P5 Coder-3B 2-shot (no training): unit non-leaked 0.20-0.45; test_comp 0.15-0.40.
  P6 every arm: tokenize (96 of 176 units, longest bodies) below the rest of the units.
  P7 every trained arm beats nn on non-leaked units (> 0.000) and on test_comp (> 0.081).
"""

import argparse
import ast
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp55_eval as EV  # noqa: E402

PAD, SEP, EOS, LSEP = 256, 257, 258, 259
PT0 = 260
NL = 10
KINDS = ["If", "Elif", "Else", "While", "For", "Try", "Except", "Return", "Raise", "Break", "Continue", "Pass",
         "Assign", "AssignT", "AugAssign", "Call", "Other"]
MAXD = 6
NPT = MAXD * len(KINDS)
VOCAB = PT0 + NPT


def line_kind(s):
    s = s.strip()
    for pre, k in (("elif ", "Elif"), ("else:", "Else"), ("if ", "If"), ("while ", "While"), ("for ", "For"),
                   ("try:", "Try"), ("except", "Except"), ("return", "Return"), ("raise", "Raise"), ("break", "Break"),
                   ("continue", "Continue"), ("pass", "Pass")):
        if s.startswith(pre):
            return k
    try:
        n = ast.parse(s).body[0]
    except SyntaxError:
        return "Other"
    if isinstance(n, ast.Assign):
        return "AssignT" if isinstance(n.targets[0], ast.Tuple) else "Assign"
    if isinstance(n, ast.AugAssign):
        return "AugAssign"
    if isinstance(n, ast.Expr):
        return "Call"
    return "Other"


def plan_of(body):
    """-> [(depth, kind, content)] per non-empty line."""
    out = []
    for l in body.rstrip("\n").split("\n"):
        if not l.strip():
            continue
        ind = len(l) - len(l.lstrip(" "))
        out.append((min(ind // 4, MAXD - 1), line_kind(l), l.strip()))
    return out


def pt(depth, kind):
    return PT0 + depth * len(KINDS) + KINDS.index(kind)


def pt_dec(t):
    t -= PT0
    return t // len(KINDS), KINDS[t % len(KINDS)]


def enc(s):
    return list(s.encode("utf-8"))


def seq_flat(prompt, body):
    p = enc(prompt) + [SEP]
    return p, enc(body) + [EOS]


def seq_planner(prompt, body):
    p = enc(prompt) + [SEP]
    return p, [pt(d, k) for d, k, _ in plan_of(body)] + [EOS]


def writer_prefix(prompt, plan):
    return enc(prompt) + [SEP] + [pt(d, k) for d, k in plan] + [LSEP]


def seq_writer(prompt, body):
    pl = plan_of(body)
    p = writer_prefix(prompt, [(d, k) for d, k, _ in pl])
    t = []
    for d, k, c in pl:
        t += [pt(d, k)] + enc(c) + [NL]
    return p, t + [EOS]


# ------------------------------------------------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask, cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        a = Fn.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, d, L, h, ctx, drop=0.1):
        super().__init__()
        self.ctx = ctx
        self.emb, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h, drop) for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, ids, pos, mask, caches=None):
        x = self.drop(self.emb(ids) + self.pos(pos.clamp(max=self.ctx - 1)))
        for i, b in enumerate(self.blocks):
            x = b(x, mask, None if caches is None else caches[i])
        return self.head(self.ln(x))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------------------------------------------------ train
def batchify(pairs, idx, dev):
    seqs = [pairs[i][0] + pairs[i][1] for i in idx]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    lab = torch.full((len(seqs), T), -100, dtype=torch.long)
    for r, i in enumerate(idx):
        p, t = pairs[i]
        s = p + t
        ids[r, :len(s)] = torch.tensor(s)
        lab[r, len(p) - 1:len(s) - 1] = torch.tensor(t)
    return ids.to(dev), lab.to(dev)


def train(model, pairs, steps, batch, lr, seed, dev, log, tag, forced_pt=False, micro=0):
    """forced_pt: writer -> no loss on plan tokens inside the body (they are forced at decode time)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    r = random.Random(seed)
    order = list(range(len(pairs)))
    model.train()
    t0 = time.time()
    cast = dev.startswith("cuda")
    for s in range(steps):
        if s * batch % len(pairs) < batch:
            r.shuffle(order)
        idx = [order[(s * batch + i) % len(pairs)] for i in range(batch)]
        lrs = lr * min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / steps))
        for g in opt.param_groups:
            g["lr"] = lrs
        opt.zero_grad(set_to_none=True)
        # review 10-27: math SDPA keeps [B,h,T,T] per layer -> gradient accumulation over micro-batches; the loss is
        # token-weighted over the whole batch, so micro only changes memory, not the objective
        mb = micro or batch
        ntok = 0
        parts = []
        for j in range(0, batch, mb):
            ids, lab = batchify(pairs, idx[j:j + mb], dev)
            if forced_pt:
                lab = torch.where(lab >= PT0, torch.full_like(lab, -100), lab)
            parts.append((ids, lab))
            ntok += int((lab != -100).sum())
        loss_sum = 0.0
        for ids, lab in parts:
            T = ids.shape[1]
            pos = torch.arange(T, device=dev).unsqueeze(0).expand_as(ids)
            mask = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cast):
                logits = model(ids, pos, mask)
            lp = Fn.cross_entropy(logits.float().view(-1, VOCAB), lab.view(-1), ignore_index=-100, reduction="sum")
            (lp / max(1, ntok)).backward()
            loss_sum += lp.item()
        loss = torch.tensor(loss_sum / max(1, ntok))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 250 == 0 or s == steps - 1:
            row = {"net": tag, "step": s, "loss": round(loss.item(), 4), "s": round(time.time() - t0, 1)}
            if cast:
                row["vram_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            log.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    return round(time.time() - t0, 1)


# --------------------------------------------------------------------------------------------------------- decoding
@torch.no_grad()
def generate(model, prefixes, max_new, temps, allow, dev, force_fn=None, seed=0):
    """batched left-padded decoding with a KV cache.
    prefixes: list of token lists; temps: per-row temperature (0 = greedy); allow: bool [VOCAB] tensor of tokens the
    model may emit; force_fn(row, generated so far) -> forced token or None. Returns generated lists (EOS excluded)."""
    B = len(prefixes)
    g = torch.Generator(device="cpu").manual_seed(seed)
    P = max(len(p) for p in prefixes)
    pad = [P - len(p) for p in prefixes]
    ids = torch.full((B, P), PAD, dtype=torch.long)
    for r, p in enumerate(prefixes):
        ids[r, pad[r]:] = torch.tensor(p)
    ids = ids.to(dev)
    padt = torch.tensor(pad, device=dev)
    ar = torch.arange(P, device=dev)
    pos = (ar.unsqueeze(0) - padt.unsqueeze(1)).clamp(min=0)
    keyok = ar.unsqueeze(0) >= padt.unsqueeze(1)                                    # [B, P]
    mask = (torch.ones(P, P, dtype=torch.bool, device=dev).tril().unsqueeze(0) & keyok.unsqueeze(1)).unsqueeze(1)
    mask = mask | (~keyok).unsqueeze(1).unsqueeze(-1) & torch.eye(P, dtype=torch.bool, device=dev)  # pad rows: self
    caches = [{} for _ in model.blocks]
    cast = dev.startswith("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cast):
        logits = model(ids, pos, mask, caches)[:, -1].float()
    out = [[] for _ in range(B)]
    done = [False] * B
    cur = torch.tensor([len(p) for p in prefixes], device=dev)
    temps_t = torch.tensor(temps, dtype=torch.float32)
    neg = torch.tensor(float("-inf"))
    allow_c = allow.cpu()
    for _ in range(max_new):
        lg = logits.cpu()
        lg = torch.where(allow_c.unsqueeze(0), lg, neg)
        greedy = lg.argmax(-1)
        probs = torch.softmax(lg / temps_t.clamp(min=1e-3).unsqueeze(1), -1)
        samp = torch.multinomial(probs, 1, generator=g).squeeze(1)
        nxt = torch.where(temps_t > 0, samp, greedy)
        nl = nxt.tolist()
        for r in range(B):
            if done[r]:
                nl[r] = PAD
                continue
            f = force_fn(r, out[r]) if force_fn else None
            if f is not None:
                nl[r] = f
            if nl[r] == EOS:
                done[r] = True
            else:
                out[r].append(nl[r])
        if all(done):
            break
        step = torch.tensor(nl, device=dev).unsqueeze(1)
        keyok = torch.cat([keyok, torch.ones(B, 1, dtype=torch.bool, device=dev)], 1)
        m = keyok.unsqueeze(1).unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cast):
            logits = model(step, cur.unsqueeze(1), m, caches)[:, -1].float()   # cur = real length
        cur = cur + 1
    return out


def dec_bytes(toks):
    return bytes(t for t in toks if t < 256).decode("utf-8", "replace")


def gen_flat(model, prompts, K, temp, dev, bs, seed, max_new):
    allow = torch.zeros(VOCAB, dtype=torch.bool)
    allow[:256] = True
    allow[EOS] = True
    rows = [(p, j) for p in prompts for j in range(K)]
    out = {}
    for b in range(0, len(rows), bs):
        chunk = rows[b:b + bs]
        res = generate(model, [enc(p) + [SEP] for p, _ in chunk], max_new, [0.0 if j == 0 else temp for _, j in chunk],
                       allow.to(dev), dev, seed=seed * 1000 + b)
        for (p, j), t in zip(chunk, res):
            out.setdefault(EV.sha(p), []).append(EV.clean_body(dec_bytes(t)))
    return out


def gen_plans(planner, prompts, K, temp, dev, bs, seed):
    allow = torch.zeros(VOCAB, dtype=torch.bool)
    allow[PT0:] = True
    allow[EOS] = True
    rows = [(p, j) for p in prompts for j in range(K)]
    plans = {}
    for b in range(0, len(rows), bs):
        chunk = rows[b:b + bs]
        res = generate(planner, [enc(p) + [SEP] for p, _ in chunk], 90, [0.0 if j == 0 else temp for _, j in chunk],
                       allow.to(dev), dev, seed=seed * 1000 + b + 7)
        for (p, j), t in zip(chunk, res):
            plans.setdefault(p, []).append([pt_dec(x) for x in t if x >= PT0] or [(0, "Return")])
    return plans


def gen_writer(writer, jobs, temp_of, dev, bs, seed, max_new):
    """jobs: [(key, prompt, plan, j)] -> {key: [body]} in job order."""
    allow = torch.zeros(VOCAB, dtype=torch.bool)
    allow[:256] = True
    allow[NL] = True
    out = {}
    for b in range(0, len(jobs), bs):
        chunk = jobs[b:b + bs]

        def force(r, so_far, chunk=chunk):
            plan = chunk[r][2]
            n_lines = sum(1 for x in so_far if x == NL)
            if not so_far or so_far[-1] == NL:
                return pt(*plan[n_lines]) if n_lines < len(plan) else EOS
            k = len(so_far) - 1 - max((i for i, x in enumerate(so_far) if x >= PT0), default=0)
            return NL if k >= 200 else None
        res = generate(writer, [writer_prefix(p, plan) for _, p, plan, _ in chunk], max_new,
                       [temp_of(j) for *_, j in chunk], allow.to(dev), dev, force_fn=force, seed=seed * 1000 + b + 13)
        for (key, p, plan, j), t in zip(chunk, res):
            lines, cur, depth = [], [], 0
            for x in t:
                if x >= PT0:
                    depth = pt_dec(x)[0]
                    cur = []
                elif x == NL:
                    lines.append("    " * depth + dec_bytes(cur))
                    cur = []
                else:
                    cur.append(x)
            body = "\n".join(lines) + "\n" if lines else "return None\n"
            out.setdefault(key, []).append(EV.clean_body(body))   # same cleaning as flat/Qwen (review 10-27)
    return out


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--gen-bs", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-frac", type=float, default=0.75)
    ap.add_argument("--arms", default="flat,pw")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--micro", type=int, default=4, help="micro-batch for gradient accumulation (memory only)")
    ap.add_argument("--probe", action="store_true", help="VRAM probe: 3 steps on the longest batch, no generation")
    ap.add_argument("--overfit", type=int, default=0,
                    help="positive control: train on the N shortest train units only and decode THOSE prompts; "
                         "greedy must reproduce the gold text (train/decode format identity, KV-cache parity)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    if dev.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    data = json.load(open(a.data))
    tr = data["train_units"]
    prompts = EV.needed_prompts(data)
    gold_body = {u["prompt"]: u["body"] for row in data["variants"].values() for u in row["units"]}
    fe_prompts = [fe["prompt"] for fe in data["func_eval"]]
    ctx = 2400
    if a.overfit:
        tr = sorted(tr, key=lambda u: len(u["prompt"]) + len(u["body"]))[:a.overfit]
        prompts = fe_prompts = [u["prompt"] for u in tr]
        gold_body.update({u["prompt"]: u["body"] for u in tr})
        cfg = {"flat": (128, 2, 4), "planner": (64, 2, 4), "writer": (128, 2, 4)}
        a.K = 1
    elif a.smoke:
        cfg = {"flat": (64, 2, 4), "planner": (48, 1, 4), "writer": (64, 2, 4)}
        a.steps, a.K, a.batch = 60, 2, 4
        prompts = fe_prompts[:3] + prompts[:3]
        fe_prompts = fe_prompts[:3]
    else:
        cfg = {"flat": (256, 8, 8), "planner": (192, 4, 6), "writer": (256, 6, 8)}
    longest = max(len(seq_writer(u["prompt"], u["body"])[0]) + len(seq_writer(u["prompt"], u["body"])[1]) for u in tr)
    longest_f = max(sum(map(len, seq_flat(u["prompt"], u["body"]))) for u in tr)
    max_body = max(len(seq_flat(u["prompt"], u["body"])[1]) for u in tr)
    assert max(longest, longest_f) < ctx, (longest, longest_f)
    max_new = max_body + 200                         # generation cap > longest gold body (gate 6)
    meta = {"seed": a.seed, "steps": a.steps, "batch": a.batch, "lr": a.lr, "K": a.K, "temp": a.temp, "ctx": ctx,
            "longest_train_seq": max(longest, longest_f), "max_new": max_new, "n_train": len(tr),
            "n_prompts": len(prompts), "device": dev, "smoke": a.smoke}
    log = []
    arms = a.arms.split(",")
    if a.probe:                                      # VRAM probe on the real dims: the longest train units, 3 steps
        tr = sorted(tr, key=lambda u: -sum(map(len, seq_writer(u["prompt"], u["body"]))))[:a.batch]
        a.steps = 3
    print(json.dumps(meta), flush=True)
    if "flat" in arms:
        torch.manual_seed(a.seed)
        m = GPT(*cfg["flat"], ctx).to(dev)
        pairs = [seq_flat(u["prompt"], u["body"]) for u in tr]
        tt = train(m, pairs, a.steps, a.batch, a.lr, a.seed, dev, log, "flat", micro=a.micro)
        if a.probe:
            probe_report(dev, "flat", a.micro)
        else:
            t0 = time.time()
            cands = gen_flat(m, prompts, a.K, a.temp, dev, a.gen_bs, a.seed, max_new)
            gm = {"params": nparams(m), "train_s": tt, "gen_s": round(time.time() - t0, 1), "micro": a.micro, **meta}
            json.dump({"arm": "flat", "seed": a.seed, "K": a.K, "meta": gm, "cands": cands},
                      open(os.path.join(a.out, f"flat_s{a.seed}.cands.json"), "w"))
            print(json.dumps({"arm": "flat", **{k: gm[k] for k in ("params", "train_s", "gen_s")}}), flush=True)
        del m
    if "pw" in arms:
        torch.manual_seed(a.seed)
        pl = GPT(*cfg["planner"], ctx).to(dev)
        wr = GPT(*cfg["writer"], ctx).to(dev)
        tp = train(pl, [seq_planner(u["prompt"], u["body"]) for u in tr], a.steps, a.batch, a.lr, a.seed, dev, log,
                   "planner", micro=a.micro)
        tw = train(wr, [seq_writer(u["prompt"], u["body"]) for u in tr], a.steps, a.batch, a.lr, a.seed + 100, dev,
                   log, "writer", forced_pt=True, micro=a.micro)
        if a.probe:
            probe_report(dev, "pw", a.micro)
            return
        t0 = time.time()
        plans = gen_plans(pl, prompts, a.K, a.temp, dev, a.gen_bs, a.seed)
        jobs = [(EV.sha(p), p, plans[p][j], j) for p in prompts for j in range(a.K)]
        cands = gen_writer(wr, jobs, lambda j: 0.0 if j == 0 else a.temp, dev, a.gen_bs, a.seed, max_new)
        gs = round(time.time() - t0, 1)
        plan_exact = sum(plans[p][0] == [(d, k) for d, k, _ in plan_of(gold_body[p])] for p in fe_prompts) / \
            max(1, len(fe_prompts))
        gm = {"params": nparams(pl) + nparams(wr), "params_planner": nparams(pl), "params_writer": nparams(wr),
              "train_s": tp + tw, "gen_s": gs, "func_eval_plan_exact_greedy": round(plan_exact, 4), **meta}
        json.dump({"arm": "pw", "seed": a.seed, "K": a.K, "meta": gm, "cands": cands},
                  open(os.path.join(a.out, f"pw_s{a.seed}.cands.json"), "w"))
        print(json.dumps({"arm": "pw", **{k: gm[k] for k in ("params", "train_s", "gen_s",
                                                              "func_eval_plan_exact_greedy")}}), flush=True)
        # diagnostic: gold plan, greedy writer, K = 1 (every prompt, so programs can be scored too)
        jobs = [(EV.sha(p), p, [(d, k) for d, k, _ in plan_of(gold_body[p])], 0) for p in prompts]
        t0 = time.time()
        cg = gen_writer(wr, jobs, lambda j: 0.0, dev, a.gen_bs, a.seed, max_new)
        json.dump({"arm": "pw_gold", "seed": a.seed, "K": 1, "meta": {**gm, "gen_s": round(time.time() - t0, 1),
                                                                       "note": "GOLD plan: diagnostic, not an arm"},
                   "cands": cg}, open(os.path.join(a.out, f"pw_gold_s{a.seed}.cands.json"), "w"))
    json.dump(log, open(os.path.join(a.out, f"trainlog_{'_'.join(arms)}_s{a.seed}.json"), "w"))
    if a.overfit:
        rep = {}
        for arm in ("flat", "pw", "pw_gold"):
            f = os.path.join(a.out, f"{arm}_s{a.seed}.cands.json")
            if os.path.exists(f):
                c = json.load(open(f))["cands"]
                rep[arm] = sum(c[EV.sha(p)][0] == gold_body[p] for p in prompts)
        print(json.dumps({"overfit_exact": rep, "n": len(prompts)}), flush=True)
        if "flat" in arms:
            print(json.dumps({"kv_parity_max_abs_diff": kv_parity(a, cfg, ctx, tr, dev)}), flush=True)


def probe_report(dev, tag, micro):
    vram = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.startswith("cuda") else None
    print(json.dumps({"probe_ok": tag, "vram_gb": vram, "micro": micro}), flush=True)
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def kv_parity(a, cfg, ctx, tr, dev):
    """incremental (KV cache, left-padded batch) logits == full forward logits on the same tokens."""
    torch.manual_seed(0)
    m = GPT(*cfg["flat"], ctx, drop=0.0).to(dev).eval()
    pre = [enc(u["prompt"]) + [SEP] for u in tr[:3]]
    seen = []

    def force(r, so_far):
        return None
    allow = torch.ones(VOCAB, dtype=torch.bool)
    allow[PAD] = False
    gen = generate(m, pre, 12, [0.0] * len(pre), allow.to(dev), dev, force_fn=force)
    worst = 0.0
    with torch.no_grad():
        for p, g in zip(pre, gen):
            full = p + g
            ids = torch.tensor([full], device=dev)
            T = len(full)
            lg = m(ids, torch.arange(T, device=dev).unsqueeze(0), torch.ones(T, T, dtype=torch.bool, device=dev).tril())
            want = lg[0, len(p) - 1:T - 1].argmax(-1).tolist()
            seen.append(want == g[:len(want)])
            worst = max(worst, 0.0 if want == g[:len(want)] else 1.0)
    return {"greedy_tokens_match": seen}


if __name__ == "__main__":
    main()
