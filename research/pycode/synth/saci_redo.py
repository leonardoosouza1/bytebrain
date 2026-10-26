"""saci-code step 4 (ledger #46) + C4: when the monitor fires, redo the understander's spec from there; does it pay?

Step 2 (`saci_monitor.py`) showed the student's own token uncertainty flags failing specs mid-generation (S3 bank,
5-fold CV, AUROC .914; at 10% false alarms on passes it fires on 106/152 failures, median 2 tokens after the first
token that differs from the gold string). His step: "ele avisa o que precisa e isso gera novamente alguns passos".
Here the warning triggers a redo and the belt grades the result.

Data: v2pool (593 Sonnet rewrites), student s5 (Qwen2.5-Coder-0.5B + LoRA), the recorded greedy run with token rows
(`student_s5_s1_tok/raws.json`, same 593 raws as the eval) and its belt traces (end_pass). Monitor = step 2 exactly:
the fold model that did not see the request, threshold = 90th percentile of the passes' CV scores. Fire index f =
first token where the fold model on the running max of the bank crosses the threshold.

Arms (only fired requests are touched unless the arm says otherwise; k samples at T=1, no top-k/top-p/penalty):
  second_fire    token f replaced by the model's best other token there, greedy after (decoding as the eval)
  second_back    same at b = the highest-entropy token in [0, f]
  back_k         k samples continuing from gen[:b]
  rewind4_k      k samples continuing from gen[:max(0, f-4)]   ("gera novamente alguns passos")
  rewind8_k      k samples continuing from gen[:max(0, f-8)]
  randpos_k      k samples from a uniform random position in [0, f] (control: the monitor's WHERE)
  full_k         k samples from the start (control: redo only the tail vs redo all)
  full_k_random  full_k on random requests, as many as fired (control: the trigger)
  full_k_always  full_k on every request (control: the trigger)
  vote_*         C4: execution vote over the same candidates (vote_full_k, vote_full_k_always, vote_rewind8_k)
Selector (gold-free; candidates = the original + the k samples): specs that pass the bridge with no flag; among them
the lowest monitor score (the request's fold model on the candidate's own token rows); none valid -> the original.
Vote (gold-free): run each valid candidate's reference module (the spec's own blocks, `E.oracle_plan`) on the probe
inputs, group by outputs, keep the largest group (ties: lowest monitor score); none valid -> the original.
Grade: the belt as in the eval (planner net + writer net, e2e/ckpt, bwrap) -> end_pass. Gold enters only here.
Oracle ceiling (diagnostic, never a result): a fired failure with >= 1 candidate that passes.

Registered 2026-10-26 19:05, before any run of this file.
Positive controls (smoke and full): PC1 the monitor reproduces step 2: 106 fired failures, 44-45 fired passes.
PC2 the 593 original raws graded here give 441. PC3 greedy continuation from gen[:b+1] reproduces the recorded raw on
>= 95% of the checked fired requests. PC4 sampling: processed scores == raw logits (the model's generation_config
carries top_k 20 / top_p 0.8 / T 0.7 / penalty 1.05; all must be off). Any PC FAIL -> no verdict.
Offline arithmetic (recorded rows, no model): b equals the first wrong token in 34/106 fired failures and comes after
it in 56 (the error stays in the kept prefix); a start f-r is at or before it in 46/53/63/81/98 of 106 for
r = 0/2/4/6/8. r = 4 and 8 were chosen after seeing this on the same pool (dev use, gate 4): rewind vs back is not
a held-out comparison.
P1 (his idea): rewind4_k fixes >= 20 of the 106 fired failures and breaks <= fixes/2 of the fired passes -> net >= +10.
P2 (where): rewind8_k fixes >= rewind4_k fixes > back_k fixes (b keeps the error in 56/106).
P3 (no selector): second_back net <= 0 (the false alarms are changed blindly).
P4 (trigger): full_k net >= full_k_random net + 10; full_k_always net <= full_k net + 5.
P5 (C4): vote_full_k_always net >= +5.
Ceiling: full_k oracle >= 50/106 fired failures.
KILL (step 4): every selector arm (back_k, rewind4_k, rewind8_k, randpos_k, full_k) net <= 0 -> on this understander a
fire-triggered redo does not pay at k=8; the monitor stays a warning only.
Amendment 2026-10-26 ~19:12, after an Opus review and before any run (no number seen):
A1 candidate rows carry the TOP-1 logprob in the nll slot, not the sampled token's. The monitor was fit on greedy rows
(chosen = argmax, nll in [0, 1.2]); a T=1 draw of p=.05 at one token moved the fold score by a median -6.8 logits, so
"lowest score" picked the rarest sample (142/150 fired originals would drop below threshold from one draw). For greedy
rows top-1 == chosen, so PC1 and the original's score are unchanged.
A2 PC2 is per request: every original graded here equals its trace end_pass (593/593), and the trace grade wins in the
merge. Planner batch dependence (learned pad embedding inside the conv window) measured on CPU: 0/501 plans differ
alone vs batched vs traced.
A3 a replaced token that is <|im_end|> ends the spec (no greedy continuation after the end token).
A4 P4b is dropped as a test: the 443 non-fired requests hold 397 passes and 46 failures, so "always" can fix <= 46 and
break up to 397; it holds whatever the trigger does (gate 7). Fixes/breaks on non-fired requests are reported per arm
as a diagnostic. Smoke: P4 is n/a (scope = smoke fired only); the smoke random subset is sorted (hash-seed free).
Limits: one pool (593), one student seed, k=8, the monitor and its threshold were fit on this pool by 5-fold CV;
first-wrong is measured against the gold string, so an equivalent spec counts as wrong early.

Usage: saci_redo.py --out DIR [--k 8] [--smoke N]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_pipeline as E  # noqa: E402
import multifn as MF  # noqa: E402
import saci_monitor as SM  # noqa: E402
import student_understander as SU  # noqa: E402

RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
V = f"{RUN}/e2e/v2pool"
SEL_ARMS = ("back_k", "rewind4_k", "rewind8_k", "randpos_k", "full_k")


def rows_of(logits, chosen):
    """raw logits [T, V], chosen ids [T] -> [[id, top-1 logprob, entropy, margin]]; equals
    `student_understander --token-scores` on greedy rows."""
    lp = logits.float().log_softmax(-1)
    top2 = lp.topk(2, -1).values
    ent = -(lp.exp() * lp).sum(-1)
    clp = top2[:, 0]        # top-1 logprob, not the chosen token's: the monitor was fit on greedy rows (amendment A1)
    return [[int(t), round(float(c), 4), round(float(e), 4), round(float(m), 4)]
            for t, c, e, m in zip(chosen, clp, ent, top2[:, 0] - top2[:, 1])]


class Student:
    def __init__(self, lora, mem_frac):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.cuda.set_per_process_memory_fraction(mem_frac)
        self.tok = AutoTokenizer.from_pretrained(SU.MODEL, cache_dir=SU.CACHE, local_files_only=True)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        base = AutoModelForCausalLM.from_pretrained(SU.MODEL, cache_dir=SU.CACHE, local_files_only=True,
                                                    dtype=torch.bfloat16).to("cuda")
        self.model = PeftModel.from_pretrained(base, lora).merge_and_unload().eval()
        fmt = json.load(open(os.path.join(os.path.dirname(os.path.normpath(lora)), "result.json"))).get(
            "spec_format", "expr")
        self.max_new = SU.MAX_NEW[fmt]
        self.pc4 = [0, 0]                                   # sampling calls where scores == logits, calls

    def prompt_ids(self, text):
        return self.tok(SU.prompt(text), add_special_tokens=False).input_ids

    def decode(self, gen):
        return self.tok.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def forced(self, p, gen):
        """teacher-forced logits at each generated position: [len(gen), V]."""
        x = torch.tensor([p + gen], device="cuda")
        return self.model(input_ids=x).logits[0, len(p) - 1:len(p) + len(gen) - 1].float()

    @torch.no_grad()
    def greedy(self, prefix, n_new):
        ids = torch.tensor([prefix], device="cuda")
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                                  do_sample=False, eos_token_id=self.im_end, pad_token_id=self.tok.pad_token_id,
                                  repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
        return out[0, len(prefix):].tolist()

    @torch.no_grad()
    def sample(self, prefix, n_new, k, seed):
        """k continuations of prefix at T=1 on the full distribution -> [(new ids up to and with im_end, rows)]."""
        torch.manual_seed(seed)
        ids = torch.tensor([prefix], device="cuda")
        out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new,
                                  do_sample=True, num_return_sequences=k, temperature=1.0, top_k=0, top_p=1.0,
                                  repetition_penalty=1.0, eos_token_id=self.im_end, pad_token_id=self.tok.pad_token_id,
                                  output_logits=True, output_scores=True, return_dict_in_generate=True)
        self.pc4[0] += all(a is b or torch.equal(a, b) for a, b in zip(out.logits, out.scores))
        lg = torch.stack(out.logits, 1)
        self.pc4[1] += 1
        res = []
        for j in range(k):
            seq = out.sequences[j, len(prefix):].tolist()
            n = seq.index(self.im_end) + 1 if self.im_end in seq else len(seq)
            res.append((seq[:n], rows_of(lg[j, :n], out.sequences[j, len(prefix):len(prefix) + n])))
        return res


class Monitor:
    """step 2's S3 bank, rebuilt exactly (same request order, same CV folds)."""

    def __init__(self, raws, tr):
        self.ids = [i for i in tr if i in raws and raws[i].get("tok")]
        self.y = np.array([not tr[i]["end_pass"] for i in self.ids])
        B = {i: SM.bank(raws[i]["tok"]) for i in self.ids}
        X3 = np.array([B[i].max(0) for i in self.ids])
        self.s3, models = SM.cv_scores(X3, self.y)
        self.thr = float(np.quantile(self.s3[~self.y], 0.9))
        self.model = dict(zip(self.ids, models))
        self.fire = {}
        for i in self.ids:
            run = np.maximum.accumulate(B[i], 0)
            self.fire[i] = next((t for t in range(len(run)) if self.model[i](run[t:t + 1])[0] > self.thr), None)

    def score(self, i, rows):
        return float(self.model[i](SM.bank(rows).max(0)[None])[0])


def valid(raw):
    _, b, fl = E.check_spec(raw)
    return b if b and not fl else None


def select(cands):
    """cands [(raw, score)] with cands[0] the original -> raw: lowest score among bridge-valid, else the original."""
    ok = [(s, n) for n, (r, s) in enumerate(cands) if valid(r)]
    return cands[min(ok)[1]][0] if ok else cands[0][0]


def vote(cands, sig):
    """execution vote: largest group of identical reference outputs; ties and within the group: lowest score."""
    groups = {}
    for r, s in cands:
        g = sig.get(r)
        if g is not None:
            groups.setdefault(g, []).append((s, r))
    if not groups:
        return cands[0][0]
    best = max(groups.values(), key=lambda m: (len(m), -min(m)[0]))
    return min(best)[1]


def signatures(raws):
    """raw -> reference-output signature (gold-free: the spec's own blocks on the probe inputs) or None."""
    todo, out = {}, {}
    B = E.Batch()
    for r in dict.fromkeys(raws):
        b = valid(r)
        if b is None:
            out[r] = None
            continue
        keys = sorted(f"d:{o}" for o in b[1]) if b[0] == "dispatch" else ["x:"]
        todo[r] = (B.add(E.assemble(b[0], E.oracle_plan(b), MF.block_text), keys), keys)
    B.run()
    for r, (h, keys) in todo.items():
        out[r] = json.dumps([[k, B[h][k]] for k in keys])
    return out


def grade(pairs, reqs, P, W):
    """[(id, raw)] -> {(id, raw): end_pass} through the eval's belt."""
    uniq = list(dict.fromkeys(pairs))
    res = {}
    for c in range(0, len(uniq), 400):
        chunk = uniq[c:c + 400]
        T = E.run_belt([reqs[i] for i, _ in chunk], [r for _, r in chunk], P, W)
        res.update({key: t["end_pass"] for key, t in zip(chunk, T)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=0, help="N fired failures + N fired passes only")
    ap.add_argument("--lora", default=f"{RUN}/e2e/student_s5_s1/lora")
    ap.add_argument("--ckpt", default=f"{RUN}/e2e/ckpt")
    ap.add_argument("--mem-frac", type=float, default=0.25)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pc3-n", type=int, default=30, help="fired requests checked for the greedy-prefix parity")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.perf_counter()
    raws = json.load(open(f"{V}/student_s5_s1_tok/raws.json"))
    tr = {t["id"]: t for t in map(json.loads, open(f"{V}/traces_student_s5.jsonl"))}
    reqs = {q["id"]: q for q in json.load(open(f"{V}/requests.json"))}
    M = Monitor(raws, tr)
    fired = [i for i in M.ids if M.fire[i] is not None]
    fail = {i: not tr[i]["end_pass"] for i in M.ids}
    res = {"pc1": {"fired_fail": sum(fail[i] for i in fired), "fired_pass": sum(not fail[i] for i in fired),
                   "threshold": M.thr}}
    res["pc1"]["pass"] = res["pc1"]["fired_fail"] == 106 and res["pc1"]["fired_pass"] in (44, 45)
    print(json.dumps(res["pc1"]), flush=True)
    P, W = E.Planner("net", a.ckpt), E.Writer("net", a.ckpt)
    orig = {i: raws[i]["raw"] for i in M.ids}
    assert all(orig[i] == tr[i]["raw_spec"] for i in M.ids), "token-score raws differ from the traced raws"
    g0 = grade([(i, orig[i]) for i in M.ids], reqs, P, W)
    res["pc2"] = {"original_pass": sum(g0[(i, orig[i])] for i in M.ids),
                  "differ_from_trace": sum(g0[(i, orig[i])] != tr[i]["end_pass"] for i in M.ids)}
    res["pc2"]["pass"] = res["pc2"]["original_pass"] == 441 and res["pc2"]["differ_from_trace"] == 0
    print(json.dumps(res["pc2"]), flush=True)

    rng = np.random.default_rng(0)
    rand_set = set(rng.choice(M.ids, size=len(fired), replace=False).tolist())
    if a.smoke:
        fired = [i for i in fired if fail[i]][:a.smoke] + [i for i in fired if not fail[i]][:a.smoke]
        rand_set = set(sorted(rand_set)[:2 * a.smoke])
        full_set = set(fired) | rand_set
    else:
        full_set = set(M.ids)
    S = Student(a.lora, a.mem_frac)
    rows, pc3, gen_ms = {}, [0, 0], {}
    for n, i in enumerate(M.ids):
        if i not in full_set:
            continue
        p = S.prompt_ids(reqs[i]["text"])
        gen = [r[0] for r in raws[i]["tok"]]
        f = M.fire[i]
        cand = {"original": [(orig[i], M.score(i, raws[i]["tok"]))]}
        t0 = time.perf_counter()
        cand["full_k"] = [(S.decode(g), M.score(i, rw)) for g, rw in S.sample(p, S.max_new, a.k, 1000 + n)]
        ms = {"full_k": 1000 * (time.perf_counter() - t0)}
        row = {"id": i, "fail": fail[i], "fire": f, "len": len(gen), "cand": cand, "ms": ms}
        if i in fired:
            ent = [r[2] for r in raws[i]["tok"]]
            b = int(np.argmax(ent[:f + 1]))
            starts = {"back_k": b, "rewind4_k": max(0, f - 4), "rewind8_k": max(0, f - 8),
                      "randpos_k": int(np.random.default_rng(2000 + n).integers(0, f + 1))}
            row.update(back=b, starts=starts)
            lg = S.forced(p, gen)
            row["forced_argmax_agree"] = int((lg.argmax(-1).cpu() == torch.tensor(gen)).sum())
            for arm, t in (("second_fire", f), ("second_back", b)):
                t0 = time.perf_counter()
                order = lg[t].argsort(descending=True).tolist()
                alt = next(x for x in order if x != gen[t])
                new = gen[:t] + [alt]
                if alt != S.im_end:
                    new += S.greedy(p + new, max(1, S.max_new - t - 1))
                cand[arm] = [(S.decode(new), None)]
                ms[arm] = 1000 * (time.perf_counter() - t0)
            for arm, t in starts.items():
                t0 = time.perf_counter()
                cand[arm] = [(S.decode(gen[:t] + g), M.score(i, raws[i]["tok"][:t] + rw))
                             for g, rw in S.sample(p + gen[:t], S.max_new - t, a.k, 3000 + 10 * n + len(cand))]
                ms[arm] = 1000 * (time.perf_counter() - t0)
            if pc3[1] < a.pc3_n:
                pc3[1] += 1
                tail = S.greedy(p + gen[:b + 1], max(1, S.max_new - b - 1)) if gen[b] != S.im_end else []
                pc3[0] += S.decode(gen[:b + 1] + tail) == orig[i]
        rows[i] = row
        if len(rows) % 50 == 0:
            print(f"{len(rows)}/{len(full_set)} requests sampled, {time.perf_counter() - t_start:.0f} s", flush=True)
    res["pc3"] = {"reproduced": pc3[0], "checked": pc3[1], "pass": pc3[1] > 0 and pc3[0] >= 0.95 * pc3[1]}
    res["pc4"] = {"equal_calls": S.pc4[0], "calls": S.pc4[1], "pass": S.pc4[0] == S.pc4[1] > 0}
    print(json.dumps({k: res[k] for k in ("pc3", "pc4")}), flush=True)

    # choices per arm; requests an arm does not touch keep the original
    every = [r for row in rows.values() for c in row["cand"].values() for r, _ in c]
    sig = signatures(every + list(orig.values()))
    chosen = {}
    for i in M.ids:
        row = rows.get(i)
        o = row["cand"]["original"] if row else [(orig[i], None)]
        fired_i = i in fired
        ch = {}
        for arm in SEL_ARMS:
            ch[arm] = select(o + row["cand"][arm]) if fired_i and arm in row["cand"] else orig[i]
        for arm in ("second_fire", "second_back"):
            ch[arm] = row["cand"][arm][0][0] if fired_i else orig[i]
        ch["full_k_random"] = select(o + row["cand"]["full_k"]) if i in rand_set and row else orig[i]
        ch["full_k_always"] = select(o + row["cand"]["full_k"]) if row else orig[i]
        ch["vote_full_k"] = vote(o + row["cand"]["full_k"], sig) if fired_i else orig[i]
        ch["vote_full_k_always"] = vote(o + row["cand"]["full_k"], sig) if row else orig[i]
        ch["vote_rewind8_k"] = vote(o + row["cand"]["rewind8_k"], sig) if fired_i else orig[i]
        chosen[i] = ch
    pairs = [(i, r) for i in M.ids for r in chosen[i].values()]
    pairs += [(i, r) for i in fired for arm in SEL_ARMS for r, _ in rows[i]["cand"][arm]]
    g = {**grade(pairs, reqs, P, W), **g0}                  # the originals keep the checked grade (A2)
    scope = fired if a.smoke else M.ids
    arms = {}
    for arm in chosen[M.ids[0]]:
        fx = sum(fail[i] and g[(i, chosen[i][arm])] for i in scope)
        br = sum(not fail[i] and not g[(i, chosen[i][arm])] for i in scope)
        arms[arm] = {"pass": sum(g[(i, chosen[i][arm])] for i in scope), "fixes": fx, "breaks": br, "net": fx - br,
                     "fixes_silent": sum(fail[i] and g[(i, chosen[i][arm])] and not
                                         any(f["from"] == "validator" for f in tr[i]["flags"]) for i in scope),
                     "changed": sum(chosen[i][arm] != orig[i] for i in scope),
                     "nonfired_fixes": sum(fail[i] and g[(i, chosen[i][arm])] for i in scope if i not in fired),
                     "nonfired_breaks": sum(not fail[i] and not g[(i, chosen[i][arm])] for i in scope
                                            if i not in fired)}
    res["scope"] = {"requests": len(scope), "original_pass": sum(g0[(i, orig[i])] for i in scope),
                    "fired": len(fired), "fired_fail": sum(fail[i] for i in fired)}
    res["arms"] = arms
    res["oracle_any_pass_on_fired_fail"] = {arm: sum(any(g[(i, r)] for r, _ in rows[i]["cand"][arm])
                                                     for i in fired if fail[i]) for arm in SEL_ARMS}
    res["gen_ms_median"] = {arm: round(float(np.median([rows[i]["ms"][arm] for i in fired if arm in rows[i]["ms"]])), 1)
                            for arm in ("full_k", "second_fire", "second_back") + SEL_ARMS[:4]}
    res["forced_argmax_agree"] = [sum(rows[i]["forced_argmax_agree"] for i in fired),
                                  sum(rows[i]["len"] for i in fired)]
    res["p4"] = "n/a in smoke (scope = smoke fired only)" if a.smoke else "full"
    res["config"] = {"k": a.k, "smoke": a.smoke, "lora": a.lora, "ckpt": a.ckpt, "max_new": S.max_new,
                     "wall_s": round(time.perf_counter() - t_start, 1)}
    json.dump(res, open(f"{a.out}/redo.json", "w"), indent=1)
    with open(f"{a.out}/rows.jsonl", "w") as fh:
        for i in M.ids:
            if i in rows:
                fh.write(json.dumps({**rows[i], "chosen": chosen[i],
                                     "chosen_pass": {k: g[(i, r)] for k, r in chosen[i].items()}}) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
