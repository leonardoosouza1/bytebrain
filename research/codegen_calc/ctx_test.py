"""Context test: does a bank of signatures replace the full source as context?

The expression evaluator is split into 7 pieces (parser_ref.py). Each piece is generated with
one of three contexts, then judged inside a module whose other pieces are the reference ones:
  none   only the module header and the target stub
  full   header + full source of every other piece (what an agent re-reading the file would send)
  bank   header + only signatures and docstrings of the other pieces (from an exact symbol bank)
Per arm: greedy + K samples per piece, visible-example selection, hidden pass rate, prompt tokens.
End to end: the 7 selected pieces of an arm are assembled alone and evaluate() is judged.

Usage: ctx_test.py --out DIR [--k 8] [--score-only]
"""

import argparse
import json
import os
import random
import time

import evaluate as ev
import judge
import parser_ref as pr
import spec

ARMS = ["none", "full", "bank"]
ASK_NONE = "Implement the function below for a calculator expression parser in Python.\n\n```python\n{ctx}\n\n{stub}```"
ASK_CTX = ("Here is the module so far:\n\n```python\n{ctx}```\n\n"
           "Implement the missing function below. It must work with the functions above.\n\n```python\n{stub}```")

VALID = ["1+2*3", "(1+2)*3", "2^3^2", "-2^2", "10/4", "2*(3+4)*5", "sqrt(16)+1", "-(3-5)", "1.5*4", "2^-1",
         "ln(e)", "log(1000)", "sin(0)+cos(0)", "pi*2", "3-2-1", "8/2/2", "--3", "abs(-7)", "(((4)))",
         "7-+2", "100*(2+12)/14", "e^2", "5*-1", "2^0.5"]
INVALID = ["1/0", "2+", "(1+2", "3 4", "foo(2)", "", "1..2", "sqrt(-1)", "1+2)", "2 $ 3", "sin 3", "()"]


def stub(name):
    p = pr.PIECES[name]
    return f'{p["sig"]}\n    """{p["doc"]}"""\n'


def context(arm, target):
    others = [n for n in pr.ORDER if n != target]
    if arm == "none":
        return pr.HEADER
    if arm == "full":
        return pr.HEADER + "\n\n" + pr.source(others) + "\n"
    return pr.HEADER + "\n\n" + "\n\n".join(stub(n) + "    ...\n" for n in others)


def cases(name, ref):
    """(visible, hidden) argument tuples for one piece."""
    r = random.Random(f"ctx-{name}")
    if name in ("tokenize", "evaluate"):
        pool = [(e,) for e in VALID + INVALID]
        vis = [("2*(3+4)",), ("1..2",)] if name == "tokenize" else [("1+2*3",), ("2^3^2",), ("(1+2",)]
    else:
        pool = []
        for e in VALID + ["1/0", "2+", "(1+2", "1+2)", "()"]:
            toks = ref["tokenize"](e)
            for pos in sorted({0, min(1, len(toks)), min(2, len(toks))}):
                pool.append((toks, pos))
        r.shuffle(pool)
        pool = pool[:40]
        vis = [(ref["tokenize"]("2*3+1"), 0), (ref["tokenize"]("(4)"), 0)]
    return vis, pool


def expected(ref, name, args):
    try:
        v = ref[name](*args)
        return ("ok", tuple(v) if isinstance(v, (tuple, list)) else v)
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__)


def with_examples(name, ref, vis):
    lines = []
    for a in vis:
        kind, v = expected(ref, name, a)
        call = f"{name}({', '.join(repr(x) for x in a)})"
        lines.append(f">>> {call}\n{v!r}" if kind == "ok" else f">>> {call}\nTraceback (most recent call last):\n{v}")
    ex = "\n".join("    " + ln for blk in lines for ln in blk.split("\n"))
    p = pr.PIECES[name]
    return f'{p["sig"]}\n    """{p["doc"]}\n\n{ex}\n    """\n'


def generate_all(args, ref):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import gen
    tok = AutoTokenizer.from_pretrained(gen_model(args))
    model = AutoModelForCausalLM.from_pretrained(gen_model(args), dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    torch.manual_seed(0)
    out = {}
    for name in pr.ORDER:
        vis, _ = cases(name, ref)
        st = with_examples(name, ref, vis)
        for arm in ARMS:
            ctx = context(arm, name)
            user = ASK_NONE.format(ctx=pr.HEADER, stub=st) if arm == "none" else ASK_CTX.format(ctx=ctx, stub=st)
            prefix = pr.HEADER + "\n\n" + st
            text = gen.prompt_ids(tok, user, prefix)
            n_prompt = len(tok(text)["input_ids"])
            t0 = time.time()
            g, _ = gen.generate(model, tok, text, 1, 512, False)
            s, n_tok = gen.generate(model, tok, text, args.k, 512, True)
            dt = time.time() - t0
            out[f"{name}/{arm}"] = {"codes": [prefix + x for x in g + s], "prompt_tokens": n_prompt,
                                    "gen_s": round(dt, 2), "gen_tokens": n_tok}
            print(f"{name:12s} {arm:5s} prompt {n_prompt:5d} tok  {dt:5.1f}s", flush=True)
    return out


def gen_model(args):
    return args.model


def judge_piece(name, codes, ref):
    vis, hidden = cases(name, ref)
    others = pr.source([n for n in pr.ORDER if n != name])
    calls = [list(a) for a in vis + hidden]
    jobs = [{"code": ev.clean(c) + "\n\n" + others, "fname": name, "calls": calls} for c in codes]
    res = judge.run_jobs(jobs)
    ev_vis = [expected(ref, name, a) for a in vis]
    ev_hid = [expected(ref, name, a) for a in hidden]
    rows = []
    for o in res:
        if o["load"] != "ok":
            rows.append({"vis": -1, "hidden": False, "sig": None})
            continue
        v = sum(judge.matches(e, g) for e, g in zip(ev_vis, o["out"][:len(vis)]))
        h = [judge.matches(e, g) for e, g in zip(ev_hid, o["out"][len(vis):])]
        rows.append({"vis": v, "hidden": all(h), "hidden_frac": sum(h) / len(h),
                     "sig": "|".join(judge.signature(g) for g in o["out"])})
    return rows


def score(cand, ref):
    report = {"pieces": {}, "arms": {}}
    picks = {arm: {} for arm in ARMS}
    for name in pr.ORDER:
        for arm in ARMS:
            c = cand[f"{name}/{arm}"]
            rows = judge_piece(name, c["codes"], ref)
            idx = list(range(len(rows)))
            iv = ev.pick_visible([{**r, "visible": r["vis"]} for r in rows], idx)
            picks[arm][name] = c["codes"][iv]
            report["pieces"][f"{name}/{arm}"] = {
                "greedy": rows[0]["hidden"], "sample@1": sum(r["hidden"] for r in rows[1:]) / max(1, len(rows) - 1),
                "visible": rows[iv]["hidden"], "oracle": any(r["hidden"] for r in rows),
                "prompt_tokens": c["prompt_tokens"], "gen_s": c["gen_s"]}
            p = report["pieces"][f"{name}/{arm}"]
            print(f"{name:12s} {arm:5s} greedy {p['greedy']:d} s@1 {p['sample@1']:.2f} vis {p['visible']:d} "
                  f"oracle {p['oracle']:d} prompt {p['prompt_tokens']}", flush=True)
    _, hidden = cases("evaluate", ref)
    for arm in ARMS:
        ps = [report["pieces"][f"{n}/{arm}"] for n in pr.ORDER]
        module = "\n\n".join(ev.clean(picks[arm][n]) for n in pr.ORDER)
        res = judge.run_jobs([{"code": module, "fname": "evaluate", "calls": [list(a) for a in hidden]}])[0]
        e2e = res["load"] == "ok" and all(judge.matches(expected(ref, "evaluate", a), g)
                                          for a, g in zip(hidden, res["out"]))
        e2e_frac = (sum(judge.matches(expected(ref, "evaluate", a), g) for a, g in zip(hidden, res["out"]))
                    / len(hidden)) if res["load"] == "ok" else 0.0
        report["arms"][arm] = {k: round(sum(float(p[k]) for p in ps) / len(ps), 3)
                               for k in ("greedy", "sample@1", "visible", "oracle")}
        report["arms"][arm]["prompt_tokens_mean"] = round(sum(p["prompt_tokens"] for p in ps) / len(ps))
        report["arms"][arm]["gen_s_total"] = round(sum(p["gen_s"] for p in ps), 1)
        report["arms"][arm]["end_to_end_evaluate"] = e2e
        report["arms"][arm]["end_to_end_case_frac"] = round(e2e_frac, 3)
        report["arms"][arm]["module"] = module
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ref = pr.module()
    path = os.path.join(args.out, "ctx_candidates.json")
    if not args.score_only:
        cand = generate_all(args, ref)
        with open(path, "w") as fh:
            json.dump(cand, fh)
    cand = json.load(open(path))
    report = score(cand, ref)
    for arm in ARMS:
        a = {k: v for k, v in report["arms"][arm].items() if k != "module"}
        print(arm, json.dumps(a), flush=True)
    with open(os.path.join(args.out, "ctx_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
