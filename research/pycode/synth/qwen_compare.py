"""Like-for-like reference: a real code LLM on the same synth2 task (ledger #23, registered 2026-10-23, CPU).

Same input as our planner+writer: the function head (name spells the program) is given, the model writes the body.
Knowledge parity: the prompt holds few-shot train functions chosen to cover every non-held unit at least once
(our system saw all of them in training). Greedy (temperature 0), k=1, no selector, same strict bwrap grade.
Model: Qwen2.5-Coder-0.5B-Instruct q8_0 on a running llama-server (CPU, -ngl 0), raw completion endpoint.
Prediction: test_a/test_num >= 80% (it reads the words), len3..6 lower but well above our R2 control (2%),
since a pretrained LM does not need a learned address to keep going; params 494M vs our ~162k.

Ledger #30 additions (registered 2026-10-26): --chat (Qwen chat template, function head prefilled inside a code fence,
body-only request; default stays the raw few-shot completion), --full (whole splits instead of the --n sample),
--suite {synth2,multifn,compose} (multifn: spec name `calc_add_sub_...` -> whole program; compose: prefix-tree name
`calc_hyp_avg_a_b_sub_a_b` -> whole program; shots = train programs covering every def/call symbol), graded scores and
failure taxonomy from partial_score.py next to pass1, JSON rewritten after each split.
multifn/compose raw prompts are `# <spec name>\\n` after the shots, with no extra instructions (parity with our planner).

Usage: qwen_compare.py --out DIR [--url http://127.0.0.1:8141 --n 100 --suite synth2 --chat --full --blank-ok --smoke]
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import compose as CP  # noqa: E402
import multifn as MF  # noqa: E402
import partial_score as PS  # noqa: E402
import synth2  # noqa: E402
from bridge_cpu import Data  # noqa: E402
from synth2 import UNITS, grade, head, source  # noqa: E402

STOP = ["\ndef ", "\n\n", "```", "\nclass ", "\nif __name__"]
STOP_BLANK_OK = [s for s in STOP if s != "\n\n"]   # --blank-ok: a blank line inside a body no longer cuts it (see report)
STOP_CHAT = ["```", "<|im_end|>", "\ndef ", "\nclass ", "\nif __name__"]
STOP_PROG = ["\n# calc_", "```", "\nif __name__"]    # whole programs: PEP8 blank lines between defs must not cut
SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def cover_shots(train, seed=0):
    """greedy set cover over train tasks: every non-held unit appears in some shot."""
    need = {u for u in UNITS if u not in synth2.HELD}
    r, pool, shots = random.Random(seed), list(train), []
    r.shuffle(pool)
    while need:
        best = max(pool, key=lambda t: len(need & set(t[0])))
        shots.append(best)
        need -= set(best[0])
    return shots


def cover_generic(train, cover, seed=0):
    """same greedy cover for multifn/compose: cover(task) -> set of symbols; need = every symbol seen in train."""
    r, pool, shots = random.Random(seed), list(train), []
    r.shuffle(pool)
    need = set().union(*(cover(t) for t in pool))
    while need:
        best = max(pool, key=lambda t: len(need & cover(t)))
        if not need & cover(best):
            break
        shots.append(best)
        need -= cover(best)
    return shots


def prompt_of(shots, task):
    return "".join(source(*t) + "\n\n" for t in shots) + head(*task)


def chat_wrap(user, prefill):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefill}")


def make_suite(name):
    """-> dict: sets, names, smoke split, raw/chat prompt builders, text->source, reference, stub, scorer, n_predict."""
    if name == "synth2":
        data = Data()
        shots = cover_shots(data.sets["train"])
        ex = "".join(f"```python\n{source(*s)}```\n\n" for s in shots)
        ask = ("Write the body of this function so that it matches its name. Reply with the function body only: "
               "the indented lines, no def line, no explanation.\n\n")
        return {"sets": data.sets, "names": ["test_a", "test_num", "len3", "len4", "len5", "len6"], "smoke": "len3",
                "check_task": data.sets["len4"][0], "n_shots": len(shots), "stop": STOP, "stop_chat": STOP_CHAT,
                "raw": lambda t: prompt_of(shots, t),
                "chat": lambda t: chat_wrap("Examples (the name spells the program):\n\n" + ex + ask
                                            + f"```python\n{head(*t)}```", "```python\n" + head(*t)),
                "src": lambda t, x: head(*t) + x.rstrip() + "\n", "ref": lambda t: source(*t),
                "stub": lambda t: head(*t) + "    return None\n",
                "n_pred": lambda ts: max(len(source(*t)) for t in ts),      # bytes >= tokens: never truncates a right body
                "score": PS.score_synth2}
    if name == "multifn":
        data = MF.Data()
        MF.PLAN_FORMAT = "grouped"
        prog = lambda sp: MF.assemble(MF.plan_of(sp))          # noqa: E731
        shots = cover_generic(data.sets["train"], lambda sp: {s for s in MF.plan_of(sp) if s.startswith("def:")})
        spec = MF.spec_name
        ask = ("Write the complete Python program for the spec named in the last comment: one def per op in the name "
               "and every helper those defs call, then `def main(op, a, b=0)` that dispatches each op in the name. "
               "Reply with code only.\n\n")
        return {"sets": data.sets, "names": ["test_comb", "len4", "len5", "len6", "len7", "len8", "basic", "full"],
                "smoke": "len4", "check_task": data.sets["len4"][0], "n_shots": len(shots), "stop": STOP_PROG,
                "stop_chat": ["```", "<|im_end|>"],
                "raw": lambda t: "".join(f"# {spec(s)}\n{prog(s)}\n" for s in shots) + f"# {spec(t)}\n",
                "chat": lambda t: chat_wrap("".join(f"```python\n# {spec(s)}\n{prog(s)}```\n\n" for s in shots) + ask
                                            + f"# {spec(t)}", "```python\n"),
                "src": lambda t, x: x.rstrip() + "\n", "ref": prog,
                "stub": lambda t: "def main(op, a, b=0):\n    return None\n",
                "n_pred": lambda ts: 2 * max(len(prog(t)) for t in ts),     # a right program may be longer than the reference
                "score": PS.score_multifn}
    data = CP.Data()
    prog = lambda p: CP.assemble(CP.plan_of(p), MF.block_text)  # noqa: E731
    shots = cover_generic(data.sets["train"], lambda p: {s for s in CP.plan_of(p) if s[:4] in ("def:", "call")})
    spec = CP.spec_name
    ask = ("The spec name is a call tree in prefix notation over the arguments a and b. Write the complete Python program "
           "for the spec named in the last comment: every helper it calls (and their dependencies), then `def main(a, b=0)` "
           "returning the tree. Reply with code only.\n\n")
    return {"sets": data.sets, "names": ["test_same", "calls4", "calls5", "calls6", "calls8"], "smoke": "calls4",
            "check_task": data.sets["calls4"][0], "n_shots": len(shots), "stop": STOP_PROG, "stop_chat": ["```", "<|im_end|>"],
            "raw": lambda t: "".join(f"# {spec(s)}\n{prog(s)}\n" for s in shots) + f"# {spec(t)}\n",
            "chat": lambda t: chat_wrap("".join(f"```python\n# {spec(s)}\n{prog(s)}```\n\n" for s in shots) + ask
                                        + f"# {spec(t)}", "```python\n"),
            "src": lambda t, x: x.rstrip() + "\n", "ref": prog, "stub": lambda t: "def main(a, b=0):\n    return None\n",
            "n_pred": lambda ts: 2 * max(len(prog(t)) for t in ts), "score": PS.score_compose}


def complete(url, prompt, n_predict, stop):
    """-> (text, tokens, stop_type); stop_type 'limit' = hit n_predict (possible truncation), 'error' = request failed."""
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
                       "stop": stop, "cache_prompt": True}).encode()
    req = urllib.request.Request(f"{url}/completion", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as f:
            d = json.loads(f.read())
    except Exception as e:  # noqa: BLE001 - one dead request must not kill a whole split
        print(json.dumps({"request_error": repr(e)[:200]}), flush=True)
        return "", 0, "error"
    return d["content"], d.get("tokens_predicted", 0), d.get("stop_type", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8141")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--par", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default="qwen05")
    ap.add_argument("--params", type=int, default=494_032_768)
    ap.add_argument("--suite", default="synth2", choices=["synth2", "multifn", "compose"])
    ap.add_argument("--chat", action="store_true", help="Qwen chat template instead of raw few-shot completion")
    ap.add_argument("--full", action="store_true", help="whole splits instead of the --n sample")
    ap.add_argument("--blank-ok", action="store_true", help="raw synth2: do not stop at a blank line (default stops, as the old run)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    su = make_suite(a.suite)
    # smoke: the judge must give the reference 1.0 and a stub body < 1.0 (known numbers), through the graded scorer
    t = su["check_task"]
    r_, b_ = su["score"]([t], [su["ref"](t)])[0], su["score"]([t], [su["stub"](t)])[0]
    smoke = {"suite": a.suite, "smoke_ref": r_["strict"], "smoke_broken": b_["strict"], "smoke_broken_label": b_["label"],
             "n_shots": su["n_shots"], "prompt_chars": len((su["chat"] if a.chat else su["raw"])(t))}
    if a.suite == "synth2":                                  # scorer must agree with the old grade()
        smoke["grade_ref"], smoke["grade_broken"] = grade([t], [[source(*t)]])[0][0], grade([t], [[su["stub"](t)]])[0][0]
        assert (smoke["grade_ref"], smoke["grade_broken"]) == (r_["strict"], b_["strict"])
    print(json.dumps(smoke), flush=True)
    assert r_["strict"] == 1.0 and r_["line_frac"] == 1.0 and b_["strict"] < 1.0
    names = [su["smoke"]] if a.smoke else su["names"]
    if a.smoke:
        a.n = 4
    stop = su["stop_chat"] if a.chat else (STOP_BLANK_OK if a.blank_ok and a.suite == "synth2" else su["stop"])
    tag = a.tag + ("_chat" if a.chat and not a.tag.endswith("_chat") else "")
    stem = (f"{tag}" if a.suite == "synth2" else f"{tag}_{a.suite}") + ("_smoke" if a.smoke else "_all" if a.full else "_full")
    path = f"{a.out}/{stem}.json"
    res = {"arm": tag, "suite": a.suite, "params": a.params, "n_shots": su["n_shots"], "chat": a.chat, "stop": stop,
           "sample": "whole splits" if a.full else f"Random(1).sample(n={a.n})",
           "decoding": "greedy k=1, no selector", "eval": {}}
    for k in names:
        pool = su["sets"][k]
        tasks = pool if a.full and not a.smoke else random.Random(1).sample(pool, min(a.n, len(pool)))
        t0 = time.perf_counter()
        n_pred = su["n_pred"](tasks)
        prompts = [(su["chat"] if a.chat else su["raw"])(t) for t in tasks]
        with ThreadPoolExecutor(a.par) as ex:
            outs = list(ex.map(lambda p: complete(a.url, p, n_pred, stop), prompts))
        gen_s = time.perf_counter() - t0
        srcs = [su["src"](t, body) for t, (body, _, _) in zip(tasks, outs)]
        sc = su["score"](tasks, srcs)
        sm = PS.summarize(sc)
        ok = sum(s["pass"] for s in sc)
        res["eval"][k] = {"n": len(tasks), "pass1": sm.pop("pass1"),
                          "text_exact": round(sum(s == su["ref"](t) for s, t in zip(srcs, tasks)) / len(tasks), 4),
                          "tokens": sum(n for _, n, _ in outs), "gen_s": round(gen_s, 1),
                          "correct_per_s": round(ok / gen_s, 3),
                          "n_predict": n_pred, "hit_limit": sum(st == "limit" for _, _, st in outs),
                          "request_errors": sum(st == "error" for _, _, st in outs),
                          "scores": {m: v for m, v in sm.items() if m not in ("n", "taxonomy", "first_wrong_unit_hist")},
                          "taxonomy": sm["taxonomy"], "first_wrong_unit_hist": sm.get("first_wrong_unit_hist", {}),
                          "fails": [s for s, q in zip(srcs, sc) if not q["pass"]][:3],
                          "fail_labels": [q["label"] for q in sc if not q["pass"]][:3]}
        print(json.dumps({k: {m: v for m, v in res["eval"][k].items() if m not in ("fails", "fail_labels")}}), flush=True)
        json.dump(res, open(path, "w"), indent=1)               # after each split: a dead run keeps what finished


if __name__ == "__main__":
    main()
