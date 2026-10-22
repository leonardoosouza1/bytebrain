"""Regenerate a whole function on llama-server, many samples, pick by visible cases, confirm hidden.

For functions that are wrong at the root (evaluate became a split()-based RPN stack), a region fix
cannot help. Prompt = `import math` + signature + docstring (+ optional plan comment); the model writes
the body as a plain completion. Each answer replaces the function in the file (AST span), is judged on the
seed-2 cases (what the user showed) and on the hidden seeds 0,1.

Usage: whole_function.py --name evaluate --out DIR [--k 128] [--url http://127.0.0.1:8139]
"""

import argparse
import ast
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import fim_repair as F
import localize as L

STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\n# ", "\nprint(", "```", "<|endoftext|>", "<|im_end|>"]
PLANS = {
    "plain": "",
    "plan": ("    # Plan: tokenize the string into numbers, names, operators and parentheses, then parse with\n"
             "    # recursive descent: expr (+ -) -> term (* /) -> unary (- +) -> power (^, right assoc) -> atom\n"
             "    # (number, constant, function call, parenthesised expr). Any leftover token raises ValueError.\n"),
}


def header_of(source, name):
    """-> (signature + docstring text, (first line, last line) of the function, 1-based)."""
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.split("\n")
    doc_end = fn.body[0].end_lineno
    return "\n".join(lines[fn.lineno - 1:doc_end]) + "\n", (fn.lineno, fn.end_lineno)


def replace_function(source, span, text):
    lines = source.split("\n")
    return "\n".join(lines[:span[0] - 1] + text.rstrip("\n").split("\n") + lines[span[1]:])


def complete(url, text, max_new, temperature, seed):
    import urllib.request
    body = {"prompt": text, "n_predict": max_new, "temperature": temperature, "top_p": 0.95, "seed": seed,
            "stop": STOPS, "cache_prompt": True}
    req = urllib.request.Request(url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out["content"], out.get("tokens_predicted", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="evaluate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=900)
    ap.add_argument("--url", default="http://127.0.0.1:8139")
    args = ap.parse_args()
    source = json.load(open(os.path.join(F.HERE, "prompts.json")))["source"]
    head, span = header_of(source, args.name)
    pool = ThreadPoolExecutor(32)
    report = {"name": args.name, "k": args.k, "arms": {}}
    for arm, plan in PLANS.items():
        prompt = "import math\n\n\n" + head + plan
        t0 = time.perf_counter()
        outs = list(pool.map(lambda s: complete(args.url, prompt, args.max_new, args.temperature, 2000 + s),
                             range(args.k)))
        gen_s = time.perf_counter() - t0
        rows = []
        for text, n_tok in outs:
            code = head + plan + text
            try:
                ast.parse(code)
            except SyntaxError:
                rows.append({"vis": False, "hid": False, "syntax": False, "tokens": n_tok})
                continue
            new = replace_function(source, span, code)
            vis = L.judge_module(new, names=[args.name], seeds=(2,))[args.name]
            hid = L.judge_module(new, names=[args.name])[args.name] if vis else None
            rows.append({"vis": vis, "hid": hid, "syntax": True, "tokens": n_tok, "code": code if vis else None})
        n_vis = sum(r["vis"] for r in rows)
        n_hid = sum(bool(r["hid"]) for r in rows)
        # hidden pass rate over ALL samples needs a hidden run for non-visible ones too (oracle count)
        oracle = 0
        for (text, _), r in zip(outs, rows):
            if r["syntax"] and not r["vis"]:
                new = replace_function(source, span, head + plan + text)
                oracle += L.judge_module(new, names=[args.name])[args.name]
        oracle += n_hid
        a = {"gen_s": round(gen_s, 1), "tokens_mean": round(sum(r["tokens"] for r in rows) / len(rows)),
             "syntax_ok": sum(r["syntax"] for r in rows), "visible_pass": n_vis, "visible_and_hidden": n_hid,
             "hidden_pass_any": oracle, "picks": [r["code"] for r in rows if r["hid"]][:3]}
        report["arms"][arm] = a
        print(f"{arm:6s} gen {gen_s:6.1f}s tok {a['tokens_mean']:4d} syntax {a['syntax_ok']:3d}/{args.k} "
              f"visible {n_vis:3d} -> hidden {n_hid:3d}   hidden any {oracle}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"whole_{args.name}_k{args.k}.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
