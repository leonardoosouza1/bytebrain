"""Generate calculator candidates with a frozen coder model.

Arms produced here (scored later by evaluate.py):
  whole     one greedy pass that writes the whole module from all stubs
  greedy    one greedy pass per function, anchored on its signature + docstring
  samples   K sampled passes per function, same anchor

Usage: gen.py --out DIR [--k 16] [--model Qwen/Qwen2.5-Coder-0.5B-Instruct]
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import spec

ASK = "Implement this function of a scientific calculator in Python. Use only the standard library.\n\n```python\n{}```"
ASK_WHOLE = ("Write a Python module `calc.py` for a scientific calculator that implements every function below. "
             "Use only the standard library.\n\n```python\nimport math\n\n\n{}```")
HEAD = "import math\n\n\n"


def prompt_ids(tok, user, prefix):
    msgs = [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "```python\n" + prefix
    return text


def cut(text):
    return text.split("```")[0]


def generate(model, tok, text, n, max_new, sample):
    enc = tok([text], return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new, stop_strings=["```"], tokenizer=tok, pad_token_id=tok.eos_token_id)
    if sample:
        kw.update(do_sample=True, temperature=0.8, top_p=0.95, num_return_sequences=n)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    L = enc["input_ids"].shape[1]
    texts = tok.batch_decode(out[:, L:], skip_special_tokens=True)
    n_tok = int((out[:, L:] != tok.eos_token_id).sum())
    return [cut(t) for t in texts], n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    torch.manual_seed(args.seed)
    res = {"model": args.model, "k": args.k, "funcs": {}}

    t0 = time.time()
    stubs = "\n\n".join(spec.stub(e) for e in spec.SPEC)
    text = prompt_ids(tok, ASK_WHOLE.format(stubs), HEAD)
    body, n_tok = generate(model, tok, text, 1, 6000, False)
    res["whole"] = {"code": HEAD + body[0], "s": round(time.time() - t0, 2), "tokens": n_tok}
    print(f"whole: {res['whole']['s']}s {n_tok} tok", flush=True)

    for e in spec.SPEC:
        st = spec.stub(e)
        text = prompt_ids(tok, ASK.format(st), HEAD + st)
        max_new = 1024 if e["name"] == "evaluate" else 384
        t1 = time.time()
        g, g_tok = generate(model, tok, text, 1, max_new, False)
        t2 = time.time()
        s, s_tok = generate(model, tok, text, args.k, max_new, True)
        t3 = time.time()
        res["funcs"][e["name"]] = {"greedy": HEAD + st + g[0], "samples": [HEAD + st + x for x in s],
                                   "greedy_s": round(t2 - t1, 2), "samples_s": round(t3 - t2, 2),
                                   "greedy_tok": g_tok, "samples_tok": s_tok}
        print(f"{e['name']}: greedy {t2 - t1:.1f}s samples {t3 - t2:.1f}s", flush=True)
    res["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, f"candidates_s{args.seed}.json"), "w") as fh:
        json.dump(res, fh)
    print(f"done {res['total_s']}s", flush=True)


if __name__ == "__main__":
    main()
