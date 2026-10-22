"""Edit by address with fill-in-the-middle instead of an instruction.

The 0.5B instruct model ignores "write only the region" and echoes the block unchanged
(run_patch_model.py: 0/144). Qwen2.5-Coder was trained with FIM tokens, so the address becomes the
prompt itself: code before the region is the prefix, code after it the suffix, the model writes the middle.
Arms: hint (failing calls as a comment just above the region) and nohint.
Each answer goes through the same patch + re-judge as run_patch_model.py.

Usage: fim_repair.py --out DIR [--k 8]
"""

import argparse
import json
import os
import time

os.environ.setdefault("HF_HOME", "/media/llm-data/models/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import gen_patch_prompts as G  # noqa: E402
import localize as L  # noqa: E402
import run_patch_model as R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMES = ["span_stmt", "span_suffix", "span_deep"]  # "function" has no region markers (whole rewrite)
STOPS = ["<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|fim_prefix|>", "<|repo_name|>"]


def split_block(p):
    """-> (prefix, suffix, indent, failing test lines) from the prepared instruction prompt."""
    user = p["user"]
    code = user.split("```python\n", 1)[1].rsplit("```", 1)[0]
    lines = code.split("\n")
    i = next(n for n, ln in enumerate(lines) if ln.strip() == G.MARK_START)
    j = next(n for n, ln in enumerate(lines) if ln.strip() == G.MARK_END)
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    tests = [ln[2:] for ln in user.split("\n") if ln.startswith("- ")]
    return "\n".join(lines[:i]) + "\n", "\n".join(lines[j + 1:]), indent, tests


def cut_to_region(text, indent):
    """Keep the middle only while it stays inside the region: stop at the first non-empty line indented
    less than the region (an empty suffix lets the model run on into new defs and test prints)."""
    keep = []
    for ln in text.split("\n"):
        if ln.strip() and not ln.startswith(indent):
            break
        keep.append(ln)
    return "\n".join(keep)


def fim_text(p, hint):
    prefix, suffix, indent, tests = split_block(p)
    if hint:
        prefix += "".join(f"{indent}# failing: {t}\n" for t in tests)
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"


def generate(tok, model, text, max_new, k, temperature):
    import torch
    enc = tok([text], return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new, stop_strings=STOPS, tokenizer=tok, pad_token_id=tok.eos_token_id)
    if k:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=k)
    else:
        kw.update(do_sample=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    s = time.perf_counter() - t0
    n = enc["input_ids"].shape[1]
    texts = tok.batch_decode(out[:, n:], skip_special_tokens=False)
    for st in STOPS:
        texts = [t.split(st)[0] for t in texts]
    n_new = int((out[:, n:] != tok.eos_token_id).sum())
    return texts, s, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = json.load(open(os.path.join(HERE, "prompts.json")))
    source = data["source"]
    baseline = L.judge_module(source)
    import torch
    torch.manual_seed(args.seed)
    tok, model = R.load_model(data["model"])
    results = []
    for p in data["prompts"]:
        if p["scheme"] not in SCHEMES:
            continue
        for hint in (True, False):
            text = fim_text(p, hint)
            max_new = 3 * (p["target_tokens"] or 60) + 40
            batches = [("greedy", *generate(tok, model, text, max_new, 0, 0))]
            if args.k:
                batches.append(("sample", *generate(tok, model, text, max_new, args.k, args.temperature)))
            r = {"name": p["name"], "scheme": p["scheme"], "hint": hint, "target_tokens": p["target_tokens"],
                 "prompt_tokens": len(tok(text)["input_ids"]), "answers": []}
            for kind, ts, s, n_new in batches:
                for t in ts:
                    a = R.judge_answer(source, p, R.clean_answer(cut_to_region(t, split_block(p)[2])), baseline)
                    a.update(kind=kind, answer=t, gen_s=round(s / len(ts), 3), new_tokens=round(n_new / len(ts), 1))
                    r["answers"].append(a)
            ok = [a for a in r["answers"] if a.get("fixed") and not a.get("regressions")]
            r["greedy_fixed"] = bool(r["answers"][0].get("fixed") and not r["answers"][0].get("regressions"))
            r["n_fixed"] = len(ok)
            results.append(r)
            a0 = r["answers"][0]
            print(f"{p['name']:10s} {p['scheme']:12s} hint {hint:d} prompt {r['prompt_tokens']:4d} "
                  f"greedy fixed {r['greedy_fixed']:d} rej {a0.get('rejected')} tok {a0.get('new_tokens')} "
                  f"fixed {len(ok)}/{len(r['answers'])}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"fim_results_s{args.seed}_k{args.k}.json")
    with open(path, "w") as fh:
        json.dump({"k": args.k, "results": results}, fh, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
