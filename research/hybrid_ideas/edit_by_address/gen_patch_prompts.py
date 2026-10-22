"""Build the repair prompts the model would get in each address scheme (no model is loaded).

For each failing function of calc_generated.py:
  1. run its hidden cases (seed 1) in the sandbox with traceback capture (localize.py)
  2. vote a site line (innermost traceback line, else return site) -> addresses (addressing.locate)
  3. emit one prompt per scheme:
       whole_file    whole module + failing tests -> model rewrites the whole module
       function      imports + signatures it calls + the function -> model rewrites the function
       span_stmt     the function with the located function-body statement marked -> model writes the region
       span_suffix   same, region = located statement .. end of the function body
       span_deep     same, region = innermost statement containing the site line
  4. store, per prompt: address, chat text, response prefix, prompt/context tokens, the oracle target
     (what a correct answer for that region is, from fixes.py "minimal" family) and its token count,
     and whether the region can contain the fix at all (covers_fix).

Usage: gen_patch_prompts.py [--out prompts.json]
Tokenizer only: HF_HOME=/media/llm-data/models/hf HF_HUB_OFFLINE=1.
"""

import argparse
import ast
import json
import os

os.environ.setdefault("HF_HOME", "/media/llm-data/models/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import addressing as A  # noqa: E402
import fixes as F  # noqa: E402
import localize as L  # noqa: E402

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "```python\n"
SCHEMES = ["whole_file", "function", "span_stmt", "span_suffix", "span_deep"]
SPAN_ADDR = {"span_stmt": "symbol_stmt", "span_suffix": "symbol_suffix", "span_deep": "path_deep"}
MARK_START, MARK_END = "# >>> EDIT REGION START", "# <<< EDIT REGION END"


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def n_tok(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True) + PREFIX


def _lines_of(source, start, end):
    """char offsets -> [first_line, last_line] (1-based, inclusive); insertion -> [l, l-1]."""
    first = source.count("\n", 0, start) + 1
    last = source.count("\n", 0, end) if end > start else first - 1
    return first, last


def region_target(orig_func, fixed_func, a, b):
    """Function-relative line region [a, b) of orig_func; the replacement that turns orig_func into
    fixed_func if everything outside the region is unchanged, else None (region misses the fix)."""
    F_, G = orig_func.splitlines(), fixed_func.splitlines()
    tail = len(F_) - b
    if F_[:a] != G[:a] or len(G) < a + tail or (tail and F_[b:] != G[len(G) - tail:]):
        return None
    return "\n".join(G[a:len(G) - tail])


def tests_text(fail, k=3):
    """Up to k failing tests, preferring distinct site lines (different symptoms first)."""
    seen, pick = set(), []
    for r in fail:
        key = (tuple(r["info"]["tb"][-1:]), r["info"]["ret"])
        if key not in seen:
            seen.add(key)
            pick.append(r)
    pick += [r for r in fail if r not in pick]
    fail = pick
    return "\n".join(f"- {r['msg']}" for r in fail[:k])


def build(tok, source, name, fail, loc, fixed_source):
    tree = ast.parse(source)
    func, node = A.function_source(source, name, tree)
    fixed_func, _ = A.function_source(fixed_source, name)
    fstart, fend, _ = A.resolve(source, {"kind": "symbol", "name": name, "stmt": None}, tree)
    f_first = _lines_of(source, fstart, fend)[0]
    imports = A.module_imports(tree)
    sigs = A.called_signatures(source, name, tree)
    tests = tests_text(fail)
    site = loc["site_line"]
    site_text = source.splitlines()[site - 1].strip()
    where = f"line {site} (`{site_text}`), found by {loc['site_how'].replace('_', ' ')}"
    out = []

    # whole file
    user = ("This Python module `calc.py` has a bug. These tests fail:\n" + tests +
            f"\nThe failure is in `{name}`, at {where}.\n"
            "Rewrite the whole module with the bug fixed. Keep every other function unchanged.\n\n"
            f"```python\n{source}```")
    out.append(dict(scheme="whole_file", address={"kind": "file"}, context=source, user=user,
                    target=fixed_source, target_kind="minimal"))

    # function + what it calls
    ctx = "\n".join(imports) + "\n\n"
    if sigs:
        ctx += "# functions defined elsewhere in the module (signatures only):\n" + \
               "\n".join(s + " ..." for s in sigs) + "\n\n"
    ctx += func
    user = (f"The function `{name}` below has a bug. These tests fail:\n" + tests +
            f"\nThe failure is at {where}.\n"
            f"Write the corrected `{name}` function (the whole function, same signature).\n\n"
            f"```python\n{ctx}```")
    out.append(dict(scheme="function", address=loc["addresses"]["symbol_func"], context=ctx, user=user,
                    target=fixed_func, target_kind="minimal"))

    # span schemes: same function context, region marked with x/y coordinates
    for scheme, key in SPAN_ADDR.items():
        addr = loc["addresses"][key]
        s, e, indent = A.resolve(source, addr, tree)
        l1, l2 = _lines_of(source, s, e)
        a, b = l1 - f_first, l2 - f_first + 1
        flines = func.splitlines()
        pad = " " * (indent or 0)
        marked = "\n".join(flines[:a] + [pad + MARK_START] + flines[a:b] + [pad + MARK_END] + flines[b:]) + "\n"
        region = "\n".join(flines[a:b])
        col = indent or 0
        ctx = "\n".join(imports) + "\n\n"
        if sigs:
            ctx += "# functions defined elsewhere in the module (signatures only):\n" + \
                   "\n".join(x + " ..." for x in sigs) + "\n\n"
        ctx += marked
        user = (f"The function `{name}` below has a bug. These tests fail:\n" + tests +
                f"\nThe failure is at {where}.\n"
                f"Only the region between `{MARK_START}` and `{MARK_END}` may change: "
                f"lines {l1}-{l2} of the file (x = line, y = column {col}).\n"
                f"Write ONLY the new code for that region (it may be more or fewer lines), "
                f"indented as in the function. Do not repeat code outside the region.\n\n"
                f"```python\n{ctx}```")
        tgt, how = region_target(func, fixed_func, a, b), "minimal"
        if tgt is None:
            tgt, how = F.alt_target(name, scheme), "alternative"
        if tgt is None:
            how = None
        # failing calls that never execute the region: no edit confined to it can change their result
        region_set = set(range(l1, l2 + 1))
        unreached = [r["call"] for r in fail if not region_set & set(r["info"]["lines"])]
        out.append(dict(scheme=scheme, address=addr, region_lines=[l1, l2], region=region,
                        context=ctx, user=user, target=tgt, target_kind=how,
                        failing_calls_not_reaching_region=unreached))

    for p in out:
        p["name"] = name
        p["chat"] = chat(tok, p["user"])
        p["prefix"] = PREFIX
        p["prompt_tokens"] = n_tok(tok, p["chat"])
        p["context_tokens"] = n_tok(tok, p["context"])
        p["covers_fix"] = p["target"] is not None  # an oracle edit inside the region exists (verified by
        # run_patch_model.py --oracle); False = none constructed, see failing_calls_not_reaching_region
        p["target_tokens"] = n_tok(tok, p["target"]) if p["target"] is not None else None
        # generation budget: 1.5x the oracle answer (+64), or the function size when the region misses
        ref_len = p["target_tokens"] if p["target_tokens"] is not None else n_tok(tok, fixed_func)
        p["max_new_tokens"] = int(ref_len * 1.5) + 64
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "prompts.json"))
    args = ap.parse_args()
    tok = tokenizer()
    source = F.load_generated()
    prompts, locs = [], {}
    for name in F.FAILING:
        fc = L.failing_calls(source, name, seed=2)  # prompt cases: seed 2; final judge uses seeds 0,1
        loc, votes = L.vote_address(source, name, fc["fail"])
        locs[name] = {"votes": votes, "site_line": loc["site_line"], "site_how": loc["site_how"],
                      "addresses": loc["addresses"], "n_fail": len(fc["fail"]), "n_pass": len(fc["pass"])}
        # each prompt is judged against the module with ONLY this function fixed by the model
        prompts += build(tok, source, name, fc["fail"], loc, fixed_function_only(source, name))
    with open(args.out, "w") as fh:
        json.dump({"model": MODEL, "source_path": F.GEN_PATH, "source": source, "locate": locs,
                   "prompts": prompts}, fh, indent=1)
    for p in prompts:
        print(f"{p['name']:10s} {p['scheme']:12s} prompt {p['prompt_tokens']:5d}  context {p['context_tokens']:5d}  "
              f"target {p['target_tokens']!s:>5}  covers_fix {p['covers_fix']}")
    print("wrote", args.out)


def fixed_function_only(source, name):
    return F.apply_family(source, name, "minimal")


if __name__ == "__main__":
    main()
