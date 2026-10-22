"""Python-function corpora for the byte-LM domain test (local sources only, no download).

Three arms built from the SAME set of functions, so only the formatting differs:
  raw   - the function's source text as written (comments, docstrings, original layout);
  clean - ast.unparse of the function with docstrings removed (no comments, canonical layout);
  small - clean, restricted to self-contained functions (module level, every free name is a builtin)
          and at most 600 bytes: the narrowest domain.
Functions are deduplicated on their clean form and split by source FILE into train/holdout, so no holdout
function's file is ever seen in training. Each corpus is functions joined by a blank line.

Usage: build_corpus.py --out /media/llm-data/datasets/pycode [--roots DIR ...]
"""

import argparse
import ast
import builtins
import hashlib
import json
import os
import random
import symtable
import warnings

ROOTS = ["/usr/lib/python3.12", "/home/leonardo/projects/LLM/.venv-rocm/lib/python3.12/site-packages"]
BUILTINS = set(dir(builtins))


def strip_docstrings(fn):
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            b0 = node.body[0]
            if isinstance(b0, ast.Expr) and isinstance(b0.value, ast.Constant) and isinstance(b0.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return fn


def free_names(src):
    """Global names a function reads (symtable), or None if it cannot be analysed."""
    try:
        top = symtable.symtable(src, "<f>", "exec")
    except (SyntaxError, ValueError):
        return None
    out, stack = set(), list(top.get_children())
    while stack:
        t = stack.pop()
        out |= {s.get_name() for s in t.get_symbols() if s.is_global() and s.is_referenced()}
        stack.extend(t.get_children())
    return out


def functions(path):
    try:
        text = open(path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        return
    if not text.isascii():
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return
    lines = text.splitlines()
    for parent in [tree] + [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in parent.body:
            if not isinstance(fn, ast.FunctionDef) or fn.decorator_list:
                continue
            raw = lines[fn.lineno - 1:fn.end_lineno]
            ind = len(raw[0]) - len(raw[0].lstrip())
            raw = "\n".join(l[ind:] for l in raw)
            try:
                clean = ast.unparse(strip_docstrings(ast.parse(raw).body[0]))
            except (SyntaxError, IndentationError, ValueError, RecursionError):
                continue
            yield raw, clean, parent is tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="*", default=ROOTS)
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--max-bytes", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seen, rows = set(), []
    for root in args.roots:
        for d, _, fs in os.walk(root):
            if "/test" in d or "__pycache__" in d:
                continue
            for f in sorted(fs):
                if not f.endswith(".py") or f.startswith("test_"):
                    continue
                p = os.path.join(d, f)
                for raw, clean, top in functions(p):
                    if not (40 <= len(clean) <= args.max_bytes) or clean in seen:
                        continue
                    seen.add(clean)
                    fn = free_names(clean)
                    rows.append({"file": p, "raw": raw, "clean": clean,
                                 "self_contained": top and fn is not None and fn <= BUILTINS})
    random.Random(0).shuffle(rows)
    hold = lambda r: int(hashlib.sha1(r["file"].encode()).hexdigest(), 16) % 1000 < args.holdout * 1000  # noqa: E731
    stats = {}
    for arm, pick in (("raw", lambda r: r["raw"]), ("clean", lambda r: r["clean"]),
                      ("small", lambda r: r["clean"] if r["self_contained"] and len(r["clean"]) <= 600 else None)):
        for split in ("train", "holdout"):
            body = [pick(r) for r in rows if hold(r) == (split == "holdout")]
            body = [b for b in body if b]
            data = ("\n\n".join(body) + "\n\n").encode()
            with open(os.path.join(args.out, f"{arm}_{split}.txt"), "wb") as fh:
                fh.write(data)
            stats[f"{arm}_{split}"] = {"functions": len(body), "bytes": len(data)}
    with open(os.path.join(args.out, "holdout_functions.json"), "w") as fh:
        json.dump([{"file": r["file"], "clean": r["clean"], "self_contained": r["self_contained"]}
                   for r in rows if hold(r)], fh)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
