"""Extract per-function symbol event streams from Python source files.

For every function: the token stream (as category ids) plus, at each token, which
local symbol (parameter or locally assigned name) occurs there and whether it is a
load or a store. A query is a load of a local symbol at a point where at least two
local symbols already exist; the task is to say which of them is used.

Usage: extract.py OUT.pt ROOT[:LABEL] [ROOT[:LABEL] ...]
Files are split by path hash: label 'stdlib' -> train (80%) / val (20%); any
other label -> 'out' (a different code base, never trained on).
"""

import ast
import hashlib
import io
import keyword
import sys
import tokenize
from pathlib import Path

import torch

MIN_T, MAX_T, MAX_S = 24, 384, 24
SKIP_DIRS = {"test", "tests", "idlelib", "lib2to3", "site-packages", "__pycache__", "turtledemo"}

KINDS = ["PAD", "LOCAL", "NAME", "NUMBER", "STRING", "NEWLINE", "INDENT", "DEDENT", "OP?", "OTHER"]
OPS = ["(", ")", "[", "]", "{", "}", ":", ",", ".", "=", "+=", "-=", "*=", "+", "-", "*", "/", "//",
       "%", "**", "==", "!=", "<", ">", "<=", ">=", "->", "@", "|", "&", "^", "~", ";", "<<", ">>",
       ":=", "|=", "&=", "/=", "**="]
VOCAB = KINDS + ["kw:" + k for k in keyword.kwlist] + ["op:" + o for o in OPS]
CAT = {v: i for i, v in enumerate(VOCAB)}


def locals_of(fn):
    """Parameters plus every name stored inside fn (excluding nested defs/classes)."""
    params = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    if fn.args.vararg:
        params.append(fn.args.vararg.arg)
    if fn.args.kwarg:
        params.append(fn.args.kwarg.arg)
    stores = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores.append(node.id)
    names = list(dict.fromkeys(params + stores))
    return params, names


def has_nested_scope(fn):
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
               for n in ast.walk(fn) if n is not fn)


def store_positions(fn):
    pos = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            pos.add((node.lineno, node.col_offset))
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            pos.add((node.target.lineno, node.target.col_offset))
    for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs + [fn.args.vararg, fn.args.kwarg]:
        if a is not None:
            pos.add((a.lineno, a.col_offset))
    return pos


def file_functions(src):
    tree = ast.parse(src)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or has_nested_scope(fn):
            continue
        params, names = locals_of(fn)
        if not 2 <= len(names) <= MAX_S:
            continue
        sym = {n: i for i, n in enumerate(names)}
        stores = store_positions(fn)
        start, end = (fn.lineno, fn.col_offset), (fn.end_lineno, fn.end_col_offset)
        cats, sids, isst = [], [], []
        for t in toks:
            if t.start < start or t.start >= end:
                continue
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            sid, st = -1, 0
            if t.type == tokenize.NAME and keyword.iskeyword(t.string):
                c = CAT["kw:" + t.string]
            elif t.type == tokenize.NAME and t.string in sym:
                c, sid = CAT["LOCAL"], sym[t.string]
                st = 1 if t.start in stores else 0
            elif t.type == tokenize.NAME:
                c = CAT["NAME"]
            elif t.type == tokenize.NUMBER:
                c = CAT["NUMBER"]
            elif t.type in (tokenize.STRING, tokenize.FSTRING_START, tokenize.FSTRING_MIDDLE, tokenize.FSTRING_END):
                c = CAT["STRING"]
            elif t.type == tokenize.NEWLINE:
                c = CAT["NEWLINE"]
            elif t.type == tokenize.INDENT:
                c = CAT["INDENT"]
            elif t.type == tokenize.DEDENT:
                c = CAT["DEDENT"]
            elif t.type == tokenize.OP:
                c = CAT.get("op:" + t.string, CAT["OP?"])
            else:
                c = CAT["OTHER"]
            cats.append(c)
            sids.append(sid)
            isst.append(st)
        if not MIN_T <= len(cats) <= MAX_T:
            continue
        is_param = [1 if n in params else 0 for n in names]
        out.append((cats, sids, isst, is_param, names))
    return out


def main():
    out_path = sys.argv[1]
    splits = {"train": [], "val": [], "out": []}
    nfiles = 0
    for spec in sys.argv[2:]:
        root, _, label = spec.partition(":")
        label = label or "stdlib"
        for p in sorted(Path(root).rglob("*.py")):
            if SKIP_DIRS & set(p.relative_to(root).parts):
                continue
            try:
                fns = file_functions(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError, tokenize.TokenError, ValueError, RecursionError):
                continue
            nfiles += 1
            if label == "stdlib":
                h = int(hashlib.sha1(str(p.relative_to(root)).encode()).hexdigest(), 16) % 5
                split = "val" if h == 0 else "train"
            else:
                split = "out"
            splits[split] += fns
    data = {"vocab": VOCAB, "max_t": MAX_T, "max_s": MAX_S}
    for k, fns in splits.items():
        n = len(fns)
        cats = torch.zeros((n, MAX_T), dtype=torch.int16)
        sids = torch.full((n, MAX_T), -1, dtype=torch.int8)
        isst = torch.zeros((n, MAX_T), dtype=torch.int8)
        isparam = torch.zeros((n, MAX_S), dtype=torch.int8)
        lens = torch.zeros(n, dtype=torch.int16)
        names = []
        for i, (c, s, st, ip, nm) in enumerate(fns):
            L = len(c)
            cats[i, :L] = torch.tensor(c, dtype=torch.int16)
            sids[i, :L] = torch.tensor(s, dtype=torch.int8)
            isst[i, :L] = torch.tensor(st, dtype=torch.int8)
            isparam[i, :len(ip)] = torch.tensor(ip, dtype=torch.int8)
            lens[i] = L
            names.append(nm)
        data[k] = {"cats": cats, "sids": sids, "store": isst, "is_param": isparam, "len": lens, "names": names}
        print(f"{k}: {n} functions")
    print(f"files parsed: {nfiles}")
    torch.save(data, out_path)


if __name__ == "__main__":
    main()
