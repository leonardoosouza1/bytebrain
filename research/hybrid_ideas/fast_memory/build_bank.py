"""Build a flat symbol/context bank from Python source files.

Layout (struct-of-arrays, all ids dense u32 starting at 0):
  names      : interned identifier strings  (name_id -> str), pool + offsets
  scopes     : file_id u8, kind u8, start_line, end_line, parent_scope
  symbols    : name_id, scope_id, kind u8, def_line, live_lo, live_hi, sig_off, val_off
  postings   : CSR, uses[post_start[s]:post_start[s+1]] = packed addr32
               addr32 = file:4 | line:16 | col:12
  line index : CSR per (file, line) -> live symbol ids (precomputed interval query)
  local nibble streams: per function, the token-level local-symbol stream with
               each local id packed in 4 bits (two per byte), 0xF = "not a local"

Usage: build_bank.py OUTDIR FILE [FILE ...]
Writes OUTDIR/bank.bin (little-endian sections, see dump()) and OUTDIR/bank_meta.json.
"""

import ast
import builtins
import io
import json
import struct
import sys
import tokenize
from pathlib import Path

KIND = {"module": 0, "function": 1, "class": 2, "param": 3, "local": 4, "global": 5,
        "import": 6, "builtin": 7}
NIBBLE_NONE = 0xF


def pack_addr(file_id, line, col):
    assert file_id < 16 and line < 65536
    return (file_id << 28) | (line << 12) | min(col, 4095)


class Bank:
    def __init__(self):
        self.names, self.name_id = [], {}
        self.scopes = []          # dict(file, kind, lo, hi, parent, name)
        self.syms = []            # dict(name_id, scope, kind, def_line, sig, val, uses[])
        self.sym_key = {}         # (scope, name_id) -> sym id
        self.files = []
        self.nibble_streams = []  # (n_locals, [local id or None per token])

    def intern(self, s):
        i = self.name_id.get(s)
        if i is None:
            i = self.name_id[s] = len(self.names)
            self.names.append(s)
        return i

    def add_sym(self, scope, name, kind, line, sig="", val=""):
        key = (scope, self.intern(name))
        if key in self.sym_key:
            return self.sym_key[key]
        sid = len(self.syms)
        self.sym_key[key] = sid
        self.syms.append(dict(name_id=key[1], scope=scope, kind=KIND[kind], def_line=line,
                              sig=sig, val=val, uses=[]))
        return sid


def signature(fn):
    s = f"{fn.name}({ast.unparse(fn.args)})"
    if fn.returns is not None:
        s += " -> " + ast.unparse(fn.returns)
    return s


def add_file(bank, path):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fid = len(bank.files)
    nlines = src.count("\n") + 1
    bank.files.append(str(path))
    builtin_scope = None

    def new_scope(kind, lo, hi, parent, name):
        bank.scopes.append(dict(file=fid, kind=KIND[kind], lo=lo, hi=hi, parent=parent, name=name))
        return len(bank.scopes) - 1

    mod = new_scope("module", 1, nlines, -1, Path(path).stem)
    scope_of_node = {tree: mod}

    # pass 1: declare scopes and definitions
    def declare(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
                bank.add_sym(scope, child.name, kind, child.lineno, sig=signature(child))
                sc = new_scope("function", child.lineno, child.end_lineno, scope, child.name)
                scope_of_node[child] = sc
                a = child.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
                    if arg is not None:
                        bank.add_sym(sc, arg.arg, "param", arg.lineno)
                declare(child, sc)
            elif isinstance(child, ast.ClassDef):
                bank.add_sym(scope, child.name, "class", child.lineno,
                             sig=f"class {child.name}({', '.join(ast.unparse(b) for b in child.bases)})")
                sc = new_scope("class", child.lineno, child.end_lineno, scope, child.name)
                scope_of_node[child] = sc
                declare(child, sc)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for al in child.names:
                    nm = (al.asname or al.name).split(".")[0]
                    bank.add_sym(scope, nm, "import", child.lineno, sig=ast.unparse(child))
            else:
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    kind = "global" if bank.scopes[scope]["kind"] == KIND["module"] else "local"
                    val = ""
                    bank.add_sym(scope, child.id, kind, child.lineno, val=val)
                if isinstance(child, ast.Assign) and isinstance(child.value, ast.Constant) \
                        and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    kind = "global" if bank.scopes[scope]["kind"] == KIND["module"] else "local"
                    sid = bank.add_sym(scope, child.targets[0].id, kind, child.lineno)
                    if not bank.syms[sid]["val"]:
                        bank.syms[sid]["val"] = repr(child.value.value)[:64]
                declare(child, scope)

    declare(tree, mod)

    # pass 2: resolve every Name occurrence to the innermost scope that defines it
    def resolve(scope, name_id):
        s = scope
        while s != -1:
            # class scopes are not visible from nested functions (Python rule), keep it simple:
            sid = bank.sym_key.get((s, name_id))
            if sid is not None and (s == scope or bank.scopes[s]["kind"] != KIND["class"]):
                return sid
            s = bank.scopes[s]["parent"]
        return None

    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            sc = scope_of_node.get(child, scope)
            if isinstance(child, ast.Name):
                nid = bank.intern(child.id)
                sid = resolve(scope, nid)
                if sid is None:
                    nonlocal builtin_scope
                    if builtin_scope is None:
                        builtin_scope = new_scope("module", 0, 0, -1, "<builtins/unresolved>")
                    kind = "builtin" if hasattr(builtins, child.id) else "global"
                    sid = bank.add_sym(builtin_scope, child.id, kind, 0)
                bank.syms[sid]["uses"].append(pack_addr(fid, child.lineno, child.col_offset))
            walk(child, sc)

    walk(tree, mod)

    # nibble streams: per function, token-level local-symbol stream (for the 4-bit test)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for node, sc in scope_of_node.items():
        if bank.scopes[sc]["kind"] != KIND["function"]:
            continue
        local_ids = [sid for (s, _), sid in bank.sym_key.items() if s == sc]
        name_to_local = {bank.names[bank.syms[sid]["name_id"]]: i for i, sid in enumerate(local_ids)}
        stream = []
        for t in toks:
            if t.start[0] < node.lineno or t.start[0] > node.end_lineno:
                continue
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            stream.append(name_to_local.get(t.string) if t.type == tokenize.NAME else None)
        bank.nibble_streams.append((len(local_ids), stream))
    return fid


def finalize(bank):
    """Compute live intervals and the per-line index."""
    for s in bank.syms:
        s["uses"].sort()
        lines = [(u >> 12) & 0xFFFF for u in s["uses"]]
        sc = bank.scopes[s["scope"]]
        if s["kind"] in (KIND["param"], KIND["local"]):
            lo = min([s["def_line"]] + lines)
            hi = max([s["def_line"]] + lines)
        elif sc["hi"] == 0:          # builtins / unresolved: no source range
            lo, hi = 0, 0
        else:                        # module/class level: visible from definition to scope end
            lo, hi = s["def_line"], sc["hi"]
        s["live_lo"], s["live_hi"] = lo, hi


def dump(bank, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # string pools
    def pool(strings):
        blob, offs = bytearray(), [0]
        for s in strings:
            blob += s.encode()
            offs.append(len(blob))
        return bytes(blob), offs

    name_blob, name_offs = pool(bank.names)
    sig_strings = [s["sig"] for s in bank.syms]
    sig_blob, sig_offs = pool(sig_strings)
    n = len(bank.syms)
    post_start, posts = [0], []
    for s in bank.syms:
        posts += s["uses"]
        post_start.append(len(posts))
    file_of = [bank.scopes[s["scope"]]["file"] for s in bank.syms]
    # nibble streams: concatenate; record (offset, n_tokens) per function; only functions
    # with <= 15 locals are packable (0xF is reserved for "not a local")
    nib_bytes, nib_index, overflow = bytearray(), [], 0
    for nloc, stream in bank.nibble_streams:
        if nloc > 15:
            overflow += 1
            continue
        vals = [NIBBLE_NONE if v is None else v for v in stream]
        if len(vals) % 2:
            vals.append(NIBBLE_NONE)
        off = len(nib_bytes)
        for i in range(0, len(vals), 2):
            nib_bytes.append(vals[i] | (vals[i + 1] << 4))
        nib_index.append((off, len(stream)))

    sections = []
    def u32(arr): return struct.pack(f"<{len(arr)}I", *arr)
    def u16(arr): return struct.pack(f"<{len(arr)}H", *arr)
    def u8(arr): return bytes(arr)
    sections.append(("name_offs", u32(name_offs)))
    sections.append(("name_blob", name_blob))
    sections.append(("sym_name", u32([s["name_id"] for s in bank.syms])))
    sections.append(("sym_scope", u32([s["scope"] for s in bank.syms])))
    sections.append(("sym_kind", u8([s["kind"] for s in bank.syms])))
    sections.append(("sym_file", u8(file_of)))
    sections.append(("sym_def", u32([s["def_line"] for s in bank.syms])))
    sections.append(("sym_lo", u32([s["live_lo"] for s in bank.syms])))
    sections.append(("sym_hi", u32([s["live_hi"] for s in bank.syms])))
    sections.append(("sig_offs", u32(sig_offs)))
    sections.append(("sig_blob", sig_blob))
    sections.append(("post_start", u32(post_start)))
    sections.append(("posts", u32(posts)))
    by_scope = [[] for _ in bank.scopes]
    for i, sym in enumerate(bank.syms):
        by_scope[sym["scope"]].append(i)
    ss_start, ss = [0], []
    for lst in by_scope:
        ss += lst
        ss_start.append(len(ss))
    sections.append(("scope_file", u8([sc["file"] for sc in bank.scopes])))
    sections.append(("scope_kind", u8([sc["kind"] for sc in bank.scopes])))
    sections.append(("scope_lo", u32([sc["lo"] for sc in bank.scopes])))
    sections.append(("scope_hi", u32([sc["hi"] for sc in bank.scopes])))
    sections.append(("scope_parent", u32([sc["parent"] & 0xFFFFFFFF for sc in bank.scopes])))
    sections.append(("scope_sym_start", u32(ss_start)))
    sections.append(("scope_syms", u32(ss)))
    sections.append(("nib_index", u32([x for p in nib_index for x in p])))
    sections.append(("nib_bytes", bytes(nib_bytes)))
    with open(outdir / "bank.bin", "wb") as f:
        f.write(struct.pack("<I", len(sections)))
        for name, data in sections:
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<Q", len(data)) + data)
    meta = dict(files=bank.files, n_symbols=n, n_names=len(bank.names), n_scopes=len(bank.scopes),
                n_postings=len(posts), n_functions=len(bank.nibble_streams),
                nibble_overflow_functions=overflow,
                section_bytes={k: len(v) for k, v in sections})
    (outdir / "bank_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def build(files):
    bank = Bank()
    for p in files:
        add_file(bank, p)
    finalize(bank)
    return bank


if __name__ == "__main__":
    b = build(sys.argv[2:])
    print(json.dumps(dump(b, sys.argv[1]), indent=1))
