"""Edit-by-address primitives: three address schemes, resolution to a source span, patch, locate.

Address schemes (all resolve to the same thing: a [start, end) slice of the source text):
  span    {"kind": "span", "start": [line, col], "end": [line, col]}
          line 1-based, col 0-based in characters ("x, y" on the text grid)
  path    {"kind": "path", "path": "body[38].body[2:4]"}
          AST path from the module; last step may be a slice [i:j] (j == i means insertion before i)
  symbol  {"kind": "symbol", "name": "gcd", "stmt": [i, j]}
          top-level function by name + statement range [i, j) of its body; "stmt": None = whole def

Statement-level addresses (path/symbol) re-indent the new text to the target block's indentation,
so the model can answer with dedented code. A span address inserts the text verbatim.
"""

import ast
import re
import textwrap


PATH_RE = re.compile(r"\w+\[\d+(?::\d+)?\](?:\.\w+\[\d+(?::\d+)?\])*")


class PatchRejected(Exception):
    pass


def line_starts(source):
    starts, pos = [0], 0
    for line in source.splitlines(keepends=True):
        pos += len(line)
        starts.append(pos)
    return starts


def _offset(source, starts, line, col_bytes):
    """ast col offsets are UTF-8 byte offsets; convert to a character offset."""
    line_text = source[starts[line - 1]:starts[line] if line < len(starts) else len(source)]
    return starts[line - 1] + len(line_text.encode("utf-8")[:col_bytes].decode("utf-8", "ignore"))


def functions(tree):
    return {n.name: (i, n) for i, n in enumerate(tree.body) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _walk_path(tree, path):
    """'body[38].body[2:4]' -> (parent_list, i, j, parent_node)."""
    if not PATH_RE.fullmatch(path):
        raise PatchRejected(f"bad path {path!r}")
    steps = re.findall(r"(\w+)\[(\d+)(?::(\d+))?\]", path)
    node = tree
    for k, (field, a, b) in enumerate(steps):
        seq = getattr(node, field, None)
        if not isinstance(seq, list):
            raise PatchRejected(f"no list field {field} at step {k}")
        i = int(a)
        j = int(b) if b else i + 1
        if not (0 <= i <= j <= len(seq)) or (k < len(steps) - 1 and (b or i >= len(seq))):
            raise PatchRejected(f"index out of range at step {k}")
        if k == len(steps) - 1:
            return seq, i, j, node
        node = seq[i]
    raise PatchRejected("unreachable")


def _stmt_range_offsets(source, starts, seq, i, j, parent):
    """Char offsets covering whole lines of seq[i:j]; insertion point if i == j."""
    if i < j:
        first, last = seq[i], seq[j - 1]
        start = starts[first.lineno - 1]
        end = starts[last.end_lineno] if last.end_lineno < len(starts) else len(source)
        indent = first.col_offset
    else:  # insertion before seq[i] (or after the last statement)
        if i < len(seq):
            start = end = starts[seq[i].lineno - 1]
            indent = seq[i].col_offset
        else:
            last = seq[-1]
            start = end = starts[last.end_lineno] if last.end_lineno < len(starts) else len(source)
            indent = last.col_offset
    return start, end, indent


def resolve(source, address, tree=None):
    """-> (start_char, end_char, indent or None). indent None means verbatim (span) insertion."""
    tree = tree or ast.parse(source)
    starts = line_starts(source)
    kind = address["kind"]
    if kind == "span":
        (l1, c1), (l2, c2) = address["start"], address["end"]
        n = len(starts) - 1
        if not (1 <= l1 <= l2 <= n + 1) or (l1 == l2 and c1 > c2):
            raise PatchRejected("span out of range")
        return starts[l1 - 1] + c1, starts[l2 - 1] + c2, None
    if kind == "path":
        seq, i, j, parent = _walk_path(tree, address["path"])
        if not all(isinstance(s, ast.stmt) for s in seq):
            raise PatchRejected("path does not address statements")
        return _stmt_range_offsets(source, starts, seq, i, j, parent)
    if kind == "symbol":
        fn = functions(tree).get(address["name"])
        if fn is None:
            raise PatchRejected(f"no function {address['name']}")
        idx, node = fn
        if address.get("stmt") is None:
            return _stmt_range_offsets(source, starts, tree.body, idx, idx + 1, tree)
        i, j = address["stmt"]
        if not (0 <= i <= j <= len(node.body)):
            raise PatchRejected("stmt range out of range")
        return _stmt_range_offsets(source, starts, node.body, i, j, node)
    raise PatchRejected(f"unknown address kind {kind}")


def _reindent(text, indent):
    body = textwrap.dedent(text.strip("\n"))
    out = textwrap.indent(body, " " * indent, lambda line: line.strip() != "")
    return out + "\n"


def _sig(node):
    return ast.dump(node.args)


def apply_patch(source, address, new_text, tree=None, guard=True):
    """Replace the addressed region with new_text; re-parse; reject if the module breaks.

    guard: the set of top-level function names and their signatures must be unchanged
    (a patch must not delete, rename or re-sign a function the rest of the file relies on).
    Returns (new_source, new_tree).
    """
    tree = tree or ast.parse(source)
    start, end, indent = resolve(source, address, tree)
    text = new_text if indent is None else _reindent(new_text, indent)
    new = source[:start] + text + source[end:]
    try:
        new_tree = ast.parse(new)
    except SyntaxError as e:
        raise PatchRejected(f"syntax: {e.msg} at line {e.lineno}") from None
    if guard:
        before = {k: _sig(n) for k, (_, n) in functions(tree).items()}
        after = {k: _sig(n) for k, (_, n) in functions(new_tree).items()}
        if before.keys() != after.keys():
            raise PatchRejected(f"function set changed: -{sorted(before.keys() - after.keys())} "
                                f"+{sorted(after.keys() - before.keys())}")
        bad = [k for k in before if before[k] != after[k]]
        if bad:
            raise PatchRejected(f"signature changed: {bad}")
    return new, new_tree


def _covers(node, line):
    return node.lineno <= line <= node.end_lineno


def locate(source, line, tree=None):
    """Map a candidate line number (traceback / return site) to addresses in every scheme.

    Returns dict with
      symbol      function name
      stmt        index of the function-body statement containing the line (top level of the def)
      deep_path   AST path of the innermost statement containing the line
      addresses   {"span_line", "span_stmt", "path_stmt", "path_deep", "symbol_stmt", "symbol_suffix",
                   "symbol_func"}
    """
    tree = tree or ast.parse(source)
    for mi, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _covers(node, line):
            break
    else:
        return None
    si = next((k for k, s in enumerate(node.body) if _covers(s, line)), None)
    if si is None:
        return {"symbol": node.name, "stmt": None, "addresses": {
            "symbol_func": {"kind": "symbol", "name": node.name, "stmt": None}}}
    # innermost statement
    path, cur = f"body[{mi}].body[{si}]", node.body[si]
    while True:
        nxt = None
        for field in ("body", "orelse", "finalbody", "handlers"):
            seq = getattr(cur, field, None)
            if isinstance(seq, list):
                for k, s in enumerate(seq):
                    if isinstance(s, (ast.stmt, ast.excepthandler)) and _covers(s, line):
                        nxt = (field, k, s)
                        break
            if nxt:
                break
        if not nxt or isinstance(nxt[2], ast.excepthandler):
            break
        path += f".{nxt[0]}[{nxt[1]}]"
        cur = nxt[2]
    top = node.body[si]
    line_text = source.splitlines()[line - 1]
    indent = len(line_text) - len(line_text.lstrip())
    return {
        "symbol": node.name, "stmt": si, "deep_path": path,
        "deep_lines": [cur.lineno, cur.end_lineno], "stmt_lines": [top.lineno, top.end_lineno],
        "suffix_lines": [top.lineno, node.body[-1].end_lineno],
        "addresses": {
            "span_line": {"kind": "span", "start": [line, indent], "end": [line, len(line_text)]},
            "span_stmt": {"kind": "span", "start": [top.lineno, top.col_offset],
                          "end": [top.end_lineno, top.end_col_offset]},
            "path_stmt": {"kind": "path", "path": f"body[{mi}].body[{si}]"},
            "path_deep": {"kind": "path", "path": path},
            "symbol_stmt": {"kind": "symbol", "name": node.name, "stmt": [si, si + 1]},
            "symbol_suffix": {"kind": "symbol", "name": node.name, "stmt": [si, len(node.body)]},
            "symbol_func": {"kind": "symbol", "name": node.name, "stmt": None},
        },
    }


def function_source(source, name, tree=None):
    tree = tree or ast.parse(source)
    _, node = functions(tree)[name]
    start, end, _ = resolve(source, {"kind": "symbol", "name": name, "stmt": None}, tree)
    return source[start:end], node


def called_signatures(source, name, tree=None):
    """Signature lines of top-level functions that `name` calls (its outgoing context)."""
    tree = tree or ast.parse(source)
    fmap = functions(tree)
    _, node = fmap[name]
    called = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in fmap \
                and n.func.id != name and n.func.id not in called:
            called.append(n.func.id)
    lines = source.splitlines()
    return [lines[fmap[c][1].lineno - 1] for c in called]


def module_imports(tree):
    seen, out = set(), []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            s = ast.unparse(n)
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def apply_patch_fast(source, address, new_text, tree):
    """Same contract as apply_patch, but validates by re-parsing only the enclosing top-level def.

    Sound when the addressed region lies inside one top-level function: the rest of the module is
    byte-identical, and a top-level def is a self-contained parse unit (any unclosed bracket/string
    in the patch makes the def's own text fail to parse). Falls back to apply_patch otherwise.
    On success the passed tree is CONSUMED (later nodes are line-shifted in place) and a new Module
    sharing them is returned; on rejection the tree is untouched. Returns (new_source, new_tree).
    """
    start, end, indent = resolve(source, address, tree)
    starts = line_starts(source)
    line = source.count("\n", 0, start) + 1
    host = next(((k, n) for k, n in enumerate(tree.body)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= n.end_lineno),
                None)
    if host is None:
        return apply_patch(source, address, new_text, tree)
    k, node = host
    first = (node.decorator_list[0].lineno if node.decorator_list else node.lineno)
    d0 = starts[first - 1]
    d1 = starts[node.end_lineno] if node.end_lineno < len(starts) else len(source)
    if not (d0 <= start <= end <= d1):
        return apply_patch(source, address, new_text, tree)
    text = new_text if indent is None else _reindent(new_text, indent)
    new_def = source[d0:start] + text + source[end:d1]
    if new_def and not new_def.endswith("\n"):
        return apply_patch(source, address, new_text, tree)
    try:
        sub = ast.parse(new_def)
    except SyntaxError as e:
        raise PatchRejected(f"syntax: {e.msg} at line {e.lineno + first - 1}") from None
    if len(sub.body) != 1 or not isinstance(sub.body[0], type(node)) or sub.body[0].name != node.name \
            or _sig(sub.body[0]) != _sig(node):
        raise PatchRejected("patch changed the host function's identity or signature")
    new_node = ast.increment_lineno(sub.body[0], first - 1)
    shift = new_def.count("\n") - source[d0:d1].count("\n")
    body = list(tree.body)
    body[k] = new_node
    if shift:
        for n in body[k + 1:]:
            ast.increment_lineno(n, shift)
    return source[:d0] + new_def + source[d1:], ast.Module(body=body, type_ignores=[])
