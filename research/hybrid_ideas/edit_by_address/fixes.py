"""Correct versions of the 4 failing generated functions, expressed as addressed patches.

Two fix families per function:
  minimal  the smallest hand edit of the generated code that makes it correct
  ref      the generated body (after the docstring) replaced by calc_ref's body
Both are applied through addressing.apply_patch, so they also exercise the patch code.
"""

import ast
import os
import sys
import textwrap

import addressing as A

HERE = os.path.dirname(os.path.abspath(__file__))
CALC_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "codegen_calc"))
GEN_PATH = "/media/llm-data/universe-runs/codegen-calc-20261021/calc_generated.py"
FAILING = ["nth_root", "factorial", "gcd", "evaluate"]

if CALC_DIR not in sys.path:
    sys.path.insert(0, CALC_DIR)


def load_generated():
    with open(GEN_PATH) as fh:
        return fh.read()


def _ref_source():
    with open(os.path.join(CALC_DIR, "calc_ref.py")) as fh:
        return fh.read()


def ref_body(name):
    """calc_ref body of `name` (docstring dropped), dedented. evaluate gets its helpers nested."""
    src = _ref_source()
    tree = ast.parse(src)
    fmap = A.functions(tree)
    _, node = fmap[name]
    body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
        else node.body
    lines = src.splitlines()
    text = textwrap.dedent("\n".join(lines[body[0].lineno - 1:body[-1].end_lineno]))
    if name == "factorial" or name == "nth_root" or name == "gcd":
        return text
    if name == "evaluate":
        consts = [ast.get_source_segment(src, n) for n in tree.body
                  if isinstance(n, ast.Assign) and n.targets[0].id in ("_FUNCS", "_CONSTS")]
        _, tok = fmap["_tokenize"]
        tok_src = "\n".join(lines[tok.lineno - 1:tok.end_lineno])
        return "\n".join(consts) + "\n\n" + tok_src + "\n\n" + text
    raise KeyError(name)


def _body_after_doc(tree, name):
    _, node = A.functions(tree)[name]
    return 1, len(node.body)


def fixes(source, tree=None):
    """name -> family -> list of (address, new_text), applied in order (addresses refer to the
    source as it is when that patch is applied; each list here has a single patch)."""
    tree = tree or ast.parse(source)
    fm = A.functions(tree)
    out = {}
    for name in FAILING:
        i, j = _body_after_doc(tree, name)
        out[name] = {"ref": [({"kind": "symbol", "name": name, "stmt": [i, j]}, ref_body(name))]}
    gi = fm["gcd"][0]
    out["gcd"]["minimal"] = [({"kind": "path", "path": f"body[{gi}].body[1].body[0]"}, "return abs(a)")]
    out["nth_root"]["minimal"] = [({"kind": "symbol", "name": "nth_root", "stmt": [3, 5]},
                                   "root = math.pow(abs(x), 1 / n)\nif x < 0:\n    return -root\nelse:\n    return root")]
    out["factorial"]["minimal"] = [({"kind": "symbol", "name": "factorial", "stmt": [1, 1]},
                                    "if isinstance(n, float) and n.is_integer():\n    n = int(n)")]
    out["evaluate"]["minimal"] = out["evaluate"]["ref"]  # the RPN-on-split() parser cannot be salvaged
    return out


# Region-specific alternative fixes: when the located region does not contain the minimal fix,
# a different edit confined to that region can still be correct. key: (function, scheme)
def alt_target(name, scheme):
    if name == "nth_root" and scheme in ("span_stmt", "span_deep"):
        return "if x < 0:\n    return -math.pow(-x, 1 / n)\nroot = math.pow(x, 1 / n)"
    if name == "factorial" and scheme == "span_deep":
        return ("if isinstance(n, float) and n.is_integer() and n >= 0:\n    n = int(n)\n"
                "else:\n    raise ValueError('Input must be a non-negative integer.')")
    if name == "evaluate" and scheme in ("span_stmt", "span_suffix"):
        return ref_body("evaluate")  # replaces the token loop; the old `return stack[0]` becomes dead code
    return None


def apply_family(source, name, family):
    tree = ast.parse(source)
    for addr, text in fixes(source, tree)[name][family]:
        source, tree = A.apply_patch(source, addr, text, tree)
    return source
