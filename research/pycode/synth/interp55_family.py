"""Mini-interpreter family (ledger #55 / master queue I6, 10-27): variants + hierarchical spec + reference code.

His words (#55): "quero escrever um projeto pequeno, mas complexo, calculadora cientifica de certa forma é facil, pois
já fez, agora precisamos ir além". Target: a mini-interpreter of arithmetic expressions with variables.

A VARIANT is a feature configuration (11 axes, 18432 combinations):
  ops      '+-*' plus any of '/%^'             prec    std (^ > */% > +-, ^ right-assoc) | flat (one level, left)
  var      plain (x) | dollar ($x)             assign  eq (x = e) | let (let x = e) | walrus (x := e, echoes)
  num      int ('/' floors) | float            err     raise (InterpError) | code (return (value, err) tuples)
  builtins none | abs | abs,min,max            comment none | hash ('#' to end of line)
  pstyle   rd (one function per level) | climb (precedence climbing with a table)
  sstyle   funcs ((toks, i) -> (x, i)) | class (Parser with peek/advance methods)
  estyle   ast (parser builds tuples, evaluate() walks) | direct (parser evaluates with env, no evaluate())
Every variant has the same observable main() semantics for the same (ops, prec, var, assign, num, builtins, comment,
estyle); pstyle/sstyle/err only change the code structure (estyle changes the error ORDER: direct evaluates while
parsing, so an unknown variable can surface before a later syntax error).

The program is written by THIS generator (no LLM): gold is exact and free. Units (8-13 per variant): tokenize,
parse levels (parse_expr/parse_term/parse_power or parse_binary) + parse_unary + parse_atom (functions or Parser
methods + __init__/peek/advance), apply_op, call_builtin (if builtins), evaluate (ast), run_line, main.
Hierarchical spec (ledger #33 idea, main -> children): {"module", "prelude", "main": "main", "tree": {fn: [callees]},
"functions": [{"name", "args", "contract", "calls": [signatures], "examples": [[call, expected], ...]}]}.
Each unit's prompt for a writer = signature + callee signatures + contract (English, generated from the features the
unit depends on). Assert: prompt -> body is a function (one body per prompt across all 18432 variants).

Tests: every unit gets ACCEPT examples (4; in the spec, used for selection) and GRADE cases (16; hidden, disjoint
inputs). Expected values come from running the reference in-process (trusted code written here). Independent check:
an ORACLE interpreter (second implementation, one generic algorithm) must reproduce main() of every reference on its
grade + accept programs -> catches templating bugs in the generator.

Split by COMPOSITION: a variant is held out if it contains any of these feature combinations (each value of every
axis is still seen in train, only the combination is new):
  H1 num=float & var=dollar & comment=hash                 (tokenize body never seen with all three)
  H2 err=code & estyle=direct & sstyle=class               (parser methods)
  H3 assign=let & estyle=direct & pstyle=climb             (run_line)
  H4 num=int & err=code & '%' in ops & '^' in ops          (apply_op)
  H5 builtins=abs & var=dollar & sstyle=funcs              (parse_atom)
Sets: train = every distinct unit prompt of the 11320 non-held variants (the unit space is small: 1226 distinct prompts
over all 18432 variants). func_eval = every prompt that occurs ONLY in held variants (one representative held variant
supplies the sibling context and tests). test_comp = 160 held variants (whole programs, mix of novel + seen units);
test_seen = 120 non-held variants (all units are train prompts: memorisation / assembly control, not a seed split).

Usage:
  interp55_family.py build --out DIR [--seed 0]   -> DIR/data.json (train units, eval variants with tests)
  interp55_family.py selfcheck                    -> oracle vs references on 300 random variants (no files)
"""

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import random
import sys

AXES = {
    "ops": ["+-*", "+-*/", "+-*%", "+-*^", "+-*/%", "+-*/^", "+-*%^", "+-*/%^"],
    "prec": ["std", "flat"],
    "var": ["plain", "dollar"],
    "assign": ["eq", "let", "walrus"],
    "num": ["int", "float"],
    "err": ["raise", "code"],
    "builtins": ["none", "abs", "all"],
    "comment": ["none", "hash"],
    "pstyle": ["rd", "climb"],
    "sstyle": ["funcs", "class"],
    "estyle": ["ast", "direct"],
}
AXIS_NAMES = list(AXES)
BUILTINS = {"none": (), "abs": ("abs",), "all": ("abs", "min", "max")}
PRELUDE = "class InterpError(Exception):\n    pass\n"
HOLD_RULES = {
    "H1": lambda c: c["num"] == "float" and c["var"] == "dollar" and c["comment"] == "hash",
    "H2": lambda c: c["err"] == "code" and c["estyle"] == "direct" and c["sstyle"] == "class",
    "H3": lambda c: c["assign"] == "let" and c["estyle"] == "direct" and c["pstyle"] == "climb",
    "H4": lambda c: c["num"] == "int" and c["err"] == "code" and "%" in c["ops"] and "^" in c["ops"],
    "H5": lambda c: c["builtins"] == "abs" and c["var"] == "dollar" and c["sstyle"] == "funcs",
}


def held_rules(c):
    return [k for k, f in HOLD_RULES.items() if f(c)]


def all_configs():
    for vals in itertools.product(*AXES.values()):
        yield dict(zip(AXIS_NAMES, vals))


def cfg_id(c):
    return "_".join(str(AXES[k].index(c[k])) for k in AXIS_NAMES)


# =================================================================================================== code generation
class Ctx:
    """Statement fragments that depend on (sstyle, estyle, err)."""

    def __init__(self, c):
        self.c = c
        self.cls = c["sstyle"] == "class"
        self.direct = c["estyle"] == "direct"
        self.code = c["err"] == "code"
        self.X = "val" if self.direct else "node"
        self.levels = levels(c)
        self.entry = "parse_binary" if c["pstyle"] == "climb" else "parse_expr"

    def call(self, fn, pos="i", extra=""):
        if self.cls:
            return f"self.{fn}({extra})"
        args = ["toks", pos] + ([extra] if extra else []) + (["env"] if self.direct else [])
        return f"{fn}({', '.join(args)})"

    def entry_call(self, pos="i"):
        return self.call(self.entry, pos, "1" if self.c["pstyle"] == "climb" else "")

    def bind(self, L, d, target, callstr):
        """target, i = call (+ err check)."""
        if self.cls:
            if self.code:
                L.append((d, f"{target}, err = {callstr}"))
                L.append((d, "if err:"))
                L.append((d + 1, "return None, err"))
            else:
                L.append((d, f"{target} = {callstr}"))
        else:
            if self.code:
                L.append((d, f"{target}, i, err = {callstr}"))
                L.append((d, "if err:"))
                L.append((d + 1, "return None, i, err"))
            else:
                L.append((d, f"{target}, i = {callstr}"))

    def fail(self, msg):
        if not self.code:
            return f"raise InterpError('{msg}')"
        return f"return None, '{msg}'" if self.cls else f"return None, i, '{msg}'"

    def ret(self, val, pos="i"):
        if self.cls:
            return f"return {val}, None" if self.code else f"return {val}"
        return f"return {val}, {pos}, None" if self.code else f"return {val}, {pos}"

    def peek(self):
        return "self.peek()" if self.cls else "toks[i]"

    def combine(self, L, d, op, left, right):
        """left = left op right."""
        if not self.direct:
            L.append((d, f"{left} = ('bin', {op}, {left}, {right})"))
        elif self.code:
            L.append((d, f"{left}, err = apply_op({op}, {left}, {right})"))
            L.append((d, "if err:"))
            L.append((d + 1, "return None, err" if self.cls else "return None, i, err"))
        else:
            L.append((d, f"{left} = apply_op({op}, {left}, {right})"))


def levels(c):
    """rd levels: [(name, ops)] from lowest precedence; power level (right-assoc) marked by name parse_power."""
    ops = c["ops"]
    if c["prec"] == "flat":
        return [("parse_expr", ops)]
    out = [("parse_expr", "".join(o for o in "+-" if o in ops)),
           ("parse_term", "".join(o for o in "*/%" if o in ops))]
    if "^" in ops:
        out.append(("parse_power", "^"))
    return out


def prec_table(c):
    p = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3}
    return {o: (1 if c["prec"] == "flat" else p[o]) for o in c["ops"]}


def body_level(x, name, ops, sub):
    L, X = [], x.X
    if x.cls:
        x.bind(L, 0, X, f"self.{sub}()")
        L.append((0, f"while self.peek()[0] == 'op' and self.peek()[1] in '{ops}':"))
        L.append((1, "op = self.advance()[1]"))
        x.bind(L, 1, "right", f"self.{sub}()")
    else:
        x.bind(L, 0, X, x.call(sub))
        L.append((0, f"while toks[i][0] == 'op' and toks[i][1] in '{ops}':"))
        L.append((1, "op = toks[i][1]"))
        x.bind(L, 1, "right", x.call(sub, "i + 1"))
    x.combine(L, 1, "op", X, "right")
    L.append((0, x.ret(X)))
    return L


def body_power(x):
    L = []
    if x.cls:
        x.bind(L, 0, "base", "self.parse_unary()")
        L.append((0, "if self.peek() == ('op', '^'):"))
        L.append((1, "self.advance()"))
        x.bind(L, 1, "exp", "self.parse_power()")
    else:
        x.bind(L, 0, "base", x.call("parse_unary"))
        L.append((0, "if toks[i] == ('op', '^'):"))
        x.bind(L, 1, "exp", x.call("parse_power", "i + 1"))
    x.combine(L, 1, "'^'", "base", "exp")
    L.append((0, x.ret("base")))
    return L


def body_binary(x):
    c, L, X = x.c, [], x.X
    tab = prec_table(c)
    L.append((0, "prec = {" + ", ".join(f"'{o}': {p}" for o, p in tab.items()) + "}"))
    right_pow = c["prec"] == "std" and "^" in c["ops"]
    if x.cls:
        x.bind(L, 0, X, "self.parse_unary()")
        L.append((0, "while self.peek()[0] == 'op' and self.peek()[1] in prec and prec[self.peek()[1]] >= min_prec:"))
        L.append((1, "op = self.advance()[1]"))
    else:
        x.bind(L, 0, X, x.call("parse_unary"))
        L.append((0, "while toks[i][0] == 'op' and toks[i][1] in prec and prec[toks[i][1]] >= min_prec:"))
        L.append((1, "op = toks[i][1]"))
    L.append((1, "nxt = prec[op] if op == '^' else prec[op] + 1" if right_pow else "nxt = prec[op] + 1"))
    if x.cls:
        x.bind(L, 1, "right", "self.parse_binary(nxt)")
    else:
        x.bind(L, 1, "right", x.call("parse_binary", "i + 1", "nxt"))
    x.combine(L, 1, "op", X, "right")
    L.append((0, x.ret(X)))
    return L


def body_unary(x):
    L, X = [], x.X
    neg = f"-{X}" if x.direct else f"('neg', {X})"
    if x.cls:
        L.append((0, "if self.peek() == ('op', '-'):"))
        L.append((1, "self.advance()"))
        x.bind(L, 1, X, "self.parse_unary()")
        L.append((1, x.ret(neg)))
        L.append((0, "return self.parse_atom()"))
    else:
        L.append((0, "if toks[i] == ('op', '-'):"))
        x.bind(L, 1, X, x.call("parse_unary", "i + 1"))
        L.append((1, x.ret(neg)))
        L.append((0, f"return {x.call('parse_atom')}"))
    return L


def body_atom(x):
    c, L = x.c, []
    bl = BUILTINS[c["builtins"]]
    envn = "self.env" if x.cls else "env"
    nxt = "" if x.cls else "i + 1"

    def ret1(v):          # value after consuming one token
        return x.ret(v, "i + 1")

    def var_branch(d):
        if x.direct:
            L.append((d, f"if val not in {envn}:"))
            L.append((d + 1, x.fail("unknown variable")))
            L.append((d, ret1(f"{envn}[val]")))
        else:
            L.append((d, ret1("('var', val)")))
    L.append((0, "kind, val = self.advance()" if x.cls else "kind, val = toks[i]"))
    L.append((0, "if kind == 'num':"))
    L.append((1, ret1("val" if x.direct else "('num', val)")))
    if c["var"] == "dollar":
        L.append((0, "if kind == 'var':"))
        var_branch(1)
    if bl or c["var"] == "plain":
        L.append((0, "if kind == 'id':"))
        if bl:
            if x.cls:
                L.append((1, "if self.peek() == ('op', '('):"))
                L.append((2, "self.advance()"))
                L.append((2, "args = []"))
                L.append((2, "if self.peek() != ('op', ')'):"))
                L.append((3, "while True:"))
                x.bind(L, 4, "arg", x.entry_call())
                L.append((4, "args.append(arg)"))
                L.append((4, "if self.peek() == ('op', ','):"))
                L.append((5, "self.advance()"))
                L.append((4, "else:"))
                L.append((5, "break"))
                L.append((2, "if self.advance() != ('op', ')'):"))
                L.append((3, x.fail("syntax error")))
                if not x.direct:
                    L.append((2, x.ret("('call', val, args)")))
                elif x.code:
                    L.append((2, "return call_builtin(val, args)"))
                else:
                    L.append((2, "return call_builtin(val, args)"))
            else:
                L.append((1, "if toks[i + 1] == ('op', '('):"))
                L.append((2, "i += 2"))
                L.append((2, "args = []"))
                L.append((2, "if toks[i] != ('op', ')'):"))
                L.append((3, "while True:"))
                x.bind(L, 4, "arg", x.entry_call())
                L.append((4, "args.append(arg)"))
                L.append((4, "if toks[i] == ('op', ','):"))
                L.append((5, "i += 1"))
                L.append((4, "else:"))
                L.append((5, "break"))
                L.append((2, "if toks[i] != ('op', ')'):"))
                L.append((3, x.fail("syntax error")))
                if not x.direct:
                    L.append((2, x.ret("('call', val, args)", "i + 1")))
                elif x.code:
                    L.append((2, "res, err = call_builtin(val, args)"))
                    L.append((2, "if err:"))
                    L.append((3, "return None, i, err"))
                    L.append((2, "return res, i + 1, None"))
                else:
                    L.append((2, "return call_builtin(val, args), i + 1"))
        if c["var"] == "plain":
            var_branch(1)
    L.append((0, "if kind == 'op' and val == '(':"))
    if x.cls:
        x.bind(L, 1, x.X, x.entry_call())
        L.append((1, "if self.advance() != ('op', ')'):"))
        L.append((2, x.fail("syntax error")))
        L.append((1, x.ret(x.X)))
    else:
        x.bind(L, 1, x.X, x.entry_call("i + 1"))
        L.append((1, "if toks[i] != ('op', ')'):"))
        L.append((2, x.fail("syntax error")))
        L.append((1, x.ret(x.X, "i + 1")))
    L.append((0, x.fail("syntax error")))
    del nxt
    return L


def body_tokenize(c):
    code = c["err"] == "code"
    fail = "return None, 'bad token'" if code else "raise InterpError('bad token')"
    chars = c["ops"] + "()" + ("," if c["builtins"] != "none" else "")
    L = [(0, "toks = []"), (0, "i = 0"), (0, "while i < len(src):"), (1, "ch = src[i]"),
         (1, "if ch == ' ':"), (2, "i += 1")]
    if c["comment"] == "hash":
        L += [(1, "elif ch == '#':"), (2, "break")]
    L += [(1, "elif ch.isdigit():"), (2, "j = i"), (2, "while j < len(src) and src[j].isdigit():"), (3, "j += 1")]
    if c["num"] == "float":
        L += [(2, "if j < len(src) and src[j] == '.':"), (3, "j += 1"),
              (3, "while j < len(src) and src[j].isdigit():"), (4, "j += 1"),
              (2, "toks.append(('num', float(src[i:j])))")]
    else:
        L += [(2, "toks.append(('num', int(src[i:j])))")]
    L += [(2, "i = j")]
    if c["var"] == "dollar":
        L += [(1, "elif ch == '$':"), (2, "j = i + 1"),
              (2, "while j < len(src) and (src[j].isalnum() or src[j] == '_'):"), (3, "j += 1"),
              (2, "if j == i + 1:"), (3, fail), (2, "toks.append(('var', src[i + 1:j]))"), (2, "i = j")]
    L += [(1, "elif ch.isalpha() or ch == '_':"), (2, "j = i"),
          (2, "while j < len(src) and (src[j].isalnum() or src[j] == '_'):"), (3, "j += 1"),
          (2, "toks.append(('id', src[i:j]))"), (2, "i = j")]
    if c["assign"] == "walrus":
        L += [(1, "elif ch == ':' and src[i + 1:i + 2] == '=':"), (2, "toks.append(('assign', ':='))"), (2, "i += 2")]
    else:
        L += [(1, "elif ch == '=':"), (2, "toks.append(('assign', '='))"), (2, "i += 1")]
    L += [(1, f"elif ch in '{chars}':"), (2, "toks.append(('op', ch))"), (2, "i += 1"),
          (1, "else:"), (2, fail), (0, "toks.append(('end', None))"),
          (0, "return toks, None" if code else "return toks")]
    return L


def body_apply(c):
    code = c["err"] == "code"
    r = (lambda v: f"return {v}, None") if code else (lambda v: f"return {v}")
    z = "return None, 'division by zero'" if code else "raise InterpError('division by zero')"
    L = []
    for o in c["ops"]:
        L.append((0, f"if op == '{o}':"))
        if o in "/%":
            L += [(1, "if b == 0:"), (2, z)]
        expr = {"+": "a + b", "-": "a - b", "*": "a * b", "%": "a % b", "^": "a ** b",
                "/": "a // b" if c["num"] == "int" else "a / b"}[o]
        L.append((1, r(expr)))
    L.append((0, "return None, 'syntax error'" if code else "raise InterpError('syntax error')"))
    return L


def body_builtin(c):
    code = c["err"] == "code"
    r = (lambda v: f"return {v}, None") if code else (lambda v: f"return {v}")
    f = (lambda m: f"return None, '{m}'") if code else (lambda m: f"raise InterpError('{m}')")
    L = []
    for b in BUILTINS[c["builtins"]]:
        L.append((0, f"if name == '{b}':"))
        if b == "abs":
            L += [(1, "if len(args) != 1:"), (2, f("bad arity")), (1, r("abs(args[0])"))]
        else:
            L += [(1, "if not args:"), (2, f("bad arity")), (1, r(f"{b}(args)"))]
    L.append((0, f("unknown function")))
    return L


def body_evaluate(c):
    code = c["err"] == "code"
    has_call = c["builtins"] != "none"
    L = [(0, "kind = node[0]"), (0, "if kind == 'num':"), (1, "return node[1], None" if code else "return node[1]"),
         (0, "if kind == 'var':"), (1, "if node[1] not in env:"),
         (2, "return None, 'unknown variable'" if code else "raise InterpError('unknown variable')"),
         (1, "return env[node[1]], None" if code else "return env[node[1]]"), (0, "if kind == 'neg':")]
    if code:
        L += [(1, "val, err = evaluate(node[1], env)"), (1, "if err:"), (2, "return None, err"), (1, "return -val, None")]
    else:
        L += [(1, "return -evaluate(node[1], env)")]
    if has_call:
        L.append((0, "if kind == 'call':"))
        if code:
            L += [(1, "args = []"), (1, "for a in node[2]:"), (2, "val, err = evaluate(a, env)"), (2, "if err:"),
                  (3, "return None, err"), (2, "args.append(val)")]
        else:
            L += [(1, "args = [evaluate(a, env) for a in node[2]]")]
        L.append((1, "return call_builtin(node[1], args)"))
    if code:
        L += [(0, "left, err = evaluate(node[2], env)"), (0, "if err:"), (1, "return None, err"),
              (0, "right, err = evaluate(node[3], env)"), (0, "if err:"), (1, "return None, err")]
    else:
        L += [(0, "left = evaluate(node[2], env)"), (0, "right = evaluate(node[3], env)")]
    L.append((0, "return apply_op(node[1], left, right)"))
    return L


def body_run_line(c, x):
    code = x.code
    f = "return None, 'syntax error'" if code else "raise InterpError('syntax error')"
    VK = "var" if c["var"] == "dollar" else "id"
    L = []
    if code:
        L += [(0, "toks, err = tokenize(line)"), (0, "if err:"), (1, "return None, err")]
    else:
        L += [(0, "toks = tokenize(line)")]
    if c["comment"] == "hash":
        L += [(0, "if toks[0][0] == 'end':"), (1, "return None, None" if code else "return None")]
    L += [(0, "name = None"), (0, "start = 0")]
    if c["assign"] == "let":
        L += [(0, "if toks[0] == ('id', 'let'):"),
              (1, f"if len(toks) < 4 or toks[1][0] != '{VK}' or toks[2] != ('assign', '='):"), (2, f),
              (1, "name = toks[1][1]"), (1, "start = 3")]
    else:
        a = ":=" if c["assign"] == "walrus" else "="
        L += [(0, f"if toks[0][0] == '{VK}' and toks[1] == ('assign', '{a}'):"), (1, "name = toks[0][1]"),
              (1, "start = 2")]
    tgt = "value" if x.direct else "node"
    ecall = (f"p.{x.entry}(1)" if c["pstyle"] == "climb" else f"p.{x.entry}()")
    if x.cls:
        L += [(0, "p = Parser(toks, env)" if x.direct else "p = Parser(toks)"), (0, "p.i = start")]
        if code:
            L += [(0, f"{tgt}, err = {ecall}"), (0, "if err:"), (1, "return None, err")]
        else:
            L += [(0, f"{tgt} = {ecall}")]
        L += [(0, "if p.peek()[0] != 'end':"), (1, f)]
    else:
        fcall = x.call(x.entry, "start", "1" if c["pstyle"] == "climb" else "")
        if code:
            L += [(0, f"{tgt}, i, err = {fcall}"), (0, "if err:"), (1, "return None, err")]
        else:
            L += [(0, f"{tgt}, i = {fcall}")]
        L += [(0, "if toks[i][0] != 'end':"), (1, f)]
    if not x.direct:
        if code:
            L += [(0, "value, err = evaluate(node, env)"), (0, "if err:"), (1, "return None, err")]
        else:
            L += [(0, "value = evaluate(node, env)")]
    L += [(0, "if name is not None:"), (1, "env[name] = value")]
    if c["assign"] != "walrus":
        L += [(1, "return None, None" if code else "return None")]
    L += [(0, "return str(value), None" if code else "return str(value)")]
    return L


def body_main(c):
    L = [(0, "env = {}"), (0, "out = []"), (0, "for line in lines:"), (1, "if not line.strip():"), (2, "continue")]
    if c["err"] == "code":
        L += [(1, "res, err = run_line(line, env)"), (1, "if err:"), (2, "out.append('error: ' + err)"),
              (1, "elif res is not None:"), (2, "out.append(res)")]
    else:
        L += [(1, "try:"), (2, "res = run_line(line, env)"), (1, "except InterpError as e:"),
              (2, "out.append('error: ' + str(e))"), (2, "continue"), (1, "if res is not None:"), (2, "out.append(res)")]
    L.append((0, "return out"))
    return L


# ------------------------------------------------------------------------------------------------------- contracts
def conv_sentence(x, what):
    c = x.c
    if x.cls:
        s = ("Method of Parser (fields self.toks, self.i" + (", self.env" if x.direct else "") +
             "); read the current token with self.peek() and consume it with self.advance(). ")
        s += (f"Returns ({what}, None) on success or (None, message) on error, never raises. " if x.code
              else f"Returns the {what}; errors raise InterpError(message). ")
    else:
        s = ("Takes the token list and the index i of the current token" + (" and the env dict" if x.direct else "") +
             ". ")
        s += (f"Returns ({what}, i, None) with i after the parsed part, or (None, i, message) on error, never raises. "
              if x.code else f"Returns ({what}, i) with i after the parsed part; errors raise InterpError(message). ")
    if x.direct:
        s += "Evaluates while parsing: numbers are values, binary operators are combined with apply_op. "
    else:
        s += ("Builds AST tuples: ('num', v), ('var', name), ('neg', x), ('bin', op, left, right)"
              + (", ('call', name, args)" if c["builtins"] != "none" else "") + ". ")
    return s


def unit_list(c):
    """[(unit name, args, body lines, contract)] in a fixed order (leaves first is NOT assumed)."""
    x = Ctx(c)
    U = []
    what = "value" if x.direct else "node"
    U.append(("tokenize", "src", body_tokenize(c), tokenize_contract(c)))
    if x.cls:
        U.append(("Parser.__init__", "self, toks, env" if x.direct else "self, toks",
                  [(0, "self.toks = toks"), (0, "self.i = 0")] + ([(0, "self.env = env")] if x.direct else []),
                  "Store the token list in self.toks, set self.i = 0" + (", store env in self.env" if x.direct else "")
                  + "."))
        U.append(("Parser.peek", "self", [(0, "return self.toks[self.i]")], "Return the current token (no move)."))
        U.append(("Parser.advance", "self", [(0, "tok = self.toks[self.i]"), (0, "self.i += 1"), (0, "return tok")],
                  "Return the current token and move self.i one step forward."))
    pre = "Parser." if x.cls else ""
    sig_tail = "" if x.cls else (", env" if x.direct else "")
    selfa = "self" if x.cls else "toks, i"
    if c["pstyle"] == "rd":
        lv = x.levels
        for k, (name, ops) in enumerate(lv):
            if name == "parse_power":
                U.append((pre + name, selfa + sig_tail, body_power(x),
                          conv_sentence(x, what) + "Power level: operand from parse_unary; if the next token is '^' "
                          "consume it and parse the exponent with parse_power (right-associative)."))
            else:
                sub = lv[k + 1][0] if k + 1 < len(lv) else "parse_unary"
                U.append((pre + name, selfa + sig_tail, body_level(x, name, ops, sub),
                          conv_sentence(x, what) + f"Binary level for the operators '{ops}', left-associative: "
                          f"operands from {sub}, loop while the next token is one of them."))
    else:
        tab = prec_table(c)
        rp = c["prec"] == "std" and "^" in c["ops"]
        U.append((pre + "parse_binary", ("self, min_prec" if x.cls else "toks, i, min_prec") + sig_tail,
                  body_binary(x),
                  conv_sentence(x, what) + "Precedence climbing with the local table prec = " +
                  json.dumps(tab).replace('"', "'") + ": operand from parse_unary, then while the next token is an "
                  "operator with precedence >= min_prec consume it and parse the right side with min_prec = its "
                  "precedence + 1" + (" ('^' is right-associative: its own precedence)" if rp else "") + "."))
    U.append((pre + "parse_unary", selfa + sig_tail, body_unary(x),
              conv_sentence(x, what) + "A leading '-' negates the following parse_unary; otherwise parse_atom."))
    bl = BUILTINS[c["builtins"]]
    atom = "Atom: ('num', v) token is a number; "
    if c["var"] == "dollar":
        atom += "('var', name) token is a variable" + ("" if bl else "; ('id', name) is a syntax error") + "; "
    else:
        atom += "('id', name) token is a variable" + ("" if not bl else " unless followed by '('") + "; "
    if bl:
        atom += (f"('id', name) followed by '(' is a call with comma-separated arguments parsed by {x.entry}" +
                 (" (min_prec 1)" if c["pstyle"] == "climb" else "") + " up to ')' (empty list allowed)" +
                 ("; call_builtin(name, args) gives its value" if x.direct else "") + "; ")
        if c["var"] == "dollar":
            atom += "any other ('id', name) is a syntax error; "
    if x.direct:
        atom += "a variable missing from env is 'unknown variable'; "
    atom += (f"'(' expression ')' via {x.entry}" + (" (min_prec 1)" if c["pstyle"] == "climb" else "") +
             "; anything else, or a missing ')', is 'syntax error'.")
    U.append((pre + "parse_atom", selfa + sig_tail, body_atom(x), conv_sentence(x, what) + atom))
    code = x.code
    ret_s = ("Returns (result, None) or (None, message), never raises." if code
             else "Errors raise InterpError(message).")
    U.append(("apply_op", "op, a, b", body_apply(c),
              f"Apply the binary operator op (one of '{c['ops']}') to the numbers a and b. " +
              ("'/' is floor division (//). " if "/" in c["ops"] and c["num"] == "int" else "") +
              ("'/' is true division. " if "/" in c["ops"] and c["num"] == "float" else "") +
              ("'/' or '%' by zero is 'division by zero'. " if ("/" in c["ops"] or "%" in c["ops"]) else "") +
              ("'^' is power (**). " if "^" in c["ops"] else "") + "Any other op is 'syntax error'. " + ret_s))
    if bl:
        U.append(("call_builtin", "name, args", body_builtin(c),
                  "Builtins: abs(x) with exactly one argument" +
                  (", min(...) and max(...) with one or more arguments (args is a list)" if len(bl) > 1 else "") +
                  ". Wrong argument count is 'bad arity'; any other name is 'unknown function'. " + ret_s))
    if not x.direct:
        U.append(("evaluate", "node, env", body_evaluate(c),
                  "Evaluate an AST tuple with variables from the env dict: ('num', v) -> v; ('var', name) -> env "
                  "value or 'unknown variable'; ('neg', x) -> minus x; " +
                  ("('call', name, args) -> evaluate args in order, then call_builtin; " if bl else "") +
                  "('bin', op, left, right) -> evaluate left, then right, then apply_op. " + ret_s))
    U.append(("run_line", "line, env", body_run_line(c, x), run_line_contract(c, x)))
    U.append(("main", "lines", body_main(c),
              "Run each line that is not blank with run_line, sharing one env dict (starts empty). Collect the "
              "outputs that are not None; an error becomes the output 'error: ' + message. Return the list. " +
              ("run_line returns (output, err)." if code else "run_line raises InterpError(message).")))
    return U


def tokenize_contract(c):
    s = "Scan src left to right; return the token list ending with ('end', None). Spaces are skipped. "
    if c["comment"] == "hash":
        s += "'#' starts a comment: stop scanning there. "
    s += ("A run of digits is ('num', int(text)). " if c["num"] == "int" else
          "Digits, then optionally '.' and more digits, is ('num', float(text)). ")
    if c["var"] == "dollar":
        s += "'$' followed by letters, digits or '_' is ('var', name without '$'); a '$' with no name is an error. "
    s += "A letter or '_' followed by letters, digits or '_' is ('id', word). "
    s += ("':=' is ('assign', ':='). " if c["assign"] == "walrus" else "'=' is ('assign', '='). ")
    chars = c["ops"] + "()" + ("," if c["builtins"] != "none" else "")
    s += f"Each character of '{chars}' is ('op', ch). Any other character is the error 'bad token'. "
    s += ("Returns (tokens, None) or (None, 'bad token'), never raises." if c["err"] == "code"
          else "Errors raise InterpError('bad token').")
    return s


def run_line_contract(c, x):
    VK = "var" if c["var"] == "dollar" else "id"
    s = "Run one source line with the env dict. Tokenize it" + (" (returns (tokens, err))" if x.code else "") + ". "
    if c["comment"] == "hash":
        s += "If the first token is ('end', None) the line is empty: output None. "
    if c["assign"] == "let":
        s += (f"If the first token is ('id', 'let') the line must be let, a ('{VK}', name) token, ('assign', '=') "
              "and at least one more token (else 'syntax error'); the expression starts at index 3. ")
    else:
        a = ":=" if c["assign"] == "walrus" else "="
        s += f"If the tokens start with ('{VK}', name), ('assign', '{a}') it is an assignment; the expression starts " \
             "at index 2. "
    if x.cls:
        s += ("Build Parser(toks" + (", env" if x.direct else "") + "), set its i to the start index, call " +
              (f"{x.entry}(1)" if c["pstyle"] == "climb" else f"{x.entry}()") + "; ")
        s += "if p.peek() is not the end token it is 'syntax error'. "
    else:
        s += ("Parse with " + x.call(x.entry, "start", "1" if c["pstyle"] == "climb" else "") +
              "; if toks[i] is not the end token it is 'syntax error'. ")
    if not x.direct:
        s += "Evaluate the node with evaluate(node, env). "
    s += "An assignment stores the value in env and outputs " + \
         ("str(value)" if c["assign"] == "walrus" else "None") + "; an expression outputs str(value). "
    s += ("Returns (output, None) or (None, message), never raises." if x.code
          else "Returns the output; errors raise InterpError(message).")
    return s


def text_of(L):
    return "\n".join("    " * d + t for d, t in L) + "\n"


def assemble(c, bodies):
    """bodies: {unit name: body text (indent 0)} -> module source."""
    out = [PRELUDE]
    cls_done = False
    for name, args, _, _ in unit_list(c):
        b = bodies[name]
        if name.startswith("Parser."):
            if not cls_done:
                out.append("class Parser:")
                cls_done = True
            m = name.split(".", 1)[1]
            out.append(f"    def {m}({args}):\n" + "".join("        " + l + "\n" if l.strip() else "\n"
                                                           for l in b.rstrip("\n").split("\n")))
        else:
            out.append(f"def {name}({args}):\n" + "".join("    " + l + "\n" if l.strip() else "\n"
                                                         for l in b.rstrip("\n").split("\n")))
    return "\n".join(out)


def callees(c):
    """unit -> sibling units it calls (from the reference AST)."""
    U = unit_list(c)
    names = {n for n, *_ in U}
    out = {}
    for name, args, L, _ in U:
        src = f"def f({args}):\n" + "".join("    " + l + "\n" for l in text_of(L).rstrip("\n").split("\n"))
        hit = set()
        for nd in ast.walk(ast.parse(src)):
            if isinstance(nd, ast.Call):
                f = nd.func
                if isinstance(f, ast.Name) and f.id in names:
                    hit.add(f.id)
                elif isinstance(f, ast.Name) and f.id == "Parser":
                    hit.add("Parser.__init__")
                elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in ("self", "p"):
                    if "Parser." + f.attr in names:
                        hit.add("Parser." + f.attr)
        out[name] = sorted(hit - {name}) + ([name] if name in hit else [])
    return out


def prompt_of(c, name, args, contract, calls_sig):
    head = ("class Parser:  def " + name.split(".", 1)[1] if name.startswith("Parser.") else "def " + name)
    return (f"{head}({args}):\ncalls: " + ("; ".join(calls_sig) if calls_sig else "-") + "\ncontract: " + contract +
            "\n")


def short_ret(c, name):
    """one-line return convention of a unit, for callee signatures."""
    code = c["err"] == "code"
    direct = c["estyle"] == "direct"
    cls = c["sstyle"] == "class"
    w = "value" if direct else "node"
    if name == "tokenize":
        return "(tokens, err)" if code else "tokens"
    if name in ("apply_op", "call_builtin", "evaluate"):
        return "(value, err)" if code else "value"
    if name == "run_line":
        return "(output, err)" if code else "output"
    if name == "main":
        return "outputs"
    if name in ("Parser.__init__",):
        return "Parser"
    if name in ("Parser.peek", "Parser.advance"):
        return "token"
    if cls:
        return f"({w}, err)" if code else w
    return f"({w}, i, err)" if code else f"({w}, i)"


def spec_units(c):
    """-> list of dicts {name, args, body, contract, calls, prompt} (reference), tree."""
    U = unit_list(c)
    sig = {n: a for n, a, _, _ in U}
    cal = callees(c)
    out = []
    for name, args, L, contract in U:
        cs = []
        for k in cal[name]:
            nm = ("Parser(" + sig[k].replace("self, ", "") + ")") if k == "Parser.__init__" else \
                 (("self." + k.split(".", 1)[1] + "(" + sig[k].replace("self, ", "").replace("self", "") + ")")
                  if k.startswith("Parser.") else f"{k}({sig[k]})")
            cs.append(f"{nm} -> {short_ret(c, k)}")
        out.append({"name": name, "args": args, "body": text_of(L), "contract": contract, "calls": cal[name],
                    "prompt": prompt_of(c, name, args, contract, cs)})
    return out


# =================================================================================================== test inputs
VARS = ["x", "y", "z", "a", "b", "n", "total", "k2"]


class Reject(Exception):
    pass


def lit(c, r):
    if c["num"] == "int":
        return str(r.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]))
    return r.choice(["0", "1", "2", "3", "5", "10", "2.5", "0.5", "1.25", "4.", "7"])


def vname(c, n):
    return ("$" + n) if c["var"] == "dollar" else n


def gen_expr(c, r, defined, depth=0, undefined_p=0.0):
    k = r.random()
    if depth > 2 or k < 0.3:
        u = r.random()
        if u < 0.5 or not defined:
            if r.random() < undefined_p:
                return vname(c, r.choice([v for v in VARS if v not in defined] or ["q"]))
            return lit(c, r)
        return vname(c, r.choice(sorted(defined)))
    if k < 0.4:
        return "-" + gen_expr(c, r, defined, depth + 1, undefined_p)
    if k < 0.5:
        return "(" + gen_expr(c, r, defined, depth + 1, undefined_p) + ")"
    bl = BUILTINS[c["builtins"]]
    if k < 0.6 and bl:
        f = r.choice(bl)
        n = 1 if f == "abs" else r.choice([1, 2, 3])
        return f + "(" + ", ".join(gen_expr(c, r, defined, depth + 1, undefined_p) for _ in range(n)) + ")"
    n = r.choice([2, 2, 3])
    parts = [gen_expr(c, r, defined, depth + 1, undefined_p)]
    for _ in range(n - 1):
        o = r.choice(c["ops"])
        sp = r.choice(["", " "])
        rhs = str(r.choice([0, 1, 2, 3])) if o == "^" else gen_expr(c, r, defined, depth + 1, undefined_p)
        parts += [sp + o + sp, rhs]
    return "".join(parts)


def gen_line(c, r, defined, kind=None):
    """-> (line, assigned name or None)."""
    kind = kind or r.choices(["expr", "assign", "error"], [0.45, 0.3, 0.25])[0]
    if kind == "assign":
        n = r.choice(VARS)
        e = gen_expr(c, r, defined)
        a = {"eq": f"{vname(c, n)} = {e}", "let": f"let {vname(c, n)} = {e}", "walrus": f"{vname(c, n)} := {e}"}
        line = a[c["assign"]]
        if c["comment"] == "hash" and r.random() < 0.2:
            line += "  # set " + n
        return line, n
    if kind == "expr":
        line = gen_expr(c, r, defined)
        if c["comment"] == "hash" and r.random() < 0.2:
            line += " # note"
        return line, None
    e = gen_expr(c, r, defined)
    bl = BUILTINS[c["builtins"]]
    opts = ["badchar", "syntax_tail", "syntax_open", "syntax_two", "undef", "zero", "comment_char", "badassign"]
    if bl:
        opts += ["unknown_fn", "arity"]
    if c["assign"] == "let":
        opts.append("let_bad")
    w = r.choice(opts)
    if w == "badchar":
        return e + " " + r.choice("@&?!;") + " 1", None
    if w == "syntax_tail":
        return e + " " + r.choice(c["ops"].replace("-", "")), None
    if w == "syntax_open":
        return "(" + e, None
    if w == "syntax_two":
        return e + " " + lit(c, r), None
    if w == "undef":
        return gen_expr(c, r, set(), undefined_p=1.0) + " + " + vname(c, "q"), None
    if w == "zero":
        if "/" in c["ops"] or "%" in c["ops"]:
            o = r.choice([o for o in "/%" if o in c["ops"]])
            return f"{e} {o} (1 - 1)", None
        return e + " ) ", None
    if w == "comment_char":
        return e + " # tail", None
    if w == "badassign":
        return (f"{vname(c, 'x')} = 3" if c["assign"] == "walrus" else f"{vname(c, 'x')} := 3"
                if c["assign"] != "let" else f"{vname(c, 'x')} = 3"), None
    if w == "unknown_fn":
        return r.choice(["foo", "sqrt", "minimum"]) + "(" + lit(c, r) + ")", None
    if w == "arity":
        return r.choice(["abs(1, 2)", "abs()"] + (["max()", "min()"] if len(bl) > 1 else [])), None
    if w == "let_bad":
        return r.choice(["let 3 = 4", "let", f"let {vname(c, 'x')}"]), None
    raise AssertionError(w)


def gen_program(c, r, n_lines):
    defined, lines = set(), []
    for _ in range(n_lines):
        line, n = gen_line(c, r, defined)
        lines.append(line)
        if n:
            defined.add(n)
    return lines


# =================================================================================================== oracle
class OErr(Exception):
    pass


def oracle_main(c, lines):
    """Independent generic interpreter: same semantics, one algorithm for all variants (precedence climbing)."""
    tab = prec_table(c)
    right_pow = c["prec"] == "std"
    direct = c["estyle"] == "direct"
    bl = BUILTINS[c["builtins"]]
    num = int if c["num"] == "int" else float

    def tokenize(s):
        out, i = [], 0
        chars = c["ops"] + "()" + ("," if bl else "")
        while i < len(s):
            ch = s[i]
            if ch == " ":
                i += 1
                continue
            if ch == "#" and c["comment"] == "hash":
                break
            if ch.isdigit():
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                if num is float and j < len(s) and s[j] == ".":
                    j += 1
                    while j < len(s) and s[j].isdigit():
                        j += 1
                out.append(("num", num(s[i:j])))
                i = j
                continue
            if ch == "$" and c["var"] == "dollar":
                j = i + 1
                while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                    j += 1
                if j == i + 1:
                    raise OErr("bad token")
                out.append(("var", s[i + 1:j]))
                i = j
                continue
            if ch.isalpha() or ch == "_":
                j = i
                while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                    j += 1
                out.append(("id", s[i:j]))
                i = j
                continue
            if c["assign"] == "walrus" and s[i:i + 2] == ":=":
                out.append(("assign", ":="))
                i += 2
                continue
            if c["assign"] != "walrus" and ch == "=":
                out.append(("assign", "="))
                i += 1
                continue
            if ch in chars:
                out.append(("op", ch))
                i += 1
                continue
            raise OErr("bad token")
        out.append(("end", None))
        return out

    def apply(op, a, b):
        if op in "/%" and b == 0:
            raise OErr("division by zero")
        if op == "^" and (abs(b) > 6 or abs(a) > 1e6):
            raise Reject()
        v = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "%": lambda: a % b, "^": lambda: a ** b,
             "/": lambda: a // b if num is int else a / b}[op]()
        if isinstance(v, complex) or abs(v) > 1e12:
            raise Reject()
        return v

    def builtin(name, args):
        if name not in bl:
            raise OErr("unknown function")
        if name == "abs":
            if len(args) != 1:
                raise OErr("bad arity")
            return abs(args[0])
        if not args:
            raise OErr("bad arity")
        return min(args) if name == "min" else max(args)

    class P:
        def __init__(s, toks, env):
            s.t, s.i, s.env = toks, 0, env

        def binary(s, mp):
            x = s.unary()
            while s.t[s.i][0] == "op" and s.t[s.i][1] in tab and tab[s.t[s.i][1]] >= mp:
                op = s.t[s.i][1]
                s.i += 1
                y = s.binary(tab[op] if (op == "^" and right_pow) else tab[op] + 1)
                x = apply(op, x, y) if direct else ("bin", op, x, y)
            return x

        def unary(s):
            if s.t[s.i] == ("op", "-"):
                s.i += 1
                v = s.unary()
                return -v if direct else ("neg", v)
            return s.atom()

        def var(s, name):
            if not direct:
                return ("var", name)
            if name not in s.env:
                raise OErr("unknown variable")
            return s.env[name]

        def atom(s):
            k, v = s.t[s.i]
            s.i += 1
            if k == "num":
                return v if direct else ("num", v)
            if k == "var":
                return s.var(v)
            if k == "id":
                if bl and s.t[s.i] == ("op", "("):
                    s.i += 1
                    args = []
                    if s.t[s.i] != ("op", ")"):
                        while True:
                            args.append(s.binary(1))
                            if s.t[s.i] == ("op", ","):
                                s.i += 1
                            else:
                                break
                    if s.t[s.i] != ("op", ")"):
                        raise OErr("syntax error")
                    s.i += 1
                    return builtin(v, args) if direct else ("call", v, args)
                if c["var"] == "plain":
                    return s.var(v)
                raise OErr("syntax error")
            if (k, v) == ("op", "("):
                x = s.binary(1)
                if s.t[s.i] != ("op", ")"):
                    raise OErr("syntax error")
                s.i += 1
                return x
            raise OErr("syntax error")

    def ev(n, env):
        k = n[0]
        if k == "num":
            return n[1]
        if k == "var":
            if n[1] not in env:
                raise OErr("unknown variable")
            return env[n[1]]
        if k == "neg":
            return -ev(n[1], env)
        if k == "call":
            return builtin(n[1], [ev(a, env) for a in n[2]])
        return apply(n[1], ev(n[2], env), ev(n[3], env))

    VK = "var" if c["var"] == "dollar" else "id"
    env, out = {}, []
    for line in lines:
        if not line.strip():
            continue
        try:
            t = tokenize(line)
            if t[0][0] == "end" and c["comment"] == "hash":
                continue
            name, start = None, 0
            if c["assign"] == "let":
                if t[0] == ("id", "let"):
                    if len(t) < 4 or t[1][0] != VK or t[2] != ("assign", "="):
                        raise OErr("syntax error")
                    name, start = t[1][1], 3
            elif t[0][0] == VK and t[1] == ("assign", ":=" if c["assign"] == "walrus" else "="):
                name, start = t[0][1], 2
            p = P(t, env)
            p.i = start
            x = p.binary(1)
            if t[p.i][0] != "end":
                raise OErr("syntax error")
            v = x if direct else ev(x, env)
            if name is not None:
                env[name] = v
                if c["assign"] != "walrus":
                    continue
            out.append(str(v))
        except OErr as e:
            out.append("error: " + str(e))
    return out


# =================================================================================================== canon + run
def canon(v):
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return repr(v)
        v = round(v, 6)
        return repr(0.0 if v == 0 else v)
    if isinstance(v, int):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return ("L[" if isinstance(v, list) else "T[") + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, dict):
        return "D{" + ",".join(canon(k) + ":" + canon(v[k]) for k in sorted(v, key=repr)) + "}"
    return repr(v)


HARNESS = '''
def _e3(r):
    return ('E', r[2]) if r[2] is not None else (r[0], r[1])
def _e2(r):
    return ('E', r[1]) if r[1] is not None else r[0]
def _m(p, name, k=None, *a):
    if k is not None:
        p.i = k
    r = getattr(p, name)(*a)
    return (r, p.i)
def _mc(p, name, k=None, *a):
    if k is not None:
        p.i = k
    r = getattr(p, name)(*a)
    return ('E', r[1]) if r[1] is not None else (r[0], p.i)
def _rl(line, env):
    out = run_line(line, env)
    return (out, env)
'''


def eval_call(g, call):
    """in-process evaluation of a test call in namespace g (TRUSTED reference code only)."""
    try:
        return canon(eval(call, g))
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ == "InterpError":
            return "!InterpError:" + str(e)
        return "!" + type(e).__name__


def ref_namespace(c):
    src = assemble(c, {u["name"]: u["body"] for u in spec_units(c)})
    g = {}
    exec(src + HARNESS, g)
    return g, src


def unit_calls(c, unit, r, n, avoid=()):
    """n distinct test call strings for one unit (inputs drawn from r; strings in avoid are skipped)."""
    code = c["err"] == "code"
    cls = c["sstyle"] == "class"
    direct = c["estyle"] == "direct"
    g, _ = REFCACHE(c)
    bl = BUILTINS[c["builtins"]]
    out, tries = [], 0
    seen = set(avoid)
    while len(out) < n and tries < 400:
        tries += 1
        call = None
        if unit == "main":
            call = f"main({gen_program(c, r, r.choice([3, 4, 5, 6]))!r})"
        elif unit == "run_line":
            prog = gen_program(c, r, r.choice([1, 2, 3]))
            env = {}
            okenv = oracle_env(c, prog[:-1])
            env = okenv if okenv is not None else {}
            line = prog[-1]
            call = f"_rl({line!r}, {env!r})" if not code else f"(lambda e: (_e2(run_line({line!r}, e)), e))({env!r})"
        elif unit == "tokenize":
            line, _ = gen_line(c, r, set(r.sample(VARS, 3)))
            call = f"_e2(tokenize({line!r}))" if code else f"tokenize({line!r})"
        elif unit == "apply_op":
            o = c["ops"][len(out) % len(c["ops"])] if len(out) < len(c["ops"]) else r.choice(c["ops"])
            a, b = lit_val(c, r), lit_val(c, r) if o != "^" else r.choice([0, 1, 2, 3])
            if o in "/%" and r.random() < 0.3:
                b = 0 if c["num"] == "int" else 0.0
            if r.random() < 0.05:
                o = r.choice("&@")
            call = f"apply_op({o!r}, {a!r}, {b!r})"
            call = f"_e2({call})" if code else call
        elif unit == "call_builtin":
            name = r.choice(list(bl) + ["foo", "sqrt"]) if r.random() < 0.85 else r.choice(bl)
            k = r.choice([0, 1, 1, 2, 3])
            if len(out) < len(bl):                     # coverage: every builtin once with a valid arity
                name, k = bl[len(out)], (1 if bl[len(out)] == "abs" else r.choice([1, 2, 3]))
            args = [lit_val(c, r) * r.choice([1, -1]) for _ in range(k)]
            call = f"call_builtin({name!r}, {args!r})"
            call = f"_e2({call})" if code else call
        elif unit == "evaluate":
            defined = set(r.sample(VARS, 3))
            line = gen_expr(c, r, defined, undefined_p=0.1)
            env = {v: lit_val(c, r) for v in defined if r.random() < 0.85}
            node = ref_parse(c, g, line)
            if node is None:
                continue
            call = f"evaluate({node!r}, {env!r})"
            call = f"_e2({call})" if code else call
        elif unit.startswith("Parser.") and unit in ("Parser.__init__", "Parser.peek", "Parser.advance"):
            toks = ref_tokens(c, g, gen_expr(c, r, set(r.sample(VARS, 3))))
            if toks is None:
                continue
            envs = {v: lit_val(c, r) for v in VARS[:3]}
            ctor = f"Parser({toks!r}, {envs!r})" if direct else f"Parser({toks!r})"
            k = r.randrange(len(toks) - 1) if len(toks) > 1 else 0
            if unit == "Parser.__init__":
                call = f"(lambda p: (p.toks, p.i" + (", p.env" if direct else "") + f"))({ctor})"
            elif unit == "Parser.peek":
                call = f"_m({ctor}, 'peek', {k})"
            else:
                call = f"_m({ctor}, 'advance', {k})"
        else:  # parse units
            defined = set(r.sample(VARS, 3))
            line = gen_expr(c, r, defined, undefined_p=0.1) if r.random() < 0.7 else gen_line(c, r, defined,
                                                                                               "error")[0]
            toks = ref_tokens(c, g, line)
            if toks is None:
                continue
            env = {v: lit_val(c, r) for v in defined if r.random() < 0.85}
            m = unit.split(".", 1)[1] if cls else unit
            extra = "1" if m == "parse_binary" else ""
            if m == "parse_binary" and r.random() < 0.3:
                extra = str(r.choice([2, 3]))
            if cls:
                ctor = f"Parser({toks!r}, {env!r})" if direct else f"Parser({toks!r})"
                call = f"{'_mc' if code else '_m'}({ctor}, {m!r}, 0" + (f", {extra}" if extra else "") + ")"
            else:
                args = [repr(toks), "0"] + ([extra] if extra else []) + ([repr(env)] if direct else [])
                call = f"{m}({', '.join(args)})"
                call = f"_e3({call})" if code else call
        if call is None or call in seen:
            continue
        exp = eval_call(g, call)
        if exp.startswith("!") and not exp.startswith("!InterpError"):
            continue                                   # reference crash on an odd input: not a contract case
        seen.add(call)
        out.append([call, exp])
    return out


def lit_val(c, r):
    v = r.choice([0, 1, 2, 3, 4, 5, 7, 10, 12])
    return v if c["num"] == "int" else r.choice([float(v), v + 0.5, v * 0.25])


def ref_tokens(c, g, line):
    try:
        t = g["tokenize"](line)
    except Exception:  # noqa: BLE001
        return None
    if c["err"] == "code":
        t = t[0]
    return t


def ref_parse(c, g, line):
    t = ref_tokens(c, g, line)
    if t is None:
        return None
    try:
        if c["sstyle"] == "class":
            p = g["Parser"](t)
            r = p.parse_binary(1) if c["pstyle"] == "climb" else p.parse_expr()
            if c["err"] == "code":
                r = None if r[1] else r[0]
        else:
            r = g["parse_binary"](t, 0, 1) if c["pstyle"] == "climb" else g["parse_expr"](t, 0)
            if c["err"] == "code":
                r = None if r[2] else r[0]
            else:
                r = r[0]
    except Exception:  # noqa: BLE001
        return None
    return r


def oracle_env(c, lines):
    """env after running lines (via the oracle semantics re-implemented as main with a probe) -> dict or None."""
    g, _ = REFCACHE(c)
    env = {}
    for line in lines:
        try:
            if c["err"] == "code":
                g["run_line"](line, env)
            else:
                try:
                    g["run_line"](line, env)
                except Exception as e:  # noqa: BLE001
                    if type(e).__name__ != "InterpError":
                        return None
        except Exception:  # noqa: BLE001
            return None
    return env


_REF = {}


def REFCACHE(c):
    k = cfg_id(c)
    if k not in _REF:
        if len(_REF) > 64:
            _REF.clear()
        _REF[k] = ref_namespace(c)
    return _REF[k]


def safe_program(c, r, n_lines):
    """program whose oracle run does not Reject (huge powers)."""
    for _ in range(50):
        p = gen_program(c, r, n_lines)
        try:
            oracle_main(c, p)
            return p
        except Reject:
            continue
    return None


# =================================================================================================== build
def unit_tests(c, seed):
    """{unit: (accept[4], grade[16])} with disjoint call strings (different random streams)."""
    out = {}
    for u in spec_units(c):
        ra = random.Random(f"acc|{cfg_id(c)}|{u['name']}|{seed}")
        rg = random.Random(f"grade|{cfg_id(c)}|{u['name']}|{seed}")
        acc = unit_calls(c, u["name"], ra, 4 if u["name"] != "main" else 3)
        grd = unit_calls(c, u["name"], rg, 16 if u["name"] != "main" else 10, avoid={a for a, _ in acc})
        out[u["name"]] = (acc, grd)
    return out


def oracle_check(c, tests):
    """main() tests of the reference vs the oracle; -> (n_checked, n_mismatch, first mismatch)."""
    n = bad = 0
    first = None
    for call, exp in tests["main"][0] + tests["main"][1]:
        lines = ast.literal_eval(call[len("main("):-1])
        try:
            want = canon(oracle_main(c, lines))
        except Reject:
            continue
        n += 1
        if want != exp:
            bad += 1
            first = first or {"lines": lines, "ref": exp, "oracle": want}
    return n, bad, first


def body_key(text):
    try:
        return hashlib.sha1(ast.dump(ast.parse("def f():\n" + "".join("    " + l + "\n"
                                                                        for l in text.split("\n")))).encode()
                            ).hexdigest()
    except SyntaxError:
        return hashlib.sha1(text.encode()).hexdigest()


def build(out, seed, n_comp=160, n_seen=120):
    """train = every unit prompt of every NON-held variant; func eval = every prompt that occurs only in held variants
    (one representative held variant each); program eval = n_comp held variants (test_comp) + n_seen non-held
    variants (test_seen: every unit is a train prompt -> memorisation / assembly control)."""
    os.makedirs(out, exist_ok=True)
    r = random.Random(seed)
    cfgs = list(all_configs())
    p2b, unit_count = {}, 0
    for c in cfgs:
        for u in spec_units(c):
            unit_count += 1
            b = p2b.setdefault(u["prompt"], u["body"])
            assert b == u["body"], ("prompt is not enough to determine the body", u["name"], cfg_id(c))
    held = [c for c in cfgs if held_rules(c)]
    free = [c for c in cfgs if not held_rules(c)]
    r.shuffle(free)
    r.shuffle(held)
    tr_prompts, tr_bodies = {}, {}
    for c in free:
        for u in spec_units(c):
            tr_prompts.setdefault(u["prompt"], (u["name"], u["body"]))
            tr_bodies.setdefault(body_key(u["body"]), u["name"])
    rep = {}
    for c in held:
        for u in spec_units(c):
            if u["prompt"] not in tr_prompts and u["prompt"] not in rep:
                rep[u["prompt"]] = c
    test_comp, test_seen = held[:n_comp], free[:n_seen]
    data = {"axes": AXES, "hold_rules": list(HOLD_RULES), "seed": seed, "prelude": PRELUDE, "harness": HARNESS,
            "n_configs": len(cfgs), "n_held_configs": len(held), "n_distinct_prompts_all": len(p2b),
            "train_units": [{"name": n, "prompt": p, "body": b} for p, (n, b) in sorted(tr_prompts.items())]}
    need = {cfg_id(c): c for c in test_comp + test_seen + list(rep.values())}
    oc = {"checked": 0, "mismatch": 0, "examples": []}
    rows = {}
    for k in sorted(need):
        c = need[k]
        T = unit_tests(c, seed)
        n, bad, first = oracle_check(c, T)
        oc["checked"] += n
        oc["mismatch"] += bad
        if first and len(oc["examples"]) < 5:
            oc["examples"].append({"cfg": c, **first})
        units = []
        for u in spec_units(c):
            acc, grd = T[u["name"]]
            units.append({**u, "accept": acc, "grade": grd, "novel": u["prompt"] not in tr_prompts,
                          "body_seen": body_key(u["body"]) in tr_bodies})
        spec = {"module": "interp_" + k, "main": "main", "prelude": PRELUDE,
                "tree": {u["name"]: u["calls"] for u in units},
                "functions": [{"name": u["name"], "args": u["args"], "contract": u["contract"],
                               "calls": u["calls"], "examples": u["accept"]} for u in units]}
        rows[k] = {"cfg": c, "id": k, "held": held_rules(c), "spec": spec, "units": units}
    data["variants"] = rows
    data["test_comp"] = [cfg_id(c) for c in test_comp]
    data["test_seen"] = [cfg_id(c) for c in test_seen]
    data["func_eval"] = [{"prompt": p, "variant": cfg_id(c)} for p, c in sorted(rep.items())]
    data["oracle_check"] = oc
    json.dump(data, open(os.path.join(out, "data.json"), "w"))
    fe = [next(u for u in rows[x["variant"]]["units"] if u["prompt"] == x["prompt"]) for x in data["func_eval"]]
    by = {}
    for u in fe:
        by[u["name"]] = by.get(u["name"], 0) + 1
    stats = {"configs": len(cfgs), "held": len(held), "free": len(free), "units_all": unit_count,
             "distinct_prompts_all": len(p2b), "train_units_distinct": len(tr_prompts),
             "func_eval_novel_distinct": len(fe), "func_eval_body_seen": sum(u["body_seen"] for u in fe),
             "func_eval_by_unit": dict(sorted(by.items())),
             "test_comp_programs": len(test_comp), "test_seen_programs": len(test_seen),
             "test_comp_units": sum(len(rows[k]["units"]) for k in data["test_comp"]),
             "test_comp_novel_units": sum(u["novel"] for k in data["test_comp"] for u in rows[k]["units"]),
             "test_comp_programs_with_novel": sum(any(u["novel"] for u in rows[k]["units"]) for k in data["test_comp"]),
             "test_seen_novel_units": sum(u["novel"] for k in data["test_seen"] for u in rows[k]["units"]),
             "tests_per_unit": {"accept": 4, "grade": 16, "main_accept": 3, "main_grade": 10},
             "min_grade_cases": min(len(u["grade"]) for k in rows for u in rows[k]["units"]),
             "oracle": {k: oc[k] for k in ("checked", "mismatch")}}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))
    if oc["mismatch"]:
        print("ORACLE MISMATCH examples:", json.dumps(oc["examples"][:3], indent=1))


def selfcheck(n, seed):
    r = random.Random(seed)
    cfgs = list(all_configs())
    bad = checked = 0
    for c in r.sample(cfgs, n):
        for k in range(3):
            p = safe_program(c, r, 6)
            if p is None:
                continue
            g, _ = REFCACHE(c)
            ref = canon(g["main"](p))
            want = canon(oracle_main(c, p))
            checked += 1
            if ref != want:
                bad += 1
                if bad <= 3:
                    print("MISMATCH", cfg_id(c), c, p, ref, want, sep="\n  ")
    print(json.dumps({"selfcheck_programs": checked, "mismatch": bad}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("selfcheck")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.seed)
    else:
        selfcheck(a.n, a.seed)


if __name__ == "__main__":
    main()
