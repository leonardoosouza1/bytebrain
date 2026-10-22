"""Reference expression evaluator split into pieces (ground truth for the context test).

HEADER is module-level state every piece may use. PIECES are generated one at a time;
each piece is judged with the reference versions of all the other pieces in the module,
so a piece passes only if it uses its siblings' API correctly.
"""

HEADER = '''import math

FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
         "log": math.log10, "ln": math.log, "abs": abs}
CONSTS = {"pi": math.pi, "e": math.e}
'''

PIECES = {}

PIECES["tokenize"] = dict(
    sig="def tokenize(expr):",
    doc="Split a calculator expression into tokens. Numbers become floats; names (keys of FUNCS or "
        "CONSTS) and the operators + - * / ^ ( ) become strings. Spaces are skipped. Raise ValueError "
        "on an unknown name, a malformed number (like '1..2') or any other character.",
    body='''    tokens, i = [], 0
    while i < len(expr):
        c = expr[i]
        if c.isspace():
            i += 1
        elif c.isdigit() or c == ".":
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == "."):
                j += 1
            try:
                tokens.append(float(expr[i:j]))
            except ValueError:
                raise ValueError("bad number") from None
            i = j
        elif c.isalpha():
            j = i
            while j < len(expr) and expr[j].isalpha():
                j += 1
            name = expr[i:j]
            if name not in FUNCS and name not in CONSTS:
                raise ValueError("unknown name")
            tokens.append(name)
            i = j
        elif c in "+-*/^()":
            tokens.append(c)
            i += 1
        else:
            raise ValueError("bad character")
    return tokens
''')

PIECES["parse_atom"] = dict(
    sig="def parse_atom(tokens, pos):",
    doc="Parse one atom starting at tokens[pos] and return (value, next_pos). An atom is a float token, "
        "a constant name (CONSTS), a function call NAME '(' expr ')' with NAME in FUNCS (apply the function), "
        "or '(' expr ')'. Use parse_expr for the inner expression. Raise ValueError if tokens run out or "
        "the token cannot start an atom, or a ')' is missing.",
    body='''    if pos >= len(tokens):
        raise ValueError("unexpected end")
    t = tokens[pos]
    if isinstance(t, float):
        return t, pos + 1
    if t == "(":
        v, pos = parse_expr(tokens, pos + 1)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("missing )")
        return v, pos + 1
    if t in CONSTS:
        return CONSTS[t], pos + 1
    if t in FUNCS:
        if pos + 1 >= len(tokens) or tokens[pos + 1] != "(":
            raise ValueError("missing (")
        v, pos = parse_expr(tokens, pos + 2)
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("missing )")
        return FUNCS[t](v), pos + 1
    raise ValueError("unexpected token")
''')

PIECES["parse_power"] = dict(
    sig="def parse_power(tokens, pos):",
    doc="Parse atom ('^' exponent)? and return (value, next_pos). The exponent is parsed with parse_unary, "
        "which makes ^ right associative (2^3^2 == 512) and allows 2^-1. Use parse_atom for the base.",
    body='''    v, pos = parse_atom(tokens, pos)
    if pos < len(tokens) and tokens[pos] == "^":
        e, pos = parse_unary(tokens, pos + 1)
        v = v ** e
    return v, pos
''')

PIECES["parse_unary"] = dict(
    sig="def parse_unary(tokens, pos):",
    doc="Parse optional leading '-' or '+' signs and return (value, next_pos). '-' negates the result of "
        "parse_unary on the rest; '+' is ignored; otherwise defer to parse_power. So -2^2 == -4.",
    body='''    if pos < len(tokens) and tokens[pos] == "-":
        v, pos = parse_unary(tokens, pos + 1)
        return -v, pos
    if pos < len(tokens) and tokens[pos] == "+":
        return parse_unary(tokens, pos + 1)
    return parse_power(tokens, pos)
''')

PIECES["parse_term"] = dict(
    sig="def parse_term(tokens, pos):",
    doc="Parse unary (('*'|'/') unary)* left to right and return (value, next_pos). Operands come from "
        "parse_unary. Raise ZeroDivisionError when dividing by zero.",
    body='''    v, pos = parse_unary(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("*", "/"):
        op = tokens[pos]
        r, pos = parse_unary(tokens, pos + 1)
        if op == "*":
            v = v * r
        else:
            if r == 0:
                raise ZeroDivisionError("division by zero")
            v = v / r
    return v, pos
''')

PIECES["parse_expr"] = dict(
    sig="def parse_expr(tokens, pos):",
    doc="Parse term (('+'|'-') term)* left to right and return (value, next_pos). Operands come from parse_term.",
    body='''    v, pos = parse_term(tokens, pos)
    while pos < len(tokens) and tokens[pos] in ("+", "-"):
        op = tokens[pos]
        r, pos = parse_term(tokens, pos + 1)
        v = v + r if op == "+" else v - r
    return v, pos
''')

PIECES["evaluate"] = dict(
    sig="def evaluate(expr):",
    doc="Evaluate a calculator expression string and return a float: tokenize it, parse it with parse_expr "
        "from position 0, and raise ValueError if tokens are left over (or the expression is empty). "
        "Errors from tokenize/parse propagate.",
    body='''    tokens = tokenize(expr)
    if not tokens:
        raise ValueError("empty")
    v, pos = parse_expr(tokens, 0)
    if pos != len(tokens):
        raise ValueError("trailing tokens")
    return float(v)
''')

ORDER = ["tokenize", "parse_atom", "parse_power", "parse_unary", "parse_term", "parse_expr", "evaluate"]


def source(names=None):
    names = ORDER if names is None else names
    return "\n\n".join(PIECES[n]["sig"] + "\n" + _docstr(n) + PIECES[n]["body"] for n in names)


def _docstr(name):
    return f'    """{PIECES[name]["doc"]}"""\n'


def module():
    ns = {}
    exec(HEADER + "\n\n" + source(), ns)  # noqa: S102 - trusted reference code
    return ns
