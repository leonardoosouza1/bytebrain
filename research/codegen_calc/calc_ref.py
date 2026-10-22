"""Reference scientific calculator ("ground truth" for the code-generation test).

Every public function here is specified in spec.py; generated code is judged by
comparing its behaviour against these functions on hidden inputs.
"""

import math


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    return a * b


def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b


def power(base, exponent):
    return base ** exponent


def square(x):
    return x * x


def sqrt(x):
    if x < 0:
        raise ValueError("negative input")
    return math.sqrt(x)


def cbrt(x):
    r = abs(x) ** (1.0 / 3.0)
    return -r if x < 0 else r


def nth_root(x, n):
    if n == 0:
        raise ValueError("zeroth root")
    if x < 0:
        if n % 2 == 0:
            raise ValueError("even root of negative")
        return -((-x) ** (1.0 / n))
    return x ** (1.0 / n)


def reciprocal(x):
    if x == 0:
        raise ZeroDivisionError("division by zero")
    return 1 / x


def negate(x):
    return -x


def absolute(x):
    return abs(x)


def percent(x):
    return x / 100


def percent_of(p, x):
    return p * x / 100


def log10(x):
    if x <= 0:
        raise ValueError("log of non-positive")
    return math.log10(x)


def ln(x):
    if x <= 0:
        raise ValueError("log of non-positive")
    return math.log(x)


def log_base(x, base):
    if x <= 0 or base <= 0 or base == 1:
        raise ValueError("invalid log")
    return math.log(x) / math.log(base)


def exp(x):
    return math.exp(x)


def factorial(n):
    if isinstance(n, float):
        if not n.is_integer():
            raise ValueError("non-integer")
        n = int(n)
    if n < 0:
        raise ValueError("negative")
    return math.factorial(n)


def _check_nr(n, r):
    if n < 0 or r < 0 or r > n:
        raise ValueError("invalid n, r")


def permutations(n, r):
    _check_nr(n, r)
    return math.perm(n, r)


def combinations(n, r):
    _check_nr(n, r)
    return math.comb(n, r)


def _to_rad(x, degrees):
    return math.radians(x) if degrees else x


def sin(x, degrees=False):
    return math.sin(_to_rad(x, degrees))


def cos(x, degrees=False):
    return math.cos(_to_rad(x, degrees))


def tan(x, degrees=False):
    return math.tan(_to_rad(x, degrees))


def _from_rad(y, degrees):
    return math.degrees(y) if degrees else y


def asin(x, degrees=False):
    if x < -1 or x > 1:
        raise ValueError("out of domain")
    return _from_rad(math.asin(x), degrees)


def acos(x, degrees=False):
    if x < -1 or x > 1:
        raise ValueError("out of domain")
    return _from_rad(math.acos(x), degrees)


def atan(x, degrees=False):
    return _from_rad(math.atan(x), degrees)


def sinh(x):
    return math.sinh(x)


def cosh(x):
    return math.cosh(x)


def tanh(x):
    return math.tanh(x)


def deg_to_rad(x):
    return math.radians(x)


def rad_to_deg(x):
    return math.degrees(x)


def mod(a, b):
    if b == 0:
        raise ZeroDivisionError("modulo by zero")
    return a % b


def floor_div(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a // b


def round_to(x, digits):
    return round(x, digits)


def gcd(a, b):
    return math.gcd(a, b)


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


_FUNCS = {"sqrt": sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": log10, "ln": ln,
          "abs": abs}
_CONSTS = {"pi": math.pi, "e": math.e}


def evaluate(expr):
    """Recursive-descent evaluator.

    grammar:  expr   := term (('+'|'-') term)*
              term   := unary (('*'|'/') unary)*
              unary  := '-' unary | '+' unary | power
              power  := atom ('^' unary)?          (right associative, binds tighter than unary minus on its left)
              atom   := number | const | func '(' expr ')' | '(' expr ')'
    """
    toks = _tokenize(expr)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take(expected=None):
        t = peek()
        if t is None or (expected is not None and t != expected):
            raise ValueError("syntax error")
        pos[0] += 1
        return t

    def p_expr():
        v = p_term()
        while peek() in ("+", "-"):
            v = v + p_term() if take() == "+" else v - p_term()
        return v

    def p_term():
        v = p_unary()
        while peek() in ("*", "/"):
            if take() == "*":
                v = v * p_unary()
            else:
                d = p_unary()
                if d == 0:
                    raise ZeroDivisionError("division by zero")
                v = v / d
        return v

    def p_unary():
        if peek() == "-":
            take()
            return -p_unary()
        if peek() == "+":
            take()
            return p_unary()
        return p_power()

    def p_power():
        v = p_atom()
        if peek() == "^":
            take()
            v = v ** p_unary()
        return v

    def p_atom():
        t = peek()
        if t is None:
            raise ValueError("syntax error")
        if isinstance(t, float):
            take()
            return t
        if t == "(":
            take()
            v = p_expr()
            take(")")
            return v
        if t in _CONSTS:
            take()
            return _CONSTS[t]
        if t in _FUNCS:
            take()
            take("(")
            v = p_expr()
            take(")")
            return _FUNCS[t](v)
        raise ValueError("syntax error")

    v = p_expr()
    if peek() is not None:
        raise ValueError("syntax error")
    return float(v)


def _tokenize(s):
    toks, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            try:
                toks.append(float(s[i:j]))
            except ValueError:
                raise ValueError("bad number") from None
            i = j
        elif c.isalpha():
            j = i
            while j < len(s) and s[j].isalpha():
                j += 1
            name = s[i:j]
            if name not in _FUNCS and name not in _CONSTS:
                raise ValueError("unknown name")
            toks.append(name)
            i = j
        elif c in "+-*/^()":
            toks.append(c)
            i += 1
        else:
            raise ValueError("bad character")
    return toks
