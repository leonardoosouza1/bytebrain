"""Scientific-calculator spec: what the model sees, and the hidden inputs it is judged on.

Each entry has
  sig       signature line shown to the model
  doc       behaviour + error contract shown to the model
  examples  argument tuples shown as doctest examples (also the "visible tests")
  edge      hidden edge-case argument tuples
  gen       random hidden-input generator (rng -> args)
Expected outputs always come from calc_ref.
"""

import random

import calc_ref


def f(r, lo=-50, hi=50):
    return round(r.uniform(lo, hi), r.choice([0, 1, 2, 3]))


def pos(r, lo=0.01, hi=1000):
    return round(r.uniform(lo, hi), r.choice([1, 2, 3]))


def i(r, lo=-30, hi=30):
    return r.randint(lo, hi)


EXPRS = ["1+2*3", "(1+2)*3", "2^3^2", "-2^2", "10/4", "2*(3+4)*5", "sqrt(16)+1", "-(3-5)", "1.5*4",
         "2^-1", "ln(e)", "log(1000)", "sin(0)+cos(0)", "pi*2", "3-2-1", "8/2/2", "--3", "abs(-7)",
         "(((4)))", "2*pi", "1/0", "2+", "(1+2", "3 4", "foo(2)", "", "1..2", "2^0.5", "sqrt(-1)",
         "7-+2", "100*(2+12)/14", "tan(0)", "e^2", "0.1+0.2", "5*-1", "1+2)"]

SPEC = [
    dict(name="add", sig="def add(a, b):", doc="Return a + b.", examples=[(2, 3), (-1.5, 0.5)],
         edge=[(0, 0), (1e10, 1)], gen=lambda r: (f(r), f(r))),
    dict(name="subtract", sig="def subtract(a, b):", doc="Return a - b.", examples=[(5, 3), (0, 2.5)],
         edge=[(0, 0)], gen=lambda r: (f(r), f(r))),
    dict(name="multiply", sig="def multiply(a, b):", doc="Return a * b.", examples=[(4, 2.5), (-3, 3)],
         edge=[(0, 9)], gen=lambda r: (f(r), f(r))),
    dict(name="divide", sig="def divide(a, b):",
         doc="Return a / b. Raise ZeroDivisionError if b == 0.", examples=[(7, 2), (1, 0)],
         edge=[(0, 5), (-9, 3), (5, 0.0)], gen=lambda r: (f(r), f(r))),
    dict(name="power", sig="def power(base, exponent):", doc="Return base raised to exponent.",
         examples=[(2, 10), (9, 0.5)], edge=[(0, 0), (2, -2), (-2, 3)],
         gen=lambda r: (pos(r, 0.1, 10), round(r.uniform(-4, 4), 1))),
    dict(name="square", sig="def square(x):", doc="Return x squared.", examples=[(3,), (-1.5,)],
         edge=[(0,)], gen=lambda r: (f(r),)),
    dict(name="sqrt", sig="def sqrt(x):", doc="Square root. Raise ValueError if x < 0.",
         examples=[(16,), (2,), (-4,)], edge=[(0,), (-0.001,)], gen=lambda r: (f(r, -20, 500),)),
    dict(name="cbrt", sig="def cbrt(x):", doc="Real cube root; works for negative x (cbrt(-8) == -2).",
         examples=[(27,), (-8,)], edge=[(0,), (-0.001,), (2,)], gen=lambda r: (f(r, -500, 500),)),
    dict(name="nth_root", sig="def nth_root(x, n):",
         doc="Real n-th root of x (n is a non-zero integer). Negative x is allowed only for odd n "
             "(result is negative). Raise ValueError if n == 0 or if x < 0 and n is even.",
         examples=[(32, 5), (-27, 3), (-16, 4)], edge=[(1, 0), (0, 3), (81, 4), (-1, 5), (8, -3)],
         gen=lambda r: (f(r, -200, 200), r.choice([1, 2, 3, 4, 5, 7]))),
    dict(name="reciprocal", sig="def reciprocal(x):", doc="Return 1 / x. Raise ZeroDivisionError if x == 0.",
         examples=[(4,), (0,)], edge=[(-0.5,), (0.0,)], gen=lambda r: (f(r),)),
    dict(name="negate", sig="def negate(x):", doc="Return -x.", examples=[(5,), (-2.5,)], edge=[(0,)],
         gen=lambda r: (f(r),)),
    dict(name="absolute", sig="def absolute(x):", doc="Return the absolute value of x.",
         examples=[(-3,), (2.5,)], edge=[(0,)], gen=lambda r: (f(r),)),
    dict(name="percent", sig="def percent(x):", doc="Convert a percentage to a fraction: percent(50) == 0.5.",
         examples=[(50,), (12.5,)], edge=[(0,), (-20,)], gen=lambda r: (f(r, -500, 500),)),
    dict(name="percent_of", sig="def percent_of(p, x):", doc="Return p percent of x (percent_of(10, 200) == 20).",
         examples=[(10, 200), (50, 3)], edge=[(0, 5), (150, 40)], gen=lambda r: (f(r, 0, 200), f(r))),
    dict(name="log10", sig="def log10(x):", doc="Base-10 logarithm. Raise ValueError if x <= 0.",
         examples=[(1000,), (0,)], edge=[(1,), (0.01,), (-5,)], gen=lambda r: (f(r, -10, 5000),)),
    dict(name="ln", sig="def ln(x):", doc="Natural logarithm. Raise ValueError if x <= 0.",
         examples=[(1,), (-1,)], edge=[(0,), (2.718281828459045,)], gen=lambda r: (f(r, -10, 500),)),
    dict(name="log_base", sig="def log_base(x, base):",
         doc="Logarithm of x in the given base. Raise ValueError if x <= 0, base <= 0 or base == 1.",
         examples=[(8, 2), (100, 10), (5, 1)], edge=[(1, 3), (0, 2), (9, -3), (9, 0.5)],
         gen=lambda r: (f(r, -5, 1000), r.choice([2, 3, 10, 0.5, 1, 7.5, -2]))),
    dict(name="exp", sig="def exp(x):", doc="Return e ** x.", examples=[(0,), (1,)], edge=[(-3,)],
         gen=lambda r: (f(r, -10, 10),)),
    dict(name="factorial", sig="def factorial(n):",
         doc="n! for a non-negative integer n. A float with an integral value (5.0) is accepted. "
             "Raise ValueError for negative or non-integral n. Returns an int.",
         examples=[(5,), (0,), (-1,)], edge=[(5.0,), (2.5,), (1,), (20,)], gen=lambda r: (r.randint(-3, 25),)),
    dict(name="permutations", sig="def permutations(n, r):",
         doc="Number of ordered arrangements nPr = n! / (n-r)!. Raise ValueError if n < 0, r < 0 or r > n.",
         examples=[(5, 2), (3, 4)], edge=[(0, 0), (6, 6), (4, -1)], gen=lambda r: (r.randint(-1, 15), r.randint(-1, 15))),
    dict(name="combinations", sig="def combinations(n, r):",
         doc="Binomial coefficient nCr. Raise ValueError if n < 0, r < 0 or r > n.",
         examples=[(5, 2), (2, 3)], edge=[(0, 0), (10, 0), (-2, 1)], gen=lambda r: (r.randint(-1, 25), r.randint(-1, 25))),
    dict(name="sin", sig="def sin(x, degrees=False):", doc="Sine of x; x is in degrees when degrees=True, else radians.",
         examples=[(0,), (90, True)], edge=[(180, True), (3.141592653589793,)],
         gen=lambda r: (f(r, -400, 400), r.random() < 0.5)),
    dict(name="cos", sig="def cos(x, degrees=False):", doc="Cosine of x; x is in degrees when degrees=True, else radians.",
         examples=[(0,), (60, True)], edge=[(90, True), (-1.2,)], gen=lambda r: (f(r, -400, 400), r.random() < 0.5)),
    dict(name="tan", sig="def tan(x, degrees=False):", doc="Tangent of x; x is in degrees when degrees=True, else radians.",
         examples=[(0,), (45, True)], edge=[(-45, True), (1,)], gen=lambda r: (f(r, -80, 80), r.random() < 0.5)),
    dict(name="asin", sig="def asin(x, degrees=False):",
         doc="Arc sine; result in degrees when degrees=True, else radians. Raise ValueError if x is outside [-1, 1].",
         examples=[(1,), (0.5, True), (2,)], edge=[(-1, True), (1.0001,)], gen=lambda r: (round(r.uniform(-1.2, 1.2), 3), r.random() < 0.5)),
    dict(name="acos", sig="def acos(x, degrees=False):",
         doc="Arc cosine; result in degrees when degrees=True, else radians. Raise ValueError if x is outside [-1, 1].",
         examples=[(1,), (0, True), (-3,)], edge=[(-1,), (-1.0001, True)], gen=lambda r: (round(r.uniform(-1.2, 1.2), 3), r.random() < 0.5)),
    dict(name="atan", sig="def atan(x, degrees=False):", doc="Arc tangent; result in degrees when degrees=True, else radians.",
         examples=[(1, True), (0,)], edge=[(-1e6,)], gen=lambda r: (f(r, -100, 100), r.random() < 0.5)),
    dict(name="sinh", sig="def sinh(x):", doc="Hyperbolic sine.", examples=[(0,), (1,)], edge=[(-2,)], gen=lambda r: (f(r, -10, 10),)),
    dict(name="cosh", sig="def cosh(x):", doc="Hyperbolic cosine.", examples=[(0,), (1,)], edge=[(-2,)], gen=lambda r: (f(r, -10, 10),)),
    dict(name="tanh", sig="def tanh(x):", doc="Hyperbolic tangent.", examples=[(0,), (1,)], edge=[(-2,)], gen=lambda r: (f(r, -10, 10),)),
    dict(name="deg_to_rad", sig="def deg_to_rad(x):", doc="Convert degrees to radians.", examples=[(180,), (90,)],
         edge=[(0,), (-45,)], gen=lambda r: (f(r, -720, 720),)),
    dict(name="rad_to_deg", sig="def rad_to_deg(x):", doc="Convert radians to degrees.", examples=[(3.141592653589793,), (1,)],
         edge=[(0,)], gen=lambda r: (f(r, -10, 10),)),
    dict(name="mod", sig="def mod(a, b):", doc="Remainder a % b with Python semantics. Raise ZeroDivisionError if b == 0.",
         examples=[(7, 3), (-7, 3), (1, 0)], edge=[(7, -3), (5.5, 2)], gen=lambda r: (i(r), i(r))),
    dict(name="floor_div", sig="def floor_div(a, b):", doc="Floor division a // b. Raise ZeroDivisionError if b == 0.",
         examples=[(7, 2), (-7, 2), (3, 0)], edge=[(7, -2), (0, 5)], gen=lambda r: (i(r), i(r))),
    dict(name="round_to", sig="def round_to(x, digits):", doc="Round x to the given number of decimal digits (like round()).",
         examples=[(3.14159, 2), (2.5, 0)], edge=[(1234.5678, -2), (-1.005, 1)], gen=lambda r: (f(r, -1000, 1000), r.randint(-2, 4))),
    dict(name="gcd", sig="def gcd(a, b):", doc="Greatest common divisor of two integers (always >= 0; gcd(0, 0) == 0).",
         examples=[(12, 18), (-4, 6)], edge=[(0, 0), (0, 7), (17, 5)], gen=lambda r: (i(r, -100, 100), i(r, -100, 100))),
    dict(name="lcm", sig="def lcm(a, b):", doc="Least common multiple of two integers (always >= 0; 0 if either is 0).",
         examples=[(4, 6), (-3, 5)], edge=[(0, 9), (7, 7)], gen=lambda r: (i(r, -50, 50), i(r, -50, 50))),
    dict(name="is_prime", sig="def is_prime(n):", doc="True if the integer n is prime, else False.",
         examples=[(7,), (1,), (9,)], edge=[(2,), (0,), (-7,), (97,), (7919,)], gen=lambda r: (r.randint(-5, 400),)),
    dict(name="evaluate", sig="def evaluate(expr):",
         doc="Evaluate a calculator expression string and return a float. Supports numbers (with decimals), "
             "+ - * /, ^ for power (right associative, binds tighter than unary minus: -2^2 == -4), parentheses, "
             "unary minus/plus, functions sqrt sin cos tan (radians) log (base 10) ln abs, and constants pi e. "
             "Raise ZeroDivisionError on division by zero and ValueError on any invalid expression. "
             "Do not use eval().",
         examples=[("1+2*3",), ("2^3^2",), ("(1+2",)], edge=[(e,) for e in EXPRS],
         gen=lambda r: (r.choice(EXPRS),)),
]


def hidden_cases(entry, n_random=12, seed=0):
    r = random.Random(f"hidden-{entry['name']}-{seed}")
    cases = list(entry["edge"]) + [entry["gen"](r) for _ in range(n_random)]
    seen, out = set(), []
    for c in cases:
        if repr(c) not in seen:
            seen.add(repr(c))
            out.append(c)
    return out


def fuzz_cases(entry, n=24):
    """Inputs for behaviour clustering. No expected outputs are used with these."""
    r = random.Random(f"fuzz-{entry['name']}")
    return list(entry["examples"]) + [entry["gen"](r) for _ in range(n)]


def expected(entry, args):
    fn = getattr(calc_ref, entry["name"])
    try:
        return ("ok", fn(*args))
    except Exception as e:  # noqa: BLE001 - the error type is the expected value
        return ("err", type(e).__name__)


def fmt_example(entry, args):
    call = f"{entry['name']}({', '.join(repr(a) for a in args)})"
    kind, v = expected(entry, args)
    return f">>> {call}\n{v!r}" if kind == "ok" else f">>> {call}\nTraceback (most recent call last):\n{v}"


def stub(entry):
    ex = "\n".join("    " + line for e in entry["examples"] for line in fmt_example(entry, e).split("\n"))
    return f'{entry["sig"]}\n    """{entry["doc"]}\n\n{ex}\n    """\n'
