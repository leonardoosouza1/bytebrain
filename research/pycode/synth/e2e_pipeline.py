"""End-to-end validation belt with stage blame and judges (ledger #36-#39, registered 2026-10-26).

request (pt-BR) -> UNDERSTANDER (gold | Qwen via llama-server) -> spec JSON -> bridge (rule: spec -> name)
  -> PLANNER (oracle | net) -> plan symbols -> WRITER (oracle | net) -> function texts -> assembler (rule) -> code
  -> bwrap exec -> per-capability grade against the gold reference.
The calculator world has gold for every stage, so the harness measures where the belt breaks:
  conditional correctness: a stage is wrong only if it broke its contract FOR THE INPUT IT RECEIVED
    understander: reference(spec) computes every requested capability
    planner:      plan + oracle texts == reference(spec) on the spec's own calls
    writer:       every written function, alone among oracle helpers, equals the oracle function
  oracle swap counterfactuals: CF_U gold spec, CF_P oracle plan, CF_W oracle texts (does fixing one stage repair it?)
  machine validators + cross-stage flags (no gold needed): their precision is measured against the oracle blame
  fault injection with known location: calibrates the harness (check) and the judges
  judges (RUBRIC.md): Qwen first, Sonnet final; score computed by code; a fix is accepted only if execution confirms it
Loose first: no send-back retries here; flags are recorded, not acted on.

Subcommands:
  check     oracle belt on every request (expect 100%, no flags, nothing broken) + injected faults (expect the
            oracle blame to land on the injected stage)
  train     train + save dispatch planner, compose planner, writer (CPU); reproduce the known eval numbers
  run       belt over the request set -> traces_<tag>.jsonl + summary_<tag>.json
  judge     Qwen judge (+ Sonnet final judge) on traces -> verdicts, agreement with the oracle, verified fixes
  one       serve one request text end to end and print code + per-stage latency (#41 preview)

Usage:
  e2e_pipeline.py check --out DIR
  e2e_pipeline.py train --ckpt DIR [--seed 0 --steps 2000 --threads 4]
  e2e_pipeline.py run --out DIR --tag T --understander gold|qwen [--url U] --planner net|oracle --writer net|oracle
                      [--ckpt DIR] [--inject rotate]
  e2e_pipeline.py judge --out DIR --traces FILE --qwen-url U [--sonnet-mode off|all|escalate --max-sonnet-calls 40
                      --n 60 --ckpt DIR --claude-config-dir D]
  e2e_pipeline.py one --ckpt DIR --url U "faça uma função de soma"
"""

import argparse
import ast
import builtins
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockaddr as BA  # noqa: E402
import compose as CP  # noqa: E402
import multifn as MF  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUBRIC_PATH = os.path.join(HERE, "..", "RUBRIC.md")
MF.PLAN_FORMAT = "interleaved"                       # the format that reached 100% on len4-8 and the full calculator
MF.MAX_OUT = max(34, len(MF.plan_of(tuple(MF.OPS))) + 4)
OPS, BIN, CASES = MF.OPS, CP.BIN, MF.CASES
FACT_CASES = list(CASES)                             # judges see every case the truth uses (no expected values)
BUILTIN_NAMES = set(dir(builtins)) - set(OPS)
STAGES = ["understander", "planner", "writer", "assembler"]

# ----------------------------------------------------------------------------------------------- requests (pt-BR)

PT = {"add": ["soma", "somar dois números", "adição"], "sub": ["subtração", "subtrair"],
      "mul": ["multiplicação", "multiplicar"], "div": ["divisão", "dividir"], "mod": ["resto da divisão"],
      "maxx": ["maior entre dois números", "máximo de dois valores"], "sqrt": ["raiz quadrada"],
      "neg": ["inverter o sinal", "negativo de um número"], "abs": ["valor absoluto"],
      "square": ["quadrado de um número", "elevar ao quadrado"], "pow": ["potência", "elevar a à potência b"],
      "fact": ["fatorial"], "hyp": ["hipotenusa", "hipotenusa de um triângulo retângulo"],
      "avg": ["média de dois números", "média"], "pct": ["porcentagem (a por cento de b)", "a por cento de b"]}
PX = {"add": ["a soma de {x} e {y}", "{x} mais {y}"], "sub": ["{x} menos {y}", "a diferença entre {x} e {y}"],
      "mul": ["{x} vezes {y}", "o produto de {x} e {y}"], "div": ["{x} dividido por {y}"],
      "avg": ["a média de {x} e {y}"], "pct": ["{x} por cento de {y}"],
      "hyp": ["a hipotenusa de catetos {x} e {y}"], "maxx": ["o maior entre {x} e {y}"],
      "mod": ["o resto de {x} dividido por {y}"]}
FRAMES_D = ["crie uma calculadora que faça {L}", "quero um programa em python com {L}",
            "faça uma calculadora com as operações: {L}", "preciso de uma função main que escolha entre {L}"]
FRAMES_X = ["crie uma função main(a, b) que retorna {P}", "faça um programa cuja main(a, b) devolve {P}"]
OPEN = [("crie uma calculadora básica", ["add", "sub", "mul", "div"]),
        ("faça uma calculadora simples com as quatro operações", ["add", "sub", "mul", "div"]),
        ("crie uma calculadora científica", ["add", "sub", "mul", "div", "sqrt", "pow"]),
        ("quero uma calculadora científica em python", ["add", "sub", "mul", "div", "sqrt", "pow"])]


def say_list(ws):
    return ws[0] if len(ws) == 1 else ", ".join(ws[:-1]) + " e " + ws[-1]


def verbalize(prefix, r):
    def go(i):
        t = prefix[i]
        if t in ("a", "b"):
            return t, i + 1, True
        x, j, lx = go(i + 1)
        y, j, ly = go(j)
        x, y = (x if lx else f"({x})"), (y if ly else f"({y})")
        return r.choice(PX[t]).format(x=x, y=y), j, False
    return go(0)[0]


def expr_of(prefix):
    return CP.render_expr(CP.plan_of(prefix))


def make_requests(seed=0):
    r, md, cd, reqs = random.Random(seed), MF.Data(), CP.Data(), []

    def add_d(ops, split, text=None):
        ops = list(ops)
        text = text or r.choice(FRAMES_D).format(L=say_list([r.choice(PT[o]) for o in ops]))
        reqs.append({"id": f"d{len(reqs):03d}", "world": "dispatch", "split": split, "text": text, "caps": ops,
                     "gold": {"main": "dispatch", "calls": ops}})

    def add_x(prefix, split):
        reqs.append({"id": f"x{len(reqs):03d}", "world": "expr", "split": split,
                     "text": r.choice(FRAMES_X).format(P=verbalize(prefix, r)), "caps": ["formula"],
                     "prefix": list(prefix), "gold": {"main": "expr", "expr": expr_of(prefix)}})
    for o in OPS:
        add_d((o,), "one")
    for k, n in (("test_comb", 40), ("len4", 25), ("len5", 15)):
        for sp in md.sets[k][:n]:
            add_d(sp, k)
    for text, ops in OPEN:
        add_d(ops, "open", text)
    for k, n in (("test_same", 40), ("calls4", 25), ("calls5", 15)):
        for sp in cd.sets[k][:n]:
            add_x(sp, k)
    return reqs

# ----------------------------------------------------------------------------------------------- exec (bwrap)


RUNNER = r"""
import io, json, signal, sys
def _alarm(*_):
    raise TimeoutError
signal.signal(signal.SIGALRM, _alarm)
items = json.load(sys.stdin)
real, res = sys.stdout, []
sys.stdout = io.StringIO()
for src, calls in items:
    g = {}
    try:
        signal.alarm(2)
        exec(src, g)
        signal.alarm(0)
    except BaseException:
        signal.alarm(0)
        res.append(None)
        continue
    row = []
    for fn, args in calls:
        try:
            signal.alarm(1)
            v = g[fn](*args)
            signal.alarm(0)
            row.append(repr(round(v, 6) if isinstance(v, float) else v))
        except BaseException as e:
            signal.alarm(0)
            row.append("!" + type(e).__name__)
    res.append(row)
    sys.stdout = io.StringIO()
real.write(json.dumps(res))
"""


def run_many(items):
    """items: [(src, [(fn, args), ...])] -> [None | [repr per call]]; None = module failed to load."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    n_calls = sum(len(c) for _, c in items)
    try:
        p = subprocess.run(cmd, input=json.dumps(items), capture_output=True, text=True,
                           timeout=60 + 2 * len(items) + n_calls // 50)
        if p.returncode and "bwrap" in p.stderr:
            raise RuntimeError("sandbox failed: " + p.stderr[:300])
        out = json.loads(p.stdout)
        if len(out) != len(items):
            raise ValueError
        return out
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        if len(items) == 1:
            return [None]
        h = len(items) // 2
        return run_many(items[:h]) + run_many(items[h:])


def calls_for(key):
    kind, _, o = key.partition(":")
    if kind == "d":
        return [("main", [o, a, b]) for a, b in CASES]
    if kind == "x":
        return [("main", [a, b]) for a, b in CASES]
    if kind == "f":
        return [(o, [a, b]) for a, b in CASES]
    if kind == "jd":
        return [("main", [o, a, b]) for a, b in FACT_CASES]
    if kind == "jx":
        return [("main", [a, b]) for a, b in FACT_CASES]
    raise KeyError(key)


class Batch:
    """collect (module, keys) jobs, run them in one bwrap pass, read results by handle and key."""

    def __init__(self):
        self.jobs, self.res = [], []

    def add(self, src, keys):
        self.jobs.append((src, list(dict.fromkeys(keys))))
        return len(self.jobs) - 1

    def run(self):
        items = [(src, [c for k in keys for c in calls_for(k)]) for src, keys in self.jobs]
        outs = []
        for i in range(0, len(items), 300):
            outs += run_many(items[i:i + 300])
        for (src, keys), row in zip(self.jobs, outs):
            d, pos = {}, 0
            for k in keys:
                n = len(calls_for(k))
                d[k] = None if row is None else row[pos:pos + n]
                pos += n
            self.res.append(d)
        return self

    def __getitem__(self, h):
        return self.res[h]


ORACLE_DEFS = "".join(MF.block_text(s) for s in MF.plan_of_grouped(tuple(OPS)) if s.startswith("def:"))
REF_ALL_SRC = MF.assemble(MF.plan_of_grouped(tuple(OPS)))
_REF = {}


def ref_all():
    """gold outputs: d:<op> through main, f:<op> per function (computed once)."""
    if not _REF:
        b = Batch()
        h = b.add(REF_ALL_SRC, [f"d:{o}" for o in OPS] + [f"f:{o}" for o in OPS])
        _REF.update(b.run()[h])
        assert all(v is not None and not any(x.startswith("!") for x in v) for v in _REF.values()), \
            "sandbox failed: reference outputs missing"
    return _REF

# ----------------------------------------------------------------------------------------------- spec -> bridge


def flag(frm, to, kind, fn="", evidence=""):
    return {"from": frm, "to": to, "kind": kind, "fn": fn, "evidence": str(evidence)[:160]}


def parse_expr(e):
    """'hyp(avg(a, b), b)' -> prefix tuple; (None, reason) when outside the compose world."""
    try:
        node = ast.parse(e, mode="eval").body
    except SyntaxError:
        return None, "syntax"
    out = []

    def go(n):
        if isinstance(n, ast.Name) and n.id in ("a", "b"):
            out.append(n.id)
            return
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) == 2 and not n.keywords:
            if n.func.id not in BIN:
                raise ValueError(f"oov:{n.func.id}")
            out.append(n.func.id)
            go(n.args[0])
            go(n.args[1])
            return
        raise ValueError("shape:" + ast.dump(n)[:40])
    try:
        go(node)
    except (ValueError, RecursionError) as x:
        return None, str(x)
    return tuple(out), ""


STEP_RE = re.compile(r"s([0-9]{1,3}) = (\w+)\((\w+), (\w+)\)$", re.ASCII)   # ASCII: int() never sees "²"


def steps_of(prefix):
    """prefix tree -> addressed steps in post-order (ledger #43 A1): ["s1 = pct(b, a)", "s2 = mod(s1, a)", ...].
    A request is said as a sequence ("take b percent of a, then the remainder of that by a"); each step gets an
    address, so the understander never has to write the tree outside-in. The last step is the result."""
    steps = []

    def go(i):
        t = prefix[i]
        if t in ("a", "b"):
            return t, i + 1
        x, j = go(i + 1)
        y, j = go(j)
        steps.append(f"s{len(steps) + 1} = {t}({x}, {y})")
        return f"s{len(steps)}", j
    go(0)
    return steps


def parse_steps(steps, max_len=99):
    """addressed steps -> prefix tuple; (None, reason naming the step) when malformed. A step used twice (a shared
    sub-result) is expanded into the tree; a step never used is an error (the request said it, the formula lost it)."""
    if not isinstance(steps, list) or not steps:
        return None, "shape:steps"
    trees, used = [], set()
    for k, s in enumerate(steps, 1):
        m = STEP_RE.match(s.strip()) if isinstance(s, str) else None
        if not m or int(m.group(1)) != k:
            return None, f"shape:step{k}"
        op, sub = m.group(2), [m.group(2)]
        if op not in BIN:
            return None, f"oov:{op}"
        for v in m.group(3, 4):
            if v in ("a", "b"):
                sub.append(v)
            elif v[0] == "s" and v[1:].isdigit() and len(v) <= 4 and 1 <= int(v[1:]) < k:
                sub.extend(trees[int(v[1:]) - 1])
                used.add(int(v[1:]))
            else:
                return None, f"ref:step{k}"
        if len(sub) > max_len:
            return None, f"size:step{k}"
        trees.append(tuple(sub))
    dead = [k for k in range(1, len(steps)) if k not in used]
    if dead:
        return None, f"unused:step{dead[0]}"
    return trees[-1], ""


# ---- grammar mask (ledger #43 A0/A2): is this text a valid PREFIX of a spec in json.dumps form? Each parser returns
# (end, incomplete) pairs; incomplete = the text ended inside a valid parse, so a continuation exists.

def _lit(s, i, lit):
    seg = s[i:i + len(lit)]
    if not lit.startswith(seg):
        return None
    return (i + len(lit), False) if len(seg) == len(lit) else (len(s), True)


def _expr_p(s, i, depth=0):
    if i >= len(s):
        return [(len(s), True)]
    out = [r for r in (_lit(s, i, "a"), _lit(s, i, "b")) if r]
    for op in BIN if depth < 30 else ():                 # 30 > any gold depth (5); check_spec has no cap
        r = _lit(s, i, op + "(")
        if not r or r[1]:
            out += [r] if r else []
            continue
        for p, inc in _expr_p(s, r[0], depth + 1):
            r2 = (p, True) if inc else _lit(s, p, ", ")
            if not r2 or r2[1]:
                out += [r2] if r2 else []
                continue
            for q, inc2 in _expr_p(s, r2[0], depth + 1):
                r3 = (q, True) if inc2 else _lit(s, q, ")")
                out += [r3] if r3 else []
    return out


def _calls_p(s, i):
    def one(j):
        return [r for r in (_lit(s, j, f'"{o}"') for o in OPS) if r]
    out, frontier = [], one(i)
    while frontier:
        nxt = []
        for p, inc in frontier:
            if inc:
                out.append((p, True))
                continue
            out += [r for r in (_lit(s, p, "]}"),) if r]
            c = _lit(s, p, ", ")
            if c:
                out += [c] if c[1] else []
                nxt += [] if c[1] else one(c[0])
        frontier = nxt
    return out


def _steps_p(s, i, max_steps=12, max_len=99):
    out, frontier = [], [(i, 1, frozenset(), ())]       # sizes[j]: tokens of step j+1's expanded tree (parse_steps)
    while frontier:
        nxt = []
        for p, k, used, sizes in frontier:
            head = _lit(s, p, f'"s{k} = ')
            if not head or head[1]:
                out += [head] if head else []
                continue
            args = ["a", "b"] + [f"s{j}" for j in range(1, k)]
            for op in BIN:
                r1 = _lit(s, head[0], op + "(")
                if not r1 or r1[1]:
                    out += [r1] if r1 else []
                    continue
                for x in args:
                    r2 = _lit(s, r1[0], x + ", ")
                    if not r2 or r2[1]:
                        out += [r2] if r2 else []
                        continue
                    for y in args:
                        r3 = _lit(s, r2[0], y + ')"')
                        if not r3 or r3[1]:
                            out += [r3] if r3 else []
                            continue
                        size = 1 + sum(1 if v in ("a", "b") else sizes[int(v[1:]) - 1] for v in (x, y))
                        if size > max_len:
                            continue
                        u = used | {v for v in (x, y) if v[0] == "s"}
                        if all(f"s{j}" in u for j in range(1, k)):      # close only when no step is left unused
                            out += [r for r in (_lit(s, r3[0], "]}"),) if r]
                        if k - sum(f"s{j}" in u for j in range(1, k + 1)) > 2 * (max_steps - k):
                            continue                     # too many unused steps left for the steps that remain
                        c = _lit(s, r3[0], ", ")
                        if c and c[1]:
                            out.append(c)
                        elif c and k < max_steps:
                            nxt.append((c[0], k + 1, u, sizes + (size,)))
        frontier = nxt
    return out


def spec_prefix_ok(s, complete=False, steps=False):
    """True when s can still grow into a valid spec (complete=True: s IS one). Formats: dispatch list, and the formula
    as one nested expr (steps=False) or as addressed steps (steps=True)."""
    ends = []
    for head, body in (('{"main": "dispatch", "calls": [', _calls_p),
                       ('{"main": "expr", "steps": [', _steps_p) if steps else
                       ('{"main": "expr", "expr": "', lambda t, j: [((q, True) if inc else _lit(t, q, '"}'))
                                                                    for q, inc in _expr_p(t, j)])):
        r = _lit(s, 0, head)
        if r:
            ends += [r] if r[1] else [e for e in body(s, r[0]) if e]
    return any(p == len(s) and not (complete and inc) for p, inc in ends)


def canon_spec(spec):
    """a steps spec -> the same spec as one nested expr (the gold format); anything else unchanged."""
    if isinstance(spec, dict) and set(spec) == {"main", "steps"} and spec["main"] == "expr":  # extra keys: no credit
        prefix, _ = parse_steps(spec["steps"])
        if prefix is not None:
            return {"main": "expr", "expr": expr_of(prefix)}
    return spec


def to_steps_spec(spec):
    """gold spec -> addressed-steps spec; dispatch specs unchanged."""
    if spec.get("main") != "expr":
        return spec
    prefix, why = parse_expr(spec["expr"])
    assert prefix is not None, why
    steps = steps_of(prefix)
    assert steps, f"leaf-only formula {spec['expr']!r} has no steps form"
    return {"main": "expr", "steps": steps}


def check_spec(raw):
    """understander raw text -> (spec dict | None, bridged (world, tuple) | None, flags). The bridge is a rule."""
    try:
        s = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, None, [flag("validator", "understander", "json_error", "", raw)]
    if not isinstance(s, dict) or s.get("main") not in ("dispatch", "expr"):
        return s, None, [flag("validator", "understander", "schema_error", "", raw)]
    if s["main"] == "dispatch":
        calls = s.get("calls")
        if not isinstance(calls, list) or not all(isinstance(c, str) for c in calls):
            return s, None, [flag("validator", "understander", "schema_error", "", raw)]
        fl = [flag("validator", "understander", "oov_op", c) for c in calls if c not in OPS]
        ops = tuple(dict.fromkeys(c for c in calls if c in OPS))
        if not ops:
            return s, None, fl + [flag("validator", "understander", "empty_spec")]
        return s, ("dispatch", ops), fl
    if "steps" in s:                                     # addressed steps (A1); the nested expr string stays accepted
        e = s["steps"]
        prefix, why = parse_steps(e)
    else:
        e = s.get("expr")
        if not isinstance(e, str):
            return s, None, [flag("validator", "understander", "schema_error", "", raw)]
        prefix, why = parse_expr(e)
    if prefix is None:
        kind = "oov_op" if why.startswith("oov:") else "expr_parse_error"
        where = why.split(":")[-1]                       # oov: the bad id; steps: the step that broke (micro blame)
        return s, None, [flag("validator", "understander", kind,
                              where if kind == "oov_op" or (where.startswith("step") and where[4:].isdigit())
                              else "", e)]
    return s, ("expr", prefix), []


def spec_of(bridged):
    w, t = bridged
    return {"main": "dispatch", "calls": list(t)} if w == "dispatch" else {"main": "expr", "expr": expr_of(t)}


def gold_bridged(req):
    return ("dispatch", tuple(req["caps"])) if req["world"] == "dispatch" else ("expr", tuple(req["prefix"]))


def oracle_plan(bridged):
    w, t = bridged
    return MF.plan_of(t) if w == "dispatch" else CP.plan_of(t)


def assemble(world, plan, text_of):
    if world == "dispatch":
        body = [s for s in plan if s.startswith(("def:", "disp:"))]
        defs = "".join(text_of(s) for s in dict.fromkeys(s for s in body if s.startswith("def:")))
        disp = "".join(text_of(s) for s in body if s.startswith("disp:"))
        return defs + "def main(op, a, b=0):\n" + disp + "    return None\n"
    return CP.assemble(plan, text_of)


def bridged_name(bridged):
    return MF.spec_name(bridged[1]) if bridged[0] == "dispatch" else CP.spec_name(bridged[1])

# ----------------------------------------------------------------------------------------------- backends


def use_world(world, meta):
    if world == "dispatch":
        BA.SYM, BA.START, BA.MAX_OUT = MF.SYM, len(MF.SYM), meta["dispatch_max_out"]
    else:
        BA.SYM, BA.START, BA.MAX_OUT = CP.SYM, len(CP.SYM), meta["compose_max_out"]


def load_meta(ckpt):
    return json.load(open(os.path.join(ckpt, "meta.json")))


class Planner:
    def __init__(self, kind, ckpt=None):
        self.kind, self.nets = kind, {}
        if kind == "net":
            self.meta = load_meta(ckpt)
            for w in ("dispatch", "expr"):
                use_world(w, self.meta)
                net = BA.Net("addr", gate_bias=-4.0)
                net.load_state_dict(torch.load(os.path.join(ckpt, f"planner_{w}.pt")))
                self.nets[w] = net.eval()

    def plan(self, bridged_list):
        """[(world, tuple)] -> [symbol list]; one greedy batch per world."""
        out = [None] * len(bridged_list)
        for w in ("dispatch", "expr"):
            idx = [i for i, b in enumerate(bridged_list) if b[0] == w]
            if not idx:
                continue
            if self.kind == "oracle":
                for i in idx:
                    out[i] = oracle_plan(bridged_list[i])
                continue
            use_world(w, self.meta)
            sym, nb = (MF.SYM, MF.name_bytes) if w == "dispatch" else (CP.SYM, CP.name_bytes)
            specs = [bridged_list[i][1] for i in idx]
            rows = []
            for j in range(0, len(specs), 256):
                rows += self.nets[w].greedy(nb(specs[j:j + 256])).tolist()
            for i, row in zip(idx, rows):
                s = [sym[k] for k in row]
                out[i] = s[:s.index("<eos>") + 1] if "<eos>" in s else s
        return out


class Writer:
    def __init__(self, kind, ckpt=None):
        self.kind = kind
        if kind == "net":
            meta = load_meta(ckpt)
            self.net = MF.Writer(meta["writer_dim"])
            self.net.load_state_dict(torch.load(os.path.join(ckpt, "writer.pt")))
            self.net.eval()

    def texts(self, syms):
        syms = sorted({s for s in syms if s in MF.SYM[3:]})
        if self.kind == "oracle" or not syms:
            return {s: MF.block_text(s) for s in syms}
        return dict(zip(syms, self.net.write(torch.tensor([MF.SYM.index(s) for s in syms]))))

# ----------------------------------------------------------------------------------------------- understander


U_SYS = """You turn a request written in Portuguese into a JSON spec for a Python code generator. Never write code.
Operations (id: meaning):
add: a + b | sub: a - b | mul: a * b | div: a / b | mod: remainder of a / b | maxx: larger of a and b
sqrt: square root of a | neg: -a | abs: absolute value of a | square: a squared | pow: a to the power b
fact: factorial of a | hyp: hypotenuse with legs a and b | avg: average of a and b | pct: a percent of b
Two kinds of request:
1. a calculator offering operations -> {"main": "dispatch", "calls": [ids in the order asked]}
2. one formula returned by main(a, b) -> {"main": "expr", "expr": "<nested calls>"}; only add sub mul div mod maxx
   hyp avg pct, each with exactly two arguments; the leaves are a and b.
Answer with the JSON only."""
U_SHOTS = [("crie uma calculadora que faça soma e raiz quadrada", {"main": "dispatch", "calls": ["add", "sqrt"]}),
           ("quero um programa com fatorial, média de dois números e subtração",
            {"main": "dispatch", "calls": ["fact", "avg", "sub"]}),
           ("crie uma função main(a, b) que retorna a soma de a e b", {"main": "expr", "expr": "add(a, b)"}),
           ("faça um programa cuja main(a, b) devolve a média de (a vezes b) e b",
            {"main": "expr", "expr": "avg(mul(a, b), b)"})]
SPEC_SCHEMA = {"type": "object", "properties": {"main": {"type": "string", "enum": ["dispatch", "expr"]},
                                                "calls": {"type": "array", "items": {"type": "string"}},
                                                "expr": {"type": "string"}}, "required": ["main"]}


def chat(system, turns, user):
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    for u, a in turns:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    return s + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def llama(url, prompt, n_predict, schema=None):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "cache_prompt": False,
            "stop": ["<|im_end|>"]}
    if schema:
        body["json_schema"] = schema
    rq = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=600) as f:
        return json.loads(f.read())["content"]


def understand_qwen(text, url):
    return llama(url, chat(U_SYS, [(u, json.dumps(a)) for u, a in U_SHOTS], text), 160, SPEC_SCHEMA)


def sendback_msg(flags):
    """validator flags aimed at the understander -> one corrective message; uses no gold."""
    L = []
    for f in flags:
        if f["to"] != "understander":
            continue
        if f["kind"] == "oov_op":
            L.append(f'"{f["fn"]}" is not an operation id; use only the listed ids (for example maxx, not max).')
        elif f["kind"] == "expr_parse_error" and f["fn"].startswith("step") and f["fn"][4:].isdigit():
            L.append(f'{f["fn"].replace("step", "step s")} is wrong: each step is "sK = op(x, y)" numbered from s1, '
                     "x and y are a, b or an earlier step, and every step except the last is used by a later one.")
        elif f["kind"] == "expr_parse_error":
            L.append("expr must be nested calls of add sub mul div mod maxx hyp avg pct, each with exactly two "
                     "arguments, leaves a and b only; no infix operators, no numbers, no other names. Keep the kind "
                     "the request asks: main(a, b) returning one formula -> expr; a calculator offering operations "
                     "-> dispatch.")
        elif f["kind"] in ("json_error", "schema_error"):
            L.append("answer with one JSON object of one of the two kinds.")
        elif f["kind"] == "empty_spec":
            L.append("calls must list at least one operation id.")
    return " ".join(dict.fromkeys(L))


def understand_sendback(text, url, rounds):
    """understander + send-back channel: re-ask while a validator rejects the spec, at most `rounds` times."""
    turns = [(u, json.dumps(a)) for u, a in U_SHOTS]
    raw, hist = llama(url, chat(U_SYS, turns, text), 160, SPEC_SCHEMA), []
    prompt_user = text
    for _ in range(rounds):
        msg = sendback_msg(check_spec(raw)[2])
        if not msg:
            break
        hist.append({"raw": raw, "msg": msg})
        turns = turns + [(prompt_user, raw)]
        prompt_user = f"The validator rejected that spec: {msg} Answer again for the same request: {text}"
        raw = llama(url, chat(U_SYS, turns, prompt_user), 160, SPEC_SCHEMA)
    return raw, hist

# ----------------------------------------------------------------------------------------------- fault injection


FAULTS = [("understander", "drop_op"), ("understander", "swap_op"), ("understander", "add_op"),
          ("planner", "drop_def"), ("planner", "drop_disp"), ("planner", "swap_disp"), ("planner", "dup_def"),
          ("writer", "sign_flip"), ("writer", "const_change"), ("writer", "drop_guard"), ("writer", "call_swap")]


def subtree_end(prefix, i):
    need, j = 1, i
    while need:
        need += 1 if prefix[j] not in ("a", "b") else -1
        j += 1
    return j


def inject_spec(kind, b, r):
    w, t = b
    t = list(t)
    if w == "dispatch":
        others = [o for o in OPS if o not in t]
        if kind == "drop_op" and len(t) >= 2:
            del t[r.randrange(len(t))]
        elif kind == "swap_op":
            t[r.randrange(len(t))] = r.choice(others)
        elif kind == "add_op":
            t.append(r.choice(others))
        else:
            return None
        return (w, tuple(t))
    ops = [i for i, x in enumerate(t) if x not in ("a", "b")]
    if not ops:
        return None
    i = r.choice(ops)
    if kind == "drop_op":
        e1 = subtree_end(t, i + 1)
        t = t[:i] + t[i + 1:e1] + t[subtree_end(t, i):]
    elif kind == "swap_op":
        t[i] = r.choice([o for o in BIN if o != t[i]])
    else:
        return None
    return (w, tuple(t))


def inject_plan(kind, world, plan, r):
    p = list(plan)
    defs = [i for i, s in enumerate(p) if s.startswith("def:")]
    if kind == "drop_def" and defs:
        del p[r.choice(defs)]
    elif kind == "drop_disp" and world == "dispatch" and [s for s in p if s.startswith("disp:")]:
        del p[r.choice([i for i, s in enumerate(p) if s.startswith("disp:")])]
    elif kind == "swap_disp":
        pre = "disp:" if world == "dispatch" else "call:"
        idx = [i for i, s in enumerate(p) if s.startswith(pre)]
        if not idx:
            return None
        i = r.choice(idx)
        pool = [o for o in (OPS if world == "dispatch" else BIN) if pre + o not in p]
        if not pool:
            return None
        p[i] = pre + r.choice(pool)
    elif kind == "dup_def" and defs:
        i = r.choice(defs)
        p.insert(i + 1, p[i])
    else:
        return None
    return p


def inject_text(kind, text, r):
    head, body = text.split("\n", 1) if "\n" in text else (text, "")
    if kind == "sign_flip":
        for x, y in ((" + ", " - "), (" - ", " + "), (" * ", " / ")):
            if x in body:
                return head + "\n" + body.replace(x, y, 1)
    elif kind == "const_change":
        for x, y in (("100", "10"), (", 2)", ", 3)"), ("range(2,", "range(3,"), ("** 0.5", "** 0.25"),
                     ("r = 1", "r = 2"), ("sub(0,", "sub(1,")):
            if x in body:
                return head + "\n" + body.replace(x, y, 1)
    elif kind == "drop_guard":
        for g in ("    if b == 0:\n        return None\n", "    if a < 0:\n        return None\n"):
            if g in body:
                return head + "\n" + body.replace(g, "", 1)
    elif kind == "call_swap":
        m = re.search(r"\b(add|sub|mul|div|sqrt|square|neg)\(", body)
        if m:
            new = r.choice([o for o in ("add", "sub", "mul", "div") if o != m.group(1)])
            return head + "\n" + body[:m.start()] + new + body[m.start() + len(m.group(1)):]
    return None


def rng_for(*parts):
    """stable per-request RNG (built-in hash() of str changes per process)."""
    return random.Random(int(hashlib.md5(repr(parts).encode()).hexdigest(), 16) % 2 ** 31)


def pick_fault(i, round_=0):
    return FAULTS[(i + 5 * round_) % len(FAULTS)]

# ----------------------------------------------------------------------------------------------- the belt


def called_names(src):
    """function names called inside function bodies, and names defined, of a module (None if it does not parse)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None, None
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return defined, called


def def_name_ok(text, o):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False, "parse_error"
    fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    return (fns == [o], "" if fns == [o] else "wrong_name:" + ",".join(fns))


def run_belt(reqs, raws, planner, writer, injects=None, overrides=None, lat_u=None):
    """reqs, raw understander outputs -> traces. injects[i] = (stage, kind) or None.
    overrides[i] = {"skip": stage, "plan": [...], "texts": {sym: text}} (fix verification)."""
    n = len(reqs)
    injects = injects or [None] * n
    overrides = overrides or [None] * n
    T = [{"id": q["id"], "world": q["world"], "split": q["split"], "text": q["text"], "caps": q["caps"],
          "raw_spec": raws[i], "inject": None, "lat_ms": {"understander": (lat_u or [0] * n)[i]}}
         for i, q in enumerate(reqs)]
    bridged, gold_b = [], [gold_bridged(q) for q in reqs]
    for i, q in enumerate(reqs):
        s, b, fl = check_spec(raws[i])
        inj, ov = injects[i], overrides[i] or {}
        if b and inj and inj[0] == "understander" and ov.get("skip") != "understander":
            nb = inject_spec(inj[1], b, rng_for(q["id"], inj[1]))
            if nb:
                b, T[i]["inject"] = nb, {"stage": inj[0], "kind": inj[1], "fn": ""}
                s = spec_of(b)
        T[i].update(spec=s, bridged=list(b) if b else None, flags=fl)
        bridged.append(b)
    # planner: predicted spec (chained) and gold spec (isolated / CF_U)
    t0 = time.perf_counter()
    todo = [b for b in bridged if b] + gold_b
    plans_all = planner.plan(todo) if todo else []
    plans, k = [], 0
    for b in bridged:
        plans.append(plans_all[k] if b else [])
        k += 1 if b else 0
    gplans = plans_all[k:]
    for i in range(n):
        T[i]["lat_ms"]["planner"] = round(1000 * (time.perf_counter() - t0) / n, 3)
    for i, q in enumerate(reqs):
        inj, ov = injects[i], overrides[i] or {}
        if "plan" in (ov or {}):
            plans[i] = ov["plan"]
        elif bridged[i] and inj and inj[0] == "planner":
            p = inject_plan(inj[1], bridged[i][0], plans[i], rng_for(q["id"], inj[1]))
            if p is not None:
                plans[i], T[i]["inject"] = p, {"stage": "planner", "kind": inj[1], "fn": ""}
                gp = inject_plan(inj[1], gold_b[i][0], gplans[i], rng_for(q["id"], inj[1], "gold"))
                gplans[i] = gp if gp is not None else gplans[i]
    # writer: one pass over every symbol any module needs
    t0 = time.perf_counter()
    need = {s for p in plans + gplans for s in p} | {s for b in bridged if b for s in oracle_plan(b)}
    NT = writer.texts(need)
    for i in range(n):
        T[i]["lat_ms"]["writer"] = round(1000 * (time.perf_counter() - t0) / n, 3)
    wt = []
    for i, q in enumerate(reqs):
        inj, ov = injects[i], overrides[i] or {}
        own = {}
        if inj and inj[0] == "writer":                   # a writer fix overrides one text below, the fault stays
            r = rng_for(q["id"], inj[1])
            cands = list(dict.fromkeys(s for s in plans[i] if s.startswith("def:")))
            r.shuffle(cands)
            for s in cands:
                nt = inject_text(inj[1], NT[s], r)
                if nt is not None:
                    own[s], T[i]["inject"] = nt, {"stage": "writer", "kind": inj[1], "fn": s[4:]}
                    break
        own.update(ov.get("texts", {}))
        wt.append(lambda s, own=own: own.get(s, NT.get(s, "")))
    # modules
    B, H, fn_cache = Batch(), [], {}
    for i, q in enumerate(reqs):
        b, gb, plan = bridged[i], gold_b[i], plans[i]
        pw = b[0] if b else q["world"]
        kg = [f"d:{o}" for o in q["caps"]] if q["world"] == "dispatch" else ["x:"]
        ks = ([f"d:{o}" for o in b[1]] if b[0] == "dispatch" else ["x:"]) if b else []
        src = assemble(pw, plan, wt[i])
        disp_ops = re.findall(r'op == "(\w+)"', src)
        kj = ([f"jd:{o}" for o in dict.fromkeys(list(b[1] if b and b[0] == "dispatch" else []) + disp_ops)]
              if pw == "dispatch" else ["jx:"])
        h = {"E": B.add(src, kg + ks + kj), "src": src, "kg": kg, "ks": ks, "kj": kj, "pw": pw}
        if q["world"] == "expr":
            h["G"] = B.add(assemble("expr", oracle_plan(gb), MF.block_text), ["x:"])
        if b:
            h["REF_S"] = B.add(assemble(pw, oracle_plan(b), MF.block_text), kg + ks)
            h["CF_W"] = B.add(assemble(pw, plan, MF.block_text), kg + ks)
            h["CF_P"] = B.add(assemble(pw, oracle_plan(b), wt[i]), kg)
        h["CF_U"] = B.add(assemble(gb[0], gplans[i], wt[i]), kg)
        h["CF_UW"] = B.add(assemble(gb[0], gplans[i], MF.block_text), kg)
        h["fn"] = {}
        for s in dict.fromkeys(x for x in plan if x.startswith(("def:", "disp:"))):
            k, o = s.split(":")
            if k == "def":
                key = (s, wt[i](s))
                if key not in fn_cache:
                    fn_cache[key] = B.add(ORACLE_DEFS + wt[i](s) + "def _end():\n    pass\n", [f"f:{x}" for x in OPS])
                h["fn"][s] = fn_cache[key]
            else:
                h["fn"][s] = B.add(ORACLE_DEFS + "def main(op, a, b=0):\n" + wt[i](s) + "    return None\n", [f"d:{o}"])
        H.append(h)
    t0 = time.perf_counter()
    B.run()
    lat_x = round(1000 * (time.perf_counter() - t0) / max(1, n), 3)
    R = ref_all()
    for i, q in enumerate(reqs):
        t, h, b, plan = T[i], H[i], bridged[i], plans[i]
        gold = (lambda k: R[k]) if q["world"] == "dispatch" else (lambda k, g=B[h["G"]]: g[k])
        E = B[h["E"]]
        alt = q["world"] == "dispatch" and len(q["caps"]) == 1 and b is not None and b[0] == "expr"

        def cap_status(out, src=None):
            st = {}
            for cap, k in zip(q["caps"], h["kg"]):
                if out[k] is not None and out[k] == gold(k):
                    st[cap] = "correct"
                elif alt and out.get("x:") is not None and out["x:"] == R[f"f:{cap}"]:
                    st[cap] = "correct"                  # one-op request answered as main(a, b) = op(a, b)
                elif src is not None and q["world"] == "dispatch" and f'op == "{cap}"' not in src:
                    st[cap] = "missing"
                else:
                    st[cap] = "wrong"
            return st
        cs = cap_status(E, h["src"])
        u_caps = cap_status(B[h["REF_S"]]) if b else {c: "missing" for c in q["caps"]}
        u_ok = all(v == "correct" for v in u_caps.values())
        p_ok, p_bad = None, []
        if b:
            rs, cw = B[h["REF_S"]], B[h["CF_W"]]
            p_bad = [k for k in h["ks"] if cw[k] is None or cw[k] != rs[k]]
            p_ok = not p_bad
        w_bad = []
        for s, hh in h["fn"].items():
            k, o = s.split(":")
            keys = [f"f:{x}" for x in OPS] if k == "def" else [f"d:{o}"]
            ok = all(B[hh][x] is not None and B[hh][x] == R[x] for x in keys)
            if k == "def":
                nok, why = def_name_ok(wt[i](s), o)
                ok = ok and nok
                if not nok:
                    t["flags"].append(flag("validator", "writer", why.split(":")[0], o, why))
            if not ok:
                w_bad.append(o if k == "def" else s)
        w_ok = not w_bad
        end_pass = all(v == "correct" for v in cs.values())
        broken = [st for st, ok in (("understander", u_ok), ("planner", p_ok is not False), ("writer", w_ok)) if not ok]
        root = None if end_pass else (broken[0] if broken else "unexplained")
        allc = (lambda out: all(v == "correct" for v in cap_status(out).values()))
        cf = {"u": allc(B[h["CF_U"]]), "ref_s": allc(B[h["REF_S"]]) if b else False,
              "p": allc(B[h["CF_P"]]) if b else False, "w": allc(B[h["CF_W"]]) if b else False}
        # machine validators that need no gold
        if b and b[0] == "dispatch":
            for o in b[1]:
                if f"disp:{o}" not in plan:
                    t["flags"].append(flag("validator", "planner", "missing_disp", o))
            for s in plan:
                if s.startswith("disp:") and s[5:] not in b[1]:
                    t["flags"].append(flag("validator", "planner", "extra_disp", s[5:]))
        if b and b[0] == "expr" and CP.render_expr(plan) != expr_of(b[1]):
            t["flags"].append(flag("validator", "planner", "tree_mismatch", "", CP.render_expr(plan)))
        if b and "<eos>" not in plan:
            t["flags"].append(flag("validator", "planner", "no_eos"))
        defined, called = called_names(h["src"])
        if defined is None:
            t["flags"].append(flag("validator", "unknown", "module_parse_error"))
        else:
            for y in sorted(called - defined - BUILTIN_NAMES):
                t["flags"].append(flag("writer", "planner", "undefined_helper", y))
        notes = []
        for k in h["ks"]:
            bad = [v for v in (E[k] or []) if v.startswith("!")]
            if E[k] is None:
                t["flags"].append(flag("exec", "unknown", "module_error", k))
                break
            if bad:
                t["flags"].append(flag("exec", "unknown", "exec_error", k, bad[0]))
                if E[k] == B[h["REF_S"]][k]:              # the requested formula itself raises (None into +)
                    notes.append(f"inherent_exception:{k}")
        for f in t["flags"]:
            f["true"] = f["to"] in broken
        t.update(plan=plan, plan_oracle=oracle_plan(b) if b else [], plan_on_gold=gplans[i],
                 texts={s: wt[i](s) for s in dict.fromkeys(plan) if s in MF.SYM[3:]}, src=h["src"],
                 facts={k: E[k] for k in h["kj"]}, cap_status=cs,
                 score=round(sum(v == "correct" for v in cs.values()) / len(cs), 4), end_pass=end_pass,
                 u_ok=u_ok, u_caps=u_caps, p_ok=p_ok, p_bad=p_bad,
                 p_iso_ok=allc(B[h["CF_UW"]]), w_ok=w_ok, w_bad=w_bad, broken=broken, root=root, cf=cf,
                 oracle_notes=notes)
        t["lat_ms"]["exec"] = lat_x
    return T


def summarize(T):
    n = len(T)
    rate = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
    s = {"n": n, "pass": rate([t["end_pass"] for t in T]), "score": rate([t["score"] for t in T]),
         "u_ok": rate([t["u_ok"] for t in T]),
         "p_ok_chained": rate([t["p_ok"] for t in T if t["p_ok"] is not None]),
         "p_ok_isolated": rate([t["p_iso_ok"] for t in T]),
         "w_ok_chained": rate([t["w_ok"] for t in T]),
         "root": {}, "by_split": {}, "flags": {}, "cf_repair_on_fail": {}, "lat_ms": {}}
    s["product_of_stages"] = round(s["u_ok"] * (s["p_ok_chained"] or 0) * s["w_ok_chained"], 4)
    for t in T:
        if t["root"]:
            s["root"][t["root"]] = s["root"].get(t["root"], 0) + 1
        sp = s["by_split"].setdefault(t["split"], {"n": 0, "pass": 0})
        sp["n"] += 1
        sp["pass"] += t["end_pass"]
        for f in t["flags"]:
            fk = s["flags"].setdefault(f"{f['to']}:{f['kind']}", {"n": 0, "true": 0})
            fk["n"] += 1
            fk["true"] += f["true"]
    fails = [t for t in T if not t["end_pass"]]
    for k in ("u", "p", "w", "ref_s"):
        s["cf_repair_on_fail"][k] = rate([t["cf"][k] for t in fails])
    for k in ("understander", "planner", "writer", "exec"):
        s["lat_ms"][k] = rate([t["lat_ms"].get(k, 0) for t in T])
    inj = {}
    for t in T:
        if t["inject"]:
            e = inj.setdefault(f"{t['inject']['stage']}:{t['inject']['kind']}",
                               {"n": 0, "visible": 0, "root_match": 0, "fn_match": 0, "stage_broken": 0})
            e["n"] += 1
            e["visible"] += not t["end_pass"]
            e["root_match"] += t["root"] == t["inject"]["stage"]
            e["stage_broken"] += t["inject"]["stage"] in t["broken"]
            e["fn_match"] += bool(t["inject"]["fn"]) and t["inject"]["fn"] in t["w_bad"]
    s["inject"] = inj
    return s

# ----------------------------------------------------------------------------------------------- subcommands


def load_requests(out):
    p = os.path.join(out, "requests.json")
    if not os.path.exists(p):
        json.dump(make_requests(), open(p, "w"), indent=1, ensure_ascii=False)
    return json.load(open(p))


def cmd_check(a):
    os.makedirs(a.out, exist_ok=True)
    reqs = load_requests(a.out)
    raws = [json.dumps(q["gold"]) for q in reqs]
    P, W = Planner("oracle"), Writer("oracle")
    T0 = run_belt(reqs, raws, P, W)
    bad0 = [t["id"] for t in T0 if not t["end_pass"] or t["broken"] or [f for f in t["flags"] if f["to"] in STAGES]]
    rows = [summarize(T0)]
    T1 = []
    for rnd in range(2):
        T1 += run_belt(reqs, raws, P, W, injects=[pick_fault(i, rnd) for i in range(len(reqs))])
    s1 = summarize(T1)
    wrong_root = [(t["id"], t["inject"], t["root"]) for t in T1 if t["inject"] and not t["end_pass"]
                  and t["root"] != t["inject"]["stage"]]
    unexplained = [t["id"] for t in T1 if t["root"] == "unexplained"]
    res = {"clean": rows[0], "clean_bad_ids": bad0, "inject": s1["inject"], "inject_root": s1["root"],
           "inject_flags": s1["flags"], "wrong_root": wrong_root[:20], "n_wrong_root": len(wrong_root),
           "unexplained": unexplained[:20],
           "verdict": "CHECK PASS" if not bad0 and not wrong_root and not unexplained else "CHECK FAIL"}
    json.dump(res, open(os.path.join(a.out, "check.json"), "w"), indent=1)
    with open(os.path.join(a.out, "traces_check_inject.jsonl"), "w") as f:
        for t in T1:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


def cmd_train(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.ckpt, exist_ok=True)
    md, cd = MF.Data(), CP.Data()
    meta = {"seed": a.seed, "steps": a.steps, "writer_dim": a.writer_dim, "plan_format": MF.PLAN_FORMAT,
            "dispatch_max_out": MF.MAX_OUT,
            "compose_max_out": max(len(CP.plan_of(s)) for k in cd.sets for s in cd.sets[k]) + 4}
    t0 = time.perf_counter()
    use_world("dispatch", meta)
    dp = MF.train_planner(md, a.seed, a.steps)
    writer, _ = MF.train_writer(md, a.writer_dim, a.seed)
    use_world("expr", meta)
    cp = CP.train_planner(cd, a.seed, a.steps)
    meta["train_wall_s"] = round(time.perf_counter() - t0, 1)
    torch.save(dp.state_dict(), os.path.join(a.ckpt, "planner_dispatch.pt"))
    torch.save(cp.state_dict(), os.path.join(a.ckpt, "planner_expr.pt"))
    torch.save(writer.state_dict(), os.path.join(a.ckpt, "writer.pt"))
    strip = (lambda ev: {k: {m: x for m, x in v.items() if not m.startswith("example")} for k, v in ev.items()})
    use_world("dispatch", meta)
    meta["eval_dispatch"] = strip(MF.evaluate(dp, writer, md, ["test_comb", "len4", "len8", "full"]))
    use_world("expr", meta)
    meta["eval_expr"] = strip(CP.evaluate(cp, writer, cd, ["test_same", "calls4", "calls8"]))
    json.dump(meta, open(os.path.join(a.ckpt, "meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


def cmd_run(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    reqs = load_requests(a.out)
    if a.limit:
        reqs = reqs[:a.limit]
    raws, lat, hists = [], [], []
    F = json.load(open(a.raws_file)) if a.understander == "file" else None   # {id: {"raw", "ms"}} from another model
    for q in reqs:
        t0, hist = time.perf_counter(), []
        try:
            if a.understander == "gold":
                raw = json.dumps(q["gold"])
            elif a.understander == "file":
                raw = F[q["id"]]["raw"]
            else:
                raw, hist = understand_sendback(q["text"], a.url, a.sendback)
        except OSError as e:                             # server down: recorded as an understander json_error
            raw = f"<error {e}>"
        raws.append(raw)
        hists.append(hist)
        lat.append(F[q["id"]]["ms"] if F else round(1000 * (time.perf_counter() - t0), 1))
    P, W = Planner(a.planner, a.ckpt), Writer(a.writer, a.ckpt)
    injects = [pick_fault(i) for i in range(len(reqs))] if a.inject == "rotate" else None
    T = run_belt(reqs, raws, P, W, injects=injects, lat_u=lat)
    for t, hist in zip(T, hists):
        t["sendback"], t["belt"] = hist, {"planner": a.planner, "writer": a.writer}
    s = summarize(T)
    sb = [t for t in T if t["sendback"]]
    s["sendback"] = {"rounds": a.sendback, "triggered": len(sb), "pass_after": sum(t["end_pass"] for t in sb),
                     "still_flagged": sum(any(f["to"] == "understander" for f in t["flags"]) for t in sb)}
    s["config"] = {k: getattr(a, k) for k in ("understander", "url", "planner", "writer", "ckpt", "inject", "limit",
                                              "sendback", "raws_file")}
    with open(os.path.join(a.out, f"traces_{a.tag}.jsonl"), "w") as f:
        for t in T:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    json.dump(s, open(os.path.join(a.out, f"summary_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(s, ensure_ascii=False), flush=True)

# ----------------------------------------------------------------------------------------------- judges


VERDICT_SCHEMA = {"type": "object", "properties": {
    "capabilities": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string", "enum": list(OPS) + ["formula"]},
        "status": {"type": "string", "enum": ["correct", "wrong", "missing", "extra"]}},
        "required": ["name", "status"]}, "maxItems": 16},
    "blame": {"type": "array", "items": {"type": "object", "properties": {
        "stage": {"type": "string", "enum": STAGES}, "function": {"type": "string"}, "evidence": {"type": "string"}},
        "required": ["stage", "function", "evidence"]}, "maxItems": 4},
    "fix": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {
        "stage": {"type": "string", "enum": STAGES[:3]}, "function": {"type": "string"},
        "replacement": {"type": "string"}}, "required": ["stage", "function", "replacement"]}]},
    "justification": {"type": "string"}}, "required": ["capabilities", "blame", "fix", "justification"]}


def rubric_text():
    return open(RUBRIC_PATH).read()


def evidence(t):
    L = [f"REQUEST: {t['text']}",                   # steps specs are shown as the nested expr the rubric documents
         f"SPEC (understander output): {json.dumps(canon_spec(t['spec']), ensure_ascii=False)}",
         f"PLAN (planner output): {' '.join(t['plan']) if t['plan'] else '(none)'}", "CODE (assembled):", t["src"],
         "EXECUTION (bwrap facts: what the code returned; no expected values are given):"]
    for k, vals in t["facts"].items():
        kind, _, o = k.partition(":")
        for (x, y), v in zip(FACT_CASES, vals or ["<module failed to load>"] * len(FACT_CASES)):
            L.append(f'  main("{o}", {x}, {y}) -> {v}' if kind == "jd" else f"  main({x}, {y}) -> {v}")
    fl = [f"{f['kind']}({f['fn']}) noticed by {f['from']}, points at {f['to']}" for f in t["flags"]]
    L.append("MACHINE FLAGS: " + ("; ".join(fl) if fl else "none"))
    return "\n".join(L)


def judge_qwen(t, url):
    sys_ = rubric_text() + "\n\nYou are the FIRST judge. Answer with the JSON object of section 2 only."
    raw = llama(url, chat(sys_, [], evidence(t)), 700, VERDICT_SCHEMA)
    try:
        return json.loads(raw), raw
    except json.JSONDecodeError:
        return None, raw


def last_json(text):
    dec, best, i = json.JSONDecoder(), None, 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, j = dec.raw_decode(text, i)
                best, i = obj, j
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return best


def judge_sonnet(t, qv, config_dir, cwd, budget):
    first = (json.dumps(qv, ensure_ascii=False) if qv is not None else
             "(none: the first judge produced no valid JSON, so qwen_ok is false and `corrected` must be the full "
             "verdict)")
    prompt = ("You are the FINAL judge of a code-generation belt. Follow this rule exactly; it is the truth you "
              "judge by.\n\n" + rubric_text() + "\n\nEVIDENCE:\n" + evidence(t) + "\n\nFIRST JUDGE VERDICT:\n" +
              first + "\n\nAnswer with the JSON object of section 4 only, no other text.")
    env = dict(os.environ)
    if config_dir:
        env["CLAUDE_CONFIG_DIR"] = config_dir
    for attempt in range(3):
        if budget["left"] <= 0:
            return None, "budget"
        budget["left"] -= 1
        try:
            p = subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", "", "--output-format", "json"],
                               input=prompt,  # no tools: the judge cannot open requests.json (gold) next to its cwd
                               capture_output=True, text=True, timeout=300, env=env, cwd=cwd)
            text = json.loads(p.stdout).get("result", "")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            text = f"<error {type(e).__name__}>"
        if "operation blocked by hook" in text[:300]:
            continue
        obj = last_json(text)
        if isinstance(obj, dict) and "qwen_ok" in obj:
            return obj, text[-2000:]
    return None, text[-2000:]


def grade_verdict(v, t):
    """agreement of a verdict with the oracle truth of trace t; the score is computed here, not by the judge."""
    if not isinstance(v, dict):
        return {"valid": False, "pass_agree": False, "cap_agree": 0.0, "root_agree": False if not t["end_pass"] else None,
                "fn_agree": None, "false_alarm": None, "fully_ok": False, "score": None}
    caps = {}
    for c in v.get("capabilities") or []:
        if isinstance(c, dict) and isinstance(c.get("name"), str):
            caps[c["name"].strip().lower()] = c.get("status")
    listed = [s for s in caps.values() if s != "extra"]
    score = sum(s == "correct" for s in listed) / len(listed) if listed else 0.0
    jpass = bool(listed) and score == 1.0
    ag = [name in caps and (caps[name] == "correct") == (st == "correct") for name, st in t["cap_status"].items()]
    blame = [x for x in (v.get("blame") or []) if isinstance(x, dict)]
    root_ok = None if t["end_pass"] or t["root"] == "unexplained" else bool(blame) and blame[0].get("stage") == t["root"]
    fname = (blame[0].get("function") or "").strip() if blame else ""
    fn_ok = None if t["root"] != "writer" else bool(blame) and (
        fname in t["w_bad"] or (fname == "main" and any(x.startswith("disp:") for x in t["w_bad"])))
    fa = (not jpass or bool(blame)) if t["end_pass"] else None
    return {"valid": True, "score": round(score, 4), "pass_agree": jpass == t["end_pass"],
            "cap_agree": round(sum(ag) / len(ag), 4), "root_agree": root_ok, "fn_agree": fn_ok, "false_alarm": fa,
            "fully_ok": jpass == t["end_pass"] and all(ag) and root_ok in (None, True)}


def machine_contradiction(v, t):
    """escalation trigger that needs no gold: verdict says correct where the facts show an exception, or no blame
    although a validator raised a flag, or an invalid verdict."""
    if not isinstance(v, dict):
        return "invalid"
    caps = {str(c.get("name")).strip().lower(): c.get("status") for c in v.get("capabilities") or []
            if isinstance(c, dict)}
    for k, vals in t["facts"].items():
        o = k.partition(":")[2] or "formula"
        if caps.get(o) == "correct" and (vals is None or any(str(x).startswith("!") for x in vals)):
            return f"correct_but_exception:{o}"
    if t["flags"] and not v.get("blame"):
        return "flags_but_no_blame"
    return ""


def verify_fix(fix, t, req, planner, writer):
    """apply one proposed fix and rerun the belt; accepted only if execution confirms it (anti false signal)."""
    if not isinstance(fix, dict) or fix.get("stage") not in STAGES[:3] or not isinstance(fix.get("replacement"), str):
        return {"applied": False, "why": "no_fix"}
    st, rep = fix["stage"], fix["replacement"]
    if st != "understander" and not t["bridged"]:
        return {"applied": False, "why": "no_spec_to_plan"}
    raw, ov = t["raw_spec"], {"skip": st} if st == "understander" else {}
    if st == "understander":
        raw = rep
    elif st == "planner":
        sym = MF.SYM if (t["bridged"] or [req["world"]])[0] == "dispatch" else CP.SYM
        p = [x for x in rep.replace(",", " ").split() if x in sym]
        ov["plan"] = p if p and p[-1] == "<eos>" else p + ["<eos>"]
    else:
        fn = (fix.get("function") or "").strip()
        if f"def:{fn}" not in MF.SYM:
            return {"applied": False, "why": "unknown_function"}
        ov["texts"] = {f"def:{fn}": rep if rep.endswith("\n") else rep + "\n"}
    inj = (t["inject"]["stage"], t["inject"]["kind"]) if t["inject"] else None
    t0 = run_belt([req], [t["raw_spec"]], planner, writer, injects=[inj])[0]
    if (t0["score"], t0["plan"], t0["root"]) != (t["score"], t["plan"], t["root"]):
        return {"applied": False, "why": "unstable_baseline"}
    t2 = run_belt([req], [raw], planner, writer, injects=[inj], overrides=[ov])[0]
    regress = [c for c, s in t0["cap_status"].items() if s == "correct" and t2["cap_status"][c] != "correct"]
    right_stage = st == t0["root"] and st not in t2["broken"]
    return {"applied": True, "stage": st, "score_before": t0["score"], "score_after": t2["score"],
            "repaired": t2["end_pass"], "regress": regress, "right_stage": right_stage,
            "accepted": t2["score"] > t0["score"] and not regress and right_stage,
            "target": {"understander": spec_of(tuple(t2["bridged"])) if t2["bridged"] else None, "planner": t2["plan"],
                       "writer": rep}[st], "input": {"understander": req["text"],
                                                    "planner": bridged_name(t2["bridged"]) if t2["bridged"] else "",
                                                    "writer": f"def:{(fix.get('function') or '').strip()}"}[st]}


def pick_traces(T, n):
    """stratified: failures first (balanced by root), then passes; deterministic."""
    key = (lambda t: hashlib.md5(t["id"].encode() + json.dumps(t["inject"]).encode()).hexdigest())
    fails = sorted([t for t in T if not t["end_pass"]], key=key)
    by = {}
    for t in fails:
        by.setdefault(t["root"], []).append(t)
    out = []
    while len(out) < (2 * n) // 3 and any(by.values()):
        for k in sorted(by):
            if by[k] and len(out) < (2 * n) // 3:
                out.append(by[k].pop(0))
    passes = sorted([t for t in T if t["end_pass"]], key=key)
    out += passes[:n - len(out)]
    rest = [t for k in sorted(by) for t in by[k]]
    return out + rest[:n - len(out)]


def cmd_judge(a):
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    T = [json.loads(l) for l in open(a.traces)]
    reqs = {q["id"]: q for q in load_requests(a.out)}
    belt = T[0].get("belt") or {"planner": "net" if a.ckpt else "oracle", "writer": "net" if a.ckpt else "oracle"}
    P, W = Planner(belt["planner"], a.ckpt), Writer(belt["writer"], a.ckpt)
    sel = pick_traces(T, a.n)
    if a.only_fails:                                     # replay pool: only failed traces need a correction
        sel = [t for t in sel if not t["end_pass"]]
    cwd = os.path.join(a.out, "sonnet_cwd")
    os.makedirs(cwd, exist_ok=True)
    budget, rows = {"left": a.max_sonnet_calls}, []
    out_path = os.path.join(a.out, f"judge_{a.tag}.jsonl")
    with open(out_path, "w") as f:
        for t in sel:
            t0 = time.perf_counter()
            qv, qraw = judge_qwen(t, a.qwen_url)
            row = {"id": t["id"], "inject": t["inject"], "truth": {k: t[k] for k in ("end_pass", "score", "root", "w_bad",
                                                                                      "cap_status")},
                   "qwen": qv, "qwen_raw": None if qv else qraw[-500:], "qwen_ms": round(1000 * (time.perf_counter() - t0)),
                   "qwen_grade": grade_verdict(qv, t)}
            req = reqs[t["id"]]
            row["fix_qwen"] = verify_fix((qv or {}).get("fix"), t, req, P, W)
            g, fx = row["qwen_grade"], (qv or {}).get("fix")
            row["qwen_truly_ok"] = bool(g["fully_ok"] and g["fn_agree"] in (None, True)
                                        and (fx is None or row["fix_qwen"].get("accepted")))
            why = ""
            if a.sonnet_mode == "all":
                why = "all"
            elif a.sonnet_mode == "escalate":
                why = machine_contradiction(qv, t) or ("audit" if int(hashlib.md5(t["id"].encode()).hexdigest(), 16) % 5 == 0
                                                       else "")
            row["sonnet_why"], final = why, qv
            if why:
                t0 = time.perf_counter()
                sv, sraw = judge_sonnet(t, qv, a.claude_config_dir, cwd, budget)
                row.update(sonnet=sv, sonnet_raw=None if sv else sraw, sonnet_ms=round(1000 * (time.perf_counter() - t0)))
                if sv and qv is None:
                    sv["qwen_ok"] = False                    # an invalid first verdict can never be "ok"
                if sv and not sv.get("qwen_ok") and isinstance(sv.get("corrected"), dict):
                    final = sv["corrected"]
            row["final"], row["final_grade"] = final, grade_verdict(final, t)
            row["final_is_qwen"] = final is qv
            row["fix_final"] = row["fix_qwen"] if final is qv else verify_fix((final or {}).get("fix"), t, req, P, W)
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"id": t["id"], "root": t["root"], "q": row["qwen_grade"]["fully_ok"], "why": why,
                              "s_ok": (row.get("sonnet") or {}).get("qwen_ok"), "final": row["final_grade"]["fully_ok"],
                              "left": budget["left"]}), flush=True)
    rate = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
    s = {"n": len(rows), "sonnet_calls": a.max_sonnet_calls - budget["left"]}
    for who in ("qwen", "final"):
        g = [r[f"{who}_grade"] for r in rows]
        s[who] = {m: rate([x[m] for x in g if x[m] is not None]) for m in
                  ("valid", "pass_agree", "cap_agree", "root_agree", "fn_agree", "false_alarm", "fully_ok")}
    called = [r for r in rows if r.get("sonnet")]
    s["sonnet"] = {"called": len(called), "valid": len(called),
                   "says_qwen_ok": sum(bool(r["sonnet"].get("qwen_ok")) for r in called),
                   "agrees_with_truth_on_qwen_ok": rate([bool(r["sonnet"].get("qwen_ok")) == r["qwen_truly_ok"]
                                                         for r in called]),
                   "failed_calls": sum(1 for r in rows if r["sonnet_why"] and not r.get("sonnet"))}
    for k in ("fix_qwen", "fix_final"):
        fx = [r[k] for r in rows if k in r and r[k].get("applied")]
        s[k] = {"proposed": len(fx), "accepted": sum(x["accepted"] for x in fx), "repaired": sum(x["repaired"] for x in fx)}
    with open(os.path.join(a.out, f"replay_{a.tag}.jsonl"), "w") as f:
        for r in rows:
            for k, src in (("fix_qwen", "qwen"), ("fix_final", "final")):
                x = r.get(k) if k == "fix_qwen" or not r.get("final_is_qwen") else None
                if x and x.get("accepted"):
                    f.write(json.dumps({"stage": x["stage"], "input": x["input"], "target": x["target"],
                                        "source": src, "trace": r["id"], "repaired": x["repaired"]},
                                       ensure_ascii=False) + "\n")
    json.dump(s, open(os.path.join(a.out, f"judge_summary_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(s), flush=True)


def cmd_one(a):
    torch.set_num_threads(a.threads)
    P, W = Planner("net", a.ckpt), Writer("net", a.ckpt)
    for rep in range(2):                                 # second pass = warm latency
        for text in a.text:
            lat, t0 = {}, time.perf_counter()
            raw, _ = understand_sendback(text, a.url, 2)
            lat["understander"] = time.perf_counter()
            s, b, fl = check_spec(raw)
            if not b:
                print(json.dumps({"text": text, "raw": raw, "flags": fl}, ensure_ascii=False))
                continue
            plan = P.plan([b])[0]
            lat["planner"] = time.perf_counter()
            tx = W.texts(plan)
            lat["writer"] = time.perf_counter()
            src = assemble(b[0], plan, lambda x: tx.get(x, ""))
            keys = [f"jd:{o}" for o in b[1]] if b[0] == "dispatch" else ["jx:"]
            bt = Batch()
            h = bt.add(src, keys)
            facts = bt.run()[h]
            lat["exec"] = time.perf_counter()
            prev, ms = t0, {}
            for k in ("understander", "planner", "writer", "exec"):
                ms[k], prev = round(1000 * (lat[k] - prev), 1), lat[k]
            ms["total"] = round(1000 * (lat["exec"] - t0), 1)
            print(json.dumps({"pass": rep, "text": text, "spec": s, "plan": plan, "facts": facts, "ms": ms},
                             ensure_ascii=False))
            if rep == 1:
                print(src)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c = sub.add_parser("train")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--steps", type=int, default=2000)
    c.add_argument("--writer-dim", type=int, default=64)
    c.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("run")
    c.add_argument("--out", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--understander", choices=["gold", "qwen", "file"], default="gold")
    c.add_argument("--raws-file", default=None, help="understander=file: JSON {id: {raw, ms}}")
    c.add_argument("--url", default="http://127.0.0.1:8141")
    c.add_argument("--planner", choices=["net", "oracle"], default="net")
    c.add_argument("--writer", choices=["net", "oracle"], default="net")
    c.add_argument("--ckpt", default=None)
    c.add_argument("--inject", choices=["none", "rotate"], default="none")
    c.add_argument("--sendback", type=int, default=0, help="max send-back rounds to the understander")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("judge")
    c.add_argument("--out", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--traces", required=True)
    c.add_argument("--qwen-url", default="http://127.0.0.1:8141")
    c.add_argument("--sonnet-mode", choices=["off", "all", "escalate"], default="off")
    c.add_argument("--max-sonnet-calls", type=int, default=0)
    c.add_argument("--claude-config-dir", default=None)
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--only-fails", action="store_true", help="judge only failed traces (pool for #37 replay)")
    c.add_argument("--ckpt", default=None)
    c.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("one")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--url", default="http://127.0.0.1:8141")
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("text", nargs="+")
    a = ap.parse_args()
    {"check": cmd_check, "train": cmd_train, "run": cmd_run, "judge": cmd_judge, "one": cmd_one}[a.cmd](a)


if __name__ == "__main__":
    main()
