"""Candidate executor; runs inside bwrap (see judge.py).

stdin: JSON list of jobs {"code": str, "fname": str, "calls": [[arg, ...], ...]}
stdout: JSON list (one per job) of {"load": "ok"|error, "us": float, "out": [result, ...]}
result: ["ok", value] | ["err", [exception mro names]] | ["timeout"]
Only a few pure modules can be imported; eval/exec/open are unavailable.
"""

import builtins
import io
import json
import resource
import signal
import sys
import time

resource.setrlimit(resource.RLIMIT_AS, (1024 * 2**20, 1024 * 2**20))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
resource.setrlimit(resource.RLIMIT_CPU, (240, 240))

ALLOWED = {"math", "cmath", "operator", "functools", "re", "decimal", "fractions", "numbers", "typing",
           "string", "itertools", "collections"}
_real_import = builtins.__import__


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of {name} not allowed")
    return _real_import(name, globals, locals, fromlist, level)


SAFE = {k: v for k, v in vars(builtins).items() if k not in ("eval", "exec", "open", "compile", "input",
                                                             "breakpoint", "exit", "quit", "help")}
SAFE["__import__"] = _import


class Timeout(Exception):
    pass


def on_alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, on_alarm)


def norm(v):
    if isinstance(v, bool):
        return ["bool", v]
    if isinstance(v, (int, float)):
        if isinstance(v, int) and abs(v) > 2**63:
            return ["num", float(v)]
        return ["num", v]
    if isinstance(v, (tuple, list)) and len(v) <= 64:
        return ["seq", [norm(x) for x in v]]
    if isinstance(v, str):
        return ["str", v[:80]]
    return ["other", repr(v)[:80]]


def call(fn, args):
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    try:
        return ["ok", norm(fn(*args))]
    except Timeout:
        return ["timeout"]
    except BaseException as e:  # noqa: BLE001 - any failure is data
        return ["err", [c.__name__ for c in type(e).__mro__]]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def run(job):
    env = {"__builtins__": SAFE, "__name__": "candidate"}
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        exec(compile(job["code"], "<candidate>", "exec"), env)
    except BaseException as e:  # noqa: BLE001
        return {"load": type(e).__name__, "us": 0.0, "out": []}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    fn = env.get(job["fname"])
    if not callable(fn):
        return {"load": "missing", "us": 0.0, "out": []}
    t0 = time.perf_counter()
    out = [call(fn, a) for a in job["calls"]]
    us = (time.perf_counter() - t0) * 1e6 / max(len(job["calls"]), 1)
    return {"load": "ok", "us": us, "out": out}


jobs = json.loads(sys.stdin.read())
real_stdout = sys.stdout
sys.stdout = io.StringIO()  # candidates may print; keep the result channel clean
results = [run(j) for j in jobs]
real_stdout.write(json.dumps(results))
