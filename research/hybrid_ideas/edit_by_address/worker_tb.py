"""Candidate executor variant that also reports WHERE a call failed; runs inside bwrap.

Same sandbox contract as codegen_calc/judge_worker.py (restricted imports, no eval/exec/open,
rlimits, per-call alarm), plus per call:
  "tb"      list of candidate line numbers of the traceback (outermost -> innermost),
            only frames whose filename is "<candidate>"
  "ret"     line of the first `return` event in a candidate frame (the deepest return,
            i.e. where a returned value originated); used when there is no exception
  "lines"   sorted executed candidate lines (for spectrum-based localization)

stdin: JSON list of jobs {"code": str, "fname": str, "calls": [[arg, ...], ...]}
stdout: JSON list of {"load": "ok"|error, "out": [[result, info], ...]}
"""

import builtins
import io
import json
import resource
import signal
import sys

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
CAND = "<candidate>"
MAX_LINES_EVENTS = 200000


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
    return ["other", repr(v)[:80]]


def call(fn, args):
    lines, state = set(), {"ret": None, "n": 0}

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != CAND:
            return None
        if event == "line":
            state["n"] += 1
            if state["n"] < MAX_LINES_EVENTS:
                lines.add(frame.f_lineno)
        elif event == "return" and state["ret"] is None and arg is not None:
            state["ret"] = frame.f_lineno
        return tracer

    signal.setitimer(signal.ITIMER_REAL, 0.5)
    tb_lines = []
    sys.settrace(tracer)
    try:
        res = ["ok", norm(fn(*args))]
    except Timeout:
        res = ["timeout"]
    except BaseException as e:  # noqa: BLE001 - any failure is data
        res = ["err", [c.__name__ for c in type(e).__mro__]]
        tb = e.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == CAND:
                tb_lines.append(tb.tb_lineno)
            tb = tb.tb_next
    finally:
        sys.settrace(None)
        signal.setitimer(signal.ITIMER_REAL, 0)
    return [res, {"tb": tb_lines, "ret": state["ret"], "lines": sorted(lines)}]


def run(job):
    env = {"__builtins__": SAFE, "__name__": "candidate"}
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        exec(compile(job["code"], CAND, "exec"), env)
    except BaseException as e:  # noqa: BLE001
        return {"load": type(e).__name__, "out": []}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    fn = env.get(job["fname"])
    if not callable(fn):
        return {"load": "missing", "out": []}
    return {"load": "ok", "out": [call(fn, a) for a in job["calls"]]}


jobs = json.loads(sys.stdin.read())
real_stdout = sys.stdout
sys.stdout = io.StringIO()
results = [run(j) for j in jobs]
real_stdout.write(json.dumps(results))
