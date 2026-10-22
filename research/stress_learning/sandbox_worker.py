"""Sandboxed snippet executor (runs inside bwrap, see code_exec_learning.run_sandboxed).

Reads a JSON list of code strings on stdin, writes a JSON list of outputs.
Each snippet runs with a whitelist of builtins (no import, no open, no eval),
a 1 s alarm and an address-space limit. Output is whatever print() produced,
stripped; errors become "ERROR:<type>".
"""

import json
import resource
import signal
import sys

resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (8, 8))
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

SAFE = {name: __builtins__.__dict__[name] if hasattr(__builtins__, "__dict__") else __builtins__[name]
        for name in ("range", "len", "sum", "max", "min", "sorted", "abs", "str", "int",
                     "list", "enumerate", "reversed", "True", "False", "None")}


class Timeout(Exception):
    pass


def on_alarm(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, on_alarm)


def run(code):
    out = []

    def _print(*args, sep=" ", end="\n"):
        out.append(sep.join(str(a) for a in args) + end)

    env = {"__builtins__": {**SAFE, "print": _print}}
    signal.alarm(1)
    try:
        exec(compile(code, "<snippet>", "exec"), env)
    except Timeout:
        return "ERROR:Timeout"
    except Exception as e:  # noqa: BLE001 - any snippet failure is data
        return f"ERROR:{type(e).__name__}"
    finally:
        signal.alarm(0)
    return "".join(out).strip()


codes = json.loads(sys.stdin.read())
sys.stdout.write(json.dumps([run(c) for c in codes]))
