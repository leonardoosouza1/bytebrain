"""Host side of the candidate executor: sandboxed runs and output matching."""

import json
import math
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def run_jobs(jobs):
    """Run candidate jobs with no network, read-only /usr, no home, no /media."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
           "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
           "--proc", "/proc", "--dev", "/dev",
           "--ro-bind", os.path.join(HERE, "judge_worker.py"), "/w.py",
           "/usr/bin/python3", "-I", "-S", "/w.py"]
    p = subprocess.run(cmd, input=json.dumps(jobs), capture_output=True, text=True,
                       timeout=60 + sum(1 + 0.6 * len(j["calls"]) for j in jobs))
    if p.returncode != 0:
        raise RuntimeError(f"sandbox failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def run_jobs_isolated(jobs, workers=6, per_job_s=None):
    """One sandbox per job with a hard wall-clock kill.

    run_jobs puts every job in one worker, so a candidate that swallows the SIGALRM
    Timeout (bare `except:` inside a loop) stalls the whole batch until the batch timeout.
    Here such a job alone returns load "timeout".
    """
    from concurrent.futures import ThreadPoolExecutor

    def one_limited(job):
        cmd_timeout = per_job_s or 5 + 0.6 * len(job["calls"])
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
               "--proc", "/proc", "--dev", "/dev",
               "--ro-bind", os.path.join(HERE, "judge_worker.py"), "/w.py",
               "/usr/bin/python3", "-I", "-S", "/w.py"]
        try:
            p = subprocess.run(cmd, input=json.dumps([job]), capture_output=True, text=True, timeout=cmd_timeout)
        except subprocess.TimeoutExpired:
            return {"load": "timeout", "us": 0.0, "out": []}
        if p.returncode != 0:
            return {"load": "crash", "us": 0.0, "out": []}
        return json.loads(p.stdout)[0]

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one_limited, jobs))


def matches(exp, got):
    """exp: ("ok", value) | ("err", name) from calc_ref; got: worker result."""
    if exp[0] == "err":
        return got[0] == "err" and exp[1] in got[1]
    if got[0] != "ok":
        return False
    return _match_value(exp[1], got[1])


def _match_value(e, got):
    kind, v = got
    if isinstance(e, (tuple, list)):
        return kind == "seq" and len(v) == len(e) and all(_match_value(a, b) for a, b in zip(e, v))
    if isinstance(e, str):
        return kind == "str" and v == e
    if isinstance(e, bool):
        return kind == "bool" and v == e
    if kind != "num":
        return False
    if isinstance(e, float) and math.isnan(e):
        return isinstance(v, float) and math.isnan(v)
    return math.isclose(v, e, rel_tol=1e-9, abs_tol=1e-9)


def _sig_value(kind, v):
    if kind == "num":
        return f"n:{float(v):.9g}"
    if kind == "seq":
        return "(" + ",".join(_sig_value(*x) for x in v) + ")"
    return f"{kind}:{v}"


def signature(got):
    """Behaviour fingerprint of one call, for consensus clustering."""
    if got[0] == "ok":
        kind, v = got[1]
        return _sig_value(kind, v)
    if got[0] == "err":
        # coarse: the standard base the error derives from
        for name in ("ZeroDivisionError", "ValueError", "TypeError"):
            if name in got[1]:
                return f"e:{name}"
        return "e:other"
    return "timeout"
