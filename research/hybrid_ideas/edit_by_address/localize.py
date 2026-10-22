"""Failing test -> traceback / return site -> address. Sandboxed with the same bwrap recipe as judge.py.

judge.run_jobs is reused for pass/fail; run_jobs_tb runs our worker_tb.py (line numbers + coverage).
"""

import ast
import json
import math
import os
import subprocess
import sys

import addressing as A
import fixes as F  # also puts codegen_calc on sys.path

import judge  # noqa: E402  (codegen_calc)
import spec  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = {e["name"]: e for e in spec.SPEC}


def run_jobs_tb(jobs):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
           "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
           "--proc", "/proc", "--dev", "/dev",
           "--ro-bind", os.path.join(HERE, "worker_tb.py"), "/w.py",
           "/usr/bin/python3", "-I", "-S", "/w.py"]
    p = subprocess.run(cmd, input=json.dumps(jobs), capture_output=True, text=True,
                       timeout=60 + sum(1 + 0.6 * len(j["calls"]) for j in jobs))
    if p.returncode != 0:
        raise RuntimeError(f"sandbox failed: {p.stderr[-500:]}")
    return json.loads(p.stdout)


def judge_module(source, names=None, seeds=(0, 1)):
    """-> {name: bool} all hidden cases (both seeds) pass, via codegen_calc judge.run_jobs."""
    names = names or [e["name"] for e in spec.SPEC]
    jobs, meta = [], []
    for n in names:
        for s in seeds:
            cases = spec.hidden_cases(SPEC[n], seed=s)
            jobs.append({"code": source, "fname": n, "calls": [list(a) for a in cases]})
            meta.append((n, cases))
    out = judge.run_jobs(jobs)
    res = {n: True for n in names}
    for (n, cases), o in zip(meta, out):
        ok = o["load"] == "ok" and all(judge.matches(spec.expected(SPEC[n], a), g) for a, g in zip(cases, o["out"]))
        res[n] = res[n] and ok
    return res


def _fmt_got(g):
    if g[0] == "ok":
        return repr(g[1][1])
    if g[0] == "err":
        return f"raised {g[1][0]}"
    return "timed out"


def _fmt_exp(e):
    return repr(e[1]) if e[0] == "ok" else f"raise {e[1]}"


def failing_calls(source, name, seed=1):
    """Run the hidden cases of `name` with traceback capture; return failing + passing call records."""
    entry = SPEC[name]
    cases = spec.hidden_cases(entry, seed=seed)
    out = run_jobs_tb([{"code": source, "fname": name, "calls": [list(a) for a in cases]}])[0]
    if out["load"] != "ok":
        return {"load": out["load"], "fail": [], "pass": []}
    fail, ok = [], []
    for a, (g, info) in zip(cases, out["out"]):
        e = spec.expected(entry, a)
        rec = {"args": list(a), "got": g, "exp": list(e), "info": info,
               "call": f"{name}({', '.join(repr(x) for x in a)})",
               "msg": f"{name}({', '.join(repr(x) for x in a)}) -> {_fmt_got(g)}; expected {_fmt_exp(e)}"}
        (ok if judge.matches(e, g) else fail).append(rec)
    return {"load": "ok", "fail": fail, "pass": ok}


def site_line(rec):
    """Address source for one failing call: innermost traceback line, else the return site."""
    tb = rec["info"]["tb"]
    if tb:
        return tb[-1], "traceback"
    if rec["info"]["ret"] is not None:
        return rec["info"]["ret"], "return_site"
    return None, "none"


def ochiai(fail, passed):
    """Spectrum-based suspiciousness per line: ef / sqrt(F * (ef + ep))."""
    nf = len(fail)
    lines = {ln for r in fail + passed for ln in r["info"]["lines"]}
    score = {}
    for ln in lines:
        ef = sum(ln in r["info"]["lines"] for r in fail)
        ep = sum(ln in r["info"]["lines"] for r in passed)
        score[ln] = ef / math.sqrt(nf * (ef + ep)) if ef else 0.0
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def vote_address(source, name, recs, tree=None):
    """Majority site line over failing calls -> locate() result (+ the vote)."""
    tree = tree or ast.parse(source)
    votes = {}
    for r in recs:
        ln, how = site_line(r)
        if ln is not None:
            votes.setdefault((ln, how), []).append(r["call"])
    if not votes:
        return None, votes
    (ln, how), _ = max(votes.items(), key=lambda kv: (len(kv[1]), -kv[0][0]))
    loc = A.locate(source, ln, tree)
    if loc is None:
        return None, votes
    loc["site_line"], loc["site_how"] = ln, how
    return loc, {f"{k[0]}:{k[1]}": v for k, v in votes.items()}
