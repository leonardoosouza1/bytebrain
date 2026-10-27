"""Grammar and name truth of functions sampled from a byte-LM checkpoint.

For each sample: prompt -> bytes until the first non-blank line at column 0 after the def line (end of the function).
  terminated - the function ended inside the byte budget;
  parses     - ast.parse accepts it (grammar truth);
  names_ok   - every global name it reads is a builtin or a parameter/local (static name truth, symtable);
  runs       - names_ok functions called once in bwrap with fuzz arguments: no NameError/SyntaxError/
               UnboundLocalError (other exceptions on random inputs are allowed; it is a liveness check).
Prompts: "def " (free generation) and the signature lines of held-out functions (completion).

Usage: eval_samples.py CKPT --out FILE.json [--n 256] [--bytes 800] [--temp 0.8]
"""

import argparse
import ast
import builtins
import json
import os
import random
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from fast_gen import CachedGen  # noqa: E402
from battery_knowing import ByteLM  # noqa: E402
from build_corpus import free_names  # noqa: E402

BUILTINS = set(dir(builtins))
HOLDOUT = "/media/llm-data/datasets/pycode/holdout_functions.json"
RUNNER = r'''
import json, sys, random
src, fname, nargs = json.loads(sys.stdin.read())
import io
real_out, sys.stdout = sys.stdout, io.StringIO()     # generated code may print: keep the result line clean
g = {}
try:
    exec(src, g)
except Exception as e:
    real_out.write(json.dumps("load:" + type(e).__name__) + "\n"); sys.exit()
bad = ("NameError", "UnboundLocalError", "SyntaxError")
r = random.Random(0)
pool = [0, 1, 3, -2, 2.5, "ab", "", [1, 2], (1, 2), {"a": 1}, None, True]
res = "ok"
for _ in range(4):
    try:
        g[fname](*[r.choice(pool) for _ in range(nargs)])
    except BaseException as e:
        if type(e).__name__ in bad:
            res = type(e).__name__; break
real_out.write(json.dumps(res) + "\n")
'''


def cut_function(text):
    """-> (function source, terminated)."""
    lines = text.split("\n")
    for i, l in enumerate(lines[1:], 1):
        if l.strip() and not l[0].isspace():
            return "\n".join(lines[:i]).rstrip() + "\n", True
    return text, False


def run_bwrap(src, fname, nargs):
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",  # review #48: no host env in the sandbox
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    try:
        p = subprocess.run(cmd, input=json.dumps([src, fname, nargs]), capture_output=True, text=True, timeout=3)
        return json.loads(p.stdout) if p.returncode == 0 else "crash"
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return "timeout"


def judge(src):
    r = {"parses": False, "names_ok": False, "runs": False}
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return r
    r["parses"] = True
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    fr = free_names(src)
    if fn is None or fr is None or not fr <= BUILTINS:
        return r
    r["names_ok"] = True
    a = fn.args
    if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
        return r
    nargs = len(a.args) - len(a.defaults)
    r["runs"] = run_bwrap(src, fn.name, nargs) == "ok"
    return r


def summarize(rows):
    n = len(rows)
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in ("terminated", "parses", "names_ok", "runs")} | {"n": n}


@torch.no_grad()
def generate_masked(g, prompt, n_seq, n_bytes, temp, start):
    """CachedGen.generate with the GPU syntax automaton in the loop: bytes that would raise an L1 error get
    probability 0 (rows with no legal byte left are sampled unmasked). `start` = prompt bytes fed to the automaton."""
    from gpu_syntax import SyntaxAutomaton
    dev = g.m.device
    sa = SyntaxAutomaton(n_seq, device=dev)
    for c in start:
        sa.step(torch.full((n_seq,), c, device=dev))
    g.reset()
    ids = torch.tensor([list(prompt)], dtype=torch.long, device=dev).repeat(n_seq, 1)
    g.hist = ids.clone()
    logits = g._forward(ids)
    out = torch.zeros(n_seq, n_bytes, dtype=torch.uint8, device=dev)
    for i in range(n_bytes):
        ok = sa.allowed()
        ok |= ~ok.any(1, keepdim=True)
        p = torch.softmax(logits.float() / temp, -1) * ok
        nxt = torch.multinomial(p / p.sum(-1, keepdim=True), 1).squeeze(-1)
        sa.step(nxt)
        out[:, i] = nxt.to(torch.uint8)
        g.hist = torch.cat([g.hist, nxt[:, None]], 1)
        if g.pos >= g.ctx:
            tail = g.hist[:, -g.keep:]
            g.reset()
            g.hist = tail
            logits = g._forward(tail)
        else:
            logits = g._forward(nxt[:, None])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--bytes", type=int, default=800)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--mem-frac", type=float, default=0.06)
    ap.add_argument("--mask", action="store_true", help="GPU syntax automaton masks illegal bytes while generating")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    torch.manual_seed(0)
    m = ByteLM(args.ckpt)
    g = CachedGen(m)
    t0 = time.perf_counter()
    report = {"ckpt": args.ckpt, "step": m.step, "temp": args.temp, "mask": args.mask}
    gen = (lambda pr, k, start: generate_masked(g, pr, k, args.bytes, args.temp, start)) if args.mask else \
        (lambda pr, k, start: g.generate(pr, k, args.bytes, temp=args.temp))
    out = gen(b"\n\ndef ", args.n, b"def ").cpu().numpy()
    free = []
    for row in out:
        src, term = cut_function("def " + bytes(row).decode("utf-8", "replace"))
        free.append({"src": src, "terminated": term, **judge(src)})
    report["free"] = summarize(free)
    hold = json.load(open(HOLDOUT))
    random.Random(1).shuffle(hold)
    hold = [h for h in hold if len(h["clean"]) <= 600][:args.n]
    comp = []
    for h in hold:  # one prompt each; batched by 1 row (prompts differ)
        sig = h["clean"].split("\n")[0] + "\n"
        # a prompt longer than the context would index the position table out of range (GPU hardware fault)
        prompt = (b"\n\n" + sig.encode())[-(g.ctx // 2):]
        row = gen(prompt, 1, sig.encode())[0].cpu().numpy()
        src, term = cut_function(sig + bytes(row).decode("utf-8", "replace"))
        comp.append({"src": src, "terminated": term, "self_contained": h["self_contained"], **judge(src)})
    report["completion"] = summarize(comp)
    report["wall_s"] = round(time.perf_counter() - t0, 1)
    report["examples"] = {"free": [r["src"] for r in free[:12]], "completion": [r["src"] for r in comp[:12]]}
    report["rows"] = {"free": free, "completion": comp}
    with open(args.out, "w") as fh:
        json.dump(report, fh)
    print(json.dumps({k: report[k] for k in ("step", "free", "completion", "wall_s")}))


if __name__ == "__main__":
    main()
