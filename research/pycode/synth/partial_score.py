"""Graded correctness of ANY generated function/program vs its reference (ledger #30, registered 2026-10-26, CPU).

Request (#30): "se o modelo completo gera uma função, o quanto dela está correta?" - pass/fail is too coarse.
All execution of generated code happens in the existing bwrap runners (synth2.run_many, multifn.run_many,
compose.run_many); this module only builds the programs to run and compares outputs. No new sandbox.

Scores per function (dict keys; every score is in 0..1 unless said otherwise):
  parse               1 if ast.parse accepts the whole source
  exec                1 if the function defines and runs on every input without exception or timeout
  io_frac             share of the inputs (synth2: 40; multifn: ops x 20 cases; compose: 20 cases) with the reference output
  state_frac          synth2 only: share of inputs where the list state before the final return equals the reference state
  strict              synth2: share where output AND state are right (= synth2.grade); multifn/compose: = io_frac
  pass                True iff strict == 1 (same as the old pass1)
  line_frac           share of reference body lines found in order (difflib SequenceMatcher on stripped lines)
  block_frac          share of planned blocks whose text is exactly present (synth2: unit/final lines in order;
                      multifn/compose: def blocks and dispatch blocks / main block, any order); None without a plan
  first_wrong_line    0-based index of the first body line that differs from the reference (None if identical)
  first_wrong_unit    synth2 only: 0-based index of the first unit whose cumulative list state is never reproduced by any
                      prefix of the generated body (prefix programs return xs after k lines, run in bwrap); None if all reached
  units_reached_frac  synth2 only: share of units whose cumulative state is reproduced by some prefix
  op_frac             multifn only: share of requested ops right on every case
Flags used by the failure taxonomy: skipped (fewer body lines than the reference / a required def missing),
extra (more lines, nothing missing), wrong_const (same lines once digits are masked, digits differ).
classify() gives ONE label per function: pass | parse_error | runtime_error | wrong_constant | skipped_unit |
extra_line | wrong_output, in that precedence.

Caveats: prefix states can match early by accident (sort twice, abs twice), so first_wrong_unit is a lower bound on
the true breakage point. block_frac is text-exact, so a correct elif instead of if, or a docstring, costs credit
while io_frac stays 1: read both. A multi-line unit (loop) cut inside a prefix fails to run and just does not count.

LLM judge: only a stub (judge_prompt / parse_judge); it calls no model. It MUST be calibrated against io_frac
(calibrate()) on functions with known partial scores before its numbers are trusted or reported.

Usage: partial_score.py check      (reference must be 1.0 and a stub body must not, all three suites; runs bwrap)
"""

import ast
import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import compose as CP  # noqa: E402
import multifn as MF  # noqa: E402
import synth2 as S2  # noqa: E402

FAILS = ["parse_error", "runtime_error", "wrong_constant", "skipped_unit", "extra_line", "wrong_output"]
MEANS = ["parse", "exec", "io_frac", "state_frac", "strict", "line_frac", "block_frac", "units_reached_frac", "op_frac"]
_NUM = re.compile(r"\d+")


def _chunks(fn, items, n=256):
    out = []
    for i in range(0, len(items), n):
        out += fn(items[i:i + n])
    return out


def _parses(src):
    try:
        ast.parse(src)
        return True
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False


def _def_at(lines):
    return next((i for i, l in enumerate(lines) if l.startswith("def ")), -1)


def body_lines(src):
    """non-blank lines (right-stripped) after the first top-level def line."""
    ls = src.split("\n")
    return [l.rstrip() for l in ls[_def_at(ls) + 1:] if l.strip()]


def all_lines(src):
    return [l.rstrip() for l in src.split("\n") if l.strip()]


def core(lines):
    """body without trailing top-level return lines (the final)."""
    lines = list(lines)
    while lines and lines[-1].startswith("    return"):
        lines.pop()
    return lines


def line_frac(gen, ref):
    g, r = [l.strip() for l in gen], [l.strip() for l in ref]
    if not r:
        return 1.0
    m = difflib.SequenceMatcher(None, g, r, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks()) / len(r)


def first_wrong_line(gen, ref):
    for i, (a, b) in enumerate(zip(gen, ref)):
        if a != b:
            return i
    return None if len(gen) == len(ref) else min(len(gen), len(ref))


def blocks_found(gen, blocks, ordered):
    """share of blocks (lists of lines, indentation included) present exactly as contiguous lines in gen."""
    gen, cur, ok = [l.rstrip() for l in gen], 0, 0
    for blk in blocks:
        blk = [l.rstrip() for l in blk]
        for i in range(cur if ordered else 0, len(gen) - len(blk) + 1):
            if gen[i:i + len(blk)] == blk:
                ok += 1
                cur = i + len(blk) if ordered else cur
                break
    return ok / len(blocks) if blocks else 1.0


def shape(gen, ref, ordered=True):
    g, r = [l.strip() for l in gen], [l.strip() for l in ref]
    if not ordered:
        g, r = sorted(g), sorted(r)
    return {"skipped": len(g) < len(r), "extra": len(g) > len(r),
            "wrong_const": g != r and [_NUM.sub("#", x) for x in g] == [_NUM.sub("#", x) for x in r]}


def classify(s):
    if s["pass"]:
        return "pass"
    if not s["parse"]:
        return "parse_error"
    if not s["exec"]:
        return "runtime_error"
    for flag, label in (("wrong_const", "wrong_constant"), ("skipped", "skipped_unit"), ("extra", "extra_line")):
        if s[flag]:
            return label
    return "wrong_output"


# ------------------------------------------------------------------ synth2
def _prefixes(src):
    """programs returning xs after k body lines (k = 1..n), keeping the generated head; the final return is dropped."""
    ls = src.rstrip("\n").split("\n")
    i = _def_at(ls)
    if i < 0:
        return []
    body = core([l for l in ls[i + 1:] if l.strip()])
    return ["\n".join(ls[:i + 1] + body[:j] + ["    return xs"]) + "\n" for j in range(1, len(body) + 1)]


def _ref_states(units):
    """reference list state after k units, k = 1..n, on S2.INPUTS (reference code, run in-process like synth2.reference)."""
    out = []
    for k in range(1, len(units) + 1):
        src = S2.head(units, "") + "".join(f"    {l}\n" for x in units[:k] for l in S2.UNITS[x]) + "    return xs\n"
        g = {}
        exec(src.replace(f"def {S2.name_of(units, '')}(", "def _ref(", 1), g)
        f = g["_ref"]
        out.append([repr(f(list(xs), k_) if S2.nargs(units) == 2 else f(list(xs))) for xs, k_ in S2.INPUTS])
    return out


def score_synth2(tasks, srcs):
    """tasks [(units, final)], srcs [full source each] -> [score dict]."""
    items, spans = [], []
    for (u, f), s in zip(tasks, srcs):
        nm, na = S2.name_of(u, f), S2.nargs(u)
        pre = _prefixes(s) if _parses(s) else []
        spans.append((len(items), len(pre)))
        items += [(s, nm, na), (S2.state_variant(s), nm, na)] + [(p, nm, na) for p in pre]
    outs = _chunks(S2.run_many, items)
    res = []
    for (u, f), s, (b, npre) in zip(tasks, srcs, spans):
        ref, ref_st = S2.reference(u, f), S2.reference(u, "")
        o, st = outs[b], outs[b + 1]
        n = len(ref)
        io = sum(a == r for a, r in zip(o, ref)) / n if o else 0.0
        sf = sum(a == r for a, r in zip(st, ref_st)) / n if st else 0.0
        strict = sum(a == r and x == y for a, r, x, y in zip(o, ref, st, ref_st)) / n if o and st else 0.0
        gen, refb = body_lines(s), body_lines(S2.source(u, f))
        sh = shape(core(gen), core(refb))
        blocks = [["    " + l for l in S2.UNITS[x]] for x in u] + [["    " + S2.FINAL[f]]]
        if npre:
            pouts = [p for p in outs[b + 2:b + 2 + npre] if p is not None]
            reached = [any(p == r for p in pouts) for r in _ref_states(u)]
            fwu = next((k for k, x in enumerate(reached) if not x), None)
            urf = sum(reached) / len(reached)
        else:
            fwu, urf = None, 0.0
        res.append({"parse": float(bool(npre) or _parses(s)), "exec": float(o is not None and "!" not in o),
                    "io_frac": io, "state_frac": sf, "strict": strict, "pass": strict == 1.0,
                    "line_frac": line_frac(gen, refb), "block_frac": blocks_found(gen, blocks, True),
                    "first_wrong_line": first_wrong_line(gen, refb), "first_wrong_unit": fwu,
                    "first_wrong_unit_name": u[fwu] if fwu is not None else None, "units_reached_frac": urf, **sh})
    for r in res:
        r["label"] = classify(r)
    return res


# ------------------------------------------------------------------ multifn / compose
def _blocks_multifn(spec):
    syms = list(dict.fromkeys(s for s in MF.plan_of(spec) if s.startswith(("def:", "disp:"))))
    return [MF.block_text(s).rstrip("\n").split("\n") for s in syms]


def _missing_defs(src, ops):
    return any(not re.search(rf"^def {o}\(", src, re.M) for o in ops)


def _finish(r):
    r["label"] = classify(r)
    return r


def score_multifn(specs, srcs):
    """specs [tuple of ops], srcs [full program each] -> [score dict]; main(op, a, b) on ops x 20 cases."""
    uniq = sorted(set(specs))
    refs = {sp: MF.assemble(MF.plan_of(sp)) for sp in uniq}
    ref_out = dict(zip(uniq, _chunks(MF.run_many, [(refs[sp], list(sp)) for sp in uniq])))
    outs = _chunks(MF.run_many, [(s, list(sp)) for s, sp in zip(srcs, specs)])
    nc, res = len(MF.CASES), []
    for sp, s, o in zip(specs, srcs, outs):
        ref = ref_out[sp]
        io = sum(a == r for a, r in zip(o, ref)) / len(ref) if o else 0.0
        per_op = [all(a == r for a, r in zip(o[i * nc:(i + 1) * nc], ref[i * nc:(i + 1) * nc])) if o else False
                  for i in range(len(sp))]
        bad = o is None or any(a.startswith("!") and a != r for a, r in zip(o, ref))
        gen, refl = all_lines(s), all_lines(refs[sp])
        sh = shape(gen, refl, ordered=False)
        need = [x[4:] for x in MF.plan_of(sp) if x.startswith("def:")]
        sh["skipped"] = _missing_defs(s, need)
        sh["extra"] = sh["extra"] and not sh["skipped"]
        res.append(_finish({"parse": float(_parses(s)), "exec": float(not bad), "io_frac": io, "strict": io,
                            "pass": io == 1.0, "op_frac": sum(per_op) / len(sp), "line_frac": line_frac(gen, refl),
                            "block_frac": blocks_found(gen, _blocks_multifn(sp), False),
                            "first_wrong_line": first_wrong_line(gen, refl), **sh}))
    return res


def score_compose(prefixes, srcs):
    """prefixes [prefix token tuple], srcs [full program each] -> [score dict]; main(a, b) on 20 cases."""
    uniq = sorted(set(prefixes))
    refs = {p: CP.assemble(CP.plan_of(p), MF.block_text) for p in uniq}
    ref_out = dict(zip(uniq, _chunks(CP.run_many, [refs[p] for p in uniq])))
    outs = _chunks(CP.run_many, list(srcs))
    res = []
    for p, s, o in zip(prefixes, srcs, outs):
        ref = ref_out[p]
        io = sum(a == r for a, r in zip(o, ref)) / len(ref) if o else 0.0
        bad = o is None or any(a.startswith("!") and a != r for a, r in zip(o, ref))
        gen, refl = all_lines(s), all_lines(refs[p])
        plan = CP.plan_of(p)
        blocks = [MF.block_text(x).rstrip("\n").split("\n") for x in dict.fromkeys(x for x in plan if x.startswith("def:"))]
        blocks.append(["def main(a, b=0):", f"    return {CP.render_expr(plan)}"])
        sh = shape(gen, refl, ordered=False)
        sh["skipped"] = _missing_defs(s, [x[4:] for x in plan if x.startswith("def:")])
        sh["extra"] = sh["extra"] and not sh["skipped"]
        res.append(_finish({"parse": float(_parses(s)), "exec": float(not bad), "io_frac": io, "strict": io,
                            "pass": io == 1.0, "line_frac": line_frac(gen, refl),
                            "block_frac": blocks_found(gen, blocks, False),
                            "first_wrong_line": first_wrong_line(gen, refl), **sh}))
    return res


SCORERS = {"synth2": score_synth2, "multifn": score_multifn, "compose": score_compose}


def score_many(suite, tasks, srcs):
    return SCORERS[suite](tasks, srcs)


def summarize(scores):
    """means of every score present + failure taxonomy counts + share of failures broken at each first_wrong_unit."""
    n = len(scores)
    out = {"n": n, "pass1": round(sum(s["pass"] for s in scores) / n, 4)}
    for k in MEANS:
        v = [s[k] for s in scores if s.get(k) is not None]
        if v:
            out[k] = round(sum(v) / len(v), 4)
    out["taxonomy"] = {k: sum(s["label"] == k for s in scores) for k in FAILS}
    fw = [s["first_wrong_unit"] for s in scores if s.get("first_wrong_unit") is not None]
    if fw:
        out["first_wrong_unit_hist"] = {str(k): fw.count(k) for k in sorted(set(fw))}
    return out


# ------------------------------------------------------------------ LLM judge (stub: no model is called)
def judge_prompt(spec_text, source):
    """prompt for a judge model (Qwen or Claude); the caller sends it. Output must be parsed with parse_judge."""
    return ("You grade a Python function against its specification. Judge only what the code does, not its style.\n"
            "Reply with JSON only: {\"score\": <number 0..1 = share of the specification the code implements correctly>, "
            "\"reason\": \"<one sentence naming the first thing that is wrong, or 'correct'>\"}\n\n"
            f"SPECIFICATION:\n{spec_text}\n\nCODE:\n```python\n{source.rstrip()}\n```\n")


def parse_judge(text):
    """judge reply -> {"score": float in 0..1 or None when unparsable, "reason": str}; 1 < score <= 100 is read as percent."""
    score, reason = None, ""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            score, reason = d.get("score"), str(d.get("reason", ""))
        except (json.JSONDecodeError, AttributeError):
            pass
    if score is None:
        m = re.search(r"score\"?\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.I)
        score = m.group(1) if m else None
    try:
        score = float(score)
    except (TypeError, ValueError):
        return {"score": None, "reason": reason or "unparsable judge reply"}
    if 1 < score <= 100:
        score /= 100
    return {"score": min(1.0, max(0.0, score)), "reason": reason}


def calibrate(judge_scores, io_fracs):
    """judge vs bwrap io_frac on the same functions -> pearson r and mean absolute error (None pairs dropped)."""
    p = [(j, i) for j, i in zip(judge_scores, io_fracs) if j is not None]
    if len(p) < 3:
        return {"n": len(p), "pearson": None, "mae": None}
    mj, mi = sum(j for j, _ in p) / len(p), sum(i for _, i in p) / len(p)
    cov = sum((j - mj) * (i - mi) for j, i in p)
    vj, vi = sum((j - mj) ** 2 for j, _ in p), sum((i - mi) ** 2 for _, i in p)
    return {"n": len(p), "pearson": round(cov / (vj * vi) ** 0.5, 4) if vj > 0 and vi > 0 else None,
            "mae": round(sum(abs(j - i) for j, i in p) / len(p), 4)}


def check():
    """each suite: reference scores pass, a stub body does not (known numbers before trusting any report)."""
    t = [["add_2", "sort"], "sum"]
    sp, pf = ("add", "hyp"), ("hyp", "a", "b")
    cases = {"synth2": ([tuple(t)], S2.source(*t), S2.head(*t) + "    return None\n"),
             "multifn": ([sp], MF.assemble(MF.plan_of(sp)), "def main(op, a, b=0):\n    return None\n"),
             "compose": ([pf], CP.assemble(CP.plan_of(pf), MF.block_text), "def main(a, b=0):\n    return None\n")}
    for name, (tasks, ref, bad) in cases.items():
        r, b = score_many(name, tasks, [ref])[0], score_many(name, tasks, [bad])[0]
        print(json.dumps({"suite": name, "ref_pass": r["pass"], "ref_line": r["line_frac"], "ref_block": r["block_frac"],
                          "stub_pass": b["pass"], "stub_label": b["label"], "stub_io": b["io_frac"]}))
        assert r["pass"] and r["line_frac"] == 1.0 and r["block_frac"] == 1.0 and not b["pass"]


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
    else:
        print(__doc__)
