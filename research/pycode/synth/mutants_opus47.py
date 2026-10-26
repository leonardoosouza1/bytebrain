"""#47 bad side: mutants of the verified Opus programs with a known first wrong line.
A mutant is kept only if bwrap execution differs from the good program on >= 1 call (equivalent mutants dropped)."""
import json, re, sys
sys.path.insert(0, "/home/leonardo/projects/LLM/bytebrain/research/pycode/synth")
import e2e_pipeline as E

D = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/saci_code"
recs = [json.loads(l) for l in open(f"{D}/opus_examples.jsonl")]
SWAP = {" + ": " - ", " - ": " + ", " * ": " + ", " / ": " * ", " % ": " // ", " // ": " / ", " > ": " < ", " < ": " > "}

def calls_of(r):
    g = r["gold"]
    if g["main"] == "dispatch":
        return [("main", [o, a, b]) for o in g["calls"] for a, b in E.CASES]
    return [("main", [a, b]) for a, b in E.CASES]

def mutants(r):
    lines = r["program"].split("\n")
    out = []
    main_i = next(i for i, l in enumerate(lines) if l.startswith("def main("))
    for i, l in enumerate(lines):                       # sign/operator flip in a helper body (his a+b -> a-b)
        if i < main_i and l.startswith("    ") and "return" in l:
            for k, v in SWAP.items():
                if k in l:
                    out.append(("op_flip", i, l.replace(k, v, 1)))
                    break
        if "range(b)" in l:
            out.append(("off_by_one", i, l.replace("range(b)", "range(b + 1)")))
        if "range(2, a + 1)" in l:
            out.append(("off_by_one", i, l.replace("range(2, a + 1)", "range(2, a)")))
    if r["gold"]["main"] == "expr":                    # swap the two args of the outermost call in main
        i = main_i + 1
        m = re.match(r"(\s*return )(\w+)\((.*)\)$", lines[i])
        if m:
            depth, cut = 0, None
            for j, ch in enumerate(m.group(3)):
                depth += ch == "("; depth -= ch == ")"
                if ch == "," and depth == 0:
                    cut = j
            if cut is not None:
                x, y = m.group(3)[:cut].strip(), m.group(3)[cut + 1:].strip()
                out.append(("arg_swap", i, f"{m.group(1)}{m.group(2)}({y}, {x})"))
    else:                                               # dispatch: two branches call each other's helper
        br = [i for i, l in enumerate(lines) if re.match(r"\s+return \w+\(a, b\)$", l) and i > main_i]
        if len(br) >= 2:
            i, j = br[0], br[1]
            fi = re.search(r"return (\w+)\(", lines[i]).group(1)
            fj = re.search(r"return (\w+)\(", lines[j]).group(1)
            out.append(("wrong_branch", i, lines[i].replace(fi + "(", fj + "(")))
    res = []
    for kind, i, new in out:
        ls = list(lines); ls[i] = new
        res.append({"kind": kind, "line": i + 1, "orig": lines[i], "mut": new, "program": "\n".join(ls)})
    return res

jobs, meta = [], []
for r in recs:
    jobs.append((r["program"], calls_of(r))); meta.append((r, None))
    for m in mutants(r):
        jobs.append((m["program"], calls_of(r))); meta.append((r, m))
outs = E.run_many(jobs)
good = {}
kept, dropped = [], 0
for (r, m), o in zip(meta, outs):
    if m is None:
        good[r["id"]] = o; continue
    g = good[r["id"]]
    if o is None or g is None:
        dropped += 1; continue
    n_diff = sum(x != y for x, y in zip(o, g))
    if n_diff == 0:
        dropped += 1; continue
    kept.append({"id": f"{r['id']}_{m['kind']}_L{m['line']}", "src": r["id"], "text": r["text"], "gold": r["gold"],
                 "kind": m["kind"], "first_wrong_line": m["line"], "orig_line": m["orig"], "mut_line": m["mut"],
                 "program": m["program"], "calls_differing": n_diff, "calls_total": len(g)})
with open(f"{D}/opus_mutants.jsonl", "w") as f:
    for k in kept:
        f.write(json.dumps(k, ensure_ascii=False) + "\n")
from collections import Counter
print("mutants kept", len(kept), "dropped_equivalent_or_failed", dropped, Counter(k["kind"] for k in kept))
for k in kept[:4]:
    print(k["id"], repr(k["orig_line"]), "->", repr(k["mut_line"]), f"{k['calls_differing']}/{k['calls_total']}")
