"""Measurements for the edit-by-address idea (CPU only, tokenizer only, no model weights).

  1. minimal edit span per failing function (char-level diff generated -> fixed), in every address scheme
  2. localization: does the address voted from the failing tests contain the minimal edit? Ochiai rank
  3. tokens: output to regenerate (file / function / regions / minimal diff), context per scheme
  4. latency: parse, locate, resolve, apply_patch (full re-parse) vs apply_patch_fast (local re-parse),
     sandbox re-judge
Writes measure.json next to this file. Needs prompts.json (run gen_patch_prompts.py first).
"""

import ast
import json
import os
import statistics
import time

os.environ.setdefault("HF_HOME", "/media/llm-data/models/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import addressing as A  # noqa: E402
import fixes as F  # noqa: E402
import gen_patch_prompts as G  # noqa: E402
import localize as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def char_diff(a, b):
    """Minimal single contiguous edit: common prefix/suffix. -> (start, end_in_a, end_in_b)."""
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[-1 - s] == b[-1 - s]:
        s += 1
    return p, len(a) - s, len(b) - s


def to_xy(src, off):
    line = src.count("\n", 0, off) + 1
    col = off - (src.rfind("\n", 0, off) + 1)
    return [line, col]


def enclosing(src, tree, l1, l2):
    """Smallest statement (as AST path + symbol stmt range) whose lines contain [l1, l2]."""
    best = None
    for mi, node in enumerate(tree.body):
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= l1 and l2 <= node.end_lineno):
            continue
        idx = [k for k, s in enumerate(node.body) if s.end_lineno >= l1 and s.lineno <= max(l2, l1)]
        if not idx:  # pure insertion between statements
            k = next((k for k, s in enumerate(node.body) if s.lineno >= l1), len(node.body))
            return {"symbol": node.name, "stmt": [k, k], "path": f"body[{mi}].body[{k}:{k}]"}
        best = {"symbol": node.name, "stmt": [idx[0], idx[-1] + 1],
                "path": f"body[{mi}].body[{idx[0]}:{idx[-1] + 1}]"}
        if len(idx) == 1:  # descend while one child covers the whole range
            path, cur = f"body[{mi}].body[{idx[0]}]", node.body[idx[0]]
            while True:
                nxt = None
                for field in ("body", "orelse", "finalbody"):
                    for k, s in enumerate(getattr(cur, field, []) or []):
                        if isinstance(s, ast.stmt) and s.lineno <= l1 and l2 <= s.end_lineno:
                            nxt = (field, k, s)
                if not nxt:
                    break
                path += f".{nxt[0]}[{nxt[1]}]"
                cur = nxt[2]
            best["path"] = path
        return best
    return best


def timeit(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return {"median_us": round(statistics.median(ts), 1), "p90_us": round(sorted(ts)[int(0.9 * n)], 1), "n": n}


def main():
    tok = G.tokenizer()
    nt = lambda t: G.n_tok(tok, t)  # noqa: E731
    src = F.load_generated()
    tree = ast.parse(src)
    prompts = json.load(open(os.path.join(HERE, "prompts.json")))
    res = {"file": {"lines": src.count("\n"), "chars": len(src), "tokens": nt(src),
                    "functions": len(A.functions(tree))}, "funcs": {}}

    for name in F.FAILING:
        r = {}
        fsrc, node = A.function_source(src, name, tree)
        r["function_lines"] = [node.lineno, node.end_lineno]
        r["function_tokens"] = nt(fsrc)
        for fam in ("minimal", "ref"):
            fixed = F.apply_family(src, name, fam)
            s, ea, eb = char_diff(src, fixed)
            l1, l2 = to_xy(src, s)[0], to_xy(src, max(ea - 1, s))[0] if ea > s else to_xy(src, s)[0]
            r[fam] = {"span_xy": {"start": to_xy(src, s), "end": to_xy(src, ea)},
                      "removed_chars": ea - s, "inserted_chars": eb - s,
                      "removed_text": src[s:ea], "inserted_text": fixed[s:eb],
                      "inserted_tokens": nt(fixed[s:eb]), "removed_tokens": nt(src[s:ea]),
                      "lines": [l1, l2], "enclosing": enclosing(src, tree, l1, l2),
                      "fixed_function_tokens": nt(A.function_source(fixed, name)[0])}
        # localization quality
        fc = L.failing_calls(src, name, seed=1)
        loc = prompts["locate"][name]
        mlines = r["minimal"]["lines"]
        cover = {}
        for scheme, key in G.SPAN_ADDR.items():
            s_, e_, _ = A.resolve(src, loc["addresses"][key], tree)
            a_l = [src.count("\n", 0, s_) + 1, src.count("\n", 0, e_)]
            cover[scheme] = {"region_lines": a_l, "contains_minimal_edit": a_l[0] <= mlines[0] and mlines[1] <= a_l[1]}
        r["locate"] = {"site_line": loc["site_line"], "site_how": loc["site_how"], "votes": loc["votes"],
                       "n_fail_seed1": loc["n_fail"], "n_pass_seed1": loc["n_pass"], "regions": cover}
        och = L.ochiai(fc["fail"], fc["pass"])
        edit_lines = set(range(mlines[0], mlines[1] + 1))
        rank = next((k + 1 for k, (ln, _) in enumerate(och) if ln in edit_lines), None)
        top = och[0][1] if och else 0
        r["ochiai"] = {"top5": [[ln, round(s, 3)] for ln, s in och[:5]], "first_edit_line_rank": rank,
                       "lines_tied_at_top": sum(1 for _, s in och if s == top), "executed_lines": len(och)}
        # tokens per scheme from the prepared prompts
        r["schemes"] = {p["scheme"]: {"prompt_tokens": p["prompt_tokens"], "context_tokens": p["context_tokens"],
                                      "output_tokens": p["target_tokens"], "target_kind": p.get("target_kind"),
                                      "unreached": len(p.get("failing_calls_not_reaching_region", []))}
                        for p in prompts["prompts"] if p["name"] == name}
        res["funcs"][name] = r
        print(name, "minimal span", r["minimal"]["span_xy"], "ins_tok", r["minimal"]["inserted_tokens"],
              "enclosing", r["minimal"]["enclosing"], "ochiai rank", rank, flush=True)

    # latency
    lat = {}
    lat["ast_parse_file"] = timeit(lambda: ast.parse(src), 300)
    lat["locate"] = timeit(lambda: A.locate(src, 523, tree), 300)
    gi = A.functions(tree)["gcd"][0]
    cases = {
        "span_xy (gcd line 523)": ({"kind": "span", "start": [523, 15], "end": [523, 16]}, "abs(a)"),
        "ast_path (gcd body[1].body[0])": ({"kind": "path", "path": f"body[{gi}].body[1].body[0]"}, "return abs(a)"),
        "symbol+stmt (nth_root [3,5))": ({"kind": "symbol", "name": "nth_root", "stmt": [3, 5]},
                                        F.fixes(src, tree)["nth_root"]["minimal"][0][1]),
        "symbol whole func (evaluate)": ({"kind": "symbol", "name": "evaluate", "stmt": None},
                                         A.function_source(F.apply_family(src, "evaluate", "ref"), "evaluate")[0]),
    }
    lat["resolve_only"] = timeit(lambda: A.resolve(src, cases["ast_path (gcd body[1].body[0])"][0], tree), 300)
    for label, (addr, text) in cases.items():
        lat[f"apply_patch full-reparse, cached tree: {label}"] = timeit(lambda: A.apply_patch(src, addr, text, tree), 300)
        lat[f"apply_patch full-reparse, no tree: {label}"] = timeit(lambda: A.apply_patch(src, addr, text), 100)
        # apply_patch_fast line-shifts the nodes AFTER the host in the tree it is given; those nodes are
        # never read for this address, so reusing one scratch tree across repetitions is valid for timing
        scratch = ast.parse(src)
        lat[f"apply_patch_fast local-reparse: {label}"] = timeit(lambda: A.apply_patch_fast(src, addr, text, scratch), 300)
    lat["reject (syntax) full"] = timeit(lambda: _reject(lambda: A.apply_patch(src, cases["ast_path (gcd body[1].body[0])"][0], "return (a", tree)), 300)
    lat["reject (syntax) fast"] = timeit(lambda: _reject(lambda: A.apply_patch_fast(src, cases["ast_path (gcd body[1].body[0])"][0], "return (a", tree)), 300)
    fixed_gcd = F.apply_family(src, "gcd", "minimal")
    lat["sandbox re-judge 1 function (2 seeds)"] = timeit(lambda: L.judge_module(fixed_gcd, ["gcd"]), 10)
    lat["sandbox re-judge all 39 (2 seeds)"] = timeit(lambda: L.judge_module(fixed_gcd), 5)
    lat["sandbox traceback run 1 function"] = timeit(lambda: L.failing_calls(src, "gcd"), 10)
    res["latency"] = lat
    for k, v in lat.items():
        print(f"{k:70s} {v['median_us']:>12.1f} us (p90 {v['p90_us']})")
    with open(os.path.join(HERE, "measure.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote measure.json")


def _reject(fn):
    try:
        fn()
    except A.PatchRejected:
        return True
    return False


if __name__ == "__main__":
    main()
