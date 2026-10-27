"""Out-of-template test set for the understander (ledger #34, #29): Sonnet rewrites the 179 e2e requests the way a
person would type them in a chat, then a second Sonnet pass checks each rewrite against the gold spec; only rewrites
it confirms are kept. Ids and gold are unchanged, so `e2e_pipeline.py run --out <dir>` grades them as before.

Why: every number so far uses requests built from 6 sentence frames; a fine-tuned student can learn the frames.
Cost: 2 Sonnet calls per batch of 30 (12 for 179), no tools, retried when the claude-mem hook blocks the call.

`--from-train N` rewrites N items of the student's TRAINING pairs instead (arm s5: compose train trees + random
4-5 call trees that exclude every test tree, multifn train op lists) and writes `<dst>/train_para.json`, the
augmentation set for `student_understander.py --extra` (arm s5p). Test requests are never sent in this mode.
"""

import argparse
import json
import os
import subprocess

GEN = """Rewrite each Portuguese request below the way a Brazilian developer would naturally type it to a coding
assistant in a chat. Keep the meaning EXACTLY:
- calculator requests: the same operations in the same order, any wording; it must still be a calculator / program
  offering those operations;
- formula requests: main receives a and b and returns the formula; keep every operation, every operand (a or b),
  the operand order and the nesting. You may replace parentheses by words only if the grouping stays unambiguous.
Vary style across items (informal, imperative, question, short, long). Do not reuse the original sentence frames
"crie uma função main(a, b) que retorna", "faça um programa cuja main(a, b) devolve", "crie uma calculadora que
faça", "quero um programa em python com", "faça uma calculadora com as operações:", "preciso de uma função main que
escolha entre". Use Portuguese with accents.
Answer with one JSON array only: [{"id": "...", "text": "..."}], same ids, same count.

REQUESTS:
"""

CHECK = """For each item, decide whether the REWRITE asks for exactly the GOLD spec.
GOLD formats: {"main": "dispatch", "calls": [ops in order]} = a calculator offering those operations;
{"main": "expr", "expr": "..."} = main(a, b) returns that formula. Operation ids: add a+b, sub a-b, mul a*b, div a/b,
mod remainder of a/b, maxx larger of a and b, sqrt square root of a, neg -a, abs |a|, square a*a, pow a**b,
fact a!, hyp hypotenuse with legs a and b, avg average, pct a percent of b.
An item is "same" only if a careful reader could build exactly the gold from the rewrite alone (same operations,
order, operands, nesting; for calculators the same operations in the same order).
Answer with one JSON array only: [{"id": "...", "same": true|false, "why": "<short, only when false>"}].

ITEMS:
"""


def sonnet(prompt, cwd, budget):
    for _ in range(3):
        if budget["left"] <= 0:
            return None
        budget["left"] -= 1
        p = subprocess.run(["claude", "-p", "--model", "sonnet", "--tools", "", "--output-format", "json"],
                           input=prompt, capture_output=True, text=True, timeout=600, cwd=cwd)
        try:
            text = json.loads(p.stdout).get("result", "")
        except json.JSONDecodeError:
            continue
        if "operation blocked by hook" in text[:300]:
            continue
        i, j = text.find("["), text.rfind("]")
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="e2e run dir holding requests.json")
    ap.add_argument("--dst", required=True, help="new run dir for the rewritten requests.json")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--max-calls", type=int, default=24)
    ap.add_argument("--from-train", type=int, default=0, help="rewrite N training pairs (augmentation), not tests")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--world", choices=["any", "dispatch", "expr"], default="any", help="--from-train: spec kind")
    ap.add_argument("--gen-extra", default="", help="extra style instruction appended to the rewrite prompt")
    ap.add_argument("--name", default="train_para", help="--from-train: output file stem")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="--from-train: request files whose texts, orig_texts or golds may not be sampled (ledger #7)")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    cwd = os.path.join(a.dst, "sonnet_cwd")
    os.makedirs(cwd, exist_ok=True)
    tests = json.load(open(os.path.join(a.src, "requests.json")))
    if a.from_train:
        import random
        import student_understander as SU
        pairs, _ = SU.train_pairs("s5", a.seed, tests)
        pairs = [p for p in pairs if a.world == "any" or p[1]["main"] == a.world]
        ex = [q for f in a.exclude for q in json.load(open(f))]
        bad_t = {q["text"] for q in ex} | {q["orig_text"] for q in ex if "orig_text" in q}
        bad_g = {json.dumps(q["gold"], sort_keys=True) for q in ex}
        n0 = len(pairs)
        pairs = [p for p in pairs if p[0] not in bad_t and json.dumps(p[1], sort_keys=True) not in bad_g]
        print(json.dumps({"pairs": n0, "excluded": n0 - len(pairs), "exclude_files": a.exclude}), flush=True)
        pairs = random.Random(a.seed).sample(pairs, a.from_train)
        reqs = [{"id": f"t{i:04d}", "split": "train", "text": t, "gold": g} for i, (t, g) in enumerate(pairs)]
    else:  # open: gold is our arbitrary choice for "calculadora científica", a rewrite cannot be checked against it
        reqs = [q for q in tests if q["split"] != "open"]
    budget, out, log = {"left": a.max_calls}, [], []
    for i in range(0, len(reqs), a.batch):
        B = reqs[i:i + a.batch]
        rew = sonnet(GEN.replace("\nREQUESTS:", a.gen_extra + "\nREQUESTS:") + json.dumps([{"id": q["id"], "text": q["text"]} for q in B], ensure_ascii=False), cwd,
                     budget)
        if not rew:
            log.append({"batch": i, "error": "generation"})
            continue
        rw = {x["id"]: x["text"] for x in rew if isinstance(x, dict) and "id" in x and "text" in x}
        items = [{"id": q["id"], "GOLD": q["gold"], "REWRITE": rw[q["id"]]} for q in B if q["id"] in rw]
        chk = sonnet(CHECK + json.dumps(items, ensure_ascii=False), cwd, budget)
        ok = {x["id"]: x for x in (chk or []) if isinstance(x, dict) and "id" in x}
        for q in B:
            c = ok.get(q["id"], {})
            keep = q["id"] in rw and c.get("same") is True
            log.append({"id": q["id"], "orig": q["text"], "rewrite": rw.get(q["id"]), "same": c.get("same"),
                        "why": c.get("why")})
            if keep:
                out.append({**q, "text": rw[q["id"]], "orig_text": q["text"]})
        print(json.dumps({"batch": i, "kept": len(out), "calls_left": budget["left"]}), flush=True)
    name = a.name if a.from_train else "requests"
    json.dump(out, open(os.path.join(a.dst, name + ".json"), "w"), indent=1, ensure_ascii=False)
    json.dump(log, open(os.path.join(a.dst, (a.name if a.from_train else "paraphrase") + "_log.json"), "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps({"n_src": len(reqs), "kept": len(out), "calls_used": a.max_calls - budget["left"]}), flush=True)


if __name__ == "__main__":
    main()
