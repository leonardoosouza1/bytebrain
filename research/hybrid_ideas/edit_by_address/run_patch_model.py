"""Close the loop on the prepared prompts: answer -> patch by address -> re-judge.

  --oracle   use each prompt's stored oracle target as the "answer" (CPU only, no model). Proves that
             the address + patch + re-judge plumbing is correct and that the region can hold a fix.
  default    load the frozen coder model on GPU and generate the answers (greedy + optional samples).

For every answer: strip fences/markers, apply_patch at the prompt's address (whole_file: replace the
module, guarded), then judge the patched module on the hidden cases (seeds 0 and 1) of the target
function, plus every other function (regression check: an address-confined patch must not break them).

Usage:
  run_patch_model.py --oracle
  run_patch_model.py --k 8 --out /media/llm-data/universe-runs/hybrid-ideas-20261021/edit_by_address
"""

import argparse
import json
import os
import time

os.environ.setdefault("HF_HOME", "/media/llm-data/models/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import addressing as A  # noqa: E402
import gen_patch_prompts as G  # noqa: E402
import localize as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DEFAULT = "/media/llm-data/universe-runs/hybrid-ideas-20261021/edit_by_address"


def clean_answer(text):
    text = text.split("```")[0]
    lines = [ln for ln in text.split("\n") if ln.strip() not in (G.MARK_START, G.MARK_END)]
    return "\n".join(lines).rstrip() + "\n"


def patch(source, p, answer):
    """-> (patched_source | None, reject_reason | None, apply_us)."""
    t0 = time.perf_counter()
    try:
        if p["scheme"] == "whole_file":
            n = source.count("\n")
            end = [n + 1, 0] if source.endswith("\n") else [n + 1, len(source.splitlines()[-1])]
            new, _ = A.apply_patch(source, {"kind": "span", "start": [1, 0], "end": end}, answer)
        else:
            new, _ = A.apply_patch(source, p["address"], answer)
        return new, None, (time.perf_counter() - t0) * 1e6
    except A.PatchRejected as e:
        return None, str(e), (time.perf_counter() - t0) * 1e6


def judge_answer(source, p, answer, baseline):
    new, why, us = patch(source, p, answer)
    rec = {"apply_us": round(us, 1), "rejected": why}
    if new is None:
        rec.update(fixed=False, regressions=None)
        return rec
    t0 = time.perf_counter()
    res = L.judge_module(new)
    rec["judge_s"] = round(time.perf_counter() - t0, 3)
    rec["fixed"] = res[p["name"]]
    rec["regressions"] = [k for k, v in res.items() if baseline[k] and not v]
    return rec


VRAM_CAP_GB = 4.0


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    # hard VRAM cap: sdpa materialises [batch, heads, L, L] on this GPU; a 9x4k whole-file prompt asked for 16 GB
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GB * 1e9 / total), 0)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    return tok, model


def generate(tok, model, text, max_new, k, temperature):
    import torch
    enc = tok([text], return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new, stop_strings=["```"], tokenizer=tok, pad_token_id=tok.eos_token_id)
    if k:
        kw.update(do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=k)
    else:
        kw.update(do_sample=False)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    s = time.perf_counter() - t0
    L_ = enc["input_ids"].shape[1]
    texts = tok.batch_decode(out[:, L_:], skip_special_tokens=True)
    n_new = int((out[:, L_:] != tok.eos_token_id).sum())
    return texts, s, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(HERE, "prompts.json"))
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--k", type=int, default=0, help="extra sampled answers per prompt (model mode)")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schemes", default=",".join(G.SCHEMES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--long-tokens", type=int, default=1500,
                    help="prompts longer than this get greedy only (samples would not fit the VRAM cap)")
    args = ap.parse_args()
    data = json.load(open(args.prompts))
    source = data["source"]
    baseline = L.judge_module(source)
    schemes = args.schemes.split(",")
    tok = model = None
    if not args.oracle:
        import torch
        torch.manual_seed(args.seed)
        tok, model = load_model(data["model"])
    results = []
    for p in data["prompts"]:
        if p["scheme"] not in schemes:
            continue
        r = {"name": p["name"], "scheme": p["scheme"], "prompt_tokens": p["prompt_tokens"],
             "target_tokens": p["target_tokens"], "covers_fix": p["covers_fix"], "answers": []}
        if args.oracle:
            if p["target"] is None:
                r["answers"].append({"kind": "oracle", "skipped": "no oracle edit for this region"})
            else:
                a = judge_answer(source, p, p["target"], baseline)
                a["kind"] = "oracle"
                r["answers"].append(a)
        else:
            texts, s, n_new = generate(tok, model, p["chat"], p["max_new_tokens"], 0, 0)
            batches = [("greedy", texts, s, n_new)]
            if args.k and p["prompt_tokens"] > args.long_tokens:
                print(f"{p['name']} {p['scheme']}: {p['prompt_tokens']} prompt tokens > {args.long_tokens}, greedy only",
                      flush=True)
            elif args.k:
                batches.append(("sample", *generate(tok, model, p["chat"], p["max_new_tokens"], args.k,
                                                    args.temperature)))
            for kind, ts, s, n_new in batches:
                for t in ts:
                    a = judge_answer(source, p, clean_answer(t), baseline)
                    a.update(kind=kind, answer=t, gen_s=round(s / len(ts), 3), new_tokens=round(n_new / len(ts), 1))
                    r["answers"].append(a)
        ok = [a for a in r["answers"] if a.get("fixed") and not a.get("regressions")]
        r["greedy_fixed"] = bool(r["answers"] and r["answers"][0].get("fixed") and not r["answers"][0].get("regressions"))
        r["any_fixed"] = bool(ok)
        results.append(r)
        a0 = r["answers"][0]
        print(f"{p['name']:10s} {p['scheme']:12s} first: fixed {a0.get('fixed')} rej {a0.get('rejected')} "
              f"regr {a0.get('regressions')} apply {a0.get('apply_us')}us  any_fixed {r['any_fixed']} "
              f"({len(ok)}/{len(r['answers'])})", flush=True)
    out_dir = args.out or (HERE if args.oracle else OUT_DEFAULT)
    os.makedirs(out_dir, exist_ok=True)
    name = "oracle_results.json" if args.oracle else f"model_results_s{args.seed}_k{args.k}.json"
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump({"oracle": args.oracle, "k": args.k, "results": results}, fh, indent=1)
    print("wrote", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
