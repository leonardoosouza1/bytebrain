"""World-token margin (ledger #43, A1 follow-up): did the steps format push the held frame to "expr", or was s5
already near a coin flip there?

A1 (steps format) answered the held-out frame "preciso de uma função main que escolha entre {L}" as expr 17/18; s5 (same
pairs, order, seed, expr format) answered it as dispatch 18/18. With one seed per arm the two readings cannot be told
apart from pass counts. This reads the decision itself: the prompt plus the shared prefix of the two possible targets
('{"main": "' in json.dumps) is fed once, and the log-probability of the first token of each world is read at the
position where the two targets diverge. margin = log p(dispatch) - log p(expr).

Smoke (positive controls, same prompt and tokenizer as the eval):
  sign: sign(margin) equals the world the recorded greedy generation chose (raws.json, by prefix), except where
        |margin| < 0.5 (bf16 forward vs KV-cache generate). PASS = no disagreement with |margin| >= 0.5.
  magnitude: on the first 20 requests, generate() with output_logits gives the same margin at the decision step
        within 0.3 nats.
Prediction (stated before the run): s5 margins on the held frame are all > 0 (it chose dispatch 18/18) and small:
median < 2 and min < 1 nat -> the decision sat near the boundary. KILL of that reading: median > 5 (s5 was
confident; the format moved it). 2..5 = no verdict. s5st: held median < 0 (it chose expr 17/18).
Limit: a margin is the distance from the decision boundary, not the seed variance. A small s5 margin says a seed
could flip it, it does not show that one did; that needs a second seed. Report by split (one / test_comb / len4 /
len5) next to the other dispatch frames as the baseline.

Usage: world_margin.py --run-dir E2E --lora ADAPTER_DIR [--lora ADAPTER_DIR ...] --out OUT_JSON
"""

import argparse
import gc
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_understander as SU  # noqa: E402

HELD = "preciso de uma função main que escolha entre"


def split_point(tok):
    """Shared token prefix of the two world targets, and the first diverging token of each."""
    d = tok(json.dumps({"main": "dispatch", "calls": ["add"]}), add_special_tokens=False).input_ids
    x = tok(json.dumps({"main": "expr", "expr": "add(a, b)"}), add_special_tokens=False).input_ids
    k = next(i for i, (p, q) in enumerate(zip(d, x)) if p != q)
    return d[:k], d[k], x[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lora", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mem-frac", type=float, default=0.5)
    a = ap.parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    tok = AutoTokenizer.from_pretrained(SU.MODEL, cache_dir=SU.CACHE, local_files_only=True)
    pre, t_d, t_x = split_point(tok)
    print("prefix", repr(tok.decode(pre)), "dispatch tok", repr(tok.decode([t_d])), "expr tok", repr(tok.decode([t_x])))
    reqs = SU.E.load_requests(a.run_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    res = {"prefix": tok.decode(pre), "tok_dispatch": tok.decode([t_d]), "tok_expr": tok.decode([t_x]), "arms": {}}
    for lora in a.lora:
        tag = os.path.basename(os.path.dirname(os.path.normpath(lora)))
        raws = json.load(open(os.path.join(a.run_dir, tag, "raws.json")))
        base = AutoModelForCausalLM.from_pretrained(SU.MODEL, cache_dir=SU.CACHE, local_files_only=True,
                                                    dtype=torch.bfloat16).to("cuda")
        model = PeftModel.from_pretrained(base, lora).merge_and_unload().eval()
        rows = []
        with torch.no_grad():
            for q in reqs:
                ids = tok(SU.prompt(q["text"]), add_special_tokens=False).input_ids + pre
                lp = model(torch.tensor([ids], device="cuda")).logits[0, -1].float().log_softmax(-1)
                raw = raws[q["id"]]["raw"]
                chose = next((w for w in ("dispatch", "expr") if raw.startswith('{"main": "' + w)), None)
                rows.append({"id": q["id"], "world": q["world"], "split": q.get("split"), "held": q["text"].startswith(HELD),
                             "margin": round(float(lp[t_d] - lp[t_x]), 4),
                             "mass_two": round(float(lp[t_d].exp() + lp[t_x].exp()), 4), "chose": chose})
            mag = []                                    # magnitude control vs the KV-cache path of the eval
            for q, r in zip(reqs[:20], rows[:20]):
                ids = torch.tensor([tok(SU.prompt(q["text"]), add_special_tokens=False).input_ids], device="cuda")
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=len(pre) + 1,
                                     do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None,
                                     top_k=None, output_logits=True, return_dict_in_generate=True)
                assert out.sequences[0, ids.shape[1]:ids.shape[1] + len(pre)].tolist() == pre, q["id"]
                g = out.logits[len(pre)][0].float().log_softmax(-1)
                mag.append(abs(float(g[t_d] - g[t_x]) - r["margin"]))
        ok = [r for r in rows if r["chose"] in ("dispatch", "expr")]
        dis = [r for r in ok if (r["margin"] > 0) != (r["chose"] == "dispatch")]
        held = sorted(r["margin"] for r in rows if r["held"])
        disp = sorted(r["margin"] for r in rows if r["world"] == "dispatch" and not r["held"])
        expr = sorted(r["margin"] for r in rows if r["world"] == "expr")
        med = lambda v: round(statistics.median(v), 4) if v else None  # noqa: E731
        by_split = {sp: med([r["margin"] for r in rows if r["held"] and r["split"] == sp])
                    for sp in sorted({r["split"] for r in rows if r["held"]})}
        summ = {"smoke_sign_agree": f"{len(ok) - len(dis)}/{len(ok)}",
                "smoke_sign_pass": all(abs(r["margin"]) < 0.5 for r in dis),
                "smoke_mag_max_abs_diff": round(max(mag), 4), "smoke_mag_pass": max(mag) < 0.3,
                "held_n": len(held), "held_median": med(held), "held_by_split": by_split,
                "held_min_max": [held[0], held[-1]] if held else None,
                "dispatch_other_median": med(disp), "dispatch_other_min": disp[0] if disp else None,
                "expr_median": med(expr), "expr_max": expr[-1] if expr else None,
                "mass_two_min": min(r["mass_two"] for r in rows)}
        print(tag, json.dumps(summ), flush=True)
        res["arms"][tag] = {"summary": summ, "rows": rows}
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)     # after every arm
        del model, base
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
