"""Student understander (ledger #34): fine-tune a small Qwen to transpile a pt-BR request into the spec JSON.

His words (10-26): "extrair de um qwen maior e retreinar ou fine tuning... que faça essa transpilação... um json".
Why now: with the Qwen-3B understander (few-shot + grammar + 2 send-back rounds) the belt passes 78.8%, and all 38
failures are understander errors, mostly nested formulas read into the wrong tree (e2e traces_q3b_sb2.jsonl).

Student = Qwen2.5-Coder-0.5B-Instruct + LoRA, labels = generator gold (closed world, gold is free).
Train data never contains a test item:
  - dispatch: op lists from multifn train split (<= 3 ops); formulas: compose train split (<= 3 calls);
  - arm s5 adds random 4-5 call trees, excluding every compose test set (test_same, calls4..8);
  - frames FRAMES_D[3] and FRAMES_X[1] are held out of training, so "seen frame" vs "unseen frame" is measured;
  - any training text identical to a test text is dropped (count reported).
The OPEN requests ("calculadora científica") are never in training: the student has to fail or guess them.
Eval: greedy, batch 1 (no padding), on the 179 e2e requests; writes {id: {raw, ms}} for `e2e_pipeline.py run
--understander file`, which runs the real planner + writer + bwrap on it.
Prediction (stated before the run): s3 test_same >= 0.9 exact, calls4/5 <= 0.5 (length generalisation);
s5 calls4/5 >= 0.9; unseen frame within 10 points of seen frame; open split 0-2/4.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose as CP  # noqa: E402
import e2e_pipeline as E  # noqa: E402
import multifn as MF  # noqa: E402

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
CACHE = "/media/llm-data/models/hf/hub"
S_SYS = "Turn the Portuguese request into the JSON spec for the code generator. Answer with the JSON only."
HOLD_D, HOLD_X = E.FRAMES_D[3], E.FRAMES_X[1]


def frame_of(text):
    for f in E.FRAMES_D + E.FRAMES_X:
        head = f.split("{")[0]
        if text.startswith(head):
            return f
    return None


def train_pairs(arm, seed, tests):
    r = random.Random(seed)
    md, cd = MF.Data(), CP.Data()
    fd = [f for f in E.FRAMES_D if f != HOLD_D]
    fx = [f for f in E.FRAMES_X if f != HOLD_X]
    out = []
    for ops in md.sets["train"]:
        for _ in range(2):
            text = r.choice(fd).format(L=E.say_list([r.choice(E.PT[o]) for o in ops]))
            out.append((text, {"main": "dispatch", "calls": list(ops)}))
    trees = list(cd.sets["train"])
    if arm == "s5":
        banned = {tuple(p) for k in cd.sets if k != "train" for p in cd.sets[k]}
        banned |= {tuple(q["prefix"]) for q in tests if "prefix" in q}
        extra = set()
        while len(extra) < 3000:
            p = tuple(CP.rand_tree(r, r.choice([4, 5])))
            if p not in banned:
                extra.add(p)
        trees += sorted(extra)
    for p in trees:
        out.append((r.choice(fx).format(P=E.verbalize(p, r)), {"main": "expr", "expr": E.expr_of(p)}))
    test_texts = {q["text"] for q in tests}
    kept = [(t, g) for t, g in out if t not in test_texts]
    r.shuffle(kept)
    return kept, len(out) - len(kept)


def prompt(text):
    return E.chat(S_SYS, [], text)


def encode(tok, text, gold):
    p = tok(prompt(text), add_special_tokens=False).input_ids
    y = tok(json.dumps(gold) + "<|im_end|>", add_special_tokens=False).input_ids
    return p, y


def collate(rows, pad):
    L = max(len(p) + len(y) for p, y in rows)
    ids = torch.full((len(rows), L), pad)
    lab = torch.full((len(rows), L), -100)
    am = torch.zeros((len(rows), L), dtype=torch.long)
    for i, (p, y) in enumerate(rows):
        n = len(p) + len(y)
        ids[i, :n] = torch.tensor(p + y)
        lab[i, len(p):n] = torch.tensor(y)
        am[i, :n] = 1
    return ids, lab, am


def loss_of(model, ids, lab, am):
    """lm_head only on target positions: the full [B, T, 151k] logits would cost ~1 GB."""
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=am).last_hidden_state
    y = lab[:, 1:]
    m = y != -100
    logits = base.lm_head(h[:, :-1][m]).float()
    return F.cross_entropy(logits, y[m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="e2e run dir (holds requests.json)")
    ap.add_argument("--arm", choices=["s3", "s5"], default="s3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=4, help="micro-batch for gradient accumulation")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--mem-frac", type=float, default=0.35)
    ap.add_argument("--smoke", action="store_true", help="30 steps, eval on 12 requests")
    ap.add_argument("--lora", default=None, help="eval only: saved adapter dir (no training)")
    ap.add_argument("--eval-dir", default=None, help="eval only: run dir whose requests.json is evaluated")
    ap.add_argument("--extra", nargs="+", default=None,
                    help="augmentation: JSON files [{text, gold}] (paraphrase_requests --from-train)")
    ap.add_argument("--extra-reps", type=int, default=3, help="times each extra pair is repeated in the train set")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="more request files whose texts/golds may not be trained on (default: <out>/para/requests.json)")
    ap.add_argument("--steps", type=int, default=0, help="fixed optimizer steps (like-for-like with another arm)")
    ap.add_argument("--tag-suffix", default="p", help="arm name suffix when --extra is used (s5p, s5v2, ...)")
    a = ap.parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.manual_seed(a.seed)
    dev = "cuda"
    tag = (f"student_{a.arm}{a.tag_suffix if a.extra else ''}_s{a.seed}" + ("_smoke" if a.smoke else "") if not a.lora else
           os.path.basename(os.path.dirname(os.path.normpath(a.lora))))   # eval only: name of the trained arm
    odir = os.path.join(a.out, tag)
    os.makedirs(odir, exist_ok=True)
    tests = E.load_requests(a.out)
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=True)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir=CACHE, local_files_only=True,
                                                 dtype=torch.bfloat16).to(dev)
    if a.lora:                                           # eval only, on another request set (e.g. rewritten text)
        assert a.eval_dir and os.path.abspath(a.eval_dir) != os.path.abspath(a.out), "eval-dir must be another run dir"
        odir = os.path.join(a.eval_dir, tag)
        os.makedirs(odir, exist_ok=True)
        model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
        res = evaluate(model, tok, im_end, json.load(open(os.path.join(a.eval_dir, "requests.json"))), dev)
        res.update({"tag": tag, "lora": a.lora, "eval_dir": a.eval_dir})
        json.dump(res.pop("raws"), open(os.path.join(odir, "raws.json"), "w"), ensure_ascii=False, indent=0)
        json.dump(res, open(os.path.join(odir, "result.json"), "w"), indent=1, ensure_ascii=False)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        return
    pairs, dropped = train_pairs(a.arm, a.seed, tests)
    n_extra, extra_dropped = 0, 0
    if a.extra:              # never a test request (sets A and B): same text or same gold (bar `one`); no held-out frame
        excl = a.exclude if a.exclude is not None else [p for p in [os.path.join(a.out, "para", "requests.json")]
                                                         if os.path.exists(p)]
        allq = tests + [q for f in excl for q in json.load(open(f))]
        bad_t = {q["text"] for q in allq} | {q["orig_text"] for q in allq if "orig_text" in q}
        bad_g = {json.dumps(q["gold"], sort_keys=True) for q in allq if q["split"] != "one"}
        raw_ex = [x for f in a.extra for x in json.load(open(f))]
        held = [f.split("{")[0].strip().lower() for f in (HOLD_D, HOLD_X)] + ["preciso de uma função main", "escolha entre"]
        ex = [(x["text"], x["gold"]) for x in raw_ex   # held-frame wording anywhere, any case
              if x["text"] not in bad_t and json.dumps(x["gold"], sort_keys=True) not in bad_g
              and not any(h in x["text"].lower() for h in held)]
        n_extra, extra_dropped = len(ex), len(raw_ex) - len(ex)
        pairs = pairs + ex * a.extra_reps
        random.Random(a.seed).shuffle(pairs)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc = [encode(tok, t, g) for t, g in pairs]
    steps = 30 if a.smoke else a.steps or max(1, int(a.epochs * len(enc) / a.batch))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    print(json.dumps({"tag": tag, "train_pairs": len(enc), "dropped_equal_to_test": dropped, "extra": n_extra,
                      "extra_dropped": extra_dropped,
                      "steps": steps,
                      "lora_params": n_lora, "held_out_frames": [HOLD_D, HOLD_X]}, ensure_ascii=False), flush=True)
    model.train()
    t0, r, log = time.perf_counter(), random.Random(a.seed), []
    order = list(range(len(enc)))
    for s in range(steps):
        if s * a.batch % len(enc) < a.batch:
            r.shuffle(order)
        lr = a.lr * min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = [order[(s * a.batch + i) % len(enc)] for i in range(a.batch)]
        opt.zero_grad()
        loss = 0.0
        for j in range(0, a.batch, a.micro):            # gradient accumulation: same math as one batch
            ids, lab, am = (x.to(dev) for x in collate([enc[i] for i in idx[j:j + a.micro]], tok.pad_token_id))
            part = loss_of(model, ids, lab, am) * (len(idx[j:j + a.micro]) / a.batch)
            part.backward()
            loss += part.detach()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if s % 25 == 0 or s == steps - 1:
            log.append([s, round(loss.item(), 4)])
            print(json.dumps({"step": s, "loss": round(loss.item(), 4), "lr": round(lr, 6),
                              "s": round(time.perf_counter() - t0, 1),
                              "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
    train_s = round(time.perf_counter() - t0, 1)
    if not a.smoke:
        model.save_pretrained(os.path.join(odir, "lora"))
    model = model.merge_and_unload()                     # fused weights: same outputs, no LoRA overhead per token
    model.eval()
    ev = evaluate(model, tok, im_end, tests[:6] + tests[-6:] if a.smoke else tests, dev)
    raws = ev.pop("raws")
    res = {"tag": tag, "arm": a.arm, "seed": a.seed, "train_pairs": len(enc), "dropped_equal_to_test": dropped,
           "extra": n_extra, "extra_dropped": extra_dropped, "extra_file": a.extra, "extra_reps": a.extra_reps,
           "steps": steps, "train_s": train_s, "lora_params": n_lora, "loss_log": log, **ev}
    json.dump(raws, open(os.path.join(odir, "raws.json"), "w"), ensure_ascii=False, indent=0)
    json.dump(res, open(os.path.join(odir, "result.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False), flush=True)


def evaluate(model, tok, im_end, evalq, dev):
    raws, rows = {}, []
    with torch.no_grad():
        for q in evalq:
            ids = torch.tensor([tok(prompt(q["text"]), add_special_tokens=False).input_ids], device=dev)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=120,
                                 do_sample=False, eos_token_id=im_end, pad_token_id=tok.pad_token_id,
                                 repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
            # plain greedy: the model's generation_config sets repetition_penalty 1.05 (hits repeated ops in trees)
            torch.cuda.synchronize()
            ms = round(1000 * (time.perf_counter() - t1), 1)
            raw = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            raws[q["id"]] = {"raw": raw, "ms": ms}
            try:
                exact = json.loads(raw) == q["gold"]
            except json.JSONDecodeError:
                exact = False
            fr = frame_of(q["text"])                     # a rewrite that happens to echo a frame head is still a rewrite
            frame = ("rewritten" if "orig_text" in q else "open" if fr is None else
                     "unseen" if fr in (HOLD_D, HOLD_X) else "seen")
            rows.append({"id": q["id"], "split": q["split"], "exact": exact, "ms": ms, "frame": frame})
    hl = [x for x in rows if x["split"] not in ("one", "open")]   # one: its op list is in training; open: our gold
    agg = {}
    for key in ("split", "frame"):
        for row in rows:
            d = agg.setdefault(f"{key}:{row[key]}", [0, 0])
            d[0] += row["exact"]
            d[1] += 1
    return {"exact": round(sum(x["exact"] for x in rows) / len(rows), 4),
            "exact_headline": round(sum(x["exact"] for x in hl) / max(1, len(hl)), 4),
            "exact_by": {k: f"{v[0]}/{v[1]}" for k, v in sorted(agg.items())},
            "ms_median": sorted(x["ms"] for x in rows)[len(rows) // 2],
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "raws": raws}


if __name__ == "__main__":
    main()
