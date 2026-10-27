"""I5 (master queue 10-27, ledger #5 / #22 / #34 / #41 / #55 / #56): the 5 scientific ops of sciops.py wired into OUR
belt behind new files (multifn.py and the other modules of the running queue are not edited).

His words: "gere python de verdade e executavel... calculadora cientifica" (#5) and "calculadora cientifica de certa
forma é facil, pois já fez, agora precisamos ir além" (#55). Target request, his phrasing:
  "crie uma calculadora científica com seno, cosseno, exponencial, log, soma e divisão"
  -> understander (Qwen2.5-Coder-0.5B + LoRA, spec JSON) -> check_spec_x (rule) -> planner (blockaddr net, 20-op
     table) -> writer (byte GRU, 20-op table) -> assembler (rule) -> bwrap tests.
What does the work (gate 11): the understander is a learned transpiler; the planner learns each op's dependency
closure from the name; the WRITER IS A LEARNED LOOKUP (one symbol -> one fixed body text, 20 bodies); bodies of the
sci ops are hand-written in sciops.py (Taylor / atanh series, no imports). This item wires and measures the belt, it
does not make the writer compose new code (that is I6).

Arms (all on GPU in the full run; seeds paired: understander seed k <-> nets seed k-1, same data-order seed):
  nets, per seed s in 0 1 2 (multifn recipes, equal step budgets):
    planner P_s     blockaddr "addr" net on the 20-op Data (train 1-3 op specs, belt sci specs excluded), 2000 steps
    W0_s  POSITIVE CONTROL: old tables (15 ops), fresh, multifn recipe 1500 steps, d64 -> must give the old writer
          number: gold-plan pass 15/15 ops + full 15-op calculator (e2e ckpt meta eval_dispatch test_comb / len4 /
          len8 / full pass1 = 1.0; multifn_il 3/3 seeds 100%, README sec 7). Also the e2e ckpt writer.pt itself is
          loaded into WriterX and must give 15/15 (eval path parity).
    WE_s  W0_s grown to the 20-op tables (WriterX.from_mf: old rows copied, new rows fresh init), +1500 steps on the
          20-op pool (old + new symbols, uniform)
    WF_s  fresh d64 on the 20-op pool, 1500 steps (same recipe as W0)
    WF128_s fresh d128, 1500 steps (capacity arm; sci bodies are up to 346 bytes vs 93 for the old ops)
  understander, per seed k in 1 2 3 (student_understander s5 recipe: LoRA r16 a32 on q/k/v/o/gate/up/down, AdamW
          2e-4 no wd, batch 32 (micro 4), warmup 30 + cosine, 1 epoch, prompt S_SYS, greedy max_new 120):
    US_k  s5 train pairs (data seed k) + 1500 sci pairs (TRAIN frames only), shuffled with Random(k); steps =
          len(pairs) // 32
    UC_k  equal-budget control: the same s5 pairs (data seed k), NO sci pairs, the same number of steps as US_k
          (gate: UC_1 step-0 loss must equal s5's logged 2.607 +- .02 -> same pairs/order/loss path as s5; LoRA B=0 so NOT an init check)
    s5    the existing adapter (e2e/student_s5_s1), regenerated through this file's gen on set A (parity with the
          recorded raws) and on SCI (the "before" row: it has never seen a sci op)
  baselines on SCI (same tests): const = most common calls list among the sci TRAIN labels; const_all = all 20 ops
          (passes lenient, must fail strict); rule = keyword matcher over the same pt-BR word lists (one-cue rule,
          gate 1); q3b_direct = frozen Qwen2.5-Coder-3B-Instruct writes the whole program from the request
          (few-shot with 2 old-op programs, the 20-op table with each op's exact semantics, `import math` allowed,
          greedy, max_new 900) -> same bwrap tests. Greedy is deterministic: one run, no seeds.
Sets:
  SCI   151 held-out pt-BR requests: 150 random sci calculators (2-6 ops, >= 1 sci op, op set never a training spec,
        frames disjoint from every training frame) + his request verbatim. Gold = dispatch calls in the order said.
  A, B  e2e requests.json (179 templated) and para/requests.json (174 Sonnet rewrites), graded by e2e_pipeline.run_belt
        unchanged (grader v1 = how s5's 97.8 / 85.6 were measured).
Tests (multifn_sci.grade, bwrap): every requested op through main(op, ...): sci ops on 16 held-out floats each vs
  math.* (|got - want| <= 1e-6 |want| + 1e-9; log(x <= 0) -> None), old ops on multifn.CASES vs the multifn reference
  module; STRICT: main(other id, 1.5, 2) is None for every one of the 20 ids not asked + 2 unknown ids.
  pass = all requested ops right AND strict. exec = module loads and nothing raises that the truth does not.
Writer on gold plan: per op, plan_of((op,)) written entirely by the writer, assembled, graded as above.

PREDICTIONS (written 13:24 10-27, before any GPU run; CPU smoke only proves plumbing):
  P1 positive control: W0 15/15 ops + full-15 pass, 3/3 seeds; e2e ckpt writer 15/15 through WriterX.
  P2 writer, gold plan, sci ops (5): WF64 mean >= 4/5 per seed; WE64 >= WF64 - 0.3 op; WF128 5/5 in >= 2/3 seeds;
     every extended arm keeps the 15 old ops 15/15 (3/3 seeds). Full 20-op program: pass in >= 2/3 seeds for WF128.
  P3 planner P_s plan_exact (gold spec): test_comb >= .95, len4 >= .9, len6 >= .7 (review #44: len6 mean .76 for 7
     seeds of the old planner); SCI gold specs >= .85 mean.
  P4 understander on SCI: US spec set-exact >= 90% mean over 3 seeds; his request right in >= 2/3 seeds;
     UC and s5 <= 5% (they never saw a sci id).
  P5 belt on SCI (strict pass, mean of 3 seeds): US + P + WF128 >= 80%; US + P + WE64 >= 75%; US + P + WF64 >= 70%.
     exec rate >= 90% for US + P + WF128.
  P6 baselines on SCI strict pass: const <= 2%, const_all 0% (lenient ~100%), rule >= 85% (the templated set is
     keyword-shaped; the rule is a real competitor here, not a floor), q3b_direct 40-75%.
  P7 regression (grader v1, net planner/writer): US mean over 3 seeds on A >= 96.8 (s5 97.8 - 1) and on B >= 84.6
     (s5 85.6 - 1); |US - UC| <= 1.5 pt on A and on B (paired by seed). The 4 'open' A requests ("calculadora
     científica" without a list, gold = add sub mul div sqrt pow) may drop to 0/4 for US (s5 1/4): reported apart.
AMENDMENT 13:35 10-27 (after the CPU smoke, before any GPU run; P1-P7 above are NOT changed): the s5 adapter,
  which never saw a sci id, answered 3/3 smoke SCI requests exactly (incl. his request): the ids sin/cos/exp/log/
  tan are the English names the base model already knows, so the old belt failed sci requests in check_spec
  (oov_op), not in the understander. P4's 's5 <= 5%' is therefore expected to FAIL; new prediction: s5 spec
  set-exact on SCI >= 60%, US >= s5 + 10 pt. Row s5+P_s+W (old understander, new nets) added to the report.
KILL / adapt: if P1 fails, nothing else is read (eval path bug). If the writer arms fail P2 at 1500 steps, adapt with
  more steps (equal across arms), not by dropping the arm.

Subcommands:
  nets         --out O --device cuda --seeds 0 1 2 [--planner-steps 2000 --writer-steps 1500]
  understander --out O --arm US|UC --seed k [--device cuda --mem-frac 0.35] [--smoke]
  gen          --out O --lora DIR --name s5 [--sets A SCI]          (eval only of an existing adapter)
  qwen3b       --out O [--model Qwen/Qwen2.5-Coder-3B-Instruct --mem-frac 0.75]
  report       --out O                                             (CPU: belts, baselines, regression, predictions)
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2e_pipeline as E  # noqa: E402
import multifn as MF  # noqa: E402
import multifn_sci as X  # noqa: E402
import student_understander as SU  # noqa: E402

RUN = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
E2E = f"{RUN}/e2e"
S5_LORA = f"{E2E}/student_s5_s1/lora"
S5_LOSS0 = 2.607
S5 = {"A": 0.9777, "B": 0.8563}
NET_SEEDS = (0, 1, 2)
U_SEEDS = (1, 2, 3)

# ----------------------------------------------------------------------------------------------- requests (pt-BR)

PT_SCI = {"sin": ["seno", "seno de um ângulo"], "cos": ["cosseno", "cosseno de um ângulo"], "tan": ["tangente"],
          "exp": ["exponencial", "função exponencial", "e elevado a x"],
          "log": ["log", "logaritmo", "logaritmo natural"]}
WORDS = {**E.PT, **PT_SCI}
FRAMES_TRAIN = ["faça uma calculadora científica que tenha {L}", "quero uma calculadora em python com as funções {L}",
                "monte uma calculadora científica com as operações: {L}"] + [f for f in E.FRAMES_D if f != SU.HOLD_D]
FRAMES_EVAL = ["crie uma calculadora científica com {L}",                    # his phrasing
               "preciso de uma calculadora científica que calcule {L}",
               "gere um programa python executável de calculadora científica com {L}",
               SU.HOLD_D]                                                   # s5's held frame
HIS = ("crie uma calculadora científica com seno, cosseno, exponencial, log, soma e divisão",
       ["sin", "cos", "exp", "log", "add", "div"])
assert not set(FRAMES_TRAIN) & set(FRAMES_EVAL)


def say(r, ops):
    return E.say_list([r.choice(WORDS[o]) for o in ops])


def sci_eval_requests():
    r, out, seen = random.Random(5601), [], set()
    for k in (2, 3, 4, 5, 6):
        n = 0
        while n < 30:
            ks = r.randint(1, min(5, k))
            ops = r.sample(X.SCI, ks) + r.sample(MF.OPS, k - ks)
            r.shuffle(ops)
            key = tuple(sorted(ops))
            if key in seen:
                continue
            seen.add(key)
            text = r.choice(FRAMES_EVAL).format(L=say(r, ops))
            out.append({"id": f"s{len(out):03d}", "world": "dispatch", "split": f"sci_len{k}", "text": text,
                        "caps": ops, "gold": {"main": "dispatch", "calls": ops}})
            n += 1
    out.append({"id": "his", "world": "dispatch", "split": "his", "text": HIS[0], "caps": HIS[1],
                "gold": {"main": "dispatch", "calls": HIS[1]}})
    return out


def sci_train_pairs(seed, evalq, n=1500):
    """sci requests for training: 1-4 ops, >= 1 sci op, op SET never one of the eval sets, TRAIN frames only."""
    r = random.Random(1000 + seed)
    banned = {tuple(sorted(q["caps"])) for q in evalq}
    out = []
    while len(out) < n:
        k = r.choice([1, 2, 2, 3, 3, 4])
        ks = r.randint(1, min(5, k))
        ops = r.sample(X.SCI, ks) + r.sample(MF.OPS, k - ks)
        r.shuffle(ops)
        if tuple(sorted(ops)) in banned:
            continue
        out.append((r.choice(FRAMES_TRAIN).format(L=say(r, ops)), {"main": "dispatch", "calls": ops}))
    return out


def load_sets(out):
    p = os.path.join(out, "sci_requests.json")
    if not os.path.exists(p):
        json.dump(sci_eval_requests(), open(p, "w"), ensure_ascii=False, indent=1)
    return {"A": json.load(open(f"{E2E}/requests.json")), "B": json.load(open(f"{E2E}/para/requests.json")),
            "SCI": json.load(open(p))}


def leak_report(evalq, pairs, sets):
    tr_t = {t for t, _ in pairs}
    tr_sets = {tuple(sorted(g["calls"])) for _, g in pairs if g["main"] == "dispatch" and set(g["calls"]) & set(X.SCI)}
    return {"sci_text_in_train": sum(q["text"] in tr_t for q in evalq),
            "sci_opset_in_sci_train": sum(tuple(sorted(q["caps"])) in tr_sets for q in evalq),
            "A_text_in_train": sum(q["text"] in tr_t for q in sets["A"]),
            "B_text_in_train": sum(q["text"] in tr_t for q in sets["B"]),
            "eval_frame_heads_in_train": sum(any(t.startswith(f.split("{")[0]) for f in FRAMES_EVAL) for t in tr_t
                                             if any(o in t for o in ("seno", "cosseno", "tangente", "exponencial",
                                                                     "log")))}

# ----------------------------------------------------------------------------------------------- spec bridge


def check_spec_x(raw):
    """raw -> (ops tuple | None, flags). Dispatch only (the sci ops are unary; the compose world stays binary)."""
    try:
        s = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, ["json_error"]
    if not isinstance(s, dict) or s.get("main") != "dispatch":
        return None, ["not_dispatch"]
    calls = s.get("calls")
    if not isinstance(calls, list) or not all(isinstance(c, str) for c in calls):
        return None, ["schema_error"]
    fl = [f"oov_op:{c}" for c in calls if c not in X.OPS]
    ops = tuple(dict.fromkeys(c for c in calls if c in X.OPS))
    return (ops or None), fl + ([] if ops else ["empty_spec"])


def rule_spec(text):
    """one-cue baseline: longest pt-BR word of the lists found in the text -> op, in order of appearance."""
    t, hits = text.lower(), []
    phrases = sorted(((w, o) for o, ws in WORDS.items() for w in ws), key=lambda x: -len(x[0]))
    taken = [False] * len(t)
    for w, o in phrases:
        for m in re.finditer(re.escape(w.lower()), t):
            if not any(taken[m.start():m.end()]) and (m.start() == 0 or not t[m.start() - 1].isalnum()) and \
                    (m.end() == len(t) or not t[m.end()].isalnum()):
                hits.append((m.start(), o))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    ops = list(dict.fromkeys(o for _, o in sorted(hits)))
    return json.dumps({"main": "dispatch", "calls": ops})

# ----------------------------------------------------------------------------------------------- understander


def gen(model, tok, im_end, reqs, dev, max_new=120):
    """= student_understander.evaluate without mask/scores (same prompt, greedy, batch 1, no padding)."""
    raws = {}
    with torch.no_grad():
        for q in reqs:
            ids = torch.tensor([tok(SU.prompt(q["text"]), add_special_tokens=False).input_ids], device=dev)
            t1 = time.perf_counter()
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                 do_sample=False, eos_token_id=im_end, pad_token_id=tok.pad_token_id,
                                 repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
            raws[q["id"]] = {"raw": tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(),
                             "ms": round(1000 * (time.perf_counter() - t1), 1)}
    return raws


def load_base(dev, model_name=SU.MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=SU.CACHE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=SU.CACHE, local_files_only=True,
                                                 dtype=torch.bfloat16).to(dev)
    return tok, model, tok.convert_tokens_to_ids("<|im_end|>")


def sub_sets(sets, smoke):
    return {k: (v[:2] + v[-1:] if smoke else v) for k, v in sets.items()}


def cmd_understander(a):
    from peft import LoraConfig, get_peft_model
    dev = a.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.set_num_threads(a.threads)
    sets = load_sets(a.out)
    tag = f"{a.arm}_s{a.seed}" + ("_smoke" if a.smoke else "")
    odir = os.path.join(a.out, "u", tag)
    if os.path.exists(os.path.join(odir, "result.json")):
        print("done already:", odir)
        return
    os.makedirs(odir, exist_ok=True)
    torch.manual_seed(a.seed)                            # same order as student_understander.main
    old, dropped = SU.train_pairs("s5", a.seed, sets["A"])
    sci = sci_train_pairs(a.seed, sets["SCI"])
    us = old + sci
    random.Random(a.seed).shuffle(us)
    n_steps = len(us) // a.batch                          # US budget; UC gets the same number of steps
    pairs = us if a.arm == "US" else old
    leak = leak_report(sets["SCI"], pairs, sets)
    assert leak["sci_text_in_train"] == 0 and leak["sci_opset_in_sci_train"] == 0 and leak["eval_frame_heads_in_train"] == 0, leak
    tok, model, im_end = load_base(dev)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    enc = [SU.encode(tok, t, g) for t, g in pairs]
    steps = a.smoke_steps if a.smoke else n_steps
    batch = 2 if a.smoke else a.batch
    micro = 1 if a.smoke else a.micro
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    head = {"tag": tag, "arm": a.arm, "seed": a.seed, "train_pairs": len(enc), "s5_pairs": len(old),
            "sci_pairs": len(sci) if a.arm == "US" else 0, "dropped_equal_to_A": dropped, "steps": steps,
            "us_steps": n_steps, "leak": leak}
    print(json.dumps(head, ensure_ascii=False), flush=True)
    model.train()
    t0, r, log = time.perf_counter(), random.Random(a.seed), []
    order = list(range(len(enc)))
    for s in range(steps):                               # = student_understander.main's loop
        if s * batch % len(enc) < batch:
            r.shuffle(order)
        lr = 2e-4 * min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = [order[(s * batch + i) % len(enc)] for i in range(batch)]
        opt.zero_grad()
        loss = 0.0
        for j in range(0, batch, micro):
            ids, lab, am = (x.to(dev) for x in SU.collate([enc[i] for i in idx[j:j + micro]], tok.pad_token_id))
            part = SU.loss_of(model, ids, lab, am) * (len(idx[j:j + micro]) / batch)
            part.backward()
            loss += part.detach()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if s % 25 == 0 or s == steps - 1:
            log.append([s, round(float(loss), 4)])
            print(json.dumps({"step": s, "loss": round(float(loss), 4), "lr": round(lr, 6),
                              "s": round(time.perf_counter() - t0, 1)}), flush=True)
    gate = None
    if a.arm == "UC" and a.seed == 1 and not a.smoke:
        gate = abs(log[0][1] - S5_LOSS0) <= 0.02
        print(json.dumps({"gate_step0_loss_vs_s5": log[0][1], "s5": S5_LOSS0, "ok": gate}), flush=True)
    train_s = round(time.perf_counter() - t0, 1)
    if not a.smoke:
        model.save_pretrained(os.path.join(odir, "lora"))
    model = model.merge_and_unload().eval()
    for k, reqs in sub_sets(sets, a.smoke).items():
        raws = gen(model, tok, im_end, reqs, dev)
        json.dump(raws, open(os.path.join(odir, f"raws_{k}.json"), "w"), ensure_ascii=False, indent=0)
        print(k, "exact", sum(_exact(raws[q["id"]]["raw"], q) for q in reqs), "/", len(reqs), flush=True)
    json.dump({**head, "loss_log": log, "train_s": train_s, "gate_loss0": gate}, open(os.path.join(odir, "result.json"), "w"),
              indent=1, ensure_ascii=False)


def _exact(raw, q):
    try:
        return E.canon_spec(json.loads(raw)) == q["gold"]
    except json.JSONDecodeError:
        return False


def cmd_gen(a):
    from peft import PeftModel
    dev = a.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.set_num_threads(a.threads)
    sets = sub_sets(load_sets(a.out), a.smoke)
    odir = os.path.join(a.out, "u", a.name + ("_smoke" if a.smoke else ""))
    os.makedirs(odir, exist_ok=True)
    tok, model, im_end = load_base(dev)
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
    rp = os.path.join(odir, "gen_result.json")
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    for k in a.sets:
        p = os.path.join(odir, f"raws_{k}.json")
        if os.path.exists(p):
            continue
        raws = gen(model, tok, im_end, sets[k], dev)
        json.dump(raws, open(p, "w"), ensure_ascii=False, indent=0)
        res[k] = sum(_exact(raws[q["id"]]["raw"], q) for q in sets[k])
        if k == "A" and a.name == "s5":                    # parity with the recorded s5 eval (gate 12 style)
            rec = json.load(open(f"{E2E}/student_s5_s1/raws.json"))
            res["A_parity"] = sum(rec[q["id"]]["raw"] == raws[q["id"]]["raw"] for q in sets[k])
        print(k, json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(odir, "gen_result.json"), "w"), indent=1)

# ----------------------------------------------------------------------------------------------- Coder-3B direct


Q_SYS = """You write one complete Python module from a request written in Portuguese.
The module must define main(op, a, b=0): op is an operation id (a string); main returns that operation applied to a
(and b). Support exactly the operations the request asks for; for any other op id main must return None.
Operation ids and their exact meaning:
add: a + b | sub: a - b | mul: a * b | div: a / b, None when b == 0 | mod: a % b, None when b == 0
maxx: the larger of a and b | sqrt: a ** 0.5, None when a < 0 | neg: -a | abs: absolute value of a | square: a * a
pow: product of b factors a (1 when b < 1) | fact: product 2 * 3 * ... * a (1 when a < 2)
hyp: sqrt(a*a + b*b) | avg: (a + b) / 2 | pct: a * b / 100
sin: sine of a (radians) | cos: cosine of a | tan: tangent of a | exp: e to the power a | log: natural log of a, None
when a <= 0
You may import math. No input(), no printing, no files. Answer with the code only, in one ```python block."""
Q_SYS += "\nPortuguese words used for each id:\n" + "\n".join(f"{o}: " + ", ".join(WORDS[o]) for o in X.OPS)
Q_SHOTS = [("crie uma calculadora que faça soma e raiz quadrada", ("add", "sqrt")),
           ("quero um programa com fatorial, média de dois números e subtração", ("fact", "avg", "sub"))]


def q_prompt(text):
    shots = [(u, "```python\n" + X.assemble(X.plan_of(ops)) + "```") for u, ops in Q_SHOTS]
    return E.chat(Q_SYS, shots, text)


def extract_code(s):
    m = re.findall(r"```(?:python)?\n(.*?)```", s, re.S)
    if m:
        return m[0]
    return re.sub(r"^\s*```(?:python)?\n", "", s)          # cut at max_new: strip the opening fence


def cmd_qwen3b(a):
    dev = a.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.set_num_threads(a.threads)
    reqs = sub_sets(load_sets(a.out), a.smoke)["SCI"]
    p = os.path.join(a.out, "u", "q3b_direct" + ("_smoke" if a.smoke else ""), "programs.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tok, model, im_end = load_base(dev, a.model)
    model.eval()
    res = json.load(open(p)) if os.path.exists(p) else {}
    with torch.no_grad():
        for q in reqs:
            if q["id"] in res:
                continue
            ids = torch.tensor([tok(q_prompt(q["text"]), add_special_tokens=False).input_ids], device=dev)
            t1 = time.perf_counter()
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.max_new,
                                 do_sample=False, eos_token_id=im_end, pad_token_id=tok.pad_token_id,
                                 repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)
            raw = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            res[q["id"]] = {"raw": raw, "ms": round(1000 * (time.perf_counter() - t1), 1),
                            "new_tokens": int(out.shape[1] - ids.shape[1])}
            json.dump(res, open(p, "w"), ensure_ascii=False, indent=0)
    print("q3b programs", len(res), "capped at max_new:", sum(v["new_tokens"] >= a.max_new for v in res.values()))

# ----------------------------------------------------------------------------------------------- nets


def writer_eval(net):
    """gold plan, every symbol written by the net: per op its own program; plus the full program of the net's ops."""
    ops = [s[4:] for s in net.syms if s.startswith("def:")]
    plans = {o: X.plan_of((o,)) for o in ops}
    full = X.plan_of(tuple(ops))
    texts = net.texts({s for p in list(plans.values()) + [full] for s in p})
    items = [X.program_items(X.assemble(plans[o], texts.get), (o,)) for o in ops]
    items.append(X.program_items(X.assemble(full, texts.get), tuple(ops)))
    rows = X.run_all(items)
    g = [X.grade(r, (o,)) for r, o in zip(rows, ops)]
    gf = X.grade(rows[-1], tuple(ops))
    return {"op_pass": {o: x["pass"] for o, x in zip(ops, g)},
            "text_exact": {s: texts[s] == X.block_text(s) for s in texts},
            "old_pass": sum(x["pass"] for o, x in zip(ops, g) if o in MF.OPS),
            "sci_pass": sum(x["pass"] for o, x in zip(ops, g) if o in X.SCI),
            "full_pass": gf["pass"], "full_n_ops": len(ops)}


def mf_old_writer_state(path):
    return {k: v.cpu() for k, v in torch.load(path, map_location="cpu").items()}


def cmd_nets(a):
    torch.set_num_threads(a.threads)
    dev = a.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    sets = load_sets(a.out)
    ndir = os.path.join(a.out, "nets")
    os.makedirs(ndir, exist_ok=True)
    data = X.Data(0, exclude=[q["caps"] for q in sets["SCI"]])
    pool_x = sorted({s for sp in data.sets["train"] for s in X.plan_of(sp) if s != "<eos>"})
    pool_old = sorted({s for sp in MF.Data().sets["train"] for s in MF.plan_of_grouped(sp) if s != "<eos>"})
    assert set(pool_x) == set(X.SYM[3:]) and set(pool_old) == set(MF.SYM[3:]), "pool misses a symbol"
    # eval path parity: the e2e ckpt writer (multifn.Writer, CPU-trained) through WriterX
    ck = X.WriterX.from_mf(mf_old_writer_state(f"{E2E}/ckpt/writer.pt"), 64, MF.SYM, MF.CH)
    ck_ev = writer_eval(ck)
    print(json.dumps({"ckpt_writer_parity": {k: ck_ev[k] for k in ("old_pass", "full_pass")}}), flush=True)
    json.dump(ck_ev, open(os.path.join(ndir, "ckpt_writer_eval.json"), "w"), indent=1)
    for s in a.seeds:
        p = os.path.join(ndir, f"nets_s{s}.json")
        if os.path.exists(p):
            continue
        res, t0 = {"seed": s}, time.perf_counter()
        pl = X.train_planner(data, s, a.planner_steps, dev)
        torch.save(pl.state_dict(), os.path.join(ndir, f"planner_s{s}.pt"))
        pl_cpu = load_planner(os.path.join(ndir, f"planner_s{s}.pt"))
        res["planner_s"] = round(time.perf_counter() - t0, 1)
        ev = {}
        for k in ("test_comb", "len4", "len5", "len6", "sci_len4", "sci5", "full"):
            plans = X.plan_specs(pl_cpu, data.sets[k])
            ev[k] = round(sum(p == X.plan_of(sp) for p, sp in zip(plans, data.sets[k])) / len(data.sets[k]), 4)
        plans = X.plan_specs(pl_cpu, [tuple(q["caps"]) for q in sets["SCI"]])
        ev["SCI_gold_specs"] = round(sum(p == X.plan_of(tuple(q["caps"])) for p, q in zip(plans, sets["SCI"]))
                                     / len(sets["SCI"]), 4)
        res["planner_plan_exact"] = ev
        print(json.dumps({"seed": s, "planner": ev}), flush=True)
        arms = {}
        torch.manual_seed(s)
        w0 = X.WriterX(64, MF.SYM, MF.CH)
        arms["W0"] = (w0, X.train_writer(w0, pool_old, s, a.writer_steps, device=dev))
        we = X.WriterX.from_mf({k: v.cpu() for k, v in w0.to_mf_state().items()}, 64, seed=s)
        arms["WE64"] = (we, X.train_writer(we, pool_x, s, a.writer_steps, device=dev))
        for name, d in (("WF64", 64), ("WF128", 128)):
            torch.manual_seed(s)
            w = X.WriterX(d)
            arms[name] = (w, X.train_writer(w, pool_x, s, a.writer_steps, device=dev))
        for name, (w, last_loss) in arms.items():
            w.cpu()
            torch.save(w.state_dict(), os.path.join(ndir, f"writer_{name}_s{s}.pt"))
            wev = writer_eval(w)
            res[name] = {**wev, "last_loss": round(last_loss, 5), "params": sum(p.numel() for p in w.parameters())}
            print(json.dumps({"seed": s, "arm": name, "old_pass": wev["old_pass"], "sci_pass": wev["sci_pass"],
                              "full_pass": wev["full_pass"], "op_pass": wev["op_pass"], "loss": round(last_loss, 5)}),
                  flush=True)
        res["wall_s"] = round(time.perf_counter() - t0, 1)
        json.dump(res, open(p, "w"), indent=1)


def load_planner(path):
    X.use_planner_globals()
    net = E.BA.Net("addr", gate_bias=-4.0)
    net.load_state_dict({k: v.cpu() for k, v in torch.load(path, map_location="cpu").items()})
    return net.eval()


def load_writer(path, name):
    d = 128 if name.endswith("128") else 64
    net = X.WriterX(d, MF.SYM, MF.CH) if name == "W0" else X.WriterX(d)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()

# ----------------------------------------------------------------------------------------------- report (CPU)


class PlannerX:
    """duck type of e2e_pipeline.Planner: dispatch -> the 20-op planner, expr -> the e2e ckpt compose planner."""

    def __init__(self, net):
        self.old, self.net = E.Planner("net", f"{E2E}/ckpt"), net

    def plan(self, bl):
        out = [None] * len(bl)
        dx = [i for i, b in enumerate(bl) if b[0] == "dispatch"]
        ex = [i for i, b in enumerate(bl) if b[0] != "dispatch"]
        if ex:
            for i, p in zip(ex, self.old.plan([bl[i] for i in ex])):
                out[i] = p
        if dx:
            for i, p in zip(dx, X.plan_specs(self.net, [bl[i][1] for i in dx])):
                out[i] = p
        return out


class WriterWrap:
    def __init__(self, net):
        self.net = net

    def texts(self, syms):
        return self.net.texts(syms)


def sci_belt(reqs, specs, planner, writer):
    """specs[i]: ops tuple or None (no program). planner None = oracle plan; writer None = oracle texts."""
    idx = [i for i, s in enumerate(specs) if s]
    plans = {}
    if idx:
        ps = X.plan_specs(planner, [specs[i] for i in idx]) if planner else [X.plan_of(specs[i]) for i in idx]
        plans = dict(zip(idx, ps))
    need = {s for p in plans.values() for s in p}
    texts = writer.texts(need) if writer else {s: X.block_text(s) for s in need if s in X.SYM[3:]}
    items = [X.program_items(X.assemble(plans[i], lambda s: texts.get(s, "")), tuple(q["caps"])) for i, q in
             enumerate(reqs) if i in plans]
    rows = dict(zip(idx, X.run_all(items)))
    out = []
    for i, q in enumerate(reqs):
        g = X.grade(rows[i], tuple(q["caps"])) if i in rows else X.grade(None, tuple(q["caps"]))
        g["u_set"] = bool(specs[i]) and set(specs[i]) == set(q["caps"])
        g["p_ok"] = i in plans and plans[i] == X.plan_of(specs[i])
        out.append(g)
    return out


def direct_rows(reqs, progs):
    items = [X.program_items(extract_code(progs[q["id"]]["raw"]), tuple(q["caps"])) for q in reqs]
    return [X.grade(r, tuple(q["caps"])) for r, q in zip(X.run_all(items), reqs)]


def agg(rows, reqs):
    n = len(rows)
    r = (lambda k: round(sum(x[k] for x in rows) / n, 4))
    sci_ops = [ok for x in rows for o, ok in x["op_ok"].items() if o in X.SCI]
    res = {"n": n, "pass": r("pass"), "pass_lenient": r("pass_lenient"), "exec": r("exec"), "loads": r("loads"),
           "sci_op_acc": round(sum(sci_ops) / max(1, len(sci_ops)), 4)}
    if "u_set" in rows[0]:
        res["u_set"], res["p_ok"] = r("u_set"), r("p_ok")
    for lo, hi, name in ((2, 4, "len2_4"), (5, 6, "len5_6")):
        sel = [x["pass"] for x, q in zip(rows, reqs) if q["split"].startswith("sci_len") and lo <= int(q["split"][7:]) <= hi]
        res[f"pass_{name}"] = round(sum(sel) / max(1, len(sel)), 4)
    his = [x["pass"] for x, q in zip(rows, reqs) if q["id"] == "his"]
    res["his_pass"] = his[0] if his else None
    return res


def raws_to_specs(raws, reqs):
    return [check_spec_x(raws[q["id"]]["raw"])[0] if q["id"] in raws else None for q in reqs]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def cmd_report(a):
    assert not E.GRADER_V2, "s5 constants are grader v1; unset PYCODE_GRADER"
    torch.set_num_threads(a.threads)
    sets = load_sets(a.out)
    sci, ndir, udir = sets["SCI"], os.path.join(a.out, "nets"), os.path.join(a.out, "u")
    rep = {"sci_n": len(sci), "nets": {}, "sci_belt": {}, "baselines": {}, "regression": {}}
    seeds = [s for s in NET_SEEDS if os.path.exists(f"{ndir}/nets_s{s}.json")]
    useeds = [k for k in U_SEEDS if all(os.path.exists(f"{udir}/{u}_s{k}/raws_SCI.json") for u in ("US", "UC"))]
    if a.smoke:
        seeds, useeds = [0], [1]
    for s in seeds:
        rep["nets"][s] = json.load(open(f"{ndir}/nets_s{s}.json"))
    # baselines (no seed)
    sp_train = sci_train_pairs(1, sci)
    from collections import Counter
    common = Counter(tuple(g["calls"]) for _, g in sp_train).most_common(1)[0][0]
    rep["baselines"]["const_most_common"] = {"spec": list(common), **agg(sci_belt(sci, [common] * len(sci), None, None), sci)}
    rep["baselines"]["const_all20"] = agg(sci_belt(sci, [tuple(X.OPS)] * len(sci), None, None), sci)
    rule_raws = {q["id"]: {"raw": rule_spec(q["text"])} for q in sci}
    rep["baselines"]["rule_oracle_nets"] = agg(sci_belt(sci, raws_to_specs(rule_raws, sci), None, None), sci)
    rep["baselines"]["gold_spec_oracle_nets"] = agg(sci_belt(sci, [tuple(q["caps"]) for q in sci], None, None), sci)
    for name in ("q3b_direct", "q3b_direct_smoke"):
        p = f"{udir}/{name}/programs.json"
        if os.path.exists(p):
            progs = json.load(open(p))
            rq = [q for q in sci if q["id"] in progs]
            rep["baselines"][name] = {**agg(direct_rows(rq, progs), rq), "n": len(rq)}
    s5d = f"{udir}/s5" + ("_smoke" if a.smoke else "")
    if os.path.exists(f"{s5d}/raws_SCI.json"):
        r5 = json.load(open(f"{s5d}/raws_SCI.json"))
        rq = [q for q in sci if q["id"] in r5]
        rep["baselines"]["s5_oracle_nets"] = agg(sci_belt(rq, raws_to_specs(r5, rq), None, None), rq)
        gr = json.load(open(f"{s5d}/gen_result.json")) if os.path.exists(f"{s5d}/gen_result.json") else {}
        rep["s5_parity_A"] = gr.get("A_parity")
    # full belt per seed pair (U seed k <-> nets seed k-1), every writer arm; plus gold spec through the nets
    for k in useeds:
        s = k - 1 if not a.smoke else 0
        if s not in rep["nets"]:
            continue
        P = load_planner(f"{ndir}/planner_s{s}.pt")
        for un in ("US", "UC"):
            ud = f"{udir}/{un}_s{k}" + ("_smoke" if a.smoke else "")
            if not os.path.exists(f"{ud}/raws_SCI.json"):
                continue
            raws = json.load(open(f"{ud}/raws_SCI.json"))
            rq = [q for q in sci if q["id"] in raws]
            specs = raws_to_specs(raws, rq)
            for wn in ("WE64", "WF64", "WF128"):
                W = load_writer(f"{ndir}/writer_{wn}_s{s}.pt", wn)
                rep["sci_belt"][f"{un}_s{k}+P_s{s}+{wn}_s{s}"] = agg(sci_belt(rq, specs, P, W), rq)
            rep["sci_belt"][f"{un}_s{k}+oracle_nets"] = agg(sci_belt(rq, specs, None, None), rq)
    r5 = json.load(open(f"{s5d}/raws_SCI.json")) if os.path.exists(f"{s5d}/raws_SCI.json") else None
    for s in seeds:
        P = load_planner(f"{ndir}/planner_s{s}.pt")
        for wn in ("WE64", "WF64", "WF128"):
            W = load_writer(f"{ndir}/writer_{wn}_s{s}.pt", wn)
            rep["sci_belt"][f"gold+P_s{s}+{wn}_s{s}"] = agg(sci_belt(sci, [tuple(q["caps"]) for q in sci], P, W), sci)
            if r5:                                       # old understander (s5, frozen) through the new nets
                rq = [q for q in sci if q["id"] in r5]
                rep["sci_belt"][f"s5+P_s{s}+{wn}_s{s}"] = agg(sci_belt(rq, raws_to_specs(r5, rq), P, W), rq)
    # regression A / B through e2e_pipeline.run_belt (grader v1), dispatch planner + writer replaced
    lim = 6 if a.smoke else None
    # positive control: recorded s5 raws through the ORIGINAL e2e nets must give exactly S5 (n=179 / 174)
    for setn in ("A", "B"):
        p = f"{E2E}/student_s5_s1/raws.json" if setn == "A" else f"{E2E}/para/student_s5_s1/raws.json"
        raws = json.load(open(p))
        reqs = [q for q in sets[setn] if q["id"] in raws][:lim]
        sm = E.summarize(E.run_belt(reqs, [raws[q["id"]]["raw"] for q in reqs], E.Planner("net", f"{E2E}/ckpt"),
                                    E.Writer("net", f"{E2E}/ckpt")))
        rep["regression"][f"{setn}:s5+e2e_ckpt_nets"] = {"n": sm["n"], "pass": sm["pass"], "u_ok": sm["u_ok"]}
    for k in useeds:
        s = k - 1 if not a.smoke else 0
        if s not in rep["nets"]:
            continue
        P = PlannerX(load_planner(f"{ndir}/planner_s{s}.pt"))
        for un in ("US", "UC", "s5"):
            ud = (f"{udir}/{un}_s{k}" if un != "s5" else f"{E2E}") + ("_smoke" if a.smoke and un != "s5" else "")
            for setn in ("A", "B"):
                if un == "s5":
                    p = f"{E2E}/student_s5_s1/raws.json" if setn == "A" else f"{E2E}/para/student_s5_s1/raws.json"
                else:
                    p = f"{ud}/raws_{setn}.json"
                if not os.path.exists(p):
                    continue
                raws = json.load(open(p))
                reqs = [q for q in sets[setn] if q["id"] in raws][:lim]
                for wn in ("WE64", "WF64", "WF128"):
                    W = WriterWrap(load_writer(f"{ndir}/writer_{wn}_s{s}.pt", wn))
                    T = E.run_belt(reqs, [raws[q["id"]]["raw"] for q in reqs], P, W)
                    sm = E.summarize(T)
                    rep["regression"][f"{setn}:{un}_s{k}+P_s{s}+{wn}"] = {
                        "n": sm["n"], "pass": sm["pass"], "u_ok": sm["u_ok"],
                        "pass_no_open": mean([t["end_pass"] for t in T if t["split"] != "open"]),
                        "open": f"{sum(t['end_pass'] for t in T if t['split'] == 'open')}/"
                                f"{sum(t['split'] == 'open' for t in T)}", "root": sm["root"]}
                if un != "s5":                                # with the ORIGINAL e2e planner + writer too
                    T = E.run_belt(reqs, [raws[q["id"]]["raw"] for q in reqs], E.Planner("net", f"{E2E}/ckpt"),
                                   E.Writer("net", f"{E2E}/ckpt"))
                    sm = E.summarize(T)
                    rep["regression"][f"{setn}:{un}_s{k}+e2e_ckpt_nets"] = {"n": sm["n"], "pass": sm["pass"],
                                                                              "u_ok": sm["u_ok"]}
    g = f"{udir}/UC_s1/result.json"
    rep["gate_UC1_loss0"] = json.load(open(g)).get("gate_loss0") if os.path.exists(g) else None
    ce = f"{ndir}/ckpt_writer_eval.json"
    rep["ckpt_writer_parity"] = ({k: json.load(open(ce))[k] for k in ("old_pass", "full_pass")}
                                 if os.path.exists(ce) else None)
    rep["predictions"] = verdicts(rep, seeds, useeds)
    out = os.path.join(a.out, "report" + ("_smoke" if a.smoke else "") + ".json")
    json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep["predictions"], indent=1, ensure_ascii=False))
    print("baselines", json.dumps({k: {m: v.get(m) for m in ("n", "pass", "pass_lenient", "exec", "sci_op_acc")}
                                   for k, v in rep["baselines"].items()}))
    for k, v in rep["sci_belt"].items():
        print("belt", k, json.dumps({m: v.get(m) for m in ("n", "pass", "exec", "u_set", "p_ok", "pass_len5_6", "his_pass")}))
    for k, v in rep["regression"].items():
        print("regr", k, json.dumps(v))
    print("wrote", out)


def verdicts(rep, seeds, useeds):
    N, B, R = rep["nets"], rep["sci_belt"], rep["regression"]
    v = {}
    if seeds:
        v["P1_W0_15of15_full"] = all(N[s]["W0"]["old_pass"] == 15 and N[s]["W0"]["full_pass"] for s in seeds)
        v["P2_sci_pass_mean"] = {w: mean([N[s][w]["sci_pass"] for s in seeds]) for w in ("WF64", "WE64", "WF128")}
        v["P2_old_kept"] = {w: [N[s][w]["old_pass"] for s in seeds] for w in ("WF64", "WE64", "WF128")}
        v["P2_full20"] = {w: [N[s][w]["full_pass"] for s in seeds] for w in ("WF64", "WE64", "WF128")}
        v["P3_planner_mean"] = {k: mean([N[s]["planner_plan_exact"][k] for s in seeds])
                                for k in N[seeds[0]]["planner_plan_exact"]}
    v["P4_s5_u_set"] = rep["baselines"].get("s5_oracle_nets", {}).get("u_set")
    v["P6_baselines_pass"] = {k: b.get("pass") for k, b in rep["baselines"].items() if isinstance(b, dict)}
    for un in ("US", "UC"):
        v[f"P4_{un}_u_set_mean"] = mean([B.get(f"{un}_s{k}+oracle_nets", {}).get("u_set") for k in useeds])
        v[f"P4_{un}_his"] = [B.get(f"{un}_s{k}+oracle_nets", {}).get("his_pass") for k in useeds]
        for w in ("WE64", "WF64", "WF128"):
            v[f"P5_{un}_{w}_pass_mean"] = mean([B.get(f"{un}_s{k}+P_s{k - 1}+{w}_s{k - 1}", {}).get("pass") for k in useeds])
            v[f"P5_{un}_{w}_exec_mean"] = mean([B.get(f"{un}_s{k}+P_s{k - 1}+{w}_s{k - 1}", {}).get("exec") for k in useeds])
    v["gate_UC1_loss0"] = rep.get("gate_UC1_loss0")
    v["ckpt_writer_parity"] = rep.get("ckpt_writer_parity")
    v["note_WE64_budget"] = "WE64 = W0 (1500 steps) + 1500 more = 3000 total; WF64/WF128 = 1500: not equal budget"
    for setn in ("A", "B"):
        pc = R.get(f"{setn}:s5+e2e_ckpt_nets", {}).get("pass")
        v[f"P7_{setn}_positive_control"] = {"pass": pc, "expected": S5[setn], "ok": pc == S5[setn]}
        us = mean([R.get(f"{setn}:US_s{k}+e2e_ckpt_nets", {}).get("pass") for k in useeds])
        uc = mean([R.get(f"{setn}:UC_s{k}+e2e_ckpt_nets", {}).get("pass") for k in useeds])
        v[f"P7_{setn}_e2e_nets"] = {"US": us, "UC": uc, "s5_ref": S5[setn],     # understander-only change
                                    "US_ok": None if us is None else us >= S5[setn] - 0.01}
        for w in ("WE64", "WF64", "WF128"):
            s5n = mean([R.get(f"{setn}:s5_s{k}+P_s{k - 1}+{w}", {}).get("pass") for k in useeds])
            us = mean([R.get(f"{setn}:US_s{k}+P_s{k - 1}+{w}", {}).get("pass") for k in useeds])
            uc = mean([R.get(f"{setn}:UC_s{k}+P_s{k - 1}+{w}", {}).get("pass") for k in useeds])
            v[f"P7_{setn}_{w}"] = {"US": us, "UC": uc, "s5_ref": S5[setn], "s5_same_nets": s5n,
                                   "US_ok": None if us is None else us >= S5[setn] - 0.01,
                                   "US_ok_vs_s5_same_nets": None if us is None or s5n is None else us >= s5n - 0.01,
                                   "US_minus_UC": None if us is None or uc is None else round(us - uc, 4)}
    return v

# ----------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("nets", "understander", "gen", "qwen3b", "report"):
        c = sp.add_parser(name)
        c.add_argument("--out", required=True)
        c.add_argument("--threads", type=int, default=2)
        c.add_argument("--device", default="cuda")
        c.add_argument("--mem-frac", type=float, default=0.35)
        c.add_argument("--smoke", action="store_true")
        if name == "nets":
            c.add_argument("--seeds", type=int, nargs="+", default=list(NET_SEEDS))
            c.add_argument("--planner-steps", type=int, default=2000)
            c.add_argument("--writer-steps", type=int, default=1500)
        if name == "understander":
            c.add_argument("--arm", choices=["US", "UC"], required=True)
            c.add_argument("--seed", type=int, required=True)
            c.add_argument("--batch", type=int, default=32)
            c.add_argument("--micro", type=int, default=4)
            c.add_argument("--smoke-steps", type=int, default=2)
        if name == "gen":
            c.add_argument("--lora", default=S5_LORA)
            c.add_argument("--name", default="s5")
            c.add_argument("--sets", nargs="+", default=["A", "SCI"])
        if name == "qwen3b":
            c.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
            c.add_argument("--max-new", type=int, default=900)
    a = ap.parse_args()
    assert a.out.startswith("/media/llm-data/"), "outputs only on the data partition"
    os.makedirs(a.out, exist_ok=True)
    print(json.dumps({"cmd": a.cmd, "md5": {f: hashlib.md5(open(os.path.join(os.path.dirname(__file__), f), "rb")
                                                           .read()).hexdigest()[:10]
                                            for f in ("sci_belt56.py", "multifn_sci.py", "sciops.py")}}), flush=True)
    {"nets": cmd_nets, "understander": cmd_understander, "gen": cmd_gen, "qwen3b": cmd_qwen3b,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
