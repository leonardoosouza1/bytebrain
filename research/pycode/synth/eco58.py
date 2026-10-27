"""Ecosystem loop: every module learns in the same round (ledger #58, TOP PRIORITY; folds in #37/#47/#57 and I7).

HIS WORDS (quoted, followed literally):
  #58 "pq não estava treinando tudo? pq a ideia e que seja tudo treinando, não faz sentido vc treinar separado, o
  ecosistema precisa é fazer tudo junto, aprendo junto, erra e afins. e preciso que de prioridade a ideia de ecosistema"
  #47 "não é construindo um modelo gigante cheio de parametros e sim uma arquitetura inteligente que é como se fosse um
  ecosistema, onde há dependências de toda uma cadeia"
  #37 judges (Qwen first, Sonnet final under a written RULE) say how much was wrong and where; the error retrains OUR
  modules.  #57 Qwen is a JUDGE, never the learner.

THE BELT (one request -> one program; six learning modules, all ours):
  pt-BR request -> UNDERSTANDER (Qwen2.5-Coder-0.5B + LoRA; starts from U0 = s5 adapter + chain warm-up) -> spec JSON
    spec kind "dispatch" (calculator) / "expr" (formula main(a, b))  -> PLANNER_DISPATCH / PLANNER_EXPR (blockaddr nets,
      e2e/ckpt seed 0) -> WRITER_CALC (multifn GRU d64, e2e/ckpt)
    spec kind "chain" (list pipeline, NEW here: {"main": "chain", "steps": [...], "final": f}) -> PLANNER_CHAIN
      (blockaddr addr gb-4, judge57/smoke3 base_s{seed}) -> WRITER_CHAIN (bytewriter d32 ternary, same file)
  -> assembler (rule) -> bwrap. The chain kind is where planner AND writer fail naturally (len5/6 is out of the
  planner's 1-2 unit training range; the ternary writer misses some blocks), so the ecosystem has real errors in
  every stage. The calculator kinds keep the old A/B belt (and its regression) inside the same loop.

ECO ROUND (arm ECO, his idea; K rounds, per seed, every phase gold-free):
  1 belt   current understander -> planner -> writer on this round's pool batch (pool items differ per round).
  2 qwen   Qwen2.5-Coder-3B FIRST judge (verdict: score, stage understander|planner|writer|none, location, fix).
  3 sonnet triage (flags | Qwen invalid | stage != none | score < 100 | 25% hash audit) -> Sonnet FINAL judge (serial
           `claude -p --model sonnet --tools ""`, clean CLAUDE_CONFIG_DIR, empty cwd) under RULE; agree -> Qwen's
           verdict, else Sonnet's (judge57 final_of).
  4 accept fix gate, gold-free: the corrected program runs clean in bwrap (no flags) AND Sonnet re-passes it against
           the REQUEST (pass true, score >= 90).
  5 train  ALL SIX modules update in this same round:
           - the blamed module gets the accepted fix (understander: text -> corrected spec; planner: name of the spec
             it RECEIVED -> rule plan; writer: block key it RECEIVED -> corrected text);
           - every module trains on its own outputs of fully successful episodes (no flags, final verdict none);
           - downstream inputs are the REAL upstream outputs of this round (planner rows are keyed by the spec the
             understander really wrote; writer rows by the blocks the planner really planned);
           - each module mixes its original data as replay (75% of every batch) and has a Guard57-style train-only
             guard (snapshot only while every original train item it had right is still right).
  Rows accumulate over rounds (deduplicated). Round r+1 runs the belt with the modules round r produced.
ARMS (paired seeds 0/5/6 = chain base nets of judge57; same rounds K, same steps per module per round, same batch,
  same mix, same lr schedule (K blocks with restart), same RNG draw order; rows per module per block equal to ECO's):
  ECO  the loop above.
  SEP  "the current way": same judges, the SAME accepted fix rows ECO collected (favours SEP), but each module trains
       alone on generator-gold inputs only: ECO's success rows on real upstream outputs are replaced by the same number
       of generator-gold rows (gold spec -> gold plan, gold blocks) from a separate generator seed; no belt in between.
  N    original data only, same steps.     B  U0 + base nets, zero steps (positive control for the belt).
SETS (pool vs held-out disjoint by generator seed, by name/spec AND by template: frames and connectors of held-out
  requests are never used in U0/pool/SEP; the operation lexicon is shared, as in e2e where HOLD_D/HOLD_X are held):
  pool        K rounds x (32 chain len3-6 + 16 calc: dispatch len4-6, expr calls4-5), train templates
  held-out    chain_long (len5 60 + len6 60), chain_short (len2-4 60), chainB (60 Sonnet rewrites of further held
              chain requests, never in the loop), calc_long (dispatch len5-7 30 + expr calls5-6 30, HOLD frames).
  regression  A = e2e/requests.json (179), B = e2e/para/requests.json (174).
  PRIMARY = pooled new held-out (chain_long + chain_short + chainB + calc_long), end-to-end pass@1 (bwrap grade).
KILL / PASS: ECO > SEP AND ECO > N on PRIMARY with paired-bootstrap CI low > 0 and the same sign in 3/3 seeds.
GOLD: only in build (set construction), the separate gold_*.json files, the eval grade and the report. The judge path
  (belt view files, Qwen, Sonnet, fix gate, row building) never reads them. grep: `gold` appears in cmd_eval/cmd_report
  and grading helpers only.

PREDICTIONS (written 2026-10-27 15:07, before any run; pass@1, mean of 3 seeds, batched greedy):
  P0 positive controls: cached s5 raws through this file's belt -> A 175/179 (0.9777), B 149/174 (0.8563) exactly;
     s5 adapter regenerated batch-1 on GPU reproduces >= 350/353 cached raws and the same A/B pass counts +-1;
     chain base s0 module test_a = 0.8703 on CPU, within +-0.03 on GPU.
  P1 arm B: chain_long 0.15-0.35, chain_short 0.45-0.75, chainB 0.15-0.45, calc_long 0.70-0.95; A >= 0.94, B >= 0.82.
     chain_long failures split understander 25-50% / planner 25-50% / writer 15-35%.
  P2 ECO - SEP on PRIMARY: +2..+8 pp (success rows on the understander's REAL specs and the planner's REAL plans move
     the planner/writer to the inputs they actually get; later rounds catch errors that only exist after the first
     update). KILL as stated above.
  P3 ECO - N on PRIMARY: +5..+15 pp.   P4 SEP - N: +1..+8 pp.
  P5 no regression: every arm within -2 pp of B on A and on B (guards keep the understander's train items).
  P6 ECO in-loop pool pass rises round over round (r0 -> rK-1: +10..+25 pp); planner share of failures falls most.
  P7 judges vs gold (report only): Sonnet-final stage accuracy on failing items >= 0.85; Qwen-3B <= 0.50; label
     precision of accepted rows >= 0.90.

AMENDMENT (2026-10-27 15:39, after the independent Opus review, before any GPU run; original predictions kept above):
  P2 mechanism caveat: SEP receives every ECO fix row AND gold specs on new texts of the same distribution; ECO's
     only exclusive rows are self-labels of successful episodes (near-zero gradient) plus fixes aimed at its own
     current errors. So P2 (+2..+8 pp) is NOT derivable from the mechanism; a tie ECO = SEP is a live outcome.
     Revised P2 band: -3..+5 pp. P3/P4 unchanged.
  For understander/planner fixes the bwrap gate checks only the rule-built reference program (always clean), so
     their acceptance rests on Sonnet's re-pass of its own fix; label precision vs gold is reported (P7).
  NO VERDICT if guards froze the modules (report: blocks_with_movement < 2 for any ECO/SEP seed); the GPU script
     aborts before the rounds if U0 kept no step or its 32-item chain train probe is < 0.70.
  --max-sonnet-calls is ONE total for build + all rounds (J.Sonnet counts the shared sonnet_calls.jsonl); expected
     ~8 + 4 x 25-40 calls. Ids dropped by a batch answer get one retry before exit 5.
  Arm N trains with mix 0: every batch is 100% original data (ECO/SEP: 75% original + 25% rows). Dispatch held-out
     is disjoint by ORDERED op list (the same op set in another order may occur in the pool); templates disjoint.
  Chain belt output depends slightly on batch composition (reviewer: 45/48 identical alone vs batched); all arms see
     the same eval list in the same order, so arms are comparable; in-loop vs eval numbers are not item-comparable.
DECLARED: U0 (s5 + 150 chain warm-up steps, len2-4 (every len1 name is in synth2/bridge splits), train templates) is shared by all seeds; the calc planners/writer
  are e2e/ckpt seed 0 for all seeds (they are ~100% on calc sets, so most calc movement is the understander); seed
  varies the chain nets, every RNG of the continuation, and therefore everything the loop does after round 0.
  An accepted planner fix states that the received spec was right; its row is the rule plan of that spec (like
  judge57 S3: once the error is caught, a planner fix is close to an oracle row). Batched greedy is used for every arm
  (P0 reports its agreement with batch-1).

Phases (each a separate process; GPU phases never hold two models; caches/resume by file):
  build  (CPU)  sets.json (+ chainB rewrites via Sonnet)          p0  (GPU/CPU) positive controls
  u0     (GPU)  U0 adapter                                          belt/qwen/sonnet/train --round r   (ECO round r)
  arms   (GPU)  SEP and N, K blocks each                            eval (GPU) all arms x seeds     report (CPU)
Usage: eco58.py <phase> --out DIR [--device cuda|cpu --smoke --seeds 0,5,6 --rounds 4 --round r ...]
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import textwrap
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockaddr as BA  # noqa: E402

CHAIN_BA = (BA.SYM, BA.START, BA.MAX_OUT)               # blockaddr's own world = the chain world (bridge_cpu SYM)
import bytewriter as BW  # noqa: E402
import compose as CP  # noqa: E402
import e2e_pipeline as E  # noqa: E402  (sets MF.PLAN_FORMAT = interleaved, as the e2e ckpt)
import judge_retrain57 as J  # noqa: E402
import multifn as MF  # noqa: E402
import student_understander as SU  # noqa: E402
import synth2 as S2  # noqa: E402
from bridge_cpu import Data as ChainData, blocks_of, parse  # noqa: E402
from e2e_pipeline import last_json  # noqa: E402

ROOT = "/media/llm-data/universe-runs/bytebrain-pycode-20261022"
OUT = f"{ROOT}/eco58"
CACHE = "/media/llm-data/models/hf/hub"
CKPT = f"{ROOT}/e2e/ckpt"
S5_LORA = f"{ROOT}/e2e/student_s5_s1/lora"
S5_RAWS = {"A": f"{ROOT}/e2e/student_s5_s1/raws.json", "B": f"{ROOT}/e2e/para/student_s5_s1/raws.json"}
REQS = {"A": f"{ROOT}/e2e/requests.json", "B": f"{ROOT}/e2e/para/requests.json"}
S5_KNOWN = {"A": (175, 179), "B": (149, 174)}
CHAIN_BASE = ROOT + "/judge57/smoke3/base_s{}.pt"
META = E.load_meta(CKPT)
SEEN = J.SEEN
MODS = ["understander", "planner_dispatch", "planner_expr", "planner_chain", "writer_calc", "writer_chain"]
ARMS = ["ECO", "SEP", "N"]
NEW_SETS = ["chain_long", "chain_short", "chainB", "calc_long"]
N_SHOW = 8
POOL_SEED, HELD_SEED, U0_SEED, SEP_SEED, B_SEED = 5801, 5850, 5860, 5870, 5880

# ------------------------------------------------------------------ pt-BR verbalizer for the chain kind

UNIT_PT = {"sort": ["ordena a lista", "coloca os números em ordem crescente"],
           "reverse": ["inverte a ordem da lista", "deixa a lista de trás para frente"],
           "negate": ["troca o sinal de cada número", "nega cada elemento"],
           "abs": ["pega o valor absoluto de cada número", "deixa cada elemento positivo com o valor absoluto"],
           "unique": ["remove os valores repetidos mantendo a primeira ocorrência", "tira os duplicados da lista"],
           "drop_first": ["descarta o primeiro elemento", "tira o primeiro item da lista"],
           "cumsum": ["faz a soma acumulada", "troca cada posição pelo total acumulado até ali"],
           "running_max": ["faz o máximo acumulado", "troca cada posição pelo maior valor visto até ali"],
           "dedup_adjacent": ["remove repetidos vizinhos", "junta valores iguais consecutivos em um só"],
           "add_k": ["soma k a cada número", "acrescenta k em cada elemento"],
           "keep_gt_k": ["mantém só os maiores que k", "filtra os valores acima de k"],
           "take_k": ["fica só com os k primeiros", "pega os primeiros k elementos"]}
PARAM_PT = {"add": ["soma {n} a cada número", "acrescenta {n} em cada elemento"],
            "times": ["multiplica cada número por {n}", "faz cada elemento vezes {n}"],
            "keep_gt": ["mantém só os maiores que {n}", "filtra os valores acima de {n}"],
            "keep_lt": ["mantém só os menores que {n}", "filtra os valores abaixo de {n}"],
            "clip": ["limita cada número a no máximo {n}", "corta os valores acima de {n} para {n}"],
            "mod": ["troca cada número pelo resto da divisão por {n}", "aplica o resto da divisão por {n} em cada elemento"],
            "take": ["fica só com os {n} primeiros", "pega os primeiros {n} elementos"]}
FINAL_PT = {"": ["e devolve a lista", "e retorna a lista resultante"],
            "sum": ["e retorna a soma", "e devolve o total"],
            "max": ["e retorna o maior valor (0 se a lista ficar vazia)", "e devolve o máximo, ou 0 se não sobrar nada"],
            "len": ["e retorna quantos elementos sobraram", "e devolve o tamanho da lista"],
            "first": ["e retorna o primeiro elemento (0 se vazia)", "e devolve o primeiro item, ou 0 se vazia"],
            "count_gt": ["e retorna quantos são maiores que {n}", "e devolve a contagem de valores acima de {n}"]}
FRAMES_TRAIN = ["crie uma função que recebe a lista xs{K}, {S}", "escreva uma função em python que pega a lista xs{K}, {S}"]
FRAMES_HELD = ["preciso de um código que, dada a lista xs{K}, {S}"]
CONN_TRAIN = [", depois ", " e então "]
CONN_HELD = [", em seguida "]


def unit_phrase(u, r):
    if u in UNIT_PT:
        return r.choice(UNIT_PT[u])
    op, n = u.rsplit("_", 1)
    return r.choice(PARAM_PT[op]).format(n=n)


def final_phrase(f, r):
    if f.startswith("count_gt_"):
        return r.choice(FINAL_PT["count_gt"]).format(n=f.rsplit("_", 1)[1])
    return r.choice(FINAL_PT[f])


def chain_text(task, r, held):
    units, final = task
    conn = r.choice(CONN_HELD if held else CONN_TRAIN)
    kk = " e o número k" if S2.nargs(units) == 2 else ""
    body = conn.join(unit_phrase(u, r) for u in units) + " " + final_phrase(final, r)
    return r.choice(FRAMES_HELD if held else FRAMES_TRAIN).format(K=kk, S=body)


def chain_spec(task):
    return {"main": "chain", "steps": list(task[0]), "final": task[1]}


def chain_req(task, r, held, split, iid):
    return {"id": iid, "world": "chain", "split": split, "text": chain_text(task, r, held), "task": [list(task[0]), task[1]],
            "gold": chain_spec(task)}


def gen_chain(r, lens, exclude):
    """one task per entry of lens, names unique and not in exclude (exclude is updated)."""
    out = []
    for k in lens:
        for _ in range(100000):
            t = [[r.choice(SEEN) for _ in range(k)], r.choice(J.FINALS)]
            n = S2.name_of(*t)
            if n not in exclude:
                exclude.add(n)
                out.append(t)
                break
        else:
            raise RuntimeError(f"gen_chain: no unused name of length {k}")
    return out


def gkey(gold):
    return json.dumps(gold, sort_keys=True)


def calc_req(kind, spec, r, held, split, iid):
    if kind == "dispatch":
        fr = E.FRAMES_D[3] if held else r.choice([f for f in E.FRAMES_D if f != SU.HOLD_D])
        return {"id": iid, "world": "dispatch", "split": split, "caps": list(spec), "gold": {"main": "dispatch",
                "calls": list(spec)}, "text": fr.format(L=E.say_list([r.choice(E.PT[o]) for o in spec]))}
    fr = E.FRAMES_X[1] if held else r.choice([f for f in E.FRAMES_X if f != SU.HOLD_X])
    return {"id": iid, "world": "expr", "split": split, "caps": ["formula"], "prefix": list(spec),
            "gold": {"main": "expr", "expr": E.expr_of(spec)}, "text": fr.format(P=E.verbalize(spec, r))}


def gen_calc(r, kinds, exclude, held, split, tag):
    """kinds [("dispatch", k) | ("expr", calls)] -> requests with unique golds not in exclude (updated)."""
    out = []
    for kind, k in kinds:
        while True:
            spec = tuple(r.sample(MF.OPS, k)) if kind == "dispatch" else tuple(CP.rand_tree(r, k))
            q = calc_req(kind, spec, r, held, split, f"{tag}{len(out):04d}")
            if gkey(q["gold"]) not in exclude:
                exclude.add(gkey(q["gold"]))
                out.append(q)
                break
    return out


def pool_kinds(n_chain, n_calc, r):
    lens = [3, 4, 5, 5, 6, 6] * (n_chain // 6 + 1)
    r.shuffle(lens)
    ck = [("dispatch", r.choice([4, 5, 6])) if i % 2 == 0 else ("expr", r.choice([4, 5])) for i in range(n_calc)]
    return lens[:n_chain], ck

# ------------------------------------------------------------------ worlds / modules


def set_world(w):
    if w == "chain":
        BA.SYM, BA.START, BA.MAX_OUT = CHAIN_BA
    else:
        E.use_world(w, META)


CALC = {"dispatch": (MF.spec_name, MF.plan_of, MF.SYM, MF.name_bytes),
        "expr": (CP.spec_name, CP.plan_of, CP.SYM, CP.name_bytes)}


def load_nets(seed, path, dev):
    nets = {}
    ck = torch.load(CHAIN_BASE.format(seed), map_location=dev)
    set_world("chain")
    nets["planner_chain"] = BA.Net("addr", gate_bias=-4.0)
    nets["planner_chain"].load_state_dict(ck["planner"])
    nets["writer_chain"] = BW.Writer(J.WRITER_D, J.WRITER_TER)
    nets["writer_chain"].load_state_dict(ck["writer"])
    for w in ("dispatch", "expr"):
        set_world(w)
        nets[f"planner_{w}"] = BA.Net("addr", gate_bias=-4.0)
        nets[f"planner_{w}"].load_state_dict(torch.load(os.path.join(CKPT, f"planner_{w}.pt"), map_location=dev))
    nets["writer_calc"] = MF.Writer(META["writer_dim"])
    nets["writer_calc"].load_state_dict(torch.load(os.path.join(CKPT, "writer.pt"), map_location=dev))
    if path and os.path.exists(path):
        st = torch.load(path, map_location=dev)
        for k, v in st.items():
            nets[k].load_state_dict(v)
    for n in nets.values():
        n.train()
    return nets


class CalcPlanner(E.Planner):
    def __init__(self, nets):                            # noqa: super not called on purpose (no ckpt reload)
        self.kind, self.meta = "net", META
        self.nets = {"dispatch": nets["planner_dispatch"], "expr": nets["planner_expr"]}


class CalcWriter(E.Writer):
    def __init__(self, nets):                            # noqa
        self.kind, self.net = "net", nets["writer_calc"]


def eval_mode(nets, on):
    for n in nets.values():
        n.eval() if on else n.train()

# ------------------------------------------------------------------ understander (HF)


def load_understander(lora, dev, trainable=False, merge=False):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SU.MODEL, cache_dir=CACHE, local_files_only=True)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(SU.MODEL, cache_dir=CACHE, local_files_only=True,
                                                 dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    model = PeftModel.from_pretrained(model, lora, is_trainable=trainable)
    if merge:
        model = model.merge_and_unload()
    model.eval()
    return model, tok


@torch.no_grad()
def gen_specs(model, tok, texts, dev, bs=32, max_new=160):
    """greedy; bs=1 = SU.evaluate exactly (no padding); bs>1 = left-padded batches (declared, same for every arm)."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    was = model.training
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        ps = [SU.prompt(t) for t in texts[i:i + bs]]
        enc = tok(ps, add_special_tokens=False, return_tensors="pt", padding=True)
        ids, am = enc.input_ids.to(dev), enc.attention_mask.to(dev)
        g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False,
                           eos_token_id=im_end, pad_token_id=tok.pad_token_id, repetition_penalty=1.0,
                           temperature=None, top_p=None, top_k=None)
        out += [tok.decode(row[ids.shape[1]:], skip_special_tokens=True).strip() for row in g]
    if was:
        model.train()
    return out

# ------------------------------------------------------------------ the belt (gold-free production)


def parse_raw(raw):
    """understander raw text -> (kind, spec, bridged, flags). chain bridged = (units list, final)."""
    try:
        s = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        s = None
    if isinstance(s, dict) and s.get("main") == "chain":
        st, fi = s.get("steps"), s.get("final", "")
        if not isinstance(st, list) or not all(isinstance(u, str) for u in st) or not isinstance(fi, str) or \
                set(s) - {"main", "steps", "final"}:
            return None, s, None, ["understander:schema_error"]
        bad = [u for u in st if u not in S2.UNITS] + ([fi] if fi not in S2.FINAL else [])
        if bad:
            return None, s, None, [f"understander:oov_unit:{b}" for b in bad]
        if not 1 <= len(st) <= 6:
            return None, s, None, ["understander:bad_length"]
        return "chain", s, (list(st), fi), []
    s2, b, fl = E.check_spec(raw)
    return (b[0] if b else None), s2, b, [f"{f['to']}:{f['kind']}" + (f":{f['fn']}" if f["fn"] else "") for f in fl]


def canon(kind, b):
    return chain_spec(b) if kind == "chain" else E.spec_of(b)


def rule_plan(kind, b):
    return blocks_of(*b) if kind == "chain" else E.oracle_plan(b)


def spec_name(kind, b):
    return S2.name_of(*b) if kind == "chain" else CALC[kind][0](b[1])


def reference_code(kind, b):
    return S2.source(*b) if kind == "chain" else E.assemble(kind, E.oracle_plan(b), MF.block_text)


def calc_blocks(plan):
    return [s for s in dict.fromkeys(plan) if s in MF.SYM[3:]]


def calc_keys(kind, b, src):
    ks = [f"d:{o}" for o in b[1]] if kind == "dispatch" else ["x:"]
    disp = re.findall(r'op == "(\w+)"', src)
    kj = [f"jd:{o}" for o in dict.fromkeys(list(b[1]) + disp)] if kind == "dispatch" else ["jx:"]
    return ks, kj


def execute(entries):
    """[(kind, bridged, src)] -> [(facts [[call, out]], exec flags)]; chain via synth2 bwrap, calc via e2e bwrap."""
    res = [None] * len(entries)
    ch = [i for i, e in enumerate(entries) if e[0] == "chain"]
    if ch:
        outs = J.run_srcs([(entries[i][2], entries[i][1]) for i in ch])
        for i, o in zip(ch, outs):
            task, src = entries[i][1], entries[i][2]
            nm = S2.name_of(*task)
            facts = [[f"{nm}({xs}, {k})" if S2.nargs(task[0]) == 2 else f"{nm}({xs})",
                      "<module failed to load>" if o is None else ("EXCEPTION" if o[j] == "!" else o[j])]
                     for j, (xs, k) in enumerate(S2.INPUTS[:N_SHOW])]
            res[i] = (facts, [f"exec:{f}" for f in J.flags_of(src, o, task)])
    B, hs = E.Batch(), {}
    for i, (kind, b, src) in enumerate(entries):
        if kind in ("dispatch", "expr"):
            ks, kj = calc_keys(kind, b, src)
            hs[i] = (B.add(src, ks + kj), ks, kj)
    if hs:
        B.run()
    for i, (h, ks, kj) in hs.items():
        out, fl, facts = B[h], [], []
        kind, b, src = entries[i]
        defined, _ = E.called_names(src)
        if defined is None:
            fl.append("exec:module_parse_error")
        elif "main" not in defined:
            fl.append("exec:missing_main")
        for k in ks:
            if out[k] is None:
                fl.append(f"exec:module_error:{k}")
                break
            if any(v.startswith("!") for v in out[k]):
                fl.append(f"exec:exec_error:{k}")
        for k in kj:
            _, _, o = k.partition(":")
            calls = E.calls_for(k)[:4 if kind == "dispatch" else N_SHOW]
            for j, (fn, args) in enumerate(calls):
                v = out[k][j] if out[k] is not None else "<module failed to load>"
                facts.append([f"{fn}({', '.join(repr(x) for x in args)})", v])
        res[i] = (facts[:16], fl)
    return res


def calc_validate(kind, b, plan, text_of, src):
    """the e2e gold-free validators (run_belt), as flag strings."""
    fl = []
    if kind == "dispatch":
        fl += [f"planner:missing_disp:{o}" for o in b[1] if f"disp:{o}" not in plan]
        fl += [f"planner:extra_disp:{s[5:]}" for s in plan if s.startswith("disp:") and s[5:] not in b[1]]
    elif CP.render_expr(plan) != E.expr_of(b[1]):
        fl.append("planner:tree_mismatch")
    if "<eos>" not in plan:
        fl.append("planner:no_eos")
    defined, called = E.called_names(src)
    if defined is not None:
        fl += [f"planner:undefined_helper:{y}" for y in sorted(called - defined - E.BUILTIN_NAMES)]
    for s in calc_blocks(plan):
        if s.startswith("def:"):
            ok, why = E.def_name_ok(text_of(s), s[4:])
            if not ok:
                fl.append(f"writer:{why.split(':')[0]}:{s[4:]}")
    return fl


def belt(nets, reqs, raws):
    """requests + understander raws -> gold-free records (what the judges may see) . No gold is read here."""
    recs = []
    for q, raw in zip(reqs, raws):
        kind, spec, b, fl = parse_raw(raw)
        recs.append({"id": q["id"], "text": q["text"], "raw": raw, "kind": kind, "bridged": b, "flags": fl,
                     "plan": [], "blocks": [], "src": "", "facts": []})
    eval_mode(nets, True)
    ch = [r for r in recs if r["kind"] == "chain"]
    if ch:
        set_world("chain")
        items = J.run_belt(nets["planner_chain"], nets["writer_chain"], [r["bridged"] for r in ch])
        for r, it in zip(ch, items):
            r["plan"], r["plan_parsed"] = it["syms"], it["plan"]
            r["blocks"] = [{"label": f"{'final' if bb[1] else 'step'} {bb[0] or '(return xs)'}", "key": list(bb),
                            "text": t} for bb, t in zip(it["blocks"], it["texts"])]
            r["src"] = it["src"]
            if it["plan"] is None:
                r["flags"].append("planner:malformed_plan")
            elif [list(it["plan"][0]), it["plan"][1]] != [list(r["bridged"][0]), r["bridged"][1]]:
                r["flags"].append("planner:plan_differs_from_spec")
    cc = [r for r in recs if r["kind"] in ("dispatch", "expr")]
    if cc:
        P, W = CalcPlanner(nets), CalcWriter(nets)
        plans = P.plan([r["bridged"] for r in cc])
        NT = W.texts({s for p in plans for s in p})
        for r, p in zip(cc, plans):
            tx = (lambda s: NT.get(s, ""))
            r["plan"], r["src"] = p, E.assemble(r["kind"], p, tx)
            r["blocks"] = [{"label": s, "key": s, "text": NT.get(s, "")} for s in calc_blocks(p)]
            r["flags"] += calc_validate(r["kind"], r["bridged"], p, tx, r["src"])
    eval_mode(nets, False)
    ex = [r for r in recs if r["kind"]]
    for r, (facts, fl) in zip(ex, execute([(r["kind"], r["bridged"], r["src"]) for r in ex])):
        r["facts"], r["flags"] = facts, r["flags"] + fl
    return recs

# ------------------------------------------------------------------ gold grading (eval + report only)


def grade(nets, reqs, raws, recs):
    """-> per request {pass, root}. chain: synth2 strict grade of the produced program against the gold task;
    calc: e2e run_belt end_pass/root (the same nets, the known e2e grader)."""
    out = [None] * len(reqs)
    ch = [i for i, q in enumerate(reqs) if q["world"] == "chain"]
    ok_c = [i for i in ch if recs[i]["kind"] == "chain"]
    sc = S2.grade([reqs[i]["task"] for i in ok_c], [[recs[i]["src"]] for i in ok_c]) if ok_c else []
    sc = dict(zip(ok_c, sc))
    for i in ch:
        q, r = reqs[i], recs[i]
        p = i in sc and sc[i][0] == 1.0
        if p:
            root = None
        elif r["kind"] != "chain" or [list(r["bridged"][0]), r["bridged"][1]] != q["task"]:
            root = "understander"
        elif r.get("plan_parsed") is None or [list(r["plan_parsed"][0]), r["plan_parsed"][1]] != q["task"]:
            root = "planner"
        else:
            root = "writer"
        out[i] = {"pass": int(p), "root": root}
    cc = [i for i, q in enumerate(reqs) if q["world"] != "chain"]
    if cc:
        eval_mode(nets, True)
        T = E.run_belt([reqs[i] for i in cc], [raws[i] for i in cc], CalcPlanner(nets), CalcWriter(nets))
        eval_mode(nets, False)
        for i, t in zip(cc, T):
            out[i] = {"pass": int(t["end_pass"]), "root": t["root"]}
    return out

# ------------------------------------------------------------------ judges: RULE, evidence, verdicts, fixes

RULE = """RULE (the judges follow this exactly).
A request written in Portuguese asks for a Python program. A small generator answers it in three stages:
UNDERSTANDER reads the request and writes a JSON SPEC of one of three kinds:
 1. list pipeline {"main": "chain", "steps": [unit ids in order], "final": final id}. The program is ONE function
    named by the steps (and the final, if any) joined with `_then_`, applied in order to the list `xs`; it takes a
    second argument `k` when a step id ends in `_k`. unit ids: sort reverse negate abs unique drop_first cumsum
    running_max dedup_adjacent add_k keep_gt_k take_k, and OP_N with N in 2..6: add_N (+N to each), times_N (*N),
    keep_gt_N (keep elements > N), keep_lt_N (keep < N), clip_N (min(x, N)), mod_N (x % N), take_N (first N).
    final ids: "" (return the list), sum, max (0 if empty), len, first (0 if empty), count_gt_N (how many > N).
 2. calculator {"main": "dispatch", "calls": [op ids in the order asked]}: one helper function per op and
    main(op, a, b=0) that dispatches `if op == "<id>": return <id>(a, b)`.
 3. formula {"main": "expr", "expr": "<nested calls of a and b>"}: main(a, b=0) returns that formula.
    op ids: add sub mul div mod (a % b) maxx (larger of a, b) sqrt neg abs square pow fact hyp avg pct (a percent of b);
    formulas use only add sub mul div mod maxx hyp avg pct, two arguments each.
PLANNER reads the spec and writes the PLAN (the code blocks to write). WRITER writes the code of every planned block.
Blame the FIRST stage that broke its own contract for the input it RECEIVED:
- "understander": the SPEC does not ask for what the request asks (wrong kind, missing/extra/wrong/swapped step or op,
  wrong number, wrong final, invalid JSON).
- "planner": the spec is right but the PLAN does not match the spec (missing, extra, swapped or malformed blocks).
- "writer": spec and plan are right but the code of a block does not do what its label says.
- "none": the program does what the request asks. Do not guess outputs you cannot derive.
location: planner -> index (0-based) of the first wrong plan entry; writer -> index of the first wrong BLOCK;
understander/none -> null.
score = 0-100, how much of what the request asks the program really does.
fix (null when stage is none):
- understander: {"spec": <the correct spec JSON>}
- planner: {"spec": <the spec the planner received, copied exactly>} (the plan is then rebuilt by rule from it)
- writer: {"blocks": [{"location": i, "text": "<corrected code of block i>"}, ...]} one entry per wrong block, written
  like the other blocks (list steps: no indentation, reassign xs; helper functions: `def` line at column 0 with the
  body indented 4 spaces; dispatch lines: `if op == "<id>":` at column 0 with `return <id>(a, b)` indented 4)."""

Q_ROLE = ('\n\nYou are the FIRST judge. Answer with ONE JSON object only: {"score": int, "stage": "none"|'
          '"understander"|"planner"|"writer", "location": int or null, "justification": str, "fix": object or null}.')
S_ROLE = J.S_ROLE
REPASS = ("You re-check corrected programs against the same RULE. For each item, pass = true only if the program does "
          "everything the REQUEST asks.")
STAGES = ("none", "understander", "planner", "writer")


def view(rec):
    """the only fields a judge path may read."""
    return {k: rec[k] for k in ("id", "text", "raw", "kind", "bridged", "flags", "plan", "plan_parsed", "blocks", "src",
                                "facts") if k in rec}


def evidence(rec, iid):
    L = [f"ID: {iid}", f"REQUEST: {rec['text']}", "SPEC (understander output): " + (rec["raw"][:600] or "(empty)")]
    if not rec["kind"]:
        L.append("PLAN: (none: the spec is invalid, nothing was planned or written)")
    else:
        if rec["kind"] == "chain":
            pp = rec.get("plan_parsed")
            L.append("PLAN (planner output): " + (f"steps {json.dumps(pp[0])}, final {json.dumps(pp[1])}" if pp else
                                                   "MALFORMED " + " ".join(rec["plan"])))
        else:
            L.append("PLAN (planner output): " + " ".join(rec["plan"]))
        L.append("BLOCKS (writer output, numbered; shown without the function-level indentation):")
        for k, b in enumerate(rec["blocks"]):
            L.append(f"  [{k}] {b['label']}:")
            L += [f"      {l}" for l in textwrap.dedent(b["text"]).split("\n") if l.strip()] or ["      (empty)"]
        L += ["FULL CODE:", rec["src"].rstrip(), "EXECUTION (bwrap; outputs only, no expected values):"]
        L += [f"  {c} -> {o}" for c, o in rec["facts"]] or ["  (not executed)"]
    L.append("MACHINE FLAGS: " + (", ".join(rec["flags"]) if rec["flags"] else "none"))
    return "\n".join(L)


def norm_verdict(v):
    if not isinstance(v, dict) or v.get("stage") not in STAGES:
        return None
    try:
        sc = int(v.get("score"))
    except (TypeError, ValueError):
        return None
    loc = v.get("location")
    if v["stage"] in ("planner", "writer"):
        try:
            loc = int(loc)
        except (TypeError, ValueError):
            return None
    else:
        loc = None
    fx = v.get("fix")
    return {"score": max(0, min(100, sc)), "stage": v["stage"], "location": loc,
            "justification": str(v.get("justification", ""))[:600], "fix": fx if isinstance(fx, dict) else None}


def final_of(qv, sv_raw):
    """judge57 final_of with the three-stage verdict: Sonnet agrees -> Qwen's verdict, else Sonnet's."""
    sv = norm_verdict(sv_raw) if isinstance(sv_raw, dict) else None
    agree = bool(isinstance(sv_raw, dict) and sv_raw.get("agree_with_qwen") is True)
    if sv_raw is None:
        return None, "sonnet_missing", None
    if sv is None and not (agree and qv is not None):
        return None, "sonnet_invalid", False
    if qv is None:
        return sv, "qwen_invalid" + ("_sonnet_said_agree" if agree else ""), False
    if agree:
        if sv is not None and (sv["stage"] != qv["stage"] or (qv["stage"] in ("planner", "writer") and
                                                              sv["location"] != qv["location"])):
            return sv, "agree_inconsistent", False
        return qv, "agree" if sv is not None else "agree_unverified", True
    return sv, "disagree", False


def needs_sonnet(rec, qv):
    if rec["flags"] or qv is None or qv["stage"] != "none" or qv["score"] < 100:
        return "triage"
    return "audit" if int(hashlib.md5(rec["id"].encode()).hexdigest(), 16) % 4 == 0 else ""


def fix_text(kind, label_key, text):
    """writer fix text -> the writer's own format, or None when the writer could not emit it."""
    s = J.norm_text(text)
    if not s:
        return None
    if kind == "chain":
        return s if all(c in BW.C2I for c in s) and len(s) <= 159 else None
    if str(label_key).startswith("disp:"):
        s = "".join("    " + l + "\n" for l in s.split("\n") if l)
    return s if all(c in MF.CH for c in s) and len(s) <= 159 else None


def build_fix(rec, v):
    """verdict -> {code, kind, bridged, rows} or {reject}. Rows go to the blamed module. Gold-free."""
    if v is None or v["stage"] == "none":
        return {"reject": "no_error_claimed"}
    fx = v["fix"]
    if fx is None:
        return {"reject": "no_fix"}
    if v["stage"] in ("understander", "planner"):
        sp = fx.get("spec")
        if not isinstance(sp, dict):
            return {"reject": "no_spec"}
        kind, _, b, fl = parse_raw(json.dumps(sp))
        if b is None or fl:
            return {"reject": "bad_spec"}
        c = canon(kind, b)
        same = rec["kind"] == kind and canon(rec["kind"], rec["bridged"]) == c
        if v["stage"] == "understander":
            if same:
                return {"reject": "fix_equals_output"}
            return {"code": reference_code(kind, b), "kind": kind, "bridged": b, "plan": rule_plan(kind, b),
                    "rows": [{"module": "understander", "text": rec["text"], "spec": c}]}
        if not same:                                     # a planner fix restates the spec the planner received
            return {"reject": "fix_disagrees_with_spec"}
        rp = rule_plan(kind, b)
        if rp == rec["plan"]:
            return {"reject": "fix_equals_output"}
        return {"code": reference_code(kind, b), "kind": kind, "bridged": b, "plan": rp,
                "rows": [{"module": f"planner_{kind}", "name": spec_name(kind, b), "syms": rp}]}
    # writer
    if not rec["kind"] or not rec["blocks"] or (rec["kind"] == "chain" and rec.get("plan_parsed") is None):
        return {"reject": "nothing_written"}
    bl = fx.get("blocks")
    if not isinstance(bl, list) or not bl:
        return {"reject": "bad_blocks"}
    texts, rows = [b["text"] for b in rec["blocks"]], []
    for e in bl[:6]:
        if not isinstance(e, dict):
            return {"reject": "bad_blocks"}
        try:
            k = int(e.get("location"))
        except (TypeError, ValueError):
            return {"reject": "bad_location"}
        if not 0 <= k < len(texts):
            return {"reject": "location_out_of_range"}
        key = rec["blocks"][k]["key"]
        t = fix_text(rec["kind"], key, e.get("text", ""))
        if t is None:
            return {"reject": "unencodable_text"}
        if t == texts[k]:
            return {"reject": "fix_equals_output"}
        texts[k] = t
        if rec["kind"] == "chain":
            op, dg = BW.block_key(*key)
            rows.append({"module": "writer_chain", "op": op, "dig": dg, "text": t, "block": key})
        else:
            rows.append({"module": "writer_calc", "sym": key, "text": t})
    if rec["kind"] == "chain":
        code = J.assemble(rec["bridged"], texts)
    else:
        tm = {b["key"]: t for b, t in zip(rec["blocks"], texts)}
        code = E.assemble(rec["kind"], rec["plan"], lambda s: tm.get(s, ""))
    return {"code": code, "kind": rec["kind"], "bridged": rec["bridged"], "plan": rec["plan"], "rows": rows,
            "texts": texts}


def gate(cands):
    """[(rec, fix)] -> [(facts, flags)]: the corrected program must run clean in bwrap (gold-free)."""
    res = execute([(f["kind"], f["bridged"], f["code"]) for _, f in cands])
    out = []
    for (rec, f), (facts, fl) in zip(cands, res):
        if f["kind"] != "chain":
            tm = dict(zip([b["key"] for b in rec["blocks"]], f.get("texts", [])))
            fl = fl + calc_validate(f["kind"], f["bridged"], f["plan"], lambda s: tm.get(s, MF.block_text(s)), f["code"])
        out.append((facts, fl))
    return out


def repass_evidence(rec, code, facts, iid):
    return "\n".join([f"ID: {iid}", f"REQUEST: {rec['text']}", "CORRECTED PROGRAM:", code.rstrip(),
                      "EXECUTION (bwrap; outputs only):"] + [f"  {c} -> {o}" for c, o in facts])


class QwenJudge:
    """Qwen2.5-Coder-3B (FIRST judge), HF, greedy, left-padded batches. Never trained (#57)."""

    def __init__(self, model, device, mem_frac):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(mem_frac)
        self.tok = AutoTokenizer.from_pretrained(model, cache_dir=CACHE, local_files_only=True)
        self.tok.padding_side = "left"
        kw = {"device_map": {"": 0}} if device == "cuda" else {}
        self.m = AutoModelForCausalLM.from_pretrained(model, cache_dir=CACHE, local_files_only=True,
                                                      dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                                      **kw).eval()
        self.dev = device

    @torch.no_grad()
    def ask(self, system, users, max_new):
        ps = [self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                           add_generation_prompt=True, tokenize=False) for u in users]
        enc = self.tok(ps, return_tensors="pt", padding=True, add_special_tokens=False)
        ids, am = enc.input_ids.to(self.dev), enc.attention_mask.to(self.dev)
        g = self.m.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new, do_sample=False,
                            pad_token_id=self.tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        return [self.tok.decode(r[ids.shape[1]:], skip_special_tokens=True) for r in g]

# ------------------------------------------------------------------ training (every module, guarded)


def gen_targets(syms, symlist):
    ys = [[symlist.index(s) for s in v] for v in syms]
    T = max(map(len, ys))
    y = torch.tensor([v + [0] * (T - len(v)) for v in ys])
    return torch.cat([torch.full((len(ys), 1), len(symlist)), y[:, :-1]], 1), y


def trim(row, symlist):
    s = [symlist[k] for k in row]
    return s[:s.index("<eos>") + 1] if "<eos>" in s else s


def calc_plan_ok(net, w, specs):
    set_world(w)
    _, plan_fn, sym, nb = CALC[w]
    net.eval()
    ok = set()
    for i in range(0, len(specs), 256):
        ch = specs[i:i + 256]
        for j, (row, sp) in enumerate(zip(net.greedy(nb(ch)).tolist(), ch)):
            if trim(row, sym) == plan_fn(sp):
                ok.add(i + j)
    net.train()
    return frozenset(ok)


def cont_planner_calc(net, w, train, guard_specs, rows, seed, steps, mix, lr):
    set_world(w)
    name_fn, plan_fn, sym, _ = CALC[w]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    ro, rx, nx = random.Random(seed * 1000 + 41), random.Random(seed * 1000 + 42), round(48 * mix)
    g = J.Guard57(net, lambda: calc_plan_ok(net, w, guard_specs))
    set_world(w)
    loss = torch.tensor(0.0)
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = J.lr_at(lr, s, steps)
        bo = [train[ro.randrange(len(train))] for _ in range(48)]
        ix = [rx.random() for _ in range(nx)]
        names, syms = [name_fn(t) for t in bo], [plan_fn(t) for t in bo]
        if rows:
            for j, u in enumerate(ix):
                r = rows[int(u * len(rows))]
                names[48 - nx + j], syms[48 - nx + j] = r["name"], r["syms"]
        y_in, y = gen_targets(syms, sym)
        loss = F.cross_entropy(net(J.name_tensor(names), y_in).transpose(1, 2), y, ignore_index=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        g.check(s + 1)
        set_world(w)
    return g.finish(round(loss.item(), 5))


def calc_block_ok(net, pool):
    net.eval()
    tx = net.write(torch.tensor([MF.SYM.index(b) for b in pool]))
    net.train()
    return frozenset(i for i, (t, b) in enumerate(zip(tx, pool)) if t == MF.block_text(b))


def cont_writer_calc(net, pool, rows, seed, steps, mix, lr):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    ro, rx, nx = random.Random(seed * 1000 + 51), random.Random(seed * 1000 + 52), round(64 * mix)
    g = J.Guard57(net, lambda: calc_block_ok(net, pool))
    loss = torch.tensor(0.0)
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = J.lr_at(lr, s, steps)
        bo = [pool[ro.randrange(len(pool))] for _ in range(64)]
        ix = [rx.random() for _ in range(nx)]
        keys, txt = list(bo), [MF.block_text(b) for b in bo]
        if rows:
            for j, u in enumerate(ix):
                r = rows[int(u * len(rows))]
                keys[64 - nx + j], txt[64 - nx + j] = r["sym"], r["text"]
        ys = [[MF.CH.index(c) for c in t] + [MF.END] for t in txt]
        T = max(map(len, ys))
        y = torch.tensor([v + [-100] * (T - len(v)) for v in ys])
        y_in = torch.cat([torch.full((64, 1), len(MF.CH) + 1), y[:, :-1].clamp(min=0)], 1)
        loss = F.cross_entropy(net(torch.tensor([MF.SYM.index(k) for k in keys]), y_in).transpose(1, 2), y,
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        g.check(s + 1)
    return g.finish(round(loss.item(), 5))


def gold_kb(q):
    """report only: (kind, bridged) of a request's gold."""
    if q["world"] == "chain":
        return "chain", (list(q["task"][0]), q["task"][1])
    b = E.gold_bridged(q)                                # ("dispatch", ops) -- the same shape parse_raw returns
    return b[0], b


def spec_equal(raw, gold):
    kind, _, b, fl = parse_raw(raw)
    return b is not None and not fl and canon(kind, b) == canon_gold(gold)


def canon_gold(gold):
    if gold["main"] == "chain":
        return chain_spec((gold["steps"], gold["final"]))
    if gold["main"] == "dispatch":
        return gold
    return E.canon_spec(gold)


class LoraGuard:
    """Guard57 for the understander: every `every` steps snapshot the LoRA weights if every guard item (its own
    training pairs) the round-start model got exactly right is still right; restore the last such snapshot."""

    def __init__(self, model, score, every):
        self.model, self.score, self.every = model, score, every
        self.base = score()
        self.best, self.step, self.last = self.snap(), 0, self.base

    def snap(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items() if "lora_" in k}

    def check(self, step):
        if step % self.every == 0:
            v = self.score()
            if v >= self.base:
                self.best, self.step, self.last = self.snap(), step, v

    def finish(self, loss):
        raw = self.score()
        self.model.load_state_dict(self.best, strict=False)
        return {"loss": loss, "train_ok_base": len(self.base), "train_ok_raw": len(raw),
                "raw_lost": len(self.base - raw), "train_ok_kept": len(self.last), "kept_step": self.step}


def train_understander(model, tok, rows, replay, guard_items, seed, steps, mix, lr, batch, micro, dev, guard_every):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ro, rx, nx = random.Random(seed * 1000 + 31), random.Random(seed * 1000 + 32), round(batch * mix)

    def score():
        raws = gen_specs(model, tok, [t for t, _ in guard_items], dev, bs=32, max_new=160)
        return frozenset(i for i, (r, (_, g)) in enumerate(zip(raws, guard_items)) if spec_equal(r, g))
    g = LoraGuard(model, score, guard_every)
    model.train()
    loss = torch.tensor(0.0)
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = J.lr_at(lr, s, steps)
        bo = [replay[ro.randrange(len(replay))] for _ in range(batch)]
        ix = [rx.random() for _ in range(nx)]
        if rows:
            for j, u in enumerate(ix):
                r = rows[int(u * len(rows))]
                bo[batch - nx + j] = (r["text"], r["spec"])
        enc = [SU.encode(tok, t, sp) for t, sp in bo]
        opt.zero_grad()
        tot = 0.0
        for j in range(0, batch, micro):
            ids, lab, am = (x.to(dev) for x in SU.collate(enc[j:j + micro], tok.pad_token_id))
            part = SU.loss_of(model, ids, lab, am) * (len(enc[j:j + micro]) / batch)
            part.backward()
            tot += part.detach()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        loss = tot
        g.check(s + 1)
    return g.finish(round(float(loss), 5))

# ------------------------------------------------------------------ state / files


def jl(path):
    """jsonl; a truncated last line (kill mid-write) is skipped so the phase resumes."""
    out = []
    if os.path.exists(path):
        for l in open(path):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def state_dir(a, arm, s, r):
    return f"{a.out}/u0" if r == 0 else f"{a.out}/state/{arm}_s{s}_r{r}"


def state_paths(a, arm, s, r):
    d = state_dir(a, arm, s, r)
    return f"{d}/lora", (None if r == 0 else f"{d}/nets.pt")


def setup(a):
    torch.set_num_threads(a.threads)
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
        torch.set_default_device("cuda")


def steps_of(a):
    if a.smoke:                                          # 25 = J.GUARD_EVERY: the snapshot path runs in the smoke
        return {**{m: 25 for m in MODS}, "understander": 4}
    return {"understander": a.u_steps, "planner_chain": a.net_steps, "writer_chain": a.net_steps,
            "planner_dispatch": a.net_steps // 2, "planner_expr": a.net_steps // 2, "writer_calc": a.net_steps // 2}


class Orig:
    """each module's original training data (replay) and its guard items (train items only)."""

    def __init__(self, a, sets):
        self.chain = ChainData()
        self.chain_train = self.chain.sets["train"]
        self.chain_pool_w = sorted({b for t in self.chain_train for b in BW.blocks_of_task(t)})
        md, cd = MF.Data(), CP.Data()
        self.disp_train, self.expr_train = md.sets["train"], cd.sets["train"]
        g = random.Random(17)
        self.disp_guard = g.sample(self.disp_train, 24 if a.smoke else 300)
        self.expr_guard = g.sample(self.expr_train, 24 if a.smoke else 300)
        self.calc_pool_w = sorted({s for sp in md.sets["train"] for s in MF.plan_of(sp) if s != "<eos>"})
        self.und_replay = [tuple(x) for x in sets["und_replay"]]
        gi = random.Random(19).sample(range(len(self.und_replay)), 8 if a.smoke else 48)
        self.und_guard = [self.und_replay[i] for i in gi]


def train_block(a, arm, s, r, rows, orig, seed_key):
    """one block for every module: state (arm, s, r) -> (arm, s, r+1). Same function for ECO, SEP and N."""
    d_out = state_dir(a, arm, s, r + 1)
    if os.path.exists(f"{d_out}/log.json"):
        print(f"{arm} s{s} block {r} cached", flush=True)
        return json.load(open(f"{d_out}/log.json"))
    lora_in, nets_in = state_paths(a, arm, s, r)
    os.makedirs(d_out, exist_ok=True)
    st, t0 = steps_of(a), time.time()
    nets = load_nets(s, nets_in, a.device)
    by = {m: [x for x in rows if x["module"] == m] for m in MODS}
    log = {"arm": arm, "seed": s, "block": r, "n_rows": {m: len(v) for m, v in by.items()}, "steps": st}
    mx = (lambda m: a.mix if by[m] else 0.0)
    set_world("chain")
    log["planner_chain"] = J.cont_planner(nets["planner_chain"], orig.chain_train, by["planner_chain"], seed_key,
                                          st["planner_chain"], mx("planner_chain"), a.lr)
    log["writer_chain"] = J.cont_writer(nets["writer_chain"], orig.chain_pool_w, by["writer_chain"], seed_key,
                                        st["writer_chain"], mx("writer_chain"), a.lr)
    for w, tr, gd in (("dispatch", orig.disp_train, orig.disp_guard), ("expr", orig.expr_train, orig.expr_guard)):
        log[f"planner_{w}"] = cont_planner_calc(nets[f"planner_{w}"], w, tr, gd, by[f"planner_{w}"], seed_key,
                                                st[f"planner_{w}"], mx(f"planner_{w}"), a.lr)
    log["writer_calc"] = cont_writer_calc(nets["writer_calc"], orig.calc_pool_w, by["writer_calc"], seed_key,
                                          st["writer_calc"], mx("writer_calc"), a.lr)
    torch.save({k: v.state_dict() for k, v in nets.items()}, f"{d_out}/nets.pt")
    model, tok = load_understander(lora_in, a.device, trainable=True)
    log["understander"] = train_understander(model, tok, by["understander"], orig.und_replay, orig.und_guard, seed_key,
                                             st["understander"], mx("understander"), a.u_lr, a.u_batch, a.u_micro,
                                             a.device, 2 if a.smoke else 10)
    model.save_pretrained(f"{d_out}/lora")
    log["train_s"] = round(time.time() - t0, 1)
    if a.device == "cuda":
        log["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    del model
    jdump(log, f"{d_out}/log.json")
    if r >= 1:                                           # disk: the previous block's adapter is no longer needed
        prev = state_dir(a, arm, s, r)
        if os.path.isdir(f"{prev}/lora"):
            shutil.rmtree(f"{prev}/lora")
    print(json.dumps({k: v for k, v in log.items()}), flush=True)
    return log

# ------------------------------------------------------------------ phases


def load_sets(a):
    return json.load(open(f"{a.out}/sets.json"))


def cmd_build(a):
    """sets (gold lives here; the judge path reads only belt view files) + chainB rewrites by Sonnet."""
    f = f"{a.out}/sets.json"
    if os.path.exists(f) and "chainB" in json.load(open(f)):
        print("sets cached", flush=True)
        return
    A = json.load(open(REQS["A"]))
    B = json.load(open(REQS["B"]))
    s5_pairs, _ = SU.train_pairs("s5", 1, A)
    known_names = set()
    for v in json.load(open(f"{S2.D}/splits.json")).values():
        known_names |= {S2.name_of(*t) for t in v}
    for v in ChainData().sets.values():
        known_names |= {S2.name_of(*t) for t in v}
    calc_ex = {gkey(q["gold"]) for q in A + B} | {gkey(g) for _, g in s5_pairs}
    sm = a.smoke
    rh = random.Random(HELD_SEED)
    names = set(known_names)
    held = {}
    held["chain_long"] = [chain_req(t, rh, True, "chain_long", f"hl{i:04d}") for i, t in
                          enumerate(gen_chain(rh, [5] * (2 if sm else 60) + [6] * (2 if sm else 60), names))]
    held["chain_short"] = [chain_req(t, rh, True, "chain_short", f"hs{i:04d}") for i, t in
                           enumerate(gen_chain(rh, [2, 3, 4] * (1 if sm else 20), names))]
    braw = [chain_req(t, rh, True, "chainB", f"hb{i:04d}") for i, t in
            enumerate(gen_chain(rh, [3, 4, 5, 6] * (1 if sm else 18), names))]
    kinds = [("dispatch", rh.choice([5, 6, 7])) for _ in range(2 if sm else 30)] + \
            [("expr", rh.choice([5, 6])) for _ in range(2 if sm else 30)]
    held["calc_long"] = gen_calc(rh, kinds, calc_ex, True, "calc_long", "hc")
    # train side: U0 chain warm-up, pool rounds, SEP gold fill -- disjoint from held by name/gold and template
    ru = random.Random(U0_SEED)
    u0 = [chain_req(t, ru, False, "u0", f"u{i:05d}") for i, t in
          enumerate(gen_chain(ru, [ru.choice([2, 2, 3, 3, 4, 4]) for _ in range(40 if sm else 1500)], names))]
    pool = []
    for r in range(a.rounds):
        rp = random.Random(POOL_SEED + r)
        nch, nca = (4, 2) if sm else (a.pool_chain, a.pool_calc)
        lens, ck = pool_kinds(nch, nca, rp)
        items = [chain_req(t, rp, False, "pool", f"p{r}c{i:03d}") for i, t in enumerate(gen_chain(rp, lens, names))]
        items += gen_calc(rp, ck, calc_ex, False, "pool", f"p{r}x")
        rp.shuffle(items)
        pool.append(items)
    rs = random.Random(SEP_SEED)
    nch, nca = (8, 4) if sm else (a.rounds * a.pool_chain, a.rounds * a.pool_calc)
    lens, ck = pool_kinds(nch, nca, rs)
    sep = [chain_req(t, rs, False, "sep", f"g{i:04d}") for i, t in enumerate(gen_chain(rs, lens, names))]
    sep += gen_calc(rs, ck, calc_ex, False, "sep", "gx")
    rs.shuffle(sep)
    und_replay = [(t, g) for t, g in s5_pairs] + [(q["text"], q["gold"]) for q in u0]
    # disjointness checks (provenance): names / golds / templates
    held_all = [q for v in held.values() for q in v] + braw
    tr_all = u0 + [q for p in pool for q in p] + sep
    hn = {gkey(q["gold"]) for q in held_all}
    assert not hn & {gkey(q["gold"]) for q in tr_all}, "held-out gold leaked into train side"
    assert not hn & {gkey(g) for _, g in s5_pairs}, "held-out gold in s5 train pairs"
    assert not hn & {gkey(q["gold"]) for q in A + B}, "held-out gold in A/B"
    for q in tr_all:
        assert not q["text"].startswith(FRAMES_HELD[0].split("{")[0]) and CONN_HELD[0] not in q["text"]
        assert not q["text"].startswith(E.FRAMES_D[3].split("{")[0]) and not q["text"].startswith(E.FRAMES_X[1].split("{")[0])
    sets = {"held": held, "pool": pool, "sep": sep, "u0": u0, "und_replay": und_replay, "chainB_src": braw,
            "sizes": {**{k: len(v) for k, v in held.items()}, "pool_per_round": len(pool[0]), "sep": len(sep),
                      "u0": len(u0), "und_replay": len(und_replay)}}
    # chainB: Sonnet rewrites (B-style, like e2e/para); kept only if every number of the request survives
    S = J.Sonnet(a.out, a.max_sonnet_calls)
    out = []
    bs = 3 if sm else 10
    for i in range(0, len(braw), bs):
        ch = braw[i:i + bs]
        prompt = ("Rewrite each Portuguese request below in natural Brazilian Portuguese with clearly different wording "
                  "(as a real user would ask). Keep every operation, its order, every number, the final result and the "
                  "names xs and k. Do not add or remove steps. Do not write code. Answer with ONE JSON object only: "
                  '{"verdicts": [{"id": str, "text": str}, ...]}.\n\n' +
                  "\n".join(f'{q["id"]}: {q["text"]}' for q in ch))
        vs = S.ask(prompt, f"chainB_{i}") or []
        byid = {str(v.get("id")): v.get("text") for v in vs if isinstance(v, dict)}
        for q in ch:
            t = byid.get(q["id"])
            nums = re.findall(r"\d", q["text"])
            if isinstance(t, str) and t.strip() and t != q["text"] and sorted(re.findall(r"\d", t)) == sorted(nums):
                out.append({**q, "orig_text": q["text"], "text": t.strip()})
        if sm:
            break
    sets["held"]["chainB"] = out
    sets["sizes"]["chainB"] = len(out)
    sets["sizes"]["chainB_rejected"] = len(braw) - len(out) if not sm else bs - len(out)
    jdump(sets, f)
    print(json.dumps(sets["sizes"]), flush=True)


def cmd_p0(a):
    """positive controls. CPU part: cached s5 raws -> this file's grader (e2e run_belt with these nets) must give A
    175/179 and B 149/174. Model part (--device): s5 adapter regenerated batch-1 (exact-string agreement with the cache,
    pass counts) and batched; chain base module test_a per seed (s0 0.8703 known on CPU)."""
    setup(a)
    res = {}
    nets = load_nets(0, None, a.device)
    for k in ("A", "B"):
        reqs = json.load(open(REQS[k]))
        cache = json.load(open(S5_RAWS[k]))
        raws = [cache[q["id"]]["raw"] for q in reqs]
        g = grade(nets, reqs, raws, belt(nets, reqs, raws))
        res[f"cached_{k}"] = [sum(x["pass"] for x in g), len(g)]
        print(f"POSITIVE CONTROL P0a {k}: cached s5 raws -> {res[f'cached_{k}']} (known {S5_KNOWN[k]})", flush=True)
    data = ChainData()
    for s in a.seeds:
        n = load_nets(s, None, a.device)
        set_world("chain")
        ev = J.eval_sets(n["planner_chain"], n["writer_chain"], {"test_a": data.sets["test_a"]})
        res[f"chain_test_a_s{s}"] = round(sum(ev["test_a"]) / len(ev["test_a"]), 4)
        print(f"POSITIVE CONTROL P0b chain base s{s} test_a {res[f'chain_test_a_s{s}']} (device {a.device}; "
              f"judge57 CPU s0 0.8703 s5 0.8688 s6 0.9156)", flush=True)
    if not a.no_model:
        model, tok = load_understander(S5_LORA, a.device, merge=True)
        for k in ("A", "B"):
            reqs = json.load(open(REQS[k]))
            if a.smoke:
                reqs = reqs[:4] + reqs[-4:]
            cache = json.load(open(S5_RAWS[k]))
            for tag, bs in (("b1", 1), ("batched", 32)):
                t0 = time.time()
                raws = gen_specs(model, tok, [q["text"] for q in reqs], a.device, bs=bs, max_new=SU.MAX_NEW["expr"])
                g = grade(nets, reqs, raws, belt(nets, reqs, raws))
                same = sum(r == cache[q["id"]]["raw"] for r, q in zip(raws, reqs))
                res[f"s5_{tag}_{k}"] = {"pass": [sum(x["pass"] for x in g), len(g)], "raw_equal_cache": [same, len(reqs)],
                                        "s": round(time.time() - t0, 1)}
                print(f"POSITIVE CONTROL P0c s5 {tag} {k}: pass {res[f's5_{tag}_{k}']['pass']} raw==cache "
                      f"{same}/{len(reqs)} (known {S5_KNOWN[k]})", flush=True)
        if a.device == "cuda":
            res["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    jdump(res, f"{a.out}/p0{'_nomodel' if a.no_model else ''}.json")


def cmd_u0(a):
    """U0 = s5 adapter + chain warm-up (len2-4, train templates) mixed 50/50 with s5 replay; shared by all seeds."""
    setup(a)
    if os.path.exists(f"{a.out}/u0/log.json"):
        print("u0 cached", flush=True)
        return
    sets = load_sets(a)
    os.makedirs(f"{a.out}/u0", exist_ok=True)
    s5 = [tuple(x) for x in sets["und_replay"][:len(sets["und_replay"]) - len(sets["u0"])]]
    rows = [{"text": q["text"], "spec": q["gold"]} for q in sets["u0"]]
    guard = random.Random(19).sample(s5, 8 if a.smoke else 48)
    model, tok = load_understander(S5_LORA, a.device, trainable=True)
    t0 = time.time()
    log = train_understander(model, tok, rows, s5, guard, 58, 3 if a.smoke else a.u0_steps, 0.5, 1e-4, a.u_batch,
                             a.u_micro, a.device, 2 if a.smoke else 25)
    model.save_pretrained(f"{a.out}/u0/lora")
    log["train_s"] = round(time.time() - t0, 1)
    probe = sets["u0"][:32]
    raws = gen_specs(model, tok, [q["text"] for q in probe], a.device)
    log["u0_train_probe_exact"] = [sum(spec_equal(r, q["gold"]) for r, q in zip(raws, probe)), len(probe)]
    if a.device == "cuda":
        log["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    jdump(log, f"{a.out}/u0/log.json")
    print(json.dumps(log), flush=True)


def cmd_belt(a):
    """ECO round r: current modules on this round's pool batch -> belt view (gold-free) + gold file (report only)."""
    setup(a)
    sets = load_sets(a)
    reqs = sets["pool"][a.round]
    for s in a.seeds:
        fv = f"{a.out}/belt_s{s}_r{a.round}.json"
        if os.path.exists(fv):
            print(f"belt s{s} r{a.round} cached", flush=True)
            continue
        lora, nets_p = state_paths(a, "ECO", s, a.round)
        nets = load_nets(s, nets_p, a.device)
        model, tok = load_understander(lora, a.device, merge=True)
        raws = gen_specs(model, tok, [q["text"] for q in reqs], a.device)
        del model
        recs = belt(nets, reqs, raws)
        for rc in recs:
            rc["id"] = f"s{s}_r{a.round}_{rc['id']}"
        g = grade(nets, reqs, raws, recs)                # REPORT ONLY: written to a separate file
        jdump({rc["id"]: x for rc, x in zip(recs, g)}, f"{a.out}/gold_s{s}_r{a.round}.json")
        jdump([view(rc) for rc in recs], fv)
        roots = [x["root"] for x in g if not x["pass"]]
        print(json.dumps({"belt": s, "round": a.round, "pool_pass": round(sum(x["pass"] for x in g) / len(g), 4),
                          "roots": {k: roots.count(k) for k in set(roots)},
                          "flagged": sum(bool(rc["flags"]) for rc in recs)}), flush=True)


def cmd_qwen(a):
    q = QwenJudge(a.qwen_model, a.device, a.mem_frac)
    sysq = RULE + Q_ROLE
    for s in a.seeds:
        path = f"{a.out}/qwen_s{s}_r{a.round}.jsonl"
        done = {r["id"] for r in jl(path)}
        recs = [r for r in json.load(open(f"{a.out}/belt_s{s}_r{a.round}.json")) if r["id"] not in done]
        if os.path.exists(path) and os.path.getsize(path) and open(path, "rb").read()[-1:] != b"\n":
            open(path, "a").write("\n")                  # end a line truncated by a kill before appending
        f = open(path, "a")
        for i in range(0, len(recs), a.qwen_batch):
            ch = recs[i:i + a.qwen_batch]
            t0 = time.time()
            outs = q.ask(sysq, [evidence(r, r["id"]) for r in ch], a.qwen_max_new)
            for r, raw in zip(ch, outs):
                v = norm_verdict(last_json(raw))
                f.write(json.dumps({"id": r["id"], "qwen": v, "qwen_valid": v is not None, "qwen_raw": raw[-1500:]}) + "\n")
            f.flush()
            print(json.dumps({"seed": s, "round": a.round, "batch": i, "valid": [last_json(o) is not None for o in outs],
                              "s": round(time.time() - t0, 1)}), flush=True)


def cmd_sonnet(a):
    S = J.Sonnet(a.out, a.max_sonnet_calls)
    for s in a.seeds:
        path = f"{a.out}/final_s{s}_r{a.round}.jsonl"
        if os.path.exists(path):
            print(f"final s{s} r{a.round} cached", flush=True)
            continue
        recs = {r["id"]: r for r in json.load(open(f"{a.out}/belt_s{s}_r{a.round}.json"))}
        qrows = {r["id"]: r for r in jl(f"{a.out}/qwen_s{s}_r{a.round}.jsonl")}
        miss = [i for i in recs if i not in qrows]
        assert not miss, f"qwen phase incomplete for seed {s} round {a.round}: {len(miss)}"
        rows, send = {}, []
        for iid, rc in recs.items():
            why = needs_sonnet(rc, qrows[iid]["qwen"])
            rows[iid] = {"id": iid, "why": why, "qwen_valid": qrows[iid]["qwen_valid"]}
            if why:
                send.append(iid)
            else:
                rows[iid].update(final=qrows[iid]["qwen"], note="qwen_pass_not_sent", qwen_ok=True)
        for b in range(0, len(send), a.batch):
            ids = send[b:b + a.batch]
            body = []
            for iid in ids:
                qv = qrows[iid]["qwen"]
                first = json.dumps(qv) if qv is not None else (
                    "(INVALID: the first judge produced no valid JSON; agree_with_qwen must be false and you give the "
                    "full verdict)")
                body.append(evidence(recs[iid], iid) + "\nFIRST JUDGE (Qwen) VERDICT: " + first)
            vs = S.ask(RULE + S_ROLE + "\n\n" + "\n\n=====\n\n".join(body), f"judge_s{s}_r{a.round}_{b}") or []
            byid = {str(v.get("id")): v for v in vs if isinstance(v, dict)}
            lost = [k for k, iid in enumerate(ids) if iid not in byid]
            if lost:                                     # one retry for ids the batch answer dropped
                vs = S.ask(RULE + S_ROLE + "\n\n" + "\n\n=====\n\n".join(body[k] for k in lost),
                           f"judge_s{s}_r{a.round}_{b}_retry") or []
                byid.update({str(v.get("id")): v for v in vs if isinstance(v, dict)})
            for iid in ids:
                fin, note, qok = final_of(qrows[iid]["qwen"], byid.get(iid))
                rows[iid].update(sonnet_raw=byid.get(iid), final=fin, note=note, qwen_ok=qok)
            print(json.dumps({"seed": s, "round": a.round, "batch": b, "calls": S.used,
                              "notes": [rows[i]["note"] for i in ids]}), flush=True)
        cand = []
        for iid, r in rows.items():
            r["fix"] = build_fix(recs[iid], r.get("final"))
            if "code" in r["fix"]:
                cand.append(iid)
        gates = gate([(recs[i], rows[i]["fix"]) for i in cand])
        rp = []
        for iid, (facts, fl) in zip(cand, gates):
            rows[iid]["fix_flags"] = fl
            if not fl:
                rp.append((iid, facts))
        for b in range(0, len(rp), a.batch):
            ch = rp[b:b + a.batch]
            head = (RULE + "\n\n" + REPASS + ' Answer with ONE JSON object only: {"verdicts": [{"id": str, "pass": '
                    'bool, "score": int}, ...]}.\n\n')
            prompt = head + "\n\n=====\n\n".join(repass_evidence(recs[i], rows[i]["fix"]["code"], fc, i) for i, fc in ch)
            vs = S.ask(prompt, f"repass_s{s}_r{a.round}_{b}") or []
            byid = {str(v.get("id")): v for v in vs if isinstance(v, dict)}
            lost = [(i, fc) for i, fc in ch if i not in byid]
            if lost:
                vs = S.ask(head + "\n\n=====\n\n".join(repass_evidence(recs[i], rows[i]["fix"]["code"], fc, i)
                                                   for i, fc in lost), f"repass_s{s}_r{a.round}_{b}_retry") or []
                byid.update({str(v.get("id")): v for v in vs if isinstance(v, dict)})
            for iid, _ in ch:
                rows[iid]["repass"] = byid.get(iid)
        for r in rows.values():
            r["accepted"] = bool("code" in r["fix"] and not r.get("fix_flags") and J.repass_ok(r.get("repass")))
        miss = sum(r.get("note") == "sonnet_missing" or (r.get("fix_flags") == [] and r.get("repass") is None)
                   for r in rows.values())
        with open(path + (".incomplete" if miss else ""), "w") as f:
            for r in rows.values():
                f.write(json.dumps(r) + "\n")
        print(json.dumps({"seed": s, "round": a.round, "sent": len(send), "fix_built": len(cand),
                          "accepted": sum(r["accepted"] for r in rows.values()), "calls_used": S.used}), flush=True)
        if miss:                                          # a capped/failed Sonnet must not look like a result
            print(f"ABORT: seed {s} round {a.round}: {miss} items without a Sonnet verdict/re-pass (cap {S.cap}, "
                  f"used {S.used})", flush=True)
            sys.exit(5)


def success_rows(rc):
    """fully successful episode (no flags, final verdict none) -> one row per module that took part, keyed by the
    REAL upstream output (the spec the understander wrote, the blocks the planner planned)."""
    k, b = rc["kind"], rc["bridged"]
    rows = [{"module": "understander", "text": rc["text"], "spec": canon(k, b)},
            {"module": f"planner_{k}", "name": spec_name(k, b), "syms": rc["plan"]}]
    for bl in rc["blocks"]:
        if k == "chain":
            op, dg = BW.block_key(*bl["key"])
            rows.append({"module": "writer_chain", "op": op, "dig": dg, "text": bl["text"], "block": bl["key"]})
        else:
            rows.append({"module": "writer_calc", "sym": bl["key"], "text": bl["text"]})
    return rows


def rkey(x):
    return json.dumps({k: v for k, v in x.items() if k not in ("src", "round", "kind_row")}, sort_keys=True)


def eco_rows(a, s, r):
    """cumulative ECO buffer after round r: accepted fixes + success rows of rounds 0..r (dedup, first-seen order)."""
    out, seen = [], set()
    for rr in range(r + 1):
        recs = {x["id"]: x for x in json.load(open(f"{a.out}/belt_s{s}_r{rr}.json"))}
        fin = {x["id"]: x for x in jl(f"{a.out}/final_s{s}_r{rr}.jsonl")}
        assert set(recs) <= set(fin), f"final file incomplete s{s} r{rr}"
        for iid, rc in recs.items():
            fr = fin[iid]
            new = []
            if fr["accepted"]:
                new = [{**x, "src": "fix", "round": rr} for x in fr["fix"]["rows"]]
            elif rc["kind"] and not rc["flags"] and (fr.get("final") or {}).get("stage") == "none":
                new = [{**x, "src": "success", "round": rr} for x in success_rows(rc)]
            for x in new:
                kk = rkey(x)
                if kk not in seen:
                    seen.add(kk)
                    out.append(x)
    return out


def sep_gold_rows(sets, seed):
    """generator-gold rows per module (gold spec -> gold plan -> gold blocks) from the SEP generator seed."""
    by = {m: [] for m in MODS}
    for q in sets["sep"]:
        if q["world"] == "chain":
            t = q["task"]
            by["understander"].append({"module": "understander", "text": q["text"], "spec": chain_spec(t)})
            by["planner_chain"].append({"module": "planner_chain", "name": S2.name_of(*t), "syms": blocks_of(*t)})
            for bb in BW.blocks_of_task(t):
                op, dg = BW.block_key(*bb)
                by["writer_chain"].append({"module": "writer_chain", "op": op, "dig": dg, "text": BW.block_text(*bb),
                                           "block": list(bb)})
        else:
            b = E.gold_bridged(q)
            by["understander"].append({"module": "understander", "text": q["text"], "spec": canon(b[0], b)})
            by[f"planner_{b[0]}"].append({"module": f"planner_{b[0]}", "name": spec_name(b[0], b),
                                          "syms": rule_plan(b[0], b)})
            for sy in calc_blocks(rule_plan(b[0], b)):
                by["writer_calc"].append({"module": "writer_calc", "sym": sy, "text": MF.block_text(sy)})
    g = random.Random(seed * 7 + 58)
    for m in MODS:
        u, seen = [], set()
        for x in by[m]:
            if rkey(x) not in seen:
                seen.add(rkey(x))
                u.append(x)
        g.shuffle(u)
        by[m] = u
    return by


def sep_rows(eco_buf, gold_by):
    """SEP block rows: ECO's accepted FIX rows of that block + generator-gold rows, per module, to ECO's count."""
    out = []
    for m in MODS:
        eco_m = [x for x in eco_buf if x["module"] == m]
        fx = [x for x in eco_m if x["src"] == "fix"]
        need = len(eco_m) - len(fx)
        gold = gold_by[m][:need]
        if len(gold) < need:
            print(f"WARNING SEP fill short for {m}: {len(gold)}/{need} (gold rows reused)", flush=True)
            gold = (gold_by[m] * (need // max(1, len(gold_by[m])) + 1))[:need]
        out += fx + [{**x, "src": "gold"} for x in gold]
    return out


def cmd_train(a):
    """ECO round r: rows of rounds 0..r (fixes + successes on REAL upstream outputs) -> ALL modules, one block."""
    setup(a)
    sets = load_sets(a)
    orig = Orig(a, sets)
    for s in a.seeds:
        rows = eco_rows(a, s, a.round)
        jdump(rows, f"{a.out}/rows_ECO_s{s}_r{a.round}.json")
        print(json.dumps({"seed": s, "round": a.round, "rows": {m: [sum(x["module"] == m and x["src"] == k for x in rows)
                                                                    for k in ("fix", "success")] for m in MODS}}),
              flush=True)
        train_block(a, "ECO", s, a.round, rows, orig, s * 10 + a.round)


def cmd_arms(a):
    """SEP and N: K blocks each, same steps/mix/lr/seed keys as ECO's rounds; SEP rows sized to ECO's per block."""
    setup(a)
    sets = load_sets(a)
    orig = Orig(a, sets)
    for s in a.seeds:
        gold_by = sep_gold_rows(sets, s)
        for r in range(a.rounds):
            buf = json.load(open(f"{a.out}/rows_ECO_s{s}_r{r}.json"))
            sr = sep_rows(buf, gold_by)
            jdump(sr, f"{a.out}/rows_SEP_s{s}_r{r}.json")
            train_block(a, "SEP", s, r, sr, orig, s * 10 + r)
            train_block(a, "N", s, r, [], orig, s * 10 + r)


def cmd_eval(a):
    """every arm x seed on held-out + A/B; gold grade; per-stage roots; chain module test_a."""
    setup(a)
    sets = load_sets(a)
    evs = {k: v for k, v in sets["held"].items()}
    for k in ("A", "B"):
        q = json.load(open(REQS[k]))
        evs[k] = q[:4] + q[-4:] if a.smoke else q
    data = ChainData()
    arms = [x for x in a.arms.split(",")]
    for arm in arms:
        for s in a.seeds:
            f = f"{a.out}/eval_{arm}_s{s}.json"
            if os.path.exists(f):
                print(f"eval {arm} s{s} cached", flush=True)
                continue
            r = 0 if arm == "B" else a.rounds
            lora, nets_p = state_paths(a, arm, s, r)
            nets = load_nets(s, nets_p, a.device)
            model, tok = load_understander(lora, a.device, merge=True)
            t0 = time.time()
            names = list(evs)
            allq = [q for k in names for q in evs[k]]
            raws = gen_specs(model, tok, [q["text"] for q in allq], a.device)
            del model
            recs = belt(nets, allq, raws)
            g = grade(nets, allq, raws, recs)
            res = {"arm": arm, "seed": s, "eval": {}, "roots": {}}
            i = 0
            for k in names:
                n = len(evs[k])
                res["eval"][k] = [x["pass"] for x in g[i:i + n]]
                rt = [x["root"] for x in g[i:i + n] if not x["pass"]]
                res["roots"][k] = {z: rt.count(z) for z in sorted(set(map(str, rt)))}
                i += n
            res["raws"] = {q["id"] + "@" + q["split"]: rw for q, rw in zip(allq, raws)}
            set_world("chain")
            ev = J.eval_sets(nets["planner_chain"], nets["writer_chain"], {"test_a": data.sets["test_a"]})
            res["chain_module_test_a"] = round(sum(ev["test_a"]) / len(ev["test_a"]), 4)
            res["eval_s"] = round(time.time() - t0, 1)
            res["eval_mean"] = {k: round(sum(v) / max(1, len(v)), 4) for k, v in res["eval"].items()}
            if a.device == "cuda":
                res["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            jdump(res, f)
            print(json.dumps({"arm": arm, "seed": s, "eval_mean": res["eval_mean"], "roots": res["roots"],
                              "chain_module_test_a": res["chain_module_test_a"], "s": res["eval_s"]}), flush=True)
            if arm == "B" and s == 0:
                print(f"POSITIVE CONTROL B s0 chain module test_a {res['chain_module_test_a']} (CPU known 0.8703)",
                      flush=True)


def cmd_report(a):
    sets = load_sets(a)
    out = {"seeds": a.seeds, "rounds": a.rounds, "arms": {}, "diffs": {}, "roots": {}, "loop": {}, "judges": {},
           "labels": {}, "rows": {}, "modules_moved": {}}
    res = {arm: {s: json.load(open(f"{a.out}/eval_{arm}_s{s}.json")) for s in a.seeds
                 if os.path.exists(f"{a.out}/eval_{arm}_s{s}.json")} for arm in ["B"] + ARMS}
    for arm, by in res.items():
        if not by:
            continue
        for r in by.values():
            r["eval"]["PRIMARY"] = [x for k in NEW_SETS for x in r["eval"].get(k, [])]
        ks = list(next(iter(by.values()))["eval"])
        out["arms"][arm] = {k: round(sum(sum(r["eval"][k]) / max(1, len(r["eval"][k])) for r in by.values()) / len(by), 4)
                            for k in ks}
        out["arms"][arm]["chain_module_test_a"] = [r["chain_module_test_a"] for r in by.values()]
        tot = {}
        for r in by.values():
            for k in NEW_SETS:
                for z, n in r["roots"].get(k, {}).items():
                    tot[z] = tot.get(z, 0) + n
        out["roots"][arm] = tot
    full = (lambda x: len(res.get(x, {})) == len(a.seeds))
    for x, y in (("ECO", "SEP"), ("ECO", "N"), ("SEP", "N"), ("ECO", "B"), ("SEP", "B"), ("N", "B")):
        if full(x) and full(y):
            out["diffs"][f"{x}-{y}"] = {k: J.boot_diff({s: res[x][s]["eval"][k] for s in a.seeds},
                                                       {s: res[y][s]["eval"][k] for s in a.seeds})
                                        for k in ["PRIMARY"] + NEW_SETS + ["A", "B"] if res[x][a.seeds[0]]["eval"].get(k)}
    moved = {}                                           # per arm/seed: blocks in which any module kept a step > 0
    for arm in ARMS:
        for s in a.seeds:
            n = 0
            for r in range(a.rounds):
                lf = f"{state_dir(a, arm, s, r + 1)}/log.json"
                if os.path.exists(lf):
                    lg = json.load(open(lf))
                    n += any(lg[m]["kept_step"] > 0 and lg["n_rows"][m] > 0 for m in MODS)  # moved on rows
            moved[f"{arm}_s{s}"] = n
    out["blocks_with_movement"] = moved
    if full("ECO") and full("SEP") and full("N"):
        p1, p2 = out["diffs"]["ECO-SEP"]["PRIMARY"], out["diffs"]["ECO-N"]["PRIMARY"]
        ok = all(p["ci95"][0] > 0 and p["seeds_positive"] == len(a.seeds) for p in (p1, p2))
        frozen = [k for k, v in moved.items() if not k.startswith("N_") and v < min(2, a.rounds)]
        out["verdict"] = {"rule": "ECO > SEP and ECO > N on PRIMARY, CI low > 0, same sign in every seed",
                          "ECO-SEP": p1, "ECO-N": p2,
                          "PASS": ok if not frozen else "NO VERDICT: guards froze " + ",".join(frozen),
                          "blocks_with_movement": moved}
    # in-loop learning curve (gold, report only)
    for s in a.seeds:
        curve = []
        for r in range(a.rounds):
            fg = f"{a.out}/gold_s{s}_r{r}.json"
            if os.path.exists(fg):
                g = json.load(open(fg))
                rt = [x["root"] for x in g.values() if not x["pass"]]
                curve.append({"round": r, "pool_pass": round(sum(x["pass"] for x in g.values()) / len(g), 4),
                              "roots": {z: rt.count(z) for z in sorted(set(map(str, rt)))}})
        out["loop"][f"s{s}"] = curve
    # judges vs gold, label precision of accepted rows (report only)
    J_ = {"qwen": [], "final": [], "sonnet_own": []}
    lab, prog_chain = [], []
    pool = {q["id"]: q for p in sets["pool"] for q in p}
    for s in a.seeds:
        for r in range(a.rounds):
            fg = f"{a.out}/gold_s{s}_r{r}.json"
            if not os.path.exists(fg):
                continue
            g = json.load(open(fg))
            qr = {x["id"]: x for x in jl(f"{a.out}/qwen_s{s}_r{r}.jsonl")}
            fr = {x["id"]: x for x in jl(f"{a.out}/final_s{s}_r{r}.jsonl")}
            recs = {x["id"]: x for x in json.load(open(f"{a.out}/belt_s{s}_r{r}.json"))}
            for iid, gx in g.items():
                truth = gx["root"] or "none"
                whos = [("qwen", (qr.get(iid) or {}).get("qwen")), ("final", (fr.get(iid) or {}).get("final"))]
                if (fr.get(iid) or {}).get("why"):
                    whos.append(("sonnet_own", norm_verdict((fr.get(iid) or {}).get("sonnet_raw"))))
                if truth not in STAGES:                      # e2e "unexplained": no judge can match it
                    out["judges_excluded"] = out.get("judges_excluded", 0) + 1
                    whos = []
                for who, v in whos:
                    J_[who].append({"truth": truth, "stage": (v or {}).get("stage"), "valid": v is not None,
                                    "flags": recs[iid]["flags"]})
                f_ = fr.get(iid) or {}
                if f_.get("accepted"):
                    q = pool[iid.split("_", 2)[2]]
                    gb = canon_gold(q["gold"])
                    gk, gbr = gold_kb(q)
                    fx = f_["fix"]
                    for row in fx["rows"]:
                        if row["module"] == "understander":
                            lab.append({"module": "understander", "exact": row["spec"] == gb})
                        elif row["module"].startswith("planner"):    # vs the GOLD plan of the request
                            lab.append({"module": row["module"], "exact": row["syms"] == rule_plan(gk, gbr)})
                        elif row["module"] == "writer_chain":       # behaviour on the 40 inputs, style may differ
                            lab.append({"module": "writer_chain", "exact": J.block_same(row["text"], row["block"])})
                        else:
                            lab.append({"module": "writer_calc", "exact": row["text"] == MF.block_text(row["sym"])})
                    lab.append({"module": "blame_vs_gold_root", "exact": fx["rows"][0]["module"].split("_")[0] == gx["root"]})
                    if q["world"] == "chain":                    # corrected program vs the request gold (batched below)
                        prog_chain.append((q["task"], fx["code"]))
                    else:                                        # calc: right spec AND every corrected block exact
                        ok = fx["kind"] == gk and canon(fx["kind"], fx["bridged"]) == canon(gk, gbr)
                        ok = ok and list(fx["plan"]) == rule_plan(gk, gbr)      # whole program, not only the fix
                        if fx["rows"][0]["module"] == "writer_calc":
                            ok = ok and all(t == MF.block_text(b_["key"]) for b_, t in
                                            zip(recs[iid]["blocks"], fx.get("texts", [])))
                        lab.append({"module": "program_vs_request_gold", "exact": bool(ok)})
    if prog_chain:
        for sc in S2.grade([t for t, _ in prog_chain], [[c] for _, c in prog_chain]):
            lab.append({"module": "program_vs_request_gold", "exact": sc[0] == 1.0})
    rate = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
    for who, rs in J_.items():
        fail = [x for x in rs if x["truth"] != "none"]
        out["judges"][who] = {"n": len(rs), "valid": rate([x["valid"] for x in rs]),
                              "pass_agree": rate([(x["stage"] == "none") == (x["truth"] == "none") for x in rs if x["valid"]]),
                              "stage_acc_on_fails": rate([x["stage"] == x["truth"] for x in fail]), "n_fails": len(fail)}
    fl = [x for x in J_["qwen"] if x["truth"] != "none"]
    out["judges"]["flag_rule_baseline"] = {"stage_acc_on_fails": rate([
        (next((f.split(":")[0] for f in x["flags"] if f.split(":")[0] in STAGES), "planner")) == x["truth"] for x in fl])}
    out["judges"]["constant_baseline"] = {k: rate([x["truth"] == k for x in fl]) for k in STAGES[1:]}
    out["labels"] = {m: {"n": len([x for x in lab if x["module"] == m]), "precision": rate([x["exact"] for x in lab
                                                                                          if x["module"] == m])}
                     for m in sorted({x["module"] for x in lab})}
    for arm in ARMS:
        for s in a.seeds:
            for r in range(a.rounds):
                lf = f"{state_dir(a, arm, s, r + 1)}/log.json"
                if os.path.exists(lf):
                    lg = json.load(open(lf))
                    out["rows"][f"{arm}_s{s}_r{r}"] = lg["n_rows"]
                    out["modules_moved"][f"{arm}_s{s}_r{r}"] = {m: [lg[m]["kept_step"], lg[m]["train_ok_kept"],
                                                                    lg[m]["train_ok_base"]] for m in MODS}
    notes = {}
    for s in a.seeds:
        for r in range(a.rounds):
            for x in jl(f"{a.out}/final_s{s}_r{r}.jsonl"):
                notes[x.get("note")] = notes.get(x.get("note"), 0) + 1
    out["final_notes"] = notes
    out["sonnet_calls"] = len(jl(f"{a.out}/sonnet_calls.jsonl"))
    jdump(out, f"{a.out}/verdict.json")
    print(json.dumps({k: out[k] for k in ("arms", "roots", "loop", "judges", "labels", "final_notes", "sonnet_calls")}),
          flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("PRIMARY", "A", "B")} for k, v in out["diffs"].items()}),
          flush=True)
    if "verdict" in out:
        print("VERDICT " + json.dumps(out["verdict"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "p0", "u0", "belt", "qwen", "sonnet", "train", "arms", "eval", "report"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seeds", default="0,5,6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--pool-chain", type=int, default=32)
    ap.add_argument("--pool-calc", type=int, default=16)
    ap.add_argument("--arms", default="B,ECO,SEP,N")
    ap.add_argument("--no-model", action="store_true", help="p0: only the cached-raw and chain module controls")
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--qwen-batch", type=int, default=8)
    ap.add_argument("--qwen-max-new", type=int, default=400)
    ap.add_argument("--mem-frac", type=float, default=0.8)
    ap.add_argument("--max-sonnet-calls", type=int, default=260)
    ap.add_argument("--batch", type=int, default=6, help="items per Sonnet call")
    ap.add_argument("--net-steps", type=int, default=100, help="chain planner/writer steps per block (calc nets: half)")
    ap.add_argument("--u-steps", type=int, default=30, help="understander steps per block")
    ap.add_argument("--u0-steps", type=int, default=150)
    ap.add_argument("--u-batch", type=int, default=16)
    ap.add_argument("--u-micro", type=int, default=4)
    ap.add_argument("--u-lr", type=float, default=5e-5)
    ap.add_argument("--mix", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    a.seeds = [int(x) for x in a.seeds.split(",")]
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    {"build": cmd_build, "p0": cmd_p0, "u0": cmd_u0, "belt": cmd_belt, "qwen": cmd_qwen, "sonnet": cmd_sonnet,
     "train": cmd_train, "arms": cmd_arms, "eval": cmd_eval, "report": cmd_report}[a.phase](a)


if __name__ == "__main__":
    main()
