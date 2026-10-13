"""Fast engine for the stress / sleep batteries (phase 1: pure PyTorch).

Same per-question semantics as stress_learning.py and sleep_learning.py; only
the execution changes:

- every (fact, paraphrase) pair is tokenized once into a GPU table with fixed
  length and fixed answer slots, so no per-step CPU tensor building;
- the model only changes when it trains, so the next questions are scored in
  one batched forward ("scoring window") and the window is cut at the first
  question that triggers training; all questions before it were scored with
  exactly the weights the sequential loop would have used;
- `always` reads its served answer from the training forward itself
  (item 0 is the current question, scored before the update);
- losses and statistics use answer positions only, and stay on the GPU until
  one read per window / per eval.

Optional `--dtype fp16`: frozen base in fp16 under autocast, LoRA in fp32,
fixed-scale GradScaler, answer logits in fp32. Only adopt it after
check_equivalence.py shows S matches fp32.

Arms: never, always, wake (== stress), random (matched to wake), wake_replay,
sleep_uniform, sleep_fading.
"""

import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from stress_learning import MODEL, RELATIONS, make_facts, make_stream, train_paraphrases

ARMS = ("never", "always", "wake", "random", "wake_replay", "sleep_uniform", "sleep_fading")
GATED = ("wake", "wake_replay", "sleep_uniform", "sleep_fading")


class PairTable:
    """All (fact, paraphrase) pairs tokenized once, right-padded, on the GPU."""

    def __init__(self, tok, facts, device):
        n_para = len(next(iter(RELATIONS.values())))
        self.n_para = n_para
        rows = []
        for f in facts:
            for para in range(n_para):
                q = RELATIONS[f["rel"]][para].format(e=f["entity"])
                p = tok(f"Pergunta: {q}\nResposta:", add_special_tokens=False)["input_ids"]
                t = tok(f" {f['answer']}.", add_special_tokens=False)["input_ids"]
                rows.append((p, t))
        L = max(len(p) + len(t) for p, t in rows)
        T = max(len(t) for _, t in rows)
        N = len(rows)
        ids = torch.full((N, L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((N, L), dtype=torch.long)
        pos = torch.zeros((N, T), dtype=torch.long)
        tgt = torch.zeros((N, T), dtype=torch.long)
        valid = torch.zeros((N, T), dtype=torch.bool)
        for i, (p, t) in enumerate(rows):
            n = len(p) + len(t)
            ids[i, :n] = torch.tensor(p + t)
            att[i, :n] = 1
            # hidden state at position j predicts token j+1
            pos[i, :len(t)] = torch.arange(len(p) - 1, n - 1)
            tgt[i, :len(t)] = torch.tensor(t)
            valid[i, :len(t)] = True
        self.ids, self.att, self.pos = ids.to(device), att.to(device), pos.to(device)
        self.tgt, self.valid = tgt.to(device), valid.to(device)
        self.n_tok = self.valid.sum(1).float()
        self.device = device
        self.L, self.T = L, T

    def index(self, items):
        return torch.tensor([f * self.n_para + p for f, p in items], device=self.device)


class Engine:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.fp16 = args.dtype == "fp16"
        torch.manual_seed(args.seed)
        dtype = torch.float16 if self.fp16 else torch.float32
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device)
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, cfg)
        if self.fp16:
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        self.model.eval()
        self.body = self.model.get_base_model().model  # transformer without lm_head
        emb = self.model.get_output_embeddings().weight
        self.head_w = emb.detach().float() if self.fp16 else emb  # fp32 answer logits
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=1e-4, weight_decay=0.0)
        self.scaler = (torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=10**9)
                       if self.fp16 else None)

    def _answer_stats(self, tab, idx):
        """Per row: mean answer NLL [B], exact-match hit [B], token NLL sum and count."""
        ctx = torch.autocast("cuda", torch.float16) if self.fp16 else torch.autocast("cuda", enabled=False)
        with ctx:
            h = self.body(input_ids=tab.ids[idx], attention_mask=tab.att[idx]).last_hidden_state
        pos, tgt, valid = tab.pos[idx], tab.tgt[idx], tab.valid[idx]
        hs = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))      # [B,T,H]
        logits = torch.nn.functional.linear(hs.float(), self.head_w.float())  # [B,T,V]
        logp = torch.log_softmax(logits, -1)
        tok_nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * valid
        tok_hit = (logits.argmax(-1) == tgt) | ~valid
        n = tab.n_tok[idx]
        return tok_nll.sum(1) / n, tok_hit.all(1), tok_nll.sum(), n.sum()

    @torch.no_grad()
    def score(self, tab, idx, bs=128):
        out_s, out_h = [], []
        for i in range(0, len(idx), bs):
            s, h, _, _ = self._answer_stats(tab, idx[i:i + bs])
            out_s.append(s)
            out_h.append(h)
        return torch.cat(out_s), torch.cat(out_h)

    def train_step(self, tab, idx, lr):
        """One AdamW step on the token-mean answer loss (same as HF labels loss).
        Returns pre-update (S, hit) of the rows, still on the GPU."""
        for g in self.opt.param_groups:
            g["lr"] = lr
        s, hit, nll_sum, n_sum = self._answer_stats(tab, idx)
        loss = nll_sum / n_sum
        self.opt.zero_grad(set_to_none=True)
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
        return s.detach(), hit.detach(), loss.detach()


@torch.no_grad()
def evaluate(eng, tab, facts, seen_counts):
    seen = [f for f in range(len(facts)) if seen_counts[f] > 0]
    if not seen:
        return {"n_seen": 0}
    out = {"n_seen": len(seen)}
    for split in ("train_para", "heldout_para"):
        items = [(f, p) for f in seen for p in facts[f][split]]
        _, hits = eng.score(tab, tab.index(items))
        hits = hits.tolist()
        per_fact = {}
        for (f, _), h in zip(items, hits):
            per_fact.setdefault(f, []).append(h)
        acc = {f: sum(v) / len(v) for f, v in per_fact.items()}
        total = sum(seen_counts[f] for f in seen)
        out[split] = {"acc": sum(acc.values()) / len(acc),
                      "acc_freq_weighted": sum(acc[f] * seen_counts[f] for f in seen) / total,
                      "per_fact": acc}
    return out


def run_arm(arm, args, facts, stream, tab, schedule=None):
    eng = Engine(args, "cuda")
    rng = random.Random(args.seed * 1000 + 7)  # same draw order as the old scripts
    seen = [0] * len(facts)
    buffer, buffer_set = [], set()
    events, sleeps, curve, nll_log = [], [], [], []
    counters = {"wake_steps": 0, "sleep_steps": 0, "item_grads": 0, "scored": 0,
                "consults": 0, "served_ok": 0, "forwards": 0}
    replay_n = args.matched_replay if arm == "wake_replay" else args.replay
    lr_always = args.always_lr

    random_plan = {}
    if arm == "random":
        times = rng.sample(range(len(stream)), len(schedule))
        for t, ev in zip(sorted(times), schedule):
            random_plan[t] = ev

    def remember(fid):
        if fid not in buffer_set:
            buffer_set.add(fid)
            buffer.append(fid)

    def replay_items(fid, para):
        items = [(fid, para)]
        if buffer:
            for _ in range(replay_n):
                rf = rng.choice(buffer)
                items.append((rf, rng.choice(facts[rf]["train_para"])))
        return items

    def train(items, lr, kind):
        out = eng.train_step(tab, tab.index(items), lr)
        counters[kind] += 1
        counters["item_grads"] += len(items)
        return out

    def stress_event(t, fid, para, s):
        remember(fid)
        lrs, s_now = [], s
        one = tab.index([(fid, para)])
        for _ in range(args.max_steps):
            lr = args.lr * min(s_now / args.theta, args.lr_gain_cap)
            train(replay_items(fid, para), lr, "wake_steps")
            lrs.append(lr)
            s_now = eng.score(tab, one)[0].item()
            counters["forwards"] += 1
            if s_now < args.theta_stop:
                break
        events.append({"t": t, "fid": fid, "s_before": s, "s_after": s_now, "lrs": lrs})

    def sleep_phase(t):
        if not buffer:
            return
        if arm == "sleep_uniform":
            for _ in range(args.sleep_steps):
                fids = [rng.choice(buffer) for _ in range(args.sleep_batch)]
                train([(f, rng.choice(facts[f]["train_para"])) for f in fids], args.sleep_lr, "sleep_steps")
            sleeps.append({"t": t, "steps": args.sleep_steps})
            return
        items = [(f, rng.choice(facts[f]["train_para"])) for f in buffer]
        s_all = eng.score(tab, tab.index(items))[0].tolist()
        counters["scored"] += len(items)
        fading = sorted([x for x in zip(items, s_all) if x[1] > args.fade_theta], key=lambda x: -x[1])
        n_steps = min(args.sleep_steps, -(-len(fading) // args.sleep_batch))
        queue = [it for it, _ in fading]
        for k in range(n_steps):
            batch = queue[k * args.sleep_batch:(k + 1) * args.sleep_batch]
            while len(batch) < args.sleep_batch:
                f = rng.choice(buffer)
                batch.append((f, rng.choice(facts[f]["train_para"])))
            train(batch, args.sleep_lr, "sleep_steps")
        sleeps.append({"t": t, "buffer": len(buffer), "fading": len(fading), "steps": n_steps,
                       "mean_fading_s": (sum(s for _, s in fading) / len(fading)) if fading else 0.0})

    def boundary(t):
        """Called after question t (0-based) is fully processed."""
        nonlocal last_eval
        n = t + 1
        if (n % args.eval_every == 0 and arm != "never") or n == len(stream):
            ev = evaluate(eng, tab, facts, seen)
            relearned = len(events) - len({e["fid"] for e in events})
            row = {"t": n, **{k: v for k, v in counters.items() if k != "served_ok"},
                   "relearn_events": relearned, "served_acc": counters["served_ok"] / n,
                   "n_seen": ev["n_seen"], "acc_train": ev["train_para"]["acc"],
                   "acc_heldout": ev["heldout_para"]["acc"],
                   "acc_heldout_fw": ev["heldout_para"]["acc_freq_weighted"],
                   "elapsed_s": round(time.time() - t0, 1)}
            curve.append(row)
            print(json.dumps({"arm": arm, "seed": args.seed, **row}), flush=True)
            last_eval = ev
        # sleep after the eval, never after the last question
        if arm.startswith("sleep") and n % args.sleep_every == 0 and n < len(stream):
            sleep_phase(n)

    period = math.gcd(args.eval_every, args.sleep_every)
    last_eval = None
    t0 = time.time()
    t = 0
    window = args.window
    while t < len(stream):
        if arm == "always":
            fid, para = stream[t]
            seen[fid] += 1
            remember(fid)
            s, hit, _ = train(replay_items(fid, para), lr_always, "wake_steps")
            s0, h0 = s[0].item(), bool(hit[0].item())
            nll_log.append((round(s0, 4), h0))
            counters["served_ok"] += int(h0)
            boundary(t)
            t += 1
            continue

        # scoring window: never crosses an eval/sleep boundary
        end = min(t + window, (t // period + 1) * period, len(stream))
        s_win, h_win = eng.score(tab, tab.index(stream[t:end]))
        counters["forwards"] += 1
        s_win, h_win = s_win.tolist(), h_win.tolist()
        cut = None
        for i in range(end - t):
            q = t + i
            fid, para = stream[q]
            s, hit = s_win[i], bool(h_win[i])
            seen[fid] += 1
            nll_log.append((round(s, 4), hit))
            knows = s < args.theta
            if arm == "never" or not knows:
                counters["consults"] += 1
                counters["served_ok"] += 1  # bank answer is correct by construction
            else:
                counters["served_ok"] += int(hit)
            if arm in GATED and not knows:
                stress_event(q, fid, para, s)
                cut = i
            elif arm == "random" and q in random_plan:
                remember(fid)
                for lr in random_plan[q]["lrs"]:
                    train(replay_items(fid, para), lr, "wake_steps")
                events.append({"t": q, "fid": fid, "lrs": random_plan[q]["lrs"]})
                cut = i
            boundary(q)
            if cut is not None:
                break
        if cut is None:
            t = end
            window = min(args.window, window * 2)
        else:
            t = t + cut + 1
            window = max(args.window_min, min(args.window, 2 * (cut + 1)))

    stressed = {e["fid"] for e in events}
    counts = {}
    for e in events:
        counts[e["fid"]] = counts.get(e["fid"], 0) + 1
    first_t = {}
    for e in events:
        first_t.setdefault(e["fid"], e["t"])
    early = [f for f, ft in first_t.items() if ft < len(stream) // 2]
    final_heldout = last_eval["heldout_para"]["per_fact"]
    result = {
        "arm": arm, "curve": curve, "events": events, "sleeps": sleeps,
        **{k: v for k, v in counters.items() if k != "served_ok"},
        "served_acc": counters["served_ok"] / len(stream),
        "stressed_facts": len(stressed),
        "relearned_fraction": sum(1 for v in counts.values() if v > 1) / max(1, len(stressed)),
        "retention_early_heldout": (sum(final_heldout.get(f, 0.0) for f in early) / len(early)) if early else None,
        "stream_nll_hit": nll_log,
        "final_per_fact_heldout": final_heldout,
        "final_per_fact_train": last_eval["train_para"]["per_fact"],
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "wall_s": round(time.time() - t0, 1),
    }
    del eng
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts", type=int, default=200)
    ap.add_argument("--questions", type=int, default=2000)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-stop", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--always-lr", type=float, default=3e-4)
    ap.add_argument("--lr-gain-cap", type=float, default=3.0)
    ap.add_argument("--replay", type=int, default=3)
    ap.add_argument("--matched-replay", type=int, default=7)
    ap.add_argument("--sleep-every", type=int, default=100)
    ap.add_argument("--sleep-steps", type=int, default=8)
    ap.add_argument("--sleep-batch", type=int, default=16)
    ap.add_argument("--sleep-lr", type=float, default=1e-4)
    ap.add_argument("--fade-theta", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--window-min", type=int, default=4)
    ap.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--arms", default="wake,random,always,wake_replay,sleep_uniform,sleep_fading")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    facts = make_facts(rng, args.facts)
    stream = train_paraphrases(facts, make_stream(rng, args.facts, args.questions, args.zipf))
    with open(os.path.join(args.out, f"seed{args.seed}_setup.json"), "w") as fh:
        json.dump({"args": vars(args), "facts": facts, "stream": stream}, fh)
    tab = PairTable(AutoTokenizer.from_pretrained(MODEL), facts, "cuda")

    schedule = None
    for arm in args.arms.split(","):
        assert arm in ARMS, arm
        if arm == "random":
            with open(os.path.join(args.out, f"seed{args.seed}_wake.json")) as fh:
                schedule = [{"lrs": e["lrs"]} for e in json.load(fh)["events"]]
        res = run_arm(arm, args, facts, stream, tab, schedule)
        res["args"] = vars(args)
        with open(os.path.join(args.out, f"seed{args.seed}_{arm}.json"), "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main()
