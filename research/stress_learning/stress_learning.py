"""Stress-gated learning on demand (Qwen3-0.6B + LoRA).

A question the model cannot answer produces "stress" S = mean NLL of the
correct answer (taken from a fast fact bank). If S exceeds a threshold the
model answers from the bank and consolidates the fact into LoRA weights with
a learning rate proportional to S, replaying previously learned facts.

Arms (same fact set, same question stream, fresh LoRA each):
  never   - never trains (pure bank lookup)
  always  - one training step on every question (induced learning)
  stress  - trains only when S > theta, lr ~ S, stops when S < theta_stop
  random  - replays the exact stress schedule (event count, steps, lrs) on
            randomly chosen questions: does stress select better than chance?

Facts are invented (entity and answer are random pseudo-words), so the base
model cannot know them. Each fact has 5 paraphrases: 3 used in the stream,
2 held out to test that the fact was learned and not the question string.
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

MODEL = ("/media/llm-data/models/hf/hub/models--Qwen--Qwen3-0.6B/snapshots/"
         "c1899de289a04d12100db370d81485cdf75e47ca")

SYLLABLES = ["ba", "ce", "di", "fo", "gu", "ka", "le", "mi", "no", "pu", "ra",
             "se", "ti", "vo", "xu", "za", "bel", "dor", "fen", "gar", "hul",
             "jor", "kel", "lom", "mur", "nev", "pal", "quin", "ros", "tal",
             "vin", "zem"]

RELATIONS = {
    "capital": [
        "Qual é a capital de {e}?",
        "A capital de {e} é qual cidade?",
        "Diga a capital de {e}.",
        "Em {e}, qual cidade é a capital?",
        "Qual cidade é a capital do país {e}?",
    ],
    "inventor": [
        "Quem inventou o {e}?",
        "O {e} foi inventado por quem?",
        "Qual o nome de quem inventou o {e}?",
        "Quem foi o inventor do {e}?",
        "Diga quem criou o {e}.",
    ],
    "river": [
        "Qual rio passa pela cidade de {e}?",
        "A cidade de {e} é cortada por qual rio?",
        "Diga o rio que atravessa {e}.",
        "Que rio corre em {e}?",
        "Qual é o rio da cidade de {e}?",
    ],
    "language": [
        "Que língua fala o povo {e}?",
        "O povo {e} fala qual idioma?",
        "Diga o idioma do povo {e}.",
        "Qual é a língua dos {e}?",
        "Em que língua conversam os {e}?",
    ],
}


def pseudo_word(rng, n):
    return "".join(rng.choice(SYLLABLES) for _ in range(n)).capitalize()


def make_facts(rng, n_facts):
    used, facts = set(), []
    rels = list(RELATIONS)
    while len(facts) < n_facts:
        e, a = pseudo_word(rng, 3), pseudo_word(rng, rng.choice([2, 3]))
        if e in used or a in used:
            continue
        used.update([e, a])
        rel = rels[len(facts) % len(rels)]
        order = list(range(5))
        rng.shuffle(order)
        facts.append({"id": len(facts), "rel": rel, "entity": e, "answer": a,
                      "train_para": order[:3], "heldout_para": order[3:]})
    return facts


def make_stream(rng, n_facts, n_questions, zipf_s):
    ranks = list(range(n_facts))
    rng.shuffle(ranks)
    weights = [1.0 / (ranks[i] + 1) ** zipf_s for i in range(n_facts)]
    stream = []
    for _ in range(n_questions):
        f = rng.choices(range(n_facts), weights=weights)[0]
        stream.append((f, rng.randrange(3)))
    return stream


def train_paraphrases(facts, stream):
    """Map the stream's paraphrase slot (0-2) to the fact's own train paraphrase.

    Until 2026-10-13 the slot was used directly as the paraphrase index, so
    ~42% of stream questions used a paraphrase listed as held out.
    """
    return [(f, facts[f]["train_para"][k]) for f, k in stream]


class Encoder:
    def __init__(self, tok, facts, device):
        self.tok, self.facts, self.device = tok, facts, device
        self.cache = {}

    def pair(self, fid, para):
        key = (fid, para)
        if key not in self.cache:
            f = self.facts[fid]
            q = RELATIONS[f["rel"]][para].format(e=f["entity"])
            p = self.tok(f"Pergunta: {q}\nResposta:", add_special_tokens=False)["input_ids"]
            t = self.tok(f" {f['answer']}.", add_special_tokens=False)["input_ids"]
            self.cache[key] = (p, t)
        return self.cache[key]

    def batch(self, items):
        seqs = [self.pair(f, p) for f, p in items]
        L = max(len(p) + len(t) for p, t in seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        lab = torch.full((len(seqs), L), -100, dtype=torch.long)
        for i, (p, t) in enumerate(seqs):
            n = len(p) + len(t)
            ids[i, :n] = torch.tensor(p + t)
            att[i, :n] = 1
            lab[i, len(p):n] = torch.tensor(t)
        return ids.to(self.device), att.to(self.device), lab.to(self.device)


def answer_stats(model, enc, items):
    """Per item: mean NLL of answer tokens and teacher-forced exact match
    (argmax equals target at every answer position == greedy exact match)."""
    ids, att, lab = enc.batch(items)
    hidden = model(input_ids=ids, attention_mask=att,
                   output_hidden_states=True).hidden_states[-1][:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != -100
    # project only answer positions to the vocabulary (full logits OOM)
    logits = model.get_output_embeddings()(hidden[mask]).float()
    logp = torch.log_softmax(logits, -1)
    t = tgt[mask]
    tok_nll = -logp.gather(-1, t.unsqueeze(-1)).squeeze(-1)
    tok_hit = logits.argmax(-1) == t
    row = mask.nonzero()[:, 0]
    n = mask.sum(1)
    nll = torch.zeros(len(items), device=ids.device).index_add_(0, row, tok_nll) / n
    hit = torch.zeros(len(items), device=ids.device).index_add_(0, row, tok_hit.float()) == n
    return nll.tolist(), hit.tolist()


def train_step(model, opt, enc, items, lr):
    for g in opt.param_groups:
        g["lr"] = lr
    model.train()
    ids, att, lab = enc.batch(items)
    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    model.eval()
    return loss.item()


def load_model(seed, device):
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, cfg)
    model.eval()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, weight_decay=0.0)
    return model, opt


@torch.no_grad()
def evaluate(model, enc, facts, seen_counts, bs=64):
    seen = [f for f in range(len(facts)) if seen_counts[f] > 0]
    if not seen:
        return {"n_seen": 0}
    out = {}
    for split in ("train_para", "heldout_para"):
        items = [(f, p) for f in seen for p in facts[f][split]]
        hits = []
        for i in range(0, len(items), bs):
            hits += answer_stats(model, enc, items[i:i + bs])[1]
        per_fact = {}
        for (f, _), h in zip(items, hits):
            per_fact.setdefault(f, []).append(h)
        acc = {f: sum(v) / len(v) for f, v in per_fact.items()}
        total = sum(seen_counts[f] for f in seen)
        out[split] = {
            "acc": sum(acc.values()) / len(acc),
            "acc_freq_weighted": sum(acc[f] * seen_counts[f] for f in seen) / total,
            "per_fact": acc,
        }
    out["n_seen"] = len(seen)
    return out


def run_arm(arm, args, facts, stream, device, schedule=None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = Encoder(tok, facts, device)
    model, opt = load_model(args.seed, device)
    rng = random.Random(args.seed * 1000 + 7)
    seen = [0] * len(facts)
    buffer, buffer_set = [], set()
    events, curve = [], []
    steps = consults = served_ok = 0
    nll_log = []

    random_plan = {}
    if arm == "random":
        times = rng.sample(range(len(stream)), len(schedule))
        for t, ev in zip(sorted(times), schedule):
            random_plan[t] = ev

    def replay_batch(fid, para):
        items = [(fid, para)]
        if buffer:
            for _ in range(args.replay):
                rf = rng.choice(buffer)
                items.append((rf, rng.choice(facts[rf]["train_para"])))
        return items

    def remember(fid):
        if fid not in buffer_set:
            buffer_set.add(fid)
            buffer.append(fid)

    t0 = time.time()
    for t, (fid, para) in enumerate(stream):
        seen[fid] += 1
        with torch.no_grad():
            (s,), (hit,) = answer_stats(model, enc, [(fid, para)])
        knows = s < args.theta
        nll_log.append((round(s, 4), bool(hit)))
        if arm == "never" or (arm in ("stress", "random") and not knows):
            consults += 1
            served_ok += 1  # bank answer is correct by construction
        elif arm == "always":
            served_ok += int(hit)
        else:
            served_ok += int(hit)

        if arm == "always":
            remember(fid)
            train_step(model, opt, enc, replay_batch(fid, para), args.always_lr)
            steps += 1
        elif arm == "stress" and not knows:
            remember(fid)
            lrs, losses, s_now = [], [], s
            for _ in range(args.max_steps):
                lr = args.lr * min(s_now / args.theta, args.lr_gain_cap)
                losses.append(train_step(model, opt, enc, replay_batch(fid, para), lr))
                lrs.append(lr)
                steps += 1
                with torch.no_grad():
                    (s_now,), _ = answer_stats(model, enc, [(fid, para)])
                if s_now < args.theta_stop:
                    break
            events.append({"t": t, "fid": fid, "s_before": s, "s_after": s_now,
                           "lrs": lrs, "losses": losses})
            if args.debug and len(events) % 25 == 0:
                recent = events[-25:]
                print(json.dumps({"debug": arm, "t": t, "events": len(events),
                                  "s_before": round(sum(e["s_before"] for e in recent) / 25, 3),
                                  "s_after": round(sum(e["s_after"] for e in recent) / 25, 3),
                                  "steps": sum(len(e["lrs"]) for e in recent),
                                  "loss_last": round(recent[-1]["losses"][-1], 3)}), flush=True)
        elif arm == "random" and t in random_plan:
            remember(fid)
            for lr in random_plan[t]["lrs"]:
                train_step(model, opt, enc, replay_batch(fid, para), lr)
                steps += 1
            events.append({"t": t, "fid": fid, "lrs": random_plan[t]["lrs"]})

        periodic = (t + 1) % args.eval_every == 0 and arm != "never"
        if periodic or t + 1 == len(stream):
            ev = evaluate(model, enc, facts, seen)
            row = {"t": t + 1, "steps": steps, "consults": consults,
                   "served_acc": served_ok / (t + 1), "n_seen": ev["n_seen"],
                   "acc_train": ev["train_para"]["acc"],
                   "acc_heldout": ev["heldout_para"]["acc"],
                   "acc_heldout_fw": ev["heldout_para"]["acc_freq_weighted"],
                   "elapsed_s": round(time.time() - t0, 1)}
            curve.append(row)
            print(json.dumps({"arm": arm, "seed": args.seed, **row}), flush=True)
            last_eval = ev

    # retention: facts whose first stress/training event was in the first half
    first_t = {}
    for e in events:
        first_t.setdefault(e["fid"], e["t"])
    early = [f for f, t in first_t.items() if t < len(stream) // 2]
    final_heldout = last_eval["heldout_para"]["per_fact"]
    retention = (sum(final_heldout.get(f, 0.0) for f in early) / len(early)) if early else None

    del model, opt
    torch.cuda.empty_cache()
    return {"arm": arm, "curve": curve, "events": events, "steps": steps,
            "consults": consults, "retention_early_heldout": retention,
            "n_early": len(early), "stream_nll_hit": nll_log,
            "final_per_fact_heldout": final_heldout,
            "final_per_fact_train": last_eval["train_para"]["per_fact"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--facts", type=int, default=500)
    ap.add_argument("--questions", type=int, default=5000)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--theta-stop", type=float, default=0.2)
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--always-lr", type=float, default=3e-4)
    ap.add_argument("--lr-gain-cap", type=float, default=3.0)
    ap.add_argument("--replay", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--arms", default="stress,random,always,never")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    rng = random.Random(args.seed)
    facts = make_facts(rng, args.facts)
    stream = train_paraphrases(facts, make_stream(rng, args.facts, args.questions, args.zipf))
    with open(os.path.join(args.out, f"seed{args.seed}_setup.json"), "w") as fh:
        json.dump({"args": vars(args), "facts": facts, "stream": stream}, fh)

    schedule = None
    for arm in args.arms.split(","):
        if arm == "random":
            path = os.path.join(args.out, f"seed{args.seed}_stress.json")
            with open(path) as fh:
                schedule = [{"lrs": e["lrs"]} for e in json.load(fh)["events"]]
        res = run_arm(arm, args, facts, stream, device, schedule)
        res["args"] = vars(args)
        with open(os.path.join(args.out, f"seed{args.seed}_{arm}.json"), "w") as fh:
            json.dump(res, fh)


if __name__ == "__main__":
    main()
