"""Representation unit at equal compute: does leaving bytes pay at 3M parameters?

Same synth2 data, same strict judge, same model shape (non-embedding parameters identical), same training loop and
schedule; only the unit differs:
  bytes : 256-symbol vocabulary (the current thesis)
  qwen  : Qwen2.5-Coder byte-level BPE tokens, restricted to the ids that occur in synth2 (train text + prompts)
  learned: a 256-symbol alphabet learned from the training text (line-bounded BPE, digits as slot symbols)
  addr2d : bytes + an address per byte: (row inside the function, column), computed causally from the text itself
  addrseg: addr2d + a block address: on the def line the name segment index (completed `_then_`), in the body the
           unit index (completed `    xs = ...` lines), so segment j of the name and block j of the body share j
           (addr arms add 3 small embedding tables, counted with the embeddings, not in nonemb_params)
  addrrk : relative version (Opus audit 2026-10-23: absolute block 3 / row 13 never occur in train, so addrseg's
           test_b reads an untrained embedding): every function draws r in 0..7 and the tag is (j + r) mod 8 for both
           name segment j and body block j; the row counts lines inside the current block. All tags occur at every
           position in train, so only the name<->body match carries information.
  addrctl: control for addrrk: independent rotations for the name and the body (the match is destroyed)
--quant {none,bin,ter,q2}: every nn.Linear inside the transformer blocks is trained quantized (straight-through
estimator over fp latent weights, per-output-row scale = mean |W|): bin = sign (2 states), ter = BitNet b1.58
round-clip to {-1,0,1} (3 states), q2 = {-1.5,-0.5,0.5,1.5} (4 states). Embeddings, positions, address tables, final
norm and head stay fp; activations stay fp (BitNet also quantizes activations to 8 bit: not done here, declared).
Extra split test_len4: 4 seen units + final, names outside every split (only test_num and len4 separate learning
from a hand-written rule; addr arms use the synth2 grammar `_then_` / `    xs = `, which does not transfer).
Equal FLOPs: every step sees 24576 units (bytes 48 x 512, tokens 128 x 192) for the same number of steps
(6 * N * units, N equal by construction). The token arm therefore sees ~3x more TEXT for the same compute; the
milestone near 1/ratio of the run is the equal-TEXT point.
Evaluated at milestones on test_a (unseen pairs), test_num (held number->slot bindings), test_b (3 units, unseen
length): strict pass@1, decode steps and wall time.

Usage:
  units.py smoke --unit {bytes,qwen,learned}                       (CPU+GPU light: round trip, oracle grade, 30 steps)
  units.py train --unit {bytes,qwen,learned} --out-dir DIR [--steps 3000]
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import synth2  # noqa: E402
from synth2 import grade, name_of, source  # noqa: E402
from synth import summary  # noqa: E402
from eval_samples import cut_function  # noqa: E402
from src.model import ByteGPT  # noqa: E402

TOKENIZER = ("/media/llm-data/models/hf/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/"
             "ea3f2471cf1b1f0db85067f1ef93848e38e88c25/tokenizer.json")
SHAPE = {"bytes": {"ctx": 512, "batch": 48, "gen": 380}, "addr2d": {"ctx": 512, "batch": 48, "gen": 380},
         "addrrk": {"ctx": 512, "batch": 48, "gen": 380}, "addrctl": {"ctx": 512, "batch": 48, "gen": 380},
         "addrseg": {"ctx": 512, "batch": 48, "gen": 380}, "qwen": {"ctx": 192, "batch": 128, "gen": 160},
         # 6.48 bytes/unit: ctx 96 holds ~620 bytes (bytes arm: 512); gen 68 (sample() clamps to ctx - prompt + 1)
         "learned": {"ctx": 96, "batch": 256, "gen": 68}}
LEARNED = "/media/llm-data/datasets/pycode/synth2/learned256.json"
SPLITS = ["test_a", "test_num", "test_b", "test_len4"]
MILESTONES = [0.125, 0.25, 0.35, 0.5, 1.0]
# every prompt starts with the tail of a previous function: in the corpus 80% of `def` follow the merged BPE token
# `)\n\n`, so a bare "\n\n" prompt would be off-distribution for the token arm (identical for both arms)
PREFIX = "    return sum(xs)\n\n"


class Bytes:
    """bytes, restricted (like the token arm) to the symbols seen in the training text."""

    def __init__(self, texts):
        seen = sorted({b for t in texts for b in t.encode()})
        self.to_c = {b: c for c, b in enumerate(seen)}
        self.to_q = seen
        self.vocab = len(seen)

    def encode(self, text):
        return [self.to_c[b] for b in text.encode()]

    def decode(self, ids):
        return bytes(self.to_q[i] for i in ids).decode("utf-8", "replace")

    def count(self, text):
        return len(text.encode())


ROWS, COLS, SEGS = 16, 64, 8
XS_LINE = b"\n    xs = "


def addresses(raw, rots=None, rel=False):
    """raw bytes -> [(row, col, seg)] per byte, causal (byte t sees bytes <= t). A blank line resets everything; `def `
    restarts row/segments; row 0 counts completed `_then_`; rows >= 1 count completed lines starting `    xs = `.
    rots: per `def `, a (r_name, r_body) pair: the segment output becomes (index + r) mod SEGS. rel: the row counts
    lines inside the current block (reset when a `    xs = ` line completes)."""
    row = col = seg = body = 0
    rn, rb = rots[0] if rots else (0, 0)       # bytes before the first `def ` carry the first function's rotation
    ndef = 0
    flag, win, out = False, b"", []
    for c in raw:
        win = (win + bytes([c]))[-10:]
        if c == 10:
            if win.endswith(b"\n\n"):
                row = col = seg = body = 0
                flag = False
            else:
                row = 1 if (rel and flag) else row + 1
                body += flag
                flag = False
                col = 0
        else:
            col += 1
        if win.endswith(b"def "):
            row = seg = body = 0
            if rots is not None:
                rn, rb = rots[ndef]
                ndef += 1
        if row == 0 and win.endswith(b"_then_"):
            seg += 1
        if win == XS_LINE:
            flag = True
        if rots is None:
            sb = min(seg if row == 0 else body, SEGS - 1)
        else:
            sb = (seg + rn) % SEGS if row == 0 else (body + rb) % SEGS
        out.append((min(row, ROWS - 1), min(col, COLS - 1), sb))
    return out


class AddrTracker:
    """the same rule as addresses(), vectorized over a batch for the sampler (one byte per row per step)."""

    def __init__(self, n, device, rn=None, rb=None, rel=False):
        z = lambda: torch.zeros(n, dtype=torch.long, device=device)  # noqa: E731
        self.rn, self.rb, self.rel = rn, rb, rel
        self.row, self.col, self.seg, self.body = z(), z(), z(), z()
        self.flag = torch.zeros(n, dtype=torch.bool, device=device)
        self.win = torch.zeros(n, 10, dtype=torch.long, device=device) - 1
        pat = lambda b: torch.tensor(list(b), device=device)  # noqa: E731
        self.p_nn, self.p_def, self.p_then, self.p_xs = pat(b"\n\n"), pat(b"def "), pat(b"_then_"), pat(XS_LINE)

    def _ends(self, p):
        return (self.win[:, -len(p):] == p).all(1)

    def step(self, c):
        self.win = torch.cat([self.win[:, 1:], c[:, None]], 1)
        nl = c == 10
        blank = nl & self._ends(self.p_nn)
        line = nl & ~blank
        newrow = torch.where(self.flag, 1, self.row + 1) if self.rel else self.row + 1
        self.row = torch.where(line, newrow, self.row)
        self.body = torch.where(line, self.body + self.flag.long(), self.body)
        self.flag = self.flag & ~nl
        self.col = torch.where(nl, 0, self.col + 1)
        reset = blank | self._ends(self.p_def)
        self.row, self.seg, self.body = (torch.where(reset, 0, t) for t in (self.row, self.seg, self.body))
        self.seg = self.seg + ((self.row == 0) & self._ends(self.p_then)).long()
        self.flag = self.flag | self._ends(self.p_xs)
        if self.rn is None:
            sb = torch.where(self.row == 0, self.seg, self.body).clamp(max=SEGS - 1)
        else:      # one function per prompt row: the rotation is fixed per row
            sb = torch.where(self.row == 0, (self.seg + self.rn) % SEGS, (self.body + self.rb) % SEGS)
        return torch.stack([self.row.clamp(max=ROWS - 1), self.col.clamp(max=COLS - 1), sb], -1)


class Qwen:
    """BPE ids of the full Qwen tokenizer, remapped to a compact vocabulary of the ids seen in synth2."""

    def __init__(self, texts):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(TOKENIZER)
        seen = sorted({i for t in texts for i in self.tok.encode(t).ids})
        self.to_c = {i: c for c, i in enumerate(seen)}
        self.to_q = seen
        self.vocab = len(seen)
        self.sig = hashlib.sha1(open(LEARNED, "rb").read()).hexdigest()[:12]   # a regenerated alphabet != old ckpts

    def encode(self, text):
        return [self.to_c[i] for i in self.tok.encode(text).ids]     # KeyError = a token outside the vocabulary

    def decode(self, ids):
        return self.tok.decode([self.to_q[i] for i in ids])

    def count(self, text):                                   # raw tokenizer: never fails on unseen merges
        return len(self.tok.encode(text).ids)


class Learned:
    """a 256-symbol alphabet learned from the training text: BPE capped at 256 entries whose merges never cross a
    newline, never absorb a digit (digits stay single symbols = slots) and start a new piece at `(` (so the prompt
    `def NAME(` is a whole number of units). Trained once and saved (ids must match across processes)."""

    def __init__(self, texts):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        if os.path.exists(LEARNED):
            self.tok = Tokenizer.from_file(LEARNED)
        else:
            self.tok = Tokenizer(models.BPE())
            self.tok.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split("\n", "merged_with_previous"), pre_tokenizers.Split("(", "merged_with_next"),
                pre_tokenizers.Digits(individual_digits=True)])
            chars = sorted({c for t in texts for c in t})
            self.tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=256, initial_alphabet=chars,
                                                                    min_frequency=2, show_progress=False))
            self.tok.save(LEARNED)
        self.itos = {i: t for t, i in self.tok.get_vocab().items()}
        seen = sorted({i for t in texts for i in self.tok.encode(t).ids})
        self.to_c = {i: c for c, i in enumerate(seen)}
        self.to_q = seen
        self.vocab = len(seen)
        self.sig = hashlib.sha1(open(LEARNED, "rb").read()).hexdigest()[:12]   # a regenerated alphabet != old ckpts

    def encode(self, text):
        return [self.to_c[i] for i in self.tok.encode(text).ids]     # KeyError = a unit outside the vocabulary

    def decode(self, ids):
        return "".join(self.itos[self.to_q[i]] for i in ids)

    def count(self, text):
        return len(self.tok.encode(text).ids)


def prompt_ids(unit, units, final):
    """the prompt is PREFIX + `def NAME(` (bytes) or the longest token prefix of PREFIX + reference source whose
    text is a prefix of it (tokens: the in-corpus tokenization of the name, never a split token at the boundary)."""
    head = f"{PREFIX}def {name_of(units, final)}("
    if isinstance(unit, Bytes):
        return unit.encode(head)
    ids = unit.encode(PREFIX + source(units, final))
    n = 0
    while n < len(ids) and head.startswith(unit.decode(ids[:n + 1])):
        n += 1
    return ids[:n]


class UnitGPT(ByteGPT):
    """ByteGPT with a vocabulary of any size (embedding and head replaced; blocks identical across units)."""

    def __init__(self, vocab, dim, layers, heads, ctx):
        super().__init__(dim=dim, n_layers=layers, n_heads=heads, context=ctx)
        self.tok = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)
        self.addr_mode = None


def quantize(w, mode):
    s = w.abs().mean(1, keepdim=True).clamp(min=1e-8)
    if mode == "bin":
        q = torch.where(w >= 0, 1.0, -1.0) * s
    elif mode == "ter":
        q = (w / s).round().clamp(-1, 1) * s
    elif mode == "q2":
        q = ((w / s).floor() + 0.5).clamp(-1.5, 1.5) * s
    else:
        raise ValueError(mode)
    return w + (q - w).detach()                               # forward: q; backward: identity to the latent w


class QLinear(nn.Linear):
    mode = "ter"

    def forward(self, x):
        return F.linear(x, quantize(self.weight, self.mode), self.bias)


def quantize_blocks(net, mode):
    for blk in net.blocks:
        for name, m in list(blk.named_modules()):
            if type(m) is nn.Linear:
                q = QLinear(m.in_features, m.out_features).to(m.weight.device)
                q.mode = mode
                q.load_state_dict(m.state_dict())
                parent = blk.get_submodule(name.rsplit(".", 1)[0]) if "." in name else blk
                setattr(parent, name.rsplit(".", 1)[-1], q)
    return net


class AddrGPT(UnitGPT):
    """UnitGPT + address embeddings added to the input (mode addr2d: row + col; addrseg: row + col + segment)."""

    def __init__(self, vocab, dim, layers, heads, ctx, mode):
        super().__init__(vocab, dim, layers, heads, ctx)
        self.addr_mode = mode
        self.a_row, self.a_col = nn.Embedding(ROWS, dim), nn.Embedding(COLS, dim)
        self.a_seg = nn.Embedding(SEGS, dim) if mode in ("addrseg", "addrrk", "addrctl") else None

    def addr_emb(self, a):
        e = self.a_row(a[..., 0]) + self.a_col(a[..., 1])
        return e + self.a_seg(a[..., 2]) if self.a_seg is not None else e

    def forward(self, x, addr):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))[None] + self.addr_emb(addr)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))


def _block_cached(blk, x, kc, vc, causal):
    B, L, D = x.shape
    h = blk.ln1(x)
    q, k, v = blk.qkv(h).view(B, L, 3, blk.n_heads, D // blk.n_heads).permute(2, 0, 3, 1, 4)
    if kc is not None:
        k = torch.cat([kc, k], dim=2)
        v = torch.cat([vc, v], dim=2)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    x = x + blk.proj(a.transpose(1, 2).reshape(B, L, D))
    return x + blk.mlp(blk.ln2(x)), k, v


def stop_tables(unit):
    """per unit id, the cut_function rule (a line after the def line starting with a non-blank ends the function):
    does the unit contain a newline + non-blank, does it start with a non-blank, does it end with a newline."""
    texts = [unit.decode([i]) for i in range(unit.vocab)]
    inside = [any(t[j - 1] == "\n" and not t[j].isspace() for j in range(1, len(t))) for t in texts]
    starts = [bool(t) and not t[0].isspace() for t in texts]
    return torch.tensor(inside), torch.tensor(starts), torch.tensor([t.endswith("\n") for t in texts])


def stop_step(stop, x, prev, done):
    inside, starts, ends_nl = stop
    return done | inside[x] | (prev & starts[x]), ends_nl[x]


@torch.no_grad()
def sample(net, ids, n, temp, gen=None, stop=None, to_raw=None):
    """KV-cached sampling of n units for a batch of same-length prompts; stops a row's growth at the context end and
    the whole batch once every row has ended its function (prompts end inside the def line)."""
    kc, vc = [None] * len(net.blocks), [None] * len(net.blocks)
    if stop is not None:
        stop = [t.to(ids.device) for t in stop]
        prev = done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
    n = min(n, net.context - ids.shape[1] + 1)               # last sampled unit is never fed back
    pos, x, out = 0, ids, []
    tr = None
    if net.addr_mode in ("addrrk", "addrctl"):
        g = torch.Generator(device=ids.device).manual_seed(int(ids.sum().item()) % 2 ** 31)   # eval-only rng
        rn = torch.randint(0, SEGS, (ids.shape[0],), device=ids.device, generator=g)
        rb = rn if net.addr_mode == "addrrk" else torch.randint(0, SEGS, (ids.shape[0],), device=ids.device, generator=g)
        tr = AddrTracker(ids.shape[0], ids.device, rn, rb, rel=True)
    elif net.addr_mode:
        tr = AddrTracker(ids.shape[0], ids.device)
    for _ in range(n):
        L = x.shape[1]
        h = net.tok(x) + net.pos(torch.arange(pos, pos + L, device=x.device))[None]
        if tr is not None:
            h = h + net.addr_emb(torch.stack([tr.step(to_raw[x[:, j]]) for j in range(L)], 1))
        for i, blk in enumerate(net.blocks):
            h, kc[i], vc[i] = _block_cached(blk, h, kc[i], vc[i], L > 1)
        pos += L
        logits = net.head(net.lnf(h))[:, -1].float()
        x = logits.argmax(-1, keepdim=True) if temp == 0 else \
            torch.multinomial(torch.softmax(logits / temp, -1), 1, generator=gen)
        out.append(x)
        if stop is not None:
            done, prev = stop_step(stop, x[:, 0], prev, done)
            if len(out) % 8 == 0 and bool(done.all()):
                break
    return torch.cat(out, 1)


def evaluate(net, unit, split, shape, n=200, k=4, temp=0.8):
    tasks = split[:]
    random.Random(1).shuffle(tasks)
    tasks = tasks[:n]
    prompts = [prompt_ids(unit, u, f) for u, f in tasks]
    srcs = [[None] * k for _ in tasks]
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    t0 = time.perf_counter()
    rng = torch.Generator(device="cuda").manual_seed(1)       # eval never touches the training RNG
    stop = stop_tables(unit)
    lens = []
    net.eval()
    for idx in by_len.values():
        for c in range(0, len(idx), 256 // k):
            part = idx[c:c + 256 // k]
            ids = torch.tensor([prompts[i] for i in part for _ in range(k)], device="cuda")
            gen = sample(net, ids, shape["gen"], temp, rng, stop, raw_table(unit)).cpu().tolist()
            for j, i in enumerate(part):
                for r in range(k):
                    text = unit.decode(prompts[i] + gen[j * k + r])
                    srcs[i][r] = cut_function(text[len(PREFIX):] if text.startswith(PREFIX) else text)[0]
                    lens.append(unit.count(srcs[i][r]))
    wall = time.perf_counter() - t0
    res = summary(grade(tasks, srcs))
    # units_per_function = length of the cut function in this unit (decode work a sampler that stops needs)
    res.update({"decode_wall_s": round(wall, 1), "units_per_function": round(sum(lens) / len(lens), 1),
                "example": srcs[0][0]})
    return res


def corpus(unit):
    text = open(f"{synth2.D}/train.txt").read()
    return torch.tensor(unit.encode(PREFIX + text), dtype=torch.long)


def raw_table(unit):
    """compact id -> raw byte (bytes-family units only; None for token units, which have no address)."""
    return torch.tensor(unit.to_q, device="cuda") if isinstance(unit, Bytes) else None


def rotations(mode, n, seed=0):
    r = random.Random(1000 + seed)
    out = []
    for _ in range(n):
        a = r.randrange(SEGS)
        out.append((a, a if mode == "addrrk" else r.randrange(SEGS)))
    return out


def corpus_addr(mode="addrseg", seed=0):
    raw = (PREFIX + open(f"{synth2.D}/train.txt").read()).encode()
    if mode in ("addrrk", "addrctl"):
        return torch.tensor(addresses(raw, rotations(mode, raw.count(b"def "), seed), rel=True), dtype=torch.long)
    return torch.tensor(addresses(raw), dtype=torch.long)


def len4_split(split, n=400):
    """4 distinct seen units + a final, names outside every split (Random(4), fixed)."""
    held = set(synth2.HELD)
    pool = sorted(u for u in synth2.UNITS if u not in held)
    names = {name_of(u, f) for sp in split.values() for u, f in sp}
    r, out = random.Random(4), []
    while len(out) < n:
        u, f = r.sample(pool, 4), r.choice(sorted(synth2.FINAL))
        if name_of(u, f) not in names:
            names.add(name_of(u, f))
            out.append((u, f))
    return out


def make_unit(name, split):
    texts = [PREFIX + open(f"{synth2.D}/train.txt").read()]   # vocabulary from training text only
    return {"bytes": Bytes, "addr2d": Bytes, "addrseg": Bytes, "addrrk": Bytes, "addrctl": Bytes, "qwen": Qwen,
            "learned": Learned}[name](texts)


def train(a, unit, split):
    shape = SHAPE[a.unit]
    data = corpus(unit).cuda()
    torch.manual_seed(a.seed)
    addr = a.unit.startswith("addr")
    net = (AddrGPT(unit.vocab, 256, 4, 4, shape["ctx"], a.unit) if addr else UnitGPT(unit.vocab, 256, 4, 4, shape["ctx"])).cuda()
    if a.quant != "none":
        quantize_blocks(net, a.quant)
    adata = corpus_addr(a.unit, a.seed).cuda() if addr else None
    n_nonemb = sum(p.numel() for nm, p in net.named_parameters()
                   if not nm.startswith(("tok.", "head.", "pos.", "a_row.", "a_col.", "a_seg.")))
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05)
    warm = 300

    def lr_at(s):                                            # warmup then cosine decay to 10% (as train.py)
        if s < warm:
            return 3e-4 * (s + 1) / warm
        return 3e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    marks = sorted({max(1, round(m * a.steps)) for m in MILESTONES})
    os.makedirs(a.out_dir, exist_ok=True)
    info = {"unit": a.unit, "vocab": unit.vocab, "corpus_units": len(data), "nonemb_params": n_nonemb,
            "params": sum(p.numel() for p in net.parameters()), **shape, "steps": a.steps,
            "units_per_step": shape["ctx"] * shape["batch"], "seed": a.seed, "unit_sig": getattr(unit, "sig", None),
            "quant": a.quant}
    print(json.dumps(info), flush=True)
    log = []
    train_s = 0.0
    for s in range(1, a.steps + 1):
        t0 = time.perf_counter()
        net.train()
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(s - 1)
        st = torch.randint(0, len(data) - shape["ctx"] - 1, (shape["batch"],), device="cuda")
        ix = st[:, None] + torch.arange(shape["ctx"] + 1, device="cuda")[None]
        w = data[ix]
        out = net(w[:, :-1], adata[ix[:, :-1]]) if addr else net(w[:, :-1])
        loss = F.cross_entropy(out.reshape(-1, unit.vocab), w[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        train_s += time.perf_counter() - t0
        if s in marks:
            row = {"step": s, "units_seen": s * info["units_per_step"],
                   "flops": 6 * n_nonemb * s * info["units_per_step"],
                   # + attention scores/values: 3 (fwd+bwd) x 2 x layers x ctx x dim per unit (bytes ctx 512 > 192)
                   "flops_with_attn": (6 * n_nonemb + 6 * 4 * shape["ctx"] * 256) * s * info["units_per_step"],
                   "loss": round(loss.item(), 4),
                   "train_wall_s": round(train_s, 1),
                   "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
            for sp in SPLITS:
                row[sp] = evaluate(net, unit, split[sp], shape, a.n, a.k)
                row[sp]["greedy_pass@1"] = evaluate(net, unit, split[sp], shape, a.n, 1, temp=0)["pass@1"]
            log.append(row)
            print(json.dumps({k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if x != "example"})
                              for k, v in row.items()}), flush=True)
            tag = f"{a.unit}_s{a.seed}" + ("" if a.quant == "none" else f"_{a.quant}")
            json.dump({"info": info, "log": log}, open(f"{a.out_dir}/units_{tag}.json", "w"), indent=1)
            torch.save({"model": net.state_dict(), "info": info, "step": s}, f"{a.out_dir}/ckpt_{tag}.pt")


def smoke(a, unit, split):
    """round trip; oracle sources through the same decode/cut/grade path = 1.0; prompts never leak the body."""
    text = open(f"{synth2.D}/train.txt").read()
    rt = unit.decode(unit.encode(text)) == text
    tasks = [t for sp in SPLITS for t in random.Random(2).sample(split[sp], 30)]
    srcs, leak = [], 0
    for u, f in tasks:
        p = prompt_ids(unit, u, f)
        full = unit.encode(PREFIX + source(u, f))
        leak += unit.decode(p) not in (f"{PREFIX}def {name_of(u, f)}(", f"{PREFIX}def {name_of(u, f)}")
        srcs.append([cut_function(unit.decode(p + full[len(p):])[len(PREFIX):])[0]])
    ok = sum(s[0] == 1 for s in grade(tasks, srcs))
    stop, bad = stop_tables(unit), 0
    for u, f in tasks:
        p = prompt_ids(unit, u, f)
        rest = unit.encode(PREFIX + source(u, f) + "\n" + source(*tasks[0]))[len(p):]   # next function follows
        prev = done = torch.zeros(1, dtype=torch.bool)
        for m, x in enumerate(rest):
            done, prev = stop_step(stop, torch.tensor([x]), prev, done)
            if done[0]:
                break
        cut = lambda ids: cut_function(unit.decode(p + ids)[len(PREFIX):])  # noqa: E731
        bad += not (done[0] and cut(rest[:m + 1]) == (source(u, f), True) and not cut(rest[:m])[1])
    data = corpus(unit)
    addr_mismatch = None
    if a.unit.startswith("addr"):                            # sampler tracker == corpus rule on prompt + reference
        assert len(corpus_addr(a.unit)) == len(data)
        texts = [(PREFIX + source(u, f) + "\n" + source(*tasks[0])).encode() for u, f in tasks]
        rk = a.unit in ("addrrk", "addrctl")
        addr_mismatch = 0
        for i, t in enumerate(texts):
            rn, rb = (i % SEGS, (i * 3 + (a.unit == "addrctl")) % SEGS if a.unit == "addrctl" else i % SEGS)
            if rk:   # one rotation per row in the sampler: both functions of the text share it
                tr = AddrTracker(1, "cpu", torch.tensor([rn]), torch.tensor([rb]), rel=True)
                ref = addresses(t, [(rn, rb)] * 2, rel=True)
            else:
                tr, ref = AddrTracker(1, "cpu"), addresses(t)
            got = [tuple(tr.step(torch.tensor([c]))[0].tolist()) for c in t]
            addr_mismatch += got != ref
        ex = PREFIX + source(*tasks[0])
        ea = addresses(ex.encode(), [(5, 5)], rel=True) if rk else addresses(ex.encode())
        print("addr example:", list(zip(ex, ea))[len(PREFIX):len(PREFIX) + 400:7], flush=True)
    print(json.dumps({"unit": a.unit, "addr_mismatch": addr_mismatch, "vocab": unit.vocab, "round_trip": rt, "oracle_1.0": f"{ok}/{len(tasks)}",
                      "prompts_not_name_only": leak, "stop_rule_mismatch": bad, "corpus_units": len(data),
                      "bytes_per_unit": round(len(text.encode()) / len(data), 3)}), flush=True)
    a.steps, a.n, a.k = 30, 8, 2
    a.out_dir = os.path.join(a.out_dir or "/media/llm-data/universe-runs/bytebrain-pycode-20261022/synth2", "smoke")
    train(a, unit, split)


def eval_ckpt(a, unit, split):
    """gate 5 positive control: reload a saved checkpoint and re-score the judged splits. evaluate() draws from its own
    generator (seed 1) and its own task shuffle, so the numbers must equal the run's last logged row."""
    shape, addr = SHAPE[a.unit], a.unit.startswith("addr")
    ck = torch.load(a.ckpt, map_location="cuda")
    net = (AddrGPT(unit.vocab, 256, 4, 4, shape["ctx"], a.unit) if addr else UnitGPT(unit.vocab, 256, 4, 4, shape["ctx"])).cuda()
    net.load_state_dict(ck["model"])
    out = {"ckpt": a.ckpt, "step": ck["step"]}
    for sp in ("test_a", "test_num"):
        out[sp] = {"pass@1": evaluate(net, unit, split[sp], shape, a.n, a.k)["pass@1"],
                   "greedy_pass@1": evaluate(net, unit, split[sp], shape, a.n, 1, temp=0)["pass@1"]}
    print(json.dumps({"eval_ckpt": out}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "train", "evalckpt"])
    ap.add_argument("--unit", required=True, choices=list(SHAPE))
    ap.add_argument("--out-dir")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-frac", type=float, default=0.5)
    ap.add_argument("--quant", default="none", choices=["none", "bin", "ter", "q2"])
    ap.add_argument("--ckpt", help="evalckpt: checkpoint to re-score")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    split = json.load(open(f"{synth2.D}/splits.json"))
    split["test_len4"] = len4_split(split)
    unit = make_unit(a.unit, split)
    {"smoke": smoke, "train": train, "evalckpt": eval_ckpt}[a.cmd](a, unit, split)


if __name__ == "__main__":
    main()
