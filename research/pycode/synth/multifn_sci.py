"""Extended op table for the dispatch world: multifn's 15 ops + sciops' 5 scientific ops (ledger #5 / #22 / #55, master
queue I5, 10-27). multifn.py is NOT edited (a running GPU queue imports it); everything here is a new table built on
top of it, used only by sci_belt56.py.

Index rules (old checkpoints stay valid):
  OPS  = MF.OPS + [cos, exp, log, sin, tan]                 old 15 at their old positions, new ops appended
  SYM  = MF.SYM + [def:<sci>...] + [disp:<sci>...]          SYM[:len(MF.SYM)] == MF.SYM (asserted)
  CH   = MF.CH + sorted(new characters)                     old characters keep their index; the writer's END and
                                                            START ids move from len(MF.CH) to len(CH) (remapped in
                                                            WriterX.from_mf)
Writer = the same model as multifn.Writer (symbol embedding -> byte GRU -> next byte), with nn.GRU over the whole
teacher-forced sequence instead of a GRUCell loop (same equations, same parameter shapes) so it trains on the GPU;
`from_mf` loads a multifn.Writer state dict (the e2e ckpt or a fresh old-recipe writer) and grows its rows.
Runner = the bwrap runner of e2e_pipeline (same sandbox flags, rlimits, alarms) but floats are returned with full
repr, so the sci ops can be compared with math.* at relative 1e-6 (multifn/e2e round to 6 decimals).
"""

import json
import math
import os
import random
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockaddr as BA  # noqa: E402
import multifn as MF  # noqa: E402
import sciops as SC  # noqa: E402

SCI = sorted(SC.SCI_DEFS)                                   # cos exp log sin tan
DEFS = {**MF.DEFS, **SC.SCI_DEFS}
OPS = list(MF.OPS) + SCI
SYM = list(MF.SYM) + [f"def:{o}" for o in SCI] + [f"disp:{o}" for o in SCI]
assert SYM[:len(MF.SYM)] == MF.SYM and OPS[:len(MF.OPS)] == MF.OPS and not set(MF.OPS) & set(SCI)
CASES = MF.CASES


def plan_of_grouped(spec):
    order, seen = [], set()

    def visit(o):
        if o in seen:
            return
        seen.add(o)
        for d in DEFS[o][1]:
            visit(d)
        order.append(o)
    for o in spec:
        visit(o)
    return [f"def:{o}" for o in order] + [f"disp:{o}" for o in spec] + ["<eos>"]


def plan_of(spec):
    """interleaved, as in the belt (e2e sets MF.PLAN_FORMAT = 'interleaved'): per op its dep closure, def, dispatch."""
    out = []
    for o in spec:
        out += [s for s in plan_of_grouped((o,)) if s != "<eos>"]
    return out + ["<eos>"]


def block_text(sym):
    kind, o = sym.split(":")
    if kind == "def":
        return f"def {o}(a, b=0):\n" + "".join(f"    {l}\n" for l in DEFS[o][0])
    return f"    if op == \"{o}\":\n        return {o}(a, b)\n"


def assemble(syms, text_of=block_text):
    """same layout as multifn.assemble / e2e_pipeline.assemble (dispatch world): defs (deduplicated), then main."""
    body = [s for s in syms if s.startswith(("def:", "disp:"))]
    defs = "".join(text_of(s) for s in dict.fromkeys(s for s in body if s.startswith("def:")))
    disp = "".join(text_of(s) for s in body if s.startswith("disp:"))
    return defs + "def main(op, a, b=0):\n" + disp + "    return None\n"


assert all(block_text(s) == MF.block_text(s) for s in MF.SYM[3:])
OLD_CH = list(MF.CH)
CH = OLD_CH + sorted(set("".join(block_text(s) for s in SYM[3:])) - set(OLD_CH))
MAX_OUT = len(plan_of(tuple(OPS))) + 4
MAX_TEXT = max(len(block_text(s)) for s in SYM[3:])

# ----------------------------------------------------------------------------------------------- data + planner


class Data:
    """multifn.Data over the 20-op table. exclude: sorted op sets that may not be trained (the belt's sci requests)."""

    def __init__(self, seed=0, exclude=(), n_train=2400):
        r = random.Random(seed)
        ex = {tuple(sorted(s)) for s in exclude}
        sets3 = sorted({tuple(sorted(r.sample(OPS, 3))) for _ in range(6000)})
        r.shuffle(sets3)
        held3 = set(sets3[:150])
        train = set()
        while len(train) < n_train:
            k = r.choice([1, 2, 2, 3, 3, 3])
            s = tuple(r.sample(OPS, k))
            if tuple(sorted(s)) not in held3 and tuple(sorted(s)) not in ex:
                train.add(s)
        self.sets = {"train": sorted(train), "test_comb": [tuple(r.sample(s, 3)) for s in sorted(held3)],
                     "full": [tuple(OPS)], "sci5": [tuple(SCI)]}
        for k in range(4, 7):
            self.sets[f"len{k}"] = [tuple(r.sample(OPS, k)) for _ in range(150)]
        self.sets["sci_len4"] = [tuple(r.sample(SCI, 2) + r.sample(MF.OPS, 2)) for _ in range(150)]


def use_planner_globals():
    BA.SYM, BA.START, BA.MAX_OUT = SYM, len(SYM), MAX_OUT


def name_bytes(specs):
    return MF.name_bytes(specs)                             # "calc_" + "_".join(ops) -> blockaddr byte ids


def targets(specs):
    ys = [[SYM.index(s) for s in plan_of(sp)] for sp in specs]
    T = max(map(len, ys))
    y = torch.tensor([v + [0] * (T - len(v)) for v in ys])
    return torch.cat([torch.full((len(ys), 1), len(SYM)), y[:, :-1]], 1), y


def train_planner(data, seed, steps, device="cpu"):
    """multifn.train_planner recipe (AdamW 3e-3, wd .01, batch 48, warmup 50 + cosine, clip 1) over SYM.
    blockaddr.Net builds its state tensors with factory calls, so the device is set as the default inside."""
    use_planner_globals()
    with torch.device(device):
        torch.manual_seed(seed)
        net = BA.Net("addr", gate_bias=-4.0)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
        r, tr = random.Random(seed + 100), data.sets["train"]
        for s in range(steps):
            MF.cosine_lr(opt, 3e-3, s, steps)
            sp = [tr[r.randrange(len(tr))] for _ in range(48)]
            y_in, y = targets(sp)
            loss = F.cross_entropy(net(name_bytes(sp), y_in).transpose(1, 2), y, ignore_index=0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net


@torch.no_grad()
def plan_specs(net, specs, device="cpu"):
    use_planner_globals()
    net.eval()
    out = []
    with torch.device(device):
        for i in range(0, len(specs), 256):
            for row in net.greedy(name_bytes(specs[i:i + 256])).tolist():
                s = [SYM[j] for j in row]
                out.append(s[:s.index("<eos>") + 1] if "<eos>" in s else s)
    return out

# ----------------------------------------------------------------------------------------------- writer


class WriterX(nn.Module):
    """multifn.Writer with a fused GRU. syms / ch: the symbol and character tables it was built for."""

    def __init__(self, d, syms=SYM, ch=CH):
        super().__init__()
        self.syms, self.chs = list(syms), list(ch)
        self.end, self.start = len(self.chs), len(self.chs) + 1
        self.cond = nn.Embedding(len(self.syms), d)
        self.ch = nn.Embedding(len(self.chs) + 2, d)
        self.rnn = nn.GRU(2 * d, d, batch_first=True)
        self.out = nn.Linear(d, len(self.chs) + 1)

    def forward(self, c, y_in):
        x = self.ch(y_in)
        x = torch.cat([self.cond(c)[:, None].expand(-1, x.shape[1], -1), x], 2)
        return self.out(self.rnn(x)[0])

    @torch.no_grad()
    def write_ids(self, c, max_len):
        cond, h = self.cond(c)[:, None], None
        prev, out = torch.full((c.shape[0],), self.start, device=c.device), []
        for _ in range(max_len):
            y, h = self.rnn(torch.cat([cond, self.ch(prev)[:, None]], 2), h)
            prev = self.out(y[:, 0]).argmax(-1)
            out.append(prev)
        return torch.stack(out, 1).tolist()

    def texts(self, syms, max_len=MAX_TEXT + 32):
        syms = sorted({s for s in syms if s in self.syms[3:]})
        if not syms:
            return {}
        dev = self.out.weight.device
        res = {}
        for i in range(0, len(syms), 256):
            rows = self.write_ids(torch.tensor([self.syms.index(s) for s in syms[i:i + 256]], device=dev), max_len)
            for s, row in zip(syms[i:i + 256], rows):
                row = row[:row.index(self.end)] if self.end in row else row
                res[s] = "".join(self.chs[k] for k in row)
        return res

    @classmethod
    def from_mf(cls, sd, d, syms=SYM, ch=CH, seed=0):
        """grow a multifn.Writer state dict (old SYM / CH) into a WriterX over syms / ch.
        Copied: GRU weights (cell.* -> rnn.*_l0), cond rows of the old symbols (same index: SYM extends MF.SYM),
        char rows of the old characters (same index), the old END row -> new END, old START row -> new START.
        New rows (new symbols, new characters) keep the default PyTorch init drawn under torch.manual_seed(seed)
        (Embedding N(0, 1); Linear U(+-1/sqrt(d)) for weight and bias), i.e. exactly what a fresh WriterX would
        get for those rows."""
        torch.manual_seed(seed)
        net = cls(d, syms, ch)
        old_end, old_start = len(MF.CH), len(MF.CH) + 1
        assert sd["ch.weight"].shape[0] == old_start + 1 and sd["cond.weight"].shape[0] == len(MF.SYM)
        assert list(ch[:len(MF.CH)]) == list(MF.CH) and list(syms[:len(MF.SYM)]) == list(MF.SYM)
        new = {k: v.clone() for k, v in net.state_dict().items()}
        for a, b in (("weight_ih", "weight_ih_l0"), ("weight_hh", "weight_hh_l0"), ("bias_ih", "bias_ih_l0"),
                     ("bias_hh", "bias_hh_l0")):
            new[f"rnn.{b}"] = sd[f"cell.{a}"].clone()
        new["cond.weight"][:len(MF.SYM)] = sd["cond.weight"]
        new["ch.weight"][:len(MF.CH)] = sd["ch.weight"][:len(MF.CH)]
        new["ch.weight"][net.end] = sd["ch.weight"][old_end]
        new["ch.weight"][net.start] = sd["ch.weight"][old_start]
        new["out.weight"][:len(MF.CH)] = sd["out.weight"][:len(MF.CH)]
        new["out.bias"][:len(MF.CH)] = sd["out.bias"][:len(MF.CH)]
        new["out.weight"][net.end] = sd["out.weight"][old_end]
        new["out.bias"][net.end] = sd["out.bias"][old_end]
        net.load_state_dict(new)
        return net

    def to_mf_state(self):
        """WriterX over the old tables -> a multifn.Writer state dict (for parity checks)."""
        assert self.syms == list(MF.SYM) and self.chs == list(MF.CH)
        sd = self.state_dict()
        out = {k: v for k, v in sd.items() if not k.startswith("rnn.")}
        for a, b in (("weight_ih", "weight_ih_l0"), ("weight_hh", "weight_hh_l0"), ("bias_ih", "bias_ih_l0"),
                     ("bias_hh", "bias_hh_l0")):
            out[f"cell.{a}"] = sd[f"rnn.{b}"]
        return out


def train_writer(net, pool, seed, steps=1500, batch=64, lr=3e-3, device="cpu"):
    """multifn.train_writer recipe (AdamW 3e-3, wd .01, batch 64, warmup 50 + cosine, clip 1, symbols drawn uniformly
    from pool with random.Random(seed)). The caller seeds the init (fresh: torch.manual_seed(seed) before building)."""
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    r = random.Random(seed)
    for s in range(steps):
        MF.cosine_lr(opt, lr, s, steps)
        bs = [pool[r.randrange(len(pool))] for _ in range(batch)]
        ys = [[net.chs.index(ch) for ch in block_text(b)] + [net.end] for b in bs]
        T = max(map(len, ys))
        y = torch.tensor([v + [-100] * (T - len(v)) for v in ys], device=device)
        y_in = torch.cat([torch.full((batch, 1), net.start, device=device), y[:, :-1].clamp(min=0)], 1)
        c = torch.tensor([net.syms.index(b) for b in bs], device=device)
        loss = F.cross_entropy(net(c, y_in).transpose(1, 2), y, ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    return float(loss.detach())

# ----------------------------------------------------------------------------------------------- exec + truth


RUNNER = r"""
import io, json, resource, signal, sys
for _lim, _v in ((resource.RLIMIT_AS, 1 << 30), (resource.RLIMIT_NPROC, 0), (resource.RLIMIT_FSIZE, 1 << 24)):
    resource.setrlimit(_lim, (_v, _v))
def _alarm(*_):
    raise TimeoutError
signal.signal(signal.SIGALRM, _alarm)
def _out(v):
    if isinstance(v, bool) or v is None:
        return repr(v)
    if isinstance(v, (int, float)):
        return "n" + repr(float(v)) if isinstance(v, float) or abs(v) < 2 ** 53 else "n" + repr(v)
    return "?" + repr(v)[:40]
items = json.load(sys.stdin)
real, res = sys.stdout, []
sys.stdout = io.StringIO()
for src, calls in items:
    g = {}
    try:
        signal.alarm(2)
        exec(src, g)
        signal.alarm(0)
    except BaseException:
        signal.alarm(0)
        res.append(None)
        continue
    row = []
    for fn, args in calls:
        try:
            signal.alarm(1)
            v = g[fn](*args)
            signal.alarm(0)
            row.append(_out(v))
        except BaseException as e:
            signal.alarm(0)
            row.append("!" + type(e).__name__)
    res.append(row)
    sys.stdout = io.StringIO()
real.write(json.dumps(res))
"""


def run_many(items):
    """items: [(src, [(fn, args), ...])] -> [None | [str per call]]; same sandbox as e2e_pipeline.run_many."""
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--setenv", "PATH", "/usr/bin",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev",
           "/usr/bin/python3", "-I", "-S", "-c", RUNNER]
    n_calls = sum(len(c) for _, c in items)
    try:
        p = subprocess.run(cmd, input=json.dumps(items), capture_output=True, text=True,
                           timeout=60 + 2 * len(items) + n_calls // 50)
        if p.returncode and "bwrap" in p.stderr:
            raise RuntimeError("sandbox failed: " + p.stderr[:300])
        out = json.loads(p.stdout)
        if len(out) != len(items):
            raise ValueError
        return out
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        if len(items) == 1:
            return [None]
        h = len(items) // 2
        return run_many(items[:h]) + run_many(items[h:])


def run_all(items, chunk=200):
    out = []
    for i in range(0, len(items), chunk):
        out += run_many(items[i:i + chunk])
    return out


def _sci_inputs():
    """held-out probe inputs for the sci ops: drawn here with seed 5600 (sciops.check used Random(0)); no model ever
    sees an input-output pair (the writer learns text, the understander learns specs)."""
    r = random.Random(5600)
    xs = {}
    xs["sin"] = [r.uniform(-40, 40) for _ in range(14)] + [0, 3]
    xs["cos"] = [r.uniform(-40, 40) for _ in range(14)] + [0, -2]
    tan = []
    while len(tan) < 14:
        x = r.uniform(-40, 40)
        if abs(math.cos(x)) > 0.05:
            tan.append(x)
    xs["tan"] = tan + [0, 1]
    xs["exp"] = [r.uniform(-20, 20) for _ in range(14)] + [0, 5]
    xs["log"] = [10 ** r.uniform(-6, 6) for _ in range(12)] + [1, 7, 0, -2.5]    # 0 and -2.5 -> None
    return xs


SCI_X = _sci_inputs()


def sci_truth(o, x):
    return SC.REF[o](x)


REL, ABS = 1e-6, 1e-9                                       # |got - want| <= 1e-6 |want| + 1e-9


def parse_out(s):
    """runner string -> ('none',) | ('exc', name) | ('num', float) | ('bool', b) | ('other', s)"""
    if s == "None":
        return ("none",)
    if s.startswith("!"):
        return ("exc", s[1:])
    if s.startswith("n"):
        return ("num", float(s[1:]))
    if s in ("True", "False"):
        return ("bool", s == "True")
    return ("other", s)


def same(got, want):
    """got, want: parsed outputs. Numbers: relative 1e-6 (+1e-9 absolute floor near 0); others: equal."""
    if want[0] == "num" and got[0] == "num":
        g, w = got[1], want[1]
        if math.isnan(w) or math.isnan(g) or math.isinf(w) or math.isinf(g):
            return g == w
        return abs(g - w) <= REL * abs(w) + ABS
    return got == want


def want_of(v):
    return ("none",) if v is None else ("num", float(v))


def op_calls(o):
    """calls main(o, ...) for one requested op: sci ops on held-out floats, old ops on multifn.CASES."""
    if o in SCI:
        return [("main", [o, x]) for x in SCI_X[o]]
    return [("main", [o, a, b]) for a, b in CASES]


OTHER = OPS + ["foo", "sqrt2"]                              # ids a strict program must answer with None


def none_calls(spec):
    return [("main", [o, 1.5, 2]) for o in OTHER if o not in spec]


_OLD_REF = {}


def old_ref():
    """gold outputs for the 15 old ops on CASES: the reference module of multifn (the old ops' definition)."""
    if not _OLD_REF:
        src = assemble(plan_of(tuple(MF.OPS)))
        row = run_many([(src, [c for o in MF.OPS for c in op_calls(o)])])[0]
        assert row is not None
        for i, o in enumerate(MF.OPS):
            _OLD_REF[o] = [parse_out(s) for s in row[i * len(CASES):(i + 1) * len(CASES)]]
    return _OLD_REF


def wants(o):
    if o in SCI:
        return [want_of(sci_truth(o, x)) for x in SCI_X[o]]   # math.*, never a gold program
    return old_ref()[o]


def program_items(src, spec):
    """the module, the calls testing every requested op, then the strict calls (other ids -> None)."""
    calls = [c for o in spec for c in op_calls(o)] + none_calls(spec)
    return (src, calls)


def grade(row, spec):
    """row from run_many for program_items(src, spec) -> dict(loads, exec, op_ok {op: bool}, strict_ok, pass, ...)."""
    if row is None:
        return {"loads": False, "exec": False, "op_ok": {o: False for o in spec}, "strict_ok": False,
                "pass_lenient": False, "pass": False}
    pos, op_ok, exc = 0, {}, False
    for o in spec:
        w = wants(o)
        got = [parse_out(s) for s in row[pos:pos + len(w)]]
        pos += len(w)
        exc |= any(g[0] == "exc" and x[0] != "exc" for g, x in zip(got, w))
        op_ok[o] = all(same(g, x) for g, x in zip(got, w))
    rest = [parse_out(s) for s in row[pos:]]
    strict = all(g == ("none",) for g in rest)
    exc |= any(g[0] == "exc" for g in rest)
    lenient = all(op_ok.values())
    return {"loads": True, "exec": not exc, "op_ok": op_ok, "strict_ok": strict, "pass_lenient": lenient,
            "pass": lenient and strict}
