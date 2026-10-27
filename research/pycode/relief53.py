"""Relief + water (ledger #53): frequency relief of code bytes/junctions, pour water, pools = holes.

Written 2026-10-27 10:02 (from `date`), BEFORE any run.

His words (#53): "a gente fez um projeto utilizando navier stokes ou algo do tipo, como se a gente usasse as
frequencias para criar relevos e agua para preencher, assim achava o melhor caminho ... despejar os bytes ...
veja a frequencia como falei os bytes ou juncoes, assim poderiamos achar gaps e furo no meio do codigo"

Mechanism taken from his earlier projects:
  - IA-engine (META/PROJECTS.md sec 9): "Navier-Stokes simplificado (Laplacian diffusion)" -> arm SHALLOW:
    rain on the relief, water moves down the free-surface gradient (diffusion-wave shallow water, no inertia).
  - iara-agua inpaint (JOURNAL 07-21): holes found by flood-fill per component, filled by diffusion -> arm FLOOD:
    priority-flood depression filling = the steady state of "pour water everywhere, let it settle, edges drain".
  - Resonance forest / BYTE-MATRIX sec D: multi-frequency pyramid -> relief = several context scales (1,2,4,7
    bytes) plus token and line junctions; address = (row, column) grid.

RELIEF. Statistics from clean_train.txt (52 MB, ast.unparse functions):
  byte channel : interpolated n-gram, contexts k = 1, 2, 4, 7 bytes (exact keys, no hashing), Dirichlet backoff
                 beta = 2 down to add-one unigram; channel = mean over k of -log2 P_k(byte | k previous bytes).
  token channel: regex tokens (frequent names kept, rare names -> NAME, numbers NUM, strings STR); -log2 P(token |
                 previous token in the line, <BOL> at line start), backoff to unigram. Put on the token's cells.
  line channel : junction between lines: event (indent delta in 4-col units, first token) given the previous
                 line's last token; end-of-function event (last token -> END). Put on cells 0..indent of the row
                 (end event on the last row's newline cell).
  Each channel is z-scored with mean/std from CALIBRATION functions (holdout functions that fail the eval filter,
  disjoint from eval and from the counts). S = z_byte + z_tok + z_line (missing channel = 0).
  height h = -S on the cells of each line (bytes + its newline cell); everything else is a flat plain h = 0;
  the canvas border is the outlet. Expected code = high ground, surprising code = depressions.
WATER (torch, batched; runs on CPU now, GPU later):
  FLOOD   : W = max(h, min_nbr W) iterated to the fixed point, border W = h. depth = W - h.
  SHALLOW : uniform rain 0.5 (z units) on every cell, T = 400 steps, flux to each of 4 neighbours =
            0.2 * relu(eta_i - eta_j) (eta = h + w), limited by the water present, border drains.
            depth = w at the end; flow = accumulated outflow (the "path" map, panel only).
  Also both water arms on the byte-LM relief (h = -z(LM surprisal)) to see whether water adds on a better relief.
LINE SCORE (pre-registered primary = max over the line's cells; secondary = sum):
  relief alone = max(-h); flood = max depth; shallow = max depth; lm = max z(LM surprisal). Random tie-break.
DEFECTS on held-out functions (clean_holdout.txt, filter 4-40 rows, <= 100 bytes/row, no tab, <= 1200 bytes,
  parses, text not in train). Body lines only (rows after the def header). Targets (rows of the defected text):
  delete_line {i-1, i}; delete_return ('return X' -> 'X') {i}; swap adjacent {i, i+1}; wrong_indent (+-4 cols)
  {i}; foreign_token (a token from another eval function inserted at a token start) {i}; rename (refactor that
  renames a local variable everywhere except one stale load) {stale row}; junk = POSITIVE CONTROL ('$?`'
  inserted) {i}.
METRICS: top-1 / top-3 line hit; AUROC defected vs the same functions clean (function score = max line score);
  seeds 0,1,2 (function sample + defect positions; the 1378 eligible functions are shared across seeds, so CIs are
  cluster-bootstrap over function id, 1000 resamples). Water vs relief: paired cluster-bootstrap difference.

PRE-RUN GATES (bytebrain/CLAUDE.md):
  1 baselines: random line, longest line, byte-LM surprisal (clean/ckpt_best.pt, 26M, val 1.118), parse error line
    (random ranking when the text parses; coverage reported). A number that does not beat them is not a result.
  2 cue audit: defects are programmatic. foreign_token / junk lengthen the line -> longest-line baseline shows it.
    delete_line shortens the function -> AUROC partly measures length; declared.
  3 gold enters only in scoring (targets); relief/water never see the clean twin.
  4 disjointness: counts = clean_train; eval = clean_holdout (split by file hash) minus texts equal to a train
    function; calibration = other holdout functions.
  5 positive controls: junk defect (relief must find it); LM bits/byte on eval clean must be near its val 1.118
    bits (accept 0.8-1.5). [10:08 correction after the 60-function smoke, before the full run: train.py val is
    bits/byte (divides by ln 2); the header first said nats. Smoke: eval 0.919 bits, calibration 1.055 bits.]
  Added 10:08 before the full run (secondary arm, his "despejar os bytes" read literally): BYTEPOUR = SHALLOW with
    rain only on the byte cells (each byte pours 0.5), none on the plain.
  7 arithmetic: random top-1 = mean |T|/rows ~ 0.10-0.20; the predictions below sit between that and the junk
    ceiling.
  8 equal budget: all arms see the same functions/defects; water params fixed before the run (no tuning).

ERRATUM 10:52 (after run 1, before run 2): run 1 put every function at the top-left of a fixed 42-row canvas, so
  the top rows touch the outlet and the bottom rows have up to 40 rows of plain below them. Water drained from
  the top and stayed at the bottom: shallow top-1 landed on the last row 3-6 pp more often than relief on every
  defect type (position_check.json). Fix: every cell below the function's last row is outlet too (symmetric
  top/bottom); the right side keeps a plain strip for all functions (grid always CMAX+1 wide). Run 1 kept as
  run1_*. LM surprisal now cached (lm_cache.npz). Predictions unchanged.

PREDICTIONS (pooled top-1, primary max-score; written before running):
  P0 junk (positive control): relief >= 0.85, lm >= 0.85.
  P1 foreign_token: relief 0.45 (0.30-0.60); lm 0.60.
  P2 wrong_indent: relief 0.40 (0.25-0.60) (line channel); lm 0.50; ~60% unparsable, parse line top-1 >= 0.8 there.
  P3 swap: relief 0.30 (random ~0.2); lm 0.30.
  P4 delete_line: relief 0.25 (random ~0.2); lm 0.30.
  P5 delete_return: relief 0.20; lm 0.30.
  P6 rename: relief within 5 pp of random; lm within 5 pp of random (needs cross-line name consistency).
  P7 WATER vs RELIEF ALONE: the paired 95% CI of (water - relief) top-1 includes 0 or is negative on >= 4 of the 6
     real defect types for BOTH water arms (point prediction: water does not add; p ~ 0.65). Water most likely to
     help on delete_line (a gap straddles two rows) by 0 to +5 pp.
  P8 lm > relief by >= 5 pp top-1 on all six real types (p ~ 0.7).
  P9 AUROC: relief foreign_token 0.75, wrong_indent 0.70, others 0.50-0.60; parse indicator wrong_indent ~0.8.

Run (CPU only now; GPU busy):
  cd /home/leonardo/projects/LLM/bytebrain/research/pycode && \
  HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  nice -n 5 timeout 7200 /home/leonardo/projects/LLM/.venv-rocm/bin/python -u relief53.py --device cpu
Full GPU run (when the GPU is free; one GPU process at a time):
  HSA_OVERRIDE_GFX_VERSION=10.3.0 OMP_NUM_THREADS=2 nice -n 5 setsid timeout 7200 \
  /home/leonardo/projects/LLM/.venv-rocm/bin/python -u relief53.py --device cuda --seeds 0 1 2 3 4 \
  --per-type 600 --out /media/llm-data/universe-runs/bytebrain-pycode-20261022/relief53_gpu
"""
import argparse
import ast
import hashlib
import json
import keyword
import math
import os
import re
import sys
import time

import numpy as np
import torch

TRAIN = "/media/llm-data/datasets/pycode/clean_train.txt"
HOLD = "/media/llm-data/datasets/pycode/clean_holdout.txt"
CKPT = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/clean/ckpt_best.pt"
OUT = "/media/llm-data/universe-runs/bytebrain-pycode-20261022/relief53"
BYTEBRAIN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ORDERS = (1, 2, 4, 7)
BETA = 2.0
TOPNAMES = 3000
RMAX, CMAX = 40, 100          # eval filter; grid has CMAX+1 columns (newline cell)
MAXBYTES = 1200
RAIN, STEPS, KAPPA = 0.5, 400, 0.2
DEFECTS = ("delete_line", "delete_return", "swap", "wrong_indent", "foreign_token", "rename", "junk")
LN2 = math.log(2.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ data
def split_fns(text):
    return [f.rstrip("\n") for f in re.split(r"\n\n(?=@|def |async def )", text) if f.strip()]


def h16(s):
    return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def eligible(f):
    rows = f.split("\n")
    if not (4 <= len(rows) <= RMAX) or "\t" in f:
        return False
    if max(len(r.encode()) for r in rows) > CMAX or len(f.encode()) > MAXBYTES:
        return False
    if not any(r.startswith(("def ", "async def ")) for r in rows):
        return False
    try:
        ast.parse(f)
    except Exception:
        return False
    return True


# ------------------------------------------------------------------ byte n-gram relief
def window_keys(d, w):
    n = len(d) - w + 1
    key = np.zeros(n, dtype=np.uint64)
    for j in range(w):
        key = (key << np.uint64(8)) | d[j:j + n].astype(np.uint64)
    return key


class ByteNgram:
    def __init__(self, data, cache):
        if os.path.exists(cache):
            z = np.load(cache)
            self.p0 = z["p0"]
            self.tabs = {k: (z[f"uk{k}"], z[f"ck{k}"], z[f"uc{k}"], z[f"cc{k}"]) for k in ORDERS}
            return
        uni = np.bincount(data, minlength=256).astype(np.float64) + 1.0
        self.p0 = uni / uni.sum()
        self.tabs = {}
        for k in ORDERS:
            key = window_keys(data, k + 1)
            uk, ck = np.unique(key, return_counts=True)
            uc, cc = np.unique(key >> np.uint64(8), return_counts=True)
            self.tabs[k] = (uk, ck, uc, cc)
            del key
            log(f"byte ngram k={k}: {len(uk)} keys, {len(uc)} contexts")
        np.savez(cache, p0=self.p0, **{f"{n}{k}": a for k in ORDERS for n, a in zip(("uk", "ck", "uc", "cc"),
                                                                                       self.tabs[k])})

    @staticmethod
    def _lookup(sorted_keys, counts, q):
        i = np.searchsorted(sorted_keys, q)
        i = np.minimum(i, len(sorted_keys) - 1)
        hit = sorted_keys[i] == q
        return np.where(hit, counts[i], 0).astype(np.float64)

    def surprisal(self, seq):
        """seq: bytes with left padding of 7 bytes already included; returns (len(seq)-7, len(ORDERS)) bits."""
        d = np.frombuffer(seq, dtype=np.uint8)
        pad = max(ORDERS)
        tgt = d[pad:]
        p = self.p0[tgt]
        out = []
        for k in ORDERS:
            key = window_keys(d[pad - k:], k + 1)
            uk, ck, uc, cc = self.tabs[k]
            c = self._lookup(uk, ck, key)
            cx = self._lookup(uc, cc, key >> np.uint64(8))
            p = (c + BETA * p) / (cx + BETA)
            out.append(-np.log2(p))
        return np.stack(out, 1)


# ------------------------------------------------------------------ token / line junctions
TOKRE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d[\w.]*|'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|"
                   r"==|!=|<=|>=|\*\*|//|->|\+=|-=|\*=|/=|:=|<<|>>|[^\s]")


def tokens(line):
    """list of (start_byte, end_byte, raw str); line is str (ascii mostly; byte offsets computed)."""
    out = []
    for m in TOKRE.finditer(line):
        s = len(line[:m.start()].encode())
        e = s + len(m.group().encode())
        out.append((s, e, m.group()))
    return out


class Junctions:
    def __init__(self, train_fns, cache):
        if os.path.exists(cache):
            z = json.load(open(cache))
            self.__dict__.update(z)
            return
        from collections import Counter
        names = Counter()
        for f in train_fns:
            names.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", f))
        self.vocab = [w for w, _ in names.most_common(TOPNAMES)]
        V = set(self.vocab)
        uni, pair, prevc = Counter(), Counter(), Counter()
        lev, levc, lev_uni = Counter(), Counter(), Counter()
        for f in train_fns:
            rows = f.split("\n")
            prev_last, prev_ind = None, None
            for r in rows:
                if not r.strip():
                    continue
                toks = [self._cls(t[2], V) for t in tokens(r)]
                p = "<BOL>"
                for t in toks:
                    uni[t] += 1
                    pair[p + "\x00" + t] += 1
                    prevc[p] += 1
                    p = t
                ind = len(r) - len(r.lstrip(" "))
                if prev_last is not None:
                    ev = self._delta(ind - prev_ind) + "\x00" + toks[0]
                    lev[prev_last + "\x01" + ev] += 1
                    levc[prev_last] += 1
                    lev_uni[ev] += 1
                prev_last, prev_ind = toks[-1], ind
            if prev_last is not None:
                ev = "END\x00END"
                lev[prev_last + "\x01" + ev] += 1
                levc[prev_last] += 1
                lev_uni[ev] += 1
        self.uni, self.pair, self.prevc = dict(uni), dict(pair), dict(prevc)
        self.lev, self.levc, self.lev_uni = dict(lev), dict(levc), dict(lev_uni)
        json.dump(self.__dict__, open(cache, "w"))
        log(f"junctions: {len(self.uni)} token types, {len(self.pair)} pairs, {len(self.lev)} line events")

    @staticmethod
    def _cls(t, V):
        if t[0].isalpha() or t[0] == "_":
            return t if t in V else "NAME"
        if t[0].isdigit():
            return "NUM"
        if t[0] in "'\"":
            return "STR"
        return t

    @staticmethod
    def _delta(d):
        return f"D{max(-4, min(2, d // 4))}" if d % 4 == 0 else "Dodd"

    def setup(self):
        self.V = set(self.vocab)
        self.N = sum(self.uni.values())
        self.NV = len(self.uni) + 1
        self.LN = sum(self.lev_uni.values())
        self.LV = len(self.lev_uni) + 1

    def p_tok(self, prev, t):
        pu = (self.uni.get(t, 0) + 1.0) / (self.N + self.NV)
        return (self.pair.get(prev + "\x00" + t, 0) + BETA * pu) / (self.prevc.get(prev, 0) + BETA)

    def p_line(self, prev_last, ev):
        pu = (self.lev_uni.get(ev, 0) + 1.0) / (self.LN + self.LV)
        return (self.lev.get(prev_last + "\x01" + ev, 0) + BETA * pu) / (self.levc.get(prev_last, 0) + BETA)


# ------------------------------------------------------------------ channels on the grid
def channels(text, bng, jn):
    """Returns dict of (R, CMAX+1) float arrays (nan = undefined) + line cell mask."""
    rows = text.split("\n")
    R = len(rows)
    shape = (R, CMAX + 1)
    byte = np.full(shape, np.nan)
    tok = np.full(shape, np.nan)
    line = np.full(shape, np.nan)
    mask = np.zeros(shape, dtype=bool)
    seq = b"\n" * max(ORDERS) + b"\n\n" + text.encode() + b"\n"
    s = bng.surprisal(seq)[2:].mean(1)          # drop the two "\n\n" separator bytes
    pos = 0
    prev_last, prev_ind = None, None
    for r, row in enumerate(rows):
        rb = row.encode()
        L = len(rb)
        byte[r, :L + 1] = s[pos:pos + L + 1]
        mask[r, :L + 1] = True
        pos += L + 1
        if not row.strip():
            continue
        tk = tokens(row)
        cl = [jn._cls(t[2], jn.V) for t in tk]
        p = "<BOL>"
        for (a, b, _), c in zip(tk, cl):
            tok[r, a:b] = -math.log2(jn.p_tok(p, c))
            p = c
        ind = len(row) - len(row.lstrip(" "))
        if prev_last is not None:
            ev = jn._delta(ind - prev_ind) + "\x00" + cl[0]
            line[r, :ind + 1] = -math.log2(jn.p_line(prev_last, ev))
        prev_last, prev_ind = cl[-1], ind
    # end-of-function junction on the newline cell of the last non-empty row
    last = max(i for i, row in enumerate(rows) if row.strip())
    line[last, len(rows[last].encode())] = -math.log2(jn.p_line(prev_last, "END\x00END"))
    assert pos == len(s), (pos, len(s))
    return {"byte": byte, "tok": tok, "line": line, "mask": mask}


# ------------------------------------------------------------------ byte-LM surprisal (baseline + LM relief)
class LM:
    def __init__(self, device):
        sys.path.insert(0, BYTEBRAIN)
        from src.model import ByteGPT, load_checkpoint
        cfg = torch.load(CKPT, map_location="cpu", weights_only=False)["config"]
        self.m = ByteGPT(cfg["dim"], cfg["layers"], cfg["heads"], cfg["ctx"])
        load_checkpoint(self.m, CKPT)
        self.m.eval().to(device)
        self.ctx = cfg["ctx"]
        self.device = device

    @torch.no_grad()
    def surprisal(self, texts, bs=32):
        """nats per byte for each byte of text + '\n' (context: b'\n\n' prefix). Returns list of arrays."""
        C, H = self.ctx, self.ctx // 2
        jobs = []                                   # (text idx, window start, seq slice, positions assigned)
        seqs = []
        for ti, t in enumerate(texts):
            seq = b"\n\n" + t.encode() + b"\n"
            seqs.append(seq)
            n = len(seq)
            starts = [0] + [s for s in range(H, n, H) if s + C < n + H]
            # assign each target position i (2..n-1) to the window j with i - s_j in (H, C], or j=0 for i <= C
            assign = {}
            for i in range(2, n):
                j = 0 if i <= C else math.ceil((i - C) / H)
                assign.setdefault(j * H, []).append(i)
            for s0, pos in assign.items():
                jobs.append((ti, s0, pos))
        out = [np.zeros(len(s) - 2) for s in seqs]
        jobs.sort(key=lambda j: len(seqs[j[0]][j[1]:j[1] + C]))
        for b in range(0, len(jobs), bs):
            chunk = jobs[b:b + bs]
            L = max(min(C, len(seqs[ti]) - s0) for ti, s0, _ in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            for k, (ti, s0, _) in enumerate(chunk):
                w = seqs[ti][s0:s0 + L]
                x[k, :len(w)] = torch.tensor(list(w))
            lp = torch.log_softmax(self.m(x.to(self.device)).float(), -1).cpu()
            for k, (ti, s0, pos) in enumerate(chunk):
                seq = seqs[ti]
                p = np.asarray(pos)
                tg = torch.tensor([seq[i] for i in pos])
                out[ti][p - 2] = -lp[k, torch.from_numpy(p - 1 - s0), tg].numpy()
        return out


# ------------------------------------------------------------------ water
def canvas(hs, H, W):
    """hs: list of (R, CMAX+1) height arrays (0 off-line). -> (B, H, W) with 1-cell outlet border."""
    c = torch.zeros(len(hs), H, W, dtype=torch.float32)
    for i, h in enumerate(hs):
        c[i, 1:1 + h.shape[0], 1:1 + h.shape[1]] = torch.from_numpy(h).float()
    return c


def border_mask(B, H, W, device):
    m = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    m[:, 0, :] = m[:, -1, :] = m[:, :, 0] = m[:, :, -1] = True
    return m


def outlet_mask(nrows, H, W, device):
    """border + every row below each function's last row (function rows are 1..R)."""
    m = border_mask(len(nrows), H, W, device)
    for i, r in enumerate(nrows):
        m[i, 1 + r:, :] = True
    return m


def nbr_min(W):
    big = torch.full_like(W[:, :1, :], float("inf"))
    up = torch.cat([big, W[:, :-1, :]], 1)
    dn = torch.cat([W[:, 1:, :], big], 1)
    bigc = torch.full_like(W[:, :, :1], float("inf"))
    lf = torch.cat([bigc, W[:, :, :-1]], 2)
    rt = torch.cat([W[:, :, 1:], bigc], 2)
    return torch.minimum(torch.minimum(up, dn), torch.minimum(lf, rt))


def flood(h, outlet=None):
    """Depression filling: W = max(h, min over 4-neighbours W), outlet W = h. Returns depth W - h."""
    bm = border_mask(*h.shape, h.device) if outlet is None else outlet
    W = torch.where(bm, h, torch.full_like(h, float("inf")))
    for it in range(20000):
        Wn = torch.where(bm, h, torch.maximum(h, torch.minimum(W, nbr_min(W))))
        if it % 8 == 0 and torch.equal(Wn, W):
            break
        W = Wn
    else:
        raise RuntimeError("flood did not converge")
    return W - h


def shift(x, dr, dc):
    """value at (r, c) <- x[r - dr, c - dc]; zero fill."""
    y = torch.zeros_like(x)
    H, W = x.shape[1:]
    y[:, max(dr, 0):H + min(dr, 0), max(dc, 0):W + min(dc, 0)] = \
        x[:, max(-dr, 0):H + min(-dr, 0), max(-dc, 0):W + min(-dc, 0)]
    return y


def shallow(h, rain=RAIN, steps=STEPS, kappa=KAPPA, rain_mask=None, outlet=None):
    """Diffusion-wave shallow water (simplified Navier-Stokes without inertia). Returns depth, flow.
    rain_mask: optional bool canvas; rain falls only there (BYTEPOUR arm). outlet: bool canvas (default border)."""
    bm = border_mask(*h.shape, h.device) if outlet is None else outlet
    w = torch.where(bm, torch.zeros_like(h), torch.full_like(h, rain))
    if rain_mask is not None:
        w = torch.where(rain_mask, w, torch.zeros_like(w))
    flow = torch.zeros_like(h)
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for _ in range(steps):
        eta = h + w
        f = []
        for dr, dc in dirs:
            nb = shift(eta, -dr, -dc)            # eta of neighbour at (r+dr, c+dc)
            valid = shift(torch.ones_like(eta), -dr, -dc) > 0
            f.append(torch.where(valid, kappa * torch.relu(eta - nb), torch.zeros_like(eta)))
        tot = f[0] + f[1] + f[2] + f[3]
        scale = torch.clamp(w / (tot + 1e-12), max=1.0)
        f = [fi * scale for fi in f]
        out = f[0] + f[1] + f[2] + f[3]
        inflow = sum(shift(fi, dr, dc) for fi, (dr, dc) in zip(f, dirs))
        w = w - out + inflow
        w = torch.where(bm, torch.zeros_like(w), w)
        flow += out
    return w, flow


# ------------------------------------------------------------------ defects
def body_rows(rows):
    hdr = next(i for i, r in enumerate(rows) if r.startswith(("def ", "async def ")))
    return list(range(hdr + 1, len(rows)))


def indent(r):
    return len(r) - len(r.lstrip(" "))


def make_defect(kind, text, rng, foreign_pool, rename_pool):
    """Returns (defected text, target rows set) or None if not applicable."""
    rows = text.split("\n")
    body = body_rows(rows)
    if kind == "delete_line":
        if len(body) < 2:
            return None
        i = int(rng.choice(body))
        new = rows[:i] + rows[i + 1:]
        tg = {r for r in (i - 1, i) if 0 <= r < len(new)}
        return "\n".join(new), tg
    if kind == "delete_return":
        cand = [i for i in body if re.match(r"^\s*return \S", rows[i])]
        if not cand:
            return None
        i = int(rng.choice(cand))
        new = list(rows)
        new[i] = re.sub(r"^(\s*)return ", r"\1", rows[i], count=1)
        return "\n".join(new), {i}
    if kind == "swap":
        cand = [i for i in body if i + 1 < len(rows) and rows[i] != rows[i + 1]]
        if not cand:
            return None
        i = int(rng.choice(cand))
        new = list(rows)
        new[i], new[i + 1] = rows[i + 1], rows[i]
        return "\n".join(new), {i, i + 1}
    if kind == "wrong_indent":
        i = int(rng.choice(body))
        d = indent(rows[i])
        delta = int(rng.choice([-4, 4])) if d >= 8 else 4
        new = list(rows)
        new[i] = " " * (d + delta) + rows[i].lstrip(" ")
        if len(new[i].encode()) > CMAX:
            return None
        return "\n".join(new), {i}
    if kind in ("foreign_token", "junk"):
        i = int(rng.choice(body))
        tk = [t for t in tokens(rows[i])]
        if not tk:
            return None
        s = tk[int(rng.integers(len(tk)))][0]
        ins = (foreign_pool[int(rng.integers(len(foreign_pool)))] if kind == "foreign_token" else "$?`") + " "
        rb = rows[i].encode()
        newrow = (rb[:s] + ins.encode() + rb[s:]).decode()
        if len(newrow.encode()) > CMAX:
            return None
        new = list(rows)
        new[i] = newrow
        return "\n".join(new), {i}
    if kind == "rename":
        tree = ast.parse(text)
        occ = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                occ.setdefault(n.id, []).append((n.lineno - 1, n.col_offset, n.end_col_offset,
                                                 isinstance(n.ctx, ast.Load)))
            elif isinstance(n, ast.arg):
                occ.setdefault(n.arg, []).append((n.lineno - 1, n.col_offset, n.col_offset + len(n.arg.encode()),
                                                  False))
        bodyset = set(body)
        cand = [v for v, o in occ.items() if v != "self" and any(not ld for *_, ld in o)
                and sum(1 for r, _, _, ld in o if ld and r in bodyset) >= 1 and len(o) >= 2]
        if not cand:
            return None
        v = sorted(cand)[int(rng.integers(len(cand)))]
        loads = sorted(o for o in occ[v] if o[3] and o[0] in bodyset)
        stale = loads[int(rng.integers(len(loads)))]
        words = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text))
        pool = [w for w in rename_pool if w not in words]
        new_name = pool[int(rng.integers(len(pool)))]
        rb = [r.encode() for r in rows]
        for (r, a, b, ld) in sorted(occ[v], key=lambda o: (o[0], o[1]), reverse=True):
            if (r, a, b, ld) == stale:
                continue
            if rb[r][a:b] != v.encode():
                return None                      # position mismatch: skip instance (counted via shortfall)
            rb[r] = rb[r][:a] + new_name.encode() + rb[r][b:]
        new = [x.decode() for x in rb]
        if max(len(x) for x in rb) > CMAX:
            return None
        return "\n".join(new), {stale[0]}
    raise ValueError(kind)


# ------------------------------------------------------------------ scoring helpers
def line_scores(q, mask, agg):
    """q, mask: (R, C) arrays; per row max (or sum) over the row's line cells."""
    qq = np.where(mask, q, -np.inf if agg == "max" else 0.0)
    return qq.max(1) if agg == "max" else qq.sum(1)


def parse_line(text):
    try:
        compile(text, "<f>", "exec")
        return None
    except SyntaxError as e:
        n = len(text.split("\n"))
        return min(max((e.lineno or 1) - 1, 0), n - 1)


def auroc(pos, neg):
    from scipy.stats import rankdata
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--per-type", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    dev = torch.device(a.device)
    t0 = time.time()
    cache_dir = OUT  # stats caches shared between cpu/gpu runs

    train_text = open(TRAIN, encoding="utf-8", errors="replace").read()
    train_fns = split_fns(train_text)
    train_h = {h16(f) for f in train_fns}
    log(f"train functions {len(train_fns)}")
    bng = ByteNgram(np.frombuffer(train_text.encode("utf-8", "surrogatepass"), dtype=np.uint8),
                    os.path.join(cache_dir, "byte_ngram.npz"))
    jn = Junctions(train_fns, os.path.join(cache_dir, "junctions.json"))
    jn.setup()
    rename_pool = [w for w in jn.vocab[:400] if not keyword.iskeyword(w) and w not in dir(__builtins__)
                   and w != "self" and len(w) > 1]
    del train_text, train_fns
    log("stats ready")

    hold = split_fns(open(HOLD, encoding="utf-8", errors="replace").read())
    ev, calib = [], []
    for f in hold:
        if h16(f) in train_h:
            continue
        (ev if eligible(f) else calib).append(f)
    ndup = len(hold) - len(ev) - len(calib)
    calib = [f for f in calib if len(f.encode()) <= 3000 and "\t" not in f
             and max(len(r.encode()) for r in f.split("\n")) <= CMAX and
             any(r.startswith(("def ", "async def ")) for r in f.split("\n"))][:600]
    log(f"holdout {len(hold)}: eval {len(ev)}, calib {len(calib)}, dropped as in-train {ndup}")
    if a.smoke:
        ev = ev[:60]
        a.per_type = 20

    # calibration z-stats per channel
    lm = LM(dev)
    stats = {}
    cal_ch = [channels(f, bng, jn) for f in calib[:400 if not a.smoke else 40]]
    for ch in ("byte", "tok", "line"):
        v = np.concatenate([c[ch][c["mask"] & ~np.isnan(c[ch])] for c in cal_ch])
        stats[ch] = (float(v.mean()), float(v.std()))
    cal_lm = lm.surprisal(calib[:400 if not a.smoke else 40])
    v = np.concatenate(cal_lm)
    stats["lm"] = (float(v.mean()), float(v.std()))
    log("calibration z-stats", json.dumps(stats))

    # foreign token pool: tokens from eval functions (sampled per instance from another function)
    foreign_pool = []
    for f in ev:
        for r in f.split("\n")[1:]:
            foreign_pool += [t[2] for t in tokens(r)]

    # build instances
    inst = []          # dict(seed, kind, fid, text, targets)
    for seed in a.seeds:
        for kind in DEFECTS:
            rng = np.random.default_rng([seed, DEFECTS.index(kind), 53])
            order = rng.permutation(len(ev))
            got = 0
            for fid in order:
                fid = int(fid)
                d = make_defect(kind, ev[fid], rng, foreign_pool, rename_pool)
                if d is None or d[0] == ev[fid]:
                    continue
                inst.append({"seed": seed, "kind": kind, "fid": fid, "text": d[0], "targets": sorted(d[1])})
                got += 1
                if got == a.per_type:
                    break
            log(f"seed {seed} {kind}: {got} instances")
    clean_ids = sorted({i["fid"] for i in inst})
    texts = {("c", fid): ev[fid] for fid in clean_ids}
    for k, i in enumerate(inst):
        texts[("d", k)] = i["text"]
    keys = list(texts)
    uniq = sorted(set(texts.values()))
    log(f"{len(inst)} defect instances, {len(clean_ids)} clean functions, {len(uniq)} unique texts, "
        f"{sum(len(t) for t in uniq)} bytes for the LM")

    # LM surprisal (cached by text)
    t1 = time.time()
    lm_s = {}
    lmc = os.path.join(cache_dir, "lm_cache.npz")
    if os.path.exists(lmc):
        z = np.load(lmc, allow_pickle=True)
        lm_s = dict(zip(z["texts"].tolist(), z["surp"].tolist()))
        log(f"LM cache: {len(lm_s)} texts")
    todo = [t for t in uniq if t not in lm_s]
    B = 400
    for b in range(0, len(todo), B):
        part = todo[b:b + B]
        for t, s in zip(part, lm.surprisal(part)):
            lm_s[t] = s
        log(f"LM {min(b + B, len(todo))}/{len(todo)} ({time.time() - t1:.0f} s)")
    if todo and not a.smoke:
        np.savez(lmc, texts=np.array(list(lm_s), dtype=object),
                 surp=np.array([lm_s[t] for t in lm_s], dtype=object))
    clean_nats = float(np.mean(np.concatenate([lm_s[ev[f]] for f in clean_ids])))
    log(f"GATE5 LM bits/byte on eval clean = {clean_nats / LN2:.3f} (val 1.118 bits; accept 0.8-1.5)")

    # per text: grids
    def grids(t):
        c = channels(t, bng, jn)
        z = [np.nan_to_num((c[ch] - stats[ch][0]) / stats[ch][1], nan=0.0) for ch in ("byte", "tok", "line")]
        S = np.where(c["mask"], z[0] + z[1] + z[2], 0.0)
        lmz = np.zeros_like(S)
        s = (lm_s[t] - stats["lm"][0]) / stats["lm"][1]
        pos = 0
        for r, row in enumerate(t.split("\n")):
            L = len(row.encode()) + 1
            lmz[r, :L] = s[pos:pos + L]
            pos += L
        zs = {n: np.where(c["mask"], zz, 0.0) for n, zz in zip(("byte", "tok", "line"), z)}
        return {"S": S, "LM": lmz, "mask": c["mask"], **zs}

    t2 = time.time()
    G = {t: grids(t) for t in uniq}
    log(f"grids built ({time.time() - t2:.0f} s)")

    # water, batched
    Hc, Wc = RMAX + 2, CMAX + 3
    water = {}
    for src in ("S", "LM"):
        for b in range(0, len(uniq), 512):
            part = uniq[b:b + 512]
            h = canvas([-G[t][src] for t in part], Hc, Wc).to(dev)
            om = outlet_mask([G[t]["mask"].shape[0] for t in part], Hc, Wc, dev)
            fd = flood(h, om).cpu().numpy()
            sd, fl = shallow(h, outlet=om)
            sd, fl = sd.cpu().numpy(), fl.cpu().numpy()
            if src == "S":
                rm = canvas([G[t]["mask"].astype(np.float32) for t in part], Hc, Wc).to(dev) > 0
                bp = shallow(h, rain_mask=rm, outlet=om)[0].cpu().numpy()
            for k, t in enumerate(part):
                R = G[t]["mask"].shape[0]
                water[(src, "flood", t)] = fd[k, 1:1 + R, 1:CMAX + 2]
                water[(src, "shallow", t)] = sd[k, 1:1 + R, 1:CMAX + 2]
                water[(src, "flow", t)] = fl[k, 1:1 + R, 1:CMAX + 2]
                if src == "S":
                    water[(src, "bytepour", t)] = bp[k, 1:1 + R, 1:CMAX + 2]
        log(f"water on {src} done ({time.time() - t2:.0f} s)")

    methods = ["random", "longest", "parse", "lm", "relief", "relief+flood", "relief+shallow", "relief+bytepour",
               "lm+flood", "lm+shallow", "relief_byte", "relief_tok", "relief_line"]

    def scores(t, agg, rng):
        g = G[t]
        m = g["mask"]
        rows = t.split("\n")
        R = len(rows)
        out = {
            "random": rng.random(R),
            "longest": np.array([len(r.encode()) for r in rows], float),
            "lm": line_scores(g["LM"], m, agg),
            "relief": line_scores(-(-g["S"]), m, agg),
            "relief+flood": line_scores(water[("S", "flood", t)], m, agg),
            "relief+shallow": line_scores(water[("S", "shallow", t)], m, agg),
            "relief+bytepour": line_scores(water[("S", "bytepour", t)], m, agg),
            "lm+flood": line_scores(water[("LM", "flood", t)], m, agg),
            "lm+shallow": line_scores(water[("LM", "shallow", t)], m, agg),
            "relief_byte": line_scores(g["byte"], m, agg),
            "relief_tok": line_scores(g["tok"], m, agg),
            "relief_line": line_scores(g["line"], m, agg),
        }
        pl = parse_line(t)
        pr = rng.random(R)
        if pl is not None:
            pr[pl] += 3.0
            for d in (-1, 1):
                if 0 <= pl + d < R:
                    pr[pl + d] += 2.0
        out["parse"] = pr
        for k in out:
            out[k] = out[k] + rng.random(R) * 1e-7          # random tie-break
        return out, pl is not None

    results = {"config": {"orders": ORDERS, "beta": BETA, "rain": RAIN, "steps": STEPS, "kappa": KAPPA,
                          "seeds": a.seeds, "per_type": a.per_type, "device": a.device, "n_eval_fns": len(ev),
                          "calib_stats": stats, "lm_clean_nats": clean_nats}, "per_seed": {}, "pooled": {}}
    rec = []           # per instance rows for bootstrap
    for agg in ("max", "sum"):
        rng = np.random.default_rng(53)
        clean_fs = {}
        for fid in clean_ids:
            sc, _ = scores(ev[fid], agg, rng)
            clean_fs[fid] = {m: float(sc[m].max()) for m in methods}
            clean_fs[fid]["parse_ind"] = 0.0
        for k, i in enumerate(inst):
            sc, unparse = scores(i["text"], agg, rng)
            row = {"agg": agg, "seed": i["seed"], "kind": i["kind"], "fid": i["fid"],
                   "rows": len(i["text"].split("\n")), "ntg": len(i["targets"]), "unparse": unparse}
            for m in methods:
                o = np.argsort(-sc[m])
                row[m + "@1"] = int(o[0] in i["targets"])
                row[m + "@3"] = int(any(x in i["targets"] for x in o[:3]))
                row[m + "_fs"] = float(sc[m].max())
            row["parse_ind"] = float(unparse)
            rec.append(row)
        results["clean_fs_" + agg] = {str(f): v for f, v in clean_fs.items()}

    # aggregate
    def boot(vals, fids, n=1000, seed=0):
        # cluster bootstrap over function id: resample functions, keep all their instances
        r = np.random.default_rng(seed)
        u, inv = np.unique(fids, return_inverse=True)
        sm = np.bincount(inv, weights=vals, minlength=len(u))
        ct = np.bincount(inv, minlength=len(u)).astype(float)
        pick = r.integers(0, len(u), size=(n, len(u)))
        bs = sm[pick].sum(1) / ct[pick].sum(1)
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    for agg in ("max", "sum"):
        clean_fs = results["clean_fs_" + agg]
        for kind in DEFECTS:
            rr = [r for r in rec if r["agg"] == agg and r["kind"] == kind]
            if not rr:
                continue
            fids = np.array([r["fid"] for r in rr])
            entry = {"n": len(rr), "unparsable": float(np.mean([r["unparse"] for r in rr])),
                     "random_expect@1": float(np.mean([r["ntg"] / r["rows"] for r in rr]))}
            for m in methods:
                for k in ("@1", "@3"):
                    v = np.array([r[m + k] for r in rr], float)
                    entry[m + k] = float(v.mean())
                    entry[m + k + "_ci"] = boot(v, fids)
                pos = np.array([r[m + "_fs"] for r in rr])
                negf = sorted({r["fid"] for r in rr})
                neg = np.array([clean_fs[str(f)][m] for f in negf])
                entry[m + "_auroc"] = auroc(pos, neg)
            entry["parse_ind_auroc"] = auroc(np.array([r["parse_ind"] for r in rr]), np.zeros(len(negf)))
            up = [r for r in rr if r["unparse"]]
            entry["parse@1_on_unparsable"] = float(np.mean([r["parse@1"] for r in up])) if up else None
            for wm in ("relief+flood", "relief+shallow", "relief+bytepour"):
                d = np.array([r[wm + "@1"] - r["relief@1"] for r in rr], float)
                entry[f"delta@1 {wm}-relief"] = float(d.mean())
                entry[f"delta@1 {wm}-relief_ci"] = boot(d, fids)
            for wm in ("lm+flood", "lm+shallow"):
                d = np.array([r[wm + "@1"] - r["lm@1"] for r in rr], float)
                entry[f"delta@1 {wm}-lm"] = float(d.mean())
                entry[f"delta@1 {wm}-lm_ci"] = boot(d, fids)
            d = np.array([r["lm@1"] - r["relief@1"] for r in rr], float)
            entry["delta@1 lm-relief"] = float(d.mean())
            entry["delta@1 lm-relief_ci"] = boot(d, fids)
            results["pooled"].setdefault(agg, {})[kind] = entry
            for seed in a.seeds:
                rs = [r for r in rr if r["seed"] == seed]
                results["per_seed"].setdefault(agg, {}).setdefault(kind, {})[str(seed)] = {
                    m + "@1": float(np.mean([r[m + "@1"] for r in rs])) for m in methods}
    results["config"]["wall_s"] = time.time() - t0
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "instances.jsonl"), "w") as fh:
        for r in rec:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "defects.jsonl"), "w") as fh:
        for i in inst:
            fh.write(json.dumps(i) + "\n")

    # table
    show = ["random", "longest", "parse", "lm", "relief", "relief+flood", "relief+shallow", "relief+bytepour",
            "lm+flood", "lm+shallow"]
    for agg in ("max", "sum"):
        log(f"=== top-1 (agg={agg}) pooled seeds {a.seeds}")
        log("kind".ljust(14) + "".join(m[:13].rjust(14) for m in show))
        for kind, e in results["pooled"][agg].items():
            log(kind.ljust(14) + "".join(f"{e[m + '@1']:.3f}".rjust(14) for m in show))
        log(f"=== AUROC (agg={agg})")
        for kind, e in results["pooled"][agg].items():
            log(kind.ljust(14) + "".join(f"{e[m + '_auroc']:.3f}".rjust(14) for m in show)
                + f"  parse_ind {e['parse_ind_auroc']:.3f} unparsable {e['unparsable']:.2f}")
        log(f"=== water - relief top-1 (agg={agg})")
        for kind, e in results["pooled"][agg].items():
            log(kind.ljust(14) + "  ".join(
                f"{wm}: {e[f'delta@1 {wm}-relief']:+.3f} [{e[f'delta@1 {wm}-relief_ci'][0]:+.3f},"
                f"{e[f'delta@1 {wm}-relief_ci'][1]:+.3f}]" for wm in ("relief+flood", "relief+shallow",
                                                                     "relief+bytepour")))

    panel(a.out, inst, G, water, results)
    log(f"done, wall {time.time() - t0:.0f} s")


def panel(out, inst, G, water, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    kinds = [k for k in DEFECTS]
    ex = []
    for k in kinds:
        e = next((i for i in inst if i["kind"] == k and i["seed"] == inst[0]["seed"]), None)
        if e:
            ex.append(e)
    fig = plt.figure(figsize=(34, 4.2 * len(ex) + 6), facecolor="#111")
    gs = fig.add_gridspec(len(ex) + 1, 4, height_ratios=[4.2] * len(ex) + [6])
    for r, e in enumerate(ex):
        t = e["text"]
        g = G[t]
        rows = t.split("\n")
        R = len(rows)
        wlen = max(len(x.encode()) for x in rows) + 2
        panels = [("relief height (-surprise): dark = depression", -g["S"], "terrain"),
                  ("FLOOD depth (pools)", water[("S", "flood", t)], "Blues"),
                  ("SHALLOW water depth", water[("S", "shallow", t)], "Blues"),
                  ("SHALLOW flow (paths)", np.log1p(water[("S", "flow", t)]), "magma")]
        for c, (title, img, cmap) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(np.where(g["mask"], img, np.nan)[:, :wlen], cmap=cmap, aspect="auto",
                      interpolation="nearest")
            for tr in e["targets"]:
                ax.add_patch(Rectangle((-0.5, tr - 0.5), wlen, 1, fill=False, ec="red", lw=2))
            if c == 0:
                for i, row in enumerate(rows):
                    ax.text(0, i, row[:wlen], fontsize=5.5, family="monospace", va="center", color="black")
            ax.set_title(f"{e['kind']} | {title}", color="w", fontsize=9)
            ax.tick_params(colors="w", labelsize=6)
    ax = fig.add_subplot(gs[-1, :])
    P = results["pooled"]["max"]
    ms = ["random", "longest", "parse", "lm", "relief", "relief+flood", "relief+shallow", "relief+bytepour",
          "lm+flood"]
    xs = np.arange(len(P))
    wbar = 0.8 / len(ms)
    for j, m in enumerate(ms):
        v = [P[k][m + "@1"] for k in P]
        lo = [P[k][m + "@1"] - P[k][m + "@1_ci"][0] for k in P]
        hi = [P[k][m + "@1_ci"][1] - P[k][m + "@1"] for k in P]
        ax.bar(xs + j * wbar, v, wbar, yerr=[lo, hi], label=m, capsize=2)
    ax.set_xticks(xs + 0.4)
    ax.set_xticklabels(list(P), color="w")
    ax.set_ylabel("top-1 line hit (95% CI)", color="w")
    ax.tick_params(colors="w")
    ax.set_facecolor("#222")
    ax.legend(ncol=8, fontsize=9)
    ax.set_title("#53 relief + water: does the deepest pool sit on the defect line? (red boxes = defect rows)",
                 color="w")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "relief53.png"), dpi=80, facecolor="#111")


if __name__ == "__main__":
    main()
