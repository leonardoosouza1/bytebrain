"""Python lexical/structural checker as a batched GPU automaton, advanced one byte at a time inside a sampling loop.

State per sequence (all tensors [B]): lexer state (code / comment / 1-line or triple string per quote / escapes),
bracket stack [B, 64] with pointer, indentation (stack of open indent widths, as the Python tokenizer keeps), whether the previous
logical line ended with ':' (next line must indent by one), and an error flag with the byte position where it rose.
Catches: unmatched/unclosed brackets, unterminated strings, newline inside a 1-line string, bad indentation (not a
missing indent after ':', unexpected indent, dedent to a width never opened). Does not catch grammar beyond that (e.g. `x = = 1`).

  chk = SyntaxAutomaton(B, device)
  chk.step(byte_tensor_B)          # advance; chk.err [B] bool, chk.err_pos [B]
  chk.allowed()                    # [B, 256] bool: bytes that would not raise an error now (for masking)
  chk.complete()                   # [B] bool: could the text end here (no open string/bracket, no pending indent)

Usage (validation against ast.parse): gpu_syntax.py --samples FILE.json  (eval_samples/mix_eval rows or real)
"""

import argparse
import ast
import json
import random
import time

import torch

CODE, COMMENT, ERR = 0, 1, 2
# per quote q (0 = ', 1 = "): base + offset
S1, S1E, O1, O2, T, TE, T1, T2 = range(8)
NQ = 8
N_STATES = 3 + 2 * NQ
QUOTES = (39, 34)
OPEN = {40: 41, 91: 93, 123: 125}
CLOSE = {41, 93, 125}
NL, SP, BS, HASH, COLON = 10, 32, 92, 35, 58


def qs(q, s):
    return 3 + q * NQ + s


def build_table():
    """next-state [N_STATES, 256] and code_char [N_STATES, 256] (byte is a code character, not string/comment)."""
    nxt = torch.full((N_STATES, 256), ERR, dtype=torch.long)
    code = torch.zeros((N_STATES, 256), dtype=torch.bool)
    for c in range(256):
        # CODE
        if c in QUOTES:
            nxt[CODE, c] = qs(QUOTES.index(c), O1)
        elif c == HASH:
            nxt[CODE, c] = COMMENT
        else:
            nxt[CODE, c] = CODE
        code[CODE, c] = True
        nxt[COMMENT, c] = CODE if c == NL else COMMENT
        code[COMMENT, c] = c == NL
        for q, qc in enumerate(QUOTES):
            # inside a one-line string
            nxt[qs(q, S1), c] = CODE if c == qc else qs(q, S1E) if c == BS else ERR if c == NL else qs(q, S1)
            nxt[qs(q, S1E), c] = qs(q, S1)
            # saw one opening quote: second quote -> maybe triple; else a one-line string body starts
            nxt[qs(q, O1), c] = qs(q, O2) if c == qc else qs(q, S1E) if c == BS else ERR if c == NL else qs(q, S1)
            # saw two quotes: third -> triple string; else the string was empty and c is code
            if c == qc:
                nxt[qs(q, O2), c] = qs(q, T)
            else:
                nxt[qs(q, O2), c] = nxt[CODE, c]
                code[qs(q, O2), c] = True
            for s, after in ((T, T1), (T1, T2)):
                nxt[qs(q, s), c] = qs(q, after) if c == qc else qs(q, TE) if c == BS else qs(q, T)
            nxt[qs(q, T2), c] = CODE if c == qc else qs(q, TE) if c == BS else qs(q, T)
            nxt[qs(q, TE), c] = qs(q, T)
    return nxt, code


class SyntaxAutomaton:
    def __init__(self, B, device="cuda", depth=64):
        self.nxt, self.code = (t.to(device) for t in build_table())
        self.B, self.dev, self.depth = B, device, depth
        z = lambda dt=torch.long: torch.zeros(B, dtype=dt, device=device)  # noqa: E731
        self.state, self.ptr, self.spaces, self.t = z(), z(), z(), 0
        self.ind = torch.zeros(B, 32, dtype=torch.long, device=device)   # open indent widths; ind[:, 0] = 0
        self.iptr = z()                                                  # index of the current (top) width
        self.stack = torch.zeros(B, depth, dtype=torch.long, device=device)
        self.line_start = torch.ones(B, dtype=torch.bool, device=device)
        self.line_has = z(torch.bool)      # current logical line has a non-space code char
        self.last_colon = z(torch.bool)    # last non-space code char at depth 0 was ':'
        self.expect = z(torch.bool)        # previous logical line ended with ':'
        self.err = z(torch.bool)
        self.err_pos = torch.full((B,), -1, dtype=torch.long, device=device)
        self.ar = torch.arange(B, device=device)
        self.closer = torch.zeros(256, dtype=torch.long, device=device)
        for o, c in OPEN.items():
            self.closer[o] = c

    def _indent_bad(self, spaces):
        """[B] -> bad: after ':' the width must exceed the top width; otherwise it must equal an open width."""
        top = self.ind[self.ar, self.iptr]
        k = torch.arange(32, device=self.dev)[None]
        on_stack = ((self.ind == spaces[:, None]) & (k <= self.iptr[:, None])).any(1)
        return torch.where(self.expect, spaces <= top, ~on_stack)

    @torch.no_grad()
    def step(self, c):
        c = c.long()
        s = self.state
        code = self.code[s, c] & ~self.err
        new = self.nxt[s, c]
        depth0 = self.ptr == 0
        e = new == ERR
        # brackets (code chars only)
        is_open = code & (self.closer[c] > 0)
        is_close = code & ((c == 41) | (c == 93) | (c == 125))
        top = self.stack[self.ar, (self.ptr - 1).clamp(min=0)]
        e |= is_close & ((self.ptr == 0) | (top != c))
        e |= is_open & (self.ptr >= self.depth)
        # indentation: first non-space code char of a physical line at depth 0
        first = code & self.line_start & depth0 & (c != SP) & (c != NL)
        e |= first & self._indent_bad(self.spaces)
        # update
        e &= ~self.err
        self.err_pos = torch.where(e, torch.full_like(self.err_pos, self.t), self.err_pos)
        self.err |= e
        ok = ~self.err
        push = is_open & ok
        self.stack[self.ar[push], self.ptr[push]] = self.closer[c[push]]
        self.ptr = self.ptr + push.long() - (is_close & ok).long()
        f_ok = first & ok
        # indent: push the new width; dedent: pop to the matching width
        push_i = f_ok & self.expect
        self.iptr = self.iptr + push_i.long()
        self.ind[self.ar[push_i], self.iptr[push_i]] = self.spaces[push_i]
        k = torch.arange(32, device=self.dev)[None]
        match = (self.ind == self.spaces[:, None]) & (k <= self.iptr[:, None])
        pos = torch.where(match, k.expand_as(match), torch.full_like(match, -1, dtype=torch.long)).max(1).values
        self.iptr = torch.where(f_ok & ~push_i, pos, self.iptr)
        self.expect = torch.where(first & ok, torch.zeros_like(self.expect), self.expect)
        end_line = code & (c == NL) & depth0 & ok
        self.expect = torch.where(end_line & self.line_has, self.last_colon, self.expect)
        self.line_has = torch.where(end_line, torch.zeros_like(self.line_has), self.line_has | (first & ok))
        nonsp = code & (c != SP) & (c != NL) & (self.ptr == 0) & ok
        self.last_colon = torch.where(nonsp | (is_close & ok), c == COLON, self.last_colon)
        self.line_start = torch.where(code & (c == NL) & ok, torch.ones_like(self.line_start),
                                      torch.where(first, torch.zeros_like(self.line_start), self.line_start))
        self.spaces = torch.where(code & (c == NL), torch.zeros_like(self.spaces),
                                  self.spaces + (code & self.line_start & (c == SP)).long())
        self.state = torch.where(self.err, torch.full_like(new, ERR), new)
        self.t += 1

    @torch.no_grad()
    def allowed(self):
        """[B, 256]: candidate bytes that keep the sequence error-free after this step."""
        s = self.state[:, None]
        cand = torch.arange(256, device=self.dev)[None].expand(self.B, 256)
        code = self.code[s, cand]
        ok = self.nxt[s, cand] != ERR
        top = self.stack[self.ar, (self.ptr - 1).clamp(min=0)][:, None]
        is_close = code & ((cand == 41) | (cand == 93) | (cand == 125))
        ok &= ~(is_close & ((self.ptr[:, None] == 0) | (top != cand)))
        first = code & self.line_start[:, None] & (self.ptr[:, None] == 0) & (cand != SP) & (cand != NL)
        ok &= ~(first & self._indent_bad(self.spaces)[:, None])
        return ok & ~self.err[:, None]

    def complete(self):
        in_str = (self.state >= 3) & (((self.state - 3) % NQ) != O2)
        return ~self.err & ~in_str & (self.ptr == 0) & ~(self.expect & ~self.line_has)


def check_texts(texts, device="cuda"):
    """-> (flagged [N] bool, err_pos [N]): error raised or text not complete at its end."""
    data = [t.encode() + b"\n" for t in texts]
    L = max(len(d) for d in data)
    B = len(data)
    buf = torch.full((B, L), NL, dtype=torch.long)
    for i, d in enumerate(data):
        buf[i, :len(d)] = torch.tensor(list(d))
    buf = buf.to(device)
    chk = SyntaxAutomaton(B, device)
    for t in range(L):
        chk.step(buf[:, t])
    return (chk.err | ~chk.complete()).cpu(), chk.err_pos.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="*", default=[], help="eval_samples/mix_eval json files with rows[...]['src']")
    ap.add_argument("--n-real", type=int, default=2000)
    args = ap.parse_args()
    hold = json.load(open("/media/llm-data/datasets/pycode/holdout_functions.json"))
    random.Random(3).shuffle(hold)
    real = [h["clean"] for h in hold[:args.n_real]]
    t0 = time.perf_counter()
    flag, _ = check_texts(real)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    nbytes = sum(len(t) + 1 for t in real)
    print(f"real held-out: {int(flag.sum())}/{len(real)} flagged (false positives)  "
          f"{dt:.2f}s, {1e9 * dt / (len(real) * max(len(t) for t in real)):.1f} ns per seq-byte")
    for i in [i for i in range(len(real)) if flag[i]][:3]:
        print("  FP example:", repr(real[i][:300]))
    # negative control: real functions with one random corruption
    r = random.Random(4)
    bad = []
    for t in real[:1000]:
        j = r.randrange(len(t))
        k = r.random()
        bad.append(t[:j] + r.choice("()[]{}'\":") + t[j:] if k < 0.5 else t[:j] + t[j + 1:])
    truth = []
    for t in bad:
        try:
            ast.parse(t)
            truth.append(True)
        except (SyntaxError, ValueError):
            truth.append(False)
    flag, _ = check_texts(bad)
    report(flag, truth, "corrupted real")
    for f in args.samples:
        rows = json.load(open(f))["rows"]
        srcs = [r["src"] for part in rows.values() for r in part]
        truth = [r["parses"] for part in rows.values() for r in part]
        flag, _ = check_texts(srcs)
        report(flag, truth, f)


def report(flag, truth, name):
    fails = [i for i, ok in enumerate(truth) if not ok]
    caught = sum(bool(flag[i]) for i in fails)
    fp = sum(bool(flag[i]) for i, ok in enumerate(truth) if ok)
    print(f"{name}: ast fails {len(fails)}/{len(truth)}, automaton catches {caught} "
          f"({caught / max(1, len(fails)):.1%}), flags {fp} that parse (false positives)")


if __name__ == "__main__":
    main()
