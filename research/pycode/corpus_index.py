"""Byte-level infinity-gram index of a corpus on the GPU (suffix array, prefix doubling).

The corpus is the definition of "correct": for any context, the index returns how often each next byte followed
the longest suffix of that context that occurs in the corpus. Built once (seconds), held in VRAM, queried in batch.

  idx = CorpusIndex.build(path, depth=64)          # suffixes sorted on their first `depth` bytes
  m, cnt, hist = idx.next_bytes(ctx)               # ctx: [B, L] uint8/long, right-aligned
      m    [B]      length of the longest context suffix found in the corpus (0..L)
      cnt  [B]      occurrences of that suffix
      hist [B,256]  next-byte counts after it (all occurrences up to `cap`, sampled evenly beyond)

Usage (self-check + build timing): corpus_index.py CORPUS [--depth 64]
"""

import argparse
import os
import time

import numpy as np
import torch


class CorpusIndex:
    def __init__(self, data, sa, depth):
        self.data, self.sa, self.depth = data, sa, depth
        self.n = data.numel()
        # padded view: reading past the end yields -1, below every byte (matches the sentinel rank 0)
        self.pad = torch.cat([data.long(), torch.full((depth + 1,), -1, device=data.device, dtype=torch.long)])

    @classmethod
    def build(cls, path, depth=64, device="cuda"):
        data = torch.from_numpy(np.fromfile(path, dtype=np.uint8)).to(device)
        n = data.numel()
        rank = data.long() + 1                       # 1..256; 0 = past the end
        k = 1
        while True:
            nxt = torch.zeros_like(rank)
            nxt[:n - k] = rank[k:]
            key = rank * (n + 2) + nxt
            del nxt
            key, sa = torch.sort(key)
            new = torch.empty_like(rank)
            new[sa] = torch.cumsum(torch.cat([key.new_ones(1), (key[1:] != key[:-1]).long()]), 0)
            del key
            rank = new
            k *= 2
            if k >= depth or int(rank.max()) == n:
                break
        return cls(data, sa, depth)

    @classmethod
    def cached(cls, path, depth=64, device="cuda"):
        """Load the suffix array saved next to the corpus (int32), building and saving it once if absent."""
        sa_path = f"{path}.sa{depth}.npy"
        if os.path.exists(sa_path):
            data = torch.from_numpy(np.fromfile(path, dtype=np.uint8)).to(device)
            sa = torch.from_numpy(np.load(sa_path).astype(np.int64)).to(device)
            return cls(data, sa, depth)
        idx = cls.build(path, depth, device)
        np.save(sa_path, idx.sa.to(torch.int32).cpu().numpy())
        return idx

    def _cmp(self, pos, pat, m):
        """sign(corpus[pos:pos+m] - pat[:, -m:]) lexicographic, per row. pos [B], pat [B, L] long."""
        ar = torch.arange(m, device=pat.device)
        a = self.pad[pos[:, None] + ar[None]]
        b = pat[:, pat.shape[1] - m:]
        d = a != b
        first = d.long().argmax(1)
        diff = (a.gather(1, first[:, None]) - b.gather(1, first[:, None])).squeeze(1)
        return torch.where(d.any(1), diff.sign(), torch.zeros_like(diff))

    def _bound(self, pat, m, upper):
        lo = torch.zeros(pat.shape[0], dtype=torch.long, device=pat.device)
        hi = torch.full_like(lo, self.n)
        while bool((lo < hi).any()):
            mid = (lo + hi) // 2
            c = self._cmp(self.sa[mid.clamp(max=self.n - 1)], pat, m)
            go_right = (c <= 0) if upper else (c < 0)
            act = lo < hi
            lo = torch.where(act & go_right, mid + 1, lo)
            hi = torch.where(act & ~go_right, mid, hi)
        return lo

    def range(self, pat, m):
        return self._bound(pat, m, False), self._bound(pat, m, True)

    @torch.no_grad()
    def next_bytes(self, ctx, lengths=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64), cap=4096):
        ctx = ctx.long()
        B, L = ctx.shape
        best_m = torch.zeros(B, dtype=torch.long, device=ctx.device)
        lo_b = torch.zeros_like(best_m)
        hi_b = torch.full_like(best_m, self.n)
        for m in lengths:
            if m > min(L, self.depth - 1):
                break
            lo, hi = self.range(ctx, m)
            ok = hi > lo
            best_m = torch.where(ok, torch.full_like(best_m, m), best_m)
            lo_b, hi_b = torch.where(ok, lo, lo_b), torch.where(ok, hi, hi_b)
            if not bool(ok.any()):
                break
        cnt = hi_b - lo_b
        # next byte after each occurrence: up to `cap` evenly spaced occurrences per row
        j = torch.arange(cap, device=ctx.device)
        take = torch.minimum(cnt, torch.full_like(cnt, cap))
        idx = lo_b[:, None] + (j[None] * cnt[:, None]) // take.clamp(min=1)[:, None]
        valid = j[None] < take[:, None]
        pos = self.sa[idx.clamp(max=self.n - 1)] + best_m[:, None]
        nb = self.pad[pos.clamp(max=self.n + self.depth)]
        valid &= nb >= 0
        hist = torch.zeros(B, 257, device=ctx.device)
        hist.scatter_add_(1, torch.where(valid, nb, 256), valid.float())
        return best_m, cnt, hist[:, :256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--mem-frac", type=float, default=0.4)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    t0 = time.perf_counter()
    idx = CorpusIndex.build(args.corpus, args.depth)
    torch.cuda.synchronize()
    print(f"built n={idx.n} in {time.perf_counter() - t0:.1f}s  peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    # self-check: sorted order on the first 16 bytes, and counts vs a brute-force numpy count
    raw = np.fromfile(args.corpus, dtype=np.uint8)
    r = np.random.default_rng(0)
    for i in r.integers(0, idx.n - 1, 5):
        a, b = int(idx.sa[i]), int(idx.sa[i + 1])
        assert bytes(raw[a:a + 16]) <= bytes(raw[b:b + 16]), (i, raw[a:a + 16], raw[b:b + 16])
    for pat in [b"def ", b"return self.", b"    for i in range(", b"raise ValueError("]:
        p = torch.tensor([list(pat)], device="cuda")
        lo, hi = idx.range(p, len(pat))
        brute = raw.tobytes().count(pat)  # non-overlapping; equal for these patterns
        print(f"{pat!r:24} index {int(hi - lo):8d}  brute {brute:8d}")
    ctx = torch.tensor([list(b"        if not isinstance(")], device="cuda")
    t1 = time.perf_counter()
    m, cnt, h = idx.next_bytes(ctx.repeat(256, 1))
    torch.cuda.synchronize()
    top = h[0].topk(5)
    print(f"query x256 {1e3 * (time.perf_counter() - t1):.1f} ms  m={int(m[0])} cnt={int(cnt[0])} next:",
          [(chr(int(b)), int(c)) for c, b in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
