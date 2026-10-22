"""Benchmark symbol-bank lookups from Python: dict, list-of-tuples, numpy, ctypes->Rust.

Usage: bench_py.py BANKDIR [path/to/libbank_ffi.so]
Prints one line per (query, implementation): ns/op (median of 5 repeats) and bytes.
"""

import bisect
import ctypes
import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

RNG = random.Random(0)
NQ = 4096


def load(bankdir):
    raw = (Path(bankdir) / "bank.bin").read_bytes()
    (n,), off, sec = struct.unpack_from("<I", raw), 4, {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", raw, off); off += 4
        name = raw[off:off + ln].decode(); off += ln
        (dl,) = struct.unpack_from("<Q", raw, off); off += 8
        sec[name] = raw[off:off + dl]; off += dl
    u32 = lambda k: np.frombuffer(sec[k], dtype="<u4")
    u8 = lambda k: np.frombuffer(sec[k], dtype=np.uint8)
    return dict(name_offs=u32("name_offs"), name_blob=sec["name_blob"], sym_name=u32("sym_name"),
                sym_scope=u32("sym_scope"), sym_kind=u8("sym_kind"), sym_file=u8("sym_file"),
                sym_def=u32("sym_def"), sym_lo=u32("sym_lo"), sym_hi=u32("sym_hi"),
                sig_offs=u32("sig_offs"), sig_blob=sec["sig_blob"], post_start=u32("post_start"),
                posts=u32("posts"), scope_file=u8("scope_file"), scope_kind=u8("scope_kind"),
                scope_lo=u32("scope_lo"), scope_hi=u32("scope_hi"),
                scope_parent=u32("scope_parent").astype(np.int64),
                ss_start=u32("scope_sym_start"), ss=u32("scope_syms"),
                nib_index=u32("nib_index").reshape(-1, 2), nib_bytes=u8("nib_bytes"))


def bench(fn, queries, reps=5):
    best = []
    for _ in range(reps):
        t = time.perf_counter_ns()
        for q in queries:
            fn(q)
        best.append((time.perf_counter_ns() - t) / len(queries))
    return sorted(best)[len(best) // 2]


def deep_size(obj, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    s = sys.getsizeof(obj)
    if isinstance(obj, dict):
        s += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        s += sum(deep_size(x, seen) for x in obj)
    return s


def main():
    B = load(sys.argv[1])
    so = sys.argv[2] if len(sys.argv) > 2 else None
    nsym = len(B["sym_name"])
    names = [B["name_blob"][B["name_offs"][i]:B["name_offs"][i + 1]].decode()
             for i in range(len(B["name_offs"]) - 1)]
    sigs = [B["sig_blob"][B["sig_offs"][i]:B["sig_offs"][i + 1]].decode() for i in range(nsym)]
    posts_ll = [B["posts"][B["post_start"][i]:B["post_start"][i + 1]].tolist() for i in range(nsym)]
    rows = []

    def rep(query, impl, ns, nbytes, note=""):
        rows.append(dict(query=query, impl=impl, ns=round(ns, 1), bytes=nbytes, note=note))
        print(f"{query:28s} {impl:34s} {ns:10.1f} ns  {nbytes:>9} B  {note}")

    # query workloads, weighted by real use counts (what the loop actually asks for)
    weights = np.diff(B["post_start"]).astype(float) + 1
    q_syms = RNG.choices(range(nsym), weights=weights.tolist(), k=NQ)
    q_names = [names[B["sym_name"][s]] for s in q_syms]

    # ---- name -> id ----
    d_name = {n: i for i, n in enumerate(names)}
    rep("name->id", "py dict", bench(d_name.__getitem__, q_names), deep_size(d_name))
    lot = [(n, i) for i, n in enumerate(names)]
    def lot_find(n):
        for k, v in lot:
            if k == n:
                return v
    rep("name->id", "py list-of-tuples (linear)", bench(lot_find, q_names[:512]), deep_size(lot))
    srt = sorted(lot)
    srt_keys = [k for k, _ in srt]
    rep("name->id", "py sorted list + bisect", bench(lambda n: srt[bisect.bisect_left(srt_keys, n)][1], q_names),
        deep_size(srt) + deep_size(srt_keys))
    # numpy: 64-bit FNV hash, sorted, searchsorted (one call per query, then batched)
    def fnv(s):
        h = 0xcbf29ce484222325
        for c in s.encode():
            h = ((h ^ c) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return h
    hk = np.array([fnv(n) for n in names], dtype=np.uint64)
    order = np.argsort(hk); hk_s = hk[order]; ids_s = order.astype(np.uint32)
    assert len(np.unique(hk)) == len(hk)
    rep("name->id", "numpy hash+searchsorted (1/call)",
        bench(lambda n: ids_s[np.searchsorted(hk_s, fnv(n))], q_names), hk_s.nbytes + ids_s.nbytes,
        "includes python FNV")
    qh = np.array([fnv(n) for n in q_names], dtype=np.uint64)
    t = time.perf_counter_ns(); r = ids_s[np.searchsorted(hk_s, qh)]; t = time.perf_counter_ns() - t
    assert all(names[r[i]] == q_names[i] for i in range(NQ))
    rep("name->id", "numpy searchsorted (batch 4096)", t / NQ, hk_s.nbytes + ids_s.nbytes, "hash precomputed")

    # ---- id -> signature ----
    rep("id->signature", "py list[str]", bench(sigs.__getitem__, q_syms), deep_size(sigs))
    tup = [(names[B["sym_name"][i]], int(B["sym_kind"][i]), int(B["sym_def"][i]), sigs[i]) for i in range(nsym)]
    rep("id->signature", "py list-of-tuples [i][3]", bench(lambda i: tup[i][3], q_syms), deep_size(tup))
    so_, blob = B["sig_offs"], B["sig_blob"]
    rep("id->signature", "numpy offsets + bytes.decode", bench(lambda i: blob[so_[i]:so_[i + 1]].decode(), q_syms),
        so_.nbytes + len(blob))
    so_l = so_.tolist()
    rep("id->signature", "py offsets list + bytes.decode", bench(lambda i: blob[so_l[i]:so_l[i + 1]].decode(), q_syms),
        len(blob) + deep_size(so_l))

    # ---- postings ----
    rep("uses(sym) postings", "py list-of-lists", bench(posts_ll.__getitem__, q_syms), deep_size(posts_ll))
    flat = [(int(B["sym_file"][s]) if False else s, u) for s in range(nsym) for u in posts_ll[s]]
    rep("uses(sym) postings", "py flat (sym,addr) scan", bench(lambda s: [u for k, u in flat if k == s], q_syms[:256]),
        deep_size(flat))
    ps, P = B["post_start"], B["posts"]
    rep("uses(sym) postings", "numpy CSR slice (view)", bench(lambda s: P[ps[s]:ps[s + 1]], q_syms), ps.nbytes + P.nbytes)
    rep("uses(sym) postings", "numpy CSR slice + decode lines", bench(lambda s: (P[ps[s]:ps[s + 1]] >> 12) & 0xFFFF, q_syms),
        ps.nbytes + P.nbytes)

    # ---- live at line L (locals of the innermost function + nothing else) ----
    nfile = int(B["scope_file"].max()) + 1
    file_lines = [int(B["scope_hi"][(B["scope_file"] == f) & (B["scope_kind"] == 0)].max()) for f in range(nfile)]
    q_lines = [(f := RNG.randrange(nfile), RNG.randint(1, file_lines[f])) for _ in range(NQ)]
    sf, slo, shi = B["sym_file"], B["sym_lo"], B["sym_hi"]
    kindarr = B["sym_kind"]
    is_local = (kindarr == 3) | (kindarr == 4)
    py_iv = [(int(sf[i]), int(slo[i]), int(shi[i]), i) for i in range(nsym) if is_local[i]]
    def scan(q):
        f, L = q
        return [i for (ff, lo, hi, i) in py_iv if ff == f and lo <= L <= hi]
    rep("live locals at line L", "py list-of-tuples scan", bench(scan, q_lines[:512]), deep_size(py_iv))
    def np_mask(q):
        f, L = q
        return np.flatnonzero(is_local & (sf == f) & (slo <= L) & (shi >= L))
    rep("live locals at line L", "numpy full mask (1/call)", bench(np_mask, q_lines), sf.nbytes + slo.nbytes + shi.nbytes)
    # scope-indexed: per file, function scopes sorted by start; find innermost containing L
    fscopes = {}
    for sc in range(len(B["scope_kind"])):
        if B["scope_kind"][sc] == 1:
            fscopes.setdefault(int(B["scope_file"][sc]), []).append((int(B["scope_lo"][sc]), int(B["scope_hi"][sc]), sc))
    for v in fscopes.values():
        v.sort()
    fstarts = {f: [x[0] for x in v] for f, v in fscopes.items()}
    ss_start, ss = B["ss_start"].tolist(), B["ss"].tolist()
    slo_l, shi_l, is_local_l = slo.tolist(), shi.tolist(), is_local.tolist()
    def scoped(q):
        f, L = q
        v = fscopes.get(f)
        if not v:
            return []
        j = bisect.bisect_right(fstarts[f], L) - 1
        while j >= 0 and v[j][1] < L:   # walk back to an enclosing (possibly outer) function
            j -= 1
        if j < 0:
            return []
        sc = v[j][2]
        return [i for i in ss[ss_start[sc]:ss_start[sc + 1]] if slo_l[i] <= L <= shi_l[i] and is_local_l[i]]
    # correctness vs brute scan (innermost-scope only can be a subset when nested functions exist)
    agree = sum(set(scoped(q)) <= set(scan(q)) for q in q_lines[:300])
    rep("live locals at line L", "py bisect scope + local filter", bench(scoped, q_lines),
        deep_size(fscopes) + deep_size(fstarts), f"subset-of-brute {agree}/300")

    # ---- nibble-packed local-id stream decode (per function) ----
    nb, ni = B["nib_bytes"], B["nib_index"]
    nb_b = bytes(nb)
    q_fn = [RNG.randrange(len(ni)) for _ in range(NQ)]
    def py_decode(k):
        off, n = ni[k]
        out = []
        for byte in nb_b[off:off + (n + 1) // 2]:
            out.append(byte & 0xF); out.append(byte >> 4)
        return out
    toks_per_fn = float(ni[:, 1].mean())
    t = bench(py_decode, q_fn[:1024])
    rep("nibble decode (per token)", "py loop", t / toks_per_fn, len(nb), f"{toks_per_fn:.0f} tok/fn avg")
    def np_decode(k):
        off, n = ni[k]
        b = nb[off:off + (n + 1) // 2]
        return np.stack([b & 0xF, b >> 4], 1).ravel()
    t = bench(np_decode, q_fn)
    rep("nibble decode (per token)", "numpy unpack (1 fn/call)", t / toks_per_fn, len(nb))
    # 4-bit hierarchical pointer decode: file4|func8|local4 in a u16
    ptrs = [RNG.randrange(65536) for _ in range(NQ)]
    rep("u16 ptr decode file4|fn8|loc4", "py shifts", bench(lambda p: (p >> 12, (p >> 4) & 0xFF, p & 0xF), ptrs), 2)

    # ---- ctypes -> Rust (same data, native implementation) ----
    if so:
        lib = ctypes.CDLL(so)
        lib.bank_open.argtypes = [ctypes.c_char_p]; lib.bank_open.restype = ctypes.c_int
        lib.bank_name_to_id.argtypes = [ctypes.c_char_p, ctypes.c_size_t]; lib.bank_name_to_id.restype = ctypes.c_int64
        lib.bank_uses_count.argtypes = [ctypes.c_uint32]; lib.bank_uses_count.restype = ctypes.c_uint32
        lib.bank_noop.argtypes = []; lib.bank_noop.restype = ctypes.c_int
        assert lib.bank_open(str(Path(sys.argv[1]) / "bank.bin").encode()) == 0
        enc = [n.encode() for n in q_names]
        assert all(lib.bank_name_to_id(e, len(e)) == d_name[n] for e, n in zip(enc[:200], q_names[:200]))
        rep("ctypes overhead", "ctypes noop()", bench(lambda _: lib.bank_noop(), q_syms), 0)
        rep("name->id", "ctypes -> Rust FxHashMap", bench(lambda e: lib.bank_name_to_id(e, len(e)), enc), 0,
            "native part is in bench.rs table")
        rep("uses(sym) postings", "ctypes -> Rust count", bench(lib.bank_uses_count, q_syms), 0)

    Path(sys.argv[1], "py_results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
