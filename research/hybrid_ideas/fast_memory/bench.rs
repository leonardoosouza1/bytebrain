// Native benchmark of symbol-bank lookups.
// Build: nice -n 10 rustc --edition 2021 -O -C target-cpu=native bench.rs -o bench
// Run:   nice -n 10 ./bench bank/bank.bin
#![allow(dead_code)]
include!("bank_core.rs");
use std::hint::black_box;
use std::time::Instant;

const NQ: usize = 4096;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 >> 33 }
    fn below(&mut self, n: usize) -> usize { (self.next() as usize) % n }
}

/// median over 7 runs of ns per query; each run loops the query set enough to last ~20 ms
fn bench<Q, F: FnMut(&Q) -> u64>(qs: &[Q], mut f: F) -> f64 {
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..reps { for q in qs { acc = acc.wrapping_add(f(black_box(q))); } }
        black_box(acc);
        if t.elapsed().as_millis() >= 20 { break; }
        reps *= 2;
    }
    let mut v: Vec<f64> = (0..7).map(|_| {
        let t = Instant::now();
        let mut acc = 0u64;
        for _ in 0..reps { for q in qs { acc = acc.wrapping_add(f(black_box(q))); } }
        black_box(acc);
        t.elapsed().as_nanos() as f64 / (reps * qs.len()) as f64
    }).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[3]
}

fn row(q: &str, imp: &str, ns: f64, bytes: usize, note: &str) {
    println!("{q:30} {imp:40} {ns:9.2} ns  {bytes:>9} B  {note}");
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or("bank/bank.bin".into());
    let b = load(&path);
    let nsym = b.sym_name.len();
    let nname = b.name_offs.len() - 1;
    let mut rng = Lcg(42);
    // uses-weighted query symbols: pick a random posting, map back to its symbol (+ some uniform)
    let q_syms: Vec<u32> = (0..NQ).map(|i| {
        if i % 4 == 0 { rng.below(nsym) as u32 } else {
            let p = rng.below(b.posts.len()) as u32;
            (b.post_start.partition_point(|&s| s <= p) - 1) as u32
        }
    }).collect();
    let q_names: Vec<&str> = q_syms.iter().map(|&s| b.name(b.sym_name[s as usize] as usize)).collect();
    println!("bank: {nsym} symbols, {nname} names, {} scopes, {} postings", b.scope_kind.len(), b.posts.len());

    // ---------- name -> id ----------
    let std_map: HashMap<&str, u32> = (0..nname).map(|i| (b.name(i), i as u32)).collect();
    row("name->id", "std HashMap<&str> (SipHash)", bench(&q_names, |n| std_map[*n] as u64), nname * 24, "");
    let fx = b.name_map();
    row("name->id", "FxHashMap<&str>", bench(&q_names, |n| fx[*n] as u64), nname * 24, "");
    let mut sorted: Vec<(&str, u32)> = (0..nname).map(|i| (b.name(i), i as u32)).collect();
    sorted.sort();
    row("name->id", "sorted Vec + binary search", bench(&q_names, |n| {
        let i = sorted.binary_search_by(|p| p.0.cmp(n)).unwrap(); sorted[i].1 as u64 }), nname * 20, "");
    let lin: Vec<(&str, u32)> = (0..nname).map(|i| (b.name(i), i as u32)).collect();
    row("name->id", "linear scan Vec<(&str,u32)>", bench(&q_names[..512], |n| {
        lin.iter().find(|p| p.0 == *n).unwrap().1 as u64 }), nname * 20, "");

    // scoped resolve: (scope, name_id) -> symbol, walking the scope chain
    let mut key: FxMap<u64, u32> = FxMap::default();
    for s in 0..nsym { key.insert(((b.sym_scope[s] as u64) << 32) | b.sym_name[s] as u64, s as u32); }
    let q_res: Vec<(u32, u32)> = q_syms.iter().map(|&s| {
        // ask from the innermost scope where the symbol is used (or its own scope if unused)
        let sc = b.sym_scope[s as usize];
        (sc, b.sym_name[s as usize])
    }).collect();
    row("resolve(scope,name)->sym", "FxHashMap<u64> + parent walk", bench(&q_res, |&(sc, nid)| {
        let mut s = sc;
        loop {
            if let Some(&v) = key.get(&(((s as u64) << 32) | nid as u64)) { return v as u64; }
            s = b.scope_parent[s as usize];
            if s == u32::MAX { return u64::MAX; }
        }
    }), key.len() * 16, "");

    // ---------- id -> signature / record ----------
    row("id->signature (&str)", "SoA offsets + blob", bench(&q_syms, |&i| {
        let i = i as usize; let s = &b.sig_blob[b.sig_offs[i] as usize..b.sig_offs[i + 1] as usize];
        s.len() as u64 + s.first().copied().unwrap_or(0) as u64 }), 0, "");
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct Rec { name: u32, sig_off: u32, scope: u16, def: u16, lo: u16, hi: u16, kind_file: u8, _pad: [u8; 3] }
    let recs: Vec<Rec> = (0..nsym).map(|i| Rec { name: b.sym_name[i], sig_off: b.sig_offs[i], scope: b.sym_scope[i] as u16,
        def: b.sym_def[i] as u16, lo: b.sym_lo[i] as u16, hi: b.sym_hi[i] as u16,
        kind_file: (b.sym_kind[i] << 4) | b.sym_file[i], _pad: [0; 3] }).collect();
    row("id->full record", "AoS packed 20 B record", bench(&q_syms, |&i| {
        let r = recs[i as usize]; (r.name ^ r.sig_off) as u64 + r.lo as u64 + r.hi as u64 + r.kind_file as u64 }),
        std::mem::size_of::<Rec>(), "B/symbol");
    row("id->full record", "SoA 7 arrays", bench(&q_syms, |&i| {
        let i = i as usize; (b.sym_name[i] ^ b.sig_offs[i]) as u64 + b.sym_lo[i] as u64 + b.sym_hi[i] as u64
            + b.sym_kind[i] as u64 + b.sym_file[i] as u64 }), 4 * 6 + 2, "B/symbol");

    // ---------- postings ----------
    row("uses(sym) -> slice", "CSR slice (len)", bench(&q_syms, |&s| {
        (b.post_start[s as usize + 1] - b.post_start[s as usize]) as u64 }), 4, "B/use");
    let avg_uses = q_syms.iter().map(|&s| (b.post_start[s as usize + 1] - b.post_start[s as usize]) as f64).sum::<f64>() / NQ as f64;
    row("uses(sym) -> decode all", "CSR + addr32 decode (file,line,col)", bench(&q_syms, |&s| {
        let mut a = 0u64;
        for &p in &b.posts[b.post_start[s as usize] as usize..b.post_start[s as usize + 1] as usize] {
            a += (p >> 28) as u64 + ((p >> 12) & 0xFFFF) as u64 + (p & 0xFFF) as u64;
        }
        a }), 4, &format!("avg {avg_uses:.1} uses/query"));

    // ---------- live locals at line L ----------
    let nfile = *b.scope_file.iter().max().unwrap() as usize + 1;
    let flines: Vec<u32> = (0..nfile).map(|f| (0..b.scope_kind.len()).filter(|&s| b.scope_file[s] as usize == f && b.scope_kind[s] == 0)
        .map(|s| b.scope_hi[s]).max().unwrap()).collect();
    let q_lines: Vec<(u8, u32)> = (0..NQ).map(|_| { let f = rng.below(nfile); (f as u8, 1 + rng.below(flines[f] as usize) as u32) }).collect();
    let locals: Vec<usize> = (0..nsym).filter(|&i| b.sym_kind[i] == 3 || b.sym_kind[i] == 4).collect();
    let lf: Vec<u8> = locals.iter().map(|&i| b.sym_file[i]).collect();
    let llo: Vec<u32> = locals.iter().map(|&i| b.sym_lo[i]).collect();
    let lhi: Vec<u32> = locals.iter().map(|&i| b.sym_hi[i]).collect();
    let brute = |&(f, l): &(u8, u32)| -> u64 {
        let mut c = 0u64;
        for k in 0..lf.len() { c += (lf[k] == f && llo[k] <= l && l <= lhi[k]) as u64; }
        c };
    row("live locals at line L", "SoA brute scan (branchless)", bench(&q_lines, brute), 9, &format!("{} locals scanned", lf.len()));
    // scope-indexed: per file sorted function scopes; innermost containing scope then its symbols
    let mut fsc: Vec<Vec<(u32, u32, u32)>> = vec![vec![]; nfile];
    for s in 0..b.scope_kind.len() { if b.scope_kind[s] == 1 { fsc[b.scope_file[s] as usize].push((b.scope_lo[s], b.scope_hi[s], s as u32)); } }
    for v in fsc.iter_mut() { v.sort(); }
    let scoped = |&(f, l): &(u8, u32)| -> u64 {
        let v = &fsc[f as usize];
        let mut j = v.partition_point(|x| x.0 <= l) as isize - 1;
        while j >= 0 && v[j as usize].1 < l { j -= 1; }
        if j < 0 { return 0; }
        let sc = v[j as usize].2 as usize;
        let mut c = 0u64;
        for &i in &b.ss[b.ss_start[sc] as usize..b.ss_start[sc + 1] as usize] {
            let i = i as usize; c += (b.sym_lo[i] <= l && l <= b.sym_hi[i] && (b.sym_kind[i] == 3 || b.sym_kind[i] == 4)) as u64;
        }
        c };
    let agree = q_lines.iter().filter(|q| scoped(q) <= brute(q)).count();
    row("live locals at line L", "scope index (binary search) + filter", bench(&q_lines, scoped), 12,
        &format!("<= brute on {agree}/{NQ}"));
    // precomputed per-line CSR of live locals (memory for speed)
    let mut line_start: Vec<Vec<u32>> = vec![]; let mut line_syms: Vec<u32> = vec![];
    for f in 0..nfile {
        let mut st = vec![0u32; flines[f] as usize + 2];
        let mut buckets: Vec<Vec<u32>> = vec![vec![]; flines[f] as usize + 1];
        for k in 0..lf.len() { if lf[k] as usize == f { for l in llo[k]..=lhi[k].min(flines[f]) { buckets[l as usize].push(locals[k] as u32); } } }
        for l in 0..=flines[f] as usize { st[l] = line_syms.len() as u32; line_syms.extend(&buckets[l]); }
        st[flines[f] as usize + 1] = line_syms.len() as u32;
        line_start.push(st);
    }
    let line_bytes = line_syms.len() * 4 + line_start.iter().map(|v| v.len() * 4).sum::<usize>();
    row("live locals at line L", "precomputed per-line CSR", bench(&q_lines, |&(f, l)| {
        let st = &line_start[f as usize]; (st[l as usize + 1] - st[l as usize]) as u64 }), line_bytes,
        "total bytes of index");

    // ---------- nibble-packed local stream ----------
    let nfn = b.nib_index.len() / 2;
    let q_fn: Vec<usize> = (0..NQ).map(|_| rng.below(nfn)).collect();
    let toks: f64 = q_fn.iter().map(|&k| b.nib_index[2 * k + 1] as f64).sum::<f64>() / NQ as f64;
    let scalar = bench(&q_fn, |&k| {
        let (off, n) = (b.nib_index[2 * k] as usize, b.nib_index[2 * k + 1] as usize);
        let mut a = 0u64;
        for t in 0..n { let byte = b.nib_bytes[off + t / 2]; a += ((byte >> ((t & 1) * 4)) & 0xF) as u64; }
        a });
    row("nibble stream decode", "scalar, per token", scalar / toks, 0, &format!("{toks:.0} tok/fn; 0.5 B/token"));
    // SWAR: count occurrences of local id 3 in the stream, 16 nibbles per u64
    let swar = bench(&q_fn, |&k| {
        let (off, n) = (b.nib_index[2 * k] as usize, b.nib_index[2 * k + 1] as usize);
        let bytes = &b.nib_bytes[off..off + (n + 1) / 2];
        let pat = 0x3333_3333_3333_3333u64; let lo = 0x1111_1111_1111_1111u64;
        let mut c = 0u64;
        let mut ch = bytes.chunks_exact(8);
        for w in &mut ch {
            let x = u64::from_le_bytes(w.try_into().unwrap()) ^ pat; // nibble == 0 where match
            let y = (x | (x >> 1)) | ((x | (x >> 1)) >> 2);          // bit0 of each nibble = any bit set
            c += ((!y) & lo).count_ones() as u64;
        }
        for &by in ch.remainder() { c += ((by & 0xF) == 3) as u64 + ((by >> 4) == 3) as u64; }
        c });
    row("nibble stream count(id==3)", "SWAR 16 nibbles/u64, per token", swar / toks, 0, "");
    // u16 hierarchical pointer file4|fn8|local4, and u32 addr file4|line16|col12
    let ptrs: Vec<u16> = (0..NQ).map(|_| rng.next() as u16).collect();
    row("u16 ptr decode file4|fn8|loc4", "shifts+masks", bench(&ptrs, |&p| ((p >> 12) as u64) + ((p >> 4) & 0xFF) as u64 + (p & 0xF) as u64), 2, "");
    // nibble ptr -> symbol: local slot -> global sym via per-function table (2-level)
    let fn_local_base: Vec<u32> = (0..b.scope_kind.len()).map(|s| b.ss_start[s]).collect();
    let p2: Vec<(u32, u8)> = (0..NQ).map(|_| { let s = rng.below(b.scope_kind.len()); let n = b.ss_start[s + 1] - b.ss_start[s];
        (s as u32, if n == 0 { 0 } else { (rng.below(n as usize) as u8) & 0xF }) }).collect();
    row("(scope, nibble slot) -> sym", "base[scope] + slot -> ss[]", bench(&p2, |&(s, k)| {
        let i = (fn_local_base[s as usize] + k as u32) as usize; b.ss[i.min(b.ss.len() - 1)] as u64 }), 0, "2 dependent loads");

    // ---------- scale test: 1M synthetic names (does it stay fast when it leaves L2?) ----------
    let big: Vec<String> = (0..1_000_000u32).map(|i| format!("sym_{:x}_{}", i.wrapping_mul(2654435761), i % 97)).collect();
    let bigmap: FxMap<&str, u32> = big.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let qb: Vec<&str> = (0..NQ).map(|_| big[rng.below(big.len())].as_str()).collect();
    row("name->id (1M names)", "FxHashMap<&str>", bench(&qb, |n| bigmap[*n] as u64), 0, "out of cache");
    let bigrec: Vec<Rec> = (0..1_000_000).map(|i| recs[i % nsym]).collect();
    let qr: Vec<u32> = (0..NQ).map(|_| rng.below(bigrec.len()) as u32).collect();
    row("id->full record (1M)", "AoS packed 20 B record", bench(&qr, |&i| { let r = bigrec[i as usize]; (r.name ^ r.sig_off) as u64 + r.lo as u64 }), 20, "random access, 20 MB");

    // ---------- latency (dependent chain): next index depends on the previous load ----------
    // Independent queries above overlap in the out-of-order core (throughput). A loop that must
    // wait for the answer before deciding the next step pays the full latency, measured here.
    fn chase(next: &[u32], steps: usize) -> f64 {
        let mut i = 0u32; let t = Instant::now();
        for _ in 0..steps { i = next[i as usize]; }
        black_box(i); t.elapsed().as_nanos() as f64 / steps as f64
    }
    for &n in &[nsym, 1_000_000usize, 16_000_000] {
        // random single-cycle permutation (Sattolo) so the chase visits every slot
        let mut perm: Vec<u32> = (0..n as u32).collect();
        for k in (1..n).rev() { let j = rng.below(k); perm.swap(k, j); }
        let mut next = vec![0u32; n];
        for k in 0..n { next[perm[k] as usize] = perm[(k + 1) % n]; }
        let mut v: Vec<f64> = (0..5).map(|_| chase(&next, 4_000_000)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        row("dependent load chain", &format!("u32 array, {} entries", n), v[2], n * 4, "latency, not throughput");
    }
}
