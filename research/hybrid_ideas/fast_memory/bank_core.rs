// Shared loader for bank.bin (included by bench.rs and ffi.rs via include!).
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

// FxHash (rustc's hasher): one multiply per word, no DoS protection, very fast for short keys.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);
impl Hasher for Fx {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        let mut c = bytes.chunks_exact(8);
        for w in &mut c {
            h = (h.rotate_left(5) ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
        for &b in c.remainder() {
            h = (h.rotate_left(5) ^ b as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
        self.0 = h;
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn finish(&self) -> u64 { self.0 }
}
pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

pub struct Bank {
    pub name_offs: Vec<u32>, pub name_blob: Vec<u8>,
    pub sym_name: Vec<u32>, pub sym_scope: Vec<u32>, pub sym_kind: Vec<u8>, pub sym_file: Vec<u8>,
    pub sym_def: Vec<u32>, pub sym_lo: Vec<u32>, pub sym_hi: Vec<u32>,
    pub sig_offs: Vec<u32>, pub sig_blob: Vec<u8>,
    pub post_start: Vec<u32>, pub posts: Vec<u32>,
    pub scope_file: Vec<u8>, pub scope_kind: Vec<u8>, pub scope_lo: Vec<u32>, pub scope_hi: Vec<u32>,
    pub scope_parent: Vec<u32>, pub ss_start: Vec<u32>, pub ss: Vec<u32>,
    pub nib_index: Vec<u32>, pub nib_bytes: Vec<u8>,
}

fn u32s(b: &[u8]) -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect() }

pub fn load(path: &str) -> Bank {
    let raw = std::fs::read(path).expect("read bank.bin");
    let rd32 = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
    let n = rd32(0);
    let mut off = 4;
    let mut sec: HashMap<String, Vec<u8>> = HashMap::new();
    for _ in 0..n {
        let ln = rd32(off); off += 4;
        let name = String::from_utf8(raw[off..off + ln].to_vec()).unwrap(); off += ln;
        let dl = u64::from_le_bytes(raw[off..off + 8].try_into().unwrap()) as usize; off += 8;
        sec.insert(name, raw[off..off + dl].to_vec()); off += dl;
    }
    let mut t = |k: &str| sec.remove(k).unwrap_or_else(|| panic!("missing {k}"));
    Bank {
        name_offs: u32s(&t("name_offs")), name_blob: t("name_blob"),
        sym_name: u32s(&t("sym_name")), sym_scope: u32s(&t("sym_scope")), sym_kind: t("sym_kind"),
        sym_file: t("sym_file"), sym_def: u32s(&t("sym_def")), sym_lo: u32s(&t("sym_lo")), sym_hi: u32s(&t("sym_hi")),
        sig_offs: u32s(&t("sig_offs")), sig_blob: t("sig_blob"),
        post_start: u32s(&t("post_start")), posts: u32s(&t("posts")),
        scope_file: t("scope_file"), scope_kind: t("scope_kind"), scope_lo: u32s(&t("scope_lo")),
        scope_hi: u32s(&t("scope_hi")), scope_parent: u32s(&t("scope_parent")),
        ss_start: u32s(&t("scope_sym_start")), ss: u32s(&t("scope_syms")),
        nib_index: u32s(&t("nib_index")), nib_bytes: t("nib_bytes"),
    }
}

impl Bank {
    pub fn name(&self, i: usize) -> &str {
        std::str::from_utf8(&self.name_blob[self.name_offs[i] as usize..self.name_offs[i + 1] as usize]).unwrap()
    }
    pub fn name_map(&self) -> FxMap<&str, u32> {
        (0..self.name_offs.len() - 1).map(|i| (self.name(i), i as u32)).collect()
    }
}
