// cdylib exposing a few bank lookups to Python via ctypes (to measure FFI overhead).
// Build: nice -n 10 rustc --edition 2021 -O --crate-type cdylib ffi.rs -o libbank_ffi.so
#![allow(dead_code)]
include!("bank_core.rs");
use std::sync::OnceLock;

struct State { bank: &'static Bank, names: FxMap<&'static str, u32> }
static STATE: OnceLock<State> = OnceLock::new();

#[no_mangle]
pub extern "C" fn bank_open(path: *const std::os::raw::c_char) -> i32 {
    let p = unsafe { std::ffi::CStr::from_ptr(path) }.to_str().unwrap();
    let bank: &'static Bank = Box::leak(Box::new(load(p)));
    let names = bank.name_map();
    STATE.set(State { bank, names }).ok();
    0
}

#[no_mangle]
pub extern "C" fn bank_noop() -> i32 { 0 }

#[no_mangle]
pub extern "C" fn bank_name_to_id(ptr: *const u8, len: usize) -> i64 {
    let s = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) };
    STATE.get().unwrap().names.get(s).map(|&v| v as i64).unwrap_or(-1)
}

#[no_mangle]
pub extern "C" fn bank_uses_count(sym: u32) -> u32 {
    let b = STATE.get().unwrap().bank;
    b.post_start[sym as usize + 1] - b.post_start[sym as usize]
}
