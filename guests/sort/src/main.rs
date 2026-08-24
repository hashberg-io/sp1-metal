#![no_main]
sp1_zkvm::entrypoint!(main);
extern crate alloc;
use alloc::vec::Vec;

/// Sorting, which is memory-and-compare shaped and a common member of every
/// zkVM benchmark suite.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let len = (n as usize).min(1 << 20).max(2);
    let mut v: Vec<u32> = Vec::with_capacity(len);
    let mut x: u32 = 0xDEADBEEF;
    for _ in 0..len {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        v.push(x);
    }
    v.sort_unstable();
    sp1_zkvm::io::commit(&(v[0] ^ v[len - 1]));
}
