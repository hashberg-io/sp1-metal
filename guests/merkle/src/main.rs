#![no_main]
sp1_zkvm::entrypoint!(main);
extern crate alloc;
use alloc::vec::Vec;
use sp1_zkvm::syscalls::{syscall_sha256_compress, syscall_sha256_extend};

/// A Merkle tree over n leaves, hashed with the SHA-256 precompile.
///
/// A realistic composite: precompile chips carrying most of the work, with
/// core chips doing the addressing around them.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let leaves = (n as usize).min(1 << 18).max(2).next_power_of_two();
    let mut level: Vec<[u64; 8]> = (0..leaves)
        .map(|i| {
            let mut h = IV;
            h[0] ^= i as u64;
            h
        })
        .collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks(2) {
            next.push(compress_pair(&pair[0], &pair[1]));
        }
        level = next;
    }
    sp1_zkvm::io::commit(&(level[0][0] as u32));
}

const IV: [u64; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn compress_pair(l: &[u64; 8], r: &[u64; 8]) -> [u64; 8] {
    let mut w = [0u64; 64];
    w[..8].copy_from_slice(l);
    w[8..16].copy_from_slice(r);
    let mut state = IV;
    unsafe {
        syscall_sha256_extend(&mut w);
        syscall_sha256_compress(&mut w, &mut state);
    }
    state
}
