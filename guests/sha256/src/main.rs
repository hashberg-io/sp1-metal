#![no_main]
sp1_zkvm::entrypoint!(main);
use sp1_zkvm::syscalls::{syscall_sha256_compress, syscall_sha256_extend};

/// The SHA-256 precompiles, driven directly: n rounds of extend + compress,
/// which is exactly the ShaExtend and ShaCompress chips and nothing else.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut state: [u64; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut w = [0u64; 64];
    for i in 0..n {
        for (j, slot) in w.iter_mut().enumerate().take(16) {
            *slot = ((i as u64).wrapping_mul(0x9E3779B9) ^ (j as u64)) & 0xFFFF_FFFF;
        }
        unsafe {
            syscall_sha256_extend(&mut w);
            syscall_sha256_compress(&mut w, &mut state);
        }
    }
    sp1_zkvm::io::commit(&(state[0] as u32));
}
