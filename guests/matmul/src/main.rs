#![no_main]
sp1_zkvm::entrypoint!(main);
extern crate alloc;
use alloc::vec;

/// Dense matrix multiplication over u32: the most regular arithmetic
/// available, and an upper bound on how uniform a chip mix can get.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    // n is the dimension directly: work is d^3, so a size knob that meant
    // iterations would be uninterpretable next to the linear guests.
    let d = (n as usize).min(512).max(2);
    let a: alloc::vec::Vec<u32> = (0..d * d).map(|i| (i as u32).wrapping_mul(2654435761)).collect();
    let b: alloc::vec::Vec<u32> = (0..d * d).map(|i| (i as u32).wrapping_mul(40503)).collect();
    let mut c = vec![0u32; d * d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i * d + k];
            for j in 0..d {
                c[i * d + j] = c[i * d + j].wrapping_add(aik.wrapping_mul(b[k * d + j]));
            }
        }
    }
    sp1_zkvm::io::commit(&c[0]);
}
