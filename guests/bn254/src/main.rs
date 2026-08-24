#![no_main]
sp1_zkvm::entrypoint!(main);
use sp1_zkvm::syscalls::syscall_bn254_double;

/// Repeated doubling of the BN254 G1 generator, which is (1, 2).
///
/// Doubling for the same reason as the secp256k1 guest: it is total, so no
/// intermediate can violate a precondition.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut p: [u64; 8] = [1, 0, 0, 0, 2, 0, 0, 0];
    for _ in 0..n {
        unsafe { syscall_bn254_double(&mut p) };
    }
    sp1_zkvm::io::commit(&(p[0] as u32));
}
