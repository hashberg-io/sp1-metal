#![no_main]
sp1_zkvm::entrypoint!(main);
use sp1_zkvm::syscalls::syscall_secp256k1_double;

/// Repeated doubling of the secp256k1 generator.
///
/// Doubling is chosen over addition because it is total on a prime-order
/// curve: every intermediate stays a valid affine point and none reaches
/// infinity, so the guest cannot wander into a precompile precondition it
/// fails to meet. These are the chips whose constraint DAG exceeds the
/// zerocheck node bound, which is the whole reason this guest exists.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    // The generator, as little-endian 64-bit limbs: x then y.
    let mut p: [u64; 8] = [
        0x59F2815B16F81798, 0x029BFCDB2DCE28D9, 0x55A06295CE870B07, 0x79BE667EF9DCBBAC,
        0x9C47D08FFB10D4B8, 0xFD17B448A6855419, 0x5DA4FBFC0E1108A8, 0x483ADA7726A3C465,
    ];
    for _ in 0..n {
        unsafe { syscall_secp256k1_double(&mut p) };
    }
    sp1_zkvm::io::commit(&(p[0] as u32));
}
