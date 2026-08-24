#![no_main]
sp1_zkvm::entrypoint!(main);
use sp1_zkvm::syscalls::syscall_keccak_permute;

/// The Keccak-f[1600] permutation, driven directly. The widest precompile
/// chip in common use, and the EVM's hash.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut st = [0u64; 25];
    for i in 0..n {
        st[0] ^= i as u64;
        unsafe { syscall_keccak_permute(&mut st) };
    }
    sp1_zkvm::io::commit(&(st[0] as u32));
}
