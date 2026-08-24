#![no_main]
sp1_zkvm::entrypoint!(main);
use sp1_zkvm::syscalls::syscall_uint256_mulmod;

/// Big-integer multiplication without a curve: the Uint256 precompile chip.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut x: [u64; 4] = [0x9E3779B97F4A7C15, 3, 5, 7];
    let y: [u64; 4] = [0x0123456789ABCDEF, 11, 13, 17];
    for _ in 0..n {
        unsafe { syscall_uint256_mulmod(&mut x, &y) };
    }
    sp1_zkvm::io::commit(&(x[0] as u32));
}
