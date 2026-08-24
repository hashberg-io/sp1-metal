#![no_main]
sp1_zkvm::entrypoint!(main);

/// Bitwise-shaped work: xorshift and rotates, which load the bitwise and shift
/// chips rather than the multiply ones.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut x: u32 = 0x1234_5678;
    let mut acc: u32 = 0;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x = x.rotate_left(7);
        acc = acc.wrapping_add(x & 0xFFFF);
    }
    sp1_zkvm::io::commit(&acc);
}
