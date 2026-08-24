#![no_main]
sp1_zkvm::entrypoint!(main);
/// Widening multiplication, which loads the MUL chip's high-half path rather
/// than the low-half one an ordinary multiply uses.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut x: u64 = 0x0123_4567_89AB_CDEF;
    let mut acc: u64 = 0;
    for _ in 0..n {
        let hi = ((x as u128 * 0x9E37_79B9_7F4A_7C15u128) >> 64) as u64;
        acc = acc.wrapping_add(hi);
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    }
    sp1_zkvm::io::commit(&acc);
}
