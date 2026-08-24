#![no_main]
sp1_zkvm::entrypoint!(main);
/// Control-flow shaped work: a branch whose direction the hardware cannot
/// predict, so the branch chip carries the trace rather than the ALU chips.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut x: u32 = 0x1234_5678;
    let mut a: u32 = 0;
    let mut b: u32 = 0;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        if x & 1 == 0 {
            a = a.wrapping_add(x);
        } else if x & 2 == 0 {
            b = b.wrapping_sub(x);
        } else if x & 4 == 0 {
            a ^= b;
        } else {
            b = b.rotate_left(3);
        }
    }
    sp1_zkvm::io::commit(&a.wrapping_add(b));
}
