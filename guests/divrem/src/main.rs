#![no_main]
sp1_zkvm::entrypoint!(main);
/// Division-shaped work. DIVREM's constraint DAG is unlike add or shift, so
/// the kernel generated for it is unlike any other measured here.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut x: u32 = 0x9E37_79B9;
    let mut acc: u32 = 0;
    for _ in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let d = (x | 1) >> 3 | 1;
        acc = acc.wrapping_add(x / d).wrapping_add(x % d);
    }
    sp1_zkvm::io::commit(&acc);
}
