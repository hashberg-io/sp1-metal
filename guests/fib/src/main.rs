#![no_main]
sp1_zkvm::entrypoint!(main);

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c = (a + b) % 998_244_353;
        a = b;
        b = c;
    }
    sp1_zkvm::io::commit(&a);
}
