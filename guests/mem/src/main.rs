#![no_main]
sp1_zkvm::entrypoint!(main);

/// Memory-shaped work: pseudorandom scatter and gather over a resident array,
/// which loads the memory chips rather than the arithmetic ones.
fn main() {
    let n: u32 = sp1_zkvm::io::read();
    const SIZE: usize = 1 << 14;
    let mut v: Vec<u32> = (0..SIZE as u32).collect();
    let mut idx: u32 = 1;
    for _ in 0..n {
        idx = idx.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let i = (idx as usize) & (SIZE - 1);
        let j = (idx.rotate_left(13) as usize) & (SIZE - 1);
        let t = v[i];
        v[i] = v[j] ^ t;
        v[j] = t.wrapping_add(1);
    }
    sp1_zkvm::io::commit(&v[0]);
}
