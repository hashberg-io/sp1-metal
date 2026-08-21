//! CPU mirror of the KoalaBear Montgomery arithmetic used by the Metal
//! kernels, plus conversions between raw Montgomery words and slop's field
//! types. The slop types are the parity reference; the raw mirror exists only
//! to move data in and out of GPU buffers.

use rand::Rng;
use slop_algebra::extension::BinomialExtensionField;
use slop_algebra::{AbstractExtensionField, AbstractField, PrimeField32};
use slop_koala_bear::KoalaBear;

pub type F = KoalaBear;
pub type EF = BinomialExtensionField<KoalaBear, 4>;

pub const P: u32 = 0x7f00_0001;
pub const MU: u32 = 0x8100_0001;

pub fn to_monty(x: u32) -> u32 {
    (((x as u64) << 32) % P as u64) as u32
}

pub fn from_monty(x: u32) -> u32 {
    redc(x as u64)
}

pub fn redc(x: u64) -> u32 {
    let t = (x as u32).wrapping_mul(MU);
    let u = (t as u64) * (P as u64);
    let (d, over) = x.overflowing_sub(u);
    ((d >> 32) as u32).wrapping_add(if over { P } else { 0 })
}

pub fn monty_of_f(x: F) -> u32 {
    to_monty(x.as_canonical_u32())
}

pub fn f_of_monty(m: u32) -> F {
    F::from_canonical_u32(from_monty(m))
}

pub fn monty4_of_ef(x: EF) -> [u32; 4] {
    let base = x.as_base_slice();
    [monty_of_f(base[0]), monty_of_f(base[1]), monty_of_f(base[2]), monty_of_f(base[3])]
}

pub fn ef_of_monty4(m: [u32; 4]) -> EF {
    EF::from_base_slice(&[f_of_monty(m[0]), f_of_monty(m[1]), f_of_monty(m[2]), f_of_monty(m[3])])
}

pub fn rand_f_vec<R: Rng>(rng: &mut R, n: usize) -> Vec<F> {
    (0..n).map(|_| rng.gen::<F>()).collect()
}

pub fn rand_ef_vec<R: Rng>(rng: &mut R, n: usize) -> Vec<EF> {
    (0..n).map(|_| rng.gen::<EF>()).collect()
}

pub fn monty_vec_of_f(xs: &[F]) -> Vec<u32> {
    xs.iter().map(|&x| monty_of_f(x)).collect()
}

pub fn monty4_vec_of_ef(xs: &[EF]) -> Vec<[u32; 4]> {
    xs.iter().map(|&x| monty4_of_ef(x)).collect()
}
