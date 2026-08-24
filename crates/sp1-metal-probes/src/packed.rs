// SPDX-License-Identifier: MIT OR Apache-2.0
//! The vocabulary the packed CPU baselines are written in.
//!
//! The SP1 CPU prover does its field arithmetic through the packed field types
//! (`KoalaBear::Packing`, which is the NEON type on aarch64, and the matching
//! packed extension), not through scalar `KoalaBear`. A GPU kernel that only
//! beats scalar CPU code has not beaten anything, so every CPU baseline in the
//! probe suite that is allowed to set a ratio is written against these types.
//!
//! Packing is a layout choice, and layouts are the prover's to pick: the
//! helpers below transpose into packed form *outside* the timed region, which
//! is what a CPU backend built to compete would do once, up front.

use slop_algebra::{AbstractExtensionField, AbstractField, ExtensionField, Field, PackedValue};

use crate::kb::{EF, F};

/// Packed base field: NEON on aarch64, scalar elsewhere.
pub type PF = <F as Field>::Packing;

/// Packed degree-4 extension: p3's own `BinomialExtensionField<PF, 4>`.
pub type PEF = <EF as ExtensionField<F>>::ExtensionPacking;

/// SIMD lanes per packed element.
pub const LANES: usize = PF::WIDTH;

/// One packed extension element holding `LANES` consecutive elements of `xs`.
fn pack_ext_at(xs: &[EF], base: usize) -> PEF {
    PEF::from_base_fn(|j| {
        PF::from_fn(|lane| {
            let c: &[F] = xs[base + lane].as_base_slice();
            c[j]
        })
    })
}

/// Broadcasts one extension element across all lanes.
pub fn broadcast_ext(x: EF) -> PEF {
    let c: &[F] = x.as_base_slice();
    PEF::from_base_fn(|j| PF::from(c[j]))
}

/// Transposes `xs` into packed extension elements.
pub fn pack_ext(xs: &[EF]) -> Vec<PEF> {
    assert_eq!(xs.len() % LANES, 0, "packed layout needs a length divisible by the SIMD width");
    (0..xs.len() / LANES).map(|g| pack_ext_at(xs, g * LANES)).collect()
}

/// Splits `xs` into the even- and odd-indexed halves a fold consumes, each in
/// packed layout. The interleaved pairs a fold walks are a strided access in
/// scalar layout; a packed CPU prover pays for the deinterleave once.
pub fn split_pack_base(xs: &[F]) -> (Vec<PF>, Vec<PF>) {
    let n_out = xs.len() / 2;
    assert_eq!(n_out % LANES, 0, "packed layout needs a length divisible by the SIMD width");
    let groups = n_out / LANES;
    let evens = (0..groups)
        .map(|g| PF::from_fn(|lane| xs[2 * (g * LANES + lane)]))
        .collect();
    let odds = (0..groups)
        .map(|g| PF::from_fn(|lane| xs[2 * (g * LANES + lane) + 1]))
        .collect();
    (evens, odds)
}

/// The extension counterpart of [`split_pack_base`].
pub fn split_pack_ext(xs: &[EF]) -> (Vec<PEF>, Vec<PEF>) {
    let n_out = xs.len() / 2;
    assert_eq!(n_out % LANES, 0, "packed layout needs a length divisible by the SIMD width");
    let groups = n_out / LANES;
    let evens = (0..groups)
        .map(|g| PEF::from_base_fn(|j| PF::from_fn(|lane| {
            let c: &[F] = xs[2 * (g * LANES + lane)].as_base_slice();
            c[j]
        })))
        .collect();
    let odds = (0..groups)
        .map(|g| PEF::from_base_fn(|j| PF::from_fn(|lane| {
            let c: &[F] = xs[2 * (g * LANES + lane) + 1].as_base_slice();
            c[j]
        })))
        .collect();
    (evens, odds)
}

/// Sums a packed extension element across its lanes.
pub fn sum_lanes(p: &PEF) -> EF {
    let coeffs: &[PF] = p.as_base_slice();
    let mut total = EF::zero();
    for lane in 0..LANES {
        let c: [F; 4] = std::array::from_fn(|j| coeffs[j].as_slice()[lane]);
        total += <EF as AbstractExtensionField<F>>::from_base_slice(&c);
    }
    total
}

/// Flattens packed base elements back to scalar order.
pub fn unpack_base(xs: &[PF]) -> Vec<F> {
    xs.iter().flat_map(|p| p.as_slice().iter().copied()).collect()
}

/// Flattens packed extension elements back to scalar order.
pub fn unpack_ext(xs: &[PEF]) -> Vec<EF> {
    let mut out = Vec::with_capacity(xs.len() * LANES);
    for p in xs {
        let coeffs: &[PF] = p.as_base_slice();
        for lane in 0..LANES {
            let c: [F; 4] = std::array::from_fn(|j| coeffs[j].as_slice()[lane]);
            out.push(<EF as AbstractExtensionField<F>>::from_base_slice(&c));
        }
    }
    out
}
