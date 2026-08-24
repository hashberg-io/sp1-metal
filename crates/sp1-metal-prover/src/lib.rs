// SPDX-License-Identifier: MIT OR Apache-2.0
//! Harness for the fork's Metal GKR round.
//!
//! The implementation itself lives in `sp1-hypercube` behind its `metal`
//! feature, because that is the only place `DefaultRoundOps` can point at it —
//! which is what lets an existing prover run its GKR rounds on the GPU without
//! a type changing anywhere downstream. This crate exists to test and measure
//! it from outside.

pub use sp1_hypercube::metal::{MetalRoundOps, STATS};

/// The base field the kernels are written for.
pub type F = slop_koala_bear::KoalaBear;
/// Its degree-4 extension, which is what a sumcheck round carries.
pub type EF = slop_algebra::extension::BinomialExtensionField<F, 4>;
