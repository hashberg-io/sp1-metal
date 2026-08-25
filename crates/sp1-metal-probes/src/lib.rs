// SPDX-License-Identifier: MIT OR Apache-2.0
//! Probe suite for an SP1 Metal proving backend on Apple Silicon.
//!
//! Every probe has two faces: a parity check against the slop CPU
//! implementations (small inputs, runs in `cargo test`), and a throughput
//! benchmark (large inputs, run explicitly via the `probes` binary).

pub mod backend;
pub mod commit;
pub mod gpu;
pub mod kb;
pub mod packed;
pub mod probes;

/// Re-exported from [`sp1_metal_core`], which sits below the vendored slop
/// crates so that they and this crate reach the same counters.
pub use sp1_metal_core::{compile, handoff, nocopy, queue, stats, tally};
