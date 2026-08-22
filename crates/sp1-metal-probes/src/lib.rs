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
pub mod stats;
