// Modified from slop-jagged 6.4.0 by the sp1-metal project; see CHANGES.md.
#![allow(clippy::disallowed_types)]
#[cfg(feature = "metal")]
pub mod metal;

mod basefold;
mod hadamard;
mod jagged_eval;
mod long;
mod poly;
mod populate;
mod prover;
mod sumcheck;
mod verifier;

pub use basefold::*;
pub use hadamard::*;
pub use jagged_eval::*;
pub use long::*;
pub use poly::*;
pub use prover::*;
pub use verifier::*;
