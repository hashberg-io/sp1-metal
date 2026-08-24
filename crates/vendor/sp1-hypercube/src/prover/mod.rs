// Modified from sp1-hypercube 6.4.0 by the sp1-metal project; see CHANGES.md.
//! Prover components.

mod cpu;
mod memory_permit;
#[cfg(feature = "metal")]
mod metal_tcs;
mod permits;
mod shard;
mod simple;
mod trace;
pub mod zerocheck;

pub use cpu::*;
pub use memory_permit::*;
pub use permits::*;
pub use shard::*;
pub use simple::*;
use slop_merkle_tree::BnProver;
use slop_stacked::StackedPcsProver;
use sp1_primitives::{SP1ExtensionField, SP1Field, SP1GlobalContext, SP1OuterGlobalContext};
pub use trace::*;
pub use zerocheck::*;

#[cfg(feature = "metal")]
pub use metal_tcs::{MetalMerkleProver, COMMIT_STATS};

#[cfg(not(feature = "metal"))]
pub use slop_merkle_tree::Poseidon2KoalaBear16Prover as SP1MerkleTreeProver;

/// Upstream's committer with its Merkle tree built on the GPU.
#[cfg(feature = "metal")]
pub type SP1MerkleTreeProver = MetalMerkleProver;

#[allow(clippy::disallowed_types)]
/// The CPU prover components for a jagged PCS prover in SP1.
pub type SP1InnerPcsProver = StackedPcsProver<SP1MerkleTreeProver, SP1GlobalContext>;

/// The Jagged CPU prover components for the wrap step of recursion in SP1.
pub type SP1OuterPcsProver =
    StackedPcsProver<BnProver<SP1Field, SP1ExtensionField>, SP1OuterGlobalContext>;
