//! The tensor commitment, with its expensive half on the GPU.
//!
//! Committing a shard's traces is the largest single thing the prover does —
//! 47.9 thread-seconds of a 52.8 second proof, measured — and nearly all of it
//! is one Merkle tree over the encoded rows. The tree is reachable without
//! touching upstream: `BasefoldProver` takes its committer as a type parameter,
//! so aliasing [`super::SP1MerkleTreeProver`] to this is the whole of the diff.
//!
//! Openings stay on the CPU. `prove_openings_at_indices` walks the digest
//! layers and `compute_openings_at_indices` copies rows; neither is worth a
//! dispatch, and delegating them keeps the prover data exactly the type
//! upstream builds, so the rest of the proof cannot tell the difference.
//!
//! Set `SP1_METAL_MERKLE=0` to force the CPU committer through the same binary,
//! which is how the pair of numbers either side of this is taken.

use std::borrow::Borrow;
use std::convert::Infallible;
use std::mem::ManuallyDrop;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use slop_algebra::AbstractField;
use slop_alloc::CpuBackend;
use slop_challenger::IopCtx;
use slop_commit::Message;
use slop_futures::OwnedBorrow;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_merkle_tree::{
    ComputeTcsOpenings, FieldMerkleTreeDigests, MerkleTreeTcs, MerkleTreeTcsProof,
    Poseidon2KoalaBear16Prover, TensorCsProver,
};
use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use slop_tensor::Tensor;
use sp1_primitives::{SP1Field, SP1GlobalContext};

type Digest = <SP1GlobalContext as IopCtx>::Digest;
type Digests = FieldMerkleTreeDigests<SP1Field, 8>;
type Data = (Digests, Digest, usize, usize);

/// What committing costs and how much of it the GPU took.
#[derive(Clone, Copy, Default)]
struct CommitStatsInner {
    commits: usize,
    on_gpu: usize,
    seconds: f64,
    rows: u64,
}

pub struct CommitStats {
    inner: Mutex<CommitStatsInner>,
}

impl CommitStats {
    fn record(&self, secs: f64, rows: usize, on_gpu: bool) {
        let mut s = self.inner.lock().expect("stats poisoned");
        s.commits += 1;
        s.on_gpu += usize::from(on_gpu);
        s.seconds += secs;
        s.rows += rows as u64;
    }

    /// `(commitments, of which on the GPU, thread-seconds, rows)`.
    #[must_use]
    pub fn counts(&self) -> (usize, usize, f64, u64) {
        let s = *self.inner.lock().expect("stats poisoned");
        (s.commits, s.on_gpu, s.seconds, s.rows)
    }
}

/// Time spent committing tensors, across every thread.
pub static COMMIT_STATS: CommitStats = CommitStats {
    inner: Mutex::new(CommitStatsInner { commits: 0, on_gpu: 0, seconds: 0.0, rows: 0 }),
};

/// Whether the GPU committer is in use, read once.
fn use_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SP1_METAL_MERKLE").map(|v| v != "0").unwrap_or(true)
            && sp1_metal_probes::commit::available()
    })
}

/// Commits tensors with the Merkle tree built on the GPU.
#[derive(Clone)]
pub struct MetalMerkleProver {
    inner: Poseidon2KoalaBear16Prover,
    tcs: MerkleTreeTcs<SP1GlobalContext>,
}

impl std::fmt::Debug for MetalMerkleProver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalMerkleProver")
    }
}

impl Default for MetalMerkleProver {
    fn default() -> Self {
        let (hasher, compressor) = KoalaBearDegree4Duplex::default_hasher_and_compressor();
        let tcs = MerkleTreeTcs { hasher, compressor };
        Self { inner: Poseidon2KoalaBear16Prover::new(tcs.clone()), tcs }
    }
}

/// Reinterprets Montgomery words as field elements.
///
/// `SP1Field` is `repr(transparent)` over the Montgomery word the kernels write,
/// so this is the same allocation under a different name, not a conversion.
fn as_field(layer: Vec<[u32; 8]>) -> Vec<[SP1Field; 8]> {
    let mut layer = ManuallyDrop::new(layer);
    // SAFETY: identical size and alignment, and the Vec is not dropped here.
    unsafe {
        Vec::from_raw_parts(
            layer.as_mut_ptr().cast::<[SP1Field; 8]>(),
            layer.len(),
            layer.capacity(),
        )
    }
}

impl TensorCsProver<SP1GlobalContext, CpuBackend> for MetalMerkleProver {
    type ProverError = Infallible;
    type ProverData = Data;

    fn commit_tensors<T>(
        &self,
        tensors: Message<T>,
    ) -> Result<(Digest, Self::ProverData), Self::ProverError>
    where
        T: OwnedBorrow<Tensor<SP1Field, CpuBackend>>,
    {
        let start = Instant::now();
        let borrowed: Vec<&Tensor<SP1Field, CpuBackend>> = tensors
            .iter()
            .map(|t| {
                let t: &T = t.borrow();
                t.borrow()
            })
            .collect();
        let rows = borrowed.first().map_or(0, |t| t.sizes()[0]);
        let uniform = borrowed.iter().all(|t| t.sizes()[0] == rows);
        let total_width: usize = borrowed.iter().map(|t| t.sizes()[1]).sum();

        if use_gpu() && uniform {
            let words: Vec<(&[u32], usize)> = borrowed
                .iter()
                .map(|t| {
                    let s = t.as_slice();
                    // SAFETY: SP1Field is repr(transparent) over its Montgomery word.
                    let w = unsafe {
                        std::slice::from_raw_parts(s.as_ptr().cast::<u32>(), s.len())
                    };
                    (w, t.sizes()[1])
                })
                .collect();
            if let Some(layers) = sp1_metal_probes::commit::digest_layers(&words, rows) {
                let layers: Vec<Vec<[SP1Field; 8]>> =
                    layers.into_iter().map(as_field).collect();
                let root = layers.last().expect("a tree has a root")[0];
                let log_height = rows.next_power_of_two().ilog2() as usize;
                // Two permutations, left with slop's own hasher so the published
                // commitment cannot differ from the one it would publish.
                let shape = self.tcs.hasher.hash_iter([
                    SP1Field::from_canonical_usize(log_height),
                    SP1Field::from_canonical_usize(total_width),
                ]);
                let commitment = self.tcs.compressor.compress([root, shape]);
                let digests = Digests { digest_layers: std::sync::Arc::new(layers) };
                COMMIT_STATS.record(start.elapsed().as_secs_f64(), rows, true);
                return Ok((commitment, (digests, root, log_height, total_width)));
            }
        }
        drop(borrowed);
        let out = self.inner.commit_tensors(tensors);
        COMMIT_STATS.record(start.elapsed().as_secs_f64(), rows, false);
        out
    }

    fn prove_openings_at_indices(
        &self,
        data: Self::ProverData,
        indices: &[usize],
    ) -> Result<MerkleTreeTcsProof<Digest>, Self::ProverError> {
        self.inner.prove_openings_at_indices(data, indices)
    }
}

impl ComputeTcsOpenings<SP1GlobalContext, CpuBackend> for MetalMerkleProver {
    fn compute_openings_at_indices<T>(
        &self,
        tensors: Message<T>,
        indices: &[usize],
    ) -> Tensor<SP1Field, CpuBackend>
    where
        T: OwnedBorrow<Tensor<SP1Field, CpuBackend>>,
    {
        self.inner.compute_openings_at_indices(tensors, indices)
    }
}
