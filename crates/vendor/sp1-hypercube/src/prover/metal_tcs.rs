// SPDX-License-Identifier: MIT OR Apache-2.0
// Added to sp1-hypercube 6.4.0 by the sp1-metal project; see CHANGES.md.
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
//! which is how the pair of numbers either side of this is taken, and
//! `SP1_METAL_MERKLE=check` to build both trees and compare every digest of
//! every layer.
//!
//! The checking mode was added because this was the only accelerated phase
//! without one, which made it the one place a fully checked run could not
//! look — and an intermittent wrong proof reported by a consumer
//! (`knowledge/defect-wrong-proof.md`) was a Merkle root mismatch that had
//! survived exactly such a run. Comparing whole layers rather than the root
//! is the point: the root says only that something is wrong, while the lowest
//! layer that disagrees says whether the tree was built wrongly or built
//! correctly over the wrong bytes.

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

/// How the tree is switched, read once.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Cpu,
    Gpu,
    /// Build both trees and compare them, which is slower than either.
    Check,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if !sp1_metal_probes::commit::available() {
            return Mode::Cpu;
        }
        match std::env::var("SP1_METAL_MERKLE").as_deref() {
            Ok("0") => Mode::Cpu,
            Ok("check") => Mode::Check,
            _ => Mode::Gpu,
        }
    })
}

/// Whether the GPU committer is in use, read once.
fn use_gpu() -> bool {
    mode() != Mode::Cpu
}

/// Ends the hand-off window when a commitment returns, however it returns.
///
/// The encode leaves its codeword's buffer for the Merkle kernels to take
/// rather than mapping the largest allocation in the proof twice, and this is
/// the far end of that: past here nothing can want one, and anything still
/// waiting is keyed on a pointer whose allocation is about to be dropped. See
/// `sp1_metal_core::handoff` for why an entry that outlives its allocation is
/// worse than a slow commitment.
struct EndOfHandoff;

impl Drop for EndOfHandoff {
    fn drop(&mut self) {
        sp1_metal_probes::handoff::clear();
    }
}

/// Where two trees over the same tensors first disagree.
///
/// Reported as a layer and an index rather than as a root, because which layer
/// it is answers a different question in each case: layer zero is the leaves,
/// so a disagreement there is about the bytes that were hashed, and any layer
/// above it with agreeing children is about the compression.
fn first_disagreement(
    gpu: &[Vec<[SP1Field; 8]>],
    cpu: &[Vec<[SP1Field; 8]>],
) -> Option<String> {
    if gpu.len() != cpu.len() {
        return Some(format!("{} layers against {}", gpu.len(), cpu.len()));
    }
    for (l, (a, b)) in gpu.iter().zip(cpu).enumerate() {
        if a.len() != b.len() {
            return Some(format!("layer {l} has {} digests against {}", a.len(), b.len()));
        }
        if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
            return Some(format!(
                "layer {l} digest {i} of {}: GPU {:?}, CPU {:?}",
                a.len(),
                a[i],
                b[i]
            ));
        }
    }
    None
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
        let _handoff = EndOfHandoff;
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

        let _phase = sp1_metal_probes::stats::enter(sp1_metal_probes::stats::Phase::Commit);
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
            let built = sp1_metal_probes::commit::digest_layers(&words, rows);
            drop(words);
            if let Some(layers) = built {
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
                if mode() == Mode::Check {
                    drop(borrowed);
                    let (cpu_commitment, cpu_data) = self.inner.commit_tensors(tensors)?;
                    let disagreement =
                        first_disagreement(&layers, &cpu_data.0.digest_layers);
                    assert!(
                        disagreement.is_none(),
                        "metal merkle disagrees with the CPU over {rows} rows of total \
                         width {total_width}: {}",
                        disagreement.unwrap()
                    );
                    assert_eq!(
                        (commitment, root, log_height, total_width),
                        (cpu_commitment, cpu_data.1, cpu_data.2, cpu_data.3),
                        "metal merkle agrees layer by layer but publishes a different \
                         commitment over {rows} rows"
                    );
                    COMMIT_STATS.record(start.elapsed().as_secs_f64(), rows, true);
                    return Ok((cpu_commitment, cpu_data));
                }
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
