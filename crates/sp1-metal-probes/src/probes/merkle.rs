// SPDX-License-Identifier: MIT OR Apache-2.0
//! The Merkle tree half of a BaseFold commitment.
//!
//! `commit traces` is a Reed-Solomon encode followed by a Merkle tree over the
//! codeword, and the encode probe covers the first half. This is the second:
//! slop hashes each row of the row-major codeword into a digest with
//! `PaddingFreeSponge<KoalaPerm, 16, 8, 8>` and compresses pairs up the tree
//! with `TruncatedPermutation<KoalaPerm, 2, 8, 16>`, both over the width-16
//! Poseidon2 permutation the suite already parity-checks.
//!
//! Parity is against slop's own `Poseidon2KoalaBear16Prover` — the tensor
//! commitment scheme SP1 uses — so the root this produces is the commitment
//! SP1's verifier would accept, not merely a tree of the right shape.
//!
//! The whole tree is one submission. Each layer depends on the one below it,
//! which is exactly the shape that would cost a round trip per layer if the
//! host were in the loop; there is no reason for it to be.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};
use slop_commit::Message;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_challenger::IopCtx;
use slop_merkle_tree::{MerkleTreeTcs, Poseidon2KoalaBear16Prover, TensorCsProver};
use slop_tensor::Tensor;

use crate::gpu::{dispatch_1d, set_bytes, BenchTimes, Gpu};
use crate::kb::{f_of_monty, monty_of_f, monty_vec_of_f, rand_f_vec, F};
use slop_algebra::AbstractField;
use crate::probes::poseidon2::rc_monty;
use crate::probes::{cpu_bench, times_json};

/// Encodes the whole tree into one command buffer: the leaf layer, then one
/// compression layer per level. Buffers alternate between levels.
#[allow(clippy::too_many_arguments)]
fn encode_tree(
    enc_leaves: &metal::ComputePipelineState,
    enc_compress: &metal::ComputePipelineState,
    enc_finalize: &metal::ComputePipelineState,
    gpu: &Gpu,
    rows: &metal::Buffer,
    levels: &[metal::Buffer],
    commitment: &metal::Buffer,
    rc: &metal::Buffer,
    n_rows: usize,
    width: usize,
) -> f64 {
    let log_n = n_rows.trailing_zeros() as usize;
    gpu.time_encoded(log_n + 2, {
        let level = std::cell::Cell::new(0usize);
        move |enc| {
            let l = level.get();
            if l == 0 {
                enc.set_compute_pipeline_state(enc_leaves);
                enc.set_buffer(0, Some(rows), 0);
                enc.set_buffer(1, Some(&levels[0]), 0);
                set_bytes(enc, 2, &(width as u32));
                set_bytes(enc, 3, &(n_rows as u32));
                enc.set_buffer(4, Some(rc), 0);
                dispatch_1d(enc, n_rows as u64);
            } else if l <= log_n {
                let n_parents = n_rows >> l;
                enc.set_compute_pipeline_state(enc_compress);
                enc.set_buffer(0, Some(&levels[l - 1]), 0);
                enc.set_buffer(1, Some(&levels[l]), 0);
                set_bytes(enc, 2, &(n_parents as u32));
                enc.set_buffer(3, Some(rc), 0);
                dispatch_1d(enc, n_parents as u64);
            } else {
                enc.set_compute_pipeline_state(enc_finalize);
                enc.set_buffer(0, Some(&levels[log_n]), 0);
                enc.set_buffer(1, Some(commitment), 0);
                set_bytes(enc, 2, &monty_of_f(F::from_canonical_usize(log_n)));
                set_bytes(enc, 3, &monty_of_f(F::from_canonical_usize(width)));
                enc.set_buffer(4, Some(rc), 0);
                dispatch_1d(enc, 1);
            }
            level.set(l + 1);
        }
    })
}

struct Tree {
    rows: metal::Buffer,
    levels: Vec<metal::Buffer>,
    commitment: metal::Buffer,
    rc: metal::Buffer,
}

impl Tree {
    fn new(gpu: &Gpu, vals: &[F], n_rows: usize) -> Self {
        let log_n = n_rows.trailing_zeros() as usize;
        Self {
            rows: gpu.buffer_from(&monty_vec_of_f(vals)),
            levels: (0..=log_n).map(|l| gpu.buffer_empty::<u32>((n_rows >> l) * 8)).collect(),
            commitment: gpu.buffer_empty::<u32>(8),
            rc: gpu.buffer_from(&rc_monty()),
        }
    }

    fn commitment(&self, gpu: &Gpu) -> [F; 8] {
        let raw = gpu.read_vec::<u32>(&self.commitment, 8);
        std::array::from_fn(|i| f_of_monty(raw[i]))
    }
}

fn run(gpu: &Gpu, tree: &Tree, n_rows: usize, width: usize) -> Result<f64, String> {
    let leaves = gpu.pipeline("merkle_leaves")?;
    let compress = gpu.pipeline("merkle_compress")?;
    let finalize = gpu.pipeline("merkle_finalize")?;
    Ok(encode_tree(
        &leaves,
        &compress,
        &finalize,
        gpu,
        &tree.rows,
        &tree.levels,
        &tree.commitment,
        &tree.rc,
        n_rows,
        width,
    ))
}

/// slop's own commitment scheme, and the matrix to commit to. The matrix is
/// built once: `Message` holds it behind an `Arc`, so timing a commitment does
/// not time a 500 MB clone as well.
fn slop_setup(vals: &[F], n_rows: usize, width: usize) -> (Poseidon2KoalaBear16Prover, Message<Tensor<F>>) {
    let (hasher, compressor) = KoalaBearDegree4Duplex::default_hasher_and_compressor();
    let prover = Poseidon2KoalaBear16Prover::new(MerkleTreeTcs { hasher, compressor });
    let tensor = Tensor::from(vals.to_vec()).reshape([n_rows, width]);
    (prover, Message::from(tensor))
}

/// slop's own commitment over the same matrix: the reference.
fn slop_commitment(prover: &Poseidon2KoalaBear16Prover, message: &Message<Tensor<F>>) -> [F; 8] {
    let (commitment, _) = prover.commit_tensors(message.clone()).expect("commit");
    commitment
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(91);
    // Widths on both sides of the sponge's rate, so the short final chunk is
    // exercised as well as the exact-multiple case.
    for &(n_rows, width) in &[(256usize, 8usize), (256, 16), (128, 13), (64, 1), (512, 37)] {
        let vals = rand_f_vec(&mut rng, n_rows * width);
        let tree = Tree::new(gpu, &vals, n_rows);
        run(gpu, &tree, n_rows, width)?;
        let got = tree.commitment(gpu);
        let (prover, message) = slop_setup(&vals, n_rows, width);
        let want = slop_commitment(&prover, &message);
        if got != want {
            return Err(format!(
                "merkle root mismatch at {n_rows} rows of width {width}: got {got:?}, expected {want:?}"
            ));
        }
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n_rows: usize, width: usize, reps: usize) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(92);
    let vals = rand_f_vec(&mut rng, n_rows * width);
    let tree = Tree::new(gpu, &vals, n_rows);
    run(gpu, &tree, n_rows, width)?;
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        times.push(run(gpu, &tree, n_rows, width)?);
    }
    let best = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
    let g = BenchTimes { best, mean, times };

    let (prover, message) = slop_setup(&vals, n_rows, width);
    let c = cpu_bench(reps, || {
        std::hint::black_box(slop_commitment(&prover, &message));
    });

    // Permutations the tree performs: one per rate-sized chunk of every row,
    // plus one per internal node.
    let per_row = width.div_ceil(8);
    let perms = (n_rows * per_row + n_rows - 1) as f64;
    Ok(json!({
        "probe": "merkle",
        "params": { "rows": n_rows, "width": width, "reps": reps, "permutations": perms },
        "gpu": {
            "times": times_json(&g),
            "milliseconds": g.best * 1e3,
            "mega_permutations_per_s": perms / g.best / 1e6,
            "nanoseconds_per_input_cell": g.best / (n_rows * width) as f64 * 1e9,
        },
        "cpu_slop": {
            "times": times_json(&c),
            "milliseconds": c.best * 1e3,
            "mega_permutations_per_s": perms / c.best / 1e6,
            "nanoseconds_per_input_cell": c.best / (n_rows * width) as f64 * 1e9,
            "note": "slop's Poseidon2KoalaBear16Prover, as the prover runs it today",
        },
        "gpu_over_slop": c.best / g.best,
        "caveat": "gpu_over_slop is a ratio against slop's commitment as it ships, not a hardware ratio. That commitment reaches only a fraction of the packed permutation rate this machine can do, which the poseidon2 probe measures directly; take the hardware ratio from there.",
    }))
}
