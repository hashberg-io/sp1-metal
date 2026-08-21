//! What BaseFold's commitment actually spends its time on.
//!
//! `commit traces` is the second largest phase of a shard proof, and it is not
//! hashing: slop encodes each trace with a Reed-Solomon code first — a radix-2
//! DFT with the FRI blowup — and only then builds the Merkle tree. Pricing the
//! commit phase by the Poseidon2 probe alone would be pricing the smaller half.
//!
//! This probe has no Metal kernel behind it yet. It measures slop's own CPU
//! encoder, so the phase's cost is at least attributed to the right work and
//! the split between encode and hash is visible; the kernel, and with it a real
//! ratio, is the obvious next thing the suite is missing.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};
use slop_dft::p3::Radix2DitParallel;
use slop_dft::{Dft, DftOrdering};
use slop_tensor::Tensor;

use crate::kb::{rand_f_vec, F};
use crate::probes::{cpu_bench, times_json};

/// The blowup slop's default FRI config uses: the codeword is twice the trace.
const LOG_BLOWUP: usize = 1;

pub fn bench(rows: usize, cols: usize, reps: usize) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(61);
    let cells = rows * cols;
    let src = Tensor::from(rand_f_vec(&mut rng, cells)).reshape([rows, cols]);
    let dft = Radix2DitParallel::default();
    let t = cpu_bench(reps, || {
        std::hint::black_box(
            Dft::<F>::dft(&dft, &src, LOG_BLOWUP, DftOrdering::BitReversed, 0).unwrap(),
        );
    });
    // A radix-2 DFT of N points is (N/2)·log2(N) butterflies, one multiply each.
    let n = rows << LOG_BLOWUP;
    let butterflies = (n / 2) as f64 * (n.trailing_zeros() as f64) * cols as f64;
    Ok(json!({
        "probe": "encode",
        "params": { "rows": rows, "cols": cols, "log_blowup": LOG_BLOWUP, "reps": reps },
        "cpu": {
            "times": times_json(&t),
            "giga_butterflies_per_s_best": butterflies / t.best / 1e9,
            "nanoseconds_per_input_cell": t.best / cells as f64 * 1e9,
            "gbytes_per_s_best": ((cells + (cells << LOG_BLOWUP)) * 4) as f64 / t.best / 1e9,
            "note": "slop's own CPU encoder: p3 Radix2DitParallel, bit-reversed, rayon on all cores",
        },
        "gpu": null,
    }))
}
