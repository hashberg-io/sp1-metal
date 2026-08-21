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
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::{json, Value};
use slop_algebra::{AbstractField, PackedValue, TwoAdicField};
use slop_dft::p3::Radix2DitParallel;
use slop_dft::{Dft, DftOrdering};
use slop_tensor::Tensor;

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::{f_of_monty, monty_vec_of_f, rand_f_vec, F};
use crate::packed::{LANES, PF};
use crate::probes::{cpu_bench, times_json};

/// The blowup slop's default FRI config uses: the codeword is twice the trace.
const LOG_BLOWUP: usize = 1;

pub fn bench(gpu: &Gpu, rows: usize, cols: usize, reps: usize) -> Result<Value, String> {
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
        "butterfly_stage": stage_bench(gpu, 1 << 26, 12, reps)?,
    }))
}

/// The twiddles for a stage whose half-block is `m` wide: successive powers of
/// a primitive 2m-th root of unity.
fn twiddles(m: usize) -> Vec<F> {
    let root = F::two_adic_generator(m.trailing_zeros() as usize + 1);
    let mut w = Vec::with_capacity(m);
    let mut cur = F::one();
    for _ in 0..m {
        w.push(cur);
        cur *= root;
    }
    w
}

/// One butterfly stage on the CPU, in packed lanes. Within a block the pairs
/// are `m` apart and `j` runs contiguously, so as long as `m` is at least the
/// SIMD width the loads are contiguous on both sides of the butterfly.
fn cpu_stage(a: &mut [F], tw: &[F], m: usize) {
    a.par_chunks_mut(2 * m).for_each(|block| {
        let (lo, hi) = block.split_at_mut(m);
        let lop = PF::pack_slice_mut(lo);
        let hip = PF::pack_slice_mut(hi);
        let twp = PF::pack_slice(tw);
        for k in 0..lop.len() {
            let u = lop[k];
            let t = hip[k] * twp[k];
            lop[k] = u + t;
            hip[k] = u - t;
        }
    });
}

fn scalar_stage(a: &mut [F], tw: &[F], m: usize) {
    for block in a.chunks_mut(2 * m) {
        let (lo, hi) = block.split_at_mut(m);
        for j in 0..m {
            let u = lo[j];
            let t = hi[j] * tw[j];
            lo[j] = u + t;
            hi[j] = u - t;
        }
    }
}

fn gpu_stage(
    gpu: &Gpu,
    buf: &metal::Buffer,
    tw: &metal::Buffer,
    n: usize,
    log_m: u32,
    reps: usize,
) -> Result<crate::gpu::BenchTimes, String> {
    let pso = gpu.pipeline("ntt_stage")?;
    Ok(gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(buf), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_bytes(enc, 2, &log_m);
        dispatch_1d(enc, (n / 2) as u64);
    }))
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(62);
    let n = 4096usize;
    let log_m = 5u32;
    let m = 1usize << log_m;
    let vals: Vec<F> = (0..n).map(|_| rng.gen()).collect();
    let tw = twiddles(m);
    let buf = gpu.buffer_from(&monty_vec_of_f(&vals));
    let buf_tw = gpu.buffer_from(&monty_vec_of_f(&tw));
    gpu_stage(gpu, &buf, &buf_tw, n, log_m, 0)?;
    let got: Vec<u32> = gpu.read_vec(&buf, n);
    let mut expected = vals.clone();
    scalar_stage(&mut expected, &tw, m);
    for i in 0..n {
        if f_of_monty(got[i]) != expected[i] {
            return Err(format!("ntt_stage mismatch at {i}"));
        }
    }
    let mut packed = vals;
    cpu_stage(&mut packed, &tw, m);
    if packed != expected {
        return Err("packed CPU butterfly stage disagrees with the scalar one".to_string());
    }
    Ok(())
}

/// A single butterfly stage, GPU against packed CPU. A DFT is a sequence of
/// these, so this is the ratio the encode gets when neither side manages to
/// keep a block resident across stages — the memory-pass-bound floor for both.
pub fn stage_bench(gpu: &Gpu, n: usize, log_m: u32, reps: usize) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(63);
    let m = 1usize << log_m;
    let mut vals: Vec<F> = (0..n).map(|_| rng.gen()).collect();
    let tw = twiddles(m);
    let buf = gpu.buffer_from(&monty_vec_of_f(&vals));
    let buf_tw = gpu.buffer_from(&monty_vec_of_f(&tw));
    let g = gpu_stage(gpu, &buf, &buf_tw, n, log_m, reps)?;
    let c = cpu_bench(reps, || cpu_stage(&mut vals, &tw, m));
    let bytes = (2 * n * 4) as f64;
    Ok(json!({
        "params": { "n": n, "log_m": log_m, "lanes": LANES, "reps": reps },
        "gpu": { "times": times_json(&g), "gbytes_per_s_best": bytes / g.best / 1e9 },
        "cpu_packed": { "times": times_json(&c), "gbytes_per_s_best": bytes / c.best / 1e9 },
        "gpu_over_cpu_packed": c.best / g.best,
    }))
}
