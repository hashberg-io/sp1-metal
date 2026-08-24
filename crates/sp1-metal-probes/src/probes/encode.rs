// SPDX-License-Identifier: MIT OR Apache-2.0
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

use crate::gpu::{dispatch_1d, dispatch_2d, dispatch_groups_2d, set_bytes, BenchTimes, Gpu};
use crate::kb::{f_of_monty, monty_vec_of_f, rand_f_vec, F};
use crate::packed::{LANES, PF};
use crate::probes::{cpu_bench, times_json};

/// The blowup the core prover's FRI config uses: `CORE_LOG_BLOWUP` is 2, so the
/// codeword is four times the trace, not twice.
///
/// This probe said one for a long time and the number it published was measured
/// at the wrong shape because of it. The encode kernels take the blowup as a
/// parameter, and the parity check below runs both.
const LOG_BLOWUP: usize = 2;

/// Rows and columns of the tile `dif_tail` holds in threadgroup memory, and the
/// number of stages it runs while it is resident. Must match the shader.
const TILE_ROWS: usize = 64;
const TILE_COLS: usize = 64;
const TILE_STAGES: usize = 6;

/// Kernel parameters, matching `DifParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct DifParams {
    log_n: u32,
    log_m: u32,
    cols: u32,
    log_blowup: u32,
}

/// The twiddles the whole transform reads: `w_n^j` for `j < n/2`. Stage `2^s`
/// indexes this with stride `2^(log_n - 1 - s)`.
fn encode_twiddles(n: usize) -> Vec<F> {
    let root = F::two_adic_generator(n.trailing_zeros() as usize);
    let mut w = Vec::with_capacity(n / 2);
    let mut cur = F::one();
    for _ in 0..n / 2 {
        w.push(cur);
        cur *= root;
    }
    w
}

/// Whether the tiled tail applies: it needs its six stages to be below the ones
/// the spread already did, and a whole number of tiles in both directions.
fn tail_applies(n: usize, cols: usize, log_blowup: usize) -> bool {
    n % TILE_ROWS == 0
        && cols % TILE_COLS == 0
        && n.trailing_zeros() as usize >= TILE_STAGES + log_blowup
}

/// Encodes the whole transform into one compute pass: the spread, then a global
/// pass per stage down to the tile height, then the six tiled stages.
#[allow(clippy::too_many_arguments)]
fn encode_dif(
    enc: &metal::ComputeCommandEncoderRef,
    psos: &DifPsos,
    src: &metal::Buffer,
    dst: &metal::Buffer,
    tw: &metal::Buffer,
    rows: usize,
    cols: usize,
    log_blowup: usize,
) {
    let n = rows << log_blowup;
    let log_n = n.trailing_zeros() as usize;
    let mut params = DifParams {
        log_n: log_n as u32,
        log_m: 0,
        cols: cols as u32,
        log_blowup: log_blowup as u32,
    };
    enc.set_compute_pipeline_state(&psos.0);
    enc.set_buffer(0, Some(src), 0);
    enc.set_buffer(1, Some(dst), 0);
    enc.set_buffer(2, Some(tw), 0);
    set_bytes(enc, 3, &params);
    dispatch_2d(enc, cols as u64, rows as u64);
    let lowest = if tail_applies(n, cols, log_blowup) { TILE_STAGES } else { 0 };
    let mut s = log_n - log_blowup;
    if (s - lowest) % 2 == 1 {
        s -= 1;
        params.log_m = s as u32;
        enc.set_compute_pipeline_state(&psos.1);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_bytes(enc, 2, &params);
        dispatch_2d(enc, cols as u64, (n / 2) as u64);
    }
    while s > lowest {
        params.log_m = (s - 1) as u32;
        enc.set_compute_pipeline_state(&psos.2);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_bytes(enc, 2, &params);
        dispatch_2d(enc, cols as u64, (n / 4) as u64);
        s -= 2;
    }
    if lowest > 0 {
        params.log_m = 0;
        enc.set_compute_pipeline_state(&psos.3);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_bytes(enc, 2, &params);
        dispatch_groups_2d(enc, (cols / TILE_COLS) as u64, (n / TILE_ROWS) as u64);
    }
}

type DifPsos = (
    metal::ComputePipelineState,
    metal::ComputePipelineState,
    metal::ComputePipelineState,
    metal::ComputePipelineState,
);

fn dif_psos(gpu: &Gpu) -> Result<DifPsos, String> {
    Ok((
        gpu.pipeline("dif_spread")?,
        gpu.pipeline("dif_stage")?,
        gpu.pipeline("dif_stage4")?,
        gpu.pipeline("dif_tail")?,
    ))
}

/// Runs the Metal encode once and returns the codeword in row-major order.
fn gpu_encode(
    gpu: &Gpu,
    src: &[F],
    rows: usize,
    cols: usize,
    log_blowup: usize,
) -> Result<Vec<F>, String> {
    let n = rows << log_blowup;
    let psos = dif_psos(gpu)?;
    let buf_src = gpu.buffer_from(&monty_vec_of_f(src));
    let buf_dst = gpu.buffer_empty::<u32>(n * cols);
    let buf_tw = gpu.buffer_from(&monty_vec_of_f(&encode_twiddles(n)));
    gpu.time_encoded(1, |enc| {
        encode_dif(enc, &psos, &buf_src, &buf_dst, &buf_tw, rows, cols, log_blowup)
    });
    Ok(gpu.read_vec::<u32>(&buf_dst, n * cols).into_iter().map(f_of_monty).collect())
}

fn gpu_encode_bench(gpu: &Gpu, rows: usize, cols: usize, reps: usize) -> Result<BenchTimes, String> {
    let mut rng = StdRng::seed_from_u64(64);
    let n = rows << LOG_BLOWUP;
    let psos = dif_psos(gpu)?;
    let buf_src = gpu.buffer_from(&monty_vec_of_f(&rand_f_vec(&mut rng, rows * cols)));
    let buf_dst = gpu.buffer_empty::<u32>(n * cols);
    let buf_tw = gpu.buffer_from(&monty_vec_of_f(&encode_twiddles(n)));
    Ok(gpu.bench_encoded(reps, 1, |enc| {
        encode_dif(enc, &psos, &buf_src, &buf_dst, &buf_tw, rows, cols, LOG_BLOWUP)
    }))
}

/// `w_n^k` for any `k < n`, from a table holding only the first half.
///
/// `w_n^(n/2)` is `-1`, so the upper half of the table is the negation of the
/// lower one and never has to be stored. Only the spread needs this; every
/// other stage indexes below `n / 2`.
fn tw_at(tw: &[F], k: usize, n: usize) -> F {
    let t = tw[k & (n / 2 - 1)];
    if k & (n / 2) != 0 {
        -t
    } else {
        t
    }
}

/// The same transform in packed CPU lanes. A row is `cols` contiguous elements
/// and a butterfly is elementwise across the row, so the packing runs along the
/// row and needs no transpose.
fn cpu_encode_packed(
    src: &[F],
    rows: usize,
    cols: usize,
    log_blowup: usize,
    tw: &[F],
    out: &mut [F],
) {
    let n = rows << log_blowup;
    let log_n = n.trailing_zeros() as usize;
    // The top `log_blowup` stages at once: each pairs a trace row with zero
    // padding, so what they amount to is `blowup` copies of the trace, copy `c`
    // scaled row-wise by `w_n^(j * r)` where `r` is `c` bit-reversed.
    out.par_chunks_mut(rows * cols).enumerate().for_each(|(c, block)| {
        let r = (c as u32).reverse_bits() as usize >> (32 - log_blowup);
        block.par_chunks_mut(cols).zip(src.par_chunks(cols)).enumerate().for_each(
            |(j, (out_row, src_row))| {
                let t = PF::from(tw_at(tw, j * r, n));
                let sp = PF::pack_slice(src_row);
                let op = PF::pack_slice_mut(out_row);
                for k in 0..op.len() {
                    op[k] = if r == 0 { sp[k] } else { sp[k] * t };
                }
            },
        );
    });
    for s in (0..log_n - log_blowup).rev() {
        let m = 1usize << s;
        let stride = 1usize << (log_n - 1 - s);
        out.par_chunks_mut(2 * m * cols).for_each(|block| {
            let (lo, hi) = block.split_at_mut(m * cols);
            lo.par_chunks_mut(cols).zip(hi.par_chunks_mut(cols)).enumerate().for_each(
                |(j, (lo_row, hi_row))| {
                    let t = PF::from(tw[j * stride]);
                    let lop = PF::pack_slice_mut(lo_row);
                    let hip = PF::pack_slice_mut(hi_row);
                    for k in 0..lop.len() {
                        let u = lop[k];
                        let v = hip[k];
                        lop[k] = u + v;
                        hip[k] = (u - v) * t;
                    }
                },
            );
        });
    }
}

/// slop's own encode, which is what the phase costs today.
fn slop_encode(src: &Tensor<F>, dft: &Radix2DitParallel, log_blowup: usize) -> Vec<F> {
    Dft::<F>::dft(dft, src, log_blowup, DftOrdering::BitReversed, 0).unwrap().as_slice().to_vec()
}

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
    let n = rows << LOG_BLOWUP;
    let tw = encode_twiddles(n);
    let src_slice = src.as_slice().to_vec();
    let mut packed_out = vec![F::zero(); n * cols];
    let tp =
        cpu_bench(reps, || cpu_encode_packed(&src_slice, rows, cols, LOG_BLOWUP, &tw, &mut packed_out));
    let g = gpu_encode_bench(gpu, rows, cols, reps)?;
    // Passes over the codeword: the spread reads the trace and writes the
    // codeword, each global stage reads and writes it, and the tiled tail does
    // six stages for one pass.
    let log_n = n.trailing_zeros() as usize;
    let tail = tail_applies(n, cols, LOG_BLOWUP);
    let lowest = if tail { TILE_STAGES } else { 0 };
    let global_stages = log_n - LOG_BLOWUP - lowest;
    // Two stages to a pass, plus one more if there is an odd stage left over.
    let global_passes = global_stages / 2 + global_stages % 2;
    let passes = 1 + global_passes + usize::from(tail);
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
        "cpu_packed": {
            "times": times_json(&tp),
            "giga_butterflies_per_s_best": butterflies / tp.best / 1e9,
            "nanoseconds_per_input_cell": tp.best / cells as f64 * 1e9,
            "note": "the same decimation-in-frequency transform in packed NEON lanes, rayon on all cores",
        },
        "gpu": {
            "times": times_json(&g),
            "giga_butterflies_per_s_best": butterflies / g.best / 1e9,
            "nanoseconds_per_input_cell": g.best / cells as f64 * 1e9,
            "passes_over_codeword": passes,
            "stages": log_n,
            "global_passes": global_passes,
            "tiled_tail": tail,
            "effective_gbytes_per_s": (passes * n * cols * 2 * 4) as f64 / g.best / 1e9,
        },
        "gpu_over_cpu_packed": tp.best / g.best,
        "gpu_over_slop": t.best / g.best,
        "packed_over_slop": t.best / tp.best,
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

    // The whole encode, against slop's own. The kernels ship in the vendored
    // `slop-dft` and this is the check that says they agree with what they
    // replace, so the shapes have to reach every path through them:
    //
    // - 64x64 has no global stages at all, only the spread and the tiled tail;
    // - 1024x64 has an even number of global stages, so they all pair off into
    //   radix-4 passes;
    // - 2048x64 has an odd number, which leaves one radix-2 stage at the top;
    // - the 48-column shapes have no tiled tail, since it needs whole tiles, so
    //   every stage below the spread is a global pass -- again in both parities.
    //
    // Both blowups: the one this probe used to assume, and the one the core
    // prover actually configures.
    let dft = Radix2DitParallel::default();
    for &log_blowup in &[1usize, 2] {
        for &(rows, cols) in &[
            (64usize, 64usize),
            (1024usize, 64usize),
            (2048usize, 64usize),
            (256usize, 48usize),
            (512usize, 48usize),
        ] {
            let src = rand_f_vec(&mut rng, rows * cols);
            let want =
                slop_encode(&Tensor::from(src.clone()).reshape([rows, cols]), &dft, log_blowup);
            let got = gpu_encode(gpu, &src, rows, cols, log_blowup)?;
            if got != want {
                let at = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(0);
                return Err(format!(
                    "metal encode mismatch at {rows}x{cols} blowup 2^{log_blowup}, \
                     first at index {at}: got {:?}, expected {:?}",
                    got[at], want[at]
                ));
            }
            let n = rows << log_blowup;
            let mut packed_out = vec![F::zero(); n * cols];
            cpu_encode_packed(&src, rows, cols, log_blowup, &encode_twiddles(n), &mut packed_out);
            if packed_out != want {
                return Err(format!(
                    "packed CPU encode mismatch at {rows}x{cols} blowup 2^{log_blowup}"
                ));
            }
        }
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
