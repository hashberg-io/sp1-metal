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

/// The blowup slop's default FRI config uses: the codeword is twice the trace.
const LOG_BLOWUP: usize = 1;

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

/// Whether the tiled tail applies: it needs six stages below the tile height
/// and a whole number of tiles in both directions.
fn tail_applies(n: usize, cols: usize) -> bool {
    n >= 2 * TILE_ROWS && n % TILE_ROWS == 0 && cols % TILE_COLS == 0
        && n.trailing_zeros() as usize >= TILE_STAGES + 2
}

/// Encodes the whole transform into one compute pass: the spread, then a global
/// pass per stage down to the tile height, then the six tiled stages.
#[allow(clippy::too_many_arguments)]
fn encode_dif(
    enc: &metal::ComputeCommandEncoderRef,
    psos: &(metal::ComputePipelineState, metal::ComputePipelineState, metal::ComputePipelineState),
    src: &metal::Buffer,
    dst: &metal::Buffer,
    tw: &metal::Buffer,
    rows: usize,
    cols: usize,
) {
    let n = rows << LOG_BLOWUP;
    let log_n = n.trailing_zeros() as usize;
    let mut params = DifParams { log_n: log_n as u32, log_m: (log_n - 1) as u32, cols: cols as u32 };
    enc.set_compute_pipeline_state(&psos.0);
    enc.set_buffer(0, Some(src), 0);
    enc.set_buffer(1, Some(dst), 0);
    enc.set_buffer(2, Some(tw), 0);
    set_bytes(enc, 3, &params);
    dispatch_2d(enc, cols as u64, rows as u64);
    let lowest = if tail_applies(n, cols) { TILE_STAGES } else { 0 };
    for s in (lowest..log_n - 1).rev() {
        params.log_m = s as u32;
        enc.set_compute_pipeline_state(&psos.1);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_bytes(enc, 2, &params);
        dispatch_2d(enc, cols as u64, (n / 2) as u64);
    }
    if lowest > 0 {
        params.log_m = 0;
        enc.set_compute_pipeline_state(&psos.2);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_bytes(enc, 2, &params);
        dispatch_groups_2d(enc, (cols / TILE_COLS) as u64, (n / TILE_ROWS) as u64);
    }
}

fn dif_psos(
    gpu: &Gpu,
) -> Result<(metal::ComputePipelineState, metal::ComputePipelineState, metal::ComputePipelineState), String> {
    Ok((gpu.pipeline("dif_spread")?, gpu.pipeline("dif_stage")?, gpu.pipeline("dif_tail")?))
}

/// Runs the Metal encode once and returns the codeword in row-major order.
fn gpu_encode(gpu: &Gpu, src: &[F], rows: usize, cols: usize) -> Result<Vec<F>, String> {
    let n = rows << LOG_BLOWUP;
    let psos = dif_psos(gpu)?;
    let buf_src = gpu.buffer_from(&monty_vec_of_f(src));
    let buf_dst = gpu.buffer_empty::<u32>(n * cols);
    let buf_tw = gpu.buffer_from(&monty_vec_of_f(&encode_twiddles(n)));
    gpu.time_encoded(1, |enc| encode_dif(enc, &psos, &buf_src, &buf_dst, &buf_tw, rows, cols));
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
        encode_dif(enc, &psos, &buf_src, &buf_dst, &buf_tw, rows, cols)
    }))
}

/// The same transform in packed CPU lanes. A row is `cols` contiguous elements
/// and a butterfly is elementwise across the row, so the packing runs along the
/// row and needs no transpose.
fn cpu_encode_packed(src: &[F], rows: usize, cols: usize, tw: &[F], out: &mut [F]) {
    let n = rows << LOG_BLOWUP;
    let log_n = n.trailing_zeros() as usize;
    {
        let (lo, hi) = out.split_at_mut(rows * cols);
        lo.par_chunks_mut(cols)
            .zip(hi.par_chunks_mut(cols))
            .zip(src.par_chunks(cols))
            .enumerate()
            .for_each(|(j, ((lo_row, hi_row), src_row))| {
                let t = PF::from(tw[j]);
                let sp = PF::pack_slice(src_row);
                let lop = PF::pack_slice_mut(lo_row);
                let hip = PF::pack_slice_mut(hi_row);
                for k in 0..lop.len() {
                    lop[k] = sp[k];
                    hip[k] = sp[k] * t;
                }
            });
    }
    for s in (0..log_n - 1).rev() {
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
fn slop_encode(src: &Tensor<F>, dft: &Radix2DitParallel) -> Vec<F> {
    Dft::<F>::dft(dft, src, LOG_BLOWUP, DftOrdering::BitReversed, 0).unwrap().as_slice().to_vec()
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
    let tp = cpu_bench(reps, || cpu_encode_packed(&src_slice, rows, cols, &tw, &mut packed_out));
    let g = gpu_encode_bench(gpu, rows, cols, reps)?;
    // Passes over the codeword: the spread reads the trace and writes the
    // codeword, each global stage reads and writes it, and the tiled tail does
    // six stages for one pass.
    let log_n = n.trailing_zeros() as usize;
    let global_stages = if tail_applies(n, cols) { log_n - 1 - TILE_STAGES } else { log_n - 1 };
    let passes = 1 + global_stages + usize::from(tail_applies(n, cols));
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
            "tiled_tail": tail_applies(n, cols),
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

    // The whole encode, against slop's own. Two shapes: one small enough that
    // every stage is a global pass, and one large enough to exercise the tiled
    // tail as well.
    let dft = Radix2DitParallel::default();
    for &(rows, cols) in &[(64usize, 64usize), (1024usize, 64usize)] {
        let src = rand_f_vec(&mut rng, rows * cols);
        let want = slop_encode(&Tensor::from(src.clone()).reshape([rows, cols]), &dft);
        let got = gpu_encode(gpu, &src, rows, cols)?;
        if got != want {
            let at = got.iter().zip(want.iter()).position(|(a, b)| a != b).unwrap_or(0);
            return Err(format!(
                "metal encode mismatch at {rows}x{cols}, first at index {at}: got {:?}, expected {:?}",
                got[at], want[at]
            ));
        }
        let n = rows << LOG_BLOWUP;
        let mut packed_out = vec![F::zero(); n * cols];
        cpu_encode_packed(&src, rows, cols, &encode_twiddles(n), &mut packed_out);
        if packed_out != want {
            return Err(format!("packed CPU encode mismatch at {rows}x{cols}"));
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
