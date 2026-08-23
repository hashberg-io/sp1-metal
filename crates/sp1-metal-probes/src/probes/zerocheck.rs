//! Synthetic zerocheck first round: evaluates T random degree-3 monomials per
//! row of a column-major base-field trace, combines them with powers of a
//! random extension-field alpha, weights by eq(row), and reduces to a single
//! extension-field sum. This prices the constraint-evaluation shape (the
//! dominant shard-proving cost) without a per-chip constraint transpiler;
//! the real transpiler is backend-phase work.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::{json, Value};
use slop_algebra::{AbstractField, PackedValue};

use crate::gpu::{dispatch_groups, set_bytes, Gpu, THREADGROUP_SIZE};
use crate::kb::{ef_of_monty4, monty4_vec_of_ef, monty_vec_of_f, rand_f_vec, EF, F};
use crate::packed::{broadcast_ext, pack_ext, sum_lanes, LANES, PEF, PF};
use crate::probes::{cpu_bench, times_json};

#[repr(C)]
#[derive(Clone, Copy)]
struct ZcParams {
    n: u32,
    num_cols: u32,
    num_terms: u32,
}

struct ZcData {
    trace: Vec<F>,
    eq: Vec<EF>,
    alpha_pows: Vec<EF>,
    term_cols: Vec<[u32; 4]>,
}

fn gen_data(rng: &mut StdRng, n: usize, cols: usize, terms: usize) -> ZcData {
    let trace = rand_f_vec(rng, cols * n);
    let eq: Vec<EF> = (0..n).map(|_| rng.gen()).collect();
    let alpha: EF = rng.gen();
    let mut alpha_pows = Vec::with_capacity(terms);
    let mut pow = EF::one();
    for _ in 0..terms {
        alpha_pows.push(pow);
        pow *= alpha;
    }
    let term_cols = (0..terms)
        .map(|_| {
            [
                rng.gen_range(0..cols) as u32,
                rng.gen_range(0..cols) as u32,
                rng.gen_range(0..cols) as u32,
                0,
            ]
        })
        .collect();
    ZcData { trace, eq, alpha_pows, term_cols }
}

/// The same formula the kernel computes, over rayon on all cores: this doubles
/// as the parity oracle and as the CPU side of the speedup ratio. Summation
/// order differs from the kernel's, which is exact-arithmetic irrelevant.
fn cpu_reference(data: &ZcData, n: usize) -> EF {
    (0..n)
        .into_par_iter()
        .map(|r| {
            let mut acc = EF::zero();
            for (t, cols) in data.term_cols.iter().enumerate() {
                let m = data.trace[cols[0] as usize * n + r]
                    * data.trace[cols[1] as usize * n + r]
                    * data.trace[cols[2] as usize * n + r];
                acc += data.alpha_pows[t] * m;
            }
            acc * data.eq[r]
        })
        .reduce(EF::zero, |a, b| a + b)
}

/// Everything the packed CPU pass needs that the GPU gets for free from its own
/// buffer layout: alpha powers broadcast across lanes, and eq values transposed
/// into packed extension elements. Built once, outside the timed region, as a
/// prover holding its data in packed layout would.
struct ZcPacked {
    alpha: Vec<PEF>,
    eq: Vec<PEF>,
}

impl ZcPacked {
    fn new(data: &ZcData, n: usize) -> Self {
        assert_eq!(n % LANES, 0, "packed CPU pass needs a row count divisible by the SIMD width");
        Self {
            alpha: data.alpha_pows.iter().map(|&a| broadcast_ext(a)).collect(),
            eq: pack_ext(&data.eq),
        }
    }
}

/// Rows per tile for the packed pass: at 64 columns this keeps everything one
/// tile touches inside L2, so the loop over terms streams from cache instead of
/// re-reading the whole trace once per term.
const TILE_ROWS: usize = 1024;

/// The same formula again, over packed lanes and cache-tiled: the CPU baseline
/// the Metal kernel actually has to beat.
fn cpu_packed(data: &ZcData, packed: &ZcPacked, n: usize) -> EF {
    let groups = n / LANES;
    let tile_groups = (TILE_ROWS / LANES).max(1);
    let total = (0..groups.div_ceil(tile_groups))
        .into_par_iter()
        .map(|ti| {
            let g0 = ti * tile_groups;
            let g1 = ((ti + 1) * tile_groups).min(groups);
            let mut accs = vec![PEF::zero(); g1 - g0];
            for (t, cols) in data.term_cols.iter().enumerate() {
                let a = packed.alpha[t];
                let (c0, c1, c2) =
                    (cols[0] as usize * n, cols[1] as usize * n, cols[2] as usize * n);
                for (gi, acc) in accs.iter_mut().enumerate() {
                    let r = (g0 + gi) * LANES;
                    let x0 = *PF::from_slice(&data.trace[c0 + r..c0 + r + LANES]);
                    let x1 = *PF::from_slice(&data.trace[c1 + r..c1 + r + LANES]);
                    let x2 = *PF::from_slice(&data.trace[c2 + r..c2 + r + LANES]);
                    *acc += a * (x0 * x1 * x2);
                }
            }
            let mut tile = PEF::zero();
            for (gi, acc) in accs.iter().enumerate() {
                tile += *acc * packed.eq[g0 + gi];
            }
            tile
        })
        .reduce(PEF::zero, |a, b| a + b);
    sum_lanes(&total)
}

#[allow(clippy::too_many_arguments)]
fn run_gpu(
    gpu: &Gpu,
    data: &ZcData,
    n: usize,
    cols: usize,
    terms: usize,
    reps: usize,
    kernel: &str,
) -> Result<(EF, crate::gpu::BenchTimes), String> {
    let groups = n.div_ceil(THREADGROUP_SIZE as usize);
    let buf_trace = gpu.buffer_from(&monty_vec_of_f(&data.trace));
    let buf_eq = gpu.buffer_from(&monty4_vec_of_ef(&data.eq));
    let buf_alpha = gpu.buffer_from(&monty4_vec_of_ef(&data.alpha_pows));
    let buf_terms = gpu.buffer_from(&data.term_cols);
    let buf_partials = gpu.buffer_empty::<[u32; 4]>(groups);
    let pso = gpu.pipeline(kernel)?;
    let params = ZcParams { n: n as u32, num_cols: cols as u32, num_terms: terms as u32 };
    let times = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_trace), 0);
        enc.set_buffer(1, Some(&buf_eq), 0);
        enc.set_buffer(2, Some(&buf_alpha), 0);
        enc.set_buffer(3, Some(&buf_terms), 0);
        set_bytes(enc, 4, &params);
        enc.set_buffer(5, Some(&buf_partials), 0);
        dispatch_groups(enc, groups as u64);
    });
    let partials: Vec<[u32; 4]> = gpu.read_vec(&buf_partials, groups);
    let mut total = EF::zero();
    for p in partials {
        total += ef_of_monty4(p);
    }
    Ok((total, times))
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(41);
    let (n, cols, terms) = (4096usize, 8usize, 8usize);
    let data = gen_data(&mut rng, n, cols, terms);
    let expected = cpu_reference(&data, n);
    let (got, _) = run_gpu(gpu, &data, n, cols, terms, 1, "zc_round")?;
    if got != expected {
        return Err(format!("zc_round mismatch: got {got:?}, expected {expected:?}"));
    }
    let packed = ZcPacked::new(&data, n);
    let got_packed = cpu_packed(&data, &packed, n);
    if got_packed != expected {
        return Err(format!(
            "packed CPU baseline mismatch: got {got_packed:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

pub fn bench(
    gpu: &Gpu,
    n: usize,
    cols: usize,
    terms: usize,
    reps: usize,
    with_cpu: bool,
) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(42);
    let data = gen_data(&mut rng, n, cols, terms);
    let (_, times) = run_gpu(gpu, &data, n, cols, terms, reps, "zc_round")?;
    // The same arithmetic over the same data with the trace read the way a real
    // shard's is. Only the layout differs.
    let (_, times_rm) = run_gpu(gpu, &data, n, cols, terms, reps, "zc_round_rowmajor")?;
    let base_muls_per_row = (terms * (2 + 4)) as f64 + 16.0;
    let mut result = json!({
        "probe": "zerocheck",
        "params": { "n_rows": n, "cols": cols, "terms": terms, "reps": reps },
        "times": times_json(&times),
        "mega_rows_per_s_best": n as f64 / times.best / 1e6,
        "giga_base_muls_per_s_best": n as f64 * base_muls_per_row / times.best / 1e9,
        "row_major": {
            "times": times_json(&times_rm),
            "giga_base_muls_per_s_best": n as f64 * base_muls_per_row / times_rm.best / 1e9,
            "column_major_over_row_major": times_rm.best / times.best,
            "note": "identical kernel and data, trace read as main[row * cols + col] rather than main[col * n + row]",
        },
        "trace_gbytes_per_s_best": (cols * n * 4) as f64 / times.best / 1e9,
    });
    if with_cpu {
        let t = cpu_bench(reps, || {
            std::hint::black_box(cpu_reference(&data, n));
        });
        result["cpu"] = json!({
            "times": times_json(&t),
            "mega_rows_per_s_best": n as f64 / t.best / 1e6,
            "giga_base_muls_per_s_best": n as f64 * base_muls_per_row / t.best / 1e9,
            "note": "same formula in slop field types over rayon on all cores",
        });
        result["gpu_over_cpu"] = json!(t.best / times.best);
        let packed = ZcPacked::new(&data, n);
        let tp = cpu_bench(reps, || {
            std::hint::black_box(cpu_packed(&data, &packed, n));
        });
        result["cpu_packed"] = json!({
            "times": times_json(&tp),
            "lanes": LANES,
            "tile_rows": TILE_ROWS,
            "mega_rows_per_s_best": n as f64 / tp.best / 1e6,
            "giga_base_muls_per_s_best": n as f64 * base_muls_per_row / tp.best / 1e9,
            "note": "packed (NEON) lanes, cache-tiled, rayon on all cores",
        });
        result["gpu_over_cpu_packed"] = json!(tp.best / times.best);
    }
    Ok(result)
}
