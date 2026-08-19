//! Synthetic zerocheck first round: evaluates T random degree-3 monomials per
//! row of a column-major base-field trace, combines them with powers of a
//! random extension-field alpha, weights by eq(row), and reduces to a single
//! extension-field sum. This prices the constraint-evaluation shape (the
//! dominant shard-proving cost) without a per-chip constraint transpiler;
//! the real transpiler is backend-phase work.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use slop_algebra::AbstractField;

use crate::gpu::{dispatch_groups, set_bytes, Gpu, THREADGROUP_SIZE};
use crate::kb::{ef_of_monty4, monty4_vec_of_ef, monty_vec_of_f, rand_f_vec, EF, F};
use crate::probes::times_json;

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

fn cpu_reference(data: &ZcData, n: usize) -> EF {
    let mut total = EF::zero();
    for r in 0..n {
        let mut acc = EF::zero();
        for (t, cols) in data.term_cols.iter().enumerate() {
            let m = data.trace[cols[0] as usize * n + r]
                * data.trace[cols[1] as usize * n + r]
                * data.trace[cols[2] as usize * n + r];
            acc += data.alpha_pows[t] * m;
        }
        total += acc * data.eq[r];
    }
    total
}

fn run_gpu(
    gpu: &Gpu,
    data: &ZcData,
    n: usize,
    cols: usize,
    terms: usize,
    reps: usize,
) -> Result<(EF, crate::gpu::BenchTimes), String> {
    let groups = n.div_ceil(THREADGROUP_SIZE as usize);
    let buf_trace = gpu.buffer_from(&monty_vec_of_f(&data.trace));
    let buf_eq = gpu.buffer_from(&monty4_vec_of_ef(&data.eq));
    let buf_alpha = gpu.buffer_from(&monty4_vec_of_ef(&data.alpha_pows));
    let buf_terms = gpu.buffer_from(&data.term_cols);
    let buf_partials = gpu.buffer_empty::<[u32; 4]>(groups);
    let pso = gpu.pipeline("zc_round")?;
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
    let (got, _) = run_gpu(gpu, &data, n, cols, terms, 1)?;
    if got != expected {
        return Err(format!("zc_round mismatch: got {got:?}, expected {expected:?}"));
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n: usize, cols: usize, terms: usize, reps: usize) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(42);
    let data = gen_data(&mut rng, n, cols, terms);
    let (_, times) = run_gpu(gpu, &data, n, cols, terms, reps)?;
    let base_muls_per_row = (terms * (2 + 4)) as f64 + 16.0;
    Ok(json!({
        "probe": "zerocheck",
        "params": { "n_rows": n, "cols": cols, "terms": terms, "reps": reps },
        "times": times_json(&times),
        "mega_rows_per_s_best": n as f64 / times.best / 1e6,
        "giga_base_muls_per_s_best": n as f64 * base_muls_per_row / times.best / 1e9,
        "trace_gbytes_per_s_best": (cols * n * 4) as f64 / times.best / 1e9,
    }))
}
