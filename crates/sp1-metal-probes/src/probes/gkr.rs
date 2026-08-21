//! One sumcheck round of a logup GKR layer — the largest single phase of a
//! shard proof, and the one the rest of the suite prices worst.
//!
//! The shape is taken from `sp1_hypercube::logup_gkr`'s
//! `sum_as_poly_in_last_t_variables`: walking pairs of consecutive entries of
//! the layer's four polynomials and the eq weights, the round polynomial's
//! first evaluation uses the even entry of each pair and its second uses the
//! pair's sum, and each evaluation costs a numerator term `eq·(den_0·num_1 +
//! den_1·num_0)` and a denominator term `eq·(den_0·den_1)`. Everything is in
//! the degree-4 extension, which is what makes the round expensive: five
//! extension multiplications per row against eighty bytes read, an arithmetic
//! intensity that puts both the GPU and the CPU on the compute side of their
//! rooflines.
//!
//! The probe reproduces a general circuit layer, where all four polynomials are
//! extension-valued. The first GKR layer is cheaper — its numerators are base
//! field — but every layer after it looks like this one.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use serde_json::{json, Value};
use slop_algebra::AbstractField;

use crate::gpu::{dispatch_groups, set_bytes, Gpu, THREADGROUP_SIZE};
use crate::kb::{ef_of_monty4, monty4_vec_of_ef, rand_ef_vec, EF};
use crate::packed::{split_pack_ext, LANES, PEF};
use crate::probes::{cpu_bench, times_json};

/// The five extension accumulators a round produces: numerator and denominator
/// at each of the two evaluation points, and the eq sum.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RoundSums {
    numer_first: EF,
    denom_first: EF,
    numer_second: EF,
    denom_second: EF,
    eq_sum: EF,
}

impl RoundSums {
    fn zero() -> Self {
        Self {
            numer_first: EF::zero(),
            denom_first: EF::zero(),
            numer_second: EF::zero(),
            denom_second: EF::zero(),
            eq_sum: EF::zero(),
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            numer_first: self.numer_first + other.numer_first,
            denom_first: self.denom_first + other.denom_first,
            numer_second: self.numer_second + other.numer_second,
            denom_second: self.denom_second + other.denom_second,
            eq_sum: self.eq_sum + other.eq_sum,
        }
    }
}

struct Layer {
    num_0: Vec<EF>,
    num_1: Vec<EF>,
    den_0: Vec<EF>,
    den_1: Vec<EF>,
    eq: Vec<EF>,
}

fn gen_layer(rng: &mut StdRng, n: usize) -> Layer {
    Layer {
        num_0: rand_ef_vec(rng, n),
        num_1: rand_ef_vec(rng, n),
        den_0: rand_ef_vec(rng, n),
        den_1: rand_ef_vec(rng, n),
        eq: rand_ef_vec(rng, n),
    }
}

/// The reference, in slop's field types, written the way the prover writes it.
fn cpu_reference(layer: &Layer) -> RoundSums {
    (0..layer.eq.len() / 2)
        .into_par_iter()
        .map(|j| {
            let (a, b) = (2 * j, 2 * j + 1);
            let numer =
                layer.den_0[a] * layer.num_1[a] + layer.den_1[a] * layer.num_0[a];
            let n0h = layer.num_0[a] + layer.num_0[b];
            let n1h = layer.num_1[a] + layer.num_1[b];
            let d0h = layer.den_0[a] + layer.den_0[b];
            let d1h = layer.den_1[a] + layer.den_1[b];
            let qh = layer.eq[a] + layer.eq[b];
            RoundSums {
                numer_first: layer.eq[a] * numer,
                denom_first: layer.eq[a] * (layer.den_0[a] * layer.den_1[a]),
                numer_second: qh * (d0h * n1h + d1h * n0h),
                denom_second: qh * (d0h * d1h),
                eq_sum: qh,
            }
        })
        .reduce(RoundSums::zero, RoundSums::add)
}

/// The same round in packed lanes, with the even and odd halves deinterleaved
/// up front: the CPU baseline the kernel has to beat.
struct PackedLayer {
    even: [Vec<PEF>; 5],
    odd: [Vec<PEF>; 5],
}

impl PackedLayer {
    fn new(layer: &Layer) -> Self {
        let mut even: Vec<Vec<PEF>> = Vec::with_capacity(5);
        let mut odd: Vec<Vec<PEF>> = Vec::with_capacity(5);
        for src in [&layer.num_0, &layer.num_1, &layer.den_0, &layer.den_1, &layer.eq] {
            let (e, o) = split_pack_ext(src);
            even.push(e);
            odd.push(o);
        }
        let take = |v: &mut Vec<Vec<PEF>>| -> [Vec<PEF>; 5] {
            let mut it = v.drain(..);
            std::array::from_fn(|_| it.next().unwrap())
        };
        Self { even: take(&mut even), odd: take(&mut odd) }
    }
}

/// Rows per rayon task: enough that task overhead vanishes, few enough that
/// every core stays fed.
const CPU_CHUNK: usize = 1 << 10;

fn cpu_packed(p: &PackedLayer) -> [PEF; 5] {
    let groups = p.even[0].len();
    (0..groups.div_ceil(CPU_CHUNK))
        .into_par_iter()
        .map(|ti| {
            let g0 = ti * CPU_CHUNK;
            let g1 = ((ti + 1) * CPU_CHUNK).min(groups);
            let mut acc = [PEF::zero(); 5];
            for g in g0..g1 {
                let (n0a, n1a, d0a, d1a, qa) = (
                    p.even[0][g], p.even[1][g], p.even[2][g], p.even[3][g], p.even[4][g],
                );
                let (n0b, n1b, d0b, d1b, qb) =
                    (p.odd[0][g], p.odd[1][g], p.odd[2][g], p.odd[3][g], p.odd[4][g]);
                acc[0] += qa * (d0a * n1a + d1a * n0a);
                acc[1] += qa * (d0a * d1a);
                let (n0h, n1h) = (n0a + n0b, n1a + n1b);
                let (d0h, d1h) = (d0a + d0b, d1a + d1b);
                let qh = qa + qb;
                acc[2] += qh * (d0h * n1h + d1h * n0h);
                acc[3] += qh * (d0h * d1h);
                acc[4] += qh;
            }
            acc
        })
        .reduce(|| [PEF::zero(); 5], |a, b| std::array::from_fn(|k| a[k] + b[k]))
}

fn run_gpu(
    gpu: &Gpu,
    layer: &Layer,
    reps: usize,
) -> Result<(RoundSums, crate::gpu::BenchTimes), String> {
    let n = layer.eq.len();
    let pairs = n / 2;
    let groups = pairs.div_ceil(THREADGROUP_SIZE as usize);
    let bufs: Vec<_> = [&layer.num_0, &layer.num_1, &layer.den_0, &layer.den_1, &layer.eq]
        .iter()
        .map(|v| gpu.buffer_from(&monty4_vec_of_ef(v)))
        .collect();
    let buf_partials = gpu.buffer_empty::<[u32; 4]>(groups * 5);
    let pso = gpu.pipeline("gkr_round")?;
    let pairs32 = pairs as u32;
    let times = gpu.bench_encoded(reps.max(1), 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        set_bytes(enc, 5, &pairs32);
        enc.set_buffer(6, Some(&buf_partials), 0);
        dispatch_groups(enc, groups as u64);
    });
    let partials: Vec<[u32; 4]> = gpu.read_vec(&buf_partials, groups * 5);
    let mut sums = RoundSums::zero();
    for g in 0..groups {
        sums = sums.add(RoundSums {
            numer_first: ef_of_monty4(partials[g * 5]),
            denom_first: ef_of_monty4(partials[g * 5 + 1]),
            numer_second: ef_of_monty4(partials[g * 5 + 2]),
            denom_second: ef_of_monty4(partials[g * 5 + 3]),
            eq_sum: ef_of_monty4(partials[g * 5 + 4]),
        });
    }
    Ok((sums, times))
}

fn unpack(acc: &[PEF; 5]) -> RoundSums {
    let s: Vec<EF> = acc.iter().map(crate::packed::sum_lanes).collect();
    RoundSums {
        numer_first: s[0],
        denom_first: s[1],
        numer_second: s[2],
        denom_second: s[3],
        eq_sum: s[4],
    }
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(71);
    let n = 4096usize;
    let layer = gen_layer(&mut rng, n);
    let expected = cpu_reference(&layer);
    let (got, _) = run_gpu(gpu, &layer, 1)?;
    if got != expected {
        return Err(format!("gkr_round mismatch: got {got:?}, expected {expected:?}"));
    }
    let packed = unpack(&cpu_packed(&PackedLayer::new(&layer)));
    if packed != expected {
        return Err(format!("packed CPU GKR round mismatch: got {packed:?}"));
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n: usize, reps: usize, with_cpu: bool) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(72);
    let layer = gen_layer(&mut rng, n);
    let (_, times) = run_gpu(gpu, &layer, reps)?;
    // Five extension multiplications per row, each sixteen base multiplications,
    // against five extension words read per row.
    let base_muls = n as f64 * 5.0 * 16.0;
    let bytes = (5 * n * 16) as f64;
    let mut result = json!({
        "probe": "gkr",
        "params": { "n_rows": n, "pairs": n / 2, "reps": reps },
        "times": times_json(&times),
        "mega_rows_per_s_best": n as f64 / times.best / 1e6,
        "giga_base_muls_per_s_best": base_muls / times.best / 1e9,
        "gbytes_per_s_best": bytes / times.best / 1e9,
    });
    if with_cpu {
        // The scalar version is not a straw man: sp1-hypercube's own GKR round
        // is written on scalar extension elements over rayon, so this is the
        // arithmetic the prover runs today. The distance between it and the
        // packed baseline below is CPU headroom sitting in the largest phase of
        // the proof.
        let t_scalar = cpu_bench(reps, || {
            std::hint::black_box(cpu_reference(&layer));
        });
        result["cpu_scalar"] = json!({
            "times": times_json(&t_scalar),
            "mega_rows_per_s_best": n as f64 / t_scalar.best / 1e6,
            "giga_base_muls_per_s_best": base_muls / t_scalar.best / 1e9,
            "note": "scalar extension elements over rayon, as sp1-hypercube writes it",
        });
        let packed = PackedLayer::new(&layer);
        let t = cpu_bench(reps, || {
            std::hint::black_box(cpu_packed(&packed));
        });
        result["cpu_packed"] = json!({
            "times": times_json(&t),
            "lanes": LANES,
            "mega_rows_per_s_best": n as f64 / t.best / 1e6,
            "giga_base_muls_per_s_best": base_muls / t.best / 1e9,
            "gbytes_per_s_best": bytes / t.best / 1e9,
            "note": "packed (NEON) lanes, deinterleaved once up front, rayon on all cores",
        });
        result["gpu_over_cpu_packed"] = json!(t.best / times.best);
        result["packed_over_scalar_cpu"] = json!(t_scalar.best / t.best);
    }
    Ok(result)
}
