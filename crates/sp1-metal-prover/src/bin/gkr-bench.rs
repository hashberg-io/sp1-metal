// SPDX-License-Identifier: MIT OR Apache-2.0
//! What the logup GKR round reduction costs, CPU against Metal.
//!
//! The shape is a circuit layer as the prover builds one: several chips with
//! different interaction counts, over a shard-sized number of rows. Both sides
//! are handed the same polynomial, so the only difference is where the
//! arithmetic runs.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use slop_algebra::AbstractField;
use slop_alloc::CpuBackend;
use slop_multilinear::{Mle, Padding, PaddedMle, Point};
use slop_tensor::Tensor;
use sp1_hypercube::{
    cpu_round_evals, CpuRoundOps, LogUpGkrCpuLayer, LogupRoundOps, LogupRoundPolynomial,
    PolynomialLayer,
};
use sp1_metal_prover::{MetalRoundOps, EF, F, STATS};

/// Interaction counts of a plausible mix of chips in one layer.
const CHIPS: [usize; 8] = [16, 8, 4, 32, 12, 6, 20, 10];

fn build(
    rng: &mut StdRng,
    rows: usize,
    rv: usize,
    iv: usize,
) -> LogupRoundPolynomial<F, EF, MetalRoundOps> {
    let zeros = |mle: Mle<F>| PaddedMle::padded_with_zeros(Arc::new(mle), rv as u32);
    let ones = |mle: Mle<EF>| {
        let n = mle.num_polynomials();
        PaddedMle::padded(Arc::new(mle), rv as u32, Padding::Constant((EF::one(), n, CpuBackend)))
    };
    let mut nums =
        |rng: &mut StdRng| CHIPS.iter().map(|&c| zeros(Mle::new(Tensor::<F>::rand(rng, [rows, c])))).collect();
    let numerator_0: Vec<_> = nums(rng);
    let numerator_1: Vec<_> = nums(rng);
    let mut dens =
        |rng: &mut StdRng| CHIPS.iter().map(|&c| ones(Mle::new(Tensor::<EF>::rand(rng, [rows, c])))).collect();
    let denominator_0: Vec<_> = dens(rng);
    let denominator_1: Vec<_> = dens(rng);
    let layer = LogUpGkrCpuLayer {
        numerator_0,
        denominator_0,
        numerator_1,
        denominator_1,
        num_row_variables: rv,
        num_interaction_variables: iv,
    };
    let row_point: Point<EF> = Point::from((0..rv).map(|_| rng.gen()).collect::<Vec<EF>>());
    let interaction_point: Point<EF> = Point::from((0..iv).map(|_| rng.gen()).collect::<Vec<EF>>());
    let mut point = interaction_point.clone();
    for c in row_point.iter() {
        point.add_dimension_back(*c);
    }
    LogupRoundPolynomial {
        layer: PolynomialLayer::CircuitLayer(layer),
        eq_row: Arc::new(Mle::partial_lagrange(&row_point)),
        eq_interaction: Arc::new(Mle::partial_lagrange(&interaction_point)),
        eq_adjustment: rng.gen(),
        padding_adjustment: rng.gen(),
        lambda: rng.gen(),
        point,
        ops: PhantomData,
    }
}

fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    f();
    (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(f());
            t0.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let log_rows: usize =
        std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let reps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let rows = 1usize << log_rows;
    let interactions: usize = CHIPS.iter().sum();
    let iv = interactions.next_power_of_two().trailing_zeros() as usize;

    let mut rng = StdRng::seed_from_u64(2001);
    let poly = build(&mut rng, rows, log_rows, iv);

    // Same values, seen as the CPU backend, so both sides get the same input.
    let cpu_view: &LogupRoundPolynomial<F, EF, CpuRoundOps> =
        unsafe { &*std::ptr::from_ref(&poly).cast() };

    let cpu = best(reps, || cpu_round_evals(cpu_view));
    STATS.reset();
    let gpu = best(reps, || <MetalRoundOps as LogupRoundOps<F, EF>>::round_evals(&poly));
    let (uploads, no_copy) = STATS.counts();

    let want = cpu_round_evals(cpu_view);
    let got = <MetalRoundOps as LogupRoundOps<F, EF>>::round_evals(&poly);
    assert_eq!(want, got, "benchmark disagrees with the reference");

    // The reduction is only half of a round: the fold that follows it reads the
    // same data again and is still on the CPU, so the phase-level factor is
    // bounded by how the round's time splits between the two.
    fn clone_poly<A, B>(p: &LogupRoundPolynomial<F, EF, A>) -> LogupRoundPolynomial<F, EF, B> {
        let PolynomialLayer::CircuitLayer(l) = &p.layer else { unreachable!() };
        LogupRoundPolynomial::<F, EF, B> {
            layer: PolynomialLayer::CircuitLayer(LogUpGkrCpuLayer {
                numerator_0: l.numerator_0.clone(),
                denominator_0: l.denominator_0.clone(),
                numerator_1: l.numerator_1.clone(),
                denominator_1: l.denominator_1.clone(),
                num_row_variables: l.num_row_variables,
                num_interaction_variables: l.num_interaction_variables,
            }),
            eq_row: p.eq_row.clone(),
            eq_interaction: p.eq_interaction.clone(),
            eq_adjustment: p.eq_adjustment,
            padding_adjustment: p.padding_adjustment,
            lambda: p.lambda,
            point: p.point.clone(),
            ops: PhantomData,
        }
    }
    let alpha: EF = rng.gen();
    let fold = best(reps, || {
        let fresh: LogupRoundPolynomial<F, EF, CpuRoundOps> = clone_poly(cpu_view);
        slop_sumcheck::SumcheckPolyFirstRound::fix_t_variables(fresh, alpha, 1)
    });
    let fold_metal = best(reps, || {
        let fresh: LogupRoundPolynomial<F, EF, MetalRoundOps> = clone_poly(&poly);
        slop_sumcheck::SumcheckPolyFirstRound::fix_t_variables(fresh, alpha, 1)
    });

    let entries = rows * interactions;
    println!("{} rows x {interactions} interactions across {} chips", rows, CHIPS.len());
    println!("  cpu   {:9.3} ms   {:7.1} M entries/s", cpu * 1e3, entries as f64 / cpu / 1e6);
    println!("  metal {:9.3} ms   {:7.1} M entries/s", gpu * 1e3, entries as f64 / gpu / 1e6);
    println!("  metal over cpu: {:.2}x", cpu / gpu);
    println!("  uploads: {uploads}, of which no-copy: {no_copy}");
    println!("  fold: cpu {:9.3} ms   metal {:9.3} ms   {:.2}x", fold * 1e3, fold_metal * 1e3, fold / fold_metal);
    let round_cpu = cpu + fold;
    println!(
        "  whole round: cpu {:.3} ms, reduction only on metal {:.3} ms ({:.2}x), both on metal {:.3} ms ({:.2}x)",
        round_cpu * 1e3,
        (gpu + fold) * 1e3,
        round_cpu / (gpu + fold),
        (gpu + fold_metal) * 1e3,
        round_cpu / (gpu + fold_metal)
    );
}
