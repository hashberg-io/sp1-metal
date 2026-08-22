//! The Metal GKR round reduction against the CPU one it replaces.
//!
//! Both are called on the same polynomial, so the only difference between the
//! two answers can be where the arithmetic ran. The shapes cover what the
//! prover produces: several chips with different interaction counts in one
//! layer, an even and an odd number of rows (the odd case is the one where the
//! last row pair is half padding), and a chip whose rows are padded away
//! entirely.

use std::marker::PhantomData;
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use slop_algebra::AbstractField;
use slop_alloc::CpuBackend;
use slop_multilinear::{Mle, Padding, PaddedMle, Point};
use slop_tensor::Tensor;
use sp1_hypercube::{
    cpu_round_evals, LogUpGkrCpuLayer, LogupRoundOps, LogupRoundPolynomial, PolynomialLayer,
};
use sp1_metal_prover::{MetalRoundOps, EF, F};

/// The prover leaves small layers on the CPU, because below a few thousand
/// entries preparing a dispatch costs more than the arithmetic saves. These
/// shapes are deliberately small, so the threshold is lifted to keep the
/// kernels under test.
fn force_gpu() {
    unsafe { std::env::set_var("SP1_METAL_MIN_ENTRIES", "0") };
}

fn random_layer(
    rng: &mut StdRng,
    interaction_counts: &[usize],
    num_rows: usize,
    num_row_variables: usize,
    num_interaction_variables: usize,
) -> LogUpGkrCpuLayer<F, EF> {
    let zeros = |mle: Mle<F>| {
        PaddedMle::padded_with_zeros(Arc::new(mle), num_row_variables as u32)
    };
    let ones = |mle: Mle<EF>| {
        let n = mle.num_polynomials();
        PaddedMle::padded(
            Arc::new(mle),
            num_row_variables as u32,
            Padding::Constant((EF::one(), n, CpuBackend)),
        )
    };
    let mut nums = |rng: &mut StdRng| -> Vec<PaddedMle<F>> {
        interaction_counts
            .iter()
            .map(|&c| zeros(Mle::new(Tensor::<F>::rand(rng, [num_rows, c]))))
            .collect()
    };
    let numerator_0 = nums(rng);
    let numerator_1 = nums(rng);
    let mut dens = |rng: &mut StdRng| -> Vec<PaddedMle<EF>> {
        interaction_counts
            .iter()
            .map(|&c| ones(Mle::new(Tensor::<EF>::rand(rng, [num_rows, c]))))
            .collect()
    };
    let denominator_0 = dens(rng);
    let denominator_1 = dens(rng);
    LogUpGkrCpuLayer {
        numerator_0,
        denominator_0,
        numerator_1,
        denominator_1,
        num_row_variables,
        num_interaction_variables,
    }
}

fn poly<B>(
    rng: &mut StdRng,
    layer: LogUpGkrCpuLayer<F, EF>,
    num_row_variables: usize,
    num_interaction_variables: usize,
) -> LogupRoundPolynomial<F, EF, B> {
    let row_point: Point<EF> = Point::from((0..num_row_variables).map(|_| rng.gen()).collect::<Vec<EF>>());
    let interaction_point: Point<EF> =
        Point::from((0..num_interaction_variables).map(|_| rng.gen()).collect::<Vec<EF>>());
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

/// The same layer with extension-field numerators, which is every layer after
/// the first and so exercises the other kernel.
fn random_ext_layer(
    rng: &mut StdRng,
    interaction_counts: &[usize],
    num_rows: usize,
    num_row_variables: usize,
    num_interaction_variables: usize,
) -> LogUpGkrCpuLayer<EF, EF> {
    let zeros = |mle: Mle<EF>| {
        PaddedMle::padded_with_zeros(Arc::new(mle), num_row_variables as u32)
    };
    let ones = |mle: Mle<EF>| {
        let n = mle.num_polynomials();
        PaddedMle::padded(
            Arc::new(mle),
            num_row_variables as u32,
            Padding::Constant((EF::one(), n, CpuBackend)),
        )
    };
    let mut nums = |rng: &mut StdRng| -> Vec<PaddedMle<EF>> {
        interaction_counts
            .iter()
            .map(|&c| zeros(Mle::new(Tensor::<EF>::rand(rng, [num_rows, c]))))
            .collect()
    };
    let numerator_0 = nums(rng);
    let numerator_1 = nums(rng);
    let mut dens = |rng: &mut StdRng| -> Vec<PaddedMle<EF>> {
        interaction_counts
            .iter()
            .map(|&c| ones(Mle::new(Tensor::<EF>::rand(rng, [num_rows, c]))))
            .collect()
    };
    let denominator_0 = dens(rng);
    let denominator_1 = dens(rng);
    LogUpGkrCpuLayer {
        numerator_0,
        denominator_0,
        numerator_1,
        denominator_1,
        num_row_variables,
        num_interaction_variables,
    }
}

fn ext_poly<B>(
    rng: &mut StdRng,
    layer: LogUpGkrCpuLayer<EF, EF>,
    num_row_variables: usize,
    num_interaction_variables: usize,
) -> LogupRoundPolynomial<EF, EF, B> {
    let row_point: Point<EF> =
        Point::from((0..num_row_variables).map(|_| rng.gen()).collect::<Vec<EF>>());
    let interaction_point: Point<EF> =
        Point::from((0..num_interaction_variables).map(|_| rng.gen()).collect::<Vec<EF>>());
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

#[test]
fn logup_round_reduction_matches_the_cpu_on_extension_layers() {
    force_gpu();
    let mut rng = StdRng::seed_from_u64(1002);
    let shapes: [(&[usize], usize, usize, usize); 4] = [
        (&[3, 5, 2], 8, 3, 4),
        (&[4], 16, 4, 2),
        (&[2, 7], 5, 3, 4),
        (&[1, 1, 1, 6], 32, 5, 4),
    ];
    for (counts, rows, rv, iv) in shapes {
        let layer = random_ext_layer(&mut rng, counts, rows, rv, iv);
        let metal: LogupRoundPolynomial<EF, EF, MetalRoundOps> =
            ext_poly(&mut rng.clone(), layer, rv, iv);
        let cpu = cpu_round_evals(&metal);
        let got = <MetalRoundOps as LogupRoundOps<EF, EF>>::round_evals(&metal);
        assert_eq!(got.0, cpu.0, "eval at zero differs for {counts:?} over {rows} rows");
        assert_eq!(got.1, cpu.1, "eval at the pair-sum point differs for {counts:?}");
        assert_eq!(got.2, cpu.2, "eq sum differs for {counts:?}");
    }
}

#[test]
fn logup_round_reduction_matches_the_cpu() {
    force_gpu();
    let mut rng = StdRng::seed_from_u64(1001);
    // (interaction counts per chip, rows, row variables, interaction variables)
    let shapes: [(&[usize], usize, usize, usize); 4] = [
        (&[3, 5, 2], 8, 3, 4),
        (&[4], 16, 4, 2),
        (&[2, 7], 5, 3, 4),
        (&[1, 1, 1, 6], 32, 5, 4),
    ];
    for (counts, rows, rv, iv) in shapes {
        let total: usize = counts.iter().sum();
        assert!(total <= 1 << iv, "interactions must fit the interaction variables");
        let layer = random_layer(&mut rng, counts, rows, rv, iv);
        let metal: LogupRoundPolynomial<F, EF, MetalRoundOps> =
            poly(&mut rng.clone(), layer, rv, iv);
        let cpu = cpu_round_evals(&metal);
        let got = <MetalRoundOps as LogupRoundOps<F, EF>>::round_evals(&metal);
        assert_eq!(got.0, cpu.0, "eval at zero differs for {counts:?} over {rows} rows");
        assert_eq!(got.1, cpu.1, "eval at the pair-sum point differs for {counts:?}");
        assert_eq!(got.2, cpu.2, "eq sum differs for {counts:?}");
    }
}

/// A layer whose chips straddle the threshold, so some are computed on the GPU
/// and some on the CPU and the two sets of sums are added together. This is the
/// shape a real layer has, and the arithmetic has to agree either way.
#[test]
fn a_mixed_layer_agrees_with_the_cpu() {
    unsafe { std::env::set_var("SP1_METAL_MIN_ENTRIES", "64") };
    let mut rng = StdRng::seed_from_u64(1004);
    // Counts either side of 64 entries per chip at 16 rows.
    let counts: &[usize] = &[1, 8, 2, 16, 3];
    let (rows, rv, iv) = (16usize, 4usize, 5usize);
    let layer = random_layer(&mut rng, counts, rows, rv, iv);
    let metal: LogupRoundPolynomial<F, EF, MetalRoundOps> = poly(&mut rng.clone(), layer, rv, iv);
    let cpu = cpu_round_evals(&metal);
    let got = <MetalRoundOps as LogupRoundOps<F, EF>>::round_evals(&metal);
    assert_eq!(got.0, cpu.0, "mixed layer: eval at zero differs");
    assert_eq!(got.1, cpu.1, "mixed layer: eval at the pair-sum point differs");
    assert_eq!(got.2, cpu.2, "mixed layer: eq sum differs");
}

/// The fold against slop's own, which is what the CPU default calls.
#[test]
fn fixing_the_last_variable_matches_slop() {
    force_gpu();
    let mut rng = StdRng::seed_from_u64(1003);
    for &(rows, width, vars) in &[(8usize, 3usize, 3u32), (5, 7, 3), (1, 4, 1), (33, 2, 6)] {
        let alpha: EF = rng.gen();

        let base = PaddedMle::padded_with_zeros(
            Arc::new(Mle::new(Tensor::<F>::rand(&mut rng, [rows, width]))),
            vars,
        );
        let want = base.fix_last_variable::<EF>(alpha);
        let got = <MetalRoundOps as LogupRoundOps<F, EF>>::fix_numerator(&base, alpha);
        assert_eq!(
            got.inner().as_ref().map(|m| m.guts().as_slice().to_vec()),
            want.inner().as_ref().map(|m| m.guts().as_slice().to_vec()),
            "base fold differs at {rows}x{width}"
        );
        assert_eq!(got.num_variables(), want.num_variables());

        let inner = Arc::new(Mle::new(Tensor::<EF>::rand(&mut rng, [rows, width])));
        let ext =
            PaddedMle::padded(inner, vars, Padding::Constant((EF::one(), width, CpuBackend)));
        let want = ext.fix_last_variable::<EF>(alpha);
        let got = <MetalRoundOps as LogupRoundOps<EF, EF>>::fix_numerator(&ext, alpha);
        assert_eq!(
            got.inner().as_ref().map(|m| m.guts().as_slice().to_vec()),
            want.inner().as_ref().map(|m| m.guts().as_slice().to_vec()),
            "extension fold differs at {rows}x{width}"
        );
    }
}
