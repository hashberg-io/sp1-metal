// Modified from slop-jagged 6.4.0 by the sp1-metal project; see CHANGES.md.
use std::{marker::PhantomData, sync::Arc};

use itertools::Itertools;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use slop_algebra::{interpolate_univariate_polynomial, ExtensionField, Field};
use slop_alloc::{Backend, Buffer, CpuBackend};
use slop_challenger::{FieldChallenger, VariableLengthChallenger};
use slop_multilinear::Point;

use crate::BranchingProgram;

use super::JaggedEvalSumcheckPoly;

/// A trait for the jagged assist's sum as poly.
pub trait JaggedAssistSumAsPoly<
    F: Field,
    EF: ExtensionField<F>,
    A: Backend,
    Challenger: FieldChallenger<F> + Send + Sync,
    DeviceChallenger,
>: Sized + Send + Sync
{
    /// Construct a new sumcheck instance from the parameters.
    fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: Arc<Vec<Point<F>>>,
        z_col_eq_vals: Vec<EF>,
        backend: A,
    ) -> Self;

    #[allow(clippy::too_many_arguments)]
    /// Compute the sum as a polynomial in the last varaible, storing the result in `sum_values`,
    /// then sample randomness, storing the result in `rhos`. Expected to return the evaluation
    /// of the polynomial at the sampled point.
    fn sum_as_poly_and_sample_into_point(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF, A>,
        intermediate_eq_full_evals: &Buffer<EF, A>,
        sum_values: &mut Buffer<EF, A>,
        challenger: &mut DeviceChallenger,
        claim: EF,
        rhos: Point<EF, A>,
    ) -> (EF, Point<EF, A>);

    /// Fix the last variable of the polynomial, returning a new polynomial with one less variable.
    /// The zeroth coordinate of randomness_point is used for fixing the last variable.
    fn fix_last_variable(
        poly: JaggedEvalSumcheckPoly<F, EF, Challenger, DeviceChallenger, Self, A>,
    ) -> JaggedEvalSumcheckPoly<F, EF, Challenger, DeviceChallenger, Self, A>;
}

#[derive(Debug, Clone, Default)]
pub struct JaggedAssistSumAsPolyCPUImpl<F: Field, EF: ExtensionField<F>, Challenger> {
    branching_program: BranchingProgram<EF>,
    merged_prefix_sums: Arc<Vec<Point<F>>>,
    half: EF,
    _marker: PhantomData<Challenger>,
}

impl<F: Field, EF: ExtensionField<F>, Challenger: FieldChallenger<F>>
    JaggedAssistSumAsPolyCPUImpl<F, EF, Challenger>
{
    #[inline]
    fn eval(
        &self,
        lambda: EF,
        round_num: usize,
        merged_prefix_sum: &Point<F>,
        z_col_eq_val: EF,
        intermediate_eq_full_eval: EF,
        rhos: &Point<EF>,
    ) -> EF {
        // We want to calculate eq(z_col, col_idx) * eq(x_1, (x, rho)) * h(x_2, x, rho) where
        // x_1 || x_2 = merged_prefix_sum and rho is the sumcheck random point.  Note that the
        // eq(x_col, col_idx) is already computed as `z_col_eq_val` and all but one term of eq(x_i,
        // (x, rho)) is computed as `intermediate_eq_full_eval`.

        // Split the merged prefix sum so that x_1 || x_2 = merged_prefix_sum.
        let (h_prefix_sum, eq_prefix_sum) =
            merged_prefix_sum.split_at(merged_prefix_sum.dimension() - round_num - 1);

        // Compute the remaining eq term for `eq(x_i, (x, rho))`.
        let eq_val = if lambda == EF::zero() {
            EF::one() - *eq_prefix_sum.values()[0]
        } else if lambda == self.half {
            self.half
        } else {
            unreachable!("lambda must be 0 or 1/2")
        };

        // Compute full eval of eq(x_i, (x, rho))
        let eq_eval = intermediate_eq_full_eval * eq_val;

        // Compute eval of h(x_2, x, rho).
        let mut h_prefix_sum: Point<EF> =
            h_prefix_sum.to_vec().iter().map(|x| (*x).into()).collect::<Vec<_>>().into();
        h_prefix_sum.add_dimension_back(lambda);
        h_prefix_sum.extend(rhos);
        let num_dimensions = h_prefix_sum.dimension();
        let (h_left, h_right) = h_prefix_sum.split_at(num_dimensions / 2);
        let h_eval = self.branching_program.eval(&h_left, &h_right);

        z_col_eq_val * h_eval * eq_eval
    }
}

#[cfg(feature = "metal")]
impl<F: Field, EF: ExtensionField<F>, Challenger: FieldChallenger<F>>
    JaggedAssistSumAsPolyCPUImpl<F, EF, Challenger>
{
    /// One round of the assist sumcheck on the GPU, or `None` if it declined.
    ///
    /// The prefix sums arrive as a vector of points and the kernel wants them
    /// flat, so they are copied into one buffer here. That is `n` times the
    /// point's dimension words against `n` times its dimension times a
    /// sixteen-way fold of work, so it does not show up.
    fn gpu_round(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF>,
        intermediate_eq_full_evals: &Buffer<EF>,
        rhos: &Point<EF>,
    ) -> Option<(EF, EF)> {
        let n = z_col_eq_vals.len();
        let mps_dim = self.merged_prefix_sums.first()?.dimension();
        if n == 0 || self.merged_prefix_sums.len() < n {
            return None;
        }
        if self.merged_prefix_sums.iter().take(n).any(|p| p.dimension() != mps_dim) {
            return None;
        }
        let mut flat: Vec<F> = Vec::with_capacity(n * mps_dim);
        for point in self.merged_prefix_sums.iter().take(n) {
            flat.extend(point.iter().copied());
        }
        let rho_vals: Vec<EF> = rhos.iter().copied().collect();
        let (z_row, z_index) = self.branching_program.points();
        let z_row: Vec<EF> = z_row.iter().cloned().collect();
        let z_index: Vec<EF> = z_index.iter().cloned().collect();
        let table = crate::poly::BranchingProgram::<EF>::transition_table();
        let mut out = vec![EF::zero(); 2 * n];
        if !crate::metal::jagged_assist_typed(
            &flat,
            mps_dim,
            z_col_eq_vals.as_slice(),
            intermediate_eq_full_evals.as_slice(),
            &rho_vals,
            &z_row,
            &z_index,
            &table,
            round_num,
            self.branching_program.num_vars,
            self.half,
            &mut out,
        ) {
            return None;
        }
        let mut y_0 = EF::zero();
        let mut y_half = EF::zero();
        for pair in out.chunks_exact(2) {
            y_0 += pair[0];
            y_half += pair[1];
        }
        Some((y_0, y_half))
    }
}

impl<F: Field, EF: ExtensionField<F>, Challenger: FieldChallenger<F> + Send + Sync>
    JaggedAssistSumAsPoly<F, EF, CpuBackend, Challenger, Challenger>
    for JaggedAssistSumAsPolyCPUImpl<F, EF, Challenger>
{
    fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: Arc<Vec<Point<F>>>,
        _z_col_eq_vals: Vec<EF>,
        _backend: CpuBackend,
    ) -> Self {
        let branching_program = BranchingProgram::new(z_row, z_index);

        Self {
            branching_program,
            merged_prefix_sums,
            half: EF::two().inverse(),
            _marker: PhantomData,
        }
    }

    fn sum_as_poly_and_sample_into_point(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF>,
        intermediate_eq_full_evals: &Buffer<EF>,
        sum_values: &mut Buffer<EF>,
        challenger: &mut Challenger,
        claim: EF,
        rhos: Point<EF>,
    ) -> (EF, Point<EF>) {
        let mut rhos = rhos.clone();
        // Every column's branching program is independent, and the transition
        // rule is a fixed table, so the whole round is one dispatch.
        #[cfg(feature = "metal")]
        let from_gpu = self.gpu_round(round_num, z_col_eq_vals, intermediate_eq_full_evals, &rhos);
        #[cfg(not(feature = "metal"))]
        let from_gpu: Option<(EF, EF)> = None;
        #[cfg(feature = "metal")]
        let checking = from_gpu.is_some() && crate::metal::assist_check();
        #[cfg(not(feature = "metal"))]
        let checking = false;

        // Calculate the partition chunk size.
        let chunk_size = std::cmp::max(z_col_eq_vals.len() / num_cpus::get(), 1);

        // Compute the values at x = 0 and x = 1/2.
        let (y_0, y_half) = if let (Some(pair), false) = (from_gpu, checking) {
            pair
        } else {
        self
            .merged_prefix_sums
            .par_chunks(chunk_size)
            .zip_eq(z_col_eq_vals.par_chunks(chunk_size))
            .zip_eq(intermediate_eq_full_evals.par_chunks(chunk_size))
            .map(
                |(
                    (merged_prefix_sum_chunk, z_col_eq_val_chunk),
                    intermediate_eq_full_eval_chunk,
                )| {
                    merged_prefix_sum_chunk
                        .iter()
                        .zip_eq(z_col_eq_val_chunk.iter())
                        .zip_eq(intermediate_eq_full_eval_chunk.iter())
                        .map(|((merged_prefix_sum, z_col_eq_val), intermediate_eq_full_eval)| {
                            let y_0 = self.eval(
                                EF::zero(),
                                round_num,
                                merged_prefix_sum,
                                *z_col_eq_val,
                                *intermediate_eq_full_eval,
                                &rhos,
                            );
                            let y_half = self.eval(
                                self.half,
                                round_num,
                                merged_prefix_sum,
                                *z_col_eq_val,
                                *intermediate_eq_full_eval,
                                &rhos,
                            );

                            (y_0, y_half)
                        })
                        .fold((EF::zero(), EF::zero()), |(y_0, y_2), (y_0_i, y_2_i)| {
                            (y_0 + y_0_i, y_2 + y_2_i)
                        })
                },
            )
            .reduce(
                || (EF::zero(), EF::zero()),
                |(y_0, y_2), (y_0_i, y_2_i)| (y_0 + y_0_i, y_2 + y_2_i),
            )
        };

        if let Some((g0, gh)) = from_gpu {
            assert!(
                !checking || (g0 == y_0 && gh == y_half),
                "the jagged assist round from the GPU differs from upstream's",
            );
        }

        // Store the values in the sum_values buffer.
        sum_values.as_mut_slice()[3 * round_num] = y_0;
        sum_values.as_mut_slice()[3 * round_num + 1] = y_half;
        let y_1 = claim - y_0;
        sum_values.as_mut_slice()[3 * round_num + 2] = y_1;

        // Interpolate the polynomial.
        let poly = interpolate_univariate_polynomial(
            &[EF::zero(), EF::two().inverse(), EF::one()],
            &[y_0, y_half, y_1],
        );

        // Observe and sample new randomness.
        challenger.observe_constant_length_extension_slice(&poly.coefficients);

        let alpha = challenger.sample_ext_element();
        rhos.add_dimension(alpha);

        // Return the new claim for the next round.
        (poly.eval_at_point(alpha), rhos.clone())
    }

    fn fix_last_variable(
        poly: JaggedEvalSumcheckPoly<F, EF, Challenger, Challenger, Self, CpuBackend>,
    ) -> JaggedEvalSumcheckPoly<F, EF, Challenger, Challenger, Self, CpuBackend> {
        // Add alpha to the rho point.
        let alpha = *poly.rho.first().unwrap();

        let merged_prefix_sum_dim = poly.prefix_sum_dimension as usize;

        // Update the intermediate full eq evals.
        let updated_intermediate_eq_full_evals = poly
            .merged_prefix_sums
            .chunks(merged_prefix_sum_dim)
            .zip_eq(poly.intermediate_eq_full_evals.iter())
            .map(|(merged_prefix_sum, intermediate_eq_full_eval)| {
                let x_i =
                    merged_prefix_sum.get(merged_prefix_sum_dim - 1 - poly.round_num).unwrap();
                *intermediate_eq_full_eval
                    * ((alpha * *x_i) + (EF::one() - alpha) * (EF::one() - *x_i))
            })
            .collect_vec();

        JaggedEvalSumcheckPoly::new(
            poly.bp_batch_eval,
            poly.rho,
            poly.z_col,
            poly.merged_prefix_sums,
            poly.z_col_eq_vals,
            poly.round_num + 1,
            updated_intermediate_eq_full_evals.into(),
            poly.half,
            poly.prefix_sum_dimension,
        )
    }
}
