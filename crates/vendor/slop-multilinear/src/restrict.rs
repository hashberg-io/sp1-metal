// Modified from slop-multilinear 6.4.0 by the sp1-metal project; see CHANGES.md.
use rayon::prelude::*;
use std::{cmp::max, sync::Arc};

use slop_algebra::{ExtensionField, Field};
use slop_alloc::CpuBackend;
use slop_tensor::Tensor;

use crate::{eval_mle_at_point, num_polynomials, MleEval, Point};

/// Below this many cells a fold is cheaper to do on the calling thread than to
/// hand to rayon. See [`mle_fix_last_variable`].
const SERIAL_CELLS: usize = 1 << 13;

/// Fix the last variable of an MLE with generic padding values.
pub fn mle_fix_last_variable<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    alpha: EF,
    padding_values: Arc<MleEval<F, CpuBackend>>,
) -> Tensor<EF, CpuBackend>
where
    F: Field,
    EF: ExtensionField<F>,
{
    #[cfg(feature = "metal")]
    let started = std::time::Instant::now();

    let num_polynomials = num_polynomials(mle);
    let num_non_zero_elements_out = mle.sizes()[0].div_ceil(2);
    let result_size = num_non_zero_elements_out * num_polynomials;

    let mut result: Vec<EF> = Vec::with_capacity(result_size);

    #[allow(clippy::uninit_vec)]
    unsafe {
        result.set_len(result_size);
    }

    let result_chunk_size =
        max(num_non_zero_elements_out / num_cpus::get() * num_polynomials, num_polynomials);
    let mle_chunk_size = 2 * result_chunk_size;

    // At the two ends of the interpolation the fold is a choice rather than a
    // combination: `alpha * (y - x) + x` is `x` at zero and `y` at one. Both
    // come up constantly -- splitting a layer into its even and odd halves is
    // exactly a fold at zero and a fold at one -- and taking them as written
    // costs a multiply and a subtraction per cell to compute a copy.
    let take_even = alpha.is_zero();
    let take_odd = alpha.is_one();

    // One chunk of the fold, written into its own slice of the result.
    let fold_chunk = |mle_chunk: &[F], result_chunk: &mut [EF]| {
        let num_result_rows = result_chunk.len() / num_polynomials;

        (0..num_result_rows).for_each(|i| {
            (0..num_polynomials).for_each(|j| {
                let x = mle_chunk[(2 * i) * num_polynomials + j];
                if take_even {
                    result_chunk[i * num_polynomials + j] = x.into();
                    return;
                }
                let y = mle_chunk
                    .get((2 * i + 1) * num_polynomials + j)
                    .copied()
                    .unwrap_or_else(|| padding_values[j]);
                if take_odd {
                    result_chunk[i * num_polynomials + j] = y.into();
                    return;
                }
                // return alpha * y + (EF::one() - alpha) * x, but in a more efficient
                // way that minimizes extension field
                // multiplications.
                result_chunk[i * num_polynomials + j] = alpha * (y - x) + x;
            });
        });
    };

    // Small folds run on the calling thread.
    //
    // Almost every fold in a proof is small: sumcheck halves its polynomial each
    // round, so the last rounds are a handful of rows, and there are a great
    // many of them. Measured over the acceptance workload, 98,610 of 102,882
    // folds were under a thousand cells and they cost 1176 nanoseconds a cell
    // against 1.72 for the folds that carry the actual work -- seven hundred
    // times the unit cost, for four percent of the cells.
    //
    // That is not the arithmetic. It is what it costs to hand thirty-nine cells
    // to a work-stealing pool, which is also why two sampling profiles blamed
    // rayon and could not say who was calling it: rayon really was where the
    // time went, and the answer was to stop asking it.
    if result_size < SERIAL_CELLS {
        mle.as_slice()
            .chunks(mle_chunk_size)
            .zip(result.chunks_mut(result_chunk_size))
            .for_each(|(m, r)| fold_chunk(m, r));
    } else {
        // `par_chunks` rather than `chunks(..).par_bridge()`: the bridge pulls a
        // sequential iterator through a mutex and cannot split, so it pays for
        // parallelism it does not get.
        mle.as_slice()
            .par_chunks(mle_chunk_size)
            .zip(result.par_chunks_mut(result_chunk_size))
            .for_each(|(m, r)| fold_chunk(m, r));
    }

    #[cfg(feature = "metal")]
    crate::metal::record(
        num_non_zero_elements_out,
        num_polynomials,
        started.elapsed().as_nanos() as u64,
    );

    Tensor::from(result).reshape([num_non_zero_elements_out, num_polynomials])
}

/// Fix the last variable of an MLE with constant padding.
pub fn mle_fix_last_variable_constant_padding<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    alpha: EF,
    padding_value: F,
) -> Tensor<EF, CpuBackend>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let padding_values: MleEval<_> = vec![padding_value; num_polynomials(mle)].into();
    mle_fix_last_variable(mle, alpha, Arc::new(padding_values))
}

/// Compute the evaluation at a point where the last variable is fixed to zero.
pub fn mle_fixed_at_zero<F, EF>(
    mle: &Tensor<F, CpuBackend>,
    point: &Point<EF, CpuBackend>,
) -> Tensor<EF, CpuBackend>
where
    F: Field,
    EF: ExtensionField<F>,
{
    // TODO: A smarter way to do this is pre-cache the partial_lagrange_evals that are implicit
    // in `eval_at_point` so we don't recompute it at every step of BaseFold.
    #[cfg(feature = "metal")]
    let started = std::time::Instant::now();
    let even_values = mle.as_slice().par_iter().step_by(2).copied().collect::<Vec<_>>();
    #[cfg(feature = "metal")]
    crate::metal::FIXED_AT_ZERO.record(
        even_values.len(),
        1,
        started.elapsed().as_nanos() as u64,
    );
    eval_mle_at_point(&Tensor::from(even_values), point)
}
