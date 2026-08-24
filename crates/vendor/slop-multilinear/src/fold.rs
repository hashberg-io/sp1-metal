// Modified from slop-multilinear 6.4.0 by the sp1-metal project; see CHANGES.md.
use rayon::prelude::*;
use slop_algebra::Field;
use slop_alloc::CpuBackend;
use slop_tensor::Tensor;

use crate::num_polynomials;

/// Compute the random linear combination of the even and odd coefficients of `vals`.
///
/// This is used to reduce the two evaluation claims for new_point into a
/// single evaluation claim.
pub fn fold_mle<F: Field>(guts: &Tensor<F, CpuBackend>, beta: F) -> Tensor<F, CpuBackend> {
    assert_eq!(num_polynomials(guts), 1, "this is only supported for a single polynomial");
    assert_eq!(guts.total_len() % 2, 0, "this is only supported for tensor of even length");

    #[cfg(feature = "metal")]
    let started = std::time::Instant::now();

    let fold_guts = guts
        .as_buffer()
        .par_iter()
        .step_by(2)
        .copied()
        .zip(guts.as_buffer().par_iter().skip(1).step_by(2).copied())
        .map(|(a, b)| a + beta * b)
        .collect::<Vec<_>>();
    let dim = fold_guts.len();
    #[cfg(feature = "metal")]
    crate::metal::FOLD_MLE.record(dim, 1, started.elapsed().as_nanos() as u64);
    Tensor::from(fold_guts).reshape([dim, 1])
}
