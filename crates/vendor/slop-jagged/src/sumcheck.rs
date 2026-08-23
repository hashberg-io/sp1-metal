use std::sync::Arc;

use slop_algebra::{ExtensionField, Field};
use slop_commit::{Message, Rounds};
use rayon::prelude::*;
use slop_multilinear::{Mle, Point};
use slop_tensor::Tensor;

use crate::{
    populate::partial_jagged_multilinear, HadamardProduct, JaggedLittlePolynomialProverParams,
    LongMle,
};

pub(crate) fn jagged_sumcheck_poly<F: Field, EF: ExtensionField<F>>(
    base: Rounds<Message<Mle<F>>>,
    jagged_params: &JaggedLittlePolynomialProverParams,
    row_data: Rounds<Arc<Vec<usize>>>,
    column_data: Rounds<Arc<Vec<usize>>>,
    log_stacking_height: u32,
    z_row: &Point<EF>,
    z_col: &Point<EF>,
) -> HadamardProduct<F, EF> {
    let base = base.into_iter().flatten().collect::<Message<Mle<_, _>>>();
    let long_mle = LongMle::from_message(base, log_stacking_height);
    #[cfg(feature = "metal")]
    let started = std::time::Instant::now();
    let jaggled_mle =
        partial_jagged_multilinear(jagged_params, row_data, column_data, z_row, z_col, 1);
    #[cfg(feature = "metal")]
    let after_jagged = std::time::Instant::now();

    let total_num_variables = jaggled_mle.num_variables();

    let restacked_mle =
        LongMle::from_message(restack(long_mle.components(), total_num_variables), total_num_variables);
    #[cfg(feature = "metal")]
    {
        let cells = 1usize << total_num_variables;
        crate::metal::POLY[0].record(cells, (after_jagged - started).as_nanos() as u64);
        crate::metal::POLY[1].record(cells, after_jagged.elapsed().as_nanos() as u64);
    }

    HadamardProduct { base: restacked_mle, ext: jaggled_mle }
}

/// Lays a batch of MLEs end to end in column-major order, in stacks of
/// `2^log_stacking_height`.
///
/// This is `interleave_multilinears_with_fixed_rate` at a batch size of one,
/// which is the only rate the jagged sumcheck asks for, written to do the work
/// once. Upstream makes three passes over the whole polynomial: it transposes
/// each component into a fresh allocation, appends those into a running buffer,
/// and then transposes the result again -- and that last transpose is of a
/// one-row matrix, so it permutes nothing and copies everything.
///
/// Here each component is transposed straight into its place in the output, and
/// what comes out is sliced rather than transposed. Only the transposes that
/// actually permute survive, and a component with a single polynomial does not
/// even need one of those.
///
/// The result is identical to upstream's, including the shape of the final
/// stack when the data does not fill it: the tail is padded with zeros.
fn restack<F: Field>(mles: &Message<Mle<F>>, log_stacking_height: u32) -> Message<Mle<F>> {
    let stack = 1usize << log_stacking_height;
    let total = mles.iter().map(|mle| mle.guts().total_len()).sum::<usize>();
    if total == 0 {
        // Upstream still emits one component here, of width zero.
        return Message::from(vec![Mle::new(Tensor::from(Vec::new()).reshape([stack, 0]))]);
    }
    let out_len = total.next_multiple_of(stack);
    // Uninitialised but for the tail: every cell below `total` is written by one
    // of the components, and only the padding above it has to be zero.
    #[allow(clippy::uninit_vec)]
    let mut out: Vec<F> = {
        let mut v = Vec::with_capacity(out_len);
        // SAFETY: a field element is a plain array of words with no invalid bit
        // patterns and no drop glue, and every cell is written below.
        unsafe { v.set_len(out_len) };
        v[total..].fill(F::zero());
        v
    };

    // Each component owns a disjoint range of the output, so they transpose in
    // parallel. The ranges are carved out first because that is what makes the
    // disjointness something the compiler checks rather than something this
    // comment asserts.
    let mut rest = out.as_mut_slice();
    let mut pieces = Vec::with_capacity(mles.len());
    for mle in mles.iter() {
        let (dst, tail) = rest.split_at_mut(mle.guts().total_len());
        pieces.push(dst);
        rest = tail;
    }
    pieces.into_par_iter().zip(mles.par_iter()).for_each(|(dst, mle)| {
        let guts = mle.guts();
        let (height, width) = (guts.sizes()[0], guts.sizes()[1]);
        if width == 1 {
            // A single polynomial is already in column-major order.
            dst.copy_from_slice(guts.as_slice());
        } else {
            transpose::transpose(guts.as_slice(), dst, width, height);
        }
    });

    if out_len == stack {
        // The usual case: one stack, and it is the buffer already built.
        return Message::from(vec![Mle::new(Tensor::from(out).reshape([stack, 1]))]);
    }
    let components = out
        .chunks(stack)
        .map(|chunk| Mle::new(Tensor::from(chunk.to_vec()).reshape([stack, 1])))
        .collect::<Vec<_>>();
    Message::from(components)
}
