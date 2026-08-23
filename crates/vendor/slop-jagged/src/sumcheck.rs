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

#[cfg(test)]
mod restack_tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use slop_algebra::AbstractField;
    use slop_baby_bear::BabyBear;
    use slop_stacked::interleave_multilinears_with_fixed_rate;

    use super::*;

    /// [`restack`] against the function it replaces, at shapes that reach every
    /// path through it.
    ///
    /// The proof only ever asks for the case where the data fits one stack, so
    /// the cases where it does not -- several full stacks, a partial tail, an
    /// exact fit -- are only ever covered here.
    #[test]
    fn restack_matches_upstream() {
        let mut rng = StdRng::seed_from_u64(7);
        // (component shapes as (rows, polynomials), log stacking height)
        let cases: &[(&[(usize, usize)], u32)] = &[
            (&[(8, 1)], 5),           // one component, one polynomial, fits exactly
            (&[(8, 3)], 6),           // one component, a real transpose, tail padded
            (&[(4, 2), (8, 1)], 4),   // two components, exactly two stacks
            (&[(8, 4), (4, 3)], 3),   // several full stacks and a partial tail
            (&[(16, 2), (2, 1)], 7),  // one stack with room to spare
            (&[(4, 1), (4, 1)], 3),   // an exact fit across components
        ];
        for (shapes, log_stacking_height) in cases {
            let mles = shapes
                .iter()
                .map(|&(rows, polys)| {
                    let guts: Vec<BabyBear> =
                        (0..rows * polys).map(|_| rng.gen::<BabyBear>()).collect();
                    Mle::new(Tensor::from(guts).reshape([rows, polys]))
                })
                .collect::<Vec<_>>();
            let message = Message::from(mles);
            let want =
                interleave_multilinears_with_fixed_rate(1, message.clone(), *log_stacking_height);
            let got = restack(&message, *log_stacking_height);
            assert_eq!(got.len(), want.len(), "component count at {shapes:?}");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g.guts().sizes(), w.guts().sizes(), "shape of {i} at {shapes:?}");
                assert_eq!(
                    g.guts().as_slice(),
                    w.guts().as_slice(),
                    "values of {i} at {shapes:?}"
                );
            }
        }
    }

    /// An empty batch, which upstream answers with a single zero-width
    /// component rather than with nothing.
    #[test]
    fn restack_of_nothing_matches_upstream() {
        let message: Message<Mle<BabyBear>> = Message::from(Vec::<Mle<BabyBear>>::new());
        let want = interleave_multilinears_with_fixed_rate(1, message.clone(), 4);
        let got = restack(&message, 4);
        assert_eq!(got.len(), want.len());
        assert_eq!(got[0].guts().sizes(), want[0].guts().sizes());
        let _ = BabyBear::one();
    }
}
