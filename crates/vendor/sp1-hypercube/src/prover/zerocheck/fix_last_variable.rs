// Modified from sp1-hypercube 6.4.0 by the sp1-metal project; see CHANGES.md.
use std::ops::{Add, Mul};

use slop_algebra::{ExtensionField, Field};

use super::ZeroCheckPoly;

/// Where a zerocheck column fold runs.
///
/// This fold refers to no AIR — it is `PaddedMle::fix_last_variable` and
/// nothing else — so unlike the reduction beside it, it needs no per-chip
/// generated code and the GKR round's kernels serve it directly.
#[cfg(not(feature = "metal"))]
#[inline]
fn fix_columns<K, EF>(mle: &slop_multilinear::PaddedMle<K>, alpha: EF) -> slop_multilinear::PaddedMle<EF>
where
    K: Field,
    EF: ExtensionField<K>,
{
    mle.fix_last_variable(alpha)
}

/// Where a zerocheck column fold runs. See the note on the non-Metal definition.
#[cfg(feature = "metal")]
#[inline]
fn fix_columns<K, EF>(mle: &slop_multilinear::PaddedMle<K>, alpha: EF) -> slop_multilinear::PaddedMle<EF>
where
    K: Field,
    EF: ExtensionField<K>,
{
    crate::metal::fix_padded_mle(mle, alpha)
}

/// This function will set the last variable to `alpha`.
pub fn zerocheck_fix_last_variable<
    K: Field,
    F: Field,
    EF: ExtensionField<F> + Add<K, Output = EF> + Mul<K, Output = EF> + From<K> + ExtensionField<K>,
    A,
>(
    poly: ZeroCheckPoly<K, F, EF, A>,
    alpha: EF,
) -> ZeroCheckPoly<EF, F, EF, A> {
    let preprocessed_columns =
        poly.preprocessed_columns.as_ref().map(|mle| fix_columns(mle, alpha));
    let main_columns = fix_columns(&poly.main_columns, alpha);

    if poly.main_columns.num_real_entries() == 0 {
        // If the chip is pure padding, it's contribution to sumcheck is just zero, we don't need
        // to propagate any eq_adjustment or any other data relevant to the sumcheck.
        return ZeroCheckPoly::new(
            poly.air_data,
            poly.zeta,
            preprocessed_columns,
            main_columns,
            poly.eq_adjustment,
            poly.geq_value,
            poly.padded_row_adjustment,
            poly.virtual_geq.fix_last_variable(alpha),
        );
    }

    let (rest, last) = poly.zeta.split_at(poly.zeta.dimension() - 1);
    let last = *last[0];

    // When we are fixing the last variable, we can factor out one of the eq_terms, as it will be a
    // constant. That constant is equal to (alpha * last) + (1 - alpha) * (1 - last).
    let eq_adjustment =
        poly.eq_adjustment * ((alpha * last) + (EF::one() - alpha) * (EF::one() - last));

    let has_non_padded_vars = poly.main_columns.num_real_entries() > 1;

    let geq_value = if has_non_padded_vars {
        EF::zero()
    } else {
        (EF::one() - poly.geq_value) * alpha + poly.geq_value
    };

    ZeroCheckPoly::new(
        poly.air_data,
        rest,
        preprocessed_columns,
        main_columns,
        eq_adjustment,
        geq_value,
        poly.padded_row_adjustment,
        poly.virtual_geq.fix_last_variable(alpha),
    )
}
