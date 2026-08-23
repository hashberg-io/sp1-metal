use itertools::Itertools;
use rayon::prelude::*;
use slop_algebra::{ExtensionField, Field};
use slop_alloc::CpuBackend;
use slop_matrix::dense::RowMajorMatrix;
use slop_multilinear::{Mle, PaddedMle, Padding, Point};
use std::{collections::BTreeMap, sync::Arc};

use crate::{prover::Traces, Interaction};

use super::{LogUpGkrCpuLayer, LogUpGkrOutput, LogupGkrCpuTraceGenerator};

/// A vector whose elements are about to be written and are not read first.
///
/// The circuit's first layer allocates one of these per chip for the numerators
/// and one for the denominators, and every cell of both is written before
/// anything reads them -- by the kernel when the chip goes to the GPU, and by a
/// `par_chunks_exact_mut` over the whole array when it does not. Filling them
/// first is a wasted pass over the largest arrays in the phase, and the
/// denominators are filled with ones rather than zeros, so it is a real loop
/// rather than a memset.
#[allow(clippy::uninit_vec)]
fn uninit_vec<T>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    // SAFETY: `T` is a field element -- a plain array of words with no invalid
    // bit patterns and no drop glue -- and every element is written before it
    // is read. See the note above for why that holds on both paths.
    unsafe { v.set_len(len) };
    v
}

pub(crate) fn generate_interaction_vals<F: Field, EF: ExtensionField<F>>(
    interaction: &Interaction<F>,
    preprocessed_row: &[F],
    main_row: &[F],
    is_send: bool,
    alpha: EF,
    betas: &[EF],
) -> (F, EF) {
    let mut denominator = alpha;
    let mut betas = betas.iter();
    denominator += *betas.next().unwrap() * EF::from_canonical_usize(interaction.argument_index());
    for (columns, beta) in interaction.values.iter().zip(betas) {
        let apply = columns.apply::<F, F>(preprocessed_row, main_row);
        denominator += *beta * apply;
    }
    let mut mult = interaction.multiplicity.apply::<F, F>(preprocessed_row, main_row);

    if !is_send {
        mult = -mult;
    }

    (mult, denominator)
}

impl<F: Field, EF: ExtensionField<F>, A> LogupGkrCpuTraceGenerator<F, EF, A> {
    #[allow(clippy::unused_self)]
    pub(crate) fn extract_outputs(
        &self,
        last_layer: &LogUpGkrCpuLayer<EF, EF>,
    ) -> LogUpGkrOutput<EF> {
        let numerator_0 = last_layer.numerator_0.clone();
        let numerator_1 = last_layer.numerator_1.clone();
        let denominator_0 = last_layer.denominator_0.clone();
        let denominator_1 = last_layer.denominator_1.clone();

        let mut numerator_0_interactions: Vec<EF> = numerator_0
            .into_iter()
            .flat_map(|mle| {
                let n00 = mle.fix_last_variable(EF::zero());
                let n01 = mle.fix_last_variable(EF::one());
                let n00_int = n00.eval_at::<EF>(&Point::from(vec![])).to_vec();
                let n01_int = n01.eval_at::<EF>(&Point::from(vec![])).to_vec();
                n00_int.iter().interleave(n01_int.iter()).copied().collect::<Vec<_>>()
            })
            .collect();
        numerator_0_interactions
            .resize(1 << (last_layer.num_interaction_variables + 1), EF::zero());
        let mut numerator_1_interactions: Vec<EF> = numerator_1
            .into_iter()
            .flat_map(|mle| {
                let n10 = mle.fix_last_variable(EF::zero());
                let n11 = mle.fix_last_variable(EF::one());
                let n10_int = n10.eval_at::<EF>(&Point::from(vec![])).to_vec();
                let n11_int = n11.eval_at::<EF>(&Point::from(vec![])).to_vec();
                n10_int.iter().interleave(n11_int.iter()).copied().collect::<Vec<_>>()
            })
            .collect();
        numerator_1_interactions
            .resize(1 << (last_layer.num_interaction_variables + 1), EF::zero());
        let mut denominator_0_interactions: Vec<EF> = denominator_0
            .into_iter()
            .flat_map(|mle| {
                let d00 = mle.fix_last_variable(EF::zero());
                let d01 = mle.fix_last_variable(EF::one());
                let d00_int = d00.eval_at::<EF>(&Point::from(vec![])).to_vec();
                let d01_int = d01.eval_at::<EF>(&Point::from(vec![])).to_vec();
                d00_int.iter().interleave(d01_int.iter()).copied().collect::<Vec<_>>()
            })
            .collect();
        denominator_0_interactions
            .resize(1 << (last_layer.num_interaction_variables + 1), EF::one());
        let mut denominator_1_interactions: Vec<EF> = denominator_1
            .into_iter()
            .flat_map(|mle| {
                let d10 = mle.fix_last_variable(EF::zero());
                let d11 = mle.fix_last_variable(EF::one());
                let d10_int = d10.eval_at::<EF>(&Point::from(vec![])).to_vec();
                let d11_int = d11.eval_at::<EF>(&Point::from(vec![])).to_vec();
                d10_int.iter().interleave(d11_int.iter()).copied().collect::<Vec<_>>()
            })
            .collect();
        denominator_1_interactions
            .resize(1 << (last_layer.num_interaction_variables + 1), EF::one());

        let (numerator, denominator): (Vec<_>, Vec<_>) = numerator_0_interactions
            .iter()
            .zip_eq(numerator_1_interactions.iter())
            .zip_eq(denominator_0_interactions.iter().zip_eq(denominator_1_interactions.iter()))
            .map(|((n_0, n_1), (d_0, d_1))| (*n_0 * *d_1 + *n_1 * *d_0, *d_0 * *d_1))
            .unzip();

        let numerator = Mle::from(numerator);
        let denominator = Mle::from(denominator);

        LogUpGkrOutput { numerator, denominator }
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::uninit_vec)]
    pub(crate) fn generate_first_layer(
        &self,
        interactions: &BTreeMap<String, Vec<(&Interaction<F>, bool)>>,
        main_traces: &Traces<F, CpuBackend>,
        preprocessed_traces: &Traces<F, CpuBackend>,
        alpha: EF,
        beta_seed: Point<EF>,
    ) -> LogUpGkrCpuLayer<F, EF> {
        let first_trace = main_traces.values().next().unwrap();
        let num_row_variables = first_trace.num_variables();

        let mut numerator_0 = Vec::new();
        let mut denominator_0 = Vec::new();
        let mut numerator_1 = Vec::new();
        let mut denominator_1 = Vec::new();
        let betas = Mle::partial_lagrange(&beta_seed).guts().as_slice().to_vec();
        let mut total_interactions = 0;
        for (name, interactions) in interactions.iter() {
            #[cfg(feature = "metal")]
            let chip_started = std::time::Instant::now();
            let main_trace = main_traces.get(name.as_str()).unwrap().clone();
            let height = main_trace.num_real_entries();

            let preprocessed_trace = preprocessed_traces.get(name.as_str()).cloned();
            let num_interactions = interactions.len();
            total_interactions += num_interactions;
            // Uninitialised, because both paths below write every cell: the
            // kernel fills the whole array and the CPU loop covers it with
            // `par_chunks_exact_mut`. Filling first is a wasted pass over the
            // largest thing this function touches, and the denominators are the
            // expensive half -- ones rather than zeros, so not a memset.
            let mut numer_evals = uninit_vec::<F>(height * num_interactions);
            let mut denom_evals = uninit_vec::<EF>(height * num_interactions);

            // println!("preprocessed_trace: {:?}", preprocessed_trace.num_variables());
            // Every cell here is one affine form over a row and a handful of
            // extension multiplies, with no dependence between cells, so the
            // whole chip is a single dispatch. The forms are fixed per chip and
            // go up as a table rather than as generated code.
            #[cfg(feature = "metal")]
            let on_gpu = height > 0
                && {
                    // The slice and the width come from the same place: a
                    // preprocessed table can be present but empty, and taking
                    // the width from the padded MLE while taking the values
                    // from its inner one would let the kernel index past them.
                    let prep_inner = preprocessed_trace.as_ref().and_then(|p| p.inner().as_ref());
                    crate::metal::first_layer_chip(
                        interactions,
                        main_trace.inner().as_ref().unwrap().guts().as_slice(),
                        main_trace.num_polynomials(),
                        prep_inner.map_or(&[][..], |i| i.guts().as_slice()),
                        prep_inner.map_or(0, |i| i.num_polynomials()),
                        height,
                        alpha,
                        &betas,
                        &mut numer_evals,
                        &mut denom_evals,
                    )
                };
            #[cfg(not(feature = "metal"))]
            let on_gpu = false;
            // Under `SP1_METAL_FIRST_LAYER=check` both paths run and the
            // results are compared, on every chip of a real proof.
            #[cfg(feature = "metal")]
            let expected = (on_gpu && crate::metal::first_layer_check())
                .then(|| (numer_evals.clone(), denom_evals.clone()));
            #[cfg(feature = "metal")]
            let run_cpu = !on_gpu || expected.is_some();
            #[cfg(not(feature = "metal"))]
            let run_cpu = !on_gpu;
            if height > 0 && run_cpu {
                match preprocessed_trace {
                    Some(prep) => {
                        numer_evals
                            .par_chunks_exact_mut(num_interactions)
                            .zip_eq(denom_evals.par_chunks_exact_mut(num_interactions))
                            .zip_eq(
                                prep.inner()
                                    .as_ref()
                                    .unwrap()
                                    .guts()
                                    .as_slice()
                                    .par_chunks(prep.num_polynomials())
                                    .zip(
                                        main_trace
                                            .inner()
                                            .as_ref()
                                            .unwrap()
                                            .guts()
                                            .as_slice()
                                            .par_chunks(main_trace.num_polynomials()),
                                    ),
                            )
                            .for_each(|((numer_evals, denom_evals), (prep_row, main_row))| {
                                interactions
                                    .iter()
                                    .zip(numer_evals.iter_mut())
                                    .zip(denom_evals.iter_mut())
                                    .for_each(
                                        |(((interaction, is_send), numer_eval), denom_eval)| {
                                            let (numer, denom) = generate_interaction_vals(
                                                interaction,
                                                prep_row,
                                                main_row,
                                                *is_send,
                                                alpha,
                                                &betas,
                                            );
                                            *numer_eval = numer;
                                            *denom_eval = denom;
                                        },
                                    );
                            });
                    }
                    None => {
                        numer_evals
                            .par_chunks_exact_mut(num_interactions)
                            .zip_eq(denom_evals.par_chunks_exact_mut(num_interactions))
                            .zip_eq(
                                main_trace
                                    .inner()
                                    .as_ref()
                                    .unwrap()
                                    .guts()
                                    .as_slice()
                                    .par_chunks(main_trace.num_polynomials()),
                            )
                            .for_each(|((numer_evals, denom_evals), main_row)| {
                                interactions
                                    .iter()
                                    .zip(numer_evals.iter_mut())
                                    .zip(denom_evals.iter_mut())
                                    .for_each(
                                        |(((interaction, is_send), numer_eval), denom_eval)| {
                                            let (numer, denom) = generate_interaction_vals(
                                                interaction,
                                                &[],
                                                main_row,
                                                *is_send,
                                                alpha,
                                                &betas,
                                            );
                                            *numer_eval = numer;
                                            *denom_eval = denom;
                                        },
                                    );
                            });
                    }
                }
            }

            #[cfg(feature = "metal")]
            if let Some((gpu_numer, gpu_denom)) = expected {
                assert!(
                    gpu_numer == numer_evals,
                    "first-layer numerators from the GPU differ from upstream's for chip {name}",
                );
                assert!(
                    gpu_denom == denom_evals,
                    "first-layer denominators from the GPU differ from upstream's for chip {name}",
                );
            }

            let numerator = RowMajorMatrix::new(numer_evals, num_interactions);
            let denominator = RowMajorMatrix::new(denom_evals, num_interactions);
            let numer_mle = Mle::from(numerator);
            let denom_mle = Mle::from(denominator);
            let numer_padded = PaddedMle::padded_with_zeros(Arc::new(numer_mle), num_row_variables);
            let num_polys = denom_mle.num_polynomials();
            let denom_padded = PaddedMle::padded(
                Arc::new(denom_mle),
                num_row_variables,
                Padding::Constant((EF::one(), num_polys, CpuBackend)),
            );
            let numer_0 = numer_padded.fix_last_variable(F::zero());
            let denom_0 = denom_padded.fix_last_variable(EF::zero());
            let numer_1 = numer_padded.fix_last_variable(F::one());
            let denom_1 = denom_padded.fix_last_variable(EF::one());
            numerator_0.push(numer_0);
            denominator_0.push(denom_0);
            numerator_1.push(numer_1);
            denominator_1.push(denom_1);
            #[cfg(feature = "metal")]
            crate::metal::record_gen(
                usize::from(!on_gpu),
                height * num_interactions,
                chip_started.elapsed().as_nanos() as u64,
            );
        }
        let num_interaction_variables = total_interactions.next_power_of_two().ilog2();

        LogUpGkrCpuLayer {
            numerator_0,
            denominator_0,
            numerator_1,
            denominator_1,
            num_interaction_variables: num_interaction_variables as usize,
            num_row_variables: (num_row_variables - 1) as usize,
        }
    }

    #[allow(clippy::too_many_lines)]
    #[allow(clippy::unused_self)]
    pub(crate) fn layer_transition<K>(
        &self,
        layer: &LogUpGkrCpuLayer<K, EF>,
    ) -> LogUpGkrCpuLayer<EF, EF>
    where
        K: Field + Into<EF> + Copy,
    {
        // let row_count = layer.numerator_0.first().unwrap().num_real_entries().div_ceil(2);
        let num_row_variables = layer.numerator_0.first().unwrap().num_variables();
        assert_eq!(num_row_variables, layer.num_row_variables as u32);
        let mut numerator_0 = Vec::new();
        let mut denominator_0 = Vec::new();
        let mut numerator_1 = Vec::new();
        let mut denominator_1 = Vec::new();

        // The whole layer's chips go to the device in one command buffer and
        // are waited for once, since the work is the same either way and what a
        // round trip costs is re-entering a queue three other proving threads
        // are using. Chips the kernels decline come back as `None` and take the
        // path below.
        #[cfg(feature = "metal")]
        let from_gpu = {
            let chips: Vec<_> = itertools::izip!(
                layer.numerator_0.iter(),
                layer.denominator_0.iter(),
                layer.numerator_1.iter(),
                layer.denominator_1.iter()
            )
            .filter_map(|(n0, d0, n1, d1)| {
                let n0_mle = n0.inner().as_ref()?;
                Some((
                    n0_mle.guts().as_slice(),
                    d0.inner().as_ref().unwrap().guts().as_slice(),
                    n1.inner().as_ref().unwrap().guts().as_slice(),
                    d1.inner().as_ref().unwrap().guts().as_slice(),
                    n0_mle.num_polynomials(),
                    n0_mle.guts().sizes()[0],
                    n0.num_real_entries().div_ceil(2),
                ))
            })
            .collect();
            crate::metal::transition_layer(&chips)
        };
        #[cfg(feature = "metal")]
        let mut from_gpu = from_gpu.into_iter();

        for (n0_padded, d0_padded, n1_padded, d1_padded) in itertools::izip!(
            layer.numerator_0.clone(),
            layer.denominator_0.clone(),
            layer.numerator_1.clone(),
            layer.denominator_1.clone()
        ) {
            #[cfg(feature = "metal")]
            let chip_started = std::time::Instant::now();
            let num_interactions = n0_padded.num_polynomials();
            let row_count = n0_padded.num_real_entries().div_ceil(2);
            let mut next_n0 = vec![EF::zero(); row_count * num_interactions];
            let mut next_d0 = vec![EF::one(); row_count * num_interactions];
            let mut next_n1 = vec![EF::zero(); row_count * num_interactions];
            let mut next_d1 = vec![EF::one(); row_count * num_interactions];
            #[cfg(feature = "metal")]
            let mut went_to_gpu = false;
            if let Some(n0_mle) = n0_padded.inner().as_ref() {
                let d0_mle = d0_padded.inner().as_ref().unwrap();
                let n1_mle = n1_padded.inner().as_ref().unwrap();
                let d1_mle = d1_padded.inner().as_ref().unwrap();
                // One output cell per thread and no reduction, so this is the
                // easiest thing in the phase to move; the whole layer went in
                // one command buffer above, and this takes its chip's share.
                #[cfg(feature = "metal")]
                let on_gpu = from_gpu
                    .next()
                    .flatten()
                    .map(|(a, b, c, d)| {
                        next_n0 = a;
                        next_d0 = b;
                        next_n1 = c;
                        next_d1 = d;
                    })
                    .is_some();
                #[cfg(feature = "metal")]
                let _ = (&d0_mle, &n1_mle, &d1_mle);
                #[cfg(not(feature = "metal"))]
                let on_gpu = false;
                #[cfg(feature = "metal")]
                {
                    went_to_gpu = on_gpu;
                }
                if !on_gpu {
                n0_mle
                    .guts()
                    .as_slice()
                    .par_chunks(2 * num_interactions)
                    .zip_eq(d0_mle.guts().as_slice().par_chunks(2 * num_interactions))
                    .zip_eq(n1_mle.guts().as_slice().par_chunks(2 * num_interactions))
                    .zip_eq(d1_mle.guts().as_slice().par_chunks(2 * num_interactions))
                    .zip_eq(next_n0.par_chunks_exact_mut(num_interactions))
                    .zip_eq(next_d0.par_chunks_exact_mut(num_interactions))
                    .zip_eq(next_n1.par_chunks_exact_mut(num_interactions))
                    .zip_eq(next_d1.par_chunks_exact_mut(num_interactions))
                    .for_each(
                        |(
                            (
                                (
                                    ((((n0_chunk, d0_chunk), n1_chunk), d1_chunk), next_n0_row),
                                    next_d0_row,
                                ),
                                next_n1_row,
                            ),
                            next_d1_row,
                        )| {
                            let (n_00_row, n_10_row) = n0_chunk.split_at(num_interactions);
                            let (d_00_row, d_10_row) = d0_chunk.split_at(num_interactions);
                            let (n_01_row, n_11_row) = n1_chunk.split_at(num_interactions);
                            let (d_01_row, d_11_row) = d1_chunk.split_at(num_interactions);

                            n_00_row
                                .par_iter()
                                .zip_eq(d_00_row.par_iter())
                                .zip_eq(n_01_row.par_iter())
                                .zip_eq(d_01_row.par_iter())
                                .zip_eq(next_n0_row.par_iter_mut())
                                .zip_eq(next_d0_row.par_iter_mut())
                                .for_each(|(((((n_00, d_00), n_01), d_01), next_n0), next_d0)| {
                                    let n00: EF = (*n_00).into();
                                    let n01: EF = (*n_01).into();
                                    let n0 = *d_01 * n00 + *d_00 * n01;
                                    let d0 = *d_00 * *d_01;
                                    *next_n0 = n0;
                                    *next_d0 = d0;
                                });
                            if n0_chunk.len() == 2 * num_interactions {
                                n_10_row
                                    .par_iter()
                                    .zip_eq(d_10_row.par_iter())
                                    .zip_eq(n_11_row.par_iter())
                                    .zip_eq(d_11_row.par_iter())
                                    .zip_eq(next_n1_row.par_iter_mut())
                                    .zip_eq(next_d1_row.par_iter_mut())
                                    .for_each(
                                        |(((((n_10, d_10), n_11), d_11), next_n1), next_d1)| {
                                            let n10: EF = (*n_10).into();
                                            let n11: EF = (*n_11).into();
                                            let n1 = *d_11 * n10 + *d_10 * n11;
                                            let d1 = *d_10 * *d_11;
                                            *next_n1 = n1;
                                            *next_d1 = d1;
                                        },
                                    );
                            }
                        },
                    );
                }
            }
            let next_n0_padded = PaddedMle::padded_with_zeros(
                Arc::new(Mle::from(RowMajorMatrix::new(next_n0, num_interactions))),
                num_row_variables - 1,
            );
            let next_d0_padded = PaddedMle::padded(
                Arc::new(Mle::from(RowMajorMatrix::new(next_d0, num_interactions))),
                num_row_variables - 1,
                Padding::Constant((EF::one(), num_interactions, CpuBackend)),
            );
            let next_n1_padded = PaddedMle::padded_with_zeros(
                Arc::new(Mle::from(RowMajorMatrix::new(next_n1, num_interactions))),
                num_row_variables - 1,
            );
            let next_d1_padded = PaddedMle::padded(
                Arc::new(Mle::from(RowMajorMatrix::new(next_d1, num_interactions))),
                num_row_variables - 1,
                Padding::Constant((EF::one(), num_interactions, CpuBackend)),
            );
            numerator_0.push(next_n0_padded);
            denominator_0.push(next_d0_padded);
            numerator_1.push(next_n1_padded);
            denominator_1.push(next_d1_padded);
            #[cfg(feature = "metal")]
            crate::metal::record_gen(
                2 + usize::from(!went_to_gpu),
                row_count * num_interactions,
                chip_started.elapsed().as_nanos() as u64,
            );
        }
        LogUpGkrCpuLayer {
            numerator_0,
            denominator_0,
            numerator_1,
            denominator_1,
            num_interaction_variables: layer.num_interaction_variables,
            num_row_variables: layer.num_row_variables - 1,
        }
    }
}
