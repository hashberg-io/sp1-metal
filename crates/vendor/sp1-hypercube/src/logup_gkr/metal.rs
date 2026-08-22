//! A Metal implementation of the logup GKR round.
//!
//! Lives inside the fork rather than beside it so that `DefaultRoundOps` can
//! point at it under the `metal` feature. That is what lets an existing prover
//! -- sp1-sdk's included -- run its GKR rounds on the GPU without a single type
//! changing anywhere downstream.
//!
//! The fork in `crates/vendor/sp1-hypercube` names one boundary,
//! `LogupRoundOps`, at the place where a GKR round's time goes: three sums over
//! the layer, from which the round message is interpolated. Everything else
//! about a round is arithmetic on three field elements and stays where it is.
//! This crate supplies that reduction on the GPU.
//!
//! Two things make the integration cheap. `KoalaBear` is `repr(transparent)`
//! over its Montgomery word and `BinomialExtensionField` is `repr(C)` over an
//! array of them, so a slice of field elements is already a slice of the words
//! a kernel reads — there is no conversion. And unified memory means the only
//! question is whether the pages can be handed to Metal directly; see
//! [`upload`].
//!
//! The interaction layers — the rounds after the row variables are exhausted,
//! where the whole layer is one row — are left to the CPU. They are small by
//! construction, and the point of moving a phase to the GPU is the part that is
//! not.

use std::sync::{Arc, Mutex, OnceLock};

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use slop_algebra::{AbstractExtensionField, AbstractField};
use slop_koala_bear::KoalaBear;
use slop_alloc::CpuBackend;
use slop_multilinear::{Mle, Padding, PaddedMle};
use super::{cpu_round_evals, LogupRoundOps, LogupRoundPolynomial, PolynomialLayer};

/// The base field the kernels are written for.
pub type BaseField = KoalaBear;
/// Kept for the harness crate, which names it.
pub type F = KoalaBear;
pub type EF = slop_algebra::extension::BinomialExtensionField<KoalaBear, 4>;

const THREADS: u64 = 256;

/// Shader source: the probe suite's field arithmetic verbatim, so the kernels
/// here compute in the representation everything else was parity-checked in,
/// plus the round kernels.
fn shader_source() -> String {
    let probes = include_str!("../../../../sp1-metal-probes/src/shaders.metal");
    let header = probes
        .split("// ---- end of shared field arithmetic ----")
        .next()
        .expect("shader header sentinel");
    let one = (((1u64) << 32) % 0x7f00_0001u64) as u32;
    format!("{header}\nconstant uint KB_ONE = {one}u;\n{}", include_str!("shaders.metal"))
}

struct Context {
    device: Device,
    queue: CommandQueue,
    ext: ComputePipelineState,
    base: ComputePipelineState,
    fold_ext: ComputePipelineState,
    fold_base: ComputePipelineState,
    /// Scratch buffers for per-threadgroup partials, checked out and returned.
    ///
    /// One shared buffer would be a data race: SP1 proves shards concurrently,
    /// so several rounds can be in this reduction at once, and each needs its
    /// own place to put its partial sums.
    pool: Mutex<Vec<Buffer>>,
}

impl Context {
    fn new() -> Self {
        let device = Device::system_default().expect("no Metal device found");
        let library = device
            .new_library_with_source(&shader_source(), &metal::CompileOptions::new())
            .expect("shader compilation failed");
        let pso = |name: &str| {
            let f = library.get_function(name, None).expect("kernel not found");
            device.new_compute_pipeline_state_with_function(&f).expect("pipeline")
        };
        let ext = pso("logup_round_ext");
        let base = pso("logup_round_base");
        let fold_ext = pso("fold_padded_ext");
        let fold_base = pso("fold_padded_base");
        let queue = device.new_command_queue();
        Self { device, queue, ext, base, fold_ext, fold_base, pool: Mutex::new(Vec::new()) }
    }

    /// Takes a scratch buffer of at least the size needed, out of the pool if
    /// one fits.
    fn take_partials(&self, groups: usize) -> Buffer {
        let want = (groups * 3 * 16) as u64;
        let mut pool = self.pool.lock().expect("pool poisoned");
        if let Some(i) = pool.iter().position(|b| b.length() >= want) {
            return pool.swap_remove(i);
        }
        drop(pool);
        self.device.new_buffer(want.max(4096), MTLResourceOptions::StorageModeShared)
    }

    fn return_partials(&self, buffer: Buffer) {
        let mut pool = self.pool.lock().expect("pool poisoned");
        if pool.len() < 32 {
            pool.push(buffer);
        }
    }
}

fn context() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(Context::new)
}

/// Hands a slice of field elements to Metal.
///
/// Unified memory means the pages are already where the GPU can read them, and
/// `newBufferWithBytesNoCopy` will take them as they are — but only if the
/// pointer and length are page-aligned, which the global allocator does not
/// promise. When they are, this is free; when they are not, it costs a copy,
/// and [`no_copy_fraction`] reports how often each happened.
fn upload(bytes: &[u8]) -> Buffer {
    let device = &context().device;
    let page = 16 * 1024usize;
    let aligned = bytes.as_ptr() as usize % page == 0 && bytes.len() % page == 0;
    STATS.with_borrow(aligned);
    if aligned && !bytes.is_empty() {
        // SAFETY: the buffer borrows `bytes` for the lifetime of the dispatch,
        // which is awaited before this function's caller returns.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
    } else {
        device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len().max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

/// How many uploads avoided a copy, for reporting.
pub struct UploadStats {
    inner: Mutex<(usize, usize)>,
}

impl UploadStats {
    fn with_borrow(&self, no_copy: bool) {
        let mut s = self.inner.lock().expect("stats poisoned");
        s.0 += 1;
        s.1 += usize::from(no_copy);
    }

    /// `(uploads, of which no-copy)`.
    #[must_use]
    pub fn counts(&self) -> (usize, usize) {
        *self.inner.lock().expect("stats poisoned")
    }

    /// Resets the counters.
    pub fn reset(&self) {
        *self.inner.lock().expect("stats poisoned") = (0, 0);
    }
}

/// Upload accounting, so a measurement can say how much of it was copying.
pub static STATS: UploadStats = UploadStats { inner: Mutex::new((0, 0)) };

fn as_words<T>(xs: &[T]) -> &[u8] {
    // SAFETY: KoalaBear is repr(transparent) over u32 and BinomialExtensionField
    // is repr(C) over [KoalaBear; 4], so either is a plain array of Montgomery
    // words with no padding.
    unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), std::mem::size_of_val(xs)) }
}

/// Whether a padded MLE contributes to this round at all.
fn chip_inner<T: slop_algebra::Field>(mle: &PaddedMle<T>) -> Option<Arc<Mle<T>>> {
    mle.inner().as_ref().cloned()
}

/// The reduction for one circuit layer, one dispatch per chip.
#[allow(clippy::too_many_arguments)]
fn circuit_layer<N: slop_algebra::Field>(
    pso: &ComputePipelineState,
    numerators: (&[PaddedMle<N>], &[PaddedMle<N>]),
    denominators: (&[PaddedMle<EF>], &[PaddedMle<EF>]),
    eq_interaction: &[EF],
    eq_row: &[EF],
    lambda: EF,
    numerator_words: usize,
) -> (EF, EF, EF) {
    let ctx = context();
    let lambda_words: [u32; 4] = unsafe { std::mem::transmute_copy(&lambda) };
    let mut interaction_offset = 0usize;
    let mut totals = (EF::zero(), EF::zero(), EF::zero());
    let cb = ctx.queue.new_command_buffer();
    let mut held: Vec<(Buffer, usize, [Buffer; 6])> = Vec::new();
    for (((n0, n1), d0), d1) in
        numerators.0.iter().zip(numerators.1).zip(denominators.0).zip(denominators.1)
    {
        let Some(n0_inner) = chip_inner(n0) else { continue };
        let width = n0.num_polynomials();
        let n1_inner = chip_inner(n1).expect("paired numerator");
        let d0_inner = chip_inner(d0).expect("paired denominator");
        let d1_inner = chip_inner(d1).expect("paired denominator");
        let len = n0_inner.guts().as_slice().len();
        if width == 0 || len == 0 {
            continue;
        }
        let rows = len / width;
        let n_pairs = rows.div_ceil(2);
        if n_pairs == 0 {
            continue;
        }
        // The CPU zips eq_row with the row pairs and stops at the shorter of
        // the two, so a kernel must not read past what eq_row has.
        let n_pairs = n_pairs.min(eq_row.len() / 2);
        if n_pairs == 0 {
            continue;
        }
        let n_full = (rows / 2).min(n_pairs);
        let groups = n_pairs.div_ceil(THREADS as usize);
        let partials = ctx.take_partials(groups);

        let b_n0 = upload(as_words(n0_inner.guts().as_slice()));
        let b_n1 = upload(as_words(n1_inner.guts().as_slice()));
        let b_d0 = upload(as_words(d0_inner.guts().as_slice()));
        let b_d1 = upload(as_words(d1_inner.guts().as_slice()));
        let b_eqi = upload(as_words(&eq_interaction[interaction_offset..interaction_offset + width]));
        let b_eqr = upload(as_words(&eq_row[..2 * n_pairs]));

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {
            n_pairs: u32,
            n_full_pairs: u32,
            width: u32,
        }
        let params =
            Params { n_pairs: n_pairs as u32, n_full_pairs: n_full as u32, width: width as u32 };

        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in [&b_n0, &b_n1, &b_d0, &b_d1, &b_eqi, &b_eqr].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_buffer(6, Some(&partials), 0);
        enc.set_bytes(7, std::mem::size_of::<Params>() as u64, (&params as *const Params).cast());
        enc.set_bytes(8, 16, lambda_words.as_ptr().cast());
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
        enc.end_encoding();
        // Keep the buffers alive until the whole layer has run.
        held.push((partials, groups, [b_n0, b_n1, b_d0, b_d1, b_eqi, b_eqr]));
        interaction_offset += width;
        let _ = numerator_words;
    }
    if held.is_empty() {
        return totals;
    }
    cb.commit();
    cb.wait_until_completed();
    for (partials, groups, _) in held {
        let raw = partials.contents().cast::<EF>();
        for g in 0..groups {
            // SAFETY: the kernel wrote 3 extension elements per threadgroup.
            let (a, b, c) = unsafe { (*raw.add(3 * g), *raw.add(3 * g + 1), *raw.add(3 * g + 2)) };
            totals.0 += a;
            totals.1 += b;
            totals.2 += c;
        }
        ctx.return_partials(partials);
    }
    totals
}

/// The logup GKR round reduction, on the GPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalRoundOps;

/// The kernels are written for KoalaBear and its degree-4 extension, but the
/// prover is generic over the field, so this implementation has to be too. It
/// checks at runtime which field it has been handed and defers to the CPU for
/// anything else — which costs one type comparison per round against a phase
/// that takes milliseconds.
impl<F, K> LogupRoundOps<F, K> for MetalRoundOps
where
    F: slop_algebra::Field,
    K: slop_algebra::ExtensionField<F>,
{
    fn round_evals(poly: &LogupRoundPolynomial<F, K, Self>) -> (K, K, K) {
        if is::<K, EF>() {
            if is::<F, BaseField>() {
                // SAFETY: the type ids match, so these are the same types.
                let p: &LogupRoundPolynomial<BaseField, EF, Self> =
                    unsafe { &*std::ptr::from_ref(poly).cast() };
                return unsafe { std::mem::transmute_copy(&round_evals_base(p)) };
            }
            if is::<F, EF>() {
                // SAFETY: as above.
                let p: &LogupRoundPolynomial<EF, EF, Self> =
                    unsafe { &*std::ptr::from_ref(poly).cast() };
                return unsafe { std::mem::transmute_copy(&round_evals_ext(p)) };
            }
        }
        cpu_round_evals(poly)
    }

    fn fix_numerator(mle: &PaddedMle<F>, alpha: K) -> PaddedMle<K> {
        if is::<K, EF>() {
            if is::<F, BaseField>() {
                // SAFETY: the type ids match, so these are the same types.
                let m: &PaddedMle<BaseField> = unsafe { &*std::ptr::from_ref(mle).cast() };
                let a: EF = unsafe { std::mem::transmute_copy(&alpha) };
                let out = fix_base(m, a);
                return unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) };
            }
            if is::<F, EF>() {
                // SAFETY: as above.
                let m: &PaddedMle<EF> = unsafe { &*std::ptr::from_ref(mle).cast() };
                let a: EF = unsafe { std::mem::transmute_copy(&alpha) };
                let out = fix_ext(m, a);
                return unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) };
            }
        }
        mle.fix_last_variable(alpha)
    }

    fn fix_layer(
        numerators: (&[PaddedMle<F>], &[PaddedMle<F>]),
        denominators: (&[PaddedMle<K>], &[PaddedMle<K>]),
        alpha: K,
    ) -> (Vec<PaddedMle<K>>, Vec<PaddedMle<K>>, Vec<PaddedMle<K>>, Vec<PaddedMle<K>>) {
        let handled = is::<K, EF>() && (is::<F, BaseField>() || is::<F, EF>());
        if !handled {
            return (
                numerators.0.iter().map(|m| Self::fix_numerator(m, alpha)).collect(),
                denominators.0.iter().map(|m| Self::fix_denominator(m, alpha)).collect(),
                numerators.1.iter().map(|m| Self::fix_numerator(m, alpha)).collect(),
                denominators.1.iter().map(|m| Self::fix_denominator(m, alpha)).collect(),
            );
        }
        // SAFETY: the type ids match, so these are the same types.
        let a: EF = unsafe { std::mem::transmute_copy(&alpha) };
        let ctx = context();
        let cb = ctx.queue.new_command_buffer();
        let num_pso = if is::<F, BaseField>() { &ctx.fold_base } else { &ctx.fold_ext };
        let stage_num = |ms: &[PaddedMle<F>]| -> Vec<Fold> {
            ms.iter()
                .map(|m| {
                    if is::<F, BaseField>() {
                        let m: &PaddedMle<BaseField> = unsafe { &*std::ptr::from_ref(m).cast() };
                        stage_fold(cb, num_pso, m, a)
                    } else {
                        let m: &PaddedMle<EF> = unsafe { &*std::ptr::from_ref(m).cast() };
                        stage_fold(cb, num_pso, m, a)
                    }
                })
                .collect()
        };
        let stage_den = |ms: &[PaddedMle<K>]| -> Vec<Fold> {
            ms.iter()
                .map(|m| {
                    let m: &PaddedMle<EF> = unsafe { &*std::ptr::from_ref(m).cast() };
                    stage_fold(cb, &ctx.fold_ext, m, a)
                })
                .collect()
        };
        let staged = (
            stage_num(numerators.0),
            stage_den(denominators.0),
            stage_num(numerators.1),
            stage_den(denominators.1),
        );
        cb.commit();
        cb.wait_until_completed();
        let out = (
            finish_all(staged.0),
            finish_all(staged.1),
            finish_all(staged.2),
            finish_all(staged.3),
        );
        // SAFETY: K is EF, checked above.
        unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) }
    }

    fn fix_denominator(mle: &PaddedMle<K>, alpha: K) -> PaddedMle<K> {
        if is::<K, EF>() {
            // SAFETY: the type ids match, so these are the same types.
            let m: &PaddedMle<EF> = unsafe { &*std::ptr::from_ref(mle).cast() };
            let a: EF = unsafe { std::mem::transmute_copy(&alpha) };
            let out = fix_ext(m, a);
            return unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) };
        }
        mle.fix_last_variable(alpha)
    }
}

fn is<A: 'static + ?Sized, B: 'static + ?Sized>() -> bool {
    std::any::TypeId::of::<A>() == std::any::TypeId::of::<B>()
}

fn round_evals_base(poly: &LogupRoundPolynomial<BaseField, EF, MetalRoundOps>) -> (EF, EF, EF) {
    match &poly.layer {
        PolynomialLayer::InteractionLayer(_) => cpu_round_evals(poly),
        PolynomialLayer::CircuitLayer(layer) => circuit_layer::<BaseField>(
            &context().base,
            (&layer.numerator_0, &layer.numerator_1),
            (&layer.denominator_0, &layer.denominator_1),
            poly.eq_interaction.guts().as_slice(),
            poly.eq_row.guts().as_slice(),
            poly.lambda,
            1,
        ),
    }
}

fn round_evals_ext(poly: &LogupRoundPolynomial<EF, EF, MetalRoundOps>) -> (EF, EF, EF) {
    match &poly.layer {
        PolynomialLayer::InteractionLayer(_) => cpu_round_evals(poly),
        PolynomialLayer::CircuitLayer(layer) => circuit_layer::<EF>(
            &context().ext,
            (&layer.numerator_0, &layer.numerator_1),
            (&layer.denominator_0, &layer.denominator_1),
            poly.eq_interaction.guts().as_slice(),
            poly.eq_row.guts().as_slice(),
            poly.lambda,
            4,
        ),
    }
}

fn fix_base(mle: &PaddedMle<BaseField>, alpha: EF) -> PaddedMle<EF> {
    let (padding, words) = lift_padding(mle);
    fix_padded(&context().fold_base, mle, alpha, padding, as_words(&words))
}

fn fix_ext(mle: &PaddedMle<EF>, alpha: EF) -> PaddedMle<EF> {
    let (padding, words) = lift_padding(mle);
    fix_padded(&context().fold_ext, mle, alpha, padding, as_words(&words))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FoldParams {
    out_rows: u32,
    rows: u32,
    width: u32,
}

/// Fixes a padded MLE's last variable on the GPU.
///
/// The result is an ordinary CPU-side `Mle`: unified memory means the kernel
/// can write straight into the `Vec` that will back it, when that `Vec`'s pages
/// are aligned for Metal to take them, which for the sizes that matter they
/// are. The padding is passed as a buffer so the odd-row case needs no special
/// dispatch.
fn fix_padded<T: slop_algebra::Field>(
    pso: &ComputePipelineState,
    mle: &PaddedMle<T>,
    alpha: EF,
    new_padding: Padding<EF, CpuBackend>,
    padding_words: &[u8],
) -> PaddedMle<EF>
where
    EF: slop_algebra::ExtensionField<T>,
{
    let out_vars = mle.num_variables() - 1;
    let Some(inner) = chip_inner(mle) else {
        return PaddedMle::dummy(out_vars, new_padding);
    };
    let ctx = context();
    let width = inner.num_polynomials();
    let rows = inner.guts().sizes()[0];
    let out_rows = rows.div_ceil(2);
    let total = out_rows * width;
    if total == 0 {
        return mle.fix_last_variable(alpha);
    }

    let mut values: Vec<EF> = Vec::with_capacity(total);
    // SAFETY: the kernel writes every element before anything reads them.
    unsafe { values.set_len(total) };
    let out_bytes = total * std::mem::size_of::<EF>();
    let page = 16 * 1024usize;
    let direct = values.as_ptr() as usize % page == 0 && out_bytes % page == 0;
    STATS.with_borrow(direct);
    let out_buf = if direct {
        // SAFETY: borrowed only for the dispatch, which is awaited below.
        unsafe {
            ctx.device.new_buffer_with_bytes_no_copy(
                values.as_ptr().cast(),
                out_bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
    } else {
        ctx.device.new_buffer(out_bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
    };

    let in_buf = upload(as_words(inner.guts().as_slice()));
    let pad_buf = upload(padding_words);
    let alpha_words: [u32; 4] = unsafe { std::mem::transmute_copy(&alpha) };

    let params =
        FoldParams { out_rows: out_rows as u32, rows: rows as u32, width: width as u32 };

    let cb = ctx.queue.new_command_buffer();
    encode_fold(cb, pso, &in_buf, &pad_buf, &out_buf, params, alpha_words, total);
    cb.commit();
    cb.wait_until_completed();

    if !direct {
        // SAFETY: the kernel wrote `out_bytes` bytes of extension elements.
        unsafe {
            std::ptr::copy_nonoverlapping(
                out_buf.contents().cast::<u8>(),
                values.as_mut_ptr().cast::<u8>(),
                out_bytes,
            );
        }
    }
    let guts = slop_tensor::Tensor::from(values).reshape([out_rows, width]);
    PaddedMle::padded(Arc::new(Mle::new(guts)), out_vars, new_padding)
}

/// The padding a fold produces, and its words for the kernel.
fn lift_padding<T: slop_algebra::Field>(mle: &PaddedMle<T>) -> (Padding<EF, CpuBackend>, Vec<T>)
where
    EF: slop_algebra::ExtensionField<T>,
{
    let padding = mle.clone().into_padding_values();
    let values: Vec<T> = padding.into();
    let lifted: Vec<EF> = values.iter().copied().map(<EF as AbstractExtensionField<T>>::from_base).collect();
    (Padding::Generic(Arc::new(lifted.into())), values)
}


/// Encodes one fold into an existing command buffer.
#[allow(clippy::too_many_arguments)]
fn encode_fold(
    cb: &metal::CommandBufferRef,
    pso: &ComputePipelineState,
    in_buf: &Buffer,
    pad_buf: &Buffer,
    out_buf: &Buffer,
    params: FoldParams,
    alpha_words: [u32; 4],
    total: usize,
) {
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(in_buf), 0);
    enc.set_buffer(1, Some(pad_buf), 0);
    enc.set_buffer(2, Some(out_buf), 0);
    enc.set_bytes(3, std::mem::size_of::<FoldParams>() as u64, (&params as *const FoldParams).cast());
    enc.set_bytes(4, 16, alpha_words.as_ptr().cast());
    let threads = total as u64;
    enc.dispatch_threads(
        MTLSize::new(threads, 1, 1),
        MTLSize::new(THREADS.min(threads.max(1)), 1, 1),
    );
    enc.end_encoding();
}

/// A fold whose dispatch has been encoded but not yet awaited.
struct PendingFold {
    values: Vec<EF>,
    out_buf: Buffer,
    direct: bool,
    out_rows: usize,
    width: usize,
    out_vars: u32,
    padding: Padding<EF, CpuBackend>,
    held: Vec<Buffer>,
}

impl PendingFold {
    fn finish(mut self) -> PaddedMle<EF> {
        if !self.direct {
            let bytes = self.out_rows * self.width * std::mem::size_of::<EF>();
            // SAFETY: the kernel wrote that many bytes of extension elements.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.out_buf.contents().cast::<u8>(),
                    self.values.as_mut_ptr().cast::<u8>(),
                    bytes,
                );
            }
        }
        drop(self.held);
        let guts = slop_tensor::Tensor::from(self.values).reshape([self.out_rows, self.width]);
        PaddedMle::padded(Arc::new(Mle::new(guts)), self.out_vars, self.padding)
    }
}

/// Either a fold that needed no GPU work, or one encoded into the batch.
enum Fold {
    Done(PaddedMle<EF>),
    Pending(PendingFold),
}

/// Encodes one padded MLE's fold into `cb`, or completes it outright when there
/// is nothing to compute.
fn stage_fold<T: slop_algebra::Field>(
    cb: &metal::CommandBufferRef,
    pso: &ComputePipelineState,
    mle: &PaddedMle<T>,
    alpha: EF,
) -> Fold
where
    EF: slop_algebra::ExtensionField<T>,
{
    let (padding, pad_values) = lift_padding(mle);
    let out_vars = mle.num_variables() - 1;
    let Some(inner) = chip_inner(mle) else {
        return Fold::Done(PaddedMle::dummy(out_vars, padding));
    };
    let ctx = context();
    let width = inner.num_polynomials();
    let rows = inner.guts().sizes()[0];
    let out_rows = rows.div_ceil(2);
    let total = out_rows * width;
    if total == 0 {
        return Fold::Done(mle.fix_last_variable(alpha));
    }
    let mut values: Vec<EF> = Vec::with_capacity(total);
    // SAFETY: the kernel writes every element before anything reads them.
    unsafe { values.set_len(total) };
    let out_bytes = total * std::mem::size_of::<EF>();
    let page = 16 * 1024usize;
    let direct = values.as_ptr() as usize % page == 0 && out_bytes % page == 0;
    STATS.with_borrow(direct);
    let out_buf = if direct {
        // SAFETY: borrowed until the batch is awaited, which happens before the
        // values are read.
        unsafe {
            ctx.device.new_buffer_with_bytes_no_copy(
                values.as_ptr().cast(),
                out_bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
    } else {
        ctx.device.new_buffer(out_bytes as u64, MTLResourceOptions::StorageModeShared)
    };
    let in_buf = upload(as_words(inner.guts().as_slice()));
    let pad_buf = upload(as_words(&pad_values));
    let alpha_words: [u32; 4] = unsafe { std::mem::transmute_copy(&alpha) };
    let params =
        FoldParams { out_rows: out_rows as u32, rows: rows as u32, width: width as u32 };
    encode_fold(cb, pso, &in_buf, &pad_buf, &out_buf, params, alpha_words, total);
    Fold::Pending(PendingFold {
        values,
        out_buf,
        direct,
        out_rows,
        width,
        out_vars,
        padding,
        held: vec![in_buf, pad_buf],
    })
}

fn finish_all(folds: Vec<Fold>) -> Vec<PaddedMle<EF>> {
    folds
        .into_iter()
        .map(|f| match f {
            Fold::Done(m) => m,
            Fold::Pending(p) => p.finish(),
        })
        .collect()
}
