//! A Metal implementation of the logup GKR round reduction.
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
use slop_algebra::AbstractField;
use slop_koala_bear::KoalaBear;
use slop_multilinear::{Mle, PaddedMle};
use sp1_hypercube::{cpu_round_evals, LogupRoundOps, LogupRoundPolynomial, PolynomialLayer};

pub type F = KoalaBear;
pub type EF = slop_algebra::extension::BinomialExtensionField<KoalaBear, 4>;

const THREADS: u64 = 256;

/// Shader source: the probe suite's field arithmetic verbatim, so the kernels
/// here compute in the representation everything else was parity-checked in,
/// plus the round kernels.
fn shader_source() -> String {
    let probes = include_str!("../../sp1-metal-probes/src/shaders.metal");
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
        let queue = device.new_command_queue();
        Self { device, queue, ext, base, pool: Mutex::new(Vec::new()) }
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
    for (((n0, n1), d0), d1) in
        numerators.0.iter().zip(numerators.1).zip(denominators.0).zip(denominators.1)
    {
        let Some(n0_inner) = chip_inner(n0) else { continue };
        let width = n0.num_polynomials();
        let n1_inner = chip_inner(n1).expect("paired numerator");
        let d0_inner = chip_inner(d0).expect("paired denominator");
        let d1_inner = chip_inner(d1).expect("paired denominator");
        let len = n0_inner.guts().as_slice().len();
        let rows = len / width;
        let n_pairs = rows.div_ceil(2);
        let n_full = rows / 2;
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

        let cb = ctx.queue.new_command_buffer();
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
        cb.commit();
        cb.wait_until_completed();

        let raw = partials.contents().cast::<EF>();
        for g in 0..groups {
            // SAFETY: the kernel wrote 3 extension elements per threadgroup.
            let (a, b, c) = unsafe { (*raw.add(3 * g), *raw.add(3 * g + 1), *raw.add(3 * g + 2)) };
            totals.0 += a;
            totals.1 += b;
            totals.2 += c;
        }
        ctx.return_partials(partials);
        interaction_offset += width;
        let _ = numerator_words;
    }
    totals
}

/// The logup GKR round reduction, on the GPU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalRoundOps;

impl LogupRoundOps<EF, EF> for MetalRoundOps {
    fn round_evals(poly: &LogupRoundPolynomial<EF, EF, Self>) -> (EF, EF, EF) {
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
}

impl LogupRoundOps<F, EF> for MetalRoundOps {
    fn round_evals(poly: &LogupRoundPolynomial<F, EF, Self>) -> (EF, EF, EF) {
        match &poly.layer {
            PolynomialLayer::InteractionLayer(_) => cpu_round_evals(poly),
            PolynomialLayer::CircuitLayer(layer) => circuit_layer::<F>(
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
}
