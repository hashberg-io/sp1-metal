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

use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock};

use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use slop_algebra::{AbstractExtensionField, AbstractField};
use slop_koala_bear::KoalaBear;
use slop_alloc::CpuBackend;
use slop_multilinear::{Mle, Padding, PaddedMle};
use sp1_metal_probes::stats::{self, Input, Output};
use super::{cpu_round_evals, LogupRoundOps, LogupRoundPolynomial, PolynomialLayer};

/// The base field the kernels are written for.
pub type BaseField = KoalaBear;
/// Kept for the harness crate, which names it.
pub type F = KoalaBear;
pub type EF = slop_algebra::extension::BinomialExtensionField<KoalaBear, 4>;
/// The same type, named for the places that also carry a generic `EF`.
type EF2 = EF;

const THREADS: u64 = 256;

/// The most threadgroups a round dispatch will use. Beyond this the kernel
/// strides, which keeps the partials the host adds up from growing with the
/// chip: an uncapped grid over cells would hand back hundreds of thousands of
/// them and put the work back on the CPU.
const MAX_GROUPS: usize = 2048;

/// Entries in a layer below which the round stays on the CPU.
///
/// A GKR layer has one polynomial per chip, and a shard proof walks a lot of
/// small ones: preparing a dispatch costs Metal buffer creation, which is a
/// call into the kernel, and below this size that costs more than the
/// arithmetic saves. Measured on a shard-sized layer the GPU is nine times the
/// CPU; on a layer of a few hundred entries it is far slower, and the crossover
/// is around here.
const DEFAULT_MIN_ENTRIES_FOR_GPU: usize = 1 << 14;

/// The threshold in force, overridable with `SP1_METAL_MIN_ENTRIES` — which
/// tests set to zero so that small shapes still reach the kernels.
fn min_entries_for_gpu() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("SP1_METAL_MIN_ENTRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MIN_ENTRIES_FOR_GPU)
    })
}

/// Total entries across a layer's numerators, used to decide whether the round
/// is worth dispatching.
fn layer_entries<T: slop_algebra::Field>(mles: &[PaddedMle<T>]) -> usize {
    mles.iter()
        .filter_map(|m| m.inner().as_ref().map(|i| i.guts().as_slice().len()))
        .sum()
}

/// Whether a layer is worth taking to the GPU at all.
///
/// This used to ask whether the chips big enough for their own dispatch
/// accounted for most of the layer, because everything below the threshold was
/// summed on the CPU and standing in for most of a layer that way was worse
/// than leaving the whole layer to upstream. With the small chips batched into
/// one dispatch of their own that no longer applies: every chip is covered
/// either way, so the only question is whether the layer as a whole is big
/// enough to be worth the round trip.
fn worth_dispatching<T: slop_algebra::Field>(mles: &[PaddedMle<T>]) -> bool {
    let min = min_entries_for_gpu();
    let (mut big, mut all) = (0usize, 0usize);
    for m in mles {
        let len = m.inner().as_ref().map_or(0, |i| i.guts().as_slice().len());
        all += len;
        if len >= min {
            big += len;
        }
    }
    if all == 0 {
        return false;
    }
    if batch_on_gpu() {
        return all >= min_layer_entries();
    }
    big * 2 >= all
}

/// Entries in a whole layer below which even a batched dispatch is not worth
/// it, overridable with `SP1_METAL_MIN_LAYER`.
///
/// Much lower than the per-chip threshold, and for a different reason: a chip
/// below that threshold would need a dispatch to itself, whereas a layer below
/// this one needs a single dispatch shared by all of it.
fn min_layer_entries() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("SP1_METAL_MIN_LAYER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1 << 6)
    })
}

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
    ext: ComputePipelineState,
    base: ComputePipelineState,
    fold_ext: ComputePipelineState,
    fold_base: ComputePipelineState,
    transition_ext: ComputePipelineState,
    transition_base: ComputePipelineState,
    first_layer: ComputePipelineState,
    batch_ext: ComputePipelineState,
    batch_base: ComputePipelineState,
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
        let transition_ext = pso("gkr_transition_ext");
        let transition_base = pso("gkr_transition_base");
        let first_layer = pso("first_layer");
        let batch_ext = pso("logup_round_batch_ext");
        let batch_base = pso("logup_round_batch_base");
        Self {
            device,
            ext,
            base,
            fold_ext,
            fold_base,
            transition_ext,
            transition_base,
            first_layer,
            batch_ext,
            batch_base,
            pool: Mutex::new(Vec::new()),
        }
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

    /// This thread's command queue; see [`sp1_metal_probes::nocopy::thread_queue`].
    fn queue(&self) -> CommandQueue {
        sp1_metal_probes::nocopy::thread_queue(&self.device)
    }
}

fn context() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(Context::new)
}

/// Metal takes argument data up to this size inline in the command buffer,
/// which avoids creating a buffer object -- a call into the kernel that, for
/// the small per-chip slices a GKR layer is full of, costs more than the work.
const SET_BYTES_LIMIT: usize = 4096;

/// Staging buffers, kept and reused instead of made fresh for every upload.
///
/// A brand new `MTLBuffer` has cold pages, so `newBufferWithBytes` faults them
/// in while it copies and reaches about a fifth of memcpy speed — measured at
/// 21.7 GB/s against 91.2 GB/s into a buffer that has been written before.
/// Since the copy is the largest cost on this path, that difference is worth
/// more than everything the kernels do.
///
/// The buffers are per-thread and are only released at the start of the next
/// operation, never after a wait. Every entry point here awaits its own work
/// before returning, so at the moment one begins nothing from a previous call
/// can still be reading them; releasing after a wait instead would be wrong for
/// [`fix_layer`], which stages many folds before awaiting any of them.
struct Arena {
    bufs: Vec<Buffer>,
    next: usize,
}

thread_local! {
    static ARENA: RefCell<Arena> = const { RefCell::new(Arena { bufs: Vec::new(), next: 0 }) };
}

/// The most staging buffers one thread will hold. A layer stages four arrays
/// per chip; past this an upload just makes its own buffer, so a pathological
/// shape costs time rather than unbounded memory.
const ARENA_SLOTS: usize = 256;

/// Releases this thread's staging buffers for reuse, and with them the record
/// of what is already on the device. Sound only where no dispatch of ours is
/// outstanding on this thread.
///
/// The two are released together on purpose: a cache entry names a staging
/// buffer, so recycling buffers without forgetting the entries would hand a
/// later reader a buffer that had since been overwritten.
fn arena_reset() {
    ARENA.with(|a| a.borrow_mut().next = 0);
    CACHE.with(|c| c.borrow_mut().clear());
}

/// An MLE that is already on the device.
///
/// A sumcheck round reduces the layer and then fixes it, over exactly the same
/// four MLEs per chip, so uploading each where it is needed copies every one of
/// them twice — and copying already costs more than all the arithmetic.
///
/// Keying on the data pointer is sound because the entry also holds an `Arc` to
/// the MLE: while the entry lives the allocation cannot be freed, so its
/// address cannot have been handed to something else. Entries last until the
/// next reduction begins, which is when the layer they belong to is finished.
struct Cached {
    ptr: usize,
    len: usize,
    buffer: Buffer,
    _pin: Arc<dyn std::any::Any + Send + Sync>,
}

thread_local! {
    static CACHE: RefCell<Vec<Cached>> = const { RefCell::new(Vec::new()) };
}

/// Whether an MLE already uploaded this round is reused, read once.
fn cache_uploads() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_UPLOAD_CACHE").map(|v| v != "0").unwrap_or(true))
}

/// Uploads an MLE's words, or hands back the buffer this layer already made.
fn upload_mle<T: slop_algebra::Field>(mle: &Arc<Mle<T>>) -> Buffer {
    let words = as_words(mle.guts().as_slice());
    if !cache_uploads() {
        return upload(words);
    }
    let (ptr, len) = (words.as_ptr() as usize, words.len());
    let hit = CACHE.with(|c| {
        c.borrow().iter().find(|e| e.ptr == ptr && e.len == len).map(|e| e.buffer.clone())
    });
    if let Some(buffer) = hit {
        stats::record_input(Input::Resident, len * 4, 0);
        return buffer;
    }
    let buffer = upload(words);
    CACHE.with(|c| {
        c.borrow_mut().push(Cached { ptr, len, buffer: buffer.clone(), _pin: mle.clone() })
    });
    buffer
}

/// A staging buffer of at least `len` bytes, or `None` if this thread is
/// already holding as many as it will.
fn arena_take(len: usize) -> Option<Buffer> {
    ARENA.with(|a| {
        let mut a = a.borrow_mut();
        let i = a.next;
        if i >= ARENA_SLOTS {
            return None;
        }
        a.next += 1;
        let big_enough = a.bufs.get(i).is_some_and(|b| b.length() >= len as u64);
        if !big_enough {
            let b = context()
                .device
                .new_buffer(len.max(1) as u64, MTLResourceOptions::StorageModeShared);
            if i == a.bufs.len() {
                a.bufs.push(b);
            } else {
                a.bufs[i] = b;
            }
        }
        Some(a.bufs[i].clone())
    })
}

/// Binds a small slice inline, or through a buffer if it is too large.
fn bind_small(enc: &metal::ComputeCommandEncoderRef, index: u64, bytes: &[u8]) -> Option<Buffer> {
    if bytes.len() <= SET_BYTES_LIMIT && !bytes.is_empty() {
        enc.set_bytes(index, bytes.len() as u64, bytes.as_ptr().cast());
        None
    } else {
        let b = upload(bytes);
        enc.set_buffer(index, Some(&b), 0);
        Some(b)
    }
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
    let aligned = sp1_metal_probes::nocopy::takeable(bytes.as_ptr());
    let start = std::time::Instant::now();
    let buffer = upload_inner(device, bytes, aligned);
    let elapsed = start.elapsed();
    STATS.with_borrow(aligned, bytes.len(), elapsed.as_secs_f64());
    stats::record_input(
        if aligned { Input::Wrapped } else { Input::Copied },
        bytes.len(),
        elapsed.as_nanos() as u64,
    );
    buffer
}

fn upload_inner(device: &Device, bytes: &[u8], aligned: bool) -> Buffer {
    if aligned && !bytes.is_empty() {
        // SAFETY: the buffer borrows `bytes` for the lifetime of the dispatch,
        // which is awaited before this function's caller returns.
        #[allow(unused_unsafe)]
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
    } else if let Some(buf) = arena_take(bytes.len().max(1)) {
        // SAFETY: the buffer was made with at least this many bytes, and shared
        // storage means the pointer is host-writable.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buf.contents().cast::<u8>(),
                bytes.len(),
            );
        }
        buf
    } else {
        device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len().max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

#[derive(Clone, Copy, Default)]
struct StatsInner {
    uploads: usize,
    no_copy: usize,
    upload_s: f64,
    upload_bytes: u64,
    dispatches: usize,
    encode_s: f64,
    gpu_s: f64,
    wall_s: f64,
    cpu_chip_s: f64,
    threads: usize,
}

/// What the Metal path did, so a measurement can say where the time went.
///
/// The GPU figure is the hardware's own: `GPUStartTime` and `GPUEndTime` on the
/// command buffer bracket execution only, so the difference between it and the
/// wall figure is everything around a dispatch — encoding, submission, and
/// waiting for a queue this thread does not own.
pub struct UploadStats {
    inner: Mutex<StatsInner>,
}

impl UploadStats {
    fn with_borrow(&self, no_copy: bool, bytes: usize, secs: f64) {
        let mut s = self.inner.lock().expect("stats poisoned");
        s.uploads += 1;
        s.no_copy += usize::from(no_copy);
        s.upload_bytes += bytes as u64;
        s.upload_s += secs;
    }

    fn add_encode(&self, secs: f64) {
        self.inner.lock().expect("stats poisoned").encode_s += secs;
    }

    fn add_cpu_chip(&self, secs: f64) {
        self.inner.lock().expect("stats poisoned").cpu_chip_s += secs;
    }

    /// Counts this thread once, so thread-seconds can be read against a wall clock.
    fn see_thread(&self) {
        thread_local! {
            static SEEN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        }
        SEEN.with(|seen| {
            if !seen.get() {
                seen.set(true);
                self.inner.lock().expect("stats poisoned").threads += 1;
            }
        });
    }

    /// Everything the Metal path spent, as a printable line.
    #[must_use]
    pub fn report(&self) -> String {
        let s = *self.inner.lock().expect("stats poisoned");
        format!(
            "dispatches {} on {} threads | gpu {:.3}s | blocked {:.3}s | encode {:.3}s | \
             upload {:.3}s for {:.2} GB in {} ({} no-copy) | cpu fallback {:.3}s",
            s.dispatches,
            s.threads,
            s.gpu_s,
            s.wall_s,
            s.encode_s,
            s.upload_s,
            s.upload_bytes as f64 / 1e9,
            s.uploads,
            s.no_copy,
            s.cpu_chip_s,
        )
    }

    fn record(&self, wall_s: f64, gpu_s: f64) {
        let mut s = self.inner.lock().expect("stats poisoned");
        s.dispatches += 1;
        s.wall_s += wall_s;
        // A command buffer that never ran on the GPU reports a zero or
        // decreasing pair; do not let that subtract from the total.
        if gpu_s.is_finite() && gpu_s > 0.0 {
            s.gpu_s += gpu_s;
        }
    }

    /// `(uploads, of which no-copy)`.
    #[must_use]
    pub fn counts(&self) -> (usize, usize) {
        let s = *self.inner.lock().expect("stats poisoned");
        (s.uploads, s.no_copy)
    }

    /// `(dispatches, GPU seconds, wall seconds)` spent in the Metal path.
    #[must_use]
    pub fn timings(&self) -> (usize, f64, f64) {
        let s = *self.inner.lock().expect("stats poisoned");
        (s.dispatches, s.gpu_s, s.wall_s)
    }

    /// Resets the counters.
    pub fn reset(&self) {
        *self.inner.lock().expect("stats poisoned") = StatsInner::default();
    }
}

/// Accounting for the Metal path, reported by the profiling host.
pub static STATS: UploadStats = UploadStats { inner: Mutex::new(StatsInner {
    uploads: 0,
    no_copy: 0,
    upload_s: 0.0,
    upload_bytes: 0,
    dispatches: 0,
    encode_s: 0.0,
    gpu_s: 0.0,
    wall_s: 0.0,
    cpu_chip_s: 0.0,
    threads: 0,
}) };

/// Records an output the kernel wrote, and whether it landed where it was
/// wanted or in a staging buffer that still has to be copied back.
fn record_out(direct: bool, bytes: usize) {
    stats::record_output(if direct { Output::Direct } else { Output::Staged }, bytes, 0);
}

/// Commits a command buffer, waits for it, and records what the GPU spent.
fn run_and_record(cb: &metal::CommandBufferRef) {
    let start = std::time::Instant::now();
    cb.commit();
    cb.wait_until_completed();
    let wall = start.elapsed().as_secs_f64();
    // GPUStartTime and GPUEndTime are not bound by the metal crate, but they
    // are ordinary Objective-C properties on MTLCommandBuffer.
    let (t0, t1): (f64, f64) =
        unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    STATS.record(wall, t1 - t0);
    stats::record_dispatch(wall, t1 - t0);
    stats::record_interval(t0, t1);
}

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

/// One chip's contribution to the round sums, on the CPU.
///
/// A layer's chips span orders of magnitude in size, and a dispatch costs the
/// same whether the chip has a million entries or a hundred. The small ones are
/// computed here and added to what the kernels return; the sums are additive,
/// so mixing the two is exact.
#[allow(clippy::too_many_arguments)]
fn cpu_chip<N: slop_algebra::Field>(
    n0: &[N],
    n1: &[N],
    d0: &[EF],
    d1: &[EF],
    eq_int: &[EF],
    eq_row: &[EF],
    width: usize,
    lambda: EF,
) -> (EF, EF, EF)
where
    EF: slop_algebra::ExtensionField<N>,
{
    let rows = n0.len() / width;
    let n_pairs = (rows.div_ceil(2)).min(eq_row.len() / 2);
    // Over rayon, as upstream's own reduction is: a serial fallback here would
    // be slower than the code it stands in for, which is the opposite of the
    // point.
    (0..n_pairs)
        .into_par_iter()
        .map(|p| {
        let eq0 = eq_row[2 * p];
        let eqs = eq0 + eq_row[2 * p + 1];
        let full = 2 * p + 1 < rows;
        let (mut s_num0, mut s_den0) = (EF::zero(), EF::zero());
        let (mut s_numh, mut s_denh) = (EF::zero(), EF::zero());
        let mut s_eq = EF::zero();
        let base = 2 * p * width;
        for j in 0..width {
            let e = eq_int[j];
            let (a0, a1) = (n0[base + j], n1[base + j]);
            let (b0, b1) = (d0[base + j], d1[base + j]);
            s_num0 += e * (b0 * a1 + b1 * a0);
            s_den0 += e * (b0 * b1);
            let (a0s, a1s, b0s, b1s) = if full {
                let k = base + width + j;
                (a0 + n0[k], a1 + n1[k], b0 + d0[k], b1 + d1[k])
            } else {
                (a0, a1, b0 + EF::one(), b1 + EF::one())
            };
            s_numh += e * (b0s * a1s + b1s * a0s);
            s_denh += e * (b0s * b1s);
            s_eq += e * eqs;
        }
            ((s_num0 * lambda + s_den0) * eq0, (s_numh * lambda + s_denh) * eqs, s_eq)
        })
        .reduce(
            || (EF::zero(), EF::zero(), EF::zero()),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        )
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
) -> (EF, EF, EF)
where
    EF: slop_algebra::ExtensionField<N>,
{
    STATS.see_thread();
    arena_reset();
    let ctx = context();
    let queue = ctx.queue();
    let lambda_words: [u32; 4] = unsafe { std::mem::transmute_copy(&lambda) };
    let mut interaction_offset = 0usize;
    let mut totals = (EF::zero(), EF::zero(), EF::zero());
    // Created on first use, not up front: a command buffer that is never
    // committed still occupies a slot in the queue, and a queue whose slots are
    // all held blocks the next request for one. A layer whose chips all fall to
    // the CPU asks for no dispatches at all.
    let mut command_buffer: Option<&metal::CommandBufferRef> = None;
    let mut held: Vec<(Buffer, usize, Vec<Option<Buffer>>)> = Vec::new();
    let mut small: Vec<SmallChip<N>> = Vec::new();
    let mut batch: Option<(Buffer, usize, Vec<Option<Buffer>>)> = None;
    for (((n0, n1), d0), d1) in
        numerators.0.iter().zip(numerators.1).zip(denominators.0).zip(denominators.1)
    {
        // Every chip advances the interaction offset, whether or not it
        // contributes: upstream increments it outside the `if let Some(inner)`,
        // so a chip with no rows still consumes its share of eq_interaction.
        // Skipping the increment misaligns every chip after the first empty one
        // and eventually indexes past the end.
        let width = n0.num_polynomials();
        let offset = interaction_offset;
        interaction_offset += width;
        let Some(n0_inner) = chip_inner(n0) else { continue };
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
        if len < min_entries_for_gpu() {
            // Too small to be worth a dispatch of its own, but a layer's chips
            // all add into the same three sums, so the whole tail of them can
            // share one. Collected here and dispatched together below.
            small.push(SmallChip {
                n0: n0_inner,
                n1: n1_inner,
                d0: d0_inner,
                d1: d1_inner,
                width,
                n_pairs,
                n_full,
                eqi_off: offset,
                len,
            });
            continue;
        }
        // One thread per (row pair, interaction) cell. The grid is capped and
        // each thread strides over it, so the host's sum over per-threadgroup
        // partials stays bounded however wide the chip is.
        let cells = n_pairs * width;
        let groups = cells.div_ceil(THREADS as usize).min(MAX_GROUPS);
        let partials = ctx.take_partials(groups);

        let b_n0 = upload_mle(&n0_inner);
        let b_n1 = upload_mle(&n1_inner);
        let b_d0 = upload_mle(&d0_inner);
        let b_d1 = upload_mle(&d1_inner);

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Params {
            n_pairs: u32,
            n_full_pairs: u32,
            width: u32,
        }
        let params =
            Params { n_pairs: n_pairs as u32, n_full_pairs: n_full as u32, width: width as u32 };

        let encode_start = std::time::Instant::now();
        let cb = *command_buffer.get_or_insert_with(|| queue.new_command_buffer());
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in [&b_n0, &b_n1, &b_d0, &b_d1].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        let b_eqi = bind_small(
            enc,
            4,
            as_words(&eq_interaction[offset..offset + width]),
        );
        let b_eqr = bind_small(enc, 5, as_words(&eq_row[..2 * n_pairs]));
        enc.set_buffer(6, Some(&partials), 0);
        enc.set_bytes(7, std::mem::size_of::<Params>() as u64, (&params as *const Params).cast());
        enc.set_bytes(8, 16, lambda_words.as_ptr().cast());
        enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
        enc.end_encoding();
        STATS.add_encode(encode_start.elapsed().as_secs_f64());
        // Keep the buffers alive until the whole layer has run.
        held.push((partials, groups, vec![Some(b_n0), Some(b_n1), Some(b_d0), Some(b_d1), b_eqi, b_eqr]));
        let _ = numerator_words;
    }
    // The batch of small chips joins the layer's own command buffer, so it
    // costs no extra submission.
    if !small.is_empty() {
        match stage_small_batch(
            ctx,
            &small,
            eq_interaction,
            eq_row,
            lambda_words,
            &mut command_buffer,
            &queue,
        ) {
            Some(entry) => batch = Some(entry),
            None => {
                let fallback_start = std::time::Instant::now();
                for chip in &small {
                    let c = cpu_chip::<N>(
                        chip.n0.guts().as_slice(),
                        chip.n1.guts().as_slice(),
                        chip.d0.guts().as_slice(),
                        chip.d1.guts().as_slice(),
                        &eq_interaction[chip.eqi_off..chip.eqi_off + chip.width],
                        eq_row,
                        chip.width,
                        lambda,
                    );
                    totals.0 += c.0;
                    totals.1 += c.1;
                    totals.2 += c.2;
                }
                STATS.add_cpu_chip(fallback_start.elapsed().as_secs_f64());
            }
        }
    }
    let Some(cb) = command_buffer else {
        return totals;
    };
    run_and_record(cb);
    if let Some((partials, groups, _)) = batch {
        let raw = partials.contents().cast::<EF>();
        let mut sum = (EF::zero(), EF::zero(), EF::zero());
        for g in 0..groups {
            // SAFETY: the kernel wrote 3 extension elements per threadgroup.
            let (a, b, c) = unsafe { (*raw.add(3 * g), *raw.add(3 * g + 1), *raw.add(3 * g + 2)) };
            sum.0 += a;
            sum.1 += b;
            sum.2 += c;
        }
        ctx.return_partials(partials);
        if batch_check() {
            let mut want = (EF::zero(), EF::zero(), EF::zero());
            for chip in &small {
                let c = cpu_chip::<N>(
                    chip.n0.guts().as_slice(),
                    chip.n1.guts().as_slice(),
                    chip.d0.guts().as_slice(),
                    chip.d1.guts().as_slice(),
                    &eq_interaction[chip.eqi_off..chip.eqi_off + chip.width],
                    eq_row,
                    chip.width,
                    lambda,
                );
                want.0 += c.0;
                want.1 += c.1;
                want.2 += c.2;
            }
            assert!(
                want == sum,
                "the batched round sums for {} small chips differ from the CPU's",
                small.len(),
            );
        }
        totals.0 += sum.0;
        totals.1 += sum.1;
        totals.2 += sum.2;
    }
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
        let _phase = stats::enter(stats::Phase::Gkr);
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
        let _phase = stats::enter(stats::Phase::Gkr);
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
        let _phase = stats::enter(stats::Phase::Gkr);
        let handled = is::<K, EF>()
            && (is::<F, BaseField>() || is::<F, EF>())
;
        if !handled {
            return (
                numerators.0.iter().map(|m| m.fix_last_variable(alpha)).collect(),
                denominators.0.iter().map(|m| m.fix_last_variable(alpha)).collect(),
                numerators.1.iter().map(|m| m.fix_last_variable(alpha)).collect(),
                denominators.1.iter().map(|m| m.fix_last_variable(alpha)).collect(),
            );
        }
        // SAFETY: the type ids match, so these are the same types.
        let a: EF = unsafe { std::mem::transmute_copy(&alpha) };
        pooled(|| {
        let ctx = context();
        let queue = ctx.queue();
        let cb = queue.new_command_buffer();
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
        run_and_record(cb);
        let out = (
            finish_all(staged.0),
            finish_all(staged.1),
            finish_all(staged.2),
            finish_all(staged.3),
        );
        // SAFETY: K is EF, checked above.
        unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) }
        })
    }

    fn fix_denominator(mle: &PaddedMle<K>, alpha: K) -> PaddedMle<K> {
        let _phase = stats::enter(stats::Phase::Gkr);
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

/// Runs `f` inside an autorelease pool.
///
/// Every object Metal hands back is autoreleased, and a Rust program has no
/// pool of its own: without one they accumulate for the life of the thread. A
/// benchmark of a few hundred dispatches never notices; a proof makes millions
/// and is killed for it.
fn pooled<T>(f: impl FnOnce() -> T) -> T {
    metal::objc::rc::autoreleasepool(f)
}

/// Upstream's own reduction, with a note of which reason sent it there.
///
/// The two reasons call for different work if either turns out to be
/// expensive: an interaction layer needs a kernel of its own, while a rejected
/// circuit layer only needs the gate loosened.
static CPU_ROUND: [Mutex<(usize, f64)>; 2] =
    [Mutex::new((0, 0.0)), Mutex::new((0, 0.0))];

fn timed_cpu_round<F, K>(
    poly: &LogupRoundPolynomial<F, K, MetalRoundOps>,
    interaction: bool,
) -> (K, K, K)
where
    F: slop_algebra::Field,
    K: slop_algebra::ExtensionField<F>,
    MetalRoundOps: LogupRoundOps<F, K>,
{
    let start = std::time::Instant::now();
    let out = cpu_round_evals(poly);
    let mut c = CPU_ROUND[usize::from(interaction)].lock().expect("stats poisoned");
    c.0 += 1;
    c.1 += start.elapsed().as_secs_f64();
    out
}

/// `[(rejected layers, seconds), (interaction layers, seconds)]`.
#[must_use]
pub fn cpu_round_counts() -> [(usize, f64); 2] {
    [
        *CPU_ROUND[0].lock().expect("stats poisoned"),
        *CPU_ROUND[1].lock().expect("stats poisoned"),
    ]
}

fn round_evals_base(poly: &LogupRoundPolynomial<BaseField, EF, MetalRoundOps>) -> (EF, EF, EF) {
    match &poly.layer {
        PolynomialLayer::InteractionLayer(_) => timed_cpu_round(poly, true),
        PolynomialLayer::CircuitLayer(layer) if !worth_dispatching(&layer.numerator_0) => {
            timed_cpu_round(poly, false)
        }
        PolynomialLayer::CircuitLayer(layer) => pooled(|| circuit_layer::<BaseField>(
            &context().base,
            (&layer.numerator_0, &layer.numerator_1),
            (&layer.denominator_0, &layer.denominator_1),
            poly.eq_interaction.guts().as_slice(),
            poly.eq_row.guts().as_slice(),
            poly.lambda,
            1,
        )),
    }
}

fn round_evals_ext(poly: &LogupRoundPolynomial<EF, EF, MetalRoundOps>) -> (EF, EF, EF) {
    match &poly.layer {
        PolynomialLayer::InteractionLayer(_) => timed_cpu_round(poly, true),
        PolynomialLayer::CircuitLayer(layer) if !worth_dispatching(&layer.numerator_0) => {
            timed_cpu_round(poly, false)
        }
        PolynomialLayer::CircuitLayer(layer) => pooled(|| circuit_layer::<EF>(
            &context().ext,
            (&layer.numerator_0, &layer.numerator_1),
            (&layer.denominator_0, &layer.denominator_1),
            poly.eq_interaction.guts().as_slice(),
            poly.eq_row.guts().as_slice(),
            poly.lambda,
            4,
        )),
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
    arena_reset();
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

    pooled(|| {
    let mut values: Vec<EF> = Vec::with_capacity(total);
    // SAFETY: the kernel writes every element before anything reads them.
    unsafe { values.set_len(total) };
    let out_bytes = total * std::mem::size_of::<EF>();
    let direct = sp1_metal_probes::nocopy::takeable(values.as_ptr().cast());
    record_out(direct, out_bytes);
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

    let in_buf = upload_mle(&inner);
    let pad_buf = upload(padding_words);
    let alpha_words: [u32; 4] = unsafe { std::mem::transmute_copy(&alpha) };

    let params =
        FoldParams { out_rows: out_rows as u32, rows: rows as u32, width: width as u32 };

    let queue = ctx.queue();
    let cb = queue.new_command_buffer();
    encode_fold(cb, pso, &in_buf, &pad_buf, &out_buf, params, alpha_words, total);
    run_and_record(cb);

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
    })
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
    held: Vec<Option<Buffer>>,
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
    if mle.inner().as_ref().map_or(0, |i| i.guts().as_slice().len()) < min_entries_for_gpu() {
        return Fold::Done(mle.fix_last_variable(alpha));
    }
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
    let direct = sp1_metal_probes::nocopy::takeable(values.as_ptr().cast());
    record_out(direct, out_bytes);
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
    let in_buf = upload_mle(&inner);
    let alpha_words: [u32; 4] = unsafe { std::mem::transmute_copy(&alpha) };
    let params =
        FoldParams { out_rows: out_rows as u32, rows: rows as u32, width: width as u32 };
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(&in_buf), 0);
    let pad_buf = bind_small(enc, 1, as_words(&pad_values));
    enc.set_buffer(2, Some(&out_buf), 0);
    enc.set_bytes(3, std::mem::size_of::<FoldParams>() as u64, (&params as *const FoldParams).cast());
    enc.set_bytes(4, 16, alpha_words.as_ptr().cast());
    let threads = total as u64;
    enc.dispatch_threads(
        MTLSize::new(threads, 1, 1),
        MTLSize::new(THREADS.min(threads.max(1)), 1, 1),
    );
    enc.end_encoding();
    Fold::Pending(PendingFold {
        values,
        out_buf,
        direct,
        out_rows,
        width,
        out_vars,
        padding,
        held: vec![Some(in_buf), pad_buf],
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


/// Fixes any padded MLE's last variable on the GPU, for callers outside the GKR
/// round.
///
/// Zerocheck folds its column MLEs with exactly this operation and no reference
/// to the AIR, so the same kernels serve both phases. The reduction is the part
/// of zerocheck that needs per-chip generated code; the fold does not.
pub fn fix_padded_mle<T, E>(mle: &PaddedMle<T>, alpha: E) -> PaddedMle<E>
where
    T: slop_algebra::Field,
    E: slop_algebra::ExtensionField<T>,
{
    let _phase = stats::enter(stats::Phase::Gkr);
    if is::<E, EF>()
        && mle.inner().as_ref().map_or(0, |i| i.guts().as_slice().len()) >= min_entries_for_gpu()
    {
        // SAFETY: the type ids match, so these are the same types.
        let a: EF = unsafe { std::mem::transmute_copy(&alpha) };
        if is::<T, BaseField>() {
            let m: &PaddedMle<BaseField> = unsafe { &*std::ptr::from_ref(mle).cast() };
            let out = fix_base(m, a);
            return unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) };
        }
        if is::<T, EF>() {
            let m: &PaddedMle<EF> = unsafe { &*std::ptr::from_ref(mle).cast() };
            let out = fix_ext(m, a);
            return unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) };
        }
    }
    mle.fix_last_variable(alpha)
}

/// Whether building the next circuit layer goes to the GPU, read once.
///
/// Gated separately from the round reduction because it is the one part of this
/// backend whose worth is not obvious: it does move work off the CPU, but it
/// also adds uploads, and whether that is a gain depends on how much of the
/// proof is actually running in parallel at the time.
fn transition_on_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_TRANSITION").map(|v| v != "0").unwrap_or(true))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TransitionParams {
    out_rows: u32,
    rows: u32,
    width: u32,
}

/// An output column and the buffer the kernel writes it through.
///
/// Unified memory lets the kernel write straight into the `Vec` that will back
/// the MLE when its pages are aligned for Metal to take them, which for the
/// sizes that matter they are.
struct Out {
    values: Vec<EF>,
    buffer: Buffer,
    direct: bool,
}

impl Out {
    fn new(device: &Device, total: usize) -> Self {
        let mut values: Vec<EF> = Vec::with_capacity(total);
        // SAFETY: the kernel writes every element before anything reads them.
        unsafe { values.set_len(total) };
        let bytes = total * std::mem::size_of::<EF>();
        let direct = sp1_metal_probes::nocopy::takeable(values.as_ptr().cast());
        record_out(direct, bytes);
        let buffer = if direct {
            // SAFETY: borrowed only for the dispatch, which is awaited below.
            unsafe {
                device.new_buffer_with_bytes_no_copy(
                    values.as_ptr().cast(),
                    bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
            }
        } else {
            device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
        };
        Self { values, buffer, direct }
    }

    fn finish(mut self) -> Vec<EF> {
        if !self.direct {
            let bytes = self.values.len() * std::mem::size_of::<EF>();
            let start = std::time::Instant::now();
            // SAFETY: the kernel wrote that many bytes of extension elements.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.buffer.contents().cast::<u8>(),
                    self.values.as_mut_ptr().cast::<u8>(),
                    bytes,
                );
            }
            stats::record_output(Output::Staged, 0, start.elapsed().as_nanos() as u64);
        }
        self.values
    }
}

/// Builds one chip's next circuit layer.
///
/// A layer of the logup circuit adds fractions pairwise, which is one output
/// cell per thread and no reduction at all — the easiest thing in the phase to
/// move, and 12% of a proof's thread-seconds sat in it.
///
/// Returns `None` when the chip is too small to be worth the round trip, or
/// when the fields are not the ones the kernels are written for, leaving the
/// caller to do it the way it always did.
pub fn transition_chip<K, E>(
    n0: &[K],
    d0: &[E],
    n1: &[K],
    d1: &[E],
    width: usize,
    rows: usize,
    out_rows: usize,
) -> Option<(Vec<E>, Vec<E>, Vec<E>, Vec<E>)>
where
    K: slop_algebra::Field,
    E: slop_algebra::Field,
{
    let _phase = stats::enter(stats::Phase::Gkr);
    if !is::<E, EF>() {
        return None;
    }
    let base = is::<K, BaseField>();
    if !base && !is::<K, EF>() {
        return None;
    }
    if !transition_on_gpu() {
        return None;
    }
    let total = out_rows * width;
    if total == 0 || width == 0 || total < min_entries_for_gpu() {
        return None;
    }
    // The kernel reads rows `2q` and `2q + 1` for every output row `q`, so an
    // output row count that does not match the input's would send it past the
    // end. Checked rather than assumed, since the caller derives the two
    // separately.
    if out_rows != rows.div_ceil(2) {
        return None;
    }
    if n0.len() != rows * width || n1.len() != rows * width {
        return None;
    }
    if d0.len() != rows * width || d1.len() != rows * width {
        return None;
    }
    arena_reset();
    let ctx = context();
    let out = pooled(|| {
        let outs = [
            Out::new(&ctx.device, total),
            Out::new(&ctx.device, total),
            Out::new(&ctx.device, total),
            Out::new(&ctx.device, total),
        ];
        let inputs = [
            upload(as_words(n0)),
            upload(as_words(d0)),
            upload(as_words(n1)),
            upload(as_words(d1)),
        ];
        let params = TransitionParams {
            out_rows: out_rows as u32,
            rows: rows as u32,
            width: width as u32,
        };
        let queue = ctx.queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(if base {
            &ctx.transition_base
        } else {
            &ctx.transition_ext
        });
        for (i, b) in inputs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        for (i, o) in outs.iter().enumerate() {
            enc.set_buffer(4 + i as u64, Some(&o.buffer), 0);
        }
        enc.set_bytes(
            8,
            std::mem::size_of::<TransitionParams>() as u64,
            (&params as *const TransitionParams).cast(),
        );
        enc.dispatch_threads(
            MTLSize::new(total as u64, 1, 1),
            MTLSize::new(THREADS.min(total as u64), 1, 1),
        );
        enc.end_encoding();
        run_and_record(cb);
        let [a, b, c, d] = outs;
        (a.finish(), b.finish(), c.finish(), d.finish())
    });
    // SAFETY: E is EF, checked above.
    Some(unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(out)) })
}

/// The tables the first-layer kernel walks, flattened out of a chip's
/// interactions.
///
/// Every affine form the chip evaluates -- each interaction's values and its
/// multiplicity -- becomes one entry in `val_const` and `term_off`, and its
/// terms become entries in `term_col` and `term_w`. Nothing here depends on the
/// trace, only on the chip, so it is rebuilt per shard at a cost proportional
/// to the number of terms rather than to the height.
struct FirstLayerTables {
    val_off: Vec<u32>,
    mult_val: Vec<u32>,
    term_off: Vec<u32>,
    val_const: Vec<u32>,
    term_col: Vec<u32>,
    term_w: Vec<u32>,
    arg_index: Vec<u32>,
    is_send: Vec<u32>,
}

/// The Montgomery word behind a field element the kernels read.
fn monty_word<F: slop_algebra::Field>(x: F) -> u32 {
    // SAFETY: this is only called for KoalaBear, which is repr(transparent)
    // over its Montgomery word; the caller checks the type first.
    unsafe { *std::ptr::from_ref(&x).cast::<u32>() }
}

impl FirstLayerTables {
    fn build<F: slop_algebra::Field>(interactions: &[(&crate::Interaction<F>, bool)]) -> Self {
        let mut t = FirstLayerTables {
            val_off: Vec::with_capacity(interactions.len() + 1),
            mult_val: Vec::with_capacity(interactions.len()),
            term_off: vec![0],
            val_const: Vec::new(),
            term_col: Vec::new(),
            term_w: Vec::new(),
            arg_index: Vec::with_capacity(interactions.len()),
            is_send: Vec::with_capacity(interactions.len()),
        };
        let mut push_value = |t: &mut FirstLayerTables, col: &slop_air::VirtualPairCol<F>| {
            t.val_const.push(monty_word(col.constant));
            for (pair, weight) in &col.column_weights {
                let encoded = match pair {
                    slop_air::PairCol::Preprocessed(i) => (*i as u32) | 0x8000_0000,
                    slop_air::PairCol::Main(i) => *i as u32,
                };
                t.term_col.push(encoded);
                t.term_w.push(monty_word(*weight));
            }
            t.term_off.push(t.term_col.len() as u32);
        };
        for (interaction, is_send) in interactions {
            t.val_off.push(t.val_const.len() as u32);
            for value in &interaction.values {
                push_value(&mut t, value);
            }
            // The multiplicity is one more affine form; it is appended after
            // the values so that `val_off` still bounds the values alone.
            t.mult_val.push(t.val_const.len() as u32);
            push_value(&mut t, &interaction.multiplicity);
            t.arg_index
                .push(monty_word(F::from_canonical_usize(interaction.argument_index())));
            t.is_send.push(u32::from(*is_send));
        }
        // A sentinel, so that the arrays are never read with an index the
        // kernel cannot bound. The values of interaction `i` run from
        // `val_off[i]` to `mult_val[i]`, which is where its multiplicity sits.
        t.val_off.push(t.val_const.len() as u32);
        t
    }
}

/// Whether the first layer is built on the GPU, overridable with
/// `SP1_METAL_FIRST_LAYER`.
fn first_layer_on_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_FIRST_LAYER").map(|v| v != "0").unwrap_or(true))
}

/// Whether to build the first layer both ways and compare, with
/// `SP1_METAL_FIRST_LAYER=check`.
///
/// A synthetic parity test would have to invent chips; this checks the kernel
/// against upstream's own loop on every chip of a real proof, which is a far
/// larger set of shapes than anything worth writing by hand. It roughly doubles
/// the cost of the phase, so it is a check to run, not a mode to prove in.
#[must_use]
pub fn first_layer_check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_FIRST_LAYER").is_ok_and(|v| v == "check"))
}

/// Fills one chip's numerator and denominator evaluations for the first GKR
/// layer.
///
/// Returns `false` when the chip is too small to be worth a dispatch, or the
/// fields are not the ones the kernels are written for, leaving the caller to
/// fill them the way it always did. On `true` both slices are fully written.
#[allow(clippy::too_many_arguments)]
pub(crate) fn first_layer_chip<F, EF>(
    interactions: &[(&crate::Interaction<F>, bool)],
    main: &[F],
    main_width: usize,
    prep: &[F],
    prep_width: usize,
    rows: usize,
    alpha: EF,
    betas: &[EF],
    numer_out: &mut [F],
    denom_out: &mut [EF],
) -> bool
where
    F: slop_algebra::Field,
    EF: slop_algebra::Field,
{
    if (!first_layer_on_gpu() && !first_layer_check())
        || !is::<F, BaseField>()
        || !is::<EF, EF2>()
    {
        return false;
    }
    let n = interactions.len();
    let total = rows * n;
    if n == 0 || rows == 0 || total < min_entries_for_gpu() {
        return false;
    }
    debug_assert_eq!(numer_out.len(), total);
    debug_assert_eq!(denom_out.len(), total);
    // The kernel indexes `betas` by an interaction's position, so a chip with
    // more values than the seed provides would read past the end.
    let widest = interactions.iter().map(|(i, _)| i.values.len()).max().unwrap_or(0);
    if betas.len() < widest + 1 {
        return false;
    }
    // The kernel indexes both traces by row, so refuse anything it could read
    // past rather than trusting the caller's widths.
    if main.len() < rows * main_width || prep.len() < rows * prep_width {
        return false;
    }
    let _phase = stats::enter(stats::Phase::Gkr);
    let tables = FirstLayerTables::build(interactions);

    let ctx = context();
    let queue = ctx.queue();
    metal::objc::rc::autoreleasepool(|| {
        // SAFETY: every buffer below borrows a slice that outlives the single
        // dispatch, which is awaited before this returns.
        let up = |bytes: &[u8]| unsafe { sp1_metal_probes::nocopy::read_only(&ctx.device, bytes) };
        let b_main = up(as_words(main));
        // An absent preprocessed table still has to be bound, and an empty
        // slice has no address worth taking, so bind a word of nothing. No
        // term names a preprocessed column in that case, so it is never read.
        let nothing = [0u32; 4];
        let b_prep = up(if prep.is_empty() { as_words(&nothing) } else { as_words(prep) });
        let b_betas = up(as_words(betas));
        let b_val_off = up(as_words(&tables.val_off));
        let b_mult_val = up(as_words(&tables.mult_val));
        let b_term_off = up(as_words(&tables.term_off));
        let b_val_const = up(as_words(&tables.val_const));
        let b_term_col = up(as_words(&tables.term_col));
        let b_term_w = up(as_words(&tables.term_w));
        let b_arg = up(as_words(&tables.arg_index));
        let b_send = up(as_words(&tables.is_send));

        let numer_bytes = std::mem::size_of_val(numer_out);
        let denom_bytes = std::mem::size_of_val(denom_out);
        // SAFETY: as above, and nothing reads the outputs until the wait below.
        let (b_numer, numer_direct) = unsafe {
            sp1_metal_probes::nocopy::write_into(
                &ctx.device,
                std::slice::from_raw_parts_mut(numer_out.as_mut_ptr().cast::<u8>(), numer_bytes),
            )
        };
        let (b_denom, denom_direct) = unsafe {
            sp1_metal_probes::nocopy::write_into(
                &ctx.device,
                std::slice::from_raw_parts_mut(denom_out.as_mut_ptr().cast::<u8>(), denom_bytes),
            )
        };

        #[repr(C)]
        struct Params {
            rows: u32,
            num_interactions: u32,
            main_width: u32,
            prep_width: u32,
        }
        let params = Params {
            rows: rows as u32,
            num_interactions: n as u32,
            main_width: main_width as u32,
            prep_width: prep_width as u32,
        };
        let alpha_words: [u32; 4] = unsafe { std::mem::transmute_copy(&alpha) };

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.first_layer);
        for (i, b) in [
            &b_main, &b_prep, &b_betas, &b_val_off, &b_mult_val, &b_term_off, &b_val_const,
            &b_term_col, &b_term_w, &b_arg, &b_send, &b_numer,
        ]
        .into_iter()
        .enumerate()
        {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_buffer(12, Some(&b_denom), 0);
        enc.set_bytes(13, std::mem::size_of::<Params>() as u64, (&params as *const Params).cast());
        enc.set_bytes(14, 16, alpha_words.as_ptr().cast());
        let groups = (total as u64).div_ceil(THREADS).min(MAX_GROUPS as u64).max(1);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(THREADS, 1, 1));
        enc.end_encoding();
        run_and_record(cb);

        for (direct, buffer, ptr, bytes) in [
            (numer_direct, &b_numer, numer_out.as_mut_ptr().cast::<u8>(), numer_bytes),
            (denom_direct, &b_denom, denom_out.as_mut_ptr().cast::<u8>(), denom_bytes),
        ] {
            if direct {
                continue;
            }
            let start = std::time::Instant::now();
            // SAFETY: the kernel wrote exactly this many bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(buffer.contents().cast::<u8>(), ptr, bytes);
            }
            stats::record_output(Output::Staged, bytes, start.elapsed().as_nanos() as u64);
        }
    });
    true
}

/// A chip of a layer too small to be worth its own dispatch.
struct SmallChip<N> {
    n0: Arc<Mle<N>>,
    n1: Arc<Mle<N>>,
    d0: Arc<Mle<EF>>,
    d1: Arc<Mle<EF>>,
    width: usize,
    n_pairs: usize,
    n_full: usize,
    eqi_off: usize,
    len: usize,
}

/// What the batched kernel needs to know about one chip in the batch.
#[repr(C)]
#[derive(Clone, Copy)]
struct ChipDesc {
    cells_before: u32,
    data_off: u32,
    eqi_off: u32,
    width: u32,
    n_pairs: u32,
    n_full: u32,
}

/// Whether to sum the batched chips both ways and compare, with
/// `SP1_METAL_BATCH=check`.
///
/// The batch kernel resolves a cell to its chip by binary search over an offset
/// table, which is the kind of arithmetic that is either exactly right or
/// wildly wrong, and a proof that still verifies is a coarse way to find out.
fn batch_check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_BATCH").is_ok_and(|v| v == "check"))
}

/// Whether small chips are batched at all, overridable with `SP1_METAL_BATCH`.
fn batch_on_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_BATCH").map(|v| v != "0").unwrap_or(true))
}

/// Copies a layer's small chips side by side and encodes one dispatch over all
/// of them, returning the partials to sum once the layer has run.
///
/// This is the one place the backend copies deliberately. Every other upload
/// borrows the caller's pages, but these arrays have to be adjacent for a
/// single kernel to walk them, and a chip below the dispatch threshold is at
/// most a few hundred kilobytes. The alternative -- a dispatch each -- was
/// measured and is a wash, which is what makes the copy worth paying.
///
/// Returns `None` when the batch is too small to be worth it or the fields are
/// not the ones the kernels are written for, leaving the caller to sum the
/// chips on the CPU.
fn stage_small_batch<'a, N: slop_algebra::Field>(
    ctx: &'static Context,
    small: &[SmallChip<N>],
    eq_interaction: &[EF],
    eq_row: &[EF],
    lambda_words: [u32; 4],
    command_buffer: &mut Option<&'a metal::CommandBufferRef>,
    queue: &'a CommandQueue,
) -> Option<(Buffer, usize, Vec<Option<Buffer>>)>
where
    EF: slop_algebra::ExtensionField<N>,
{
    let base = if is::<N, BaseField>() {
        true
    } else if is::<N, EF>() {
        false
    } else {
        return None;
    };
    if !batch_on_gpu() && !batch_check() {
        return None;
    }
    let total_cells: usize = small.iter().map(|c| c.n_pairs * c.width).sum();
    if small.len() < 2 || total_cells * 2 < min_layer_entries() {
        return None;
    }
    let total_len: usize = small.iter().map(|c| c.len).sum();
    let num_size = if base { 4 } else { 16 };

    let b_n0 = arena_take(total_len * num_size)?;
    let b_n1 = arena_take(total_len * num_size)?;
    let b_d0 = arena_take(total_len * 16)?;
    let b_d1 = arena_take(total_len * 16)?;

    let mut descs = Vec::with_capacity(small.len());
    let (mut cells, mut off) = (0usize, 0usize);
    for chip in small {
        descs.push(ChipDesc {
            cells_before: cells as u32,
            data_off: off as u32,
            eqi_off: chip.eqi_off as u32,
            width: chip.width as u32,
            n_pairs: chip.n_pairs as u32,
            n_full: chip.n_full as u32,
        });
        for (buffer, bytes, elem) in [
            (&b_n0, as_words(chip.n0.guts().as_slice()), num_size),
            (&b_n1, as_words(chip.n1.guts().as_slice()), num_size),
            (&b_d0, as_words(chip.d0.guts().as_slice()), 16),
            (&b_d1, as_words(chip.d1.guts().as_slice()), 16),
        ] {
            debug_assert_eq!(bytes.len(), chip.len * elem);
            // SAFETY: the staging buffer was made with room for every chip's
            // elements, and shared storage is host-writable.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().cast::<u8>().add(off * elem),
                    bytes.len(),
                );
            }
        }
        stats::record_input(Input::Copied, chip.len * (2 * num_size + 32), 0);
        cells += chip.n_pairs * chip.width;
        off += chip.len;
    }

    // The descriptors are staged rather than borrowed: this function returns
    // before the layer's command buffer runs, so a buffer over `descs` would be
    // reading a local that no longer exists. The two eq arrays are the
    // caller's and outlive the dispatch, so those can be taken where they lie.
    let desc_bytes = as_words(&descs);
    let b_desc = arena_take(desc_bytes.len())?;
    // SAFETY: the staging buffer was made with at least this many bytes, and
    // shared storage is host-writable.
    unsafe {
        std::ptr::copy_nonoverlapping(
            desc_bytes.as_ptr(),
            b_desc.contents().cast::<u8>(),
            desc_bytes.len(),
        );
    }
    stats::record_input(Input::Copied, desc_bytes.len(), 0);
    let b_eqi = upload(as_words(eq_interaction));
    let b_eqr = upload(as_words(eq_row));

    #[repr(C)]
    struct BatchParams {
        n_chips: u32,
        total_cells: u32,
    }
    let params = BatchParams { n_chips: small.len() as u32, total_cells: total_cells as u32 };
    let groups = total_cells.div_ceil(THREADS as usize).min(MAX_GROUPS);
    let partials = ctx.take_partials(groups);

    let pso = if base { &ctx.batch_base } else { &ctx.batch_ext };
    let cb = *command_buffer.get_or_insert_with(|| queue.new_command_buffer());
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    for (i, b) in [&b_n0, &b_n1, &b_d0, &b_d1, &b_eqi, &b_eqr].into_iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_buffer(6, Some(&partials), 0);
    enc.set_bytes(7, std::mem::size_of::<BatchParams>() as u64, (&params as *const BatchParams).cast());
    enc.set_bytes(8, 16, lambda_words.as_ptr().cast());
    enc.set_buffer(9, Some(&b_desc), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(groups as u64, 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
    enc.end_encoding();
    Some((partials, groups, vec![Some(b_n0), Some(b_n1), Some(b_d0), Some(b_d1), Some(b_desc), Some(b_eqi), Some(b_eqr)]))
}
