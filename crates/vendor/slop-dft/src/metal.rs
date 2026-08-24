// SPDX-License-Identifier: MIT OR Apache-2.0
// Added to slop-dft 6.4.0 by the sp1-metal project; see CHANGES.md.
//! The Reed-Solomon encode, on the GPU.
//!
//! Committing a shard's traces is the second largest phase of a proof, and it
//! is two operations: a Reed-Solomon encode of each trace, then a Merkle tree
//! over the codeword. The tree moved to the GPU first because it was reachable
//! — `BasefoldProver` takes its committer as a type parameter. The encode was
//! not: `CpuDftEncoder` holds a concrete `Arc<Radix2DitParallel>`, so there is
//! no seam above it.
//!
//! There is one below it. `Radix2DitParallel`'s only interesting property is
//! that it implements [`Dft`](crate::Dft), and that impl lives here, in a
//! forty-line crate that nothing else in the prover depends on. Replacing the
//! body of one trait method reaches every encode the prover performs without
//! forking anything above it — which is why this crate is vendored and
//! `slop-basefold-prover` is not.
//!
//! What upstream computes is not a coset LDE. `dft(src, log_blowup,
//! BitReversed, 0)` zero-pads each column to `rows << log_blowup`, takes a
//! forward DFT, and bit-reverses the rows. Decimation-in-frequency takes
//! natural-order input to bit-reversed output as it stands, so matching that
//! convention costs no permutation: the kernels are Gentleman-Sande and the
//! result is equal to upstream's element for element, which the probe suite
//! checks at two shapes, one with the tiled tail and one without.
//!
//! Anything outside that one case — a shift, a natural-order result, a
//! dimension other than the first, a field that is not KoalaBear — is handed
//! back to upstream rather than approximated. There is exactly one caller in
//! the prover and it asks for the case above.
//!
//! Set `SP1_METAL_ENCODE=0` to force the CPU encoder through the same binary,
//! and `SP1_METAL_ENCODE=check` to run both and compare, which is how the
//! parity claim above is made on a real proof rather than on a test shape.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use slop_algebra::TwoAdicField;
use slop_alloc::CpuBackend;
use slop_tensor::Tensor;
use sp1_metal_core::{nocopy, queue};
use sp1_metal_core::stats::{self, Input, Output, Phase};

use crate::DftOrdering;

/// Threads per threadgroup, matching the tiled kernel's assumption of 256.
const THREADS: u64 = 256;

/// Rows and columns of the tile `dif_tail` holds in threadgroup memory, and how
/// many stages it runs while it is resident. Must match the shader.
const TILE_ROWS: usize = 64;
const TILE_COLS: usize = 64;
const TILE_STAGES: usize = 6;

/// Below this a dispatch and its round trip cost more than the transform saves.
const MIN_CELLS: usize = 1 << 14;

/// Kernel parameters, matching `DifParams` in the shader.
#[repr(C)]
#[derive(Clone, Copy)]
struct DifParams {
    log_n: u32,
    log_m: u32,
    cols: u32,
    log_blowup: u32,
}

struct Context {
    device: Device,
    spread: ComputePipelineState,
    stage: ComputePipelineState,
    stage4: ComputePipelineState,
    tail: ComputePipelineState,
    /// One twiddle table per codeword height, built once and kept.
    ///
    /// The table is `n / 2` elements and every trace of that height reads the
    /// same one, so building it per encode would be a measurable fraction of
    /// the work it feeds. It is a copy rather than a borrow because it outlives
    /// every dispatch that reads it and nothing on the host ever looks at it
    /// again.
    twiddles: Mutex<HashMap<usize, Buffer>>,
}

/// The kernels, and the field arithmetic they are written against.
///
/// The header comes from the probe suite by path rather than by depending on
/// it: that crate sits above this one, and its parity checks compile the file
/// beside this one, so both sides of the check are the same source.
fn shader_source() -> String {
    let probes = include_str!("../../../sp1-metal-probes/src/shaders.metal");
    let header = probes
        .split("// ---- end of shared field arithmetic ----")
        .next()
        .expect("shader header sentinel");
    let one = ((1u64 << 32) % 0x7f00_0001u64) as u32;
    format!("{header}\nconstant uint KB_ONE = {one}u;\n{}", include_str!("shaders.metal"))
}

fn context() -> Option<&'static Context> {
    static CTX: OnceLock<Option<Context>> = OnceLock::new();
    CTX.get_or_init(|| {
        let device = Device::system_default()?;
        let start = std::time::Instant::now();
        let library =
            device.new_library_with_source(&shader_source(), &metal::CompileOptions::new()).ok()?;
        stats::record_compile(start.elapsed().as_secs_f64());
        let pso = |name: &str| -> Option<ComputePipelineState> {
            let f = library.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&f).ok()
        };
        Some(Context {
            spread: pso("dif_spread")?,
            stage: pso("dif_stage")?,
            stage4: pso("dif_stage4")?,
            tail: pso("dif_tail")?,
            twiddles: Mutex::new(HashMap::new()),
            device,
        })
    })
    .as_ref()
}

/// How the encode is switched, read once.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Cpu,
    Gpu,
    /// Run both and compare every codeword, which is slower than either.
    Check,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("SP1_METAL_ENCODE").as_deref() {
        Ok("0") => Mode::Cpu,
        Ok("check") => Mode::Check,
        _ => Mode::Gpu,
    })
}

/// Whether the tiled tail applies: it needs its six stages to be below the ones
/// the spread already did, and a whole number of tiles in both directions.
fn tail_applies(n: usize, cols: usize, log_blowup: usize) -> bool {
    n % TILE_ROWS == 0
        && cols % TILE_COLS == 0
        && n.trailing_zeros() as usize >= TILE_STAGES + log_blowup
}

/// `w_n^j` for `j < n / 2`, in the Montgomery words the kernels read.
///
/// A field element is `repr(transparent)` over that word, so the table is built
/// as field elements and reinterpreted rather than converted.
fn twiddle_words<F: TwoAdicField>(n: usize) -> Vec<u32> {
    let root = F::two_adic_generator(n.trailing_zeros() as usize);
    let mut cur = F::one();
    let mut w = Vec::with_capacity(n / 2);
    for _ in 0..n / 2 {
        w.push(cur);
        cur *= root;
    }
    // SAFETY: `F` is KoalaBear, checked by the caller, which is `repr(transparent)`
    // over the `u32` Montgomery word these kernels compute in.
    let mut w = std::mem::ManuallyDrop::new(w);
    unsafe { Vec::from_raw_parts(w.as_mut_ptr().cast::<u32>(), w.len(), w.capacity()) }
}

/// This height's twiddle table, built on first use and kept.
fn twiddles<F: TwoAdicField>(ctx: &Context, n: usize) -> Buffer {
    let mut cache = ctx.twiddles.lock().expect("twiddle cache poisoned");
    cache
        .entry(n)
        .or_insert_with(|| {
            let words = twiddle_words::<F>(n);
            ctx.device.new_buffer_with_data(
                words.as_ptr().cast(),
                (words.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .clone()
}

fn set_params(enc: &ComputeCommandEncoderRef, index: u64, p: &DifParams) {
    enc.set_bytes(index, std::mem::size_of::<DifParams>() as u64, (p as *const DifParams).cast());
}

/// Dispatches a grid with the column index fastest, so consecutive threads read
/// consecutive columns of the same row.
fn dispatch_2d(enc: &ComputeCommandEncoderRef, cols: u64, rows: u64) {
    let tw = THREADS.min(cols.max(1));
    let th = (THREADS / tw).max(1).min(rows.max(1));
    enc.dispatch_threads(MTLSize::new(cols, rows, 1), MTLSize::new(tw, th, 1));
}

/// Encodes the whole transform into one compute pass: the spread, a global pass
/// per stage down to the tile height, then the six tiled stages.
#[allow(clippy::too_many_arguments)]
fn encode_dif(
    enc: &ComputeCommandEncoderRef,
    ctx: &Context,
    src: &Buffer,
    dst: &Buffer,
    tw: &Buffer,
    rows: usize,
    cols: usize,
    log_blowup: usize,
) {
    let n = rows << log_blowup;
    let log_n = n.trailing_zeros() as usize;
    let mut params = DifParams {
        log_n: log_n as u32,
        log_m: 0,
        cols: cols as u32,
        log_blowup: log_blowup as u32,
    };
    enc.set_compute_pipeline_state(&ctx.spread);
    enc.set_buffer(0, Some(src), 0);
    enc.set_buffer(1, Some(dst), 0);
    enc.set_buffer(2, Some(tw), 0);
    set_params(enc, 3, &params);
    dispatch_2d(enc, cols as u64, rows as u64);
    // The spread did the top `log_blowup` stages; the tail, when it applies,
    // will do the bottom six while a tile is resident. Everything between them
    // goes two stages to a pass, since the transform is memory-bound and a
    // radix-4 butterfly costs the same multiplies as the two it replaces.
    let lowest = if tail_applies(n, cols, log_blowup) { TILE_STAGES } else { 0 };
    let mut s = log_n - log_blowup;
    if (s - lowest) % 2 == 1 {
        s -= 1;
        params.log_m = s as u32;
        enc.set_compute_pipeline_state(&ctx.stage);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_params(enc, 2, &params);
        dispatch_2d(enc, cols as u64, (n / 2) as u64);
    }
    while s > lowest {
        params.log_m = (s - 1) as u32;
        enc.set_compute_pipeline_state(&ctx.stage4);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_params(enc, 2, &params);
        dispatch_2d(enc, cols as u64, (n / 4) as u64);
        s -= 2;
    }
    if lowest > 0 {
        params.log_m = 0;
        enc.set_compute_pipeline_state(&ctx.tail);
        enc.set_buffer(0, Some(dst), 0);
        enc.set_buffer(1, Some(tw), 0);
        set_params(enc, 2, &params);
        enc.dispatch_thread_groups(
            MTLSize::new((cols / TILE_COLS) as u64, (n / TILE_ROWS) as u64, 1),
            MTLSize::new(THREADS, 1, 1),
        );
    }
}

/// Whether this call is one the kernels compute exactly.
///
/// Everything here is a property of the request rather than of the data, so a
/// caller that is refused is refused every time and the fallback is not a
/// surprise partway through a proof.
fn eligible<F: TwoAdicField>(
    src: &Tensor<F, CpuBackend>,
    shift: F,
    log_blowup: usize,
    ordering: DftOrdering,
    dim: usize,
) -> bool {
    if mode() == Mode::Cpu || TypeId::of::<F>() != TypeId::of::<p3_koala_bear::KoalaBear>() {
        return false;
    }
    if dim != 0 || ordering != DftOrdering::BitReversed || shift != F::one() {
        return false;
    }
    if src.sizes().len() != 2 {
        return false;
    }
    let (rows, cols) = (src.sizes()[0], src.sizes()[1]);
    // The spread does the top `log_blowup` stages, so the transform needs at
    // least one more below them: two rows is the smallest trace that has one.
    log_blowup >= 1 && rows >= 2 && rows.is_power_of_two() && cols > 0 && rows * cols >= MIN_CELLS
}

/// Runs the encode on the GPU, writing the codeword into `dst` where it lies.
///
/// Returns whether it ran: a caller this cannot compute exactly gets `false`
/// and must fall through to upstream's encoder.
pub fn coset_dft_into<F: TwoAdicField>(
    src: &Tensor<F, CpuBackend>,
    dst: &mut Tensor<F, CpuBackend>,
    shift: F,
    log_blowup: usize,
    ordering: DftOrdering,
    dim: usize,
) -> bool {
    if !eligible(src, shift, log_blowup, ordering, dim) {
        return false;
    }
    let Some(ctx) = context() else {
        return false;
    };
    let (rows, cols) = (src.sizes()[0], src.sizes()[1]);
    let n = rows << log_blowup;
    let total = n * cols;
    assert_eq!(dst.total_len(), total, "destination is not the codeword's shape");
    // A tensor from `with_sizes_in` has the capacity but a length of zero: the
    // upstream path fills it by extending, so it never has to say so. Writing
    // into it from a kernel does, and forgetting is silent — the codeword is
    // correct in memory and every reader sees an empty slice.
    assert!(dst.storage.capacity() >= total, "destination has no room for the codeword");
    let _phase = stats::enter(Phase::Commit);

    // Both sides are borrowed where they lie when Metal will take them. The
    // codeword is the larger of the two by the blowup factor, and it is what
    // the Merkle kernels read next, so a copy here would be paid twice.
    let in_bytes = rows * cols * 4;
    let out_bytes = n * cols * 4;
    let start = std::time::Instant::now();
    // SAFETY: the tensor's storage is `rows * cols` field elements, each a
    // 4-byte Montgomery word, and it is not written while the buffer lives.
    let b_src = unsafe {
        nocopy::read_only(
            &ctx.device,
            std::slice::from_raw_parts(src.as_slice().as_ptr().cast::<u8>(), in_bytes),
        )
    };
    let upload = start.elapsed().as_nanos() as u64;
    stats::record_input(
        if nocopy::takeable(src.as_slice().as_ptr().cast()) { Input::Wrapped } else { Input::Copied },
        in_bytes,
        upload,
    );
    // SAFETY: `dst` holds `n * cols` elements, asserted above, and is uniquely
    // borrowed for the whole of this call.
    let (b_dst, direct) = unsafe {
        nocopy::write_into(
            &ctx.device,
            std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<u8>(), out_bytes),
        )
    };
    if direct {
        // The Merkle kernels hash this codeword next, on this thread's queue,
        // and nothing on the host touches it in between. It is the largest
        // borrowed allocation in the proof, and mapping it twice is pure cost.
        sp1_metal_core::handoff::keep(dst.as_mut_ptr().cast::<u8>(), out_bytes, &b_dst);
    }
    let tw = twiddles::<F>(ctx, n);

    let queue = nocopy::thread_queue(&ctx.device);
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    encode_dif(enc, ctx, &b_src, &b_dst, &tw, rows, cols, log_blowup);
    enc.end_encoding();
    // The codeword is read next by the Merkle kernels that commit it, and they
    // run on this thread's queue behind this buffer, so the host has no reason
    // to wait here. It does have to wait when it is about to read the codeword
    // itself: to copy it out of a staging buffer, or to compare it against the
    // CPU encoder under `SP1_METAL_ENCODE=check`.
    if direct && !checking() {
        queue::submit_deferred(cb);
        stats::record_output(Output::Direct, out_bytes, 0);
    } else {
        queue::submit(cb);
    }

    if direct {
        // Recorded above when deferred.
        if checking() {
            stats::record_output(Output::Direct, out_bytes, 0);
        }
    } else {
        let start = std::time::Instant::now();
        // SAFETY: the kernel wrote exactly `out_bytes` bytes into the staging
        // buffer, and `dst` has room for them.
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_dst.contents().cast::<u8>(),
                dst.as_mut_ptr().cast::<u8>(),
                out_bytes,
            );
        }
        stats::record_output(Output::Staged, out_bytes, start.elapsed().as_nanos() as u64);
    }
    // SAFETY: the kernel initialised every one of these elements, and the
    // capacity was checked above.
    unsafe { dst.storage.set_len(total) };
    true
}

/// Whether every codeword should also be computed on the CPU and compared.
#[must_use]
pub fn checking() -> bool {
    mode() == Mode::Check
}
