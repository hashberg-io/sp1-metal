//! The jagged sumcheck's round and fold, on the GPU.
//!
//! Two operations per round over the same pair of arrays: three reductions for
//! the round polynomial, and a fold of each array to half its length.
//!
//! The fold is memory-bound — one extension multiply per sixteen bytes written
//! — which is exactly the case where a port only pays if nothing is copied. So
//! the output is written straight into the `Vec` that will back the next
//! round's MLE, through a buffer that borrows its pages, and inputs are kept
//! and reused rather than uploaded again each round. A first attempt at this
//! did neither and came out slower than the CPU, which is what that costs.
//!
//! The field arithmetic is included from the probe suite by path rather than by
//! depending on it, since that crate sits beside this one rather than below it.

use std::sync::OnceLock;

use metal::objc::{msg_send, sel, sel_impl};
use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use sp1_metal_probes::stats::{self, Input, Output};

const THREADS: u64 = 256;
/// Below this a dispatch costs more than the arithmetic saves.
const MIN_PAIRS: usize = 1 << 12;
const PAGE: usize = 16 * 1024;

const KERNELS: &str = r#"
// One thread per pair: the products at the even index, the odd one, and of the
// pairwise sums, all from one pass over the two arrays.
#define HADAMARD_ROUND(NAME, BASE_T, MUL, ADD_BASE)                                  \
kernel void NAME(device const BASE_T* base [[buffer(0)]],                            \
                 device const uint4* ext [[buffer(1)]],                              \
                 device uint4* partials [[buffer(2)]],                               \
                 constant uint& pairs [[buffer(3)]],                                 \
                 uint gid [[thread_position_in_grid]],                               \
                 uint tid [[thread_position_in_threadgroup]],                        \
                 uint tgid [[threadgroup_position_in_grid]]) {                       \
    threadgroup uint4 s0[256];                                                       \
    threadgroup uint4 s1[256];                                                       \
    threadgroup uint4 sh[256];                                                       \
    uint4 a0 = uint4(0u), a1 = uint4(0u), ah = uint4(0u);                            \
    if (gid < pairs) {                                                               \
        uint i = 2u * gid;                                                           \
        BASE_T b0 = base[i], b1 = base[i + 1u];                                      \
        uint4 e0 = ext[i], e1 = ext[i + 1u];                                         \
        a0 = MUL(e0, b0);                                                            \
        a1 = MUL(e1, b1);                                                            \
        ah = MUL(ext_add(e0, e1), ADD_BASE(b0, b1));                                 \
    }                                                                                \
    s0[tid] = a0; s1[tid] = a1; sh[tid] = ah;                                        \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                 \
    for (uint s = 128u; s > 0u; s >>= 1u) {                                          \
        if (tid < s) {                                                               \
            s0[tid] = ext_add(s0[tid], s0[tid + s]);                                 \
            s1[tid] = ext_add(s1[tid], s1[tid + s]);                                 \
            sh[tid] = ext_add(sh[tid], sh[tid + s]);                                 \
        }                                                                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                             \
    }                                                                                \
    if (tid == 0u) {                                                                 \
        partials[3u * tgid] = s0[0];                                                 \
        partials[3u * tgid + 1u] = s1[0];                                            \
        partials[3u * tgid + 2u] = sh[0];                                            \
    }                                                                                \
}

HADAMARD_ROUND(hadamard_round_base, uint, ext_mul_base, kb_add)
HADAMARD_ROUND(hadamard_round_ext, uint4, ext_mul_kara, ext_add)

// out[i] = x + alpha * (y - x), over adjacent pairs, matching what slop does to
// an MLE's last variable with zero padding.
kernel void jagged_fold_base(device const uint* in [[buffer(0)]],
                             device uint4* out [[buffer(1)]],
                             constant uint4& alpha [[buffer(2)]],
                             constant uint& n [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    uint x = in[2u * gid];
    uint y = in[2u * gid + 1u];
    uint4 r = ext_mul_base(alpha, kb_sub(y, x));
    r.x = kb_add(r.x, x);
    out[gid] = r;
}

kernel void jagged_fold_ext(device const uint4* in [[buffer(0)]],
                            device uint4* out [[buffer(1)]],
                            constant uint4& alpha [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    uint4 x = in[2u * gid];
    uint4 y = in[2u * gid + 1u];
    out[gid] = ext_add(ext_mul_kara(alpha, ext_sub(y, x)), x);
}
"#;

struct Context {
    device: Device,
    queue: CommandQueue,
    round_base: ComputePipelineState,
    round_ext: ComputePipelineState,
    fold_base: ComputePipelineState,
    fold_ext: ComputePipelineState,
}

fn context() -> Option<&'static Context> {
    static CTX: OnceLock<Option<Context>> = OnceLock::new();
    CTX.get_or_init(|| {
        let device = Device::system_default()?;
        let header = include_str!("../../../sp1-metal-probes/src/shaders.metal")
            .split("// ---- end of shared field arithmetic ----")
            .next()?;
        let one = ((1u64 << 32) % 0x7f00_0001u64) as u32;
        let source = format!("{header}\nconstant uint KB_ONE = {one}u;\n{KERNELS}");
        let library =
            device.new_library_with_source(&source, &metal::CompileOptions::new()).ok()?;
        let pso = |name: &str| -> Option<ComputePipelineState> {
            let f = library.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&f).ok()
        };
        Some(Context {
            round_base: pso("hadamard_round_base")?,
            round_ext: pso("hadamard_round_ext")?,
            fold_base: pso("jagged_fold_base")?,
            fold_ext: pso("jagged_fold_ext")?,
            queue: device.new_command_queue(),
            device,
        })
    })
    .as_ref()
}

/// Whether the jagged sumcheck's round goes to the GPU, read once.
pub(crate) fn on_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_JAGGED").map(|v| v != "0").unwrap_or(true))
}

fn as_words<T>(xs: &[T]) -> &[u8] {
    // SAFETY: KoalaBear is repr(transparent) over its Montgomery word and its
    // degree-4 extension is repr(C) over four of them.
    unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), std::mem::size_of_val(xs)) }
}

/// A read-only buffer over a slice.
///
/// Unified memory means an aligned allocation can be taken exactly as it is, so
/// nothing is copied. Every caller here awaits its dispatch before returning,
/// which is what makes borrowing the slice sound.
fn read_only(device: &Device, words: &[u8]) -> Buffer {
    let start = std::time::Instant::now();
    if !words.is_empty() && words.as_ptr() as usize % PAGE == 0 && words.len() % PAGE == 0 {
        stats::record_input(Input::Wrapped, words.len(), 0);
        // SAFETY: borrowed only for the dispatch, which is awaited below.
        return unsafe {
            device.new_buffer_with_bytes_no_copy(
                words.as_ptr().cast(),
                words.len() as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };
    }
    let buffer = device.new_buffer_with_data(
        words.as_ptr().cast(),
        words.len().max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    stats::record_input(Input::Copied, words.len(), start.elapsed().as_nanos() as u64);
    buffer
}

/// The round's three sums: the products at even indices, at odd indices, and of
/// the pairwise sums.
fn round_sums<B, E>(base: &[B], ext: &[E], base_is_ext: bool) -> Option<([u32; 4], [u32; 4], [u32; 4])> {
    // The CPU zips the two arrays rather than requiring equal lengths, so the
    // shorter one bounds every sum.
    let n = base.len().min(ext.len());
    if !on_gpu() || n % 2 != 0 || n / 2 < MIN_PAIRS {
        return None;
    }
    let _phase = stats::enter(stats::Phase::Pcs);
    let pairs = n / 2;
    let ctx = context()?;
    let pso = if base_is_ext { &ctx.round_ext } else { &ctx.round_base };
    let b_base = read_only(&ctx.device, as_words(&base[..n]));
    let b_ext = read_only(&ctx.device, as_words(&ext[..n]));
    let groups = (pairs as u64).div_ceil(THREADS);
    let partials = ctx.device.new_buffer(groups * 3 * 16, MTLResourceOptions::StorageModeShared);
    let n32 = pairs as u32;
    metal::objc::rc::autoreleasepool(|| {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&b_base), 0);
        enc.set_buffer(1, Some(&b_ext), 0);
        enc.set_buffer(2, Some(&partials), 0);
        enc.set_bytes(3, 4, (&n32 as *const u32).cast());
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(THREADS, 1, 1));
        enc.end_encoding();
        let start = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let wall = start.elapsed().as_secs_f64();
        let (t0, t1): (f64, f64) =
            unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        stats::record_dispatch(wall, t1 - t0);
    });
    stats::record_output(Output::Staged, groups as usize * 3 * 16, 0);
    let raw = partials.contents().cast::<[u32; 4]>();
    let mut out = ([0u32; 4], [0u32; 4], [0u32; 4]);
    let add = |a: &mut [u32; 4], b: [u32; 4]| {
        for k in 0..4 {
            let s = a[k] as u64 + b[k] as u64;
            a[k] = (if s >= 0x7f00_0001 { s - 0x7f00_0001 } else { s }) as u32;
        }
    };
    for g in 0..groups as usize {
        // SAFETY: the kernel wrote three extension elements per threadgroup.
        unsafe {
            add(&mut out.0, *raw.add(3 * g));
            add(&mut out.1, *raw.add(3 * g + 1));
            add(&mut out.2, *raw.add(3 * g + 2));
        }
    }
    Some(out)
}

/// The three sums as field elements, for the fields the kernels are written for.
pub(crate) fn round_sums_typed<B: 'static, E: 'static + Copy>(
    base: &[B],
    ext: &[E],
) -> Option<(E, E, E)> {
    let base_is_ext = kind::<B>()?;
    if !matches!(kind::<E>(), Some(true)) {
        return None;
    }
    let (a, b, c) = round_sums(base, ext, base_is_ext)?;
    // SAFETY: E is the degree-4 extension, repr(C) over four KoalaBears, each
    // repr(transparent) over the Montgomery word the kernel wrote.
    Some(unsafe {
        (
            std::mem::transmute_copy(&a),
            std::mem::transmute_copy(&b),
            std::mem::transmute_copy(&c),
        )
    })
}

/// `Some(true)` for the extension, `Some(false)` for the base field, `None` for
/// anything else.
fn kind<T: 'static>() -> Option<bool> {
    use std::any::TypeId;
    type Kb = slop_koala_bear::KoalaBear;
    type Ext = slop_algebra::extension::BinomialExtensionField<Kb, 4>;
    if TypeId::of::<T>() == TypeId::of::<Ext>() {
        Some(true)
    } else if TypeId::of::<T>() == TypeId::of::<Kb>() {
        Some(false)
    } else {
        None
    }
}

/// Fixes an MLE's last variable, writing the result straight into the `Vec` that
/// will back it.
///
/// This is the memory-bound half, so it only pays if nothing is copied: the
/// input is borrowed where it lies and the output buffer borrows the pages of
/// the allocation being filled.
pub(crate) fn fold_typed<T: 'static, E: 'static + Copy + Send + Sync>(
    values: &[T],
    alpha: E,
) -> Option<Vec<E>> {
    let in_is_ext = kind::<T>()?;
    if !matches!(kind::<E>(), Some(true)) || !on_gpu() {
        return None;
    }
    let n = values.len();
    if n % 2 != 0 || n / 2 < MIN_PAIRS {
        return None;
    }
    let _phase = stats::enter(stats::Phase::Pcs);
    let out_len = n / 2;
    let ctx = context()?;
    let pso = if in_is_ext { &ctx.fold_ext } else { &ctx.fold_base };
    let mut out: Vec<E> = Vec::with_capacity(out_len);
    // SAFETY: the kernel writes every element before anything reads them.
    unsafe { out.set_len(out_len) };
    let out_bytes = out_len * std::mem::size_of::<E>();
    let direct = out.as_ptr() as usize % PAGE == 0 && out_bytes % PAGE == 0;
    let b_out = if direct {
        // SAFETY: borrowed only for the dispatch, which is awaited below.
        unsafe {
            ctx.device.new_buffer_with_bytes_no_copy(
                out.as_ptr().cast(),
                out_bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
    } else {
        ctx.device.new_buffer(out_bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
    };
    let b_in = read_only(&ctx.device, as_words(values));
    let alpha_words: [u32; 4] = unsafe { std::mem::transmute_copy(&alpha) };
    let n32 = out_len as u32;
    metal::objc::rc::autoreleasepool(|| {
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&b_in), 0);
        enc.set_buffer(1, Some(&b_out), 0);
        enc.set_bytes(2, 16, alpha_words.as_ptr().cast());
        enc.set_bytes(3, 4, (&n32 as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new((out_len as u64).div_ceil(THREADS), 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
        enc.end_encoding();
        let start = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let wall = start.elapsed().as_secs_f64();
        let (t0, t1): (f64, f64) =
            unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        stats::record_dispatch(wall, t1 - t0);
    });
    if direct {
        stats::record_output(Output::Direct, out_bytes, 0);
    } else {
        let start = std::time::Instant::now();
        // SAFETY: the kernel wrote `out_bytes` bytes of extension elements.
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_out.contents().cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                out_bytes,
            );
        }
        stats::record_output(Output::Staged, out_bytes, start.elapsed().as_nanos() as u64);
    }
    Some(out)
}
