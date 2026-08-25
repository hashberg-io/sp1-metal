// SPDX-License-Identifier: MIT OR Apache-2.0
// Added to slop-jagged 6.4.0 by the sp1-metal project; see CHANGES.md.
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

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use sp1_metal_probes::stats::{self, Input, Output};

const THREADS: u64 = 256;

/// One counted step of building the jagged sumcheck polynomial.
#[derive(Default)]
pub struct Step {
    calls: std::sync::atomic::AtomicU64,
    cells: std::sync::atomic::AtomicU64,
    nanos: std::sync::atomic::AtomicU64,
}

impl Step {
    const fn new() -> Self {
        Self {
            calls: std::sync::atomic::AtomicU64::new(0),
            cells: std::sync::atomic::AtomicU64::new(0),
            nanos: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn record(&self, cells: usize, nanos: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.calls.fetch_add(1, Relaxed);
        self.cells.fetch_add(cells as u64, Relaxed);
        self.nanos.fetch_add(nanos, Relaxed);
    }
}

/// `[the jagged multilinear itself, restacking the base MLEs against it]`.
///
/// `create jagged sumcheck poly` is 2.7 thread-seconds and half of it is on the
/// GPU already, so the question is which of its two halves the rest is.
pub static POLY: [Step; 2] = [Step::new(), Step::new()];

/// The names of the steps.
pub const POLY_NAMES: [&str; 2] = ["jagged multilinear", "restack"];

/// What building the jagged sumcheck polynomial cost.
#[must_use]
pub fn poly_report() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out = String::from("\njagged poly           calls          cells   thread_s\n");
    for (name, st) in POLY_NAMES.iter().zip(POLY.iter()) {
        out.push_str(&format!(
            "{name:<18} {:>10} {:>14} {:>10.3}\n",
            st.calls.load(Relaxed),
            st.cells.load(Relaxed),
            st.nanos.load(Relaxed) as f64 / 1e9
        ));
    }
    out
}
/// Below this a dispatch costs more than the arithmetic saves.
const MIN_PAIRS: usize = 1 << 12;

const KERNELS: &str = r#"
// The jagged assist sumcheck's round: for each column, evaluate the branching
// program twice -- at lambda = 0 and lambda = 1/2 -- and weight the results.
//
// The program is a four-state dynamic program over `num_vars + 1` layers. Each
// layer reads four bits (a row bit, an index bit, and one from each of two
// prefix sums), forms the sixteen-way eq over them, and folds the state vector
// through a fixed transition table. That table is 4x16 entries computed on the
// host by calling upstream's own transition function, rather than restated
// here, so the rule cannot drift between the two implementations.
struct BpParams {
    uint n_elems;
    uint mps_dim;
    uint round_num;
    uint num_vars;
    uint zrow_dim;
    uint zidx_dim;
};

// The eq weights of a four-variable point, in the order `partial_lagrange`
// produces them: coordinate zero is the most significant bit of the index.
inline void bp_eq16(uint4 p0, uint4 p1, uint4 p2, uint4 p3, thread uint4* evals) {
    uint4 one = uint4(KB_ONE, 0u, 0u, 0u);
    uint4 coords[4] = { p0, p1, p2, p3 };
    evals[0] = one;
    uint n = 1u;
    for (uint c = 0u; c < 4u; ++c) {
        // Backwards, so that an entry is read before the slot it expands into
        // is overwritten.
        for (int i = int(n) - 1; i >= 0; --i) {
            uint4 v = evals[i];
            uint4 prod = ext_mul(v, coords[c]);
            evals[2 * i] = ext_sub(v, prod);
            evals[2 * i + 1] = prod;
        }
        n <<= 1;
    }
}

// The layer-th least significant coordinate of a point, or zero past its end.
inline uint4 bp_at(device const uint4* pt, uint dim, uint layer) {
    return layer < dim ? pt[dim - layer - 1u] : uint4(0u);
}

kernel void jagged_assist(device const uint* mps [[buffer(0)]],
                          device const uint4* z_col_eq [[buffer(1)]],
                          device const uint4* inter_eq [[buffer(2)]],
                          device const uint4* rhos [[buffer(3)]],
                          device const uint4* zrow [[buffer(4)]],
                          device const uint4* zidx [[buffer(5)]],
                          device const uint* trans [[buffer(6)]],
                          device uint4* out [[buffer(7)]],
                          constant BpParams& p [[buffer(8)]],
                          constant uint4& half_ef [[buffer(9)]],
                          uint gid [[thread_position_in_grid]],
                          uint grid [[threads_per_grid]]) {
    uint one_w = KB_ONE;
    uint4 one = uint4(one_w, 0u, 0u, 0u);
    uint split = p.mps_dim - p.round_num - 1u;
    uint dim_l = p.mps_dim / 2u;
    uint dim_r = p.mps_dim - dim_l;

    for (uint e = gid; e < p.n_elems; e += grid) {
        device const uint* row = mps + e * p.mps_dim;
        uint4 zc = z_col_eq[e];
        uint4 ie = inter_eq[e];
        // eq(x_i, (x, rho)) loses one term to the variable being fixed: at
        // lambda = 0 it is 1 - x, and at lambda = 1/2 it is 1/2.
        uint4 eq_val0 = ext_sub(one, uint4(row[split], 0u, 0u, 0u));

        for (uint which = 0u; which < 2u; ++which) {
            uint4 lambda = which == 0u ? uint4(0u) : half_ef;
            uint4 state[4] = { uint4(0u), uint4(0u), uint4(0u), uint4(0u) };
            // Success is carry = false, comparison_so_far = true, which the
            // host's index rule puts at 2.
            state[2] = one;

            for (int layer = int(p.num_vars); layer >= 0; --layer) {
                uint l = uint(layer);
                uint4 zr = bp_at(zrow, p.zrow_dim, l);
                uint4 zi = bp_at(zidx, p.zidx_dim, l);
                // The left half of the h point, then the right half; both are
                // the merged prefix sum with lambda spliced in at `split` and
                // the sumcheck randomness appended.
                uint4 a = uint4(0u);
                if (l < dim_l) {
                    uint k = dim_l - l - 1u;
                    a = k < split ? uint4(row[k], 0u, 0u, 0u)
                                  : (k == split ? lambda : rhos[k - split - 1u]);
                }
                uint4 b = uint4(0u);
                if (l < dim_r) {
                    uint k = dim_l + dim_r - l - 1u;
                    b = k < split ? uint4(row[k], 0u, 0u, 0u)
                                  : (k == split ? lambda : rhos[k - split - 1u]);
                }

                uint4 eq[16];
                bp_eq16(zr, zi, a, b, eq);

                uint4 next[4];
                for (uint ms = 0u; ms < 4u; ++ms) {
                    uint4 acc[4] = { uint4(0u), uint4(0u), uint4(0u), uint4(0u) };
                    for (uint i = 0u; i < 16u; ++i) {
                        uint o = trans[ms * 16u + i];
                        if (o < 4u) {
                            acc[o] = ext_add(acc[o], eq[i]);
                        }
                    }
                    uint4 sum = uint4(0u);
                    for (uint k = 0u; k < 4u; ++k) {
                        sum = ext_add(sum, ext_mul(acc[k], state[k]));
                    }
                    next[ms] = sum;
                }
                for (uint k = 0u; k < 4u; ++k) {
                    state[k] = next[k];
                }
            }

            // The initial state is carry = false, comparison_so_far = false.
            uint4 h_eval = state[0];
            uint4 eq_eval = ext_mul(ie, which == 0u ? eq_val0 : half_ef);
            out[2u * e + which] = ext_mul(ext_mul(zc, h_eval), eq_eval);
        }
    }
}

// The jagged little polynomial's guts: one entry per cell of the stacked
// trace area. A cell belongs to the column whose prefix-sum range contains
// it, and its value is that column's eq weight times the row's -- except in
// the final range, which is padding and evaluates to zero.
//
// The ranges are found by binary search rather than by walking, because a
// thread here starts at an arbitrary index. Duplicate prefix sums mark empty
// columns, and taking the last range that starts at or before the index skips
// them, which is what the sequential walk does by advancing past a column of
// size zero.
kernel void jagged_values(device const uint4* col_eq [[buffer(0)]],
                          device const uint4* row_eq [[buffer(1)]],
                          device const uint* prefix [[buffer(2)]],
                          device uint4* out [[buffer(3)]],
                          constant uint& n_ranges [[buffer(4)]],
                          constant uint& total [[buffer(5)]],
                          uint gid [[thread_position_in_grid]],
                          uint grid [[threads_per_grid]]) {
    for (uint idx = gid; idx < total; idx += grid) {
        uint lo = 0u;
        uint hi = n_ranges - 1u;
        while (lo < hi) {
            uint mid = (lo + hi + 1u) >> 1;
            if (prefix[mid] <= idx) { lo = mid; } else { hi = mid - 1u; }
        }
        if (lo + 1u >= n_ranges) {
            out[idx] = uint4(0u);
        } else {
            out[idx] = ext_mul(col_eq[lo], row_eq[idx - prefix[lo]]);
        }
    }
}

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
    round_base: ComputePipelineState,
    round_ext: ComputePipelineState,
    fold_base: ComputePipelineState,
    fold_ext: ComputePipelineState,
    values: ComputePipelineState,
    assist: ComputePipelineState,
}

impl Context {
    /// This thread's command queue; see [`sp1_metal_probes::nocopy::thread_queue`].
    fn queue(&self) -> CommandQueue {
        sp1_metal_probes::nocopy::thread_queue(&self.device)
    }
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
        // One compile at a time, process-wide: creating a pipeline while
        // another thread is doing the same deadlocks in the driver. See
        // `sp1_metal_core::compile`.
        let _compiling = sp1_metal_core::compile::lock();
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
            values: pso("jagged_values")?,
            assist: pso("jagged_assist")?,
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

/// Splits a dispatch's cost into what the device did and what the host waited.
///
/// The `pcs` phase blocks 13.5 ms a dispatch at six shards against 5.8 at four,
/// and an aggregate over four very different sites cannot say whether that is a
/// kernel getting bigger or a queue getting longer.
fn record_site(name: &'static str, (wall, gpu): (f64, f64)) {
    sp1_metal_core::tally::record(name, 0, (gpu * 1e9) as u64);
    sp1_metal_core::tally::record(
        match name {
            "jagged: round" => "jagged: round   wait",
            "jagged: fold" => "jagged: fold    wait",
            "jagged: values" => "jagged: values  wait",
            _ => "jagged: assist  wait",
        },
        0,
        ((wall - gpu).max(0.0) * 1e9) as u64,
    );
}

fn as_words<T>(xs: &[T]) -> &[u8] {
    // SAFETY: KoalaBear is repr(transparent) over its Montgomery word and its
    // degree-4 extension is repr(C) over four of them.
    unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u8>(), std::mem::size_of_val(xs)) }
}

/// A read-only buffer over a slice, kept for the rest of this sumcheck round.
///
/// Giving the device access to a borrowed allocation is not free, and until now
/// nobody had measured what it costs. Dispatching the fold twice over the same
/// buffers answers it: identical kernel, identical 0.027 seconds of GPU time,
/// and 0.134 seconds of waiting the first time against 0.011 the second. Nine
/// tenths of what a fold "blocks" for is the first use of its buffers, paid
/// between commit and the kernel starting.
///
/// A round reduces its two arrays and then folds the same two, so each one is
/// mapped twice for no reason. This keeps the first mapping until the fold has
/// used it, which is the whole of the array's life in this round.
///
/// Keyed on the pointer, which is sound *only* because of that lifetime: the
/// slice is borrowed by the reduction that inserted the entry and by the fold
/// that takes it, so it cannot be freed and its address cannot be handed to
/// something else in between. Entries do not outlive the round -- see
/// [`clear_round_buffers`].
fn round_buffer(device: &Device, words: &[u8]) -> Buffer {
    if !sp1_metal_core::handoff::reusing() {
        return read_only(device, words);
    }
    let key = (words.as_ptr() as usize, words.len());
    if let Some(hit) = ROUND_BUFFERS.with(|c| {
        c.borrow().iter().find(|(k, _)| *k == key).map(|(_, b)| b.clone())
    }) {
        stats::record_input(Input::Wrapped, words.len(), 0);
        return hit;
    }
    let buffer = read_only(device, words);
    ROUND_BUFFERS.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() < 4 {
            c.push((key, buffer.clone()));
        }
    });
    buffer
}

thread_local! {
    /// The buffers of the round in flight on this thread. At most the two
    /// arrays a round reduces, and they are dropped as soon as it folds them.
    static ROUND_BUFFERS: std::cell::RefCell<Vec<((usize, usize), Buffer)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Drops the round's buffers. Called by the fold, which is the last thing that
/// reads the arrays the reduction mapped.
fn clear_round_buffers() {
    ROUND_BUFFERS.with(|c| c.borrow_mut().clear());
}

/// A read-only buffer over a slice.
///
/// Unified memory means an aligned allocation can be taken exactly as it is, so
/// nothing is copied. Every caller here awaits its dispatch before returning,
/// which is what makes borrowing the slice sound.
fn read_only(device: &Device, words: &[u8]) -> Buffer {
    let start = std::time::Instant::now();
    if !words.is_empty() && sp1_metal_probes::nocopy::takeable(words.as_ptr()) {
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
    // Kept until this round's fold has used them; see `round_buffer`.
    let b_base = round_buffer(&ctx.device, as_words(&base[..n]));
    let b_ext = round_buffer(&ctx.device, as_words(&ext[..n]));
    let groups = (pairs as u64).div_ceil(THREADS);
    let partials = ctx.device.new_buffer(groups * 3 * 16, MTLResourceOptions::StorageModeShared);
    let n32 = pairs as u32;
    metal::objc::rc::autoreleasepool(|| {
        let queue = ctx.queue();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&b_base), 0);
        enc.set_buffer(1, Some(&b_ext), 0);
        enc.set_buffer(2, Some(&partials), 0);
        enc.set_bytes(3, 4, (&n32 as *const u32).cast());
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(THREADS, 1, 1));
        enc.end_encoding();
        // Timed per site: the `pcs` phase blocks 13.5 ms a dispatch at six
        // shards against 5.8 at four, and the aggregate cannot say which.
        record_site("jagged: round", sp1_metal_core::queue::submit(cb));
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
    let direct = sp1_metal_probes::nocopy::takeable(out.as_ptr().cast());
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
    let b_in = round_buffer(&ctx.device, as_words(values));
    // The round's reduction mapped this same array a moment ago and the fold is
    // the last thing that reads it, so the entry goes now.
    clear_round_buffers();
    let alpha_words: [u32; 4] = unsafe { std::mem::transmute_copy(&alpha) };
    let n32 = out_len as u32;
    metal::objc::rc::autoreleasepool(|| {
        let queue = ctx.queue();
        let cb = queue.new_command_buffer();
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
        // Timed per site: the `pcs` phase blocks 13.5 ms a dispatch at six
        // shards against 5.8 at four, and the aggregate cannot say which.
        record_site("jagged: fold", sp1_metal_core::queue::submit(cb));
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

/// Whether the jagged little polynomial's guts are filled on the GPU,
/// overridable with `SP1_METAL_JAGGED_VALUES`.
fn values_on_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_JAGGED_VALUES").map(|v| v != "0").unwrap_or(true))
}

/// Whether to fill the guts both ways and compare, with
/// `SP1_METAL_JAGGED_VALUES=check`.
///
/// The interesting cases here are the boundaries -- an empty column, which
/// shows up as a repeated prefix sum, and the final range, which is padding --
/// and a real proof's parameters exercise them where an invented one might not.
#[must_use]
pub(crate) fn values_check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_JAGGED_VALUES").is_ok_and(|v| v == "check"))
}

/// Fills the jagged little polynomial's guts, one entry per cell of the
/// stacked area.
///
/// One extension multiply per sixteen bytes written, which is the memory-bound
/// case: it only pays with nothing copied at either end, and both ends here are
/// buffers borrowing the caller's pages.
///
/// Returns `false` when the field is not the one the kernels are written for,
/// when the area is too small to be worth a dispatch, or when the output would
/// not fit in a Metal buffer -- leaving the caller to fill it as it always did.
pub(crate) fn jagged_values_typed<K: 'static + Copy>(
    col_eq: &[K],
    row_eq: &[K],
    prefix_sums: &[usize],
    out: &mut [K],
) -> bool {
    if (!values_on_gpu() && !values_check()) || !matches!(kind::<K>(), Some(true)) {
        return false;
    }
    let total = out.len();
    if total < MIN_PAIRS || prefix_sums.len() < 2 || total > u32::MAX as usize {
        return false;
    }
    let Some(ctx) = context() else {
        return false;
    };
    let out_bytes = std::mem::size_of_val(out);
    if out_bytes as u64 > ctx.device.max_buffer_length() {
        return false;
    }
    // Every non-final range indexes `row_eq` by the offset within it, and
    // `col_eq` by its own index; refuse rather than let the kernel read past
    // either.
    let widest = prefix_sums.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    if col_eq.len() + 1 < prefix_sums.len() || row_eq.len() < widest {
        return false;
    }
    let prefix32: Vec<u32> = prefix_sums.iter().map(|&x| x as u32).collect();
    if prefix_sums.iter().any(|&x| x > u32::MAX as usize) {
        return false;
    }

    let _phase = stats::enter(stats::Phase::Pcs);
    let queue = ctx.queue();
    metal::objc::rc::autoreleasepool(|| {
        // SAFETY: each buffer borrows a slice that outlives the one dispatch
        // below, which is awaited before this returns.
        let b_col = unsafe { sp1_metal_probes::nocopy::read_only(&ctx.device, as_words(col_eq)) };
        let b_row = unsafe { sp1_metal_probes::nocopy::read_only(&ctx.device, as_words(row_eq)) };
        let b_pre = unsafe { sp1_metal_probes::nocopy::read_only(&ctx.device, as_words(&prefix32)) };
        // SAFETY: as above, and nothing reads `out` until the wait below.
        let (b_out, direct) = unsafe {
            sp1_metal_probes::nocopy::write_into(
                &ctx.device,
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out_bytes),
            )
        };
        let n_ranges = prefix32.len() as u32;
        let total32 = total as u32;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.values);
        enc.set_buffer(0, Some(&b_col), 0);
        enc.set_buffer(1, Some(&b_row), 0);
        enc.set_buffer(2, Some(&b_pre), 0);
        enc.set_buffer(3, Some(&b_out), 0);
        enc.set_bytes(4, 4, (&n_ranges as *const u32).cast());
        enc.set_bytes(5, 4, (&total32 as *const u32).cast());
        let groups = (total as u64).div_ceil(THREADS).min(4096).max(1);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(THREADS, 1, 1));
        enc.end_encoding();
        // Timed per site: the `pcs` phase blocks 13.5 ms a dispatch at six
        // shards against 5.8 at four, and the aggregate cannot say which.
        record_site("jagged: values", sp1_metal_core::queue::submit(cb));
        if direct {
            return;
        }
        let start = std::time::Instant::now();
        // SAFETY: the kernel wrote exactly this many bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_out.contents().cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                out_bytes,
            );
        }
        stats::record_output(Output::Staged, out_bytes, start.elapsed().as_nanos() as u64);
    });
    true
}

/// Whether the jagged assist sumcheck's round runs on the GPU, overridable
/// with `SP1_METAL_JAGGED_ASSIST`.
fn assist_on_gpu() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_JAGGED_ASSIST").map(|v| v != "0").unwrap_or(true))
}

/// Whether to run the round both ways and compare, with
/// `SP1_METAL_JAGGED_ASSIST=check`.
#[must_use]
pub(crate) fn assist_check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_JAGGED_ASSIST").is_ok_and(|v| v == "check"))
}

/// Evaluates the jagged assist's branching program for every column, at
/// `lambda = 0` and at `lambda = 1/2`.
///
/// `merged_prefix_sums` is the flattened points, `mps_dim` words each; the
/// other per-column slices are parallel to it. Writes `2 * n` extension
/// elements: the value at zero and at a half, interleaved.
///
/// Returns `false` when the field is not the one the kernels are written for or
/// the work is too small to be worth a dispatch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jagged_assist_typed<F: 'static + Copy, E: 'static + Copy>(
    merged_prefix_sums: &[F],
    mps_dim: usize,
    z_col_eq_vals: &[E],
    intermediate_eq_full_evals: &[E],
    rhos: &[E],
    z_row: &[E],
    z_index: &[E],
    transition: &[u32],
    round_num: usize,
    num_vars: usize,
    half: E,
    out: &mut [E],
) -> bool {
    if (!assist_on_gpu() && !assist_check())
        || !matches!(kind::<F>(), Some(false))
        || !matches!(kind::<E>(), Some(true))
    {
        return false;
    }
    let n = z_col_eq_vals.len();
    // Below this the dispatch costs more than the branching program saves; the
    // work per column is `num_vars` layers of a fixed sixteen-way fold.
    if n == 0 || mps_dim == 0 || n * num_vars < 1 << 12 {
        return false;
    }
    if round_num + 1 > mps_dim
        || intermediate_eq_full_evals.len() < n
        || merged_prefix_sums.len() < n * mps_dim
        || rhos.len() < round_num
        || transition.len() != 64
        || out.len() < 2 * n
    {
        return false;
    }
    let Some(ctx) = context() else {
        return false;
    };

    let _phase = stats::enter(stats::Phase::Pcs);
    let queue = ctx.queue();
    metal::objc::rc::autoreleasepool(|| {
        // SAFETY: every buffer borrows a slice that outlives the one dispatch,
        // which is awaited before this returns, and nothing reads `out` until
        // then.
        let up = |b: &[u8]| unsafe { sp1_metal_probes::nocopy::read_only(&ctx.device, b) };
        let pad = [0u32; 4];
        let b_mps = up(as_words(&merged_prefix_sums[..n * mps_dim]));
        let b_zc = up(as_words(z_col_eq_vals));
        let b_ie = up(as_words(intermediate_eq_full_evals));
        let b_rho = up(if rhos.is_empty() { as_words(&pad) } else { as_words(rhos) });
        let b_zr = up(if z_row.is_empty() { as_words(&pad) } else { as_words(z_row) });
        let b_zi = up(if z_index.is_empty() { as_words(&pad) } else { as_words(z_index) });
        let b_tr = up(as_words(transition));
        let out_bytes = 2 * n * std::mem::size_of::<E>();
        let (b_out, direct) = unsafe {
            sp1_metal_probes::nocopy::write_into(
                &ctx.device,
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out_bytes),
            )
        };

        #[repr(C)]
        struct BpParams {
            n_elems: u32,
            mps_dim: u32,
            round_num: u32,
            num_vars: u32,
            zrow_dim: u32,
            zidx_dim: u32,
        }
        let params = BpParams {
            n_elems: n as u32,
            mps_dim: mps_dim as u32,
            round_num: round_num as u32,
            num_vars: num_vars as u32,
            zrow_dim: z_row.len() as u32,
            zidx_dim: z_index.len() as u32,
        };
        let half_words: [u32; 4] = unsafe { std::mem::transmute_copy(&half) };

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.assist);
        for (i, b) in
            [&b_mps, &b_zc, &b_ie, &b_rho, &b_zr, &b_zi, &b_tr, &b_out].into_iter().enumerate()
        {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_bytes(8, std::mem::size_of::<BpParams>() as u64, (&params as *const BpParams).cast());
        enc.set_bytes(9, 16, half_words.as_ptr().cast());
        let groups = (n as u64).div_ceil(THREADS).min(2048).max(1);
        enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(THREADS, 1, 1));
        enc.end_encoding();
        // Timed per site: the `pcs` phase blocks 13.5 ms a dispatch at six
        // shards against 5.8 at four, and the aggregate cannot say which.
        record_site("jagged: assist", sp1_metal_core::queue::submit(cb));
        if direct {
            return;
        }
        let start = std::time::Instant::now();
        // SAFETY: the kernel wrote exactly this many bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                b_out.contents().cast::<u8>(),
                out.as_mut_ptr().cast::<u8>(),
                out_bytes,
            );
        }
        stats::record_output(Output::Staged, out_bytes, start.elapsed().as_nanos() as u64);
    });
    true
}
