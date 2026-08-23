//! Where the bytes go, split by phase and by whether the GPU had to be fed.
//!
//! The counters this replaces could say how much was uploaded but not what an
//! upload cost or what it was for, and they lived in the fork, so the jagged
//! crate's dispatches were invisible to them. Both matter now. The question
//! this stage of the work turns on is whether a value a kernel reads was
//! already on the device, and no existing number answers it.
//!
//! Unified memory makes "on the device" narrower than it sounds. Every
//! allocation in this process is physically reachable by the GPU; what varies
//! is what has to happen before a kernel can read one. Three things can:
//!
//! - nothing, because a buffer for those bytes already exists — either the
//!   producer allocated them in [`MetalBackend`](crate::backend::MetalBackend)
//!   or an earlier dispatch this round already wrapped them;
//! - a buffer is created over the pages where they lie, which is cheap but not
//!   free and pins the pages for the dispatch;
//! - the bytes are copied into a staging buffer, because the pointer or the
//!   length is not page-aligned and `newBufferWithBytesNoCopy` will refuse.
//!
//! Outputs divide the same way, and their third case is the one worth naming:
//! an output that stays on the device is an input the next kernel gets for
//! free, which is the entire thesis of the work these counters exist to judge.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The stage of proving a dispatch belongs to.
///
/// Coarse on purpose: these are the rows of the table the backend is judged
/// against, so that a residency figure can be read against the speedup for the
/// same phase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// The logup GKR rounds, folds and layer transitions.
    Gkr,
    /// The constraint sumcheck.
    Zerocheck,
    /// Merkle commitment of the traces.
    Commit,
    /// The jagged polynomial commitment's sumcheck and opening.
    Pcs,
    /// Anything dispatched outside a phase, which should be nothing.
    Other,
}

const PHASES: [Phase; 5] = [Phase::Gkr, Phase::Zerocheck, Phase::Commit, Phase::Pcs, Phase::Other];

impl Phase {
    fn index(self) -> usize {
        match self {
            Phase::Gkr => 0,
            Phase::Zerocheck => 1,
            Phase::Commit => 2,
            Phase::Pcs => 3,
            Phase::Other => 4,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Phase::Gkr => "gkr",
            Phase::Zerocheck => "zerocheck",
            Phase::Commit => "commit",
            Phase::Pcs => "pcs",
            Phase::Other => "other",
        }
    }
}

/// What had to happen before a kernel could read an input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Input {
    /// A buffer for these bytes already existed. The goal state.
    Resident,
    /// A no-copy buffer was created over host pages.
    Wrapped,
    /// The bytes were copied into a staging buffer.
    Copied,
}

/// Where a kernel's output went.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Output {
    /// It stays in device memory and the host never reads it.
    Resident,
    /// The kernel wrote through a no-copy buffer into its final host home.
    Direct,
    /// The kernel wrote into a staging buffer that was then copied to the host.
    Staged,
}

#[derive(Default)]
struct Counters {
    in_resident: AtomicU64,
    in_wrapped: AtomicU64,
    in_copied: AtomicU64,
    in_copy_ns: AtomicU64,
    out_resident: AtomicU64,
    out_direct: AtomicU64,
    out_staged: AtomicU64,
    out_copy_ns: AtomicU64,
    dispatches: AtomicU64,
    gpu_ns: AtomicU64,
    wall_ns: AtomicU64,
}

impl Counters {
    const fn new() -> Self {
        Self {
            in_resident: AtomicU64::new(0),
            in_wrapped: AtomicU64::new(0),
            in_copied: AtomicU64::new(0),
            in_copy_ns: AtomicU64::new(0),
            out_resident: AtomicU64::new(0),
            out_direct: AtomicU64::new(0),
            out_staged: AtomicU64::new(0),
            out_copy_ns: AtomicU64::new(0),
            dispatches: AtomicU64::new(0),
            gpu_ns: AtomicU64::new(0),
            wall_ns: AtomicU64::new(0),
        }
    }

    fn reset(&self) {
        for c in [
            &self.in_resident,
            &self.in_wrapped,
            &self.in_copied,
            &self.in_copy_ns,
            &self.out_resident,
            &self.out_direct,
            &self.out_staged,
            &self.out_copy_ns,
            &self.dispatches,
            &self.gpu_ns,
            &self.wall_ns,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

static COUNTERS: [Counters; 5] =
    [Counters::new(), Counters::new(), Counters::new(), Counters::new(), Counters::new()];

thread_local! {
    static CURRENT: Cell<Phase> = const { Cell::new(Phase::Other) };
}

/// Restores the enclosing phase when dropped.
///
/// Phases nest in practice — the commitment runs inside the shard prover, which
/// is inside no phase at all — so a guard restores rather than clears.
pub struct PhaseGuard {
    previous: Phase,
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.previous));
    }
}

/// Attributes everything dispatched on this thread until the guard drops.
#[must_use]
pub fn enter(phase: Phase) -> PhaseGuard {
    let previous = CURRENT.with(|c| c.replace(phase));
    PhaseGuard { previous }
}

fn current() -> &'static Counters {
    &COUNTERS[CURRENT.with(Cell::get).index()]
}

/// Records that a kernel read `bytes`, and what it cost to let it.
pub fn record_input(kind: Input, bytes: usize, nanos: u64) {
    let c = current();
    let counter = match kind {
        Input::Resident => &c.in_resident,
        Input::Wrapped => &c.in_wrapped,
        Input::Copied => &c.in_copied,
    };
    counter.fetch_add(bytes as u64, Ordering::Relaxed);
    if nanos > 0 {
        c.in_copy_ns.fetch_add(nanos, Ordering::Relaxed);
    }
}

/// Records that a kernel wrote `bytes`, and where they went.
pub fn record_output(kind: Output, bytes: usize, nanos: u64) {
    let c = current();
    let counter = match kind {
        Output::Resident => &c.out_resident,
        Output::Direct => &c.out_direct,
        Output::Staged => &c.out_staged,
    };
    counter.fetch_add(bytes as u64, Ordering::Relaxed);
    if nanos > 0 {
        c.out_copy_ns.fetch_add(nanos, Ordering::Relaxed);
    }
}

/// Every command buffer's execution interval, for [`device_busy_s`].
///
/// Summing the intervals answers a different question from taking their union,
/// and with four provers submitting at once only the union is the device's
/// utilisation. Bounded so that a pathological run cannot grow it without
/// limit; past the bound the sum is still exact and only the union goes stale.
static INTERVALS: Mutex<Vec<(f64, f64)>> = Mutex::new(Vec::new());

const MAX_INTERVALS: usize = 1 << 20;

/// Kernels compiled at run time, and what it cost.
///
/// The zerocheck generates a kernel per chip, so this is paid once per distinct
/// chip and then cached. It is counted separately from everything else because
/// Metal compiles in another process: the wait shows up as this process using
/// no CPU at all, which is invisible to a sampling profiler and looks like the
/// prover having nothing to do.
static COMPILES: AtomicU64 = AtomicU64::new(0);
static COMPILE_NS: AtomicU64 = AtomicU64::new(0);

/// Records one run-time kernel compilation.
pub fn record_compile(secs: f64) {
    COMPILES.fetch_add(1, Ordering::Relaxed);
    COMPILE_NS.fetch_add((secs.max(0.0) * 1e9) as u64, Ordering::Relaxed);
}

/// `(kernels compiled, thread-seconds spent waiting for the compiler)`.
#[must_use]
pub fn compiles() -> (u64, f64) {
    (COMPILES.load(Ordering::Relaxed), COMPILE_NS.load(Ordering::Relaxed) as f64 / 1e9)
}

/// Records one command buffer: how long the caller waited, and the interval
/// during which the hardware actually ran it.
pub fn record_dispatch(wall_s: f64, gpu_s: f64) {
    let c = current();
    c.dispatches.fetch_add(1, Ordering::Relaxed);
    c.wall_ns.fetch_add((wall_s.max(0.0) * 1e9) as u64, Ordering::Relaxed);
    if gpu_s.is_finite() && gpu_s > 0.0 {
        c.gpu_ns.fetch_add((gpu_s * 1e9) as u64, Ordering::Relaxed);
    }
}

/// Records when a command buffer ran, from the hardware's own clock.
///
/// Separate from [`record_dispatch`] because the union is device-wide: which
/// phase submitted the work does not change how busy the device was.
pub fn record_interval(start: f64, end: f64) {
    if !(start.is_finite() && end.is_finite()) || end <= start {
        return;
    }
    let mut iv = INTERVALS.lock().expect("intervals poisoned");
    if iv.len() < MAX_INTERVALS {
        iv.push((start, end));
    }
}

/// Seconds during which at least one command buffer was executing, and the
/// span from the first to the last.
///
/// The first number is what "the GPU was busy" should mean. Summing per-thread
/// GPU time instead counts a device shared by four provers four times over.
#[must_use]
pub fn device_busy_s() -> (f64, f64) {
    let mut iv = INTERVALS.lock().expect("intervals poisoned").clone();
    if iv.is_empty() {
        return (0.0, 0.0);
    }
    iv.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let span = iv.iter().map(|x| x.1).fold(f64::MIN, f64::max) - iv[0].0;
    let (mut busy, mut cur) = (0.0, iv[0]);
    for &(s, e) in &iv[1..] {
        if s > cur.1 {
            busy += cur.1 - cur.0;
            cur = (s, e);
        } else if e > cur.1 {
            cur.1 = e;
        }
    }
    busy += cur.1 - cur.0;
    (busy, span)
}

/// One phase's totals, in bytes and seconds.
#[derive(Clone, Copy, Default, Debug)]
pub struct PhaseTotals {
    pub in_resident: u64,
    pub in_wrapped: u64,
    pub in_copied: u64,
    pub in_copy_s: f64,
    pub out_resident: u64,
    pub out_direct: u64,
    pub out_staged: u64,
    pub out_copy_s: f64,
    pub dispatches: u64,
    pub gpu_s: f64,
    pub wall_s: f64,
}

impl PhaseTotals {
    /// Every byte a kernel read, however it got there.
    #[must_use]
    pub fn in_bytes(&self) -> u64 {
        self.in_resident + self.in_wrapped + self.in_copied
    }

    /// Every byte a kernel wrote.
    #[must_use]
    pub fn out_bytes(&self) -> u64 {
        self.out_resident + self.out_direct + self.out_staged
    }

    /// The fraction of read bytes that needed nothing done to them, which is
    /// the number this whole instrument exists to report.
    #[must_use]
    pub fn residency(&self) -> f64 {
        let total = self.in_bytes();
        if total == 0 {
            return 0.0;
        }
        self.in_resident as f64 / total as f64
    }

    fn add(&mut self, other: &PhaseTotals) {
        self.in_resident += other.in_resident;
        self.in_wrapped += other.in_wrapped;
        self.in_copied += other.in_copied;
        self.in_copy_s += other.in_copy_s;
        self.out_resident += other.out_resident;
        self.out_direct += other.out_direct;
        self.out_staged += other.out_staged;
        self.out_copy_s += other.out_copy_s;
        self.dispatches += other.dispatches;
        self.gpu_s += other.gpu_s;
        self.wall_s += other.wall_s;
    }
}

/// What one phase has spent so far.
#[must_use]
pub fn totals(phase: Phase) -> PhaseTotals {
    let c = &COUNTERS[phase.index()];
    let load = |a: &AtomicU64| a.load(Ordering::Relaxed);
    PhaseTotals {
        in_resident: load(&c.in_resident),
        in_wrapped: load(&c.in_wrapped),
        in_copied: load(&c.in_copied),
        in_copy_s: load(&c.in_copy_ns) as f64 / 1e9,
        out_resident: load(&c.out_resident),
        out_direct: load(&c.out_direct),
        out_staged: load(&c.out_staged),
        out_copy_s: load(&c.out_copy_ns) as f64 / 1e9,
        dispatches: load(&c.dispatches),
        gpu_s: load(&c.gpu_ns) as f64 / 1e9,
        wall_s: load(&c.wall_ns) as f64 / 1e9,
    }
}

/// Every phase summed.
#[must_use]
pub fn grand_totals() -> PhaseTotals {
    let mut all = PhaseTotals::default();
    for phase in PHASES {
        all.add(&totals(phase));
    }
    all
}

/// Clears every counter.
pub fn reset() {
    for c in &COUNTERS {
        c.reset();
    }
    INTERVALS.lock().expect("intervals poisoned").clear();
    COMPILES.store(0, Ordering::Relaxed);
    COMPILE_NS.store(0, Ordering::Relaxed);
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// Device busy, second by second, as a fraction of each second.
///
/// The total alone cannot distinguish a device that is a quarter busy
/// throughout from one that is saturated for a quarter of the proof and idle
/// for the rest, and those call for opposite work.
#[must_use]
pub fn busy_histogram(buckets: usize) -> Vec<f64> {
    let iv = INTERVALS.lock().expect("intervals poisoned").clone();
    if iv.is_empty() || buckets == 0 {
        return Vec::new();
    }
    let start = iv.iter().map(|x| x.0).fold(f64::MAX, f64::min);
    let end = iv.iter().map(|x| x.1).fold(f64::MIN, f64::max);
    let width = (end - start) / buckets as f64;
    if width <= 0.0 {
        return Vec::new();
    }
    // Overlapping command buffers are merged first, so a bucket cannot exceed
    // one: the question is whether the device was doing anything, not how many
    // threads were asking.
    let mut sorted = iv;
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (s, e) in sorted {
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    let mut out = vec![0.0; buckets];
    for (s, e) in merged {
        let (mut lo, hi) = (s, e);
        while lo < hi {
            let b = (((lo - start) / width) as usize).min(buckets - 1);
            let edge = start + (b + 1) as f64 * width;
            let take = hi.min(edge) - lo;
            out[b] += take / width;
            lo += take.max(width * 1e-9);
        }
    }
    out
}

/// A table of what each phase read, wrote and ran, for a profiling host to
/// print at the end of a proof.
///
/// The GPU column sums each command buffer's own execution interval across
/// every proving thread, so where shards prove concurrently it is an upper
/// bound on how busy the device was, not a measurement of it. Reading it as
/// device utilisation without dividing by the thread count is the mistake it
/// invites.
#[must_use]
pub fn report() -> String {
    let mut out = String::from(
        "phase       dispatch   gpu_s  wall_s |     in_GB  resident  wrapped   copied   copy_s | \
         out_GB  resident   direct   staged   copy_s\n",
    );
    let row = |name: &str, t: &PhaseTotals| {
        let share = |part: u64, whole: u64| {
            if whole == 0 {
                0.0
            } else {
                part as f64 / whole as f64 * 100.0
            }
        };
        let (i, o) = (t.in_bytes(), t.out_bytes());
        format!(
            "{name:<10} {:>8} {:>7.3} {:>7.3} | {:>9.2} {:>8.1}% {:>7.1}% {:>7.1}% {:>8.3} | \
             {:>6.2} {:>8.1}% {:>7.1}% {:>7.1}% {:>8.3}\n",
            t.dispatches,
            t.gpu_s,
            t.wall_s,
            gb(i),
            share(t.in_resident, i),
            share(t.in_wrapped, i),
            share(t.in_copied, i),
            t.in_copy_s,
            gb(o),
            share(t.out_resident, o),
            share(t.out_direct, o),
            share(t.out_staged, o),
            t.out_copy_s,
        )
    };
    for phase in PHASES {
        let t = totals(phase);
        if t.dispatches == 0 && t.in_bytes() == 0 {
            continue;
        }
        out.push_str(&row(phase.name(), &t));
    }
    out.push_str(&row("TOTAL", &grand_totals()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are process-global, so everything that touches them is one
    /// test: two would race on the same atomics from different threads.
    #[test]
    fn attributes_and_sums() {
        reset();
        {
            let _g = enter(Phase::Gkr);
            record_input(Input::Resident, 100, 0);
            record_input(Input::Copied, 300, 1_000);
            record_output(Output::Direct, 50, 0);
            record_dispatch(2e-3, 1e-3);
        }
        {
            let _g = enter(Phase::Pcs);
            record_input(Input::Wrapped, 400, 0);
        }
        let gkr = totals(Phase::Gkr);
        assert_eq!(gkr.in_bytes(), 400);
        assert_eq!(gkr.residency(), 0.25);
        assert_eq!(gkr.dispatches, 1);
        assert!((gkr.in_copy_s - 1e-6).abs() < 1e-12);
        assert_eq!(totals(Phase::Pcs).in_wrapped, 400);
        assert_eq!(grand_totals().in_bytes(), 800);
        assert_eq!(totals(Phase::Other).in_bytes(), 0);
        // A phase entered inside another restores the outer one, so a
        // commitment dispatched from within GKR does not reattribute the rest
        // of it.
        reset();
        let _outer = enter(Phase::Gkr);
        {
            let _inner = enter(Phase::Commit);
            record_input(Input::Resident, 10, 0);
        }
        record_input(Input::Resident, 20, 0);
        assert_eq!(totals(Phase::Commit).in_resident, 10);
        assert_eq!(totals(Phase::Gkr).in_resident, 20);
        reset();
        assert_eq!(grand_totals().in_bytes(), 0);
        // Overlapping intervals are one busy stretch, not two.
        record_interval(10.0, 12.0);
        record_interval(11.0, 13.0);
        record_interval(20.0, 20.5);
        let (busy, span) = device_busy_s();
        assert!((busy - 3.5).abs() < 1e-9, "busy was {busy}");
        assert!((span - 10.5).abs() < 1e-9, "span was {span}");
        reset();
        assert_eq!(device_busy_s(), (0.0, 0.0));
    }
}
