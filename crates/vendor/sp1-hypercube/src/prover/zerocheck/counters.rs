//! Where zerocheck's time goes, at the granularity that separates its parts.
//!
//! `zerocheck` is the largest span in the proof at 8.5 thread-seconds and the
//! Metal region inside it accounts for 3.9. Nearly five thread-seconds are
//! somewhere else under that span and no existing number says where. Sampling
//! would blame the sumcheck driver for all of it, since everything here is
//! called from inside `reduce_sumcheck_to_evaluation`.
//!
//! So this counts instead. Four buckets partition the span: the per-chip setup
//! before the sumcheck starts, the round reduction, the fold, and whatever the
//! driver does between them -- which is what is left when the other three are
//! taken off the whole.
//!
//! The rule these exist to serve, twice paid for now: **when sampling cannot
//! separate two call sites, count at the sites rather than profiling harder.**

use std::sync::atomic::{AtomicU64, Ordering};

/// One bucket: how many times, over how many cells, for how long.
#[derive(Default)]
pub struct Bucket {
    calls: AtomicU64,
    cells: AtomicU64,
    nanos: AtomicU64,
}

impl Bucket {
    const fn new() -> Self {
        Self { calls: AtomicU64::new(0), cells: AtomicU64::new(0), nanos: AtomicU64::new(0) }
    }

    pub fn record(&self, cells: usize, nanos: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.cells.fetch_add(cells as u64, Ordering::Relaxed);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// `(calls, cells, thread-seconds)`.
    #[must_use]
    pub fn counts(&self) -> (u64, u64, f64) {
        (
            self.calls.load(Ordering::Relaxed),
            self.cells.load(Ordering::Relaxed),
            self.nanos.load(Ordering::Relaxed) as f64 / 1e9,
        )
    }
}

/// The per-chip loop that builds the `ZeroCheckPoly`s, before any round runs.
pub static SETUP: Bucket = Bucket::new();
/// The whole sumcheck, driver included. The three below are inside it.
pub static SUMCHECK: Bucket = Bucket::new();
/// The round reduction, first round and later rounds.
pub static SUM_AS_POLY_FIRST: Bucket = Bucket::new();
pub static SUM_AS_POLY: Bucket = Bucket::new();
/// The fold, first round and later rounds.
pub static FIX_FIRST: Bucket = Bucket::new();
pub static FIX: Bucket = Bucket::new();
/// Assembling the opened values once the sumcheck has finished.
pub static OPENINGS: Bucket = Bucket::new();

/// What each part of zerocheck cost, as a table.
///
/// `driver` is not measured directly: it is the sumcheck total minus the four
/// things called from inside it, which is the only way to see time that belongs
/// to no call of ours.
#[must_use]
pub fn report() -> String {
    let mut out =
        String::from("\nzerocheck            calls          cells   thread_s\n");
    let row = |out: &mut String, name: &str, b: &Bucket| {
        let (calls, cells, secs) = b.counts();
        out.push_str(&format!("{name:<15} {calls:>10} {cells:>14} {secs:>10.3}\n"));
        secs
    };
    let setup = row(&mut out, "setup", &SETUP);
    let sumcheck = row(&mut out, "sumcheck", &SUMCHECK);
    let sap0 = row(&mut out, "  round first", &SUM_AS_POLY_FIRST);
    let sap = row(&mut out, "  round", &SUM_AS_POLY);
    let fix0 = row(&mut out, "  fold first", &FIX_FIRST);
    let fix = row(&mut out, "  fold", &FIX);
    let openings = row(&mut out, "openings", &OPENINGS);
    let driver = sumcheck - sap0 - sap - fix0 - fix;
    out.push_str(&format!("{:<15} {:>10} {:>14} {driver:>10.3}\n", "  driver", "", ""));
    out.push_str(&format!(
        "{:<15} {:>10} {:>14} {:>10.3}\n",
        "TOTAL",
        "",
        "",
        setup + sumcheck + openings
    ));
    out
}

/// Why a round reduction did not reach the GPU, and how much it was worth.
///
/// The round reduction is 92% of zerocheck, and the counters above showed 2,376
/// round calls against 1,258 dispatches -- so roughly half of them run on the
/// CPU and the aggregate could not say which of `sum_as_poly`'s seven refusals
/// was responsible. Each now has its own tally, weighted by the cells it turned
/// away, because a refusal that fires often on empty chips and one that fires
/// rarely on the widest chip in the shard look identical in a count.
pub static REJECT: [Bucket; 7] = [
    Bucket::new(),
    Bucket::new(),
    Bucket::new(),
    Bucket::new(),
    Bucket::new(),
    Bucket::new(),
    Bucket::new(),
];

/// The names of the refusals, in the order `sum_as_poly` tests them.
pub const REJECT_NAMES: [&str; 7] = [
    "gpu off",
    "no main rows",
    "empty chip",
    "no preprocessed",
    "short gkr powers",
    "short eq",
    "no pipeline",
];

/// Rounds that did reach the GPU, for the denominator.
pub static ACCEPTED: Bucket = Bucket::new();

/// Building the eq weights a round is summed against.
///
/// Upstream computes these per chip per round and leaves a TODO saying it could
/// be once per sumcheck. Every chip in a shard shares the same point and the
/// same number of variables, so the table is identical across the chips of a
/// round -- about a hundred of them.
pub static LAGRANGE: Bucket = Bucket::new();
pub static LAGRANGE_HIT: Bucket = Bucket::new();

/// Where the round reductions went.
#[must_use]
pub fn reject_report() -> String {
    let mut out = String::from("\nround reduction       calls          cells   thread_s\n");
    let (calls, cells, secs) = ACCEPTED.counts();
    out.push_str(&format!("{:<18} {calls:>10} {cells:>14} {secs:>10.3}\n", "on the GPU"));
    for (name, b) in REJECT_NAMES.iter().zip(REJECT.iter()) {
        let (calls, cells, secs) = b.counts();
        if calls > 0 {
            out.push_str(&format!("{name:<18} {calls:>10} {cells:>14} {secs:>10.3}\n"));
        }
    }
    for (name, b) in [("eq built", &LAGRANGE), ("eq reused", &LAGRANGE_HIT)] {
        let (calls, cells, secs) = b.counts();
        out.push_str(&format!("{name:<18} {calls:>10} {cells:>14} {secs:>10.3}\n"));
    }
    out
}
