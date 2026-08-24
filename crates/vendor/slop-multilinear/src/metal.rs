// SPDX-License-Identifier: MIT OR Apache-2.0
// Added to slop-multilinear 6.4.0 by the sp1-metal project; see CHANGES.md.
//! Who is folding, and how much of it there is.
//!
//! The largest remaining block of CPU time in a proof — 28.5 thread-seconds of
//! it, measured — is folding MLEs, and two attempts to attribute it by sampling
//! failed for the same reason: rayon owns the stack for most of it, so the
//! profiler sees `rayon::join` rather than whoever asked. The prover has five
//! or so call sites and no way to tell them apart from outside.
//!
//! It has one from inside. Every fold in the prover funnels through
//! [`mle_fix_last_variable`](crate::mle_fix_last_variable): `Mle`,
//! `PaddedMle`, and `VirtualGeq` all call it, and the constant-padding variant
//! is a wrapper around it. A counter there sees all of them, and the shape of
//! each call — how many rows, how many polynomials, and whether it is widening
//! a base field to an extension — is enough to say which of the five it was
//! without a stack at all.
//!
//! The counters are always on when this crate is built with `metal`; they cost
//! two atomic adds against a call that touches megabytes.

use std::sync::atomic::{AtomicU64, Ordering};

/// One bucket of folds: the calls of a given shape and what they cost.
#[derive(Default)]
pub struct FoldCounter {
    calls: AtomicU64,
    rows: AtomicU64,
    cells: AtomicU64,
    nanos: AtomicU64,
}

impl FoldCounter {
    const fn new() -> Self {
        Self {
            calls: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            cells: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
        }
    }

    pub fn record(&self, rows: usize, polys: usize, nanos: u64) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.rows.fetch_add(rows as u64, Ordering::Relaxed);
        self.cells.fetch_add((rows * polys) as u64, Ordering::Relaxed);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// `(calls, rows, cells, thread-seconds)`.
    #[must_use]
    pub fn counts(&self) -> (u64, u64, u64, f64) {
        (
            self.calls.load(Ordering::Relaxed),
            self.rows.load(Ordering::Relaxed),
            self.cells.load(Ordering::Relaxed),
            self.nanos.load(Ordering::Relaxed) as f64 / 1e9,
        )
    }
}

/// Folds bucketed by width, which is what separates the call sites.
///
/// A single polynomial is basefold and the jagged sumcheck; a handful of them
/// is the GKR circuit's numerators and denominators; a wide batch is a chip's
/// whole trace under zerocheck. The boundaries are round numbers rather than
/// measured ones — the point is to tell the three apart, not to rank within
/// them.
pub static ONE_POLY: FoldCounter = FoldCounter::new();
pub static FEW_POLYS: FoldCounter = FoldCounter::new();
pub static MANY_POLYS: FoldCounter = FoldCounter::new();
/// Folds too small for the width to matter, counted apart so they cannot
/// inflate the call count of the buckets that carry the work.
pub static TINY: FoldCounter = FoldCounter::new();

/// Basefold's own fold, which halves a single polynomial by a random weight
/// rather than by fixing a variable, and the even-index restriction beside it.
/// Neither goes through the funnel above, so neither would otherwise be seen.
pub static FOLD_MLE: FoldCounter = FoldCounter::new();
pub static FIXED_AT_ZERO: FoldCounter = FoldCounter::new();

/// Below this a fold is bookkeeping rather than work.
const TINY_CELLS: usize = 1 << 10;

/// Records one fold.
pub fn record(rows: usize, polys: usize, nanos: u64) {
    let counter = if rows * polys < TINY_CELLS {
        &TINY
    } else if polys == 1 {
        &ONE_POLY
    } else if polys <= 16 {
        &FEW_POLYS
    } else {
        &MANY_POLYS
    };
    counter.record(rows, polys, nanos);
}

/// What the folds cost, as a table.
#[must_use]
pub fn report() -> String {
    let mut out = String::from(
        "\nfold width      calls           rows          cells   thread_s   ns/cell\n",
    );
    let mut total = (0u64, 0u64, 0u64, 0.0f64);
    for (name, c) in [
        ("1", &ONE_POLY),
        ("2-16", &FEW_POLYS),
        ("17+", &MANY_POLYS),
        ("tiny", &TINY),
        ("fold_mle", &FOLD_MLE),
        ("at_zero", &FIXED_AT_ZERO),
    ] {
        let (calls, rows, cells, secs) = c.counts();
        total = (total.0 + calls, total.1 + rows, total.2 + cells, total.3 + secs);
        let per = if cells == 0 { 0.0 } else { secs * 1e9 / cells as f64 };
        out.push_str(&format!(
            "{name:<10} {calls:>10} {rows:>14} {cells:>14} {secs:>10.3} {per:>9.2}\n"
        ));
    }
    let per = if total.2 == 0 { 0.0 } else { total.3 * 1e9 / total.2 as f64 };
    out.push_str(&format!(
        "{:<10} {:>10} {:>14} {:>14} {:>10.3} {per:>9.2}\n",
        "TOTAL", total.0, total.1, total.2, total.3
    ));
    out
}
