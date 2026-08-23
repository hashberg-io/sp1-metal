//! Named counters, for attributing a block of work without a profiler.
//!
//! Sampling has failed on this project four times, always the same way: rayon
//! owns the stack, so the profile blames the pool rather than whoever asked it
//! for work. Every time, a counter in the place the callers funnel through
//! answered the question in one run — and three of those four times it also
//! showed the premise was wrong, not just the location.
//!
//! So this is the fifth instance of the same instrument, and the last one that
//! needs writing: a name, a count of cells, and a duration. Names are
//! `&'static str` so the map cannot grow without a code change, and the whole
//! thing is behind one mutex because the sites that use it are coarse — once
//! per phase per shard, not once per cell.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

/// What one name has accumulated: calls, cells, seconds.
type Row = (u64, u64, f64);

static ROWS: Mutex<BTreeMap<&'static str, Row>> = Mutex::new(BTreeMap::new());

/// Records one occurrence of `name`.
pub fn record(name: &'static str, cells: usize, nanos: u64) {
    let mut rows = ROWS.lock().expect("tally poisoned");
    let row = rows.entry(name).or_insert((0, 0, 0.0));
    row.0 += 1;
    row.1 += cells as u64;
    row.2 += nanos as f64 / 1e9;
}

/// Times a closure and records it under `name`.
pub fn timed<T>(name: &'static str, cells: usize, f: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = f();
    record(name, cells, started.elapsed().as_nanos() as u64);
    out
}

/// A running measurement, recorded when it is stopped.
///
/// For the case `timed` cannot serve: a stretch of a function that borrows
/// something the closure would have to capture mutably.
#[must_use]
pub struct Span {
    name: &'static str,
    started: Instant,
}

impl Span {
    #[must_use]
    pub fn start(name: &'static str) -> Self {
        Self { name, started: Instant::now() }
    }

    pub fn stop(self, cells: usize) {
        record(self.name, cells, self.started.elapsed().as_nanos() as u64);
    }
}

/// Everything counted so far, as a table.
#[must_use]
pub fn report() -> String {
    let rows = ROWS.lock().expect("tally poisoned");
    if rows.is_empty() {
        return String::new();
    }
    let mut out = String::from("\ntally                       calls          cells   thread_s\n");
    for (name, (calls, cells, secs)) in rows.iter() {
        out.push_str(&format!("{name:<24} {calls:>10} {cells:>14} {secs:>10.3}\n"));
    }
    out
}
