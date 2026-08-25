// SPDX-License-Identifier: MIT OR Apache-2.0
//! One shader compilation at a time, process-wide.
//!
//! Creating a compute pipeline from two threads at once deadlocks inside
//! Apple's driver. Sampled during a hang, two prover threads sat in
//! `newComputePipelineStateWithFunction`, one inside
//! `MTLCompilerScheduler::buildRequest` and the other blocked in `_dlock_wait`
//! on a lock that first one holds, and neither moved again.
//!
//! Nothing in this project's own code was wrong: the pipeline cache released
//! its lock before compiling, deliberately, so that a thread wanting an
//! already-compiled kernel would not queue behind a cold compile. That is the
//! right instinct and it is what exposed the driver bug.
//!
//! So compilation is serialised here instead, in a lock that every crate in the
//! project shares because the driver's lock is per *device* and this project
//! has one. It has to be one lock rather than one per module: a zerocheck chip
//! compiling on one thread and a GKR context initialising on another are two
//! threads in the same driver call.
//!
//! **Take this around the Metal call only, never around the surrounding cache
//! lookup.** Holding it while a cache is consulted would put every thread
//! behind whichever one is compiling, which on a chip that takes ten minutes to
//! compile would stall the whole prover rather than one thread of it.
//!
//! ```ignore
//! if let Some(hit) = cache.lock().unwrap().get(key) { return hit.clone(); }
//! let _guard = sp1_metal_core::compile::lock();
//! // Re-check: another thread may have compiled this while we queued.
//! if let Some(hit) = cache.lock().unwrap().get(key) { return hit.clone(); }
//! let pipeline = device.new_compute_pipeline_state_with_function(&f);
//! ```

use std::sync::{Mutex, MutexGuard, OnceLock};

fn gate() -> &'static Mutex<()> {
    static GATE: OnceLock<Mutex<()>> = OnceLock::new();
    GATE.get_or_init(|| Mutex::new(()))
}

/// Blocks until no other thread is compiling, then admits this one.
///
/// The guard must be held across the pipeline-creation call and dropped as soon
/// as it returns. A poisoned lock is recovered rather than propagated: the
/// mutex guards nothing but the right to call into the driver, so a thread that
/// panicked while holding it has left no state behind to be inconsistent.
#[must_use]
pub fn lock() -> MutexGuard<'static, ()> {
    match gate().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}
