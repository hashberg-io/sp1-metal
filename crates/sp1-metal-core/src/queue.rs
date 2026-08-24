// SPDX-License-Identifier: MIT OR Apache-2.0
//! Committing without waiting, and why it does not work here.
//!
//! Most of what this backend waits for, it does not read. A sumcheck round's
//! fold writes the polynomial the next round reduces; the host never looks at
//! it. Waiting for that write costs a round trip — 0.34 ms of it even with
//! nothing else running, 1.6 ms with four shards proving — and a
//! `MTLCommandQueue` runs its buffers in commit order, so a later dispatch on
//! the same queue already sees the earlier one's writes. Eight and a half
//! thread-seconds of a twelve second proof go on waits of exactly that shape.
//!
//! **It does not work, and the reason is worth keeping.** Both halves of the
//! argument are per-thread: [`nocopy::thread_queue`](crate::nocopy::thread_queue)
//! gives each thread its own queue, so commit order only orders that thread's
//! dispatches, and the list of un-awaited buffers below is thread-local, so
//! only that thread can wait for them. The data is not per-thread. SP1 proves a
//! shard as a tokio task, and a task moves between worker threads — so a
//! codeword written by a deferred buffer on one worker gets read by another,
//! which has nothing to wait for and no ordering to rely on.
//!
//! What that looks like from outside is not a crash. Deferring only the
//! Reed-Solomon encode, on a guest that runs a thousand steps, produces a
//! *wrong proof* — `Query value mismatch` in the dense PCS, because the Merkle
//! tree was built over a codeword that had not finished being written. The
//! prover retries the shard, exhausts its retry budget, and the process then
//! sits with every thread parked and nothing in Metal at all, which is what
//! sent the first two attempts at diagnosis in the wrong direction.
//!
//! So [`deferring`] is off, and the switch exists to keep the experiment
//! reproducible rather than to be turned on. Making this work needs the wait
//! attached to the data — a pending buffer carried by the tensor it writes,
//! waited for by whoever reads it — and not to the thread that happened to
//! submit it. That is a different and much larger change than removing a
//! `wait_until_completed`.
//!
//! [`submit`] is what everything uses meanwhile: it waits, and it also waits
//! for anything left pending on this thread, so the two paths compose.

use std::cell::RefCell;
use std::sync::OnceLock;
use std::time::Instant;

use metal::objc::{msg_send, sel, sel_impl};
use metal::{CommandBuffer, CommandBufferRef};

use crate::stats;

thread_local! {
    /// Buffers this thread has committed and not yet waited for.
    ///
    /// They complete in this order, so waiting for the last implies the rest;
    /// the list waits for all of them anyway, because relying on that would be
    /// relying on a guarantee this file does not need.
    static PENDING: RefCell<Vec<CommandBuffer>> = const { RefCell::new(Vec::new()) };
}

/// Reads a finished buffer's GPU interval.
///
/// `GPUStartTime` and `GPUEndTime` are not bound by the metal crate, but they
/// are ordinary Objective-C properties on `MTLCommandBuffer`.
fn interval(cb: &CommandBufferRef) -> (f64, f64) {
    unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) }
}

/// Commits without waiting.
///
/// The caller must not read what the kernel writes. The next dispatch on this
/// thread will see it; the host will not, until someone calls [`drain`].
pub fn submit_deferred(cb: &CommandBufferRef) {
    if !deferring() {
        submit(cb);
        return;
    }
    cb.commit();
    let pending = PENDING.with(|p| {
        p.borrow_mut().push(cb.to_owned());
        p.borrow().len()
    });
    // A command queue serves a bounded number of command buffers, and a slot is
    // held until its buffer is released -- so a buffer kept here holds one.
    // In a correctly wired path this list holds one entry, because the next
    // dispatch drains it; the cap is what turns a missing drain into a slower
    // proof rather than a queue that never hands out another buffer.
    if pending >= MAX_PENDING {
        drain();
    }
}

/// How many command buffers may sit un-awaited before one is waited for.
const MAX_PENDING: usize = 4;

/// Whether deferring is on, read once. **Off, and see the module note.**
///
/// `SP1_METAL_DEFER=1` reproduces the failure: a wrong proof rather than a
/// crash, because the wait is tied to the thread and the data is not.
pub fn deferring() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_DEFER").map(|v| v != "0").unwrap_or(false))
}

/// Waits for everything this thread has committed and not yet waited for.
///
/// Returns the seconds spent waiting, which is zero when nothing was pending.
pub fn drain() -> f64 {
    let pending = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
    if pending.is_empty() {
        return 0.0;
    }
    let start = Instant::now();
    for cb in &pending {
        cb.wait_until_completed();
    }
    let wall = start.elapsed().as_secs_f64();
    // The wait belongs to the drain, not to any one buffer, so it is recorded
    // once; each buffer still contributes its own interval to device busy.
    for (i, cb) in pending.iter().enumerate() {
        let (t0, t1) = interval(cb);
        stats::record_dispatch(if i == 0 { wall } else { 0.0 }, t1 - t0);
        stats::record_interval(t0, t1);
    }
    wall
}

/// Commits and waits, for a caller that is about to read the result.
///
/// Anything already deferred on this thread is waited for too, since it was
/// committed first and this buffer runs behind it regardless.
///
/// Returns `(seconds spent waiting, this buffer's GPU seconds)`.
pub fn submit(cb: &CommandBufferRef) -> (f64, f64) {
    let start = Instant::now();
    cb.commit();
    // Whether giving the device access to a buffer is charged to the host here
    // or to the driver afterwards decides whether not waiting could ever help.
    let committed = start.elapsed().as_secs_f64();
    let deferred = drain();
    cb.wait_until_completed();
    let wall = start.elapsed().as_secs_f64() - deferred;
    crate::tally::record("queue: commit", 0, (committed * 1e9) as u64);
    crate::tally::record("queue: wait", 0, ((wall - committed).max(0.0) * 1e9) as u64);
    let (t0, t1) = interval(cb);
    stats::record_dispatch(wall, t1 - t0);
    stats::record_interval(t0, t1);
    (wall, t1 - t0)
}
