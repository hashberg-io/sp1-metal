// SPDX-License-Identifier: MIT OR Apache-2.0
//! Keeping a buffer between two uses of the same live allocation.
//!
//! Giving the device access to a borrowed allocation costs time proportional to
//! its size, paid between commit and the kernel starting, and it was invisible
//! until a fold was dispatched twice over the same buffers: identical kernel,
//! identical GPU time, and nine tenths less waiting the second time.
//!
//! Some allocations are handed straight from one kernel to another. The
//! Reed-Solomon encode writes a codeword and the Merkle kernels immediately hash
//! it; between them nothing on the host touches it, and it is the largest
//! borrowed allocation in the proof. Mapping it twice is pure cost.
//!
//! **The key is a pointer, so an entry that outlives its allocation is a
//! use-after-free waiting for a coincidence.** Two live allocations cannot
//! share an address, so a key that matches an allocation the entry was not made
//! for can only mean the original was freed and its address handed out again —
//! and the buffer then wired over that address is not the caller's memory in
//! any sense the language will defend.
//!
//! The table used to claim that could not happen, on the grounds that a
//! producer whose consumer never ran is evicted by the next producer. It is
//! not: it is evicted by the sixteenth. A hand-off is orphaned whenever the
//! commitment that would have taken it goes to the CPU, and — because SP1
//! proves a shard as a tokio task and a task moves between workers — whenever
//! the encode and the commit that follows it land on different threads, which
//! this table, being thread-local, cannot bridge. Both are ordinary, and both
//! leave an entry sitting on a pointer that will be freed. Measured over twelve
//! compressed `fib` proofs in one process: 265 hand-offs kept, 261 taken, four
//! orphaned — about one in seventy, or one every three proofs. That was the
//! gentler workload: thirty core proofs of larger guests later gave 94 orphans
//! in 1,560 hand-offs, one in seventeen, and two to four in *every* proof
//! rather than one in every third. The rate belongs to the workload, and the
//! reassuring figure was the smallest one measured.
//!
//! So three things bound it. A producer replaces any entry with its own key
//! rather than queueing behind it, [`take`] takes the newest match rather than
//! the oldest, and [`clear`] empties the table at the end of every commitment —
//! which is the whole window a hand-off is for, since the encode that leaves a
//! buffer here is immediately followed by the commit that wants it. What
//! remains is one commitment's worth of exposure rather than an unbounded
//! amount, which is what the paragraph above used to assert without cause.
//!
//! And then a fourth thing: it is off unless asked for. The exposure above was
//! being bounded on behalf of a saving that turned out not to exist, and the
//! codeword table produces it at about one hand-off in seventeen. See
//! [`reusing`] for the measurement that settled it.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

use metal::Buffer;

/// A pointer-keyed buffer cache's traffic: hand-offs left, taken, dropped
/// unclaimed, and — the one that matters — keyed on an address that a previous
/// entry was still sitting on.
///
/// A *collision* is the whole hazard in one number. Two live allocations
/// cannot share an address, so a producer that finds its key already present
/// has been handed the address of an allocation that has since been freed. Had
/// the older entry been the one a consumer matched, a kernel would have read
/// whatever that address holds now through a buffer wired over what it held
/// then. Counted rather than merely prevented, because "this could happen" and
/// "this happens N times a proof" are different claims and only the second is
/// a measurement.
#[derive(Debug, Default)]
pub struct Reuse {
    kept: AtomicUsize,
    taken: AtomicUsize,
    orphaned: AtomicUsize,
    collided: AtomicUsize,
}

impl Reuse {
    const fn new() -> Self {
        Self {
            kept: AtomicUsize::new(0),
            taken: AtomicUsize::new(0),
            orphaned: AtomicUsize::new(0),
            collided: AtomicUsize::new(0),
        }
    }

    /// Records a buffer left for a later reader, and how many entries for the
    /// same address it displaced.
    pub fn kept(&self, collided: usize) {
        self.kept.fetch_add(1, Ordering::Relaxed);
        self.collided.fetch_add(collided, Ordering::Relaxed);
    }

    /// Records a reader that found one.
    pub fn taken(&self) {
        self.taken.fetch_add(1, Ordering::Relaxed);
    }

    /// Records buffers dropped because the window they were left in has ended.
    pub fn orphaned(&self, n: usize) {
        self.orphaned.fetch_add(n, Ordering::Relaxed);
    }

    /// `(kept, taken, orphaned, collided)` since the process started.
    #[must_use]
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        (
            self.kept.load(Ordering::Relaxed),
            self.taken.load(Ordering::Relaxed),
            self.orphaned.load(Ordering::Relaxed),
            self.collided.load(Ordering::Relaxed),
        )
    }
}

/// The encode's codewords, handed to the Merkle kernels that hash them.
pub static CODEWORDS: Reuse = Reuse::new();

/// The two arrays a jagged round reduces, handed to the fold that reads them.
pub static ROUND_ARRAYS: Reuse = Reuse::new();

/// Whether buffers are kept for a second use, read once.
///
/// Off, because the A/B the note above used to ask for was taken and found
/// nothing to keep. Thirty pairs over the three workloads whose mapping wait
/// is the largest share of their proof — which is where this has the most it
/// could possibly save — put every wall-clock difference inside the scatter of
/// the same arm across repeats, and put the mapping wait itself on both sides
/// of zero: it favoured keeping buffers on two of the three and dropping them
/// on the third. A saving that changes sign between workloads is not one.
/// `results/2026-08-29-m4-max-reuse-ab.md` is the measurement.
///
/// `SP1_METAL_REUSE=1` still turns it on. Nothing below has changed; what
/// changed is that the exposure is opt-in rather than paid by default for a
/// return that could not be measured.
#[must_use]
pub fn reusing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("SP1_METAL_REUSE").map(|v| v != "0").unwrap_or(false))
}

/// How many hand-offs may be in flight at once. A commitment encodes several
/// codewords before hashing any of them, so this is not one.
const SLOTS: usize = 16;

thread_local! {
    static HANDOFF: RefCell<Vec<((usize, usize), Buffer)>> = const { RefCell::new(Vec::new()) };
}

/// Leaves `buffer` for whoever reads the same bytes next.
///
/// The caller must keep the allocation alive and unmoved until then.
pub fn keep(bytes: *const u8, len: usize, buffer: &Buffer) {
    if !reusing() {
        return;
    }
    keep_now(bytes, len, buffer);
}

/// The table itself, with the switch already decided.
///
/// Split out so the tests below can exercise the eviction and ordering rules
/// without depending on how [`reusing`] happens to be answering: those rules
/// are what must hold whenever the table is used at all, and a test that
/// silently does nothing when the default flips is worse than no test.
fn keep_now(bytes: *const u8, len: usize, buffer: &Buffer) {
    let key = (bytes as usize, len);
    HANDOFF.with(|h| {
        let mut h = h.borrow_mut();
        // An entry with this key describes an allocation that no longer exists:
        // this one has the address now. Keeping it could only ever hand a
        // consumer a buffer over freed memory.
        let before = h.len();
        h.retain(|(k, _)| *k != key);
        CODEWORDS.kept(before - h.len());
        if h.len() >= SLOTS {
            h.remove(0);
        }
        h.push((key, buffer.clone()));
    });
}

/// Takes the buffer left for these bytes, if there is one.
///
/// Removing it is what keeps the key from outliving the allocation.
#[must_use]
pub fn take(bytes: *const u8, len: usize) -> Option<Buffer> {
    let key = (bytes as usize, len);
    HANDOFF.with(|h| {
        let mut h = h.borrow_mut();
        // From the back: the newest entry for an address is the only one that
        // can still describe live memory.
        let i = h.iter().rposition(|(k, _)| *k == key)?;
        CODEWORDS.taken();
        Some(h.remove(i).1)
    })
}

/// Drops every buffer left here.
///
/// Called when a commitment ends, which is the end of the window any hand-off
/// was made for. Whatever is still here was left by a producer whose consumer
/// never ran, and its key is a pointer that may be freed at any moment.
pub fn clear() {
    HANDOFF.with(|h| {
        let mut h = h.borrow_mut();
        CODEWORDS.orphaned(h.len());
        h.clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::{Device, MTLResourceOptions};

    /// Two producers for one address mean the first allocation was freed and
    /// the second got its place. Handing a consumer the first buffer is a read
    /// of memory that no longer belongs to anyone, so the newest must win and
    /// the older must not be reachable at all.
    #[test]
    fn the_newest_producer_for_an_address_wins() {
        let Some(device) = Device::system_default() else {
            return;
        };
        clear();
        let stale = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        let live = device.new_buffer(128, MTLResourceOptions::StorageModeShared);
        let addr = 0x4000 as *const u8;
        keep_now(addr, 32, &stale);
        keep_now(addr, 32, &live);
        let got = take(addr, 32).expect("a buffer was kept for these bytes");
        assert_eq!(got.length(), 128, "the stale buffer was handed out");
        assert!(take(addr, 32).is_none(), "the stale buffer outlived the take");
    }

    /// A hand-off lasts one commitment. Anything still here when that ends was
    /// left by a producer whose consumer never ran -- because the commitment
    /// went to the CPU, or because the task moved to another worker between the
    /// two -- and its key is a pointer about to be freed.
    #[test]
    fn clearing_ends_the_window() {
        let Some(device) = Device::system_default() else {
            return;
        };
        clear();
        let buffer = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        keep_now(0x8000 as *const u8, 16, &buffer);
        clear();
        assert!(take(0x8000 as *const u8, 16).is_none(), "an orphan survived the commitment");
    }

    /// The table is small, and the thing it must never do when full is drop the
    /// entry someone is about to ask for.
    #[test]
    fn a_full_table_keeps_the_newest() {
        let Some(device) = Device::system_default() else {
            return;
        };
        clear();
        let buffer = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        for i in 0..SLOTS + 4 {
            keep_now((0x10000 + i * PAGE_STRIDE) as *const u8, 16, &buffer);
        }
        let newest = (0x10000 + (SLOTS + 3) * PAGE_STRIDE) as *const u8;
        assert!(take(newest, 16).is_some(), "the newest hand-off was evicted");
        let oldest = 0x10000 as *const u8;
        assert!(take(oldest, 16).is_none(), "an entry survived past the table's size");
        clear();
    }

    /// The hand-off is opt-in, because measurement found it saving nothing that
    /// could be told from zero while leaving a pointer key on a freed
    /// allocation about one hand-off in seventeen. This pins the default, which
    /// is otherwise a single word that any edit could flip back unnoticed.
    ///
    /// Skipped where the environment has an opinion, since `reusing` reads it.
    #[test]
    fn the_hand_off_is_off_unless_it_is_asked_for() {
        if std::env::var("SP1_METAL_REUSE").is_ok() {
            return;
        }
        assert!(!reusing(), "the hand-off defaulted back on");
        let Some(device) = Device::system_default() else {
            return;
        };
        clear();
        let buffer = device.new_buffer(64, MTLResourceOptions::StorageModeShared);
        keep(0xc000 as *const u8, 16, &buffer);
        assert!(take(0xc000 as *const u8, 16).is_none(), "a buffer was kept anyway");
    }

    /// Distinct made-up addresses for the eviction test. Nothing dereferences
    /// them; they only have to differ.
    const PAGE_STRIDE: usize = 0x4000;
}
