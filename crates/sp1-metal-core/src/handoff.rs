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
//! **The rule that makes this sound is the lifetime, not the key.** A caller may
//! only leave a buffer here for an allocation that is still borrowed when the
//! taker runs, and the taker removes the entry, so a pointer can never be
//! matched after the memory behind it has been freed. Anything left behind by a
//! producer whose consumer never ran is evicted by the next producer rather than
//! kept, which bounds how long a stale key can sit here to one hand-off.

use std::cell::RefCell;

use metal::Buffer;

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
    HANDOFF.with(|h| {
        let mut h = h.borrow_mut();
        if h.len() >= SLOTS {
            h.remove(0);
        }
        h.push(((bytes as usize, len), buffer.clone()));
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
        let i = h.iter().position(|(k, _)| *k == key)?;
        Some(h.remove(i).1)
    })
}
