// SPDX-License-Identifier: MIT OR Apache-2.0
//! When Metal will take memory where it lies.
//!
//! `newBufferWithBytesNoCopy` is the whole reason a GPU backend on unified
//! memory can be cheap: the pages the host just wrote are the pages the kernel
//! reads, and "uploading" is making a buffer object. The question is only when
//! it is allowed.
//!
//! It has exactly one requirement, and this project spent a long time
//! believing it had two. The pointer must be page-aligned. The length need not
//! be: Metal wires the pages spanning `[ptr, ptr + len)`, and since the pointer
//! starts a page, every page in that range holds at least one byte of the
//! caller's allocation and is therefore already mapped. Rounding up to a page
//! boundary cannot reach an unmapped page, which is why an arbitrary length is
//! safe rather than merely tolerated.
//!
//! The second, invented requirement cost about 155 GB of memcpy per proof:
//! field arrays are almost always page-aligned, because they are large enough
//! that the allocator serves them from their own mapping, and almost never a
//! whole number of pages long, because their length is a row count times an
//! interaction count.
//!
//! There *is* a second requirement, and it is about the caller rather than the
//! pointer: **no two live no-copy buffers may cover the same page.** Lending an
//! allocation once is always safe. Lending a *sub-slice* is safe only if
//! nothing else lends a slice sharing its pages at the same time — and since
//! Metal takes whole pages, two slices of one array can share a page without
//! overlapping by a single byte.
//!
//! The GKR round broke this and it cost a wrong proof, not a crash: it bound a
//! prefix of the same array once per chip, so a layer with several wide chips
//! had several buffers over that array's first page at once. See
//! `results/2026-08-26-m4-max-gkr-overlapping-lends.md`. Anything that lends
//! per item rather than per array should copy instead, which for a slice small
//! enough to be bound per item costs nothing worth measuring.

use metal::{Buffer, Device, MTLResourceOptions};

use crate::stats::{self, Input, Output};

/// The virtual page size on Apple silicon, and the alignment Metal demands of
/// a pointer it is asked to take without copying.
pub const PAGE: usize = 16 * 1024;

/// Whether Metal will take this pointer as it is.
#[must_use]
pub fn takeable(ptr: *const u8) -> bool {
    ptr as usize % PAGE == 0
}

/// A buffer over `bytes` that a kernel may read, without copying if Metal will
/// allow it.
///
/// # Safety
///
/// The buffer borrows `bytes` and does not extend its lifetime. The caller must
/// keep the slice alive, and unmodified, until the dispatch reading it has
/// completed.
#[must_use]
pub unsafe fn read_only(device: &Device, bytes: &[u8]) -> Buffer {
    if !bytes.is_empty() && takeable(bytes.as_ptr()) {
        stats::record_input(Input::Wrapped, bytes.len(), 0);
        return device.new_buffer_with_bytes_no_copy(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
    }
    // Copying is the host reading these bytes, so anything still writing them
    // has to finish first. Borrowing them is not: the dispatch that reads the
    // buffer is ordered behind the one that wrote it by the queue.
    crate::queue::drain();
    let start = std::time::Instant::now();
    let buffer = device.new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len().max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    stats::record_input(Input::Copied, bytes.len(), start.elapsed().as_nanos() as u64);
    buffer
}

/// A buffer a kernel will write, aliasing `bytes` when Metal will allow it.
///
/// Returns whether it aliases: when it does not, the caller has to copy the
/// result back out, and should say so with [`stats::record_output`].
///
/// # Safety
///
/// As [`read_only`], and the caller must not read `bytes` until the dispatch
/// writing it has completed.
#[must_use]
pub unsafe fn write_into(device: &Device, bytes: &mut [u8]) -> (Buffer, bool) {
    if !bytes.is_empty() && takeable(bytes.as_ptr()) {
        stats::record_output(Output::Direct, bytes.len(), 0);
        let buffer = device.new_buffer_with_bytes_no_copy(
            bytes.as_mut_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        return (buffer, true);
    }
    let buffer =
        device.new_buffer(bytes.len().max(1) as u64, MTLResourceOptions::StorageModeShared);
    (buffer, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::{ComputePipelineDescriptor, MTLSize};

    /// The claim the module rests on: a page-aligned pointer with a length that
    /// is not a whole number of pages still aliases, and a kernel writing
    /// through it writes the caller's own memory.
    #[test]
    fn an_odd_length_still_aliases() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let src = "kernel void bump(device uint* xs [[buffer(0)]], \
                   constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) \
                   { if (gid < n) { xs[gid] = xs[gid] + 1u; } }";
        let library = device
            .new_library_with_source(src, &metal::CompileOptions::new())
            .expect("compile");
        let f = library.get_function("bump", None).expect("kernel");
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&f));
        let pso = device.new_compute_pipeline_state(&desc).expect("pipeline");
        let queue = device.new_command_queue();
        // A megabyte and one word: large enough that the allocator gives it its
        // own page-aligned mapping, and a page and a word too long.
        let words = 64 * PAGE / 4 + 1;
        let mut v: Vec<u32> = (0..words as u32).collect();
        if !takeable(v.as_ptr().cast()) {
            return;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), words * 4)
        };
        let (buf, aliased) = unsafe { write_into(&device, bytes) };
        assert!(aliased, "a page-aligned pointer was refused");
        let n32 = words as u32;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf), 0);
        enc.set_bytes(1, 4, (&n32 as *const u32).cast());
        enc.dispatch_thread_groups(
            MTLSize::new((words as u64).div_ceil(256), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u32 + 1));
    }
}

/// This thread's command queue.
///
/// A `MTLCommandQueue` runs the command buffers committed to it in the order
/// they were committed. One queue shared by four proving threads therefore
/// makes them wait for each other even where the device could have run their
/// work side by side, which is most of the time: the shards are independent.
///
/// One queue per thread rather than one per module, because two dispatches from
/// the same thread are ordered by that thread anyway, and a queue costs a small
/// amount of driver state.
#[must_use]
pub fn thread_queue(device: &metal::Device) -> metal::CommandQueue {
    thread_local! {
        static Q: std::cell::RefCell<Option<metal::CommandQueue>> =
            const { std::cell::RefCell::new(None) };
    }
    Q.with(|q| q.borrow_mut().get_or_insert_with(|| device.new_command_queue()).clone())
}
