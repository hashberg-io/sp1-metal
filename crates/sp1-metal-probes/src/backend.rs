//! `MetalBackend`: slop's device seam, implemented for Apple GPUs.
//!
//! slop is generic over `A: Backend`, one small unsafe trait over `Allocator +
//! DeviceMemory`, and everything above it — `Buffer`, `Mle`, `Point`, `Tensor`
//! — carries the backend as a type parameter defaulting to `CpuBackend`.
//! Implementing it is what lets the prover's data structures live on the
//! device.
//!
//! On unified memory that implementation is unusually thin, and this is the one
//! place Apple's architecture is an advantage rather than the constraint the
//! rest of this project keeps running into. A `MTLResourceStorageModeShared`
//! buffer is coherent between CPU and GPU and exposes a host pointer, so an
//! allocation *is* a Metal buffer and a Metal buffer *is* host memory. Every
//! copy direction — host to device, device to host, device to device — is a
//! `memcpy`, and in the common case where a kernel reads data the host just
//! wrote there is no copy at all: `buffer_for` finds the `MTLBuffer` behind any
//! pointer slop hands out, so a `Buffer<T, MetalBackend>` can be bound to a
//! kernel where it lies.
//!
//! A CUDA backend has to stage through pinned host memory, track which side
//! owns each allocation, and pay PCIe for every crossing. None of that exists
//! here.

use std::alloc::Layout;
use std::collections::BTreeMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use metal::{Buffer, Device, MTLResourceOptions};
use slop_alloc::mem::{CopyDirection, CopyError, DeviceMemory};
use slop_alloc::{AllocError, Allocator, Backend, GlobalBackend};

/// One live allocation: the Metal buffer, and how far it extends from the host
/// pointer it exposes.
struct Allocation {
    buffer: Buffer,
    len: usize,
}

/// Bytes of freed buffers kept for reuse. A prover folds the same working set
/// down round after round, so it asks for the same handful of sizes over and
/// over; without a pool every round pays the kernel to map fresh pages, which
/// costs more than the round's arithmetic.
const POOL_BUDGET_BYTES: usize = 2 << 30;

struct Inner {
    device: Device,
    /// Live allocations keyed by the base address of their host mapping, so a
    /// pointer into the middle of one can be resolved back to its buffer.
    live: Mutex<BTreeMap<usize, Allocation>>,
    /// Freed buffers, keyed by exact byte length, waiting to be handed out again.
    pool: Mutex<(BTreeMap<usize, Vec<Buffer>>, usize)>,
}

/// A slop backend backed by Metal buffers in shared storage.
#[derive(Clone)]
pub struct MetalBackend {
    inner: Arc<Inner>,
}

impl fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBackend").field("device", &self.inner.device.name()).finish()
    }
}

impl MetalBackend {
    /// Wraps the system default Metal device.
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device found")?;
        Ok(Self {
            inner: Arc::new(Inner {
                device,
                live: Mutex::new(BTreeMap::new()),
                pool: Mutex::new((BTreeMap::new(), 0)),
            }),
        })
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// The Metal buffer behind a pointer this backend allocated, and the byte
    /// offset of the pointer within it. This is the whole of "uploading" a
    /// slop `Buffer` to the GPU: there is nothing to upload.
    pub fn buffer_for(&self, ptr: *const u8) -> Option<(Buffer, u64)> {
        let addr = ptr as usize;
        let live = self.inner.live.lock().expect("allocation table poisoned");
        let (&base, alloc) = live.range(..=addr).next_back()?;
        (addr < base + alloc.len).then(|| (alloc.buffer.clone(), (addr - base) as u64))
    }

    /// Number of allocations currently outstanding, for tests.
    pub fn live_allocations(&self) -> usize {
        self.inner.live.lock().expect("allocation table poisoned").len()
    }

    /// Bytes currently held in the reuse pool, for tests.
    pub fn pooled_bytes(&self) -> usize {
        self.inner.pool.lock().expect("pool poisoned").1
    }
}

unsafe impl Allocator for MetalBackend {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let len = layout.size().max(1);
        let pooled = {
            let mut pool = self.inner.pool.lock().expect("pool poisoned");
            let taken = pool.0.get_mut(&len).and_then(Vec::pop);
            if taken.is_some() {
                pool.1 -= len;
            }
            taken
        };
        let buffer = pooled.unwrap_or_else(|| {
            self.inner.device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
        });
        let ptr = NonNull::new(buffer.contents().cast::<u8>()).ok_or(AllocError)?;
        // Metal hands back page-aligned mappings, so any alignment slop asks
        // for is already satisfied; refuse rather than hand back a lie if not.
        if (ptr.as_ptr() as usize) % layout.align() != 0 {
            return Err(AllocError);
        }
        self.inner
            .live
            .lock()
            .expect("allocation table poisoned")
            .insert(ptr.as_ptr() as usize, Allocation { buffer, len });
        Ok(NonNull::slice_from_raw_parts(ptr, len))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        let freed = self
            .inner
            .live
            .lock()
            .expect("allocation table poisoned")
            .remove(&(ptr.as_ptr() as usize));
        if let Some(Allocation { buffer, len }) = freed {
            let mut pool = self.inner.pool.lock().expect("pool poisoned");
            if pool.1 + len <= POOL_BUDGET_BYTES {
                pool.1 += len;
                pool.0.entry(len).or_default().push(buffer);
            }
        }
    }
}

impl DeviceMemory for MetalBackend {
    /// Shared storage is coherent, so every direction is the same `memcpy` and
    /// the direction is only documentation.
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        _direction: CopyDirection,
    ) -> Result<(), CopyError> {
        std::ptr::copy_nonoverlapping(src, dst, size);
        Ok(())
    }

    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        std::ptr::write_bytes(dst, value, size);
        Ok(())
    }
}

unsafe impl Backend for MetalBackend {}

static GLOBAL: OnceLock<MetalBackend> = OnceLock::new();

impl GlobalBackend for MetalBackend {
    fn global() -> &'static Self {
        GLOBAL.get_or_init(|| MetalBackend::new().expect("no Metal device found"))
    }
}
