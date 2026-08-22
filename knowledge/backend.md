# Building the backend

The record of executing steps 5 onward of [backend-plan.md](backend-plan.md), after the three spikes and the go/no-go recorded in [spikes.md](spikes.md).

## 5. `MetalBackend`

slop's device seam is one small unsafe trait, `Backend`, over `Allocator + DeviceMemory + Clone + Debug + Send + Sync + 'static`, and everything above it — `Buffer`, `Mle`, `Point`, `Tensor`, `Evaluations` — carries the backend as a type parameter defaulting to `CpuBackend`.
`crates/sp1-metal-probes/src/backend.rs` implements it.

On unified memory the implementation is unusually thin, and this is the one place Apple's architecture is an advantage rather than the constraint the rest of the project keeps hitting.
An `MTLResourceStorageModeShared` buffer is coherent between CPU and GPU and exposes a host pointer, so an allocation *is* a Metal buffer and a Metal buffer *is* host memory.
`allocate` creates a shared buffer and hands back its `contents()` pointer, registering the mapping in an address-keyed table; `deallocate` drops the buffer; and all three copy directions in `DeviceMemory` are the same `memcpy`, because there are no two places for the data to be.
`buffer_for` runs the table backwards: given any pointer slop hands out, it finds the `MTLBuffer` behind it and the offset within it, which is the whole of "uploading" a buffer to the GPU.

**Verified end to end.** The parity check allocates a `slop_alloc::Buffer<u32, MetalBackend>` through slop's own API, fills it through its host pointer, binds it to a Metal kernel by pointer lookup, folds it on the GPU, reads the result back through the pointer, and compares against slop's `fold_mle`.
It also asserts that the allocations reached the backend and that none leak.
The refusal test recorded allocation lengths as zero, which makes the pointer lookup fail to resolve; the check failed with exactly that, and was restored.

**What the seam costs.**

| on 268 MB | |
|---|---|
| making it visible to a kernel | **0 ns** (a table lookup, below timer resolution) |
| the kernel itself (a base-field fold) | 1.17 ms |
| the staging copy a separate-memory backend would pay first | 3.69 ms at 73 GB/s |

So on this architecture the transfer a CUDA backend must perform before it can start would cost **3.17x the kernel's own runtime** for a bandwidth-shaped kernel, and here it does not exist.
That is a conservative statement of the advantage: 73 GB/s is a single-threaded host `memcpy`, and a real host-to-device transfer over PCIe would be slower still.

It does not rescue the verdict — 3.17x of nothing is still nothing when the kernel itself is only 1.4x the CPU — but it is the reason the integration work in the remaining steps is small where a CUDA port's would be large.
