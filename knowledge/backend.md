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

## 6. The PCS: commit done, opening not

`MultilinearPcsProver` is a small trait with two halves — `commit_multilinear` and `prove_trusted_evaluation` — and it is a real type parameter of `ShardProver`, so replacing it needs no fork.
The commit half is built and matches slop exactly. The opening half is not, and is the larger job.

### The Merkle tree, and a detail that would have broken everything

slop commits a tensor by hashing each row of the row-major matrix into a digest with `PaddingFreeSponge<KoalaPerm, 16, 8, 8>` and compressing pairs up the tree with `TruncatedPermutation<KoalaPerm, 2, 8, 16>`.
Both are the width-16 Poseidon2 permutation the suite already parity-checks, so both are one kernel, and neither needs the host between layers: the whole tree is a single submission.

Building it against slop's own `Poseidon2KoalaBear16Prover` surfaced something a from-scratch implementation would have got wrong.
**The commitment is not the Merkle root.**
slop finishes with `compress([root, hash([log_height, total_width])])`, binding the shape of the committed matrix into the commitment, so a backend that returned the bare root would produce commitments the verifier rejects — and would pass every internal consistency check while doing it.
Parity now covers five shapes, including widths that are not multiples of the sponge's rate; swapping the order of a node's two children fails it.

| Merkle tree over 2^21 rows of width 64, 18.9 M permutations | | |
|---|---|---|
| slop's own commitment | 5577 ms | 3.4 M perm/s |
| Metal | 36.6 ms | 516 M perm/s |

The ratio of 152x is **not** a hardware ratio and should not be quoted as one.
slop's commitment reaches 3.4 M permutations per second against the **38.5 M** the same machine's packed CPU permutation reaches in the `poseidon2` probe, so most of that factor is CPU headroom of the same kind the encode spike found.
The hardware ratio for this work is the `poseidon2` probe's 15.7x; the kernel's own 516 M perm/s against that probe's 605 M is the tree paying for reading the codeword.

### The sumcheck seam

The three traits in `slop_sumcheck::backend` are the designed extension point for a device backend, and the zerocheck path is wired through them on the CPU side.
They are now implemented for `MetalBackend` over a `MetalMle` whose values live in a `slop_alloc::Buffer<_, MetalBackend>`.

**slop's own `reduce_sumcheck_to_evaluation` then runs unmodified and produces the identical proof** — the same univariate messages, the same point, the same final evaluation — whether it folds slop's CPU `Mle` or a `MetalMle`. Reading the odd-indexed sum instead of the even one fails it at the first round.

| sumcheck over 2^22 extension values, 22 rounds | | |
|---|---|---|
| slop `Mle` on `CpuBackend` | 12.00 ms | 546 us/round |
| `MetalMle` on `MetalBackend` | 5.09 ms | 231 us/round |
| | | **2.36x** |

Getting there took three fixes and one correction, all of which are lessons for the rest of the port.
The backend's allocator needed a **reuse pool**: a sumcheck asks for the same handful of buffer sizes every round, and mapping fresh pages each time cost more than the round's arithmetic.
The reduction's scratch buffers needed to be allocated **once** rather than per round.
And the fold needed to stop blocking: command buffers on one queue run in order, so the next round's reduction is already sequenced behind the fold and the host learns nothing by waiting — but a host *read* of the folded values does have to wait, which is a real synchronisation obligation and briefly a real bug.
The correction was to the measurement: an early version timed the polynomial's construction along with the proof, which on the Metal side meant converting four million extension elements into Montgomery form, and understated the result.

2.36x against a scalar CPU baseline is modest, and the reason is the one the challenger spike found.
The driver observes each round's message into a **host** challenger and samples the next alpha there, so every round is a blocking round trip no matter how fast the kernels are — twenty-two of them here, in upstream's own control flow rather than in a probe.
The prover recovers this not by changing the driver but by having many sumchecks in flight at once, which is exactly what the challenger spike measured and what shard concurrency provides.

### What is not done

`prove_trusted_evaluation` — the jagged sumcheck, the stacking, BaseFold's folding rounds, the FRI query openings and their Merkle paths — is not implemented.
That is the larger half of the PCS phase's 19%, and with it steps 7 through 9 remain open.
