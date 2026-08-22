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

## 7. GKR, through a fork

The plan's structural claim was that GKR sits behind concrete CPU types and needs a vendored fork.
It is stronger than a reading of the types: `GkrProverImpl::prove_gkr_circuit` is public and would give a reference to check against, but it takes a `LogupGkrCpuCircuit` whose `layers` field is **private, with no public constructor**, so the circuit cannot be built, driven or parity-checked from outside the crate.
Attempting it fails to compile.

So `crates/vendor/sp1-hypercube` is upstream 6.4.0 with a recorded set of changes, and consumers reach it through `[patch.crates-io]` — which matters, because `sp1-core-machine` depends on `sp1-hypercube` too and both must resolve to the same copy for the types to unify.

### The seam

A GKR round's message is built from three sums over the layer, and everything else about a round — layer transitions, padding and eq adjustments, interpolating the message from its evaluations — is the same wherever those sums are computed.
`LogupRoundOps` names exactly that boundary, with `CpuRoundOps` holding the code that was inline, moved verbatim.
The fold went behind it too, at layer granularity rather than per polynomial, because a layer has one polynomial per chip and there are of the order of a hundred chips.
`DefaultRoundOps` selects the implementation, so turning the `metal` feature on moves the phase to the GPU **without a type changing anywhere downstream, sp1-sdk included**.

The crate's own GKR tests pass unmodified beyond the new struct field, which is what establishes the CPU path is untouched.

### What it measures, in isolation

On a shard-sized layer — 2^16 rows, 108 interactions across 8 chips — against the same layer on the CPU:

| | | |
|---|---|---|
| round reduction | 12.0x | |
| fold | 3.7x | after batching; 1.8x before |
| **whole round** | **9.06x** | 6.0x before batching, 4.3x with the fold left on the CPU |

Feeding 9.06x into the model puts a complete port at **8.15x** end to end.

Parity covers both halves — the reduction against the CPU one for base and extension layers, the fold against slop's own `fix_last_variable` — across several chip mixes and both even and odd row counts, the odd case being where the last row pair is half padding.
Squaring one denominator, or dropping the odd-row guard, each fails exactly the test that covers it.

Writing those tests found a real bug rather than a test artefact: the scratch buffer for per-threadgroup partials was a single shared allocation, which two concurrent reductions race on. SP1 proves shards concurrently, so it would have corrupted proofs under load. It is a checkout pool now.

### What it does in the real prover: not yet a win

Wired into an actual proof, the picture changes, and this is the honest result.
A synthetic layer is a handful of chips all of the same size; a real one is about a hundred chips whose sizes span orders of magnitude, and the cost of *preparing* a dispatch — creating Metal buffer objects, which is a call into the kernel — is paid per chip regardless of how little that chip has in it.

Three rounds of fixes each moved it a long way and none was enough:

- **Batching.** One command buffer per layer instead of one per chip, and one per layer's folds instead of one per polynomial. Worth 6.0x to 9.06x on the synthetic layer.
- **A size threshold.** Layers below a few thousand entries stay on the CPU, since below that the dispatch costs more than the arithmetic saves. Overridable with `SP1_METAL_MIN_ENTRIES`, which the tests set to zero so small shapes still reach the kernels.
- **Inline arguments.** The small per-chip slices — a chip's eq values, a fold's padding — go into the command buffer directly rather than into freshly created buffers.

- **A per-chip decision.** The threshold was being applied to the layer, so a layer that cleared it still dispatched its ninety small chips. Each chip and each polynomial now decides for itself, with the small ones summed on the CPU and added to what the kernels return. That is exact because the sums are additive, and a test builds a layer whose chips straddle the threshold to check it.

The lesson is the one the challenger spike ended on, now at prover scale: the arithmetic was never the problem. Every gain here came from submitting less often, and every setback from a fixed cost paid per dispatch rather than per unit of work.

- **A parallel fallback.** The per-chip CPU path was a serial loop where upstream's reduction is rayon-parallel, so every chip below the threshold was computed about ten times slower than if the seam had never been touched. This one was isolated by disabling the kernels entirely and finding the Metal build *still* slow, which is a diagnostic worth remembering.
- **A command buffer taken only when there is a dispatch for it.** The reduction took one up front and returned without committing it whenever every chip fell to the CPU. An uncommitted command buffer holds a slot in its queue, and a queue with no free slots blocks the next request for one, so a proof stalled after the first few dozen such rounds.

### Where it stands

The pieces are in place and correct: the seam, both kernels, both halves of the round, the per-chip decision, and parity at every level including layers whose chips straddle the threshold.
In isolation the round is **9.06x**.

### A measurement that was not measuring the prover

Several rounds of investigation here chased a phantom, and the record keeps them because the mistake is an easy one to repeat.

The integration appeared to hang, then to be merely slow, then to be killed. The exit status was **137** — SIGKILL — and the eventual cause was not the prover at all: the runs were invoking a copy of the binary in `/tmp`, made with `cp` from `target/release` while a `cargo build` was concurrently rewriting that file. On Apple Silicon a binary whose ad-hoc signature does not match its contents is killed by the kernel at exec, before `main`. The copy would not even print a usage error; the same binary run from `target/release` works.

So every "the Metal prover is slow" result from that stretch measured a corrupted file. Two lessons, in order of how much time each would have saved:

- **Check the exit status before theorising about performance.** 137 is not slow, and it was there from the first attempt.
- **Never benchmark a copy.** Run the artefact where the build system put it, or the thing under test is not the thing you built.

Real defects were found and fixed along the way and all of them stand — the fixes below are each backed by an isolated measurement, not by the integration runs. But no conclusion about integrated performance should be drawn from that stretch, and none is drawn here.

### The stall, and what it turned out to be

Run from `target/release`, the Metal-enabled prover used to **stall** — not run slowly, stop. A sample of the live process showed 36 threads, the main thread and sixteen tokio workers and the rayon pool, every one parked for the whole sampling window with no frame anywhere in `sp1_hypercube`, in this crate, or in Metal.

**The cause was an index, and it is fixed.** Upstream increments `interaction_offset` outside its `if let Some(inner)`, so a chip that contributes nothing still consumes its share of `eq_interaction`. This crate's loop skipped the increment when it skipped the chip, which misaligns the eq slice for every chip after the first empty one and eventually indexes past the end. A hand-built layer never exercises it, because its chips all have rows. A real one does.

With that fixed, the workload that had been hanging **proves and verifies in 15.17 s**, at the same threshold that stalled before.

Finding it took reading the CPU code against the kernel line by line rather than another experiment, after the threshold bisect had narrowed the failure to the dispatching path and the harness had ruled out size as the trigger. The parked-threads signature was a panic in a `spawn_blocking` task after all — swallowed, leaving the pipeline waiting on a sender that would never send.

### End to end

The same binary, back to back, on a core proof of a fibonacci guest:

| core proof, fibonacci guest | every layer on the CPU | the GPU path | |
|---|---|---|---|
| 200,000 iterations | 15.70 s, 15.83 s | 15.01 s, 15.23 s | 1.04x |
| 2,000,000 iterations | 64.69 s | 53.59 s | **1.21x** |

Both prove and verify at both sizes. **The Metal GKR round runs inside a real SP1 proof and the proof is accepted**, which is the thing steps 6 and 7 existed to establish.

The gain grows with the workload, which is the shape to expect and worth more than either number on its own: the smaller proof's layers mostly fall below the dispatch threshold and GKR is a small share of it, while the larger one produces layers big enough for the kernels to earn their dispatch. Neither is D7's workload, and neither is evidence for the model's 8.15x — that projection is about a compressed proof of a real lift, where GKR is 46.8% and the layers are shard-sized. What these do establish is that the port works end to end and that its benefit scales in the direction the model assumes.

Both runs shared the machine with an unrelated build, so the absolute times are not clean; they were taken back to back under the same conditions, so the ratio is.

It is specifically the GPU path. The **same binary**, with `SP1_METAL_MIN_ENTRIES` set high enough that every layer falls to the CPU, proves and verifies the same workload in **15.82 s** — indistinguishable from the build without the feature. So the fork, the feature, the seam, the runtime field dispatch and the CPU fallback are all correct and cost nothing measurable; what stalls is the part that dispatches to Metal.

That is a much smaller place to look than "somewhere in the integration", and the threshold makes it bisectable at runtime without rebuilding. At a threshold of 4,194,304 entries, where no layer in this workload reaches the GPU, the proof completes in 15.78 s; lower it and the stall returns.

Size alone is not the trigger. The harness in `crates/sp1-metal-prover` puts layers of seven million entries through the same kernels without trouble, and the parity tests cover mixed chip sizes and odd row counts. What the threshold gates is whether a *real* layer ever reaches the kernel, and real layers differ from constructed ones in ways the tests do not reach: `Padding::Generic` rather than a constant, chips whose `inner` is `None`, an `eq_row` shorter than the row pairs it is zipped against. The next test to write is one built from a layer the prover actually produced, not one assembled by hand.

What the diagnosis so far does establish is where to look. Every cause found was about *when work is submitted and what is held while it waits*, not about arithmetic — the kernels have been at their measured rates throughout.

Under `RUST_LOGGER=flat RUST_LOG=debug` the CPU build emits 211 span lines and finishes while the Metal build emits none, which looked like evidence of a stall and was not: the process was being killed before its buffered output was flushed. Silence from a killed process says nothing about progress, and the exit status says everything.

## 8. Zerocheck: the seam is easy, the kernel needs a build step

Read rather than assumed, because the first guess at this was wrong in both directions.

**The seam is cleaner than GKR's.** `zerocheck_sum_as_poly_in_last_variable` and `zerocheck_fix_last_variable` are already free functions in `prover/zerocheck/`, and the three `slop_sumcheck` backend impls just delegate to them. Factoring them behind a `ZerocheckOps` trait with a defaulted marker parameter is the same edit that worked for GKR, and smaller.

**The kernel behind it is the work, and it needs an offline build step.** The reduction's inner call is `increment_y_values`, which evaluates the chip's constraints through `air.eval(&mut ConstraintSumcheckFolder)` — a Rust function, per chip, at three interpolation points. There is no data-only formulation to port: a Metal zerocheck *is* the transpiler's generated kernels.

Two things make that a build step rather than a runtime one:

- **The transpiler cannot live in the fork.** It needs `sp1-core-machine` to enumerate chips and read their symbolic constraints, and `sp1-core-machine` depends on `sp1-hypercube`. Putting it inside would close a dependency cycle. The kernels have to be generated into a file the fork includes.
- **Compile time forces it anyway.** The spike measured eight minutes for the largest chip's kernel and 102 MB of generated Metal across all 119. That cannot happen at prover startup; it wants a Metal binary archive built ahead of time.

So step 8 is: generate three-point constraint kernels offline for the chips a workload uses, compile them into an archive, add the `ZerocheckOps` seam, and parity-check against `zerocheck_sum_as_poly_in_last_variable`. Each piece is understood; together they are days, not hours, and none of it is started.

## 9. Acceptance

The run D7 names is the ProofofSeed classical-128 lift, roughly 7.3 hours on the CPU path, and the `SP1_DUMP` pair that would reproduce it no longer exists on this machine; regenerating it needs that host run again, which is not this project's to start.
The harness for it is ready: `crates/sp1-profile-host` builds against the fork, takes `--features metal` to turn the GPU path on, proves and verifies, and now accepts `--stdin-u32` so a small guest can drive it. A fibonacci core proof through it is the standing smoke test — 18.2 s on the CPU path, and the number to beat.
