# Keeping the data on the device

A plan for the next stage of the Metal backend, written against measurements rather than intentions.
Every number below comes from one paired run of the acceptance workload — fibonacci n = 2,000,000, about 18M cycles, `SHARD_SIZE` 4 MiB, five shards, four proving threads — with each switch set together so the CPU arm really is one.

## Where it stands

| phase | CPU | Metal | | | span | CPU | Metal | |
|---|---:|---:|---:|---|---|---:|---:|---:|
| gkr | 78.4 | 28.6 | 2.74x | | prove GKR circuit | 66.5 | 19.1 | 3.48x |
| pcs | 34.5 | 13.0 | 2.64x | | commit traces | 34.3 | 8.8 | 3.89x |
| commit | 34.3 | 8.8 | 3.89x | | Dense PCS evaluation proof | 20.6 | 5.3 | 3.89x |
| zerocheck | 18.3 | 10.8 | 1.69x | | zerocheck | 18.3 | 10.8 | 1.69x |
| setup | 10.0 | 1.9 | 5.26x | | **generate GKR circuit** | 11.0 | 9.0 | **1.22x** |
| tracegen | 3.3 | 1.8 | 1.85x | | **create jagged sumcheck poly** | 2.5 | 2.5 | **1.01x** |
| **total** | **179.0** | **65.4** | **2.74x** | | setup_from_preprocessed | 8.3 | 0.9 | 9.57x |

Thread-seconds, exclusive. Wall: **64.16 s against 24.63 s, 2.60x**. Peak resident memory: **58.1 GB against 72.4 GB**.

## The three numbers this plan exists for

**The GPU is busy 1.05 seconds out of 24.6 — 4.3%.** Everything above was won by making a stage faster while it is running; almost nothing has been won by making the machine busier. There is an order of magnitude of idle silicon.

**163.45 GB is uploaded to feed those 1.05 seconds.** Copying costs 4.85 thread-seconds, 4.6x what all the arithmetic costs. Every kernel here takes host-resident data and hands back host-resident data, so between two GPU stages the data makes a round trip that neither stage needed.

**Peak memory goes up 24%, not down.** On unified memory a `StorageModeShared` buffer is process pages, so a value staged for the GPU exists twice: once in the `Vec` that owns it and once in the buffer that reads it. At 72 GB of 128 GB this already bounds how large a shard can be, and it is a direct consequence of round-tripping rather than of the GPU needing the memory.

And the tell that ties them together: the two spans that barely moved, `generate GKR circuit` at 1.22x and `create jagged sumcheck poly` at 1.01x, are both *producers*. They build an array that a kernel immediately reads. They are slow for the same reason the uploads are large — the pipeline is a sequence of host-resident values with GPU stages between them, rather than device-resident values with host stages between them.

## The invariant to establish

> A value produced on the device is consumed on the device. The host sees challenges, sizes, and the finished proof — not traces, not layers, not codewords.

That is the whole idea. Everything below is bookkeeping in service of it.

## What we already have

`MetalBackend` (`crates/sp1-metal-probes/src/backend.rs`) is a slop `Backend`: it implements `Allocator`, `DeviceMemory`, `Backend` and `GlobalBackend` over page-aligned no-copy buffers, and `buffer_for(ptr)` maps any allocation back to its Metal buffer. So `Mle<T, MetalBackend>`, `PaddedMle<T, MetalBackend>` and `Point<T, MetalBackend>` already type-check today, and slop's sumcheck driver is already generic over the backend. The type system is not the obstacle. The obstacle is a handful of places where upstream pinned `CpuBackend` into a signature.

## What blocks it, precisely

| pinned to `CpuBackend` | where | consequence |
|---|---|---|
| `ToMle::interleaved_mles() -> Message<Mle<F, CpuBackend>>` | `slop-multilinear/src/pcs.rs:99` | committed data can never be device-resident |
| `BasefoldProver<GC, P: ComputeTcsOpenings<GC, CpuBackend>>` | `slop-basefold-prover/src/prover.rs:52` | the FRI half is host-bound |
| `prove_trusted_evaluations(.., Rounds<Evaluations<GC::EF>>, ..)` | `slop-jagged/src/prover.rs:165` | the opening's claims are host-bound |

Two of these are already reachable: `slop-jagged` is vendored, and `sp1-hypercube` owns `LogupGkrCpuCircuit`, `ZeroCheckPoly` and the shard prover outright. `TensorCsProver` is *already* generic over the backend (`slop-merkle-tree/src/tcs.rs:15`), so the commitment needs nothing.

## Order of work

Each stage is separately measurable, separately revertible, and leaves the tree green. No stage begins before the one before it has moved a number.

**0. An instrument for residency.** Extend the existing counters to report, per phase: bytes uploaded, bytes written back, and the fraction of kernel inputs that were already on the device. Today's `STATS` counts uploads but cannot say what a *residency* change did. This is the only stage with no performance goal, and skipping it means flying blind through the rest — the same mistake that cost this project a wrong diagnosis twice.

**1. GKR end to end.** The largest phase at 28.6 thread-seconds, and the fork owns every type involved. Make `generate_gkr_circuit` allocate its layers in `MetalBackend`, so the round reduction, the fold and the layer transition all read data that is already there. This alone should remove most of the 163 GB, since the GKR rounds are where the uploads are. Expect `generate GKR circuit` to move off 1.22x, which is the check that residency is doing what is claimed.

**2. Zerocheck columns.** `ZeroCheckPoly` is ours. Its columns are folded every round and read by the generated constraint kernels; keeping them resident removes a round trip per round. Zerocheck is the worst ratio on the board at 1.69x, and this is why.

**3. The commitment's input.** The Merkle tree already accepts whatever `TensorCsProver` is given, so this is only about where the codeword lives. The encode stays on the CPU deliberately — 1.02x against a packed baseline, nothing to win — which means one upload per shard remains by choice. Decide then whether that upload is worth removing by vendoring `slop-basefold-prover`, or whether it is small enough to keep.

**4. The opening.** Needs `Evaluations` and `ToMle` generalised, which means vendoring `slop-multilinear` on top of the `slop-jagged` already vendored. The largest new dependency in the plan, and the one to do last, when the pattern is established and the payoff is known from stages 1 and 2.

**5. Re-acceptance.** The same three axes, and a fresh peak-memory figure, which should now be *below* the CPU path rather than 24% above it.

## How the risk is bounded

Every vendored change follows the pattern that has worked twice here: a backend type parameter defaulted to `CpuBackend`, so upstream's own tests compile and pass unchanged and the CPU path stays bit-identical. `crates/vendor/*/CHANGES.md` records each delta and why. Reverting any stage is deleting a `[patch.crates-io]` entry.

Correctness keeps the two independent checks it has now: parity against the CPU implementation for each kernel, and a proof that verifies. Residency changes *where* data lives, not what is computed, so a parity failure means a real bug rather than a tolerance question.

The one genuinely new risk is lifetime: a device-resident value whose host allocation is freed underneath it. The existing upload cache already solves this by pinning an `Arc` alongside the buffer, and `MetalBackend` owns its allocations outright, which is stronger. Prefer ownership to caching wherever the choice exists.

## What would say it worked

| | now | target |
|---|---:|---:|
| GPU busy, fraction of wall | 4.3% | > 30% |
| bytes uploaded per proof | 163.45 GB | < 20 GB |
| peak resident memory | 72.4 GB (+24% vs CPU) | <= CPU path |
| `generate GKR circuit` | 1.22x | > 3x |
| wall, acceptance workload | 2.60x | >= 4x |

## What would say it cannot work

If, after stage 1, the GPU busy fraction has not moved and the uploaded bytes have not fallen, then the cost is not where this plan says it is, and the plan should be abandoned rather than continued into stages 2-4. The measurement in stage 0 exists to make that call cheap.

The honest ceiling: the phases that stay on the CPU by choice — trace generation, the Reed-Solomon encode, the scalar tails of every sumcheck — are about 10 thread-seconds together. If residency removed *everything* else, the wall would be bounded near 5 seconds, which is where D7's 5x lives. That is the case for doing this, and also the reason not to expect more than it.
