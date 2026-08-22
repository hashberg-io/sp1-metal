# Building the backend

[verdict.md](verdict.md) concluded that a Metal backend is not worth building on this hardware, on the strength of a predicted 3.3× against a bar of 5×.
That prediction measured the GPU against a *packed* CPU baseline, because that is the fair comparison between two pieces of silicon.
[D7](desiderata.md) does not ask for the fair comparison. It asks for 5× "the same machine's CPU prover", and the prover on this machine is scalar.

Against the prover that actually ships, a complete port predicts **6.5×**, and every corner of the sensitivity band stays above 5.
This document is the plan if that is the bar you want to clear: what the number rests on, where the code seam is, what is not yet known, and in what order to find out.

## Which baseline

The two numbers answer different questions and both are right.

| | vs packed CPU | vs shipped CPU |
|---|---|---|
| logup GKR | 6.40× | 20.9× |
| zerocheck | 4.48× | 12.3× |
| PCS opening | 2.86× | 8.6× |
| commit | 2.96× | 14.2× |
| **complete port** | **3.6×** | **8.2×** |

The right-hand column is the left multiplied by the packing headroom the shipped prover leaves unclaimed: 3.26× on the GKR round and 3.0× on the folds, both measured in [measurements.md](measurements.md), and 2.75× assumed for zerocheck by analogy with the folds, which is the one factor here with no measurement behind it.
The commit row is measured on both sides rather than derived — see [spikes.md](spikes.md) — and its two columns differ by 4.8× because slop's Reed-Solomon encoder is that much slower than the same transform written packed.
Phase shares are the profiled ones: GKR 46.8%, PCS 19.0%, zerocheck 15.0%, commit 13.7%, everything else 5.5%.

Sensitivity, on the complete port against the shipped prover:

| | |
|---|---|
| PCS at a pessimistic 3× | 6.2× |
| zerocheck given no packing credit at all | 7.0× |
| commit credited only its hardware ratio (2.96×) | 6.3× |
| every ported phase free | 18.2× |

The bar survives all of them, which is the main reason this plan is worth writing.

**The load-bearing assumption is that nobody packs SP1's CPU prover underneath you.**
Every one of these numbers is a ratio against unvectorised loops, and roughly half of the win would evaporate the week upstream reaches for `KoalaBear::Packing` on the GKR round.
That has not happened through 6.x, and the `Cpu` suffixes throughout the prover suggest upstream's accelerator attention is on CUDA rather than NEON, so it is a reasonable bet.
It is still a bet, and it belongs at the top of the risk register rather than buried in a footnote.

## There is no partial path

Compressing one phase at a time, against the shipped prover:

| after porting | end to end |
|---|---|
| GKR | 1.80× |
| + zerocheck | 2.40× |
| + commit | 3.46× |
| + PCS opening | 8.24× |

Nothing before the last step clears the bar, because whatever remains unported becomes the whole cost.
D7 is a single indivisible bet, which is also what the house rule against splitting milestones would force.
The consequence for planning is that the work should not be sequenced for early payoff — there is none to be had — but to retire risk as early as possible, so that the decision to abandon can be taken cheaply if it is going to be taken at all.

## The seam

Read against slop and `sp1-hypercube` 6.4.0.

**slop is genuinely backend-generic.**
`Backend` (slop-alloc, `src/backend/mod.rs:16`) is one small unsafe trait over `Allocator + DeviceMemory + Clone + Debug + Send + Sync`, and everything above it — `Buffer`, `Mle`, `Point`, `Tensor`, `Evaluations` — carries `A: Backend = CpuBackend`.
The sumcheck driver dispatches entirely through blanket impls of three traits in slop-sumcheck `src/backend.rs`: `ComponentPolyEvalBackend` (line 9), `SumCheckPolyFirstRoundBackend` (line 32) and `SumcheckPolyBackend` (line 72).
Implement those for a device polynomial type and `prove_sumcheck` works unchanged.
This is the designed extension point, and zerocheck already uses it — `impl SumcheckPolyBackend<ZeroCheckPoly<..>, EF> for CpuBackend` is how the CPU path is wired.

**The PCS is a real generic parameter.**
`ShardProver<GC, SC, C: MultilinearPcsProver<..>>` (`src/prover/shard.rs:184`) takes the PCS prover as a type argument, so commit and opening — 33% of the proof — can be replaced against unmodified upstream.
That makes them the piece of this project that survives version bumps, and a good place to start for that reason alone.

**Unified memory makes integration nearly free, and this is the one place Apple's architecture helps.**
`MTLResourceStorageModeShared` buffers are CPU/GPU coherent, and `newBufferWithBytesNoCopy` over page-aligned allocations lets `Buffer<T, MetalBackend>` alias host memory outright.
`copy_to_device` becomes a no-op, `ToHost` is free, and CPU and GPU phases can share a buffer part-way through a proof with no staging, no PCIe, and no pinned-memory bookkeeping.
The property that caps throughput at 1.44× on streaming kernels is the same property that removes the transfer engineering a CUDA backend has to do.

**GKR and zerocheck are not behind the seam.**
`ShardProverInner` (`src/prover/shard.rs:169`) holds a concrete `GkrProverImpl<GC, SC>`, which holds a concrete `LogupGkrCpuTraceGenerator` (`src/logup_gkr/prover.rs:18`), and the round prover `prove_gkr_round` is a `pub(crate)` free function over CPU layer types (`src/logup_gkr/cpu.rs:151`).
`ZeroCheckPoly` declares `type Backend = CpuBackend` outright (`src/prover/zerocheck/mod.rs:174`).
There is no trait to implement for either.
Between them they are 62% of the proof, so a Metal backend means vendoring `sp1-hypercube` and doing the factoring by hand.
The naming throughout — `LogupGkrCpuCircuit`, `LogUpGkrCpuLayer`, `logup_gkr/cpu.rs` — says upstream intends that factoring eventually, so it is worth offering back rather than keeping.

Trace generation stays on the CPU regardless: `ShardProverInner` pins `DefaultTraceGenerator<_, _, CpuBackend>` and `MainTraceData<_, _, CpuBackend>`.
At 1.9% of the proof this costs nothing, and unified memory means the traces it produces need no copy to be read by a kernel.

## Three unknowns

Each can be answered by a spike against the existing probe harness, before any backend code is written, and each can sink the project on its own.

**A GPU-side challenger.** *(Answered — see [spikes.md](spikes.md): not blocking the host matters more than moving the sponge does.)*
A command buffer of its own costs 87 µs, sumcheck and BaseFold rounds halve every round, and each round's challenge is derived from the previous round's output — so a naive port pays a full round trip per round and the tails become the whole cost.
The `launch` probe already measured the size of this: folding 2²⁴ elements one round trip at a time takes 2.88 ms, of which 2.2 ms is round trips, against 0.63 ms for the same rounds in a single submission.
Closing it means a Poseidon2 duplex sponge and Fiat-Shamir running on the device, so challenges never touch the host.
This is a design constraint on the shape of every other kernel rather than an optimisation to add afterwards, which is why it goes first: retrofitting it is a rewrite.

**A blocked Reed-Solomon encode.** *(Answered — see [spikes.md](spikes.md): 1.72× over a packed CPU, and at the bandwidth roofline.)*
The commit phase is 13.7% and its factor spans 1.97× to 5.13×, the widest bracket in the model, because only the DFT's butterfly stage has been measured and not a real implementation.
One stage in isolation runs at 1.11× because both sides are moving memory; a version that holds a block in threadgroup memory across stages moves toward the 3.21× multiply roofline.
Which of those it lands on is entirely a question of residency, and it has to match slop's bit-reversed convention to be usable.

**A constraint transpiler.**
Zerocheck is 15% of the proof and the probe suite deliberately priced only the *shape* of constraint evaluation, never a real chip's constraints.
Turning `SymbolicExpression` into something a GPU runs is the genuine unknown-unknown of this project: either MSL codegen per chip, or a bytecode tape walked by an interpreter kernel.
SP1 has on the order of thirty chips with very different constraint sets, so the spike should take the two largest and find out whether register pressure or branch divergence kills the approach before the other twenty-eight are attempted.

Rough sizing, and these are guesses rather than estimates: two weeks each for the challenger and the encode, two to three for the transpiler spike.

## The order of work

Each step carries a note on whether it exercises the machine, and how.
This matters because the same M4 Max runs long CPU proving jobs, and some of these steps can share it while others cannot.

1. **GPU-side challenger spike.**
   A Poseidon2 duplex sponge and Fiat-Shamir on the device, plus a sumcheck round loop that batches into one submission instead of one command buffer per round.
   Goes first because it constrains the shape of every other kernel.
   *Machine: yes, GPU only.* Seconds per run, and the CPU side of the comparison is already measured, so it does not need a quiet machine.
   **Done**, and it revised its own premise: not blocking the host is worth more than moving the sponge, and the two together are worth 3.35x rather than the 4.5x assumed here. See [spikes.md](spikes.md).
2. **Blocked Reed-Solomon encode spike.**
   Bit-reversed, matching slop's convention, holding a block in threadgroup memory across stages.
   *Machine: yes, GPU and CPU.* Roughly 1 GiB working sets and seconds per run, but the packed-CPU baseline is invalidated by a competing job, so this one wants the machine idle.
   **Done**, and the bracket collapses to 1.72× against a packed CPU and 13.1× against slop's own encoder. See [spikes.md](spikes.md).
3. **Constraint transpiler spike.**
   `SymbolicExpression` to MSL codegen or to a bytecode tape, tried on the two largest chips.
   *Machine: barely.* Mostly reading `sp1-core-machine` and writing codegen, ending with one GPU run to see whether register pressure or branch divergence kills the approach.
4. **Refit the model and take the go/no-go.**
   Three measured numbers replace three brackets in `tools/fit_model.py`.
   *Machine: no.* Seconds of Python against the phase profile already on disk.
5. **`MetalBackend`.**
   Allocator over page-aligned no-copy buffers, `DeviceMemory`, `GlobalBackend`.
   The device and pipeline plumbing already exists in `crates/sp1-metal-probes/src/gpu.rs`.
   *Machine: no*, beyond compiles and unit tests.
6. **PCS opening and commit**, through the `C: MultilinearPcsProver` type parameter.
   Ships against unmodified upstream, which makes it the piece that survives version bumps.
   *Machine: yes, moderate.* The first step that builds against the full sp1-hypercube dependency tree, and it needs integration proofs rather than parity checks alone.
7. **Vendor `sp1-hypercube`**, factor the GKR prover behind a trait mirroring the sumcheck backend pattern, and implement Metal GKR.
   The largest phase and the largest diff.
   *Machine: yes.* Heavy repeated compiles of a vendored fork, plus end-to-end shard proofs to validate each layer.
8. **Zerocheck.**
   A device polynomial type implementing the three sumcheck backend traits, plus the transpiler extended across all thirty-odd chips.
   *Machine: yes.* Parity runs per chip, individually cheap and collectively a long loop.
9. **Acceptance** on D7's canonical workload, the ProofofSeed classical-128 lift, measured with `tools/span_profile.py` against the CPU run already profiled.
   *Machine: yes, heavily.* The CPU reference took about 7.3 hours uncapped, so this wants the machine to itself for the better part of a day, and again once the first result raises questions.

Steps 4 and 5 are the only ones that can proceed while a proving job holds the machine.
Step 2 is the one that needs it genuinely idle, since a busy machine understates the CPU baseline and flatters the kernel.
Step 9 needs it exclusively.

Call it three to four months of one person's attention after the spikes, with the transpiler holding most of the variance.

The existing work carries over almost entirely.
The Metal kernels for Poseidon2, the GKR round, the zerocheck round, both folds, fix-last-variable and the NTT butterfly exist and are parity-checked against slop 6.4.0; the probe suite becomes the regression harness that keeps them honest; and the span profiler is the acceptance instrument.

## What would stop this

- **A discrete GPU is a better answer to the same question.** D7's own stretch goal is to come within 5× of a single RTX 4090, and the projection in [measurements.md](measurements.md) puts a discrete card at roughly 10× using SP1's existing CUDA prover. If the goal is faster proofs rather than a Metal backend, that hardware wins without any of this work.
- **Upstream may do the factoring first.** If `sp1-hypercube` puts GKR behind a backend trait in a later release, step 5 stops being a fork and the project gets materially cheaper. Check before starting.
- **Upstream may pack the CPU prover.** That is the assumption the whole plan rests on, and it would take the projection from 6.5× back to 3.7×.
- **Maintenance.** A vendored fork of the hottest files in a fast-moving repository, pinned to `=6.4.0`, is a standing cost that does not appear anywhere in the three-to-four-month figure.
