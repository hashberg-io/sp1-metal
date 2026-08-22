# Spike results

[backend-plan.md](backend-plan.md) names three unknowns that can each sink a Metal backend, and says to answer them before writing any of it.
This is what they found, in the order the plan runs them.
Method and hardware are as in [measurements.md](measurements.md): M4 Max, best-of-ten, machine otherwise idle, every kernel parity-checked against slop 6.4.0.

## 1. The GPU-side challenger

**Question.** A command buffer of its own costs 98 us, sumcheck and BaseFold rounds halve every round, and each round's challenge comes from the previous round's output.
The plan assumed the fix was to run the duplex sponge on the device so challenges never reach the host, and priced it from the launch probe's 4.5x for a fold with the challenges precomputed.

**What was built.** The sponge SP1 actually uses — `DuplexChallenger<KoalaBear, KoalaPerm, 16, 8>`, the challenger of `KoalaBearDegree4Duplex` — as a Metal kernel over the width-16 Poseidon2 permutation the suite already had.
It is parity-checked against slop's challenger on a scripted sequence of observes and samples that crosses the rate boundary in both directions, and then again end to end: a complete BaseFold-shaped fold, message and challenge and fold per round, reproducing slop's challenges round by round and its final value.
The refusal test was to sample from the front of the sponge's output buffer instead of the back, which is the single easiest detail of p3's sponge to get wrong; it failed the parity check at the first sample and was then restored.

**Answer.** Not blocking is worth more than moving the sponge, and costs nothing to build.

| fold of 2^22 extension elements, 22 rounds | per fold | per round | |
|---|---|---|---|
| host challenger, one fold at a time | 4.12 ms | 187 us | 1.00x |
| device challenger, one fold at a time | 1.84 ms | 84 us | 2.24x |
| host challenger, four folds in flight | 1.66 ms | 75 us | **2.49x** |
| device challenger, four folds in flight | 1.23 ms | 56 us | **3.35x** |

The two are complementary, so the device sponge is still worth 1.35x once the host has stopped blocking, and worth building.
But it is not the load-bearing item the plan took it for: the first and much larger win is simply committing several folds before waiting on any of them, which is a scheduling change rather than a kernel.
The prover has the concurrency to spend — 79 shard proofs went through a queue that never drained, and each shard opens many polynomials — so this is available in the real workload and not only in the probe.

**Why the sponge is worth so much less than expected.** One Poseidon2 permutation on a single GPU thread costs **44.5 us**.
The same permutation costs 26 ns on a CPU core and 1.65 ns amortised when two million of them run concurrently on this GPU.
A duplex sponge is inherently sequential, so it gets none of that: it runs on one lane, with no other work in flight to hide the latency of dependent multiplies, and a GPU lane is a poor sequential processor.
The gap between one lane and the whole device is a factor of about 27000, and it is the reason a device-side challenger can never be free.
Moving the sponge state from device memory into registers changed nothing measurable, so the cost is the dependent arithmetic itself rather than memory traffic.

The obvious remaining move is to spread one permutation across sixteen lanes of a SIMD group, using shuffles for the linear layers instead of running all sixteen state elements on one lane.
That would cut the 44.5 us by perhaps four or five times, and with it most of what still separates the third and fourth rows above.
It has not been built.

**What this changes in the plan.** Step 1 stands, but its content changes: the constraint on every other kernel is that the host must not block, which is a property of how work is submitted, and the device sponge is a smaller optimisation layered on top.
The 4.5x the plan quoted from the launch probe is not reachable in this shape: that measurement used precomputed challenges, base-field data and no reduction pass, and the realistic ceiling with the reduction included is the 3.35x above.

## 2. The blocked Reed-Solomon encode

**Question.** The commit phase is 13.7% of the proof and its factor spanned 1.97x to 5.13x, the widest bracket in the model, because only the DFT's butterfly stage had been measured and not an implementation.
The bracket's width came from not knowing whether a blocked encode would reach the multiply roofline or stay at the memory floor.

**What was built.** A complete Metal encode, matching slop's output element for element at two shapes — one small enough that every stage is a global pass, one large enough to exercise the tiled tail.
Two observations made it much simpler than expected.
slop's encode is not a coset LDE: `Dft::dft(src, log_blowup, BitReversed, 0)` zero-pads the trace to `rows << log_blowup` and takes a plain forward DFT of each column, then bit-reverses the rows.
A decimation-in-frequency transform takes natural-order input to bit-reversed output directly, so matching slop's convention costs **no permutation at all** — the two bit-reversals p3 performs cancel against the ordering slop then asks for.
And at a blowup of two the first stage's partners are all zero, so it degenerates to `u` and `u * w^j`: the encode reads the trace once and never touches the padding.
The last six stages run while a 64-row by 64-column tile stays in threadgroup memory, which turns 21 stages into 16 passes over the codeword.
The refusal test corrupted the twiddle stride of the tiled stages only; it failed the shape that uses the tail and passed the shape that does not, then was restored.

**Answer.** 1.72x, and it is at the roofline.

| encode of a 2^20 by 64 trace at blowup 2 | | per input cell | |
|---|---|---|---|
| slop's own encoder (`Radix2DitParallel`) | 454.8 ms | 6.78 ns | 1.00x |
| the same transform in packed NEON lanes | 59.8 ms | 0.89 ns | 7.61x |
| Metal | 34.7 ms | 0.52 ns | **13.10x** |

The kernel moves 495 GB/s across its sixteen passes, which is above the 466 GB/s the triad probe measures, so there is nothing left in it.
The packed CPU encoder moves 274 GB/s over the same passes.
**The ratio is set by bandwidth and nothing else**, which also means further blocking cannot move it: cutting the pass count speeds both sides equally, and even at four passes the transform is still memory-bound rather than compute-bound, so it never approaches the 3.21x multiply ratio.
The 1.97x-to-5.13x bracket is replaced by a measurement of **2.96x** for the commit phase as a whole, mixing this encode with the Poseidon2 Merkle hashing by their measured time share.

**What this changes in the plan.** One assumption in it was wrong, in the direction that helps.
The plan gave commit the same factor in both baseline columns, reasoning that `p3-dft` already runs packed so there was no CPU headroom to claim there.
There is: slop's encoder is **7.6x** slower than the same transform written packed, most of it the two bit-reversal permutations the decimation-in-frequency convention makes unnecessary, and none of it anything to do with the GPU.
So against the prover as it ships the commit phase is worth 14.2x rather than 3.2x, and the complete port moves from 6.5x to **8.2x**; against a packed CPU it moves the other way, from 3.7x to **3.6x**.
That is the same pattern this project keeps finding, once more: almost all of the apparent GPU advantage in this phase is CPU work left undone.
