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
