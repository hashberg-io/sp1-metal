# What the machine can actually do

Measured on the M4 Max (12 performance + 4 efficiency cores, 40-core GPU, 128 GB unified at 546 GB/s nominal) on 2026-08-21, with the probe suite described in [probes.md](probes.md).
Every GPU figure is a Metal kernel parity-checked against slop 6.4.0; every CPU figure is the packed (NEON) implementation, because that is what the SP1 CPU prover runs and beating scalar CPU code would prove nothing.
Numbers below are best-of-ten. The machine was not quite idle — a background test suite held one to two of the sixteen cores throughout — which understates the CPU side, and so flatters every ratio here, by something under ten percent.

## The two ratios everything else is made of

| | GPU | CPU (packed, all cores) | ratio |
|---|---|---|---|
| Streaming (triad) | 465 GB/s | 327 GB/s | **1.42×** |
| KoalaBear multiply, register-resident | 332 G/s | 107 G/s | **3.11×** |

These bracket the whole project.
Unified memory is the reason the first number is so low: the GPU and the CPU draw on the same pool, and the CPU's twelve performance cores can already saturate two thirds of it.
The second is set by `mulhi`, which the GPU issues at 996 Gop/s — one sixth of its 5.89 Top/s add rate — and a Montgomery multiply needs two of them plus two low multiplies.
The measured multiply rate matches that composition to within one percent, so there is no headroom in this formulation of the arithmetic.

Nothing in the prover can beat 3.11× except by exploiting something other than raw throughput, and anything memory-bound is capped near 1.42×.

## Per-kernel results

| kernel | GPU | CPU (packed) | ratio | shape |
|---|---|---|---|---|
| Poseidon2-16 permutation | 603 M/s | 37.6 M/s | **16.03×** | compute, latency-bound on CPU |
| zerocheck round | 420 M rows/s (329 G muls/s) | 87.6 M rows/s (69 G muls/s) | **4.79×** | compute |
| extension fix-last-variable | 459 GB/s | 148 GB/s | **3.10×** | GPU bandwidth, CPU compute |
| extension fold | 460 GB/s | 174 GB/s | **2.64×** | as above |
| extension multiply, streamed | 9.55 G/s (459 GB/s) | 3.71 G/s (178 GB/s) | **2.58×** | as above |
| base fix-last-variable | 388 GB/s | 299 GB/s | **1.30×** | bandwidth on both sides |
| base fold | 356 GB/s | 278 GB/s | **1.28×** | bandwidth on both sides |
| base multiply, streamed | 32.1 G/s (386 GB/s) | 26.4 G/s (317 GB/s) | **1.22×** | bandwidth on both sides |
| NTT butterfly stage | 441 GB/s | 400 GB/s | **1.10×** | bandwidth on both sides |

Two of these deserve comment.

The zerocheck kernel runs at 329 G multiplies per second against a measured ceiling of 332, so it is at its roofline and cannot be improved without changing the arithmetic.
Its CPU opponent reaches 69 of a possible 107, so the 4.79× is if anything generous to the GPU: a better CPU zerocheck would close it toward 3.1×.

Poseidon2's 16× is the one result that beats the compute ratio, and it does so because the CPU side is latency-bound rather than throughput-bound.
The permutation's cube S-box is two dependent multiplies, and the twenty partial rounds apply it to a single lane of the state; p3's packed permutation reaches only about a fifth of the CPU's multiply roofline, while the kernel reaches over half of the GPU's by running thousands of permutations concurrently.
That gap is real today — it is what SP1's CPU prover does — but it is soft: a CPU implementation that interleaved several independent permutations to fill the vector pipelines would take a large bite out of it.

## Fixed costs

| | |
|---|---|
| extra dispatch in the same command buffer | 1.1 µs |
| command buffer of its own, committed and waited | 86 µs |

Sumcheck and BaseFold rounds halve their working set each round, and each round's challenge is derived from the previous round's output, so a straightforward port pays a full round trip per round.
Folding 2²⁴ elements down to a scalar costs 2.90 ms that way, of which about 2.0 ms is round trips; the same twenty-four rounds batched into one submission take 0.90 ms.
That 3.2× — 4.7× on a quieter machine — is what a GPU-side challenger would be worth, and it is a design constraint rather than an optimisation: on small rounds the fixed cost is the whole cost.

## The Reed-Solomon encode

Committing a trace is not hashing it. slop encodes each trace with a Reed-Solomon code — a radix-2 DFT at the FRI blowup of 2 — and builds the Merkle tree over the codeword, and the encode is the larger half: slop's CPU encoder spends 7.5 ns per input cell against 6.7 ns for the hashing the codeword then needs, so the phase is about 53% encode.

There is no Metal encode yet, so the encode is bounded rather than measured.
A DFT is a sequence of butterfly stages, and one stage — read a pair, one multiply, two adds, write the pair back — runs at 441 GB/s on the GPU against 400 GB/s in packed CPU lanes, a ratio of **1.10×**: both sides are moving memory, and this is the floor for an implementation that keeps nothing resident across stages.
The ceiling is the register multiply roofline, 3.11×, for one that blocks perfectly.
Mixing those with the hashing ratio puts the commit phase between **1.96× and 5.00×**.

One thing that measurement makes plain: slop's CPU encoder reaches about 42 GB/s per pass, against the 400 GB/s a hand-written packed stage manages. Comparing a Metal encode against slop's encoder as it ships would show a large ratio, and almost all of it would be CPU headroom rather than GPU advantage.

## Where the time goes

Profiled by replaying the ProofofSeed snark-lifter's compressed proof at n=8 through the span-timing host: 1007 s wall, proved and verified, 4578 s of span time over 105 span paths.

| phase | seconds | share |
|---|---|---|
| GKR (logup) | 1872 | **40.9%** |
| PCS opening (jagged / BaseFold) | 761 | 16.6% |
| zerocheck | 600 | 13.1% |
| execution | 583 | 12.7% |
| commit (encode + Merkle) | 546 | 11.9% |
| recursion scaffolding | 88 | 1.9% |
| trace generation | 75 | 1.6% |
| setup | 48 | 1.0% |
| verify, unattributed | 6 | 0.1% |

The recursion layers of a compressed proof are shard proofs themselves, so their internals land in the same buckets; the "recursion scaffolding" row is only the work of building and setting up the recursion programs.

The largest phase by a distance is the logup GKR argument, and it is the one the suite prices worst: no probe reproduces its shape, so its factor can only be bracketed between the streamed extension multiply (2.58×) and the zerocheck round (4.79×), whose rounds it resembles.

Two caveats on reading this table.
Span busy times are wall times on the thread that entered the span, and several shards prove concurrently, so these are phase shares of prover work only to the extent that every phase parallelises about equally.
And the profile was taken with one to two foreign cores busy on the machine, which flatters the GPU side by something under ten percent.

## What this means for the target

Feeding the phase shares and the per-phase factors into the model:

| | conservative | optimistic |
|---|---|---|
| serial port (GPU phases compress, CPU phases unchanged) | **1.83×** | 2.86× |
| pipelined (CPU-resident work overlapped across shards) | **2.50×** | **4.89×** |

[Desiderata](desiderata.md) D7 asks for 5× end to end. The optimistic column does not reach it, and the optimistic column assumes every bracketed factor lands at its upper end, the pipelining is perfect, and the port is complete — every phase moved, including the constraint transpiler that has not been designed.

The reason is not a weak GPU or a bad kernel. The zerocheck kernel runs at 99% of the machine's measured multiply roofline; there is nothing left to extract from it. The reason is the machine: a 40-core GPU and twelve NEON performance cores are only 3.1× apart on field multiplies and 1.4× apart on memory, because they share one pool of it. On a discrete GPU with its own 1 TB/s of memory both those numbers would be several times larger, and the same port would clear 5× comfortably. On this machine it cannot.

What would move the numbers, in order of leverage:

1. **Price GKR properly.** It is 41% of the total and its factor spans 2.58× to 4.79×; that range alone is most of the gap between the two columns above. A probe reproducing a GKR layer's sumcheck round would settle it.
2. **Write the Metal encode.** It is worth up to 3× on 12% of the total, and the butterfly stage measurement says the memory-bound floor is only 1.10×, so the answer depends entirely on how well a blocked implementation holds a working set in threadgroup memory.
3. **Derive challenges on the GPU.** At 86 µs per command-buffer round trip, the sumcheck and BaseFold tails are latency-bound; batching the rounds of one fold into a single submission was 3–5× faster in the probe.
4. **Do not port the base-field folds.** At 1.28× and 1.30× they are not worth the transfer, and PCS opening is 17% of the total spent mostly on them.

The honest reading of D7 on this hardware is that 5× is out of reach and 2.5–3× is the realistic target for a complete port. Whether that is worth the work is a judgement about what the prover is for, not a measurement.
