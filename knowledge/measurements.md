# What the machine can actually do

Measured on the M4 Max (12 performance + 4 efficiency cores, 40-core GPU, 128 GB unified at 546 GB/s nominal) on 2026-08-21, with the probe suite described in [probes.md](probes.md).
Every GPU figure is a Metal kernel parity-checked against slop 6.4.0; every CPU figure is the packed (NEON) implementation, because that is what the SP1 CPU prover runs and beating scalar CPU code would prove nothing.
Numbers below are best-of-ten on an otherwise idle machine except for one background test process holding roughly one core, which costs the CPU side something under ten percent.

## The two ratios everything else is made of

| | GPU | CPU (packed, all cores) | ratio |
|---|---|---|---|
| Streaming (triad) | 466 GB/s | 324 GB/s | **1.44×** |
| KoalaBear multiply, register-resident | 332 G/s | 106 G/s | **3.12×** |

These bracket the whole project.
Unified memory is the reason the first number is so low: the GPU and the CPU draw on the same pool, and the CPU's twelve performance cores can already saturate two thirds of it.
The second is set by `mulhi`, which the GPU issues at 996 Gop/s — one sixth of its 5.9 Top/s add rate — and a Montgomery multiply needs two of them plus two low multiplies.
The measured multiply rate matches that composition to within one percent, so there is no headroom in this formulation of the arithmetic.

Nothing in the prover can beat 3.12× except by exploiting something other than raw throughput, and anything memory-bound is capped near 1.44×.

## Per-kernel results

| kernel | GPU | CPU (packed) | ratio | shape |
|---|---|---|---|---|
| Poseidon2-16 permutation | 604 M/s | 40.6 M/s | **14.85×** | compute, latency-bound on CPU |
| zerocheck round | 420 M rows/s (329 G muls/s) | 94.7 M rows/s (74 G muls/s) | **4.44×** | compute |
| extension multiply, streamed | 9.56 G/s | 3.95 G/s | **2.42×** | GPU bandwidth, CPU compute |
| extension fold | 456 GB/s | 167 GB/s | **2.73×** | as above |
| extension fix-last-variable | 459 GB/s | 156 GB/s | **2.94×** | as above |
| base fold | 353 GB/s | 322 GB/s | **1.10×** | bandwidth on both sides |
| base fix-last-variable | 386 GB/s | 370 GB/s | **1.04×** | bandwidth on both sides |
| base multiply, streamed | 32.2 G/s (386 GB/s) | 25.6 G/s (307 GB/s) | **1.26×** | bandwidth on both sides |

Two of these deserve comment.

The zerocheck kernel runs at 329 G multiplies per second against a measured ceiling of 332, so it is at its roofline and cannot be improved without changing the arithmetic.
Its CPU opponent reaches 74 of a possible 106, so the 4.44× is if anything generous to the GPU: a better CPU zerocheck would close it toward 3.1×.

Poseidon2's 14.85× is the one result that beats the compute ratio, and it does so because the CPU side is latency-bound rather than throughput-bound.
The permutation's cube S-box is two dependent multiplies, and the twenty partial rounds apply it to a single lane of the state; p3's packed permutation reaches only about a fifth of the CPU's multiply roofline, while the kernel reaches over half of the GPU's by running thousands of permutations concurrently.
That gap is real today — it is what SP1's CPU prover does — but it is soft: a CPU implementation that interleaved several independent permutations to fill the vector pipelines would take a large bite out of it.

## Fixed costs

| | |
|---|---|
| extra dispatch in the same command buffer | 0.93 µs |
| command buffer of its own, committed and waited | 84 µs |

Sumcheck and BaseFold rounds halve their working set each round, and each round's challenge is derived from the previous round's output, so a straightforward port pays a full round trip per round.
Folding 2²⁴ elements down to a scalar costs 2.89 ms that way, of which about 2.0 ms is round trips; the same twenty-four rounds batched into one submission take 0.61 ms.
That 4.7× is what a GPU-side challenger would be worth, and it is a design constraint rather than an optimisation: on small rounds the fixed cost is the whole cost.

## Where the time goes

<!-- filled in from the span profile -->

## What this means for the target

<!-- filled in from the model -->
