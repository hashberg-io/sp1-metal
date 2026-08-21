# What the machine can actually do

Measured on the M4 Max (12 performance + 4 efficiency cores, 40-core GPU, 128 GB unified at 546 GB/s nominal) on 2026-08-21, with the probe suite described in [probes.md](probes.md).
Every GPU figure is a Metal kernel parity-checked against slop 6.4.0; every CPU figure is the packed (NEON) implementation, because that is what the SP1 CPU prover runs and beating scalar CPU code would prove nothing.
Numbers below are best-of-ten, from a single `probes all --cpu` run. The machine was not quite idle — a background test suite held one of the sixteen cores — which understates the CPU side, and so flatters every ratio here, by a few percent.

## The two ratios everything else is made of

| | GPU | CPU (packed, all cores) | ratio |
|---|---|---|---|
| Streaming (triad) | 466 GB/s | 325 GB/s | **1.44×** |
| KoalaBear multiply, register-resident | 332 G/s | 104 G/s | **3.21×** |

These bracket the whole project.
Unified memory is the reason the first number is so low: the GPU and the CPU draw on the same pool, and the CPU's twelve performance cores can already saturate two thirds of it.
The second is set by `mulhi`, which the GPU issues at 996 Gop/s — one sixth of its 5.8 Top/s add rate — and a Montgomery multiply needs two of them plus two low multiplies.
The measured multiply rate matches that composition to within one percent, so there is no headroom in this formulation of the arithmetic.

Nothing in the prover can beat 3.2× except by exploiting something other than raw throughput, and anything memory-bound is capped near 1.4×.

## Per-kernel results

| kernel | GPU | CPU (packed) | ratio | shape |
|---|---|---|---|---|
| Poseidon2-16 permutation | 605 M/s | 38.5 M/s | **15.70×** | compute, latency-bound on CPU |
| logup GKR round | 3700 M rows/s (296 G muls/s) | 578 M rows/s (46 G muls/s) | **6.40×** | compute |
| zerocheck round | 420 M rows/s (329 G muls/s) | 94 M rows/s (73 G muls/s) | **4.48×** | compute |
| extension fix-last-variable | 457 GB/s | 148 GB/s | **3.09×** | GPU bandwidth, CPU compute |
| extension fold | 458 GB/s | 160 GB/s | **2.86×** | as above |
| extension multiply, streamed | 9.49 G/s (456 GB/s) | 3.46 G/s (166 GB/s) | **2.74×** | as above |
| base fold | 354 GB/s | 281 GB/s | **1.26×** | bandwidth on both sides |
| base multiply, streamed | 32.2 G/s (386 GB/s) | 26.5 G/s (318 GB/s) | **1.21×** | bandwidth on both sides |
| base fix-last-variable | 387 GB/s | 337 GB/s | **1.15×** | bandwidth on both sides |
| NTT butterfly stage | 452 GB/s | 408 GB/s | **1.11×** | bandwidth on both sides |

Two of these deserve comment.

The zerocheck kernel runs at 329 G multiplies per second against a measured ceiling of 332, so it is at its roofline and cannot be improved without changing the arithmetic.
Its CPU opponent reaches 73 of a possible 104, so the 4.48× is if anything generous to the GPU: a better CPU zerocheck would close it toward 3.2×.

The GKR round is the same story one step further. It is the prover's largest phase, and its arithmetic — five extension multiplies per row against eighty bytes read — lands both sides on the compute half of their rooflines. The kernel reaches 296 of the machine's 332 G multiplies per second; the packed CPU baseline reaches 46 of its 104. So the measured **6.40×** is again the ratio between a kernel at its ceiling and a CPU implementation at 44% of its own, and the hardware's own answer is 3.2×.

Poseidon2's 15.7× is the one result that beats the compute ratio by a wide margin, and it does so because the CPU side is latency-bound rather than throughput-bound.
The permutation's cube S-box is two dependent multiplies, and the twenty partial rounds apply it to a single lane of the state; p3's packed permutation reaches only about a fifth of the CPU's multiply roofline, while the kernel reaches over half of the GPU's by running thousands of permutations concurrently.
That gap is real today — it is what SP1's CPU prover does — but it is soft: a CPU implementation that interleaved several independent permutations to fill the vector pipelines would take a large bite out of it.

## Fixed costs

| | |
|---|---|
| extra dispatch in the same command buffer | 1.6 µs |
| command buffer of its own, committed and waited | 87 µs |

Sumcheck and BaseFold rounds halve their working set each round, and each round's challenge is derived from the previous round's output, so a straightforward port pays a full round trip per round.
Folding 2²⁴ elements down to a scalar costs 2.88 ms that way, of which about 2.2 ms is round trips; the same twenty-four rounds batched into one submission take 0.63 ms.
That 4.5× is what a GPU-side challenger would be worth, and it is a design constraint rather than an optimisation: on small rounds the fixed cost is the whole cost.

## The Reed-Solomon encode

Committing a trace is not hashing it. slop encodes each trace with a Reed-Solomon code — a radix-2 DFT at the FRI blowup of 2 — and builds the Merkle tree over the codeword, and the encode is the larger half: slop's CPU encoder spends 7.3 ns per input cell against 6.5 ns for the hashing the codeword then needs, so the phase is about 53% encode.

There is no Metal encode yet, so the encode is bounded rather than measured.
A DFT is a sequence of butterfly stages, and one stage — read a pair, one multiply, two adds, write the pair back — runs at 452 GB/s on the GPU against 408 GB/s in packed CPU lanes, a ratio of **1.10×**: both sides are moving memory, and this is the floor for an implementation that keeps nothing resident across stages.
The ceiling is the register multiply roofline, 3.21×, for one that blocks perfectly.
Mixing those with the hashing ratio puts the commit phase between **1.97× and 5.13×**.

One thing that measurement makes plain: slop's CPU encoder reaches about 42 GB/s per pass, against the 408 GB/s a hand-written packed stage manages. Comparing a Metal encode against slop's encoder as it ships would show a large ratio, and almost all of it would be CPU headroom rather than GPU advantage.

## The CPU prover is not using this machine's vector units

The SP1 Hypercube prover does almost all of its field arithmetic on scalar `KoalaBear` and `BinomialExtensionField` values spread over rayon. Packing appears in the dependency graph only where Plonky3 puts it — inside `p3-merkle-tree`, whose Poseidon2 sponge and compression really do run on `PackedKoalaBearNeon` four states at a time, and inside `p3-dft`. Of `sp1-hypercube`'s fifty-eight source files exactly one mentions a packed type, and it is not on a hot path; `slop-sumcheck`, `slop-multilinear`, `slop-jagged`, `slop-basefold-prover` and `slop-dft` mention none at all.

That is not a build problem. On `aarch64-apple-darwin`, `target_feature = "neon"` is on by default, which is the only condition `p3-koala-bear` gates its NEON module and `KoalaBear::Packing` on; no flags, no features, nothing to configure. The packed path is available and unused.

The probes measure what it is worth, by running the same computation both ways:

| | scalar, as the prover writes it | packed | gain |
|---|---|---|---|
| logup GKR round | 177 M rows/s | 578 M rows/s | **3.26×** |
| fix-last-variable, extension | 32 GB/s | 148 GB/s | **4.65×** |
| fix-last-variable, base | 76 GB/s | 337 GB/s | **4.43×** |
| BaseFold fold, extension | 46 GB/s | 160 GB/s | **3.48×** |
| BaseFold fold, base | 122 GB/s | 281 GB/s | **2.31×** |
| Poseidon2 permutation | 21.1 M/s | 38.5 M/s | 1.82× |

The fold rows compare against slop's shipped `fold_mle` and `mle_fix_last_variable_constant_padding`, so they include layout as well as packing; the GKR row is the same algorithm in the same order, differing only in scalar versus packed lanes.

This matters twice over. It means roughly seventy percent of prover time — GKR, zerocheck and the PCS opening — is running at a third to a half of what this CPU can do, so a couple of person-weeks spent packing those loops would buy something in the same range as the whole Metal port. And it means any ratio quoted against the prover as it stands is charging the GPU for work the CPU declined to do: the GKR kernel is 6.40× a packed CPU round but 21× the round the prover actually runs, and only the first number is about the GPU.

Two smaller findings from the same audit. Building with `lto = "fat"` and `codegen-units = 1` instead of the release defaults moves nothing: +9% on the packed GKR round, −4% on the scalar one, within noise everywhere else. And `sp1-prover`'s worker counts are fixed constants — four core workers, two setup, two splicing, eight recursion provers — with a `TODO: base default values on system information` beside them; the profiled run held about fourteen of sixteen cores, so there is some headroom in the `SP1_WORKER_*` environment variables but not a great deal.

## Where the time goes

Profiled by replaying the ProofofSeed snark-lifter's compressed proof at n=8 through the span-timing host: 1007 s wall, proved and verified, 3997 s of accounted span time over 105 span paths.

| phase | seconds | share |
|---|---|---|
| GKR (logup) | 1872 | **46.8%** |
| PCS opening (jagged / BaseFold) | 761 | 19.0% |
| zerocheck | 600 | 15.0% |
| commit (encode + Merkle) | 546 | 13.7% |
| recursion scaffolding | 88 | 2.2% |
| trace generation | 75 | 1.9% |
| setup | 48 | 1.2% |
| execution, verify, unattributed | 7 | 0.2% |

A further 581 s is excluded as blocking. It arrives as a span called `collect global memory events`, which sounds like work and is not: it runs in a `spawn_blocking` closure sitting in `blocking_recv()` while the splicers feed it, and tracing counts a blocked thread inside an entered span as busy. It reported 581 s while emitting a single memory shard of 158k events, and the splicers upstream of it spent 1062 s "waiting for a worker". What it actually measured was the four-worker default throttling the pipeline. Counting it as execution put the host-resident share at 15% and the ceiling for any accelerator at 6.4×; excluding it puts them at **3.3% and 31×**.

That the prover is not host-bound also shows in the shape of the run. `prove shard with data` concurrency sits at 3.9 from the first minute to the last — pinned exactly at `DEFAULT_NUM_CORE_WORKERS = 4` — so the machine was never waiting for the front end, and 79 shard proofs went through a queue that never drained. There is more parallelism in the proof than four workers can take.

The recursion layers of a compressed proof are shard proofs themselves, so their internals land in the same buckets; the "recursion scaffolding" row is only the work of building and setting up the recursion programs.

The largest phase by a distance is the logup GKR argument, and it is the one the suite prices worst: no probe reproduces its shape, so its factor can only be bracketed between the streamed extension multiply (2.58×) and the zerocheck round (4.79×), whose rounds it resembles.

Two caveats on reading this table.
Span busy times are wall times on the thread that entered the span, and several shards prove concurrently, so these are phase shares of prover work only to the extent that every phase parallelises about equally.
And the profile was taken with one to two foreign cores busy on the machine, which flatters the GPU side by something under ten percent.

## What this means for the target

Feeding the phase shares and the per-phase factors into the model:

| | conservative | central | optimistic |
|---|---|---|---|
| serial port (GPU phases compress, CPU phases unchanged) | 2.10× | **3.05×** | 4.30× |
| pipelined (CPU-resident work overlapped across shards) | 2.20× | **3.25×** | 4.72× |

Pipelining buys almost nothing now that the host-resident share is 3.3%: there is nothing left to hide behind.

The conservative column gives every compute-bound phase the hardware's own multiply ratio, which is what a CPU implementation at its roofline would leave. The optimistic column gives each phase its measured ratio against a packed CPU baseline. The central column is the geometric mean, and is the number to plan against.

[Desiderata](desiderata.md) D7 asks for 5× end to end. No column reaches it. The optimistic one assumes every bracketed factor lands at its upper end, the pipelining is perfect, the port is complete — every phase moved, including the constraint transpiler that has not been designed — and the CPU prover stays exactly as unoptimised as it is today. and even then it stops at 4.7×.

The reason is not a weak GPU or a bad kernel. The zerocheck and GKR kernels both run within a few percent of the machine's measured multiply roofline; there is nothing left to extract from them. The reason is the machine: a 40-core GPU and twelve NEON performance cores are only 3.3× apart on field multiplies and 1.5× apart on memory, because they share one pool of it. On a discrete GPU with its own 1 TB/s of memory both those numbers would be several times larger, and the same port would clear 5× comfortably. On this machine it cannot, unless the CPU it is measured against stays as unvectorised as it is now — which is a strange thing to build a target on.

What would move the numbers, in order of leverage:

1. **Write the Metal encode.** It is worth up to 3× on 12% of the total, and the butterfly stage measurement says the memory-bound floor is only 1.11×, so the answer depends entirely on how well a blocked implementation holds a working set in threadgroup memory.
2. **Derive challenges on the GPU.** At 87 µs per command-buffer round trip, the sumcheck and BaseFold tails are latency-bound; batching the rounds of one fold into a single submission was 3–5× faster in the probe.
3. **Do not port the base-field folds.** At 1.26× and 1.15× they are barely worth the transfer, and PCS opening is 17% of the total spent mostly on them.

And one that competes with the whole project rather than helping it: **pack the CPU prover's hot loops**. GKR, zerocheck and the folds are 70% of prover time and run scalar today; the measured gains from packing the same algorithms are 2.3× to 4.7×. That is a smaller, safer piece of work with much of the same payoff, and it is worth doing first if only to know what the GPU is really being compared against.

The honest reading of D7 on this hardware is that 5× is out of reach, 3.3× is the realistic figure for a complete port, and roughly half of even that is available from the CPU without any GPU at all. Whether the rest is worth the work is a judgement about what the prover is for, not a measurement.
