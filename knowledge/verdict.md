# Verdict

The question this project set out to answer was whether an SP1 Metal backend is worth building for Apple Silicon, against the bar [desiderata](desiderata.md) D7 sets: 5x end to end versus the same machine's CPU prover.

> **Superseded in its numbers, 2026-08-27.** This document was written when the envelope was 3.71x–5.63x and before six defects were found and closed, two of them wrong proofs that read as hangs. The current measurement is [`results/bench-standard.md`](../results/bench-standard.md) and [`results/bench-full.md`](../results/bench-full.md); compressed proving is [`results/compressed-mode.tsv`](../results/compressed-mode.tsv); and [`results/bench-apps.md`](../results/bench-apps.md) is what eight whole Bitcoin and Ethereum statements measure, which is lower again at a median of 2.98x. What the document says about *what would change the verdict* is unchanged and is the part worth reading.

**The backend was built, and it works.** Four phases of a shard proof run on the GPU — the logup GKR round, the zerocheck reduction, the trace commitment and its encode, and the jagged PCS opening — and a real proof verifies. On an M4 Max, core shard proving measures **5.556x** against the same binary with every switch forced off: 10.85 s against 60.25 s, fib at n=2,000,000 and a 4 MiB shard size. Across shard sizes from 1 MiB to 16 MiB, three workload shapes and four workload sizes, the envelope is **3.71x to 5.63x**. The ladder is [`results/scaling-quiet.tsv`](../results/scaling-quiet.tsv) and [`tools/quickstart.sh`](../tools/quickstart.sh) reproduces a row of it.

**D7 is nonetheless not met — and as of 2026-08-26 that is measured rather than inferred.** Compressed proving, which is what D7's bar is written against, runs at **2.9x to 3.1x**, measured twice on 2026-08-27 and agreeing to within 0.02x (`results/compressed-mode.tsv` and the compressed rows of `results/bench-full.md`); recursion roughly halves the backend's advantage. Both of those measurements were taken on the same machine, so their agreement does not rule out a shader cache that was cold for the recursion kernels in both; running the matrix on an M5 Max put the same three points at 3.45x to 4.01x while every other point in it moved by a tenth, and [benchmark-status.md](benchmark-status.md) records settling that as an open action. D7's bar is 5x, so it is missed on either reading. The first compressed figures, 2.85x to 3.72x, were taken before the transpiler fix, when four recursion AIRs declined themselves; `results/compressed-mode.tsv` says why they are superseded rather than merely older. It could not be measured at all until the GPU zerocheck was found to compute the wrong round polynomial for four recursion AIRs (eight chip variants, both rounds each).

**5.556x is not D7's quantity.** D7 asks for wall-clock over execute to *compressed* proof on the snark-lifter's classical-128 lift. What is measured above is *core* shard proving on a synthetic workload. In this project's own recorded profile the four accelerated phases are 94.5% of prover thread-time; the remaining 5.5% — recursion, trace generation, setup, execution — is essentially untouched. Compressing 94.5% by 5.556x and leaving the rest alone puts an end-to-end figure near **4.5x**, or 4.7x if recursion keeps the 2.54x measured for it. Below the bar — and the direct measurement, when it finally came, put compressed proving lower still at about 3x rather than higher.

**And a real statement gets less than a microbenchmark.** Eight whole Bitcoin and Ethereum statements measure 2.48x–5.75x with a median of **2.98x**, about a fifth below the microbenchmark median on the same machine the same day. They also show that the microbenchmarks' "a bigger workload scores better" was really "a workload with more ordinary work around its precompiles scores better": six of the eight are flat or fall as they grow. Anyone sizing a deployment from the headline figure would be sizing it from the wrong number.

So the honest summary is that **the earlier recommendation was wrong and its measurements were not.**

## What the earlier verdict said, and how it stands

It predicted 3.3x against a *packed* CPU baseline, and it recommended not building on the strength of that number. Against the prover as it ships it predicted much more: after the spikes refit it, [backend-plan.md](backend-plan.md) put a complete port at **8.8x** integrated naively and **10.7x** with the CPU work pipelined against the GPU's, both computed by [`tools/fit_model.py`](../tools/fit_model.py) over a real profile.

- **The recommendation was wrong.** A partial port — four phases, not every GPU-able one — reached 5.56x on core shard proving against the prover as it ships.
- **The model was optimistic, not pessimistic, by roughly a factor of two.** It priced each phase by its kernel's throughput ratio and assumed the rest would follow. The built backend is bound by something the model had no term for: its own dependency structure, with the device busy about 48% of the time and the CPU using about 5 of 16 cores. [next-levers.md](next-levers.md) is five rounds of trying to close that gap, and the last three rounds moved almost nothing.
- **The packed-CPU reading of 3.3x has not been refuted**, because the CPU prover is still scalar. Nothing in this work changed that, so the comparison the earlier document called the honest one between two pieces of silicon remains unmade.

That last point is the one to carry forward. It is set out below and it is unchanged.

## Why the machine still caps this

Not for want of kernels. The zerocheck and GKR kernels run within a few percent of the machine's measured field-multiply roofline. The limit is the machine: on an M4 Max the GPU is **3.21x** the CPU on register-resident KoalaBear multiplies and **1.44x** on streaming, because unified memory means the twelve NEON performance cores already saturate two thirds of the pool the GPU draws from.

The built backend now sits against a second limit that the earlier model did not anticipate. It is bound by its own dependency structure rather than by either silicon ratio: the device is busy about 48% of the time and the CPU uses about 5 of 16 cores, and neither more shard concurrency nor deferred submission moves it. Five rounds of levers took it from 3.74x to 5.56x, and the last three moved almost nothing; [next-levers.md](next-levers.md) records every one, including eight measured negative results kept behind switches so that no question gets asked twice.

## The finding that still outranks the verdict

**The CPU prover is not using this machine's vector units.** SP1 Hypercube does its field arithmetic on scalar values; packing reaches the prover only where Plonky3 puts it, inside the Merkle tree and the DFT. Running the same algorithms packed measures 3.26x on the GKR round and 2.31–4.65x on the folds, across roughly 80% of prover time.

That is a smaller, safer piece of work than a GPU backend with much of the same payoff, and it is why the 5.556x above is quoted as what it is — a ratio against the prover as it ships, on a machine whose CPU prover has declined to use its own SIMD. Both readings are true and they measure different things. This repository now contains four CPU fixes found while chasing the GPU, none of which contain any Metal, which is a small down payment on that observation and no more.

## What would change the verdict

- **Measure D7 as written.** Execute to compressed proof, on the canonical workload, wall-clock. Compressed proving has now been measured — three guests, 2.9x to 3.1x — but on synthetic workloads, not on the snark-lifter's classical-128 lift that the bar actually names. The quantity is no longer unmeasured; the workload is still the wrong one, and both the measurement and the arithmetic above put it below the bar.
- **Pack the CPU prover.** It moves the baseline, and the honest silicon-to-silicon comparison cannot be made until it has moved.
- **Accelerate what is left.** The 5.5% outside the four phases becomes the ceiling once the phases themselves are done: at 5.56x on 94.5%, Amdahl caps the whole proof near 18x no matter what happens next to the GPU code.
- **Different Apple silicon.** The compute ratio of 3.2x and the bandwidth ratio of 1.4x are what bound the kernels. A part with a larger GPU against the same CPU would move both, and nothing about the proof system would need to change.

## What exists

A working Metal backend for SP1 Hypercube 6.4.0, as six vendored crates plus a leaf crate of shared machinery, every GPU path behind a default-off cargo feature and a runtime switch, seven of them able to run both paths and compare on every call of a real proof. [`USING.md`](../USING.md) is how to put it in your own program; [backend.md](backend.md) is how it works.

The probe suite that preceded it also stands: thirteen probes parity-checked against slop 6.4.0, a span profiler, a phase model, and the discipline that made the numbers trustworthy — measure against packed CPU baselines, never scalar ones, and check what a number measures before concluding from it. That discipline is the reason this document can say which of its own earlier claims were wrong and which were not.
