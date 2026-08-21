# Verdict

The question this project set out to answer was whether an SP1 Metal backend is worth building for Apple Silicon, against the bar [desiderata](desiderata.md) D7 sets: 5× end to end versus the CPU prover.

**It is not, on this hardware.** A complete port of every GPU-able phase predicts **3.3×** on an M4 Max, with a plausible range of 2.2× to 4.7×. The evidence is in [measurements.md](measurements.md); the suite that produced it is described in [probes.md](probes.md).

Two things outrank that verdict, and both were found while establishing it.

## Why it falls short

Not for want of kernels. The zerocheck and GKR kernels run within a few percent of the machine's measured field-multiply roofline; there is nothing left in them to extract. The limit is the machine. On an M4 Max the GPU is **3.21×** the CPU on register-resident KoalaBear multiplies and **1.44×** on streaming, because unified memory means the twelve NEON performance cores already saturate two thirds of the pool the GPU draws from. Every compute-bound phase is capped near the first number, every memory-bound phase near the second, and the prover is a mixture of the two.

The phase profile of a real proof — the ProofofSeed snark-lifter's compressed proof at n=8, 1007 s wall — is GKR 46.8%, PCS opening 19.0%, zerocheck 15.0%, commit 13.7%. The kernels measured against packed CPU baselines give 6.40×, 1.15–3.09×, 4.48× and 1.97–5.13× respectively. Amdahl does the rest.

## The two findings that matter more

**The CPU prover is not using this machine's vector units.** SP1 Hypercube does its field arithmetic on scalar values; packing reaches the prover only where Plonky3 puts it, inside the Merkle tree and the DFT. Running the same algorithms packed measures 3.26× on the GKR round and 2.31–4.65× on the folds, across roughly 80% of prover time. That is a smaller, safer piece of work than a GPU backend with much of the same payoff, and until it is done any GPU ratio quoted against the prover as it ships is charging the accelerator for work the CPU declined to do.

**The workload is GPU-shaped; this GPU is the problem.** Only 3.3% of the proof is host-resident, so the ceiling for an accelerator is about 31×, and shard proving is embarrassingly parallel with a queue that never drains. On a machine with a discrete GPU the same proof projects to roughly 90 seconds against 1007 — about 10× — using SP1's own CUDA prover rather than anything written here. A six-GPU box would keep four or five cards busy on one proof and scale near-linearly across a batch; the absence of GPU peer-to-peer does not matter, because shard proofs are independent and only small proof objects cross between cards.

That projection is a projection. Everything in [measurements.md](measurements.md) is measured on the machine it describes; the NVIDIA figures are scaled from published bandwidth and shader counts and have not been run.

## What would change the verdict

- **Pack the CPU prover first.** It moves the baseline, and the honest comparison cannot be made until it has moved.
- **A full Metal Reed-Solomon encode.** The commit phase is 13.7% and its factor spans 1.97× to 5.13× because only the DFT's butterfly stage is measured, not a blocked implementation. This is the suite's largest remaining gap.
- **A GPU-side challenger.** At 87 µs per command-buffer round trip, sumcheck and BaseFold tails are latency-bound; batching the rounds of one fold into a single submission measured 4.5× faster. This is a design constraint on any Metal backend, not an optimisation to add later.
- **Different Apple silicon.** The verdict is a statement about a compute ratio of 3.2× and a bandwidth ratio of 1.4×. A part with a larger GPU against the same CPU would move both; nothing about the proof system needs to change.

## What exists

The probe suite is the durable output. It is nine probes, each parity-checked against slop 6.4.0 and refusal-tested, a span profiler for the SP1 CPU prover, a phase model, and the discipline that made the numbers trustworthy: always measure against packed CPU baselines, never scalar ones and never the shipped loops. The Metal kernels for the hot primitives — Poseidon2, zerocheck, GKR, folds, fix-last-variable, NTT butterfly — exist and are correct, and would be the starting point for a backend if the hardware case ever changes.
