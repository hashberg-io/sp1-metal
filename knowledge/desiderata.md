# Minimum desiderata

What an SP1 Metal client must deliver to be worth shipping.
Written 2026-08-19, at the close of a day that established (during SNARK-lifter work, on an M4 Max) that no Apple-Silicon GPU proving path exists in any major zkVM: SP1's GPU backend is CUDA-only, and RISC Zero dropped its Metal segment prover in the v2 circuit rewrite.
Prior-art check the same day: no public SP1 Metal project exists (GitHub repo/issue search; Succinct's published GPU work is CUDA — `sp1-cluster`, Hypercube real-time proving on 16×5090).

## D1 — Target the current proof system, at its designed seam

The target is SP1 Hypercube (sdk/prover 6.x), not the retired Plonky3-era STARK.
The port surface is the `slop` device-backend abstraction: `slop-alloc` defines the `Backend` trait, `CpuBackend`, `DeviceMemory`, and `Buffer<T, A: Backend>`, and the tensor (`slop-tensor`) and prover layers are generic over it.
A `MetalBackend` must slot into that seam; forking the prover itself is out.
If the seam turns out not to reach some hot path (e.g. witness generation), that is a finding to record and take upstream, not a licence to fork.

## D2 — Cover the kernels that dominate, CPU-fallback the rest

On-device: the shard-proving and recursion/compression hot primitives — KoalaBear base- and extension-field arithmetic, multilinear/MLE folds, sumcheck rounds, basefold/jagged commitment work, Poseidon2, and the Merkle/hashing layers.
Everything else falls back to `CpuBackend` transparently, so the client is usable from day one with partial coverage.
The Groth16/PLONK wrap stays on CPU permanently: gnark is pure Go, takes minutes, and was verified working natively on an M4 Max (2026-08-19).

## D3 — Correctness is parity with the CPU backend, at two levels

Kernel level: every Metal primitive has a parity test against `CpuBackend` on shared vectors (field ops including extension towers, folds, permutations, hash layers), run in CI.
Pipeline level: a pre-registered binary acceptance — a real program proves end-to-end with the Metal backend and the resulting proof is accepted by the stock SP1 verifier (and, for the EVM path, by the stock on-chain verifier).
Proof bytes need not be identical to CPU-produced proofs (prover randomness); acceptance-by-unmodified-verifier is the gate.
Refusal tests follow the check-removal discipline: a parity test that cannot be made to fail by breaking the kernel proves nothing.

## D4 — Respect the hardware's arithmetic reality

KoalaBear is a 31-bit field: all field arithmetic must live in 32-bit integer lanes.
No reliance on native 64-bit integer operations (Apple GPUs emulate them); where 62-bit intermediate products arise, use explicit 32×32→hi/lo decompositions.
BN254 never touches the GPU (wrap side, CPU Go).

## D5 — Unified memory is the advantage; treat it as one budget

Zero-copy CPU↔GPU handoff (no staging copies where `MTLBuffer` sharing suffices); the design should exploit the fact that the ~100 GB working sets observed in CPU proving live in the same physical pool the GPU addresses.
The client must run within a configurable fraction of machine RAM (default ¾), because host and device now share one budget.

## D6 — Pin, track, and stay out-of-tree

Pin to a specific sp1 release (6.4.x at time of writing; the circuit line is v6.1.0) and re-run the full parity matrix on every upstream bump — upstream churns fast and a silently stale backend is worse than none.
The client must be consumable without forking sp1: a crate (plus `[patch.crates-io]` entries for the relevant `slop` crates if needed) that an existing SP1 host project enables.
Keep upstreamability open: licence choice (MIT/Apache-2 dual would match sp1; LGPL would not upstream) is an OPEN DECISION, deliberately not committed in this document.

## D7 — A measurable performance floor, end to end

Shipping bar: ≥5× the same machine's CPU prover, wall-clock, on a canonical end-to-end workload (execute → compressed proof), at equal CPU-thread budgets for the non-GPU parts.
Canonical workload candidate: a SNARK lifter's classical-128 lift (5.80e9 instructions; measured 2026-08-19 at ~7.3 h on the 16-core M4 Max CPU path, uncapped).
Stretch goal: within 5× of a single RTX 4090 on the same workload.
Kernel microbenchmarks inform but never headline; only end-to-end numbers are citable (house discipline: no number with a shortcut).

## D8 — A probe harness before any long build

First deliverable is not the backend; it is the probe suite: per-kernel parity + throughput harnesses and a cost model fitted from cheap points, so every subsequent engineering decision (which kernel next, expected payoff) is priced by measurement.
The end-to-end acceptance of D3 is pre-registered before implementation starts, and the milestone is not decomposable (house discipline: never split milestones).

## D9 — Toolchain and platform honesty

Requires full Xcode plus the separately-downloaded Metal toolchain component (`xcodebuild -downloadComponent MetalToolchain`; the license acceptance needs sudo and a real terminal) — documented, with versions pinned.
macOS 15+ on M-series only; no Docker anywhere in the pipeline.
A CI story on an Apple-Silicon runner is required before the first release tag.

## D10 — Non-goals for v1

Multi-GPU, iOS/iPadOS, real-time-proving targets, GPU-side Groth16/PLONK wrapping, other zkVMs, and x86 macOS are all explicitly out of scope for the first version.
