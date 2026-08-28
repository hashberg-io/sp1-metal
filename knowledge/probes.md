# Probe suite

The first deliverable required by [desiderata](desiderata.md) D8: per-kernel parity and throughput harnesses, plus a cost model fitted from cheap measurements, so every backend decision is priced before it is made.
Implemented 2026-08-19 and first run in full on 2026-08-21; the results and what they imply are in [measurements.md](measurements.md).

## Layout

- `crates/sp1-metal-probes` — the probe crate: Metal kernels (`src/shaders.metal`), GPU harness, CLI (`probes`), and GPU-vs-CPU parity tests.
- `crates/sp1-profile-host` — a minimal sp1-sdk 6.4.0 host that proves a given ELF on the CPU prover with span timing enabled; kept outside the workspace because sp1-sdk pulls a very large dependency tree.
- `tools/span_profile.py` — parses the flat tracing log of an SP1 CPU prover run into per-phase exclusive times (execute / tracegen / zerocheck / gkr / commit / pcs / recursion / setup / other).
- `tools/fit_model.py` — combines the phase profile with per-phase GPU speedup estimates into an end-to-end prediction, both serial and pipelined, against the D7 bar of 5×.

## The probes, in the order they should be run

1. **Span profile** (the decisive one, no Metal involved): prove a real workload on the CPU prover with `RUST_LOGGER=flat RUST_LOG=debug`, capturing both streams (`> prove.log 2>&1` — the flat logger writes spans to stdout and colours them even into a file), then run `tools/span_profile.py` on the log.
   This yields the phase-fraction vector that determines the maximum achievable end-to-end speedup; everything else refines the per-phase factors.
   Most hosts do not call `setup_logger()`, so the profiling host is usually the way in, and it does not need the guest's source: run the real host once with `SP1_DUMP=1` and sp1-sdk writes `program.bin` and `stdin.bin` for the proof it was about to generate, then exits.
   Replaying that pair through `sp1-profile-host --elf program.bin --stdin-bin stdin.bin` proves the same workload under span timing, leaving the host that produced it untouched.
   A span's busy time is wall time on the thread that entered it, and work the span fans out to rayon is counted once rather than per worker — so these are not CPU-seconds, and with several shards proving at once they sum to well over the wall clock. Reading the shares as wall-clock shares assumes every phase parallelises about equally.
2. `probes triad` — streaming bandwidth on integer lanes; the hard ceiling for fold-shaped kernels, and the number to compare against the CPU's share of the same unified-memory pool.
3. `probes imul` — register-resident add / 32×32-lo / mulhi / KoalaBear-Montgomery throughput; the mulhi rate is unpublished by Apple and decides the field-multiply ceiling.
   `--cpu` adds the CPU's own register-resident packed multiply ceiling, which is what tells you whether a kernel sitting at the GPU's roofline is beating the CPU's hardware or only beating one particular loop.
4. `probes fieldops` — streamed base- and extension-field multiplication (bandwidth-bound shape, real field ops).
5. `probes fold` — BaseFold fold and sumcheck fix-last-variable, base and extension variants, exactly mirroring `slop-multilinear`.
6. `probes poseidon2` — width-16 KoalaBear Poseidon2 (the Merkle/hashing workhorse; compute-bound).
7. `probes launch` — what a kernel costs before it computes anything: the marginal cost of one more dispatch inside a command buffer, the cost of a command buffer of its own, and the end-to-end cost of folding a vector down to a single element one round trip at a time.
   Sumcheck and BaseFold rounds halve every round and each round's challenge comes from the previous round's output, so the tail of a fold is a sequence of ever-smaller kernels each paying a full host round trip; the probe also runs the same rounds batched into one submission, which is the ceiling a GPU-side challenger would unlock.
8. `probes encode` — the Reed-Solomon encode inside a BaseFold commitment, which is a radix-2 DFT at the FRI blowup and is most of what `commit traces` spends its time on.
   It times slop's own CPU encoder, and separately runs one radix-2 butterfly stage — the DFT's inner loop — GPU against packed CPU.
   A DFT is a sequence of those stages, so the stage ratio is the floor the encode gets when neither side keeps a block resident across stages.
   It also runs a complete Metal encode and a packed CPU one, both parity-checked against slop's output element for element at a shape with and a shape without the tiled tail.
   Both are decimation-in-frequency: slop asks for bit-reversed rows, and that is what a Gentleman-Sande transform produces from natural-order input, so no permutation is needed at all; at a blowup of two the first stage's partners are the zero padding, so it degenerates to a spread.
9. `probes gkr` — one sumcheck round of a logup GKR layer, in the shape `sp1_hypercube::logup_gkr` computes it: over each pair of consecutive entries, a numerator term `eq·(den_0·num_1 + den_1·num_0)` and a denominator term `eq·(den_0·den_1)` at each of two evaluation points, all in the degree-4 extension.
   This is the prover's largest phase, and the round's arithmetic intensity — five extension multiplies per row against eighty bytes read — puts both the GPU and the CPU on the compute side of their rooflines.
10. `probes challenger` — the duplex sponge SP1 derives every Fiat-Shamir challenge from, running on the GPU, and a complete BaseFold-shaped fold with the challenger in the loop: reduce the working set to the round message, observe it, sample the fold challenge, fold.
   It times that fold four ways — challenger on the host or on the device, one fold at a time or several in flight — because the round-trip cost `probes launch` measures is a latency, and a prover with many folds available need not pay it once per fold.
   Parity is against slop's own `DuplexChallenger<KoalaBear, KoalaPerm, 16, 8>`: a scripted observe/sample sequence that crosses the sponge's rate boundary in both directions, and then the challenges and final value of the whole fold, for both the host-driven and device-driven paths.
11. `probes zerocheck` — a synthetic zerocheck first round (T degree-3 monomials per row over a column-major trace, α-combination, eq-weighting, full reduction); prices the constraint-evaluation shape without a per-chip constraint transpiler, which is backend-phase work.
12. `probes merkle` — the Merkle tree half of a BaseFold commitment, which is what `commit traces` does after the encode: rows of the codeword hashed with `PaddingFreeSponge<KoalaPerm, 16, 8, 8>` and pairs compressed up the tree with `TruncatedPermutation<KoalaPerm, 2, 8, 16>`.
   Parity is against slop's own `Poseidon2KoalaBear16Prover`, so the root is the commitment SP1's verifier would accept rather than a tree of the right shape.
13. `probes sumcheck` — slop's own `reduce_sumcheck_to_evaluation` driver running unmodified with GPU kernels underneath it.
   The driver is generic over three traits in `slop_sumcheck::backend`, which is the designed extension point for a device backend; this implements them for `MetalBackend` and checks the driver produces the identical proof — the same univariate messages, the same challenges — as it does over slop's `Mle` on the CPU.
   It is the probe that establishes the seam exists before anything is built on it.
14. `probes backend` — what slop's device seam costs on unified memory.
   A real `slop_alloc::Buffer<T, MetalBackend>` is allocated by slop, filled through its host pointer, handed to a kernel by looking up the `MTLBuffer` behind that pointer, and read back through the pointer; nothing is copied at any point.
   The benchmark prices the copy that *did not* happen, which is what a backend with separate device memory would have to move before the same kernel could run.

## The acceptance study

Separate from the probes, and the thing that says whether any of it matters: `tools/scaling.py` over a table of paired proofs, each run twice through the same binary — once with `SP1_METAL_MIN_ENTRIES` set high enough that every layer stays on the CPU, once with the GPU path as configured. Running the same binary both ways is what makes the ratio mean something: it holds the build, the machine and the workload fixed and varies only where the arithmetic runs.

The study varies three things separately, because "a bigger workload" conflates two different experiments:

- **Shard size** (`SHARD_SIZE`, default `1 << 24` cycles) at fixed total work. Larger shards mean fewer of them and bigger GKR layers in each, which is what the kernels are sensitive to.
- **Total work** at fixed shard size. This adds shards of the same size rather than making them bigger, so it isolates whatever comes from having more proofs in flight.
- **The computation itself**, across guests with different chip mixes — arithmetic, memory and bitwise — since a layer's width is the chip's interaction count.

`sp1-profile-host --execute` reports the cycle count without proving, which is how a workload is placed on those axes before spending minutes on it.

`--repeat N` proves the same statement N times through one client, which is the only way to reach a fault that lives in state carried between proofs rather than in a kernel — see [defect-wrong-proof.md](defect-wrong-proof.md), where five single-proof runs were clean against about one failure in twenty-eight from a driver.
Pair it with `--timeout SECS`: a proof that fails shard verification is retried silently and then parks, so the run has to be killed by something outside the runtime that has stopped turning.
Each failure is reported where it happened and the run continues, because with an intermittent fault the rate is the measurement.
The end of a run also prints how many buffers were handed from one kernel to another, how many were taken, and how many were left on an address a dead entry still held.

`probes check` runs every parity check in milliseconds; `probes all --reps N --out results.json` runs the full throughput ladder.
`--cpu` adds the CPU baselines described below to every probe that has one (all but `imul`); without it only the kernels are timed.
Always build with `--release` for throughput runs (`cargo run --release --bin probes -- ...`): kernel timing is GPU-side, but buffer preparation and every `--cpu` baseline are host code.
Default sizes target roughly 1 GiB working sets; timings are host-clock around command-buffer commit/complete with one warmup pass and best-of-N reporting.

## The CPU baselines

A kernel is only as interesting as what it beats, so `--cpu` times a CPU implementation of each probe alongside the kernel, warmed and best-of-N exactly as the kernel is.
The baseline that sets the ratio is always a *packed* one: `KoalaBear::Packing` is NEON on aarch64 and p3's packed extension is `BinomialExtensionField<Packing, 4>`, and these — not scalar `KoalaBear` — are what the SP1 CPU prover's hot loops are built from.
Measuring against scalar CPU code overstates the GPU by roughly an order of magnitude on the compute-bound probes, which is the difference between a plausible project and a fantasy.

Packing is a layout choice, and layouts are the prover's to pick, so the transposes happen once outside the timed region: the fold's interleaved pairs are deinterleaved up front, eq values and alpha powers are packed before the zerocheck loop, and Poseidon2 states are transposed so one packed permutation advances four states.
The zerocheck baseline is additionally tiled by rows, which keeps the columns one tile touches inside L2 rather than re-reading the trace once per constraint term.

`fold` reports two CPU numbers: slop's shipped `fold_mle` and `mle_fix_last_variable_constant_padding` as they run today, and the packed baseline.
The gap between them is CPU headroom — work a CPU backend could reclaim without any GPU — and the ratios are reported separately (`gpu_over_cpu`, `gpu_over_cpu_packed`) so a model fitted from them cannot spend that headroom twice.

CPU baselines are the one part of the suite that a busy machine invalidates: they scale with the cores actually available, so a run competing with another job understates the CPU and inflates every ratio.
Check the machine is quiet before quoting them.

## Correctness discipline

Every Metal primitive is parity-tested against the slop 6.4.0 CPU implementations on shared random vectors: base and extension field multiplication against `slop-koala-bear` types, fold and fix-last-variable against `fold_mle` and `mle_fix_last_variable_constant_padding`, Poseidon2 against `my_kb_16_perm`, and the zerocheck kernel against a slop-typed reference of the same formula.
The refusal test was performed at implementation time: corrupting the Montgomery constant `KB_MU` failed all four field-dependent parity tests, and corrupting one Poseidon2 diagonal shift failed exactly the Poseidon2 test; both were then restored to green.
The packed CPU baselines are held to the same standard, since a baseline that computes the wrong thing flatters the kernel exactly as a broken kernel would: the packed folds are checked against slop's own output and the packed zerocheck against the scalar reference.
All GPU field arithmetic lives in 32-bit lanes (desiderata D4): Montgomery reduction uses the borrow-free `MU = P⁻¹ mod 2³²` convention from p3-koala-bear, and the one place upstream accumulates in 64 bits (the Poseidon2 internal diffusion) is mirrored with explicit two-limb 32-bit arithmetic.

## Fitting the model

```sh
python3 tools/span_profile.py prove.log --json profile.json
echo '{"zerocheck": 8, "gkr": 6, "commit": 7, "pcs": 3}' > speedups.json  # from probe measurements
python3 tools/fit_model.py profile.json speedups.json
```

The pipelined prediction assumes CPU-resident phases (execute, tracegen) overlap GPU phases across shards, which unified memory makes cheap; the serial prediction is the floor for a naive integration.
Phases mapped to factor 1 (or absent) count as CPU-resident; recursion and setup default to serial-only.
The speedup factors fed to the model must come from the probe measurements above, not from paper estimates, before any number is quoted.

## Known limitations

- The zerocheck probe prices the *shape* of constraint evaluation, not a real chip's constraint set; the symbolic-constraint-to-Metal transpiler is deliberately deferred to the backend phase.
- There is no full Metal Reed-Solomon encode, only its butterfly stage, so the largest single piece of the commit phase is bounded rather than measured. This is the suite's biggest remaining gap.
- The zerocheck baselines compare like with like on arithmetic, but the CPU one reads a column-major trace with the same random column indices as the kernel; a real chip's constraint set has far more structure, and both sides would benefit from it.
- Span busy times include same-thread child time; the parser subtracts direct children per span path, which is exact for single-threaded phases and approximate where the prover multi-threads within one span.
- Throughput timings are host-clock and include command-buffer overhead; at the default ~1 GiB sizes this overhead is noise, but do not shrink sizes without moving to GPU timestamps.
