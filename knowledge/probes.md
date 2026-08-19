# Probe suite

The first deliverable required by [desiderata](desiderata.md) D8: per-kernel parity and throughput harnesses, plus a cost model fitted from cheap measurements, so every backend decision is priced before it is made.
Implemented 2026-08-19; throughput numbers are deliberately absent — no probe has been run yet (the machine was busy with a CPU proving job), and per house discipline no number is citable until measured end to end.

## Layout

- `crates/sp1-metal-probes` — the probe crate: Metal kernels (`src/shaders.metal`), GPU harness, CLI (`probes`), and GPU-vs-CPU parity tests.
- `crates/sp1-profile-host` — a minimal sp1-sdk 6.4.0 host that proves a given ELF on the CPU prover with span timing enabled; kept outside the workspace because sp1-sdk pulls a very large dependency tree.
- `tools/span_profile.py` — parses the flat tracing log of an SP1 CPU prover run into per-phase exclusive times (execute / tracegen / zerocheck / gkr / commit / pcs / recursion / setup / other).
- `tools/fit_model.py` — combines the phase profile with per-phase GPU speedup estimates into an end-to-end prediction, both serial and pipelined, against the D7 bar of 5×.

## The probes, in the order they should be run

1. **Span profile** (the decisive one, no Metal involved): prove a real workload on the CPU prover with `RUST_LOGGER=flat RUST_LOG=debug`, then run `tools/span_profile.py` on the log.
   This yields the phase-fraction vector that determines the maximum achievable end-to-end speedup; everything else refines the per-phase factors.
   Either build `crates/sp1-profile-host` against an existing guest ELF, or rerun any existing sp1-sdk host that calls `setup_logger()` with those environment variables.
2. `probes triad` — streaming bandwidth on integer lanes; the hard ceiling for fold-shaped kernels, and the number to compare against the CPU's share of the same unified-memory pool.
3. `probes imul` — register-resident add / 32×32-lo / mulhi / KoalaBear-Montgomery throughput; the mulhi rate is unpublished by Apple and decides the field-multiply ceiling.
4. `probes fieldops` — streamed base- and extension-field multiplication (bandwidth-bound shape, real field ops).
5. `probes fold` — BaseFold fold and sumcheck fix-last-variable, base and extension variants, exactly mirroring `slop-multilinear`.
6. `probes poseidon2` — width-16 KoalaBear Poseidon2 (the Merkle/hashing workhorse; compute-bound); `--cpu` adds a rayon all-cores CPU baseline using slop's own permutation.
7. `probes zerocheck` — a synthetic zerocheck first round (T degree-3 monomials per row over a column-major trace, α-combination, eq-weighting, full reduction); prices the constraint-evaluation shape without a per-chip constraint transpiler, which is backend-phase work.

`probes check` runs every parity check in milliseconds; `probes all --reps N --out results.json` runs the full throughput ladder.
Always build with `--release` for throughput runs (`cargo run --release --bin probes -- ...`): kernel timing is GPU-side, but buffer preparation and every `--cpu` baseline are host code.
Default sizes target roughly 1 GiB working sets; timings are host-clock around command-buffer commit/complete with one warmup pass and best-of-N reporting.

## Correctness discipline

Every Metal primitive is parity-tested against the slop 6.4.0 CPU implementations on shared random vectors: base and extension field multiplication against `slop-koala-bear` types, fold and fix-last-variable against `fold_mle` and `mle_fix_last_variable_constant_padding`, Poseidon2 against `my_kb_16_perm`, and the zerocheck kernel against a slop-typed reference of the same formula.
The refusal test was performed at implementation time: corrupting the Montgomery constant `KB_MU` failed all four field-dependent parity tests, and corrupting one Poseidon2 diagonal shift failed exactly the Poseidon2 test; both were then restored to green.
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
- `crates/sp1-profile-host` has not been compiled yet (its sp1-sdk build is heavy and was deferred along with the probe runs); expect possible small API fixes on first build.
- Span busy-times include same-thread child time; the parser subtracts direct children per span path, which is exact for single-threaded phases and approximate where the prover multi-threads within one span.
- Throughput timings are host-clock and include command-buffer overhead; at the default ~1 GiB sizes this overhead is noise, but do not shrink sizes without moving to GPU timestamps.
