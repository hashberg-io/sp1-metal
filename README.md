# sp1-metal

An SP1 Metal client: GPU proving for the [SP1](https://github.com/succinctlabs/sp1) zkVM on Apple Silicon, targeting the latest proof system (SP1 Hypercube, the `slop` crate family of sdk 6.x).
SP1's official GPU story is CUDA-only, and no Metal backend exists anywhere public (verified 2026-08-19); on Apple hardware the prover falls back to CPU, roughly an order of magnitude below what the on-die GPU should deliver for a 31-bit-field, sumcheck-based prover.
This project aims to close that gap by implementing a `MetalBackend` against the `slop-alloc` device-backend abstraction that the Hypercube tensor and prover layers are already generic over, with hand-written Metal kernels for the hot primitives and unified memory doing the rest.

Status: measured, and paused. The probe suite that was supposed to price the work before it began priced it out instead — a complete port predicts about 3.3× on an M4 Max against a bar of 5×, because the machine's GPU is only 3.2× its own CPU on field multiplies and 1.4× on memory. The kernels are written and correct; the case for wiring them into a backend is not there yet.

## Project structure

- [`knowledge/verdict.md`](knowledge/verdict.md) — the answer, and what would change it. **Start here.**
- [`knowledge/measurements.md`](knowledge/measurements.md) — what the machine can actually do: the two ratios that bracket the project, per-kernel results, the phase profile of a real proof, and the end-to-end model.
- [`knowledge/probes.md`](knowledge/probes.md) — the probe suite: what each probe measures, how to run it, and its limits.
- [`knowledge/desiderata.md`](knowledge/desiderata.md) — the minimum requirements this work was to meet.
- [`crates/sp1-metal-probes`](crates/sp1-metal-probes) — Metal kernels for the Hypercube hot primitives, packed-CPU baselines, and GPU-vs-CPU parity tests. `cargo test` runs the parity suite; `cargo run --release --bin probes -- all --cpu` runs the throughput ladder.
- [`crates/sp1-profile-host`](crates/sp1-profile-host) — a minimal sp1-sdk host that replays a proof under span timing. Outside the workspace: sp1-sdk pulls a very large dependency tree.
- [`tools/`](tools) — the span profiler, the summary that turns probe results into per-phase factors, and the end-to-end model.
