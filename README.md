# sp1-metal

An SP1 Metal client: GPU proving for the [SP1](https://github.com/succinctlabs/sp1) zkVM on Apple Silicon, targeting the latest proof system (SP1 Hypercube, the `slop` crate family of sdk 6.x).
SP1's official GPU story is CUDA-only, and no Metal backend exists anywhere public (verified 2026-08-19); on Apple hardware the prover falls back to CPU, roughly an order of magnitude below what the on-die GPU should deliver for a 31-bit-field, sumcheck-based prover.
This project aims to close that gap by implementing a `MetalBackend` against the `slop-alloc` device-backend abstraction that the Hypercube tensor and prover layers are already generic over, with hand-written Metal kernels for the hot primitives and unified memory doing the rest.

Status: design phase.
See `knowledge/desiderata.md` on the `design` branch for the minimum requirements this work must meet.
