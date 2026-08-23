# Vendored crates

Third-party source vendored into this repository because it has to be modified.
Each directory here is an unmodified upstream release plus a recorded set of changes; nothing else in this repository is a fork.

## `sp1-hypercube`

Vendored from `sp1-hypercube` 6.4.0 (crates.io), which is MIT OR Apache-2.0 and carries its own licence terms.
Upstream: https://github.com/succinctlabs/sp1

**Why.** The logup GKR prover is 46.8% of a shard proof and cannot be reached from outside the crate.
`GkrProverImpl` holds a concrete `LogupGkrCpuTraceGenerator`, the round prover `prove_gkr_round` is a `pub(crate)` free function over CPU layer types, and `LogupGkrCpuCircuit` keeps its layers in a private field with no public constructor — so the circuit cannot even be built to check an alternative implementation against.
The same is true of zerocheck one level down: `ZeroCheckPoly` declares `type Backend = CpuBackend` outright.
The naming throughout — `LogupGkrCpuCircuit`, `LogUpGkrCpuLayer`, `logup_gkr/cpu.rs` — indicates upstream intends this factoring eventually, so the changes below are written to be offerable back rather than kept.

**Changes from 6.4.0** are listed in [CHANGES.md](sp1-hypercube/CHANGES.md).
Consumers pick up the fork through `[patch.crates-io]`, so every crate in the dependency graph that mentions `sp1-hypercube` — `sp1-core-machine` included — resolves to this copy and the types unify.

## `slop-dft`

Vendored from `slop-dft` 6.4.0 (crates.io), which is MIT OR Apache-2.0 and carries its own licence terms.
Upstream: https://github.com/succinctlabs/sp1

**Why.** The Reed-Solomon encode is about half of `commit traces`, and the encoder is not a seam: `CpuDftEncoder` holds a concrete `Arc<Radix2DitParallel>` rather than a type parameter, so unlike the Merkle committer beside it there is nothing above it to substitute.
There is something below it. `Radix2DitParallel`'s only relevant property is that it implements `Dft`, and that impl is forty lines of a crate with four dependencies and one consumer in the whole prover.
Replacing one trait method here reaches every encode a proof performs, which is why this crate is vendored and `slop-basefold-prover` — six times the size, and a fork of the proving logic rather than of a transform — is not.

**Changes from 6.4.0** are listed in [CHANGES.md](slop-dft/CHANGES.md).
