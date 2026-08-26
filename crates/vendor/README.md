# Vendored crates

Third-party source vendored into this repository because it has to be modified.
Each directory here is an unmodified upstream release plus a recorded set of changes; nothing else in this repository is a fork.

Every crate here is `MIT OR Apache-2.0`, the same terms this repository is offered under, and none of them ships a licence file inside its crates.io package; the canonical texts are at [LICENSE-APACHE](../../LICENSE-APACHE) and [LICENSE-MIT](../../LICENSE-MIT), and the attribution is in [NOTICE](../../NOTICE).
Each crate's `CHANGES.md` lists every file that differs from the release, and each of those files carries a notice at its top saying so — which is what the Apache licence asks of a modified redistribution.

A note for anyone running `cargo doc` over these: `sp1-hypercube` has six unresolved intra-doc links, in `ir/ast.rs`, `ir/expr.rs`, `ir/lean.rs`, `machine.rs` and `air/public_values.rs`.
All six are upstream's and none is in a file this project touched, so they are left exactly as the release has them rather than quietly fixed — a fork that diverges for cosmetic reasons is harder to offer back.

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

## `slop-multilinear`

Vendored from `slop-multilinear` 6.4.0 (crates.io), which is MIT OR Apache-2.0 and carries its own licence terms.
Upstream: https://github.com/succinctlabs/sp1

**Why.** Two sampling profiles failed to say which of the prover's call sites the largest remaining block of CPU time belonged to, because rayon owns the stack for most of it.
Every fold in the prover funnels through one function here, so a counter inside it attributes all of them at once — and what it found was not arithmetic but the cost of handing tiny folds to a work-stealing pool, which is fixed in the same file.

**Changes from 6.4.0** are listed in [CHANGES.md](slop-multilinear/CHANGES.md).

## `slop-basefold-prover`

Vendored from `slop-basefold-prover` 6.4.0 (crates.io), which is MIT OR Apache-2.0 and carries its own licence terms.
Upstream: https://github.com/succinctlabs/sp1

**Why.** The dense evaluation proof's largest single piece is the random linear combination that batches a shard's MLEs before FRI begins, and it ran on one thread.
There is no seam above it: `FriCpuProver::batch` is a method on a concrete type called from a concrete prover.

**Changes from 6.4.0** are listed in [CHANGES.md](slop-basefold-prover/CHANGES.md).

## `slop-jagged`

Vendored from `slop-jagged` 6.4.0 (crates.io), which is MIT OR Apache-2.0 and carries its own licence terms.
Upstream: https://github.com/succinctlabs/sp1

**Why.** The jagged evaluation's sumcheck restacks the whole polynomial before every round, and upstream did it in three passes over the data where one will do — the last of those transposes a single-row matrix, so it permutes nothing and copies everything.
The round reduction below it is a method on a concrete CPU prover with nothing above it to substitute.

**Changes from 6.4.0** are listed in [CHANGES.md](slop-jagged/CHANGES.md).

## `p3-uni-stark`

Vendored from `p3-uni-stark` 0.4.3-succinct (crates.io), which is MIT OR Apache-2.0.
Upstream: https://github.com/Plonky3/Plonky3

**Why.** One line. `SymbolicAirBuilder` labels an AIR's preprocessed variables `Entry::Main`, so preprocessed column *j* and main column *j* come back as the same variable.
Upstream never has to tell them apart — it asks these constraints only how many there are and what degree they reach — but this project compiles them into a kernel, where a variable that says `Main` becomes a load from the main trace.
Every recursion AIR reads its opcode flags from the preprocessed trace, so every recursion kernel computed the wrong round polynomial.

**Changes from 0.4.3-succinct** are listed in [CHANGES.md](p3-uni-stark/CHANGES.md).
