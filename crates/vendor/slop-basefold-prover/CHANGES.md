# Changes from slop-basefold-prover 6.4.0

No kernels: one CPU fix that is unconditional, and a `metal` feature that adds timing spans and nothing else.

Every file that differs from the release is listed here, and each carries a notice at its top.

## Not feature-gated

### The batching random linear combination runs in parallel

`src/fri.rs`.

`FriCpuProver::batch` combines every MLE of a shard into one polynomial before the FRI rounds begin, and it did so on a single thread — `iter_mut().zip_eq(..)`, once per polynomial batch.
Every row of the accumulator is independent, so the loop is now one parallel pass over the rows with the polynomial batches iterated inside it, which also touches the accumulator once instead of once per batch.

`Dense PCS evaluation proof` falls from 5.4 to 3.1 thread-seconds over the acceptance workload.

`rayon` is added as a dependency; it was already in the tree.

## The `metal` feature

New, off by default: `metal = ["dep:sp1-metal-core"]`.

`src/prover.rs` gains six `#[cfg]`-gated spans — the grind, the batching Lagrange table, the batch itself, the fold at zero, each commit round, and the query phase — reported to the counters shared with the rest of this project.
They exist because the dense evaluation proof was the largest remaining block of CPU time and no existing number said which part of it.

`Cargo.toml` and `Cargo.toml.orig` declare the feature and its one optional dependency. `Cargo.toml.orig` is cargo's copy of the pre-packaging manifest; it carries the same additions, because a reader who opens it should not be told something different from what builds.
