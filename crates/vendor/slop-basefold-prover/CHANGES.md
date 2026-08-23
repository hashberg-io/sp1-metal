# Changes from slop-basefold-prover 6.4.0

## The batching random linear combination runs in parallel

`src/fri.rs`, not feature-gated, since it contains nothing Metal.

`FriCpuProver::batch` combines every MLE of a shard into one polynomial before the FRI rounds begin, and it did so on a single thread — `iter_mut().zip_eq(..)`, once per polynomial batch.
Every row of the accumulator is independent, so the loop is now one parallel pass over the rows with the polynomial batches iterated inside it, which also touches the accumulator once instead of once per batch.

`Dense PCS evaluation proof` falls from 5.4 to 3.1 thread-seconds over the acceptance workload.

`rayon` is added as a dependency; it was already in the tree.
