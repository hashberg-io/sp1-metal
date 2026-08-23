# Changes from slop-multilinear 6.4.0

## Small folds run on the calling thread

`src/restrict.rs`, not feature-gated, since it contains nothing Metal.

`mle_fix_last_variable` handed every fold to rayon regardless of size.
Sumcheck halves its polynomial each round, so most folds in a proof are a handful of rows: measured over the acceptance workload, 98,610 of 102,882 calls were under a thousand cells, and those calls ran at 1176 nanoseconds a cell against 1.72 for the calls that carry the work.
They now run inline, and the large ones go through `par_chunks` rather than `chunks(..).par_bridge()`, which pulls a sequential iterator through a mutex and cannot split.

## The `metal` feature

New, off by default. It adds counters, not kernels.

- **`src/metal.rs`, new.** Counts and times folds, bucketed by width, plus `fold_mle` and `mle_fixed_at_zero`.
  Every fold in the prover funnels through `mle_fix_last_variable`, so one counter there attributes what five call sites and two sampling profiles could not.
- **`src/restrict.rs`, `src/fold.rs`.** Two `#[cfg]`-gated timing blocks each.
