
## Residency accounting

`src/metal.rs` reports every buffer it makes and every command buffer it runs to the shared counters in `sp1-metal-probes`, which adds an optional dependency on that crate under the `metal` feature.
Without it the jagged dispatches were invisible to the profiling host, which made the GPU look four times less busy than it is.

## The jagged little polynomial's guts on the GPU

`partial_jagged_little_polynomial_evaluation` fills one extension element per cell of the stacked trace area, each the product of a column's eq weight and a row's, and zero in the final padding range.
It is now a single dispatch, with the ranges found by binary search rather than by the sequential walk, since a GPU thread starts at an arbitrary index.
`SP1_METAL_JAGGED_VALUES=0` keeps the CPU path; `=check` runs both and compares.

## The jagged assist sumcheck on the GPU

`JaggedAssistSumAsPolyCPUImpl::sum_as_poly_and_sample_into_point` evaluates a four-state branching program per column, twice per round.
The program's transition rule is uploaded as a 4x16 table built by calling `transition_function` itself, so the kernel cannot disagree with upstream about the rule; `BranchingProgram` gained `points()` and `transition_table()` for that.
`SP1_METAL_JAGGED_ASSIST=0` keeps the CPU path; `=check` runs both and compares.

## The jagged sumcheck polynomial is restacked in one pass

`src/sumcheck.rs`, not feature-gated, since it contains nothing Metal.

`interleave_multilinears_with_fixed_rate` at a batch size of one — the only rate the jagged sumcheck asks for — made three passes over the whole polynomial: it transposed each component into a fresh allocation, appended those into a running buffer, and transposed the result again.
That last transpose is of a one-row matrix, so it permutes nothing and copies everything.

`restack` does it once: each component transposes straight into its own disjoint range of the output, in parallel, and what comes out is sliced rather than transposed.
A component with a single polynomial needs no transpose at all, the tail is the only part that has to be zeroed, and the usual case of a single stack moves the buffer instead of copying it.

Building the polynomial falls from 2.19 to 0.42 thread-seconds.

`transpose` is added as a dependency; it was already in the tree, behind `slop-tensor`.
