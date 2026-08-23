
## Residency accounting

`src/metal.rs` reports every buffer it makes and every command buffer it runs to the shared counters in `sp1-metal-probes`, which adds an optional dependency on that crate under the `metal` feature.
Without it the jagged dispatches were invisible to the profiling host, which made the GPU look four times less busy than it is.

## The jagged little polynomial's guts on the GPU

`partial_jagged_little_polynomial_evaluation` fills one extension element per cell of the stacked trace area, each the product of a column's eq weight and a row's, and zero in the final padding range.
It is now a single dispatch, with the ranges found by binary search rather than by the sequential walk, since a GPU thread starts at an arbitrary index.
`SP1_METAL_JAGGED_VALUES=0` keeps the CPU path; `=check` runs both and compares.
