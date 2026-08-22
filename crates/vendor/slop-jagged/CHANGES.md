
## Residency accounting

`src/metal.rs` reports every buffer it makes and every command buffer it runs to the shared counters in `sp1-metal-probes`, which adds an optional dependency on that crate under the `metal` feature.
Without it the jagged dispatches were invisible to the profiling host, which made the GPU look four times less busy than it is.
