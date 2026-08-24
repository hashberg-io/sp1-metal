# sp1-metal-kernelbench

Scores the fork's Metal kernels against ceilings measured on the machine in front of you, rather than against a spec sheet.

```
cargo run --release            # the round reduction, shipped against baseline
cargo run --release --bin upload   # what an upload costs, allocating vs copying
```

Three things make it useful.

It reads `crates/vendor/sp1-hypercube/src/logup_gkr/shaders.metal` and the shared field arithmetic from `crates/sp1-metal-probes` **at run time**, so it always measures what the fork actually builds, and editing a kernel costs a rerun rather than the eight-minute rebuild that crate needs.

It times the GPU and nothing else, from `GPUStartTime` and `GPUEndTime` on the command buffer. Wall-clock around a dispatch includes encoding, submission and waiting, and deriving a bandwidth from it once cost this project a wrong diagnosis: 46 GB/s measured that way against 155 GB/s of actual execution.

It reports rates against two ceilings it measures first — a streaming kernel over the same data, and extension multiplies with no memory traffic at all. A kernel at a third of the streaming figure is not necessarily memory-starved, and the arithmetic ceiling is what tells you which.

Every run checks the shipped kernel against the baseline for bit-identical results at each chip width, so a performance change that alters an answer shows up as `match NO` rather than as a faster number.
