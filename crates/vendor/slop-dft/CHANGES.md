# Changes from slop-dft 6.4.0

## The `metal` feature

New, off by default. Without it this crate compiles to exactly what upstream publishes.

- **`src/metal.rs`, new.** The Reed-Solomon encode on the GPU, and the switch that decides when it is used.
- **`src/shaders.metal`, new.** The kernels it dispatches: the spread, a radix-4 pass that does two butterfly stages at once, a radix-2 pass for an odd stage left over, and the six tiled stages that finish the transform while a 64 by 64 tile is resident.
  The file is also read by the probe suite's `encode` probe, so the parity check and the prover compile the same source.
- **`src/p3.rs`.** `Radix2DitParallel`'s `Dft` impl tries the GPU first and falls through to upstream's body for anything the kernels do not compute exactly.
  Under `SP1_METAL_ENCODE=check` it runs both and compares element for element.
- **`src/lib.rs`.** Declares the module.
- **`Cargo.toml`.** Declares the feature and its three optional dependencies.

Nothing else is modified, and with the feature off `p3.rs` is upstream's file with two `#[cfg]` blocks that compile to nothing.

Every file that differs from the release is listed above, and each carries a notice at its top.
