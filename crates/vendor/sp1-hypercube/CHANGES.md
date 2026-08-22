# Changes from sp1-hypercube 6.4.0

Kept deliberately small and separable, so each could be offered upstream on its own.

## 1. A seam for the logup GKR round's reduction

`logup_gkr/logup_poly.rs`, `logup_gkr/cpu.rs`, `logup_gkr/prover.rs`.

A GKR round's sumcheck message is built from three sums over the layer, and everything else about a round — the layer transitions, the padding and eq adjustments, the interpolation of the message from its evaluations — is the same wherever those sums are computed.
Upstream computes them inline, so the phase is bound to the CPU.

This adds a trait at exactly that boundary:

```rust
pub trait LogupRoundOps: Sized + Send + Sync + 'static {
    fn round_evals<F, K>(poly: &LogupRoundPolynomial<F, K, Self>) -> (K, K, K)
    where F: Field, K: ExtensionField<F>;
}
```

`CpuRoundOps` holds the code that was inline, moved verbatim.
`LogupRoundPolynomial`, `prove_gkr_round` and `GkrProverImpl` gain a backend parameter defaulting to `CpuRoundOps`, so every existing use compiles unchanged and the CPU prover is bit-identical.
The crate's own GKR tests — `test_logup_poly_fix_last_variable`, `test_logup_poly_sumcheck_circuit_layer`, `test_logup_gkr_circuit_transition`, `test_logup_gkr_round_prover` — pass without modification beyond the new struct field.

The single reduction this trait names is 46.8% of a shard proof.

## 2. Restored dev-dependencies

`Cargo.toml`, `src/air/public_values.rs`.

crates.io packaging drops dev-dependencies, so the published crate cannot build its own tests.
`rand` is restored. One assertion in `public_values.rs` compared a constant against `sp1_zkvm`'s copy; `sp1-zkvm` is a guest crate and not a dependency of the published package, so that single cross-check is commented out rather than pulling a zkVM target into the build.
