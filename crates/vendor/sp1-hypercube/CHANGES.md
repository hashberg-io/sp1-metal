# Changes from sp1-hypercube 6.4.0

Everything here falls into one of two groups.

The first is unconditional: a seam, two CPU fixes, and one environment variable.
These change what upstream does — the seam is a refactor that leaves the CPU prover bit-identical, and the fixes are faster ways of computing the same values — and they are the part of this fork written to be offered back.

The second is the `metal` feature, off by default.
With it off, every file below compiles to what upstream publishes plus a comment; the `#[cfg]` blocks emit nothing and no Metal dependency enters the graph.
With it on, four phases of a shard proof run on the GPU.

Every file that differs from the release is listed here, and each carries a notice at its top.

## Not feature-gated

### 1. A seam for the logup GKR round's reduction

`src/logup_gkr/logup_poly.rs`, `src/logup_gkr/cpu.rs`, `src/logup_gkr/prover.rs`.

A GKR round's sumcheck message is built from three sums over the layer, and everything else about a round — the layer transitions, the padding and eq adjustments, the interpolation of the message from its evaluations — is the same wherever those sums are computed.
Upstream computes them inline, so the phase is bound to the CPU.

This adds a trait at exactly that boundary:

```rust
pub trait LogupRoundOps<F: Field, K: ExtensionField<F>>: Sized + Send + Sync + 'static {
    fn round_evals(poly: &LogupRoundPolynomial<F, K, Self>) -> (K, K, K);

    fn fix_numerator(mle: &PaddedMle<F>, alpha: K) -> PaddedMle<K> { .. }
    fn fix_denominator(mle: &PaddedMle<K>, alpha: K) -> PaddedMle<K> { .. }
    fn fix_layer(..) -> (Vec<PaddedMle<K>>, Vec<PaddedMle<K>>, Vec<PaddedMle<K>>, Vec<PaddedMle<K>>) { .. }
}
```

Only `round_evals` is required; the three folds have defaults that do exactly what folding one polynomial at a time did.
`fix_layer` exists because a layer holds one polynomial per chip and there are of the order of a hundred chips, so a backend that must synchronise per polynomial otherwise pays that cost a few hundred times a round.

The trait is parameterised by the field pair rather than generic in its methods, so an implementation may be specific to one field and still satisfy the bound wherever that field is in use.

`CpuRoundOps` holds the code that was inline, moved verbatim, and is therefore upstream's work rather than this project's.
`LogupRoundPolynomial`, `prove_gkr_round` and `GkrProverImpl` gain a backend parameter defaulting to `DefaultRoundOps`, which is `CpuRoundOps` unless the `metal` feature is on — so every existing use compiles unchanged, and turning the feature on moves the phase without a single type changing anywhere downstream.
The crate's own GKR tests — `test_logup_poly_fix_last_variable`, `test_logup_poly_sumcheck_circuit_layer`, `test_logup_gkr_circuit_transition`, `test_logup_gkr_round_prover` — pass without modification beyond the new struct field.

The single reduction this trait names is 46.8% of a shard proof.

### 2. The GKR first layer does not fill arrays it is about to overwrite

`src/logup_gkr/execution.rs`.

`generate_first_layer` allocated the numerator and denominator arrays with `vec![F::zero(); ..]` and `vec![EF::one(); ..]`, then wrote every cell of both.
Those are the largest allocations in the phase, and the denominators are filled with ones rather than zeros, so it is a real loop rather than a memset.
A private `uninit_vec` replaces the fill; the safety argument is that field elements are plain arrays of words with no invalid bit patterns and no drop glue, and that both the CPU path (`par_chunks_exact_mut` over the whole array) and the GPU path write every cell before anything reads one.

### 3. `SP1_METAL_PERMITS`

`src/prover/permits.rs`.

`ProverSemaphore::new` reads this variable when the cap it is given is more than one, which is the cap that decides how many shards prove at once.
`sp1-prover` hardcodes that to four — `cpu_worker_builder_with_machine` builds the semaphore itself and the SDK offers no way past it — so without this there is no way to ask whether more concurrent shards would fill the machine.
The caps of one, which serialise setup and recursion, are left alone.

The answer, for the record, was no: four and eight are indistinguishable on an M4 Max.

### 4. Restored dev-dependencies

`Cargo.toml`, `src/air/public_values.rs`.

crates.io packaging drops dev-dependencies, so the published crate cannot build its own tests — which are how this fork shows its changes leave the CPU path alone.
`rand` is restored.

The `metal` feature also names `slop-jagged/metal`, which the published slop-jagged does not have, so on its own the crate could not resolve its dependencies at all.
A `[patch.crates-io]` section points the four sibling crates at this repository's copies — the same set consumers patch one level up — which is what makes the tests above runnable rather than merely present.

One assertion in `public_values.rs` compared a constant against `sp1_zkvm`'s copy; `sp1-zkvm` is a guest crate and not a dependency of the published package, so that single cross-check is commented out rather than pulling a zkVM target into the build.

## The `metal` feature

New, off by default: `metal = ["dep:metal", "dep:sp1-metal-probes", "slop-jagged/metal"]`.

Each phase below also has a runtime switch, and most accept `=check` to run both the GPU and the CPU path on every call of a real proof and compare the results element for element.

### New files

- **`src/logup_gkr/metal.rs`** and **`src/logup_gkr/shaders.metal`.**
  `MetalRoundOps`, the `LogupRoundOps` implementation the seam above was cut for, plus the layer transition, the first layer's interaction values, and the folds.
  Seven kernels: `logup_round_ext`, `logup_round_base`, `fold_padded_ext`, `fold_padded_base`, `gkr_transition_ext`, `gkr_transition_base`, `first_layer`.
  A whole layer's chips go to the device in one command buffer and are waited for once, since the work is the same either way and what a round trip costs is re-entering a queue three other proving threads are using.
  The interaction layers — the rounds after the row variables are exhausted, where the whole layer is one row — stay on the CPU, being small by construction.
  Switches: `SP1_METAL_GKR_ROUND`, `SP1_METAL_TRANSITION`, `SP1_METAL_FIRST_LAYER`, `SP1_METAL_MIN_LAYER`, `SP1_METAL_MIN_ENTRIES`, `SP1_METAL_BATCH`, `SP1_METAL_UPLOAD_LEND`, `SP1_METAL_UPLOAD_FRESH`.
  The round reduction binds a slice of `eq_row` and of `eq_interaction` per chip and *copies* them: they start at the same page for every chip in the layer, so lending them put several no-copy buffers over one page at once, which cost a proof that did not verify.

- **`src/prover/zerocheck/metal.rs`.**
  The zerocheck round reduction, generated per chip and compiled the first time that chip is seen.
  The plan for this phase assumed the generator could not live here, because enumerating chips needs `sp1-core-machine` and that depends on this crate; it does not have to enumerate anything, since the prover is handed the AIR and this crate already asks AIRs for their symbolic constraints elsewhere.
  It also assumed compilation would have to happen ahead of time: true of the elliptic-curve precompiles, which run to 178,000 constraint nodes, and false of everything else — the sixty-two core chips compile in 15.5 seconds between them, once per process.
  Emits `zc_round` per chip, plus `zc_transpose_base` and `zc_transpose_ext`.
  A chip's kernel is compiled on a worker thread under `SP1_METAL_COMPILE_BUDGET` seconds and declined if it is not ready, with the verdict cached under `~/Library/Caches/sp1-metal`; node count does not predict compile cost, so the node bound is now only a guard against generating megabytes of source.
  Switches: `SP1_METAL_ZEROCHECK`, `SP1_METAL_COMPILE_BUDGET`, `SP1_METAL_ZC_LAYOUT`, `SP1_METAL_ZC_ILP`, `SP1_METAL_ZC_DUMP`, `SP1_METAL_ZEROCHECK_FOLD`, `SP1_METAL_ZEROCHECK_MAX_NODES`.

- **`src/prover/metal_tcs.rs`.**
  `MetalMerkleProver`: the tensor commitment with its Merkle tree on the GPU.
  Committing a shard's traces is the largest single thing the prover does — 47.9 thread-seconds of a 52.8 second proof, measured — and nearly all of it is one tree over the encoded rows.
  Openings stay on the CPU: `prove_openings_at_indices` walks the digest layers and `compute_openings_at_indices` copies rows, neither of which is worth a dispatch, and delegating them keeps the prover data exactly the type upstream builds.
  Switch: `SP1_METAL_MERKLE`.

- **`src/prover/zerocheck/counters.rs`.**
  Four buckets that partition the zerocheck span — per-chip setup, the round reduction, the fold, and what the sumcheck driver does between them — plus per-chip occupancy and the reasons a chip was declined.
  Sampling cannot separate these, because they are all called from inside `reduce_sumcheck_to_evaluation`.

### Touched files

- **`src/lib.rs`.** Re-exports the shared residency counters as `metal_stats`, so one table covers every vendored crate below this one.
- **`src/logup_gkr/mod.rs`.** Declares the module.
- **`src/logup_gkr/execution.rs`.** The first layer and the layer transition try the GPU and fall through; `SP1_METAL_FIRST_LAYER=check` compares both on every chip.
- **`src/prover/mod.rs`.** Aliases `SP1MerkleTreeProver` to `MetalMerkleProver`, and makes `zerocheck` public so the counters can be read from outside.
- **`src/prover/zerocheck/mod.rs`.** Declares the modules and times the four `ZeroCheckPoly` methods into the buckets.
- **`src/prover/zerocheck/fix_last_variable.rs`.** The column fold routes through one `fix_columns` indirection with a Metal and a non-Metal definition; this fold refers to no AIR, so it needs no generated code and the GKR round's kernels serve it directly.
- **`src/prover/zerocheck/sum_as_poly.rs`.** The GPU reduction for chips whose kernel can be generated, plus `eq_cache`: a thread-local memo of the last point's eq table.
  The sumcheck driver walks a round's polynomials in order on one thread and each asks for the weights of the same point, so the table was built, used, dropped and rebuilt about a hundred times a round.
  Over the acceptance workload that is 1,606 builds against 88.
  One entry and thread-local, because shards prove on their own threads and each is inside exactly one round at a time — which is also why it needs no eviction policy for what is a 32 MB table.
- **`src/prover/shard.rs`.** Times the four regions of `prove_shard`'s zerocheck into the buckets.
- **`Cargo.toml`.** The feature and its two optional dependencies.
