# Changes from slop-jagged 6.4.0

One CPU fix that is unconditional, and a `metal` feature that is off by default.
With the feature off, every file below compiles to what upstream publishes plus a comment.

Every file that differs from the release is listed here, and each carries a notice at its top.

Each GPU path has a runtime switch that also accepts `=check`, which runs both paths on every call of a real proof and compares the results.

## Not feature-gated

### The jagged sumcheck polynomial is restacked in one pass

`src/sumcheck.rs`.

`interleave_multilinears_with_fixed_rate` at a batch size of one — the only rate the jagged sumcheck asks for — made three passes over the whole polynomial: it transposed each component into a fresh allocation, appended those into a running buffer, and transposed the result again.
That last transpose is of a one-row matrix, so it permutes nothing and copies everything.

`restack` does it once: each component transposes straight into its own disjoint range of the output, in parallel, and what comes out is sliced rather than transposed.
A component with a single polynomial needs no transpose at all, the tail is the only part that has to be zeroed, and the usual case of a single stack moves the buffer instead of copying it.

Building the polynomial falls from 2.19 to 0.42 thread-seconds.

`restack_tests` checks the new function against upstream's at six shapes.

`transpose` is added as a dependency; it was already in the tree, behind `slop-tensor`.

## The `metal` feature

New: `metal = ["dep:metal", "dep:sp1-metal-probes", "dep:sp1-metal-core"]`.

### `src/metal.rs`, new

The kernels and the dispatch logic for everything below, plus residency accounting: every buffer made and every command buffer run is reported to the shared counters.
Without that the jagged dispatches were invisible to the profiling host, which made the GPU look four times less busy than it is.

### The jagged little polynomial's guts

`src/poly.rs`.

`partial_jagged_little_polynomial_evaluation` fills one extension element per cell of the stacked trace area, each the product of a column's eq weight and a row's, and zero in the final padding range.
It is now a single dispatch, with the ranges found by binary search rather than by the sequential walk, since a GPU thread starts at an arbitrary index.

Switch: `SP1_METAL_JAGGED_VALUES`.

### The jagged assist sumcheck

`src/jagged_eval/sumcheck_sum_as_poly.rs`.

`JaggedAssistSumAsPolyCPUImpl::sum_as_poly_and_sample_into_point` evaluates a four-state branching program per column, twice per round.
The program's transition rule is uploaded as a 4x16 table built by calling `transition_function` itself, so the kernel cannot disagree with upstream about the rule; `BranchingProgram` gained `points()` and `transition_table()` for that.

Switch: `SP1_METAL_JAGGED_ASSIST`.

### The Hadamard round's three sums

`src/hadamard.rs`.

The round message is interpolated from three sums that read the same two arrays.
One dispatch computes them together, in one pass rather than three, and the interpolation above is untouched.

### The long polynomial's fold

`src/long.rs`.

`LongMle::fix_last_variable` folds each component in turn.
The fold is memory-bound, so it is only worth moving when nothing is copied to move it: the GPU path is taken only for a single-polynomial component, where the input can be borrowed where it lies and the kernel writes into the allocation that will back the result.
Everything else takes upstream's path.

### The manifest

`Cargo.toml` and `Cargo.toml.orig` declare the feature and its three optional dependencies. `Cargo.toml.orig` is cargo's copy of the pre-packaging manifest; it carries the same additions, because a reader who opens it should not be told something different from what builds.

### Timing spans

`src/prover.rs`, `src/lib.rs`.

Two spans around the jagged parameters and the sumcheck driver, and the module declaration.
