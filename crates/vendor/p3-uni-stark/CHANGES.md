# Changes from p3-uni-stark 0.4.3-succinct

One line, and it is not feature-gated: `SymbolicAirBuilder` labels its preprocessed variables `Entry::Preprocessed` rather than `Entry::Main`.

## `src/symbolic_builder.rs`

`SymbolicAirBuilder::new` builds the variables an AIR sees when its constraints are read symbolically. The released version tags the preprocessed ones as main:

```rust
let prep_values = [0, 1].into_iter().flat_map(|offset| {
    (0..preprocessed_width)
        .map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
});
```

Preprocessed column *j* and main column *j* are therefore the same variable, and nothing downstream can tell them apart.

Upstream that is harmless. The only questions asked of these constraints are how many there are and what degree they reach, and the answer is the same either way — `SymbolicVariable::degree_multiple` returns 1 for `Preprocessed`, `Main` and `Permutation` alike, and nothing else in the crate matches on `Entry`.

It is not harmless for a consumer that compiles those constraints into code. This project generates a Metal kernel per chip from exactly this list, and a variable that says `Main` becomes a load from the main trace — so a chip whose constraints read preprocessed columns silently read the wrong buffer at the same index. Every recursion AIR does read them: `BaseAlu`, `ExtAlu`, `PrefixSumChecks` and `PublicValues` take their opcode flags from the preprocessed trace, and all four produced kernels that computed the wrong round polynomial. SP1's core chips were spared only because the three that have preprocessed columns — `Byte`, `Range` and `Program` — are pure lookup tables that assert nothing.

The other symbolic builder in the graph, `sp1_hypercube::InteractionBuilder`, already labels preprocessed columns correctly, which is the shape this change restores.

## `Cargo.toml`

The two `[[test]]` targets are dropped. The published manifest declares `tests/fib_air.rs` and `tests/mul_air.rs` but carries no dev-dependencies, so they cannot compile from the registry copy either; `tests/` is not vendored.

Nothing else is modified. Every file that differs from the release is listed above, and each carries a notice at its top.
