# Using sp1-metal

GPU proving for SP1 Hypercube on Apple Silicon, as an overlay on the crates you already depend on.
On an M4 Max it proves a shard **5.56x** faster than the same machine's CPU prover, measured end to end; across shard sizes and workloads the envelope is 3.71x to 5.63x.

You do not need to clone this repository to use it.

## Requirements

- An Apple Silicon Mac. Everything here was measured on an M4 Max; other M-series chips will differ in degree, not in kind.
- Rust 1.88 or later, which is what `sp1-hypercube` itself requires.
- **SP1 6.4.0 exactly.** See [Compatibility](#compatibility).

You do **not** need Xcode or the Command Line Tools.
The kernels are compiled from source at run time by the Metal framework, which is part of the OS: with `DEVELOPER_DIR` pointed at nothing, `xcrun metal` fails and every parity check still passes.

## Setting it up

Add three blocks to your host program's manifest — the one at your workspace root, since `[patch]` is only honoured there.

```toml
[features]
# Cargo unifies features across the graph, so naming these here reaches
# sp1-sdk's copies of the same crates. slop-jagged is not listed because
# sp1-hypercube/metal turns it on.
metal = [
    "sp1-hypercube/metal",
    "slop-dft/metal",
    "slop-multilinear/metal",
    "slop-basefold-prover/metal",
]

[dependencies]
sp1-sdk = "=6.4.0"
# Named so the feature block above can reach them, even where your code
# never mentions them directly.
sp1-hypercube = "=6.4.0"
slop-dft = "=6.4.0"
slop-multilinear = "=6.4.0"
slop-jagged = "=6.4.0"
slop-basefold-prover = "=6.4.0"

[patch.crates-io]
sp1-hypercube        = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v0.1.0" }
slop-jagged          = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v0.1.0" }
slop-multilinear     = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v0.1.0" }
slop-basefold-prover = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v0.1.0" }
slop-dft             = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v0.1.0" }
```

Then build with the feature on:

```sh
cargo build --release --features metal
```

That is the whole of the setup.
Every runtime switch defaults to on, so there is no environment variable to set and no code of yours to change: `SP1MerkleTreeProver` aliases to the GPU committer and `DefaultRoundOps` to the GPU reduction, both inside the fork, so the types your program names do not change.

Two details worth knowing rather than discovering:

- All five crates must be patched even though your code names one or none of them, because they must resolve to the same copies the SDK uses.
- Cargo finds each crate by searching the repository, so no subdirectory needs naming. `sp1-metal-core` and `sp1-metal-probes` come along as path dependencies inside the checkout.

## Telling whether it worked

Prove something twice through the same binary, once with the GPU path forced off:

```sh
cargo run --release --features metal -- ...                  # GPU
SP1_METAL_MIN_ENTRIES=999999999 SP1_METAL_MERKLE=0 \
SP1_METAL_TRANSITION=0 SP1_METAL_ZEROCHECK=0 SP1_METAL_ENCODE=0 \
SP1_METAL_JAGGED=0 SP1_METAL_FIRST_LAYER=0 SP1_METAL_JAGGED_VALUES=0 \
SP1_METAL_JAGGED_ASSIST=0 SP1_METAL_BATCH=0 SP1_METAL_UPLOAD_CACHE=0 \
  cargo run --release --features metal -- ...                # CPU
```

Both arms through one binary is the point: it is the only comparison in which nothing but the path under test differs.
Do it on an idle machine — this project has one measurement in its history that was corrupted by a build running alongside it.

## Switches

Every switch is read once, on first use. Booleans are on unless the table says otherwise; `=0` forces the CPU path.

| switch | what it gates | default |
|---|---|---|
| `SP1_METAL_MERKLE` | the trace commitment's Merkle tree | on |
| `SP1_METAL_ENCODE` | the Reed-Solomon encode | on |
| `SP1_METAL_TRANSITION` | the logup GKR layer transition | on |
| `SP1_METAL_FIRST_LAYER` | the GKR circuit's first layer | on |
| `SP1_METAL_ZEROCHECK` | the zerocheck round reduction | on |
| `SP1_METAL_ZC_DUMP` | prints the generated kernel for the named chip | off |
| `SP1_METAL_ZEROCHECK_FOLD` | cell count below which the zerocheck fold stays on the CPU | `SP1_METAL_MIN_ENTRIES` |
| `SP1_METAL_JAGGED` | the jagged phase as a whole | on |
| `SP1_METAL_JAGGED_VALUES` | the jagged little polynomial's values | on |
| `SP1_METAL_JAGGED_ASSIST` | the jagged assist sumcheck | on |
| `SP1_METAL_BATCH` | the batched fold | on |
| `SP1_METAL_UPLOAD_CACHE` | reuse of device buffers across dispatches | on |
| `SP1_METAL_REUSE` | reuse of a round's buffer between reduce and fold | on |
| `SP1_METAL_ZC_ILP` | interleaved constraint bodies in the generated kernel | on |
| `SP1_METAL_MIN_ENTRIES` | cell count below which work stays on the CPU | 16384 |
| `SP1_METAL_MIN_LAYER` | GKR layer size below which the layer stays on the CPU | 64 |
| `SP1_METAL_ZEROCHECK_MAX_NODES` | constraint-DAG nodes above which a chip is not compiled | 16384 |
| `SP1_METAL_PERMITS` | shards proved at once, where the cap is above one | as the caller set it |
| `SP1_METAL_ZC_LAYOUT` | column-major zerocheck layout | **off** — measured 21% slower |
| `SP1_METAL_DEFER` | deferred command-buffer submission | **off** — produces wrong proofs |

The last two are switches for negative results, kept so that neither question gets asked twice.

Six of them take `=check` instead of `0` or `1` — `ENCODE`, `FIRST_LAYER`, `BATCH`, `JAGGED_VALUES`, `JAGGED_ASSIST` and `ZEROCHECK`.
In that mode both paths run on every call of a real proof and their results are compared element for element.
It is slow and it is the strongest correctness evidence available short of the verifier, which also runs.

**Zerocheck checks itself even when you do not ask.** Its kernel is generated per chip from that chip's constraint DAG, and a generated kernel is only as right as the generator. The first round for each chip therefore runs both paths and compares: a chip that agrees is trusted for the life of the process, and a chip that disagrees is declined and its rounds go to the CPU. The cost is one extra CPU round per chip, not per call.

That is not belt-and-braces. Five of SP1's recursion AIRs — `BaseAlu`, `ExtAlu`, `PrefixSumChecks` and `PublicValues` — transpile to kernels that compute the wrong round polynomial, which is why compressed proofs did not verify before this check existed. They decline themselves now. `SP1_METAL_ZEROCHECK=check` turns the same comparison on for *every* call and makes a disagreement fatal, which is how those five were found.

## Compatibility

**This tracks SP1 6.4.0 exactly.** Every crate is pinned with `=6.4.0` and a consumer on any other 6.x will not resolve.

That is stated rather than solved.
The five crates here are the published releases plus a recorded set of changes, so moving to a new SP1 version means unpacking the new releases and reapplying those changes; each crate's `CHANGES.md` lists every file that differs and why, which is what makes that a morning's work rather than an archaeology exercise.

These crates are not on crates.io and will not be.
`[patch.crates-io]` matches by package name: renaming them breaks the patch, and the real names belong to upstream.

## A worked example

[`crates/sp1-profile-host`](crates/sp1-profile-host) is a working host built exactly this way — it differs only in patching by path instead of by git URL, because it lives in this repository.
Its manifest is the shortest complete answer to "what do I put where".

## Correctness

The GPU path is not trusted on the strength of its being fast.

- Proofs are verified, not merely produced; the acceptance harness reports "proved and verified".
- Five phases have a `=check` mode that compares against upstream's own code on every call.
- The forks' own test suites pass unmodified, including the four GKR tests that exist to show the CPU path is bit-identical after the seam was cut.

Run the fast parity suite yourself from a clone:

```sh
cargo test --workspace --release          # 20 tests
cargo run --release --bin probes -- check # GPU against packed CPU, small inputs
```
