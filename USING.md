# Using sp1-metal

GPU proving for SP1 Hypercube on Apple Silicon, as an overlay on the crates you already depend on.
On an M4 Max it proves a shard up to **6.0x** faster than the same machine's CPU prover, measured end to end; across fourteen guests and three workload sizes the envelope is 1.94x to 6.01x, with a median of 3.60x. Those are microbenchmarks; eight whole Bitcoin and Ethereum statements measure 2.48x to 5.75x with a median of 2.98x, and that is the figure to plan against. An M5 Max puts the same eight at 2.67x to 5.99x with a median of 3.17x. Compressed proving is about 3x.

You do not need to clone this repository to use it.

## Requirements

- An Apple Silicon Mac. Everything here was measured on an M4 Max, and the whole matrix was re-measured on an M5 Max: both arms move by 8-10% and the ratio between them does not, so other M-series chips differ in degree rather than in kind. [`results/2026-08-28-m5-max-comparison.md`](results/2026-08-28-m5-max-comparison.md) has the comparison.
- Rust 1.88 or later, which is what `sp1-hypercube` itself requires.
- `protoc`, from `brew install protobuf`. It is not this project's dependency — `sp1-prover-types` has a build script that needs it — but a machine with only Rust cannot build the host, and the failure (`Could not find protoc`) does not say where it came from.
- **SP1 6.4.0 exactly.** See [Compatibility](#compatibility).

You do **not** need Xcode or the Command Line Tools.
The kernels are compiled from source at run time by the Metal framework, which is part of the OS: with `DEVELOPER_DIR` pointed at nothing, `xcrun metal` fails and every parity check still passes.

## Setting it up

Add three blocks to your host program's manifest — the one at your workspace root, since `[patch]` is only honoured there.

```toml
[features]
# Cargo unifies features across the graph, so naming these here reaches
# sp1-sdk's copies of the same crates. Four features against six patched
# crates, because slop-jagged is turned on by sp1-hypercube/metal, and
# p3-uni-stark carries a one-line fix that is not gated at all, so it has
# no metal feature to name.
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
p3-uni-stark = "=0.4.3-succinct"

[patch.crates-io]
sp1-hypercube        = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v1.0.0" }
slop-jagged          = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v1.0.0" }
slop-multilinear     = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v1.0.0" }
slop-basefold-prover = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v1.0.0" }
slop-dft             = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v1.0.0" }
p3-uni-stark         = { git = "https://github.com/hashberg-io/sp1-metal", tag = "v1.0.0" }
```

Then build with the feature on:

```sh
cargo build --release --features metal
```

That is the whole of the setup.
Every runtime switch defaults to on, so there is no environment variable to set and no code of yours to change: `SP1MerkleTreeProver` aliases to the GPU committer and `DefaultRoundOps` to the GPU reduction, both inside the fork, so the types your program names do not change.

Two details worth knowing rather than discovering:

- All six crates must be patched even though your code names one or none of them, because they must resolve to the same copies the SDK uses. That includes `p3-uni-stark`, which has no GPU code in it and no `metal` feature. Its one line labels preprocessed columns as preprocessed rather than as main, which is harmless upstream — nothing there asks a variable which trace it came from — but not to something that compiles those constraints into a kernel. Without it, four recursion AIRs generate kernels that read the main trace where they meant the preprocessed one, and compressed proofs do not verify.
- Cargo finds each crate by searching the repository, so no subdirectory needs naming. `sp1-metal-core` and `sp1-metal-probes` come along as path dependencies inside the checkout.
- If resolving the patches fails with *"failed to authenticate when downloading repository"* on what is a public repository, that is cargo's bundled git rather than anything here. `CARGO_NET_GIT_FETCH_WITH_CLI=true cargo build` makes it shell out to `git` instead, or set `net.git-fetch-with-cli = true` in `.cargo/config.toml`. Verified against this tag from a clean checkout: all six patches then resolve to it.

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
| `SP1_METAL_GKR_ROUND` | the logup GKR round reduction | on |
| `SP1_METAL_ZC_DUMP` | prints the generated kernel for the named chip | off |
| `SP1_METAL_ZEROCHECK_FOLD` | cell count below which the zerocheck fold stays on the CPU | `SP1_METAL_MIN_ENTRIES` |
| `SP1_METAL_JAGGED` | the jagged phase as a whole | on |
| `SP1_METAL_JAGGED_VALUES` | the jagged little polynomial's values | on |
| `SP1_METAL_JAGGED_ASSIST` | the jagged assist sumcheck | on |
| `SP1_METAL_BATCH` | the batched fold | on |
| `SP1_METAL_UPLOAD_CACHE` | reuse of device buffers across dispatches | on |
| `SP1_METAL_UPLOAD_LEND` | lending host pages to the device rather than copying | on |
| `SP1_METAL_UPLOAD_FRESH` | a private buffer for every upload, sharing nothing | off — a diagnostic |
| `SP1_METAL_REUSE` | handing a buffer from the kernel that wrote an array to the one that reads it | off — measured, no gain worth its hazard |
| `SP1_METAL_ZC_ILP` | interleaved constraint bodies in the generated kernel | on |
| `SP1_METAL_MIN_ENTRIES` | cell count below which work stays on the CPU | 16384 |
| `SP1_METAL_MIN_LAYER` | GKR layer size below which the layer stays on the CPU | 64 |
| `SP1_METAL_ZEROCHECK_MAX_NODES` | constraint-DAG nodes above which a chip's kernel is not even generated | 16384 |
| `SP1_METAL_COMPILE_BUDGET` | seconds a round will wait for its chip's kernel to compile | 1 |
| `SP1_METAL_COMPILE_LOG` | prints each kernel's compile time and how much of it was queueing | off |
| `SP1_METAL_COMPILE_OPT` | `size` asks the Metal compiler to optimise for size | off — measured, no help |
| `SP1_METAL_PERMITS` | shards proved at once, where the cap is above one | as the caller set it |
| `SP1_METAL_ZC_LAYOUT` | column-major zerocheck layout | **off** — measured 21% slower |
| `SP1_METAL_DEFER` | deferred command-buffer submission | **off** — produces wrong proofs |

The last two are switches for negative results, kept so that neither question gets asked twice.

**Raising `SP1_METAL_ZEROCHECK_MAX_NODES` is not the optimisation it looks like.** Above about 20,000 nodes it admits `Global:false`, whose kernel needs on the order of half an hour in the Metal compiler; the benchmark matrix runs it at 10^6 and no proof completes in fifteen minutes while the CPU arm is unmoved. The bound is not what decides whether a chip reaches the GPU any more — the compile budget is, and a kernel is installed whenever it arrives — so what the bound still does is stop the generator emitting a kernel nobody will wait for.

Eight of them take `=check` instead of `0` or `1` — `ENCODE`, `FIRST_LAYER`, `BATCH`, `JAGGED_VALUES`, `JAGGED_ASSIST`, `ZEROCHECK`, `GKR_ROUND` and `MERKLE`.
In that mode both paths run on every call of a real proof and their results are compared element for element.
It is slow and it is the strongest correctness evidence available short of the verifier, which also runs.

**Zerocheck checks itself even when you do not ask.** Its kernel is generated per chip from that chip's constraint DAG, and a generated kernel is only as right as the generator. The first round for each chip therefore compares both paths over the first few hundred row pairs of the round: a chip that agrees is trusted for the life of the process, and a chip that disagrees is declined and its rounds go to the CPU. A prefix is enough for what the check is for — a wrong column, a wrong operation or a wrong constant is wrong on every row — and comparing the whole round instead cost the larger workloads up to half their advantage.

That is not belt-and-braces. Four of SP1's recursion AIRs, in both rounds each — `BaseAlu`, `ExtAlu`, `PrefixSumChecks` and `PublicValues` — used to transpile to kernels that computed the wrong round polynomial, which is why compressed proofs did not verify before this check existed. The cause is fixed and they no longer decline, but the check is what turned an unverifiable proof into a performance cost while it lasted. `SP1_METAL_ZEROCHECK=check` turns the same comparison on for *every* call, over the whole round, and makes a disagreement fatal.

## Defects, all found and fixed

Nothing here is open. They are recorded because each one changed what the backend does, and because a reader running an older checkout may still meet them.

**An intermittent wrong proof, which read as a hang.** With every phase accelerated, a proof failed SP1's own shard verification roughly once every twenty-eight — `Query failed to verify: root mismatch` — after which SP1 retried the shard and never returned. No invalid proof was ever accepted: this was a proof being refused, so what was at risk was the run finishing rather than the statement it proved. A hundred-proof run under `SP1_METAL_MERKLE=check` found the GPU's tree disagreeing with the CPU's at layer zero, digest zero — the leaves, so the tree was built over the wrong bytes rather than built wrongly.

The cause was the table that hands a codeword's buffer from the encode to the Merkle kernels: it was keyed on a raw pointer and could outlive the allocation that pointer named, which in a process making many proofs is a use-after-free waiting for an address to be handed out twice. The soak that produced the fault at its ninety-fourth proof ran 400 of 400 with the fix in, dropping 108 orphaned hand-offs along the way — each one an entry that would previously have sat on a pointer to freed memory. The reproduction, the counts and the mechanism are in [knowledge/2026-08-28-defect-wrong-proof.md](knowledge/2026-08-28-defect-wrong-proof.md).

It has since been run on the reporting consumer's own workload, which is where the 1-in-28 rate came from: 64 proofs with the hand-off enabled and `MERKLE=check`, no disagreement, and 64 more at the defaults. That is evidence at about the ten-percent level on its own — 64 proofs predict 2.29 failures at that rate — and it is complementary to the 400-proof soak rather than a repeat of it, being on the workload whose shard count produced the report. They have retired the workaround and measured what it had been costing them: **2.52×, 2.62× and 2.94× on three different workloads**, which is worth knowing in itself. A speedup here belongs to a workload, not to a machine.

Two things came out of it that are worth having in their own right:

- `SP1_METAL_MERKLE=check` builds both trees and compares every digest of every layer, naming the lowest layer that disagrees. This phase was the only accelerated one that could not check itself, which is why a fully checked run passed while the fault was live — and it is what caught the fault the moment it was asked to.
- `sp1-profile-host --repeat N --timeout SECS` proves through one client the way a driver does, and the timeout is a watchdog on its own OS thread — the runtime is what stops turning, so a timer inside it is no use. It says the run is parked and leaves with status 2.

**A Keccak or division program's first proof appeared to hang for ten minutes.** `KeccakPermuteControl` and `DivRem` each take about that long in the Metal compiler, paid on the first proof with no output. Node count never predicted it — `Global:true` at 15,196 nodes compiles instantly while `KeccakPermuteControl` at 12,351 does not — so the bound drawn on node count could neither catch them nor be raised past them.

A round now waits `SP1_METAL_COMPILE_BUDGET` seconds for its chip's kernel and takes the CPU if it is not ready — but the kernel is **installed whenever it arrives**, so nothing is refused for having been slow once, and a chip whose kernel lands mid-proof is on the GPU for the rest of it. A keccak proof is 6.3 s with no workaround set, where its first used to stall for ten minutes.

The one cost is that a program whose chip takes longer to compile than the proof takes to run never sees that kernel, because the process exits first — `divrem` is 18.7 s that way against 14.4 s with it.

macOS caches a compiled shader against its source and its compile options, and that cache outlives the process, so the cost is payable once. `SP1_METAL_COMPILE_BUDGET=1800` on a single run blocks until every kernel is built — 571 s for keccak — and every run after it gets those kernels at the default budget. Measured: keccak 6.30 s → 5.55 s, divrem 18.7 s → 14.3 s, both back to what they were before any of this. The two benchmark matrices in `results/` measure this without meaning to: `2026-08-27-m4-max-bench-standard.md` was taken on a cold cache and `2026-08-27-m4-max-bench-full.md` ten hours later on a warm one, and of the thirty-three workloads they share the only four that move by more than a quarter are these — divrem/standard 3.75x to 5.14x, keccak/small 2.41x to 3.19x, divrem/small 2.47x to 3.24x, keccak/standard 2.07x to 2.37x, with keccak/large moving too but by less, 1.94x to 2.16x — and every one of them moves in the GPU arm alone. `results/2026-08-27-m4-max-kernel-compile-budget.md` has the deliberate pairs. `SP1_METAL_COMPILE_LOG=1` prints what each kernel cost and how much of it was queueing.

`SP1_METAL_COMPILE_OPT=size` asks for a size-optimised compile. It is kept as a measured negative: the ten-minute chip takes 552 s that way against 566 s at the default, so this is not the optimiser being aggressive.

Two proofs that did not verify, both fixed on 2026-08-26, and both of which presented as a hang: SP1 retries a failed shard silently and leaves every thread parked, so nothing but `RUST_LOG=error` says what happened.

**Compressed proofs did not verify.** The GPU zerocheck computed the wrong round polynomial for four of SP1's recursion AIRs, because the symbolic builder that hands out an AIR's constraints labels preprocessed columns as main ones — so the transpiler read the opcode flags of every recursion chip from the wrong trace. Fixed in this repository's copy of `p3-uni-stark`; if you are reading this from before that fix, `SP1_METAL_ZEROCHECK=0` is required for compressed proofs.

**secp256k1 past about a thousand operations did not verify.** This was previously recorded here as a performance cliff — 115x *slower* than the CPU arm — and it was not slow, it was wrong. The GKR round bound a slice of the same array once per chip and lent it each time, so a layer with several wide chips had several no-copy buffers over the same pages. Those bindings are copied now. If you are reading this from before the fix, `SP1_METAL_MIN_ENTRIES=999999999` keeps that phase on the CPU.

## Compatibility

**This tracks SP1 6.4.0 exactly.** Every crate is pinned with `=6.4.0` and a consumer on any other 6.x will not resolve.

That is stated rather than solved.
The six crates here are the published releases plus a recorded set of changes, so moving to a new SP1 version means unpacking the new releases and reapplying those changes; each crate's `CHANGES.md` lists every file that differs and why, which is what makes that a morning's work rather than an archaeology exercise.

These crates are not on crates.io and will not be.
`[patch.crates-io]` matches by package name: renaming them breaks the patch, and the real names belong to upstream.

## A worked example

[`crates/sp1-profile-host`](crates/sp1-profile-host) is a working host built exactly this way — it differs only in patching by path instead of by git URL, because it lives in this repository.
Its manifest is the shortest complete answer to "what do I put where".

## Correctness

The GPU path is not trusted on the strength of its being fast.

- Proofs are verified, not merely produced; the acceptance harness reports "proved and verified".
- Eight switches have a `=check` mode that compares against upstream's own code on every call.
- The forks' own test suites pass unmodified, including the four GKR tests that exist to show the CPU path is bit-identical after the seam was cut.

Run the fast parity suite yourself from a clone:

```sh
cargo test --workspace --release          # 23 tests
cargo run --release --bin probes -- check # GPU against packed CPU, small inputs
```
