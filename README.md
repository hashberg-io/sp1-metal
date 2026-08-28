# sp1-metal

GPU proving for the [SP1](https://github.com/succinctlabs/sp1) zkVM on Apple Silicon, against the current proof system — SP1 Hypercube, the `slop` crate family of sdk 6.x.
SP1's official GPU story is CUDA-only and no Metal backend exists anywhere public, so on Apple hardware the prover falls back to CPU.
This closes part of that gap: four phases of a shard proof run on the GPU, and a real proof verifies.

**On an M4 Max, core shard proving is 6.0x faster than the same machine's CPU prover** on a large workload — 15.04 s against 90.39 s, both arms through one binary.

That is the best case, and quoting it alone would mislead. Across fourteen guests, three workload sizes and three repeats, the measured range is **1.94x to 6.01x** with a median of 3.6x, and what moves it is the size of the workload rather than what the workload computes: a small proof gets about 3.5x whatever chips it exercises, because this backend does not remove the floor underneath it. Compressed proving — which is what this project's own [D7](knowledge/desiderata.md) bar is written against — measures **2.9x to 3.1x** across three guests, so recursion roughly halves the advantage and the 5x bar is not met. Those three figures are the one place a second machine disagrees, and the disagreement points at how they were taken rather than at the hardware — see below.

Those are microbenchmarks. **Eight whole statements from Bitcoin and Ethereum — a mined header chain, Merkle-Patricia proofs, `ecrecover`, BIP-340, the public-input and pairing halves of a Groth16 verification, and a sync-committee BLS signature — measure 2.48x to 5.75x with a median of 2.98x**, about a fifth below the microbenchmarks taken on the same machine the same day. That gap is the point of measuring them: it is what someone deploying one of these would actually get.

They also invert the size effect. Only the header chain climbs as it grows, and it is the only one of the eight with much ordinary work around its precompile; the rest are flat or fall. A workload already saturated with precompile rows gains nothing from being bigger, which is what the microbenchmarks' "bigger is better" was really measuring. Full table: [`results/bench-apps.md`](results/bench-apps.md).

One consumer has now run the whole thing end to end on a statement they deploy: 219 ZKBoo responses proved in one zkVM program, wrapped to Groth16 and verified on chain, in **2 h 51 m against 6 h 28 m** for the same pass on the same machine's CPU prover. That is **2.27x**, and it is the lowest figure in this file for two reasons worth separating — it is execute-to-Groth16 rather than shard proving, and it was taken with the Merkle tree left on the CPU, because with the tree accelerated that run does not finish. See the warning below.

The fourteen microbenchmarks have each been measured twice, ten hours apart on the same machine — once on their own, once inside the larger matrix that adds the shard-size, thread-count, ablation and compressed-mode sweeps.
That is 570 timed runs, of which 564 verified, and the application statements add 144 more with no failures at all.
The six that did not are one configuration, deliberately out of range — it raises the zerocheck node bound until a chip becomes eligible whose kernel takes longer to compile than a run is allowed to take.
Of the thirty-three workloads the two matrices share, twenty-nine agree to within 0.25x.
The four that do not are keccak and divrem, and they differ only in the GPU arm: macOS caches a compiled kernel across runs, so those two are slower the first time a machine ever proves them and settle afterwards — divrem/standard at 5.14x rather than 3.75x, keccak/small at 3.19x rather than 2.41x.
The 1.94x floor quoted above is keccak on a large workload against a cold cache; warm it is 2.16x, so that floor is a first-run cost rather than a steady state, and [USING.md](USING.md) says how to pay it once, up front.

The whole matrix has also been measured on a second machine, an M5 Max.
Both arms are 8 to 10 per cent faster and the ratio between them is unchanged — a median of 3.90x against 3.93x over the sixty-four points the two machines share, with the same guest best and the same guest worst at each end — and every structural finding reproduces: the phase ablation accounts for 60 per cent of the whole rather than 61, in the same order and with the Merkle tree still the largest single item; shard size is still flat; concurrency still stops mattering above four.
The compressed points are the exception, and they move on one side only. Their CPU arms shift by the same tenth as everything else while their GPU arms move by 1.28x to 1.48x, which no other point in the matrix comes close to. A shader cache that was warm for the core guests and cold for the recursion kernels explains that; a generational step reaching recursion alone does not. So the compressed figures above are under a question about the M4 measurement rather than superseded by the M5 one — the warm readings are 3.45x to 4.01x, and the 5x bar is missed either way.
There is no M5 figure for the eight application statements: that tier ran too hot to be worth quoting, which is itself written up in [`results/m5-max.md`](results/m5-max.md) alongside the comparison.

Every figure here comes from [`results/`](results); [the verdict](knowledge/verdict.md) says which of this project's earlier claims survived and which did not.

This is experimental software and an independent fork. It is not affiliated with, endorsed by, or supported by Succinct Labs.

## Using it from your own program

You do not need to clone this. Three blocks of TOML in your host's manifest point cargo at this repository, and the Metal path is on from there — see **[USING.md](USING.md)**.

🚨 **Set `SP1_METAL_MERKLE=0` for anything you are not watching.** A consumer proving in a loop hits a Merkle root mismatch about once every twenty-eight proofs, and SP1 then retries the shard silently and parks: no output, no exit. No invalid proof is accepted — what is at risk is the run finishing, which for an overnight run is the whole of it. It has since been reproduced here and localised to the leaves of the tree, and a use-after-free on the route the codeword takes to those leaves has been fixed; the warning stands until a long run says the fix held. [The known issue](USING.md#-an-intermittent-wrong-proof-which-parks-the-run), and [what was found](knowledge/defect-wrong-proof.md).

## Seeing it work

```sh
git clone https://github.com/hashberg-io/sp1-metal && cd sp1-metal
tools/quickstart.sh
```

Proves the same workload twice through one binary, once with every GPU switch forced off, and prints both times and the ratio.
The first build is about eight minutes; the pair is about two.
It refuses to run on a loaded machine, because a pair timed under load measures the load.

**Requirements:** an Apple Silicon Mac and Rust 1.88 or later. Not Xcode — the kernels are compiled at run time by the Metal framework, which ships with the OS.

## Project structure

- **[USING.md](USING.md)** — how to use this from another project, and every runtime switch.
- [`knowledge/`](knowledge) — the notes index, which says which documents are current and which are executed plans.
- [`knowledge/verdict.md`](knowledge/verdict.md) — what was asked, what was measured, and what is still not met. **Start here.**
- [`knowledge/backend.md`](knowledge/backend.md) — how the backend works and how it got there, section by section, in the order it happened.
- [`knowledge/next-levers.md`](knowledge/next-levers.md) — five rounds of optimisation from 3.74x to 5.56x, and eight measured negative results.
- [`knowledge/measurements.md`](knowledge/measurements.md) — what the machine can do: the two ratios that bracket the project, per-kernel results, the phase profile of a real proof.
- [`knowledge/probes.md`](knowledge/probes.md) — the probe suite: what each probe measures and its limits.
- [`knowledge/desiderata.md`](knowledge/desiderata.md) — the requirements this work was to meet.
- [`crates/vendor/`](crates/vendor) — the five forked SP1 crates and one forked Plonky3 crate. Each is an unmodified release plus a `CHANGES.md` listing every file that differs and why.
- [`crates/sp1-metal-core`](crates/sp1-metal-core) — the shared machinery every dispatching crate needs. A deliberate leaf: it depends on `metal` and nothing else.
- [`crates/sp1-metal-probes`](crates/sp1-metal-probes) — 13 probes over 27 kernels, packed-CPU baselines and parity tests.
- [`crates/sp1-profile-host`](crates/sp1-profile-host) — a minimal sp1-sdk host that replays a proof under span timing, and the worked example of how to depend on this.
- [`crates/sp1-metal-prover`](crates/sp1-metal-prover) — the harness that tests and measures the fork's GKR round from outside it. The round itself lives in `sp1-hypercube`, because that is the only place `DefaultRoundOps` can point at it.
- [`crates/sp1-metal-transpile`](crates/sp1-metal-transpile) — the spike that turned a real chip's symbolic AIR constraints into Metal, which is the generator the zerocheck kernel now uses.
- [`crates/sp1-metal-kernelbench`](crates/sp1-metal-kernelbench) — scores the shipped GKR round kernel against the ceilings the probes measure, timed from the command buffer's own `GPUStartTime` so the number contains execution and nothing else.
- [`guests/`](guests) — the twenty-two guest programs every measurement is taken against, sources and built ELFs. Fourteen are microbenchmarks and composites; eight are whole Bitcoin and Ethereum statements.
- [`results/`](results) — the measurements the documents cite.
- [`tools/`](tools) — the quickstart, the benchmark suite and its reporter, the acceptance ladder, the span and sample profilers, the phase model, and the consistency checker that holds these documents to what the code says.

## Tests

There is no CI. The tests worth running need a real GPU, which hosted runners do not reliably provide, and a badge covering only the rest would claim more than it checks. They are run by hand:

```sh
python3 tools/check_consistency.py                       # the docs against the code
cargo test --workspace --release                         # 23, includes GPU parity
cargo run --release --bin probes -- check                # every kernel against a packed CPU
( cd crates/sp1-metal-prover  && cargo test --release )  # 8, the GKR and transition kernels
( cd crates/vendor/slop-jagged && cargo test --release )  # 15, includes upstream's own suite
( cd crates/vendor/sp1-hypercube && cargo test --release ) # 14, includes the four GKR tests
```

Building a vendored crate in place resolves its dev-dependencies and so rewrites the `Cargo.lock` that came with the release; `git checkout -- crates/vendor` puts it back.

The eight switches with a `=check` mode — `ZEROCHECK`, `GKR_ROUND`, `FIRST_LAYER`, `BATCH`, `ENCODE`, `JAGGED_VALUES`, `JAGGED_ASSIST` and `MERKLE` — run both the GPU and the CPU path on every call of a real proof and compare element for element; that, and the fact that proofs are verified rather than merely produced, is the strongest correctness evidence here.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option, matching the terms SP1 itself is offered under.

`crates/vendor/` redistributes five SP1 crates and one Plonky3 crate in modified form; see [NOTICE](NOTICE) for the attribution and each crate's `CHANGES.md` for what was changed.
Unless you state otherwise, any contribution you intentionally submit for inclusion shall be dual licensed as above, without additional terms or conditions.
