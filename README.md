# sp1-metal

GPU proving for the [SP1](https://github.com/succinctlabs/sp1) zkVM on Apple Silicon, against the current proof system — SP1 Hypercube, the `slop` crate family of sdk 6.x.
SP1's official GPU story is CUDA-only and no Metal backend exists anywhere public, so on Apple hardware the prover falls back to CPU.
This closes part of that gap: four phases of a shard proof run on the GPU, and a real proof verifies.

**On an M4 Max, core shard proving is 5.56x faster than the same machine's CPU prover** — 10.85 s against 60.25 s, both arms through one binary. Across shard sizes, workload shapes and workload sizes the envelope is 3.71x to 5.63x.
That figure is core proving, not the execute-to-compressed wall-clock that this project's own [D7](knowledge/desiderata.md) asks for; end to end it would land nearer 4.5x, which is below that bar. [The verdict](knowledge/verdict.md) says which earlier claims this refuted and which it did not.

This is experimental software and an independent fork. It is not affiliated with, endorsed by, or supported by Succinct Labs.

## Using it from your own program

You do not need to clone this. Three blocks of TOML in your host's manifest point cargo at this repository, and the Metal path is on from there — see **[USING.md](USING.md)**.

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
- [`knowledge/backend.md`](knowledge/backend.md) — how the backend works, phase by phase.
- [`knowledge/next-levers.md`](knowledge/next-levers.md) — five rounds of optimisation from 3.74x to 5.56x, and eight measured negative results.
- [`knowledge/measurements.md`](knowledge/measurements.md) — what the machine can do: the two ratios that bracket the project, per-kernel results, the phase profile of a real proof.
- [`knowledge/probes.md`](knowledge/probes.md) — the probe suite: what each probe measures and its limits.
- [`knowledge/desiderata.md`](knowledge/desiderata.md) — the requirements this work was to meet.
- [`crates/vendor/`](crates/vendor) — the five forked SP1 crates. Each is an unmodified release plus a `CHANGES.md` listing every file that differs and why.
- [`crates/sp1-metal-core`](crates/sp1-metal-core) — the shared machinery every dispatching crate needs. A deliberate leaf: it depends on `metal` and nothing else.
- [`crates/sp1-metal-probes`](crates/sp1-metal-probes) — 14 probes over 27 kernels, packed-CPU baselines and parity tests.
- [`crates/sp1-profile-host`](crates/sp1-profile-host) — a minimal sp1-sdk host that replays a proof under span timing, and the worked example of how to depend on this.
- [`guests/`](guests) — the three guest programs every measurement is taken against, sources and built ELFs.
- [`results/`](results) — the measurements the documents cite.
- [`tools/`](tools) — the quickstart, the acceptance ladder, the span profiler and the phase model.

## Tests

There is no CI. The tests worth running need a real GPU, which hosted runners do not reliably provide, and a badge covering only the rest would claim more than it checks. They are run by hand:

```sh
cargo test --workspace --release                         # 20, includes GPU parity
cargo run --release --bin probes -- check                # every kernel against a packed CPU
( cd crates/sp1-metal-prover  && cargo test --release )  # 8, the GKR and transition kernels
( cd crates/vendor/slop-jagged && cargo test --release )  # 15, includes upstream's own suite
( cd crates/vendor/sp1-hypercube && cargo test --release ) # 14, includes the four GKR tests
```

Building a vendored crate in place resolves its dev-dependencies and so rewrites the `Cargo.lock` that came with the release; `git checkout -- crates/vendor` puts it back.

The five phases with a `=check` mode run both the GPU and the CPU path on every call of a real proof and compare element for element; that, and the fact that proofs are verified rather than merely produced, is the strongest correctness evidence here.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option, matching the terms SP1 itself is offered under.

`crates/vendor/` redistributes five SP1 crates in modified form; see [NOTICE](NOTICE) for the attribution and each crate's `CHANGES.md` for what was changed.
Unless you state otherwise, any contribution you intentionally submit for inclusion shall be dual licensed as above, without additional terms or conditions.
