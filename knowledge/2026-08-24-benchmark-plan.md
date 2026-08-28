# Benchmark and correctness plan

> **Superseded by the suite it designed, 2026-08-27.** This was written before anything was built and it is kept for the reasoning, not the status. The suite exists: `tools/bench_suite.py`, twenty-two guests in `guests/`, four tiers, and the measurements in `results/`. [2026-08-24-benchmark-status.md](2026-08-24-benchmark-status.md) is where the work actually stands.
>
> The blocker below has cleared — the SP1 guest toolchain is installed at `~/.sp1/bin`, which is not on the `PATH`, so `export PATH="$HOME/.sp1/bin:$PATH"` first. Two of the seven sensitivities named below were answered differently than expected, and both answers are in the status document: the node bound cannot usefully be raised, and workload *size* moves the speedup far more than chip mix does.
>
> The rest stands as written.

A suite that measures this backend across the regimes it is actually sensitive to, and checks it is correct in each of them from the same runs.

Two jobs, one set of proofs, because they want the same thing: a wide spread of workloads exercised end to end. A proof that is fast and wrong is worse than useless, and the cheapest place to catch wrong is the run that was going to happen anyway.

## The blocker

**The SP1 guest toolchain is not installed on this machine.** Neither `sp1up` nor `cargo prove` is present, which is why the three existing guests ship as prebuilt ELFs. Not one new guest below can be built until that is fixed.

Step 0 is `sp1up`, and it is worth confirming it still produces ELFs that the pinned `sp1-zkvm =6.4.0` host accepts before writing any guest code — rebuild `guests/fib` first and check the acceptance row still lands where [`guests/README.md`](../guests/README.md) says it does.

## What this backend is sensitive to

A generic zkVM benchmark suite measures the zkVM. This suite has to measure the *backend*, so the axes are chosen from what is known to move its speedup.

1. **Chip mix.** The zerocheck kernel is generated per chip from that chip's constraint DAG, so which chips a program uses decides how much of zerocheck runs on the GPU at all. It also sets the GKR layer widths, since a layer's width is the chip's interaction count.
2. **The node bound.** `SP1_METAL_ZEROCHECK_MAX_NODES` is 16,384. Chips above it get no kernel and fall back to the CPU. `Global:false` already exceeds it at 22,794 nodes in the simplest workload there is, and the elliptic-curve precompiles run to 178,000. **This is the largest untested regime in the project** — see below.
3. **Shard size and shard count.** Size is measured flat from 1 to 16 MiB. Count is measured to six, where the Metal arm's spread reaches 25%.
4. **The per-program floor.** Of the 1.86 s floor, about a second is per-*program* rather than per-proof: one-time kernel compilation, 15.5 s for the sixty-two core chips on a cold shader cache. Small workloads are dominated by it and their ratios mean something different.
5. **Padding.** A chip whose `num_real_entries` is zero takes a fast path; a mostly-padded trace changes the ratio of useful to dispatched work.
6. **Proof mode.** Everything ever measured here is `--mode core`. `compressed` adds recursion, measured at 2.54x and never exercised through this backend.
7. **Memory.** At six shards the CPU arm peaks at 98 GB against the Metal arm's 47. Nothing systematically records this, and it may be the more interesting result on a 128 GB machine.

## The guests

Prior art is worth following where it exists: fibonacci, SHA-2 and SHA-3 hashing, ECDSA verification, Merkle inclusion, sorting and JSON parsing are the common spine of zkVM benchmark suites, and SP1's own published benchmarks add a Tendermint light client and Reth block execution. Those give comparability. The variants below exist because comparability is not the point here — coverage of the seven axes is.

### Tier A — regime spanners

Small, synthetic, one dominant chip family each, sized by an iteration count so the work axis is free.

| guest | stresses | status |
|---|---|---|
| `fib` | ALU add/mul, the arithmetic chips | **exists** |
| `mem` | memory chips, scatter/gather over a resident array | **exists** |
| `bits` | bitwise and shift chips | **exists** |
| `branch` | unpredictable control flow, the branch chip | new |
| `divrem` | division and remainder, whose DAG is unlike the others | new |
| `mulh` | 64-bit widening multiply chains | new |
| `padding` | deliberately leaves most chips near-empty, to exercise the zero-real-entry path | new |

`divrem` earns its place because DIVREM is the one common ALU chip whose constraint shape is nothing like add or shift, and the generated kernel is therefore nothing like the ones measured.

### Tier B — the precompile regimes

**The untested path, and the reason this suite is worth building.** Not one precompile chip appears in any measurement this project has taken. Every number in the repository comes from workloads that use fifteen core chips.

| guest | precompile chips | why |
|---|---|---|
| `sha256` | ShaCompress, ShaExtend | The most common zkVM benchmark there is, and the most common real precompile. |
| `keccak` | KeccakPermute | The EVM's hash, a very wide chip. |
| `secp256k1` | Weierstrass add/double/decompress, Fp ops | ECDSA verification. These are the chips that exceed the node bound. |
| `ed25519` | Ed25519 add/decompress | A different curve with a different chip shape. |
| `bn254-pair` | Bn254 add/double, Fp2 ops | The heaviest precompiles, and what a Groth16 wrap needs. |
| `uint256` | Uint256Mul | Big-integer arithmetic without a curve. |

Each takes an iteration count so it can be sized like Tier A: *n* hashes, *n* signature verifications, *n* pairings.

### Tier C — realistic composites

| guest | what it is | why |
|---|---|---|
| `merkle` | verify *n* inclusion proofs | hash-heavy but realistic, mixes precompile with core chips |
| `sort` | sort *n* u32s | memory and comparison heavy, a common suite member |
| `json` | parse and walk a document | control-flow and memory heavy, poor locality |
| `matmul` | dense *n*x*n* over u32 | the most regular arithmetic available, an upper bound on chip-mix uniformity |
| `tendermint` | light-client verification | SP1's own canonical benchmark; ed25519 + sha256 + deserialisation |
| `reth` | one EVM block | the flagship heavy workload; one data point, not a ladder |

Tier C is where the suite stops being about this backend and starts being comparable with published numbers. `reth` is a stretch: it may not fit the memory budget, and finding that out is itself a result.

## The configuration axes

Crossed against guests, not exhaustively — the cross product is far too large, so each axis is swept against `fib` plus two or three chosen guests.

- **Shard size** — 1, 2, 4, 8, 16 MiB. Known flat; kept as a regression check rather than a question.
- **Work** — four sizes per guest, spanning the per-program floor being dominant to being negligible.
- **Shard concurrency** — `SP1_METAL_PERMITS` at 1, 2, 4, 8. Four and eight are known indistinguishable; one and two are not measured.
- **Mode** — `core` everywhere, `compressed` on three guests. This is the only way to say anything about D7, which asks for execute-to-compressed and has never been measured.
- **The node bound** — `SP1_METAL_ZEROCHECK_MAX_NODES` raised, on the precompile guests, so the fallback chips compile instead. A direct test of whether the bound is set right.
- **Phase ablation** — with everything on, turn one phase off at a time and measure the delta. Eight configurations, and it answers a question nothing currently does: what each phase is *actually* worth in the built system, as opposed to what the model predicted for it.

## Correctness, from the same runs

Every run below is already a correctness test, because the host verifies rather than merely producing. Four things go on top.

1. **Compare the two arms' proofs byte for byte.** Field arithmetic is exact and addition is associative, so a GPU path computing the same values should produce the same proof as the CPU path, transcript included. If that holds it is by far the strongest check available — far stronger than "it verifies" — and any divergence is a real defect rather than a tolerance question. **Determinism has to be confirmed first**: prove the same input twice on one arm and compare, before trusting the cross-arm comparison. If same-arm runs already differ, find out why before going further.
2. **`=check` on the five gates that support it** — `ENCODE`, `FIRST_LAYER`, `BATCH`, `JAGGED_VALUES`, `JAGGED_ASSIST` — run on the smallest size of every guest. Both paths execute on every call and are compared element for element. It is slow, which is why it belongs on the small sizes and every guest rather than the reverse.
3. **Deliberate edge regimes**, chosen to hit the paths that ordinary workloads miss: an input of 0 and 1 iterations; sizes that make an odd number of rows, for the layer transition's identity-fraction case; a single shard; enough shards to exceed the permit count; the `padding` guest; and a precompile guest, which is the only way to exercise the no-kernel fallback under a real proof.
4. **The existing suites still pass** — 20 workspace, 8 prover, 15 jagged, 14 hypercube — run once per suite invocation, since they are seconds and they cover the seams the benchmarks do not.

## What the harness needs

`tools/scaling_study.sh` greps stderr for one number. That is fine for one ladder and will not carry a suite this size.

- **`--json` on the profile host**, emitting cycles, shard count, wall, the prove/verify split, proof bytes, and the per-phase table already printed. Parsing prose is how a suite starts lying quietly.
- **Repeats and medians.** The measured run-to-run spread is 25% on the Metal arm at six shards and about a third on zerocheck's own GPU time. Three runs minimum, median reported, range recorded. This project has drawn two wrong conclusions from single pairs.
- **Shader-cache warming.** Any configuration that changes generated shader source pays a full recompile on its first run — one such run took 108 s against 11. Warm, discard, then measure, and record that the discard happened.
- **The quiet-machine guard** from `tools/quickstart.sh`, applied to every run rather than once at the start.
- **Peak RSS**, via `/usr/bin/time -l`. Currently nothing records the memory advantage and it may be the better story.
- **Metadata per row** — git SHA, date, machine, SP1 version — because these results will be compared across months.

## Tiers, and what they cost

| tier | contents | rough cost |
|---|---|---|
| smoke | `fib`, `sha256`, `secp256k1` at one size, both arms, plus the four test suites | ~10 minutes |
| standard | all Tier A and B, three sizes, three repeats, core mode | ~2 hours |
| full | adds Tier C, the configuration sweeps, `compressed`, the ablation, and the `=check` pass | overnight |

The estimates come from the measured 71 s for one `fib` pair at n=2,000,000 and are the least reliable thing in this document; the precompile guests could be several times worse on the CPU arm, which is itself a result.

## Predictions worth writing down first

Stating these now makes the results interpretable and makes it harder to rationalise afterwards.

- **Precompile-heavy guests will show a lower speedup than the core-chip guests**, because their largest chips exceed the node bound and take zerocheck back to the CPU. If `secp256k1` lands inside the 3.71x–5.63x envelope, the bound matters less than expected and that is worth knowing.
- **Raising the node bound will make things worse, not better**, because compiling a 178,000-node chip costs more than the fallback saves. The spike measured eight minutes for the elliptic-curve precompiles.
- **`compressed` will be markedly below `core`**, since recursion measures 2.54x and is a fixed addition. This is the number D7 actually asks for.
- **The ablation will not sum.** Individual phase contributions will add up to more than the measured whole, because the system is bound by its dependency structure rather than by any phase's throughput — the device is busy under half the time.
- **The memory advantage will widen with shard count**, and may be the most useful result for anyone actually choosing hardware.

## Risks

- **The toolchain blocker is real and first.** Everything is downstream of `sp1up`.
- **Tier C guests are other people's code.** `tendermint` and `reth` mean pulling in dependencies pinned to an SP1 version, and 6.4.0 is not the current one. Budget for them not building, and treat them as optional rather than load-bearing.
- **Committed ELFs must stay reproducible.** Each new guest ships its binary like the existing three, with its hash and toolchain version recorded, or the suite inherits the problem the guests were rescued from.
- **Machine time is the binding constraint**, and this machine runs the user's own long CPU jobs. The full tier is an overnight job that has to be scheduled, not squeezed in.
