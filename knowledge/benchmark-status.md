# Benchmark suite: state, findings, and how to pick this up

> **Written 2026-08-24 as a handover.** Everything needed to continue is here or in `results/`; nothing depends on the session that produced it.

## Where things stand

| | state |
|---|---|
| Guests | **Done.** Fourteen in `guests/`, sources and ELFs committed. |
| Harness | **Done.** `tools/bench_suite.py`, `tools/bench_report.py`. |
| Standard tier | **Done, and re-run 2026-08-27 against the fixed backend.** `results/bench-standard.jsonl`, 204 runs, 34 points, no failures. Earlier runs kept beside it under dated names. |
| Full tier | **Done.** `results/bench-full.jsonl`, 406 runs, 53 points. Committed. |

The full tier is a detached `nohup` process. It survives the session that started it, flushes one JSON line per run, and can be interrupted at any point without losing what it already measured.

### Checking on it

```sh
pgrep -f bench_suite.py                       # still alive?
wc -l results/bench-full.jsonl                # rows so far
python3 tools/bench_report.py results/bench-full.jsonl
```

### If it died partway

The file is valid JSON Lines at every moment, so the report works on a partial file. To finish the remaining work, rerun — it appends, and duplicate points are averaged by guest/size/config rather than colliding:

```sh
nohup python3 -u tools/bench_suite.py --tier full --repeats 3 \
  --out results/bench-full.jsonl > /tmp/full.log 2>&1 &
```

### Two operational traps, both already hit

- **Orphaned provers.** A hung run holds its memory. Four of them once held 28.8 GB with only 10.3 GB free, which would have made the large runs swap and corrupted them. The harness now puts each run in its own process group and kills the group on timeout, but check anyway:
  `ps -Ao pid,etime,%cpu,rss,command | grep '[s]p1-profile-host'` — anything at 0.0% CPU and older than half an hour is dead weight, kill it.
- **The load average on this machine is a phantom.** It reads 7 while `top` reports 96% idle and nothing is running. Gate on `top -l 2 -n 0 -s 1` idle, never on load. The harness does this and records both per run.

## What the standard tier found

> The numbers below are from the 2026-08-24 run. The **2026-08-27 re-run against the fixed backend** is `results/bench-standard.md`: 204 runs, 34 points, **every proof verified**, median 3.60x over a range of 1.94x to 6.01x, with the compressed point at 3.12x. Its findings about *shape* — that size dominates chip mix, and that the memory advantage is flat — are unchanged; the differences are that keccak's large size is measured at all rather than timing out, that fib/large rose from 4.89x to 5.84x, and that divrem and keccak gave up some of their advantage because their slowest kernels no longer block a proof waiting to compile.

198 paired runs, 33 points, every proof verified except the keccak hangs below.

**Median 3.73x, range 2.52x - 5.73x.** `fib/standard` reached 5.70x, reproducing the 5.56x acceptance figure.

**Workload size dominates chip mix, which was not the expectation.** The suite was designed around chip mix. Every guest's *small* size lands at 3.2-3.8x and its *standard or large* size at 4.8-5.7x, whichever chips it exercises — arithmetic, memory, bitwise, control flow, division, or SHA-256. What separates the ends is the floor this backend does not remove, not the workload's shape.

**The memory advantage is real and flat.** The Metal arm peaks at about half the CPU arm's resident memory at every size: 18.25 GB against 9.62 at fib/small, 103.25 against 54.41 at fib/large. The prediction that it would widen with shard count was wrong; the prediction that precompile guests would fall below the envelope was also wrong, since sha256 reaches 5.43x.

Full table: `results/bench-standard.md`.

## Four defects, all closed on 2026-08-26

> Closed. See `knowledge/defect-plan.md` for what each turned out to be — two of the four were not what this section says they were. Defect 1, the "intermittent hang", and defect 3, the "scaling cliff", were both **wrong proofs**: SP1 retries a failed shard silently and leaves every thread parked, which reads as a hang. The section is kept as it was written because what it got wrong is the useful part.

Detail and reproduction steps in `results/kernel-compile-pathology.tsv`, `results/precompile-scaling.tsv`, `results/keccak-hang.md`, `results/gkr-overlapping-lends.md`, `results/zerocheck-invalid-proofs.md`.

1. **An intermittent hang.** `guests/keccak` at n>=2000 on the Metal arm sometimes stops dead at **0.0% CPU** with the machine idle. Roughly two runs in three, but genuinely intermittent — a point that hung in one tier completed in 8.86 s in the next. The CPU arm proves the same input in 13.3 s every time. Verification catches wrong answers; nothing catches this. **Fix this first.** The shape resembles the deferral hang in `backend.md` — threads parked with nothing outstanding in Metal — so a lost wakeup between a command buffer's completion handler and the thread waiting on it is the first suspect. `SP1_METAL_DEFER` is off, so that is not the cause.
2. **The node bound does not predict compile cost.** `KeccakPermuteControl` (8,234 / 12,351 nodes) and `DivRem` (7,624 / 11,436) each cost about ten minutes to compile, while `Global:true` at **15,196** compiles instantly. Cached afterwards, so it is one-time per program — but it lands on a user's first proof, which appears to hang. It buys nothing: keccak's warm wall is 2.77 s with those kernels and 2.79-2.85 s without. Workaround `SP1_METAL_ZEROCHECK_MAX_NODES=8000`; the real fix is a better predictor than node count.
3. **A scaling cliff on secp256k1 that is ours.** Between n=1000 and n=1500, same shard count, the CPU arm is flat (7.73 -> 7.79 s) and the Metal arm goes from 2.15 s to not finishing in 900. At n=1000 Metal is 3.6x faster; at n=1500 it is >115x slower. `Secp256k1DoubleAssign` is 167,376 / 251,064 nodes, far over the bound, so it never gets a kernel and its zerocheck falls back to the CPU — and it is the widest chip in the machine.
4. **Timeouts leaked processes** (fixed). `subprocess.run(timeout=)` kills only the direct child, so a hung prover under `/usr/bin/time` survived and held its memory.

## What the full tier found

406 runs, 53 points, 18 failures — every failure a timeout, and every timeout the hang.

### The phase ablation: the parts do not sum, and not in the direction predicted

`fib/standard`, GPU arm, one phase forced off at a time against a 10.98 s baseline:

| phase forced off | gpu_s | cost of removing it |
|---|---|---|
| MERKLE | 23.85 | **+12.87 s** |
| ZEROCHECK | 17.32 | +6.34 s |
| JAGGED | 14.57 | +3.60 s |
| TRANSITION | 13.13 | +2.15 s |
| ENCODE | 12.25 | +1.28 s |
| FIRST_LAYER | 11.15 | +0.17 s |

Those costs sum to 26.41 s. The whole GPU path is worth 48.70 s (59.68 → 10.98). **The parts account for 54% of the whole, not more than it.**

The prediction written down in advance was that they would sum to *more* than the whole, on the reasoning that the system is bound by its dependency structure rather than by any phase's throughput. That was wrong, and wrong in direction. Removing one phase leaves the others still absorbing its output on the device; only when all of them are off does the CPU pay the full uncompounded cost. The ablation therefore understates each phase rather than overstating it, and the Merkle tree at +12.87 s is the single largest thing this backend does.

### Shard size: flat, confirming the earlier ladder

5.49x, 5.55x, 5.44x, 5.74x, 5.72x at 1, 2, 4, 8 and 16 MiB. No trend worth naming.

### Shard concurrency: it matters more than four-versus-eight suggested

| permits | fib | sha256 |
|---|---|---|
| 1 | 3.96x | 3.48x |
| 2 | 5.13x | 3.91x |
| 4 | 5.44x | 4.10x |
| 8 | 5.63x | 4.14x |

The earlier finding that four and eight are indistinguishable stands — but one and two are not. Most of the backend's advantage needs at least four shards in flight, which is a real constraint on any deployment that serialises them.

### The node bound raised: catastrophic, as predicted

With `SP1_METAL_ZEROCHECK_MAX_NODES=1000000`, **0 of 3 GPU runs completed** for either curve guest, while their CPU arms were unmoved at 7.63 s. At the default bound the same guests run at 3.62x and 3.64x. Compiling a 167,000-node chip is not a trade worth making at any price.

### Compressed mode: unmeasurable

**0 of 3 GPU runs completed for every one of the three guests, and 0 of 10 attempts overall.** The CPU arms are fine — fib 27.72 s, sha256 28.09 s, secp256k1 25.15 s.

This sharpens the hang: in `core` mode it is intermittent (keccak roughly two runs in three, fib never in hundreds), and under recursion it is **total**. That is a much better clue than "intermittent" was, and it is where a fix should start looking.

It also means D7 remains unmeasured, because D7's bar is written against execute-to-compressed wall-clock and compressed does not complete.

## What the full tier covered

Beyond the guests: shard-size sweep (1/2/8/16 MiB), `SP1_METAL_PERMITS` at 1/2/8, the node bound raised to 10^6 on the curve guests, a six-way phase ablation on `fib`, and `compressed` mode on three guests.

Predictions recorded in advance, so the results can be scored rather than rationalised:

- **Raising the node bound will make things worse**, because compiling a 178,000-node chip costs more than the fallback saves.
- **`compressed` will be markedly below `core`** — recursion measures 2.54x and is a fixed addition. This is the number D7 actually asks for and it has never been measured.
- **The ablation will not sum.** Individual phase contributions should exceed the measured whole, because the system is bound by its dependency structure rather than any phase's throughput.

## Next actions, in order

1. Fix the keccak hang. It is the only defect that produces no output at all.
2. Replace the node-count bound with something that predicts compile cost.
3. Diagnose the secp256k1 cliff.
4. Fold the results into `verdict.md` — in particular, that the headline 5.56x is a *large-workload* figure and small workloads get about 3.5x whatever they compute.
