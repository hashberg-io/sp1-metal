# Benchmark suite: state, findings, and how to pick this up

> **Written 2026-08-24 as a handover.** Everything needed to continue is here or in `results/`; nothing depends on the session that produced it.

## Where things stand

| | state |
|---|---|
| Guests | **Done.** Fourteen in `guests/`, sources and ELFs committed. |
| Harness | **Done.** `tools/bench_suite.py`, `tools/bench_report.py`. |
| Standard tier | **Done.** `results/bench-standard.jsonl`, 198 runs, 33 points. Committed. |
| Full tier | **Running when this was written**, writing to `results/bench-full.jsonl`. |

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

198 paired runs, 33 points, every proof verified except the keccak hangs below.

**Median 3.73x, range 2.52x - 5.73x.** `fib/standard` reached 5.70x, reproducing the 5.56x acceptance figure.

**Workload size dominates chip mix, which was not the expectation.** The suite was designed around chip mix. Every guest's *small* size lands at 3.2-3.8x and its *standard or large* size at 4.8-5.7x, whichever chips it exercises — arithmetic, memory, bitwise, control flow, division, or SHA-256. What separates the ends is the floor this backend does not remove, not the workload's shape.

**The memory advantage is real and flat.** The Metal arm peaks at about half the CPU arm's resident memory at every size: 18.25 GB against 9.62 at fib/small, 103.25 against 54.41 at fib/large. The prediction that it would widen with shard count was wrong; the prediction that precompile guests would fall below the envelope was also wrong, since sha256 reaches 5.43x.

Full table: `results/bench-standard.md`.

## Four defects, none fixed

Detail and reproduction steps in `results/kernel-compile-pathology.tsv`, `results/precompile-scaling.tsv`, `results/keccak-hang.md`.

1. **An intermittent hang.** `guests/keccak` at n>=2000 on the Metal arm sometimes stops dead at **0.0% CPU** with the machine idle. Roughly two runs in three, but genuinely intermittent — a point that hung in one tier completed in 8.86 s in the next. The CPU arm proves the same input in 13.3 s every time. Verification catches wrong answers; nothing catches this. **Fix this first.** The shape resembles the deferral hang in `backend.md` — threads parked with nothing outstanding in Metal — so a lost wakeup between a command buffer's completion handler and the thread waiting on it is the first suspect. `SP1_METAL_DEFER` is off, so that is not the cause.
2. **The node bound does not predict compile cost.** `KeccakPermuteControl` (8,234 / 12,351 nodes) and `DivRem` (7,624 / 11,436) each cost about ten minutes to compile, while `Global:true` at **15,196** compiles instantly. Cached afterwards, so it is one-time per program — but it lands on a user's first proof, which appears to hang. It buys nothing: keccak's warm wall is 2.77 s with those kernels and 2.79-2.85 s without. Workaround `SP1_METAL_ZEROCHECK_MAX_NODES=8000`; the real fix is a better predictor than node count.
3. **A scaling cliff on secp256k1 that is ours.** Between n=1000 and n=1500, same shard count, the CPU arm is flat (7.73 -> 7.79 s) and the Metal arm goes from 2.15 s to not finishing in 900. At n=1000 Metal is 3.6x faster; at n=1500 it is >115x slower. `Secp256k1DoubleAssign` is 167,376 / 251,064 nodes, far over the bound, so it never gets a kernel and its zerocheck falls back to the CPU — and it is the widest chip in the machine.
4. **Timeouts leaked processes** (fixed). `subprocess.run(timeout=)` kills only the direct child, so a hung prover under `/usr/bin/time` survived and held its memory.

## What the full tier adds, and what to look for

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
