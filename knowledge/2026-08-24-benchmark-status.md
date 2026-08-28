# Benchmark suite: state, findings, and how to pick this up

> **Written 2026-08-24 as a handover.** Everything needed to continue is here or in `results/`; nothing depends on the session that produced it.

## Where things stand

| | state |
|---|---|
| Guests | **Done.** Twenty-two in `guests/`, sources and ELFs committed: fourteen microbenchmarks and composites, and eight whole Bitcoin and Ethereum statements added 2026-08-27. |
| Harness | **Done.** `tools/bench_suite.py`, `tools/bench_report.py`. |
| Standard tier | **Done, and re-run 2026-08-27 against the fixed backend.** `results/2026-08-27-m4-max-bench-standard.jsonl`, 204 runs, 34 points, no failures. Earlier runs kept beside it under dated names. |
| Application tier | **Done, 2026-08-27.** `results/2026-08-27-m4-max-bench-apps.jsonl` and `.md`, 144 timed runs, 24 points, no failures. |
| Full tier | **Done, and re-run 2026-08-27 against the fixed backend.** `results/2026-08-27-m4-max-bench-full.jsonl`, 366 timed runs, 64 points, six failures — all of them the one configuration that raises the node bound. The 2026-08-24 run is kept beside it under a dated name. |
| Second machine | **Measured 2026-08-28 on an M5 Max**, and the comparison is [`results/2026-08-28-m5-max-comparison.md`](../results/2026-08-28-m5-max-comparison.md). Both arms are 8-10% faster and the Metal advantage is unchanged. All three tiers are usable, and three points of the microbenchmark tiers re-taken cool confirm the ratios hold to within 4.5%. The application tier was run twice: the first attempt was thermally contaminated and is quoted nowhere, kept under a name that says so, and the re-run under a cooldown equal to each run's wall clock is the one to read. |

The full tier is a detached `nohup` process. It survives the session that started it, flushes one JSON line per run, and can be interrupted at any point without losing what it already measured.

### Checking on it

```sh
pgrep -f bench_suite.py                       # still alive?
wc -l results/2026-08-27-m4-max-bench-full.jsonl                # rows so far
python3 tools/bench_report.py results/2026-08-27-m4-max-bench-full.jsonl
```

### If it died partway

The file is valid JSON Lines at every moment, so the report works on a partial file. To finish the remaining work, rerun — it appends, and duplicate points are averaged by guest/size/config rather than colliding:

```sh
nohup python3 -u tools/bench_suite.py --tier full --repeats 3 \
  --out results/2026-08-27-m4-max-bench-full.jsonl > /tmp/full.log 2>&1 &
```

### Two operational traps, both already hit

- **Orphaned provers.** A hung run holds its memory. Four of them once held 28.8 GB with only 10.3 GB free, which would have made the large runs swap and corrupted them. The harness now puts each run in its own process group and kills the group on timeout, but check anyway:
  `ps -Ao pid,etime,%cpu,rss,command | grep '[s]p1-profile-host'` — anything at 0.0% CPU and older than half an hour is dead weight, kill it.
- **A `pgrep -f` waiter matches its own command line.** `until ! pgrep -f "bench_suite.py --tier apps"; do sleep 180; done` never exits, because the waiting shell's own command line contains the pattern. The application tier sat finished and unnoticed for three and a half hours. Match on something the waiter does not itself contain, or check for the process by pid.
- **The load average on this machine is a phantom.** It reads 7 while `top` reports 96% idle and nothing is running. Gate on `top -l 2 -n 0 -s 1` idle, never on load. The harness does this and records both per run.

## What the standard tier found

> The numbers below are from the 2026-08-24 run. The **2026-08-27 re-run against the fixed backend** is `results/2026-08-27-m4-max-bench-standard.md`: 204 runs, 34 points, **every proof verified**, median 3.60x over a range of 1.94x to 6.01x, with the compressed point at 3.12x. Its findings about *shape* — that size dominates chip mix, and that the memory advantage is flat — are unchanged; the differences are that keccak's large size is measured at all rather than timing out, that fib/large rose from 4.89x to 5.84x, and that divrem and keccak gave up some of their advantage because their slowest kernels no longer block a proof waiting to compile.

198 paired runs, 33 points, every proof verified except the keccak hangs below.

**Median 3.73x, range 2.52x - 5.73x.** `fib/standard` reached 5.70x, reproducing the 5.56x acceptance figure.

**Workload size dominates chip mix, which was not the expectation.** The suite was designed around chip mix. Every guest's *small* size lands at 3.2-3.8x and its *standard or large* size at 4.8-5.7x, whichever chips it exercises — arithmetic, memory, bitwise, control flow, division, or SHA-256. What separates the ends is the floor this backend does not remove, not the workload's shape.

**The memory advantage is real and flat.** The Metal arm peaks at about half the CPU arm's resident memory at every size: 18.25 GB against 9.62 at fib/small, 103.25 against 54.41 at fib/large. The prediction that it would widen with shard count was wrong; the prediction that precompile guests would fall below the envelope was also wrong, since sha256 reaches 5.43x.

Full table: `results/2026-08-27-m4-max-bench-standard.md`.

## Four defects, all closed on 2026-08-26

> Closed. See `knowledge/2026-08-26-defect-plan.md` for what each turned out to be — two of the four were not what this section says they were. Defect 1, the "intermittent hang", and defect 3, the "scaling cliff", were both **wrong proofs**: SP1 retries a failed shard silently and leaves every thread parked, which reads as a hang. The section is kept as it was written because what it got wrong is the useful part.

Detail and reproduction steps in `results/2026-08-24-m4-max-kernel-compile-pathology.tsv`, `results/2026-08-24-m4-max-precompile-scaling.tsv`, `results/2026-08-26-m4-max-keccak-hang.md`, `results/2026-08-26-m4-max-gkr-overlapping-lends.md`, `results/2026-08-26-m4-max-zerocheck-invalid-proofs.md`.

1. **An intermittent hang.** `guests/keccak` at n>=2000 on the Metal arm sometimes stops dead at **0.0% CPU** with the machine idle. Roughly two runs in three, but genuinely intermittent — a point that hung in one tier completed in 8.86 s in the next. The CPU arm proves the same input in 13.3 s every time. Verification catches wrong answers; nothing catches this. **Fix this first.** The shape resembles the deferral hang in `2026-08-22-backend.md` — threads parked with nothing outstanding in Metal — so a lost wakeup between a command buffer's completion handler and the thread waiting on it is the first suspect. `SP1_METAL_DEFER` is off, so that is not the cause.
2. **The node bound does not predict compile cost.** `KeccakPermuteControl` (8,234 / 12,351 nodes) and `DivRem` (7,624 / 11,436) each cost about ten minutes to compile, while `Global:true` at **15,196** compiles instantly. Cached afterwards, so it is one-time per program — but it lands on a user's first proof, which appears to hang. It buys nothing: keccak's warm wall is 2.77 s with those kernels and 2.79-2.85 s without. Workaround `SP1_METAL_ZEROCHECK_MAX_NODES=8000`; the real fix is a better predictor than node count.
3. **A scaling cliff on secp256k1 that is ours.** Between n=1000 and n=1500, same shard count, the CPU arm is flat (7.73 -> 7.79 s) and the Metal arm goes from 2.15 s to not finishing in 900. At n=1000 Metal is 3.6x faster; at n=1500 it is >115x slower. `Secp256k1DoubleAssign` is 167,376 / 251,064 nodes, far over the bound, so it never gets a kernel and its zerocheck falls back to the CPU — and it is the widest chip in the machine.
4. **Timeouts leaked processes** (fixed). `subprocess.run(timeout=)` kills only the direct child, so a hung prover under `/usr/bin/time` survived and held its memory.

## What the application tier found

144 timed runs, 24 points, **every proof verified.** Median **2.98x**, range 2.48x–5.75x. Full table: `results/2026-08-27-m4-max-bench-apps.md`.

**A real statement scores below a tight precompile loop, by about a fifth.** 2.98x here against 3.59x for the microbenchmarks, measured on the same machine the same day. That was the prediction written down when these guests were designed, and it is the number someone in those communities would actually get.

**It qualifies the microbenchmarks' main finding.** Across Tiers A, B and C a larger workload always scored better and size mattered more than chip mix. Here only `btc-headers` behaves that way:

| guest | small | standard | large |
|---|---|---|---|
| btc-headers | 3.44x | 5.55x | 5.75x |
| eth-ecrecover | 3.69x | 3.14x | 2.89x |
| btc-schnorr | 3.56x | 3.10x | 2.85x |
| eth-groth16-inputs | 3.27x | 2.82x | 2.48x |
| eth-bls-aggregate | 3.03x | 2.84x | 2.68x |
| eth-mpt | 2.68x | 2.48x | 2.69x |
| eth-pairing-check | 2.80x | 2.93x | 2.88x |
| eth-bls-verify | 3.09x | 3.02x | 3.05x |

Six of the eight are flat or *falling* with size. What separates `btc-headers` is that it has ordinary RISC-V work around its precompile — a header chain is three SHA-256 compressions and a great deal of byte handling — where the others are saturated with precompile rows. **Size lifts the ratio only when there is non-precompile work for it to lift.** A workload that is already all precompile gains nothing from being bigger, which is the honest reading of the earlier "size dominates chip mix": size was a proxy for how much ordinary work there was to spread over a shard.

**The two pairing guests are the largest workloads this project has proved** — sixteen and seventeen shards, where nothing else in the suite passes eight, and a CPU arm running to 287 s. They are also the flattest, which is the same finding at the extreme.

## What the full tier found

> **Re-measured 2026-08-27 against the fixed backend.** `results/2026-08-27-m4-max-bench-full.md`: 366 timed runs, 64 points, and the only failures left are the six that raise the node bound.
> The section below is kept as written, because what it got wrong is the useful part. Here is what the re-run changed.
>
> - **The ablation still does not sum, and still refutes the prediction in direction.** Against a 10.33 s baseline: MERKLE 23.86 (+13.53), ZEROCHECK 17.19 (+6.86), JAGGED 14.32 (+3.99), TRANSITION 12.74 (+2.41), ENCODE 12.24 (+1.91), FIRST_LAYER 11.51 (+1.18). Those sum to 29.88 s against a whole path worth 48.74 s — 61% of it, against 54% before, and the same conclusion. Merkle is still the largest single thing this backend does. The one row that moved is FIRST_LAYER: it was +0.17 s and is now +1.18 s, because the first layer now has kernels it used to decline.
> - **Shard size is still flat.** fib 5.61x / 5.54x / 5.72x / 5.75x / 5.85x and sha256 4.16x / 4.15x / 4.16x / 4.17x / 4.39x at 1, 2, 4, 8 and 16 MiB. Both guests now drift mildly upward at 16 MiB, but by less than the spread between repeats; there is still no trend worth naming.
> - **Concurrency still matters, and still only below four.** fib 3.99x / 4.98x / 5.72x / 5.64x and sha256 3.46x / 3.94x / 4.16x / 4.13x at 1, 2, 4 and 8 permits. Four in flight is where the advantage arrives and eight adds nothing.
> - **The raised node bound is still catastrophic, and now explained.** It admits `Global:false`, whose kernel needs on the order of half an hour in the Metal compiler, so no 900 s run can finish. 0 of 3 GPU runs completed for either curve guest while their CPU arms were unmoved at 7.61 and 7.59 s. This is why the bound stays at 16,384.
> - **Compressed mode is measurable, and measured.** It was never recursion that made it fail: those runs were producing wrong proofs, which SP1 retries silently. fib 3.09x, sha256 2.99x, secp256k1 2.90x, every run verified. **D7 is therefore answered, and not met** — its bar is 5x on execute-to-compressed and the backend delivers about 3x, so recursion roughly halves the advantage core proving has.

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

## The M4 figures may have been taken on a cold shader cache

> **Added 2026-08-28**, after the suite was run on a second machine.
> This is a question about the numbers already recorded here, not a finding about the new hardware.

Measuring on an M5 Max turned up a warming behaviour that nobody had had reason to look for on the M4.
Repeated runs at the default compile budget do not converge to a fully warm cache: `fib/standard` plateaued at 751 of its 1,311 zerocheck dispatches over five consecutive runs, with about 550 rounds a run still taking the CPU because their kernel had not finished compiling.
Only an explicit `SP1_METAL_COMPILE_BUDGET=1800` pass, which blocks until every kernel is built, reached 1,311 — after which the figure went from 12.5 s to 8.87 s, a difference of 30% that nothing in the run's own output announces.
The cache then survives even a reboot, so it is a once-per-machine cost rather than a per-run one.

That matters here because the suite's own warm-up is a single discarded run per point, which is enough for a chip whose kernel compiles in a second and not for one that takes minutes.
The M4 tiers were therefore warm for most guests and possibly not for all of them, and there is one place where the numbers themselves suggest they were not.

The two machines share all sixty-four points of the matrix, fifty-six of which have both arms; over those the M5's CPU arm is faster by a median of 1.10x and its GPU arm by 1.08x, and **no point in the matrix has a GPU gain above 1.17x**.
The three compressed points break that pattern and only on one side:

| point                | M4    | M5    | CPU gain | GPU gain |
| -------------------- | ----- | ----- | -------- | -------- |
| fib/small            | 3.09x | 3.70x | 1.07x    | 1.28x    |
| sha256/small         | 2.99x | 3.45x | 1.11x    | 1.28x    |
| secp256k1/small      | 2.90x | 4.01x | 1.07x    | 1.48x    |

The CPU arms move by the same 1.07-1.11x as everything else, so the workloads are unchanged and the machines are behaving as expected.
Only the GPU arms jump, and they jump outside the range every other point occupies.
A generational step that reaches recursion alone, by a third to a half, while touching nothing else is not a plausible reading; a cache that was warm for the core guests and cold for the recursion AIRs is.
Those AIRs are the largest kernel set here — they needed 120 s of blocking compile on the M5 — and they are exercised by exactly three points in the whole matrix, so they get the fewest warm-up runs of anything measured.

This is inference from the shape of the numbers, not a measurement, and it cannot be settled from the M5.
It is worth settling because compressed proving is what [2026-08-19-desiderata.md](2026-08-19-desiderata.md) D7 is judged against: the recorded answer is "about 3x", and if the recorded answer was taken cold then the backend does better than the verdict currently credits it with. It would still be short of D7's 5x either way.

## Next actions, in order

The first three of these are closed; they are kept so the order they were written in can be read against what they turned out to be.

1. ~~Fix the keccak hang.~~ Closed 2026-08-26. It was not a hang: the proof was wrong, and SP1's silent shard retry left every thread parked, which reads as one.
2. ~~Replace the node-count bound with something that predicts compile cost.~~ Closed by a different fix. Node count still does not predict compile cost and probably cannot, so the bound is no longer asked to: a round waits a second for its kernel, takes the CPU if it is not ready, and installs the kernel whenever it arrives. Nothing is refused for having been slow once.
3. ~~Diagnose the secp256k1 cliff.~~ Closed 2026-08-26. Also a wrong proof, from two no-copy Metal buffers lent over one shared page.
4. Fold the results into `2026-08-21-verdict.md` — in particular, that the headline figure is a *large-workload* one and small workloads get about 3.5x whatever they compute.
5. Re-measure the M4 on a deliberately warmed shader cache, and settle the question above. One `SP1_METAL_COMPILE_BUDGET=1800` run per guest and per mode, then the three compressed points and the `divrem` and `keccak` rows at the default budget. Those two guests are the other candidates: the gap between the cold `bench-standard` and the warm `bench-full` was already attributed to cache warmth, and a run warmed on purpose would say whether that attribution had the magnitude right. If the compressed figures move, `2026-08-21-verdict.md` and the compressed claims in `README.md` and `USING.md` move with them. While the machine is set up for it, spot-check two or three points cool as well: the M4 tiers were run with no pause between proofs at all, so they carry whatever thermal offset this machine has, and nothing recorded at the time would show it. Within-point drift is absent from that data -- no point of forty-eight is more than 5% slower by its last repeat -- but a uniform offset would not appear that way, and comparing a cool M5 against a warm M4 would overstate the newer machine.
   **The application tier makes this urgent rather than tidy.** Its M5 re-run was paced and its M4 reference was not, and there the two arms do *not* move together: the CPU arm gains a median 8.7% and the GPU arm 13.5%, lifting the Metal advantage from 2.98x to 3.17x at every point but three. That is either a real property of the newer GPU on long workloads or the M4's own heat inflating the baseline, and the apps guests are the longest in the suite, so they had the most room to throttle. The microbenchmark tiers show no such asymmetry, which argues against throttling but not decisively, since their runs are a fraction as long. Re-take the application tier on the M4 under the same regime the M5 used — `COOL_FRACTION = 1.0` and the `nominal` gate — and the question closes.
