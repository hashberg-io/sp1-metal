# The M5 Max run

Everything in this project was measured on an M4 Max until 2026-08-28, when the suite was run on an M5 Max — MacBook Pro Mac17,7, 18 cores (6 Super and 12 Performance), 128 GB, macOS 26.6.2.
This file says what the two machines differ by, which of the three tiers is fit to quote, and why one of them is not.

**Both machines report `Darwin 25.6.0 arm64`**, so the header line inside each JSON Lines file does not distinguish them.
The file name is what does: `-m5-max` is this machine, and the unsuffixed files are the M4 Max.

## Which tiers are fit to quote

- **`bench-standard-m5-max`** — usable. 204 timed runs, 34 points, no failures.
- **`bench-full-m5-max`** — usable with a caveat. 366 timed runs, 64 points, six failures, every one of them the deliberately out-of-range `nodebound=1e6` configuration that fails the same way on the M4. It ran warmer than the standard tier and reads a few per cent slow: `fib/standard` on the GPU arm is 9.20 s in the standard tier and 10.11 s here, against 8.87 s measured by hand on a cold machine.
- **`bench-apps-m5-max`** — usable. 144 timed runs, 24 points, no failures, every proof verified, and every run started with the machine reading `nominal`.
- **`bench-apps-m5-max-thermally-contaminated`** — **not usable, and quoted nowhere.** The first attempt at the application tier, thermally contaminated from end to end. It is kept under that name because the record of a ruined run is worth more than its absence, and because the thermal fields in it are what diagnosed the problem. `bench-apps-m5-max` is the re-run, taken under the longer pauses and the `nominal` gate.

Three points from the two microbenchmark tiers were re-taken cool to check them, and they stand.
Their absolute times run up to a tenth slow in the full tier, but the speedups those times produce are within 4.5% of the cool readings, because mild heat warms both arms together and cancels in the ratio:

| point            | cool  | standard tier   | full tier       |
| ---------------- | ----- | --------------- | --------------- |
| fib/standard     | 5.54x | 5.79x (+4.5%)   | 5.45x (-1.6%)   |
| bits/large       | 6.06x | 5.91x (-2.4%)   | 5.97x (-1.4%)   |
| sha256/standard  | 4.03x | 3.96x (-1.7%)   | 3.99x (-0.8%)   |

The application tier failed where these did not because its heat was severe enough to hurt the two arms *unequally* — 39% on one CPU arm against 3% on its GPU arm — and an error like that does not cancel.

## The headline

**The M5 Max is 8 to 10 per cent faster than the M4 Max on both arms, and the Metal advantage is unchanged.**
This is a clock-and-bandwidth step, not a change in what the backend can extract from the machine.

| reading                       | M4 Max          | M5 Max          |
| ----------------------------- | --------------- | --------------- |
| CPU arm, relative             | 1.00x           | **1.10x**       |
| GPU arm, relative             | 1.00x           | **1.08x**       |
| GPU over CPU, median          | 3.93x           | 3.90x           |
| GPU over CPU, range           | 2.16x – 5.97x   | 2.23x – 5.97x   |
| best point                    | bits/large 5.97x | bits/large 5.97x |
| worst point                   | keccak/large 2.16x | keccak/large 2.23x |

The floor rose and the ceiling did not, which is the whole shape of the difference.
`keccak` gains most on the GPU arm because its kernels are compile-bound; `mulh` is the one consistent GPU-side regression, from 5.45x to 5.11x at its large size.

## Everything that reproduced

The structural findings are the point of the full tier, and all of them hold on the new machine.

**The phase ablation still does not sum**, and in the same order. `fib/standard`, one phase forced off at a time, against a 10.11 s baseline:

| phase off    | M4 cost   | M5 cost   |
| ------------ | --------- | --------- |
| MERKLE       | +13.53 s  | +13.39 s  |
| ZEROCHECK    | +6.86 s   | +5.03 s   |
| JAGGED       | +3.99 s   | +4.34 s   |
| TRANSITION   | +2.41 s   | +2.13 s   |
| ENCODE       | +1.91 s   | +1.52 s   |
| FIRST_LAYER  | +1.18 s   | +0.56 s   |

Those sum to 26.97 s against a whole GPU path worth 44.96 s — 60%, against the M4's 61%, and the Merkle tree is still the largest single thing this backend does.

**Shard size is still flat**: 5.60x, 5.54x, 5.45x, 5.65x, 5.73x at 1, 2, 4, 8 and 16 MiB.
**Concurrency still only matters below four**: 3.89x, 5.08x, 5.45x, 5.72x at 1, 2, 4 and 8 permits.
**The raised node bound is still catastrophic**, failing all six of its runs while the CPU arms are unmoved.

## Compressed mode, and a question it raises about the M4

| point           | M4    | M5    | CPU gain | GPU gain |
| --------------- | ----- | ----- | -------- | -------- |
| fib/small       | 3.09x | 3.70x | 1.07x    | 1.28x    |
| sha256/small    | 2.99x | 3.45x | 1.11x    | 1.28x    |
| secp256k1/small | 2.90x | 4.01x | 1.07x    | 1.48x    |

These are the only points in the matrix whose GPU gain exceeds 1.17x, and their CPU arms move by the same tenth as everything else.
That asymmetry is much more likely to be a shader cache that was warm for the core guests and cold for the recursion AIRs than a generational step that reaches recursion alone.
It cannot be settled from this machine; see the action recorded in [benchmark-status.md](../knowledge/benchmark-status.md).
**D7 is not met either way** — its bar is 5x and the warm figures reach 3.45x to 4.01x.

## Why the first application tier was thrown away

It ran three hours of the heaviest workloads in the suite with a pause of a quarter of each run's wall clock between them, and read `fair` for 136 of its 144 runs.
Every signal the harness recorded said it was fine.
Four points re-taken hours later on an idle machine reading `nominal`:

| point                    | in the tier | re-taken cool | tier error |
| ------------------------ | ----------- | ------------- | ---------- |
| eth-ecrecover/small, CPU | 19.90 s     | 14.31 s       | +39%       |
| eth-mpt/large, CPU       | 50.10 s     | 39.62 s       | +26%       |
| eth-mpt/large, GPU       | 19.34 s     | 14.32 s       | +35%       |
| eth-ecrecover/small, GPU | 4.02 s      | 3.91 s        | +3%        |

The arms are not hurt equally, so the errors do not cancel: `eth-ecrecover/small` was inflated from a true 3.66x to a reported 4.96x, and `eth-mpt/large` deflated from 2.77x to 2.59x.
A contaminated tier does not read as noisy — it reads as a finding, which is what makes this worth the space.
[thermal.md](../knowledge/thermal.md) has the full account and what the suite does about it now.

## The application tier

The re-run took 1.84 hours of proving and 2.22 hours of waiting: a pause equal to each run's own wall clock, and then a hold until the machine read `nominal` again.
The pauses cost more than the work, which is the price of a number that means anything on a laptop.

**Here, unlike in the microbenchmarks, the two arms did not move together.**
The CPU arm is a median 8.7% faster than the M4 and the GPU arm 13.5%, so the Metal advantage rises from a median 2.98x to 3.17x, and it rises at every one of the twenty-four points but three.

Two readings fit that, and this machine cannot separate them:

- **The M4 apps figures are throttled.** They were taken with no pause between proofs, and these are the longest workloads in the suite — `eth-bls-verify/large` is a five-minute CPU run. Whatever thermal offset that machine carries, it carries most of it here.
- **The M5's GPU path genuinely gains more on this shape of work.** Against this: the standard and full tiers were also unpaced on the M4 and show no such asymmetry at all. For it: their runs are a fraction as long, so they had less room to heat.

The paced M4 re-take recorded in [benchmark-status.md](../knowledge/benchmark-status.md) settles it. Until then the M5 column stands on its own and the gain column does not.

**The CPU arm heats itself inside a single run, and no pause can fix that.**
Twenty-seven of the twenty-eight runs that finished at `fair` were CPU-arm, and all eighteen CPU runs of `eth-pairing-check` and `eth-bls-verify` began at `nominal` and ended at `fair`.
There is no cool number to be had for a four-minute all-core proof on a laptop: what is measured is what a user gets, but it is not a ratio against a CPU that stays cool, and the multipliers for those two guests are flattered by however much their own heat cost them.

| guest | size | N | cpu M4 | cpu M5 | gpu M4 | gpu M5 | M4x | M5x |
|---|---|---|---|---|---|---|---|---|
| btc-headers | small | 400 | 20.76 | 18.73 | 6.04 | 5.49 | 3.44x | 3.41x |
| btc-headers | standard | 1600 | 58.76 | 51.25 | 10.59 | 9.60 | 5.55x | 5.34x |
| btc-headers | large | 3000 | 102.76 | 94.60 | 17.88 | 15.79 | 5.75x | 5.99x |
| btc-schnorr | small | 2 | 14.92 | 13.89 | 4.18 | 3.92 | 3.56x | 3.54x |
| btc-schnorr | standard | 10 | 24.63 | 22.13 | 7.95 | 7.07 | 3.10x | 3.13x |
| btc-schnorr | large | 20 | 35.37 | 31.83 | 12.43 | 10.44 | 2.85x | 3.05x |
| eth-ecrecover | small | 2 | 16.01 | 14.89 | 4.34 | 3.93 | 3.69x | 3.79x |
| eth-ecrecover | standard | 8 | 26.71 | 23.87 | 8.51 | 7.16 | 3.14x | 3.34x |
| eth-ecrecover | large | 16 | 40.38 | 36.22 | 13.97 | 11.42 | 2.89x | 3.17x |
| eth-mpt | small | 10 | 16.86 | 15.37 | 6.28 | 5.43 | 2.68x | 2.83x |
| eth-mpt | standard | 20 | 26.45 | 23.90 | 10.65 | 8.97 | 2.48x | 2.67x |
| eth-mpt | large | 40 | 44.42 | 40.14 | 16.52 | 13.98 | 2.69x | 2.87x |
| eth-groth16-inputs | small | 2 | 16.93 | 15.07 | 5.17 | 4.77 | 3.27x | 3.16x |
| eth-groth16-inputs | standard | 8 | 26.92 | 24.21 | 9.56 | 8.08 | 2.82x | 3.00x |
| eth-groth16-inputs | large | 16 | 38.29 | 34.47 | 15.43 | 12.75 | 2.48x | 2.70x |
| eth-pairing-check | small | 1 | 112.78 | 104.99 | 40.29 | 34.88 | 2.80x | 3.01x |
| eth-pairing-check | standard | 4 | 176.83 | 168.00 | 60.36 | 52.20 | 2.93x | 3.22x |
| eth-pairing-check | large | 8 | 260.81 | 248.41 | 90.51 | 77.92 | 2.88x | 3.19x |
| eth-bls-aggregate | small | 8 | 17.50 | 16.25 | 5.79 | 5.17 | 3.03x | 3.14x |
| eth-bls-aggregate | standard | 16 | 21.80 | 20.00 | 7.68 | 6.63 | 2.84x | 3.02x |
| eth-bls-aggregate | large | 32 | 30.08 | 27.51 | 11.23 | 9.64 | 2.68x | 2.85x |
| eth-bls-verify | small | 1 | 174.35 | 163.61 | 56.47 | 48.44 | 3.09x | 3.38x |
| eth-bls-verify | standard | 2 | 210.54 | 205.34 | 69.64 | 62.56 | 3.02x | 3.28x |
| eth-bls-verify | large | 4 | 286.65 | 279.56 | 93.98 | 84.13 | 3.05x | 3.32x |

`btc-headers` is the outlier it was on the M4 — 5.99x at three thousand headers, roughly double the rest — because its GPU-side memory grows far more slowly than its CPU-side does, so it keeps the device fed long after the others stop filling it.
The two long cryptographic guests sit at a flat 3.2-3.3x across all three sizes, unmoved by N, which is what a workload dominated by a fixed pairing cost should do.

## The measurements predate the hand-off fix, and survive it

Every figure in this file was taken before the buffer hand-off fix, which landed after the last of them.
That fix changes the Merkle commit path — the largest single phase in the GPU proof — so it is the one change that could plausibly have moved these numbers.

Four points re-taken against the fixed backend, three repeats each, all at `nominal`, in `spotcheck-m5-max-postfix.jsonl`:

| point                          | CPU here | CPU published | GPU here | GPU published | x here | x published |
| ------------------------------ | -------- | ------------- | -------- | ------------- | ------ | ----------- |
| fib/standard                   | 53.02 s  | 55.07 s       | 9.20 s   | 10.11 s       | 5.76x  | 5.45x       |
| eth-mpt/large                  | 41.29 s  | 40.14 s       | 14.16 s  | 13.98 s       | 2.92x  | 2.87x       |
| eth-ecrecover/standard         | 24.74 s  | 23.87 s       | 7.28 s   | 7.16 s        | 3.40x  | 3.33x       |
| fib/small, compressed          | 26.01 s  | 25.99 s       | 6.98 s   | 7.03 s        | 3.73x  | 3.70x       |

Three of the four agree to within 3.6% on either arm, and the compressed point — chosen because the fix's own commit measured its orphaned hand-offs on compressed `fib` proofs, so it is where the new behaviour differs most — agrees to within 0.7%.
The dispatch counts are identical across all three repeats of every point and every proof verified.
`fib/standard` on the GPU arm reads 9% below its published figure because the published one comes from the full tier, which ran warm; it matches the standard tier's 9.20 s exactly.

**Nothing here needs re-running.** The fix is a correctness repair with no measurable cost.

## Setting the prover up on a new machine

Two things cost an hour each and are not in [USING.md](../USING.md).

- **`protoc` is required.** A machine with only Rust cannot build the host: `sp1-prover-types` has a build script that needs it, and fails with `Could not find protoc`. `brew install protobuf`.
- **The shader cache must be warmed deliberately.** Repeated runs at the default compile budget do not converge: `fib/standard` sat at 751 of its 1,311 zerocheck dispatches over five consecutive runs. One `SP1_METAL_COMPILE_BUDGET=1800` run per guest reaches the full count, after which `fib/standard` went from 12.5 s to 8.87 s — 30% that nothing in the run's own output announces. The cache then survives a reboot.

## The full comparison

Every point the two machines share, medians over three repeats, both warm.

| guest | size | config | cpu M4 | cpu M5 | cpu gain | gpu M4 | gpu M5 | gpu gain | M4x | M5x |
|---|---|---|---|---|---|---|---|---|---|---|
| bits | large | baseline | 89.98 | 86.30 | 1.04x | 15.08 | 14.45 | 1.04x | 5.97x | 5.97x |
| bits | small | baseline | 11.98 | 11.36 | 1.05x | 3.37 | 3.15 | 1.07x | 3.56x | 3.61x |
| bits | standard | baseline | 49.42 | 43.88 | 1.13x | 9.41 | 8.83 | 1.07x | 5.25x | 4.97x |
| bn254 | large | baseline | 7.79 | 7.88 | 0.99x | 2.22 | 2.18 | 1.02x | 3.51x | 3.61x |
| bn254 | small | baseline | 7.47 | 7.61 | 0.98x | 2.12 | 2.03 | 1.04x | 3.52x | 3.75x |
| bn254 | standard | baseline | 7.62 | 7.72 | 0.99x | 2.17 | 2.11 | 1.03x | 3.52x | 3.67x |
| bn254 | standard | nodebound=1e6 | 7.61 | 7.54 | 1.01x | - | - | - | - | - |
| branch | large | baseline | 107.97 | 103.21 | 1.05x | 18.91 | 17.99 | 1.05x | 5.71x | 5.74x |
| branch | small | baseline | 13.31 | 12.56 | 1.06x | 3.62 | 3.40 | 1.06x | 3.68x | 3.69x |
| branch | standard | baseline | 57.12 | 50.45 | 1.13x | 9.83 | 9.10 | 1.08x | 5.81x | 5.55x |
| divrem | large | baseline | 146.36 | 134.04 | 1.09x | 28.30 | 25.50 | 1.11x | 5.17x | 5.26x |
| divrem | small | baseline | 16.69 | 14.75 | 1.13x | 5.16 | 4.79 | 1.08x | 3.24x | 3.08x |
| divrem | standard | baseline | 75.00 | 63.72 | 1.18x | 14.58 | 12.86 | 1.13x | 5.14x | 4.96x |
| fib | large | baseline | 112.14 | 106.73 | 1.05x | 19.74 | 18.99 | 1.04x | 5.68x | 5.62x |
| fib | small | baseline | 13.69 | 12.84 | 1.07x | 3.65 | 3.48 | 1.05x | 3.75x | 3.69x |
| fib | small | compressed/compressed | 27.90 | 25.99 | 1.07x | 9.03 | 7.03 | 1.28x | 3.09x | 3.70x |
| fib | standard | ablate=ENCODE | - | - | - | 12.24 | 11.63 | 1.05x | - | - |
| fib | standard | ablate=FIRST_LAYER | - | - | - | 11.51 | 10.67 | 1.08x | - | - |
| fib | standard | ablate=JAGGED | - | - | - | 14.32 | 14.45 | 0.99x | - | - |
| fib | standard | ablate=MERKLE | - | - | - | 23.86 | 23.50 | 1.02x | - | - |
| fib | standard | ablate=TRANSITION | - | - | - | 12.74 | 12.24 | 1.04x | - | - |
| fib | standard | ablate=ZEROCHECK | - | - | - | 17.19 | 15.14 | 1.14x | - | - |
| fib | standard | baseline | 59.07 | 55.07 | 1.07x | 10.33 | 10.11 | 1.02x | 5.72x | 5.45x |
| fib | standard | permits=1 | 73.62 | 64.10 | 1.15x | 18.47 | 16.50 | 1.12x | 3.99x | 3.89x |
| fib | standard | permits=2 | 61.88 | 55.76 | 1.11x | 12.42 | 10.97 | 1.13x | 4.98x | 5.08x |
| fib | standard | permits=8 | 59.11 | 52.87 | 1.12x | 10.47 | 9.24 | 1.13x | 5.64x | 5.72x |
| fib | standard | shard=16MiB | 63.15 | 54.57 | 1.16x | 10.79 | 9.53 | 1.13x | 5.85x | 5.73x |
| fib | standard | shard=1MiB | 59.14 | 52.74 | 1.12x | 10.54 | 9.43 | 1.12x | 5.61x | 5.60x |
| fib | standard | shard=2MiB | 60.46 | 52.75 | 1.15x | 10.91 | 9.52 | 1.15x | 5.54x | 5.54x |
| fib | standard | shard=8MiB | 61.84 | 52.98 | 1.17x | 10.76 | 9.38 | 1.15x | 5.75x | 5.65x |
| keccak | large | baseline | 19.90 | 17.56 | 1.13x | 9.23 | 7.88 | 1.17x | 2.16x | 2.23x |
| keccak | small | baseline | 8.97 | 8.87 | 1.01x | 2.81 | 2.50 | 1.12x | 3.19x | 3.55x |
| keccak | standard | baseline | 13.25 | 12.61 | 1.05x | 5.59 | 4.80 | 1.16x | 2.37x | 2.63x |
| matmul | small | baseline | 12.82 | 12.15 | 1.06x | 3.62 | 3.34 | 1.09x | 3.54x | 3.64x |
| matmul | standard | baseline | 52.13 | 46.25 | 1.13x | 10.02 | 9.42 | 1.06x | 5.20x | 4.91x |
| mem | large | baseline | 52.17 | 46.11 | 1.13x | 10.06 | 9.01 | 1.12x | 5.18x | 5.12x |
| mem | small | baseline | 9.84 | 9.67 | 1.02x | 3.07 | 2.82 | 1.09x | 3.20x | 3.43x |
| mem | standard | baseline | 31.55 | 27.92 | 1.13x | 8.06 | 7.26 | 1.11x | 3.92x | 3.85x |
| merkle | small | baseline | 9.25 | 8.94 | 1.03x | 3.24 | 3.10 | 1.05x | 2.85x | 2.89x |
| merkle | standard | baseline | 17.81 | 15.94 | 1.12x | 5.57 | 5.12 | 1.09x | 3.20x | 3.11x |
| mulh | large | baseline | 118.81 | 108.41 | 1.10x | 21.81 | 21.22 | 1.03x | 5.45x | 5.11x |
| mulh | small | baseline | 14.22 | 12.91 | 1.10x | 3.86 | 3.66 | 1.05x | 3.68x | 3.52x |
| mulh | standard | baseline | 61.41 | 55.40 | 1.11x | 11.35 | 10.63 | 1.07x | 5.41x | 5.21x |
| secp256k1 | large | baseline | 7.71 | 7.81 | 0.99x | 2.20 | 2.05 | 1.07x | 3.51x | 3.81x |
| secp256k1 | small | baseline | 7.48 | 7.67 | 0.98x | 2.11 | 1.99 | 1.06x | 3.54x | 3.86x |
| secp256k1 | small | compressed/compressed | 25.33 | 23.73 | 1.07x | 8.73 | 5.92 | 1.48x | 2.90x | 4.01x |
| secp256k1 | standard | baseline | 7.60 | 7.71 | 0.99x | 2.12 | 2.03 | 1.04x | 3.59x | 3.80x |
| secp256k1 | standard | nodebound=1e6 | 7.59 | 7.42 | 1.02x | - | - | - | - | - |
| sha256 | large | baseline | 61.61 | 54.43 | 1.13x | 11.31 | 10.30 | 1.10x | 5.45x | 5.28x |
| sha256 | small | baseline | 14.49 | 13.19 | 1.10x | 4.24 | 3.92 | 1.08x | 3.42x | 3.36x |
| sha256 | small | compressed/compressed | 28.40 | 25.55 | 1.11x | 9.49 | 7.41 | 1.28x | 2.99x | 3.45x |
| sha256 | standard | baseline | 35.13 | 30.67 | 1.15x | 8.44 | 7.68 | 1.10x | 4.16x | 3.99x |
| sha256 | standard | permits=1 | 41.75 | 36.82 | 1.13x | 12.07 | 11.15 | 1.08x | 3.46x | 3.30x |
| sha256 | standard | permits=2 | 36.68 | 32.84 | 1.12x | 9.31 | 8.41 | 1.11x | 3.94x | 3.91x |
| sha256 | standard | permits=8 | 35.17 | 31.56 | 1.11x | 8.52 | 7.68 | 1.11x | 4.13x | 4.11x |
| sha256 | standard | shard=16MiB | 37.78 | 31.72 | 1.19x | 8.61 | 7.73 | 1.11x | 4.39x | 4.10x |
| sha256 | standard | shard=1MiB | 35.66 | 30.89 | 1.15x | 8.57 | 7.67 | 1.12x | 4.16x | 4.03x |
| sha256 | standard | shard=2MiB | 36.01 | 31.45 | 1.14x | 8.67 | 7.62 | 1.14x | 4.15x | 4.13x |
| sha256 | standard | shard=8MiB | 35.95 | 31.50 | 1.14x | 8.63 | 7.79 | 1.11x | 4.17x | 4.04x |
| sort | small | baseline | 12.99 | 12.38 | 1.05x | 3.77 | 3.50 | 1.08x | 3.44x | 3.54x |
| sort | standard | baseline | 64.95 | 59.67 | 1.09x | 11.77 | 11.75 | 1.00x | 5.52x | 5.08x |
| uint256 | large | baseline | 7.62 | 7.84 | 0.97x | 2.19 | 2.19 | 1.00x | 3.48x | 3.59x |
| uint256 | small | baseline | 7.41 | 7.68 | 0.96x | 2.17 | 2.01 | 1.08x | 3.40x | 3.82x |
| uint256 | standard | baseline | 7.48 | 7.67 | 0.98x | 2.16 | 2.06 | 1.05x | 3.46x | 3.72x |

64 shared points (0 only in M4, 0 only in M5)
CPU arm M5 vs M4: median 1.10x  range 0.96x-1.19x
GPU arm M5 vs M4: median 1.08x  range 0.99x-1.48x
GPU-over-CPU: M4 median 3.93x (2.16-5.97), M5 median 3.90x (2.23-5.97)
