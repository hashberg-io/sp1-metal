# The batched small-chip path fails intermittently when every chip goes through it

> **Found 2026-08-27. Open — not diagnosed, not fixed.** Not reachable from the shipped defaults; see below for exactly what it takes.

## What happens

```sh
SP1_METAL_MIN_ENTRIES=999999999 SP1_METAL_BATCH=1 \
  SP1_METAL_{MERKLE,TRANSITION,ZEROCHECK,UPLOAD_CACHE,JAGGED,FIRST_LAYER,JAGGED_VALUES,JAGGED_ASSIST,ENCODE}=0 \
  RUST_LOG=error crates/sp1-profile-host/target/release/sp1-profile-host \
  --elf guests/secp256k1/elf/bench-secp256k1 --stdin-u32 500 --mode core
```

Roughly two runs in three do not finish. The rest verify in about 6.5 s.

| n | outcome |
|---|---|
| 1200 | verified 6.58 s, then TIMEOUT, TIMEOUT, TIMEOUT |
| 500 | TIMEOUT, then verified 7.14 s, TIMEOUT, verified 6.49 s |

## Why this configuration and not the default

`worth_dispatching` takes a different branch when `SP1_METAL_BATCH` is on: it compares the layer's total against `SP1_METAL_MIN_LAYER` (64) rather than against `SP1_METAL_MIN_ENTRIES`. So with `MIN_ENTRIES` set very high **and** batching on, every layer is still dispatched while every chip in it is below the per-chip threshold — which sends *every* chip, however wide, through `stage_small_batch`.

The shipped default (`MIN_ENTRIES=16384`) never does that: a chip over 16 KB takes the per-chip dispatch and only the genuinely small ones batch. This regime is produced by the gate bisection in `tools/`, not by proving.

**It does not reproduce on the default path.** The 2026-08-27 standard tier is 204 runs with every gate on and every proof verified, and a size ladder over secp256k1 from n=1000 to 4000 verifies at every point.

## What has been ruled out

- **Not the lending defect fixed in `gkr-overlapping-lends.md`.** It fails and succeeds under all three upload policies: default, `SP1_METAL_UPLOAD_FRESH=1` (every upload a private copy), and `SP1_METAL_UPLOAD_LEND=0`.
- **Not the round reduction computing the wrong values**, as far as one sample goes: `SP1_METAL_GKR_ROUND=check` verified in 9.33 s. Since the failure is intermittent, one passing sample is weak evidence — repeat it.

Intermittency under every deterministic setting points at a race rather than at arithmetic.

## Where to start

1. Repeat `SP1_METAL_GKR_ROUND=check` until it either fires or passes ten times. That mode also compares the CPU round *before* the GPU round with the one after, so it catches a reduction that damages what it was lent as well as one that returns the wrong answer.
2. `SP1_METAL_PERMITS=1`. If serialising the shards hides it, it is cross-thread; if not, it is inside one layer's dispatch.
3. Read `stage_small_batch` for what changes when a chip is large: the partials it takes, the per-chip offsets it packs, and `MAX_GROUPS`.
4. `RUST_LOG=error` first, always. The two correctness defects this backend has had both read as hangs from outside and both said so only in the log.
