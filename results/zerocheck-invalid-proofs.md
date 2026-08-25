# The GPU zerocheck computes wrong values for the recursion AIRs

Found 2026-08-26 while chasing what looked like a hang. It is not a hang. It is a **correctness defect**, and it is the most serious thing this project has found.

## What happens

`--mode compressed` on the Metal path never completes. The reason is in the log, and only in the log:

```
ERROR ProveShard: task failed with retryable error:
      lift/deferred verify failed: invalid shard proof
ERROR task ... exhausted its retry budget, failing it
```

The recursion step produces a proof that does not verify. SP1 retries, exhausts its retry budget, and the process is left with every thread parked and nothing outstanding — which reads from the outside as a hang. A `sample` of the process shows 32 threads in `__psynch_cvwait`, nothing in Metal, nothing in this project's code.

## It is one phase, and the bisection is clean

Starting from all Metal switches off (which proves and verifies in 28.8 s) and turning exactly one on:

| switch on | result |
|---|---|
| `MERKLE` | verified, 15.6 s |
| `TRANSITION` | verified, 23.3 s |
| `UPLOAD_CACHE` | verified, 23.0 s |
| `JAGGED` | verified, 21.8 s |
| `FIRST_LAYER` | verified, 22.9 s |
| `JAGGED_VALUES` | verified, 23.0 s |
| `JAGGED_ASSIST` | verified, 23.1 s |
| `BATCH` | verified, 22.2 s |
| `ENCODE` | verified, 22.1 s |
| **`ZEROCHECK`** | **8 invalid shard proofs, never completes** |

Nine phases are right. One is wrong.

## Why nobody saw it

**Core-mode proofs are unaffected.** The benchmark suite proved and verified 616 of them across fourteen guests without a single failure. The zerocheck kernel is generated per chip from that chip's constraint DAG, and every chip a *core* proof uses produces a correct kernel.

Recursion uses different AIRs. Those chips have never been exercised by anything in this repository, and at least one of them transpiles to a kernel that computes the wrong thing.

**And the phase that is wrong is the only phase with no `=check` mode.** `ENCODE`, `FIRST_LAYER`, `BATCH`, `JAGGED_VALUES` and `JAGGED_ASSIST` each run both paths on every call and compare element for element under `=check`. Zerocheck has no such mode. The five phases that can check themselves are correct; the one that cannot is not. That is not a coincidence worth ignoring.

## It is not caused by the fixes in this branch

Tested against the code with the compile lock and autorelease pools reverted: the same `invalid shard proof`, identically. The defect predates them.

## What to do

1. **Give zerocheck a `=check` mode**, as the other five phases have. It is the instrument that would have caught this before it shipped, and it is the instrument that will identify which chip is wrong. This is the first thing to build.
2. With that, run `SP1_METAL_ZEROCHECK=check` over a compressed proof and let it name the chip and the row.
3. Until it is fixed, **`SP1_METAL_ZEROCHECK=0` is required for compressed proofs.** Core proofs are unaffected.

## Reproducing

```sh
RUST_LOG=error SHARD_SIZE=4194304 \
  crates/sp1-profile-host/target/release/sp1-profile-host \
  --elf guests/fib/elf/acceptance-guest --stdin-u32 250000 --mode compressed
```

Takes about three minutes to emit the first `invalid shard proof`. With `SP1_METAL_ZEROCHECK=0` added, the same command verifies in about 23 seconds.
