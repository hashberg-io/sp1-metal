# A proof that verified and a process that never exited

> **Found and fixed 2026-08-27.** One loop in `sp1_metal_core::stats::busy_histogram`. Not in the backend, and not a correctness defect.

## What it looked like

```sh
SP1_METAL_MIN_ENTRIES=999999999 SP1_METAL_BATCH=1 \
  SP1_METAL_{MERKLE,TRANSITION,ZEROCHECK,UPLOAD_CACHE,JAGGED,FIRST_LAYER,JAGGED_VALUES,JAGGED_ASSIST,ENCODE}=0 \
  ... --elf guests/secp256k1/elf/bench-secp256k1 --stdin-u32 500 --mode core
```

About two runs in three never finished. `SP1_METAL_PERMITS=1` made it stop happening, 6 for 6, which said cross-thread; every upload policy failed and succeeded alike, which said it was not the lending defect of `gkr-overlapping-lends.md`.

**All of that was reasoning about the wrong thing.** The first line of a "hung" run's log is:

```
proved and verified in 6.9216535s (mode: core)
```

followed by every table the host prints. The proof was fine. The process simply never exited, and the harness — which captured output only after killing the process group — lost the buffered lines and reported silence.

## What it was

`sample` put the main thread in `sp1_metal_core::stats::busy_histogram`, called from `report_metal`, at the top of stack for 100% of the samples. Not parked: spinning.

```rust
while lo < hi {
    let b = (((lo - start) / width) as usize).min(buckets - 1);
    let edge = start + (b + 1) as f64 * width;
    let take = hi.min(edge) - lo;
    out[b] += take / width;
    lo += take.max(width * 1e-9);   // "guaranteed progress"
}
```

A bucket boundary is named twice and by different arithmetic — divide to get the index, multiply to get the edge — and the two do not always round to the same place. When the edge lands at or just below the cursor, `take` is zero or negative, and the step that exists to guarantee progress is **a billionth of a bucket**. Crossing one bucket then needs a billion iterations.

It depends on the exact interval boundaries, which depend on which threads dispatched when, which is why it looked like a race and why one shard at a time avoided it.

## The fix

Walk the buckets instead of walking a cursor. The range of buckets an interval touches is two divisions, and the loop is over an integer range, so it terminates by construction and no boundary is named twice.

16 runs of the reproduction above, 16 verified, where before it was about one in three.

## What this cost, and the lesson

Half a day of treating a reporting bug as a correctness defect, because the symptom — a process that stops making progress — is the same symptom the two real correctness defects wore. The difference is one line of the log, and the harness threw it away.

**Read the log before the stack, and make the harness keep the output of a run it kills.** `subprocess` after `SIGKILL` returns what it has, which for a pipe that was never drained is nothing.
