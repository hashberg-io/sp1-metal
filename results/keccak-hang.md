> **Resolved 2026-08-26.** This was never a hang. See `results/zerocheck-invalid-proofs.md`: the GPU zerocheck computed the wrong round polynomial for five recursion AIRs, SP1 retried until its budget was gone, and the parked threads that remained read as a hang. Kept for the diagnosis, and because the compile deadlock and the two missing autorelease pools found along the way were real and are fixed.

# An intermittent hang in the Metal path

Found by the benchmark suite on 2026-08-24, M4 Max, git 6d15661.

## What happens

Proving `guests/keccak` at n=2000, 4 MiB shards, with `SP1_METAL_ZEROCHECK_MAX_NODES=8000`, the Metal arm sometimes never finishes. The process sits at **0.0% CPU** with the machine 90% idle — it is not slow, it is stopped.

Three repeats of the identical point:

| repeat | CPU arm | Metal arm |
|---|---|---|
| 0 | 13.31 s | 5.31 s |
| 1 | 13.49 s | **hung; killed at 900 s** |
| 2 | 13.36 s | **hung** |

Same binary, same input, same environment, same machine. The CPU arm is unaffected and stable to within 1.4%.

## What it is not

- Not compilation. The kernels were compiled and cached during the warm-up run before any of these three, and repeat 0 proves it by finishing in 5.31 s.
- Not the node-bound pathology. That one burns CPU for ten minutes; this one burns none.
- Not machine contention. `top` reports 90% idle throughout, and this project's load-average phantom was ruled out separately.

## What it resembles

This project has seen one hang before, recorded in `knowledge/backend.md`: deferred command-buffer submission produced a state where every thread parked with nothing outstanding in Metal, and the shard retried until its retry budget was exhausted. That was traced to a per-thread queue against tokio task migration, and deferral has been off by default since — `SP1_METAL_DEFER` defaults to false, so it is not the cause here, but the shape is the same.

The suspicion worth testing first is therefore a lost wakeup between a command buffer's completion handler and the thread waiting on it, which would be rare, timing-dependent, and would present exactly like this.

## Why it matters

A hang is worse than a wrong answer that verification catches, because nothing catches it. It is intermittent enough to survive a smoke test and frequent enough — two runs in three — to be hit by anyone proving a Keccak workload, which is to say anyone proving EVM execution.

## Reproducing

```sh
for i in 1 2 3; do
  SHARD_SIZE=4194304 SP1_METAL_ZEROCHECK_MAX_NODES=8000 \
    crates/sp1-profile-host/target/release/sp1-profile-host \
    --elf guests/keccak/elf/bench-keccak --stdin-u32 2000 --mode core
done
```

Watch for a run that stops making progress; `ps -Ao etime,%cpu,command | grep sp1-profile-host` will show it at 0% CPU.

## It is not keccak-specific

Later the same day, the full tier reached `compressed` mode and `guests/fib --mode compressed` stalled the same way: **0.0% CPU for over sixteen minutes** with the machine 98% idle, on the warm-up run before any timed repeat.

That widens the finding considerably. The hang was first seen on keccak at n>=2000 in `core` mode, and keccak's small size never hung; now it appears on the simplest guest in the suite in a different proof mode. So the trigger is not the Keccak chips, not the workload size, and not the node bound. Whatever it is, `fib` in `core` mode has proved through it hundreds of times without incident, which is why it took a fourteen-guest suite across two proof modes to surface.

Two consequences for whoever picks this up:

- **`compressed` mode is unmeasured because of it.** That is the mode D7's bar is actually written against, so the one number this project most needs is the one the hang blocks.
- **Look for something shared** by keccak-at-scale and fib-under-recursion rather than for something in either. A lost wakeup between a command buffer's completion handler and the thread awaiting it fits both, and fits the earlier deferral hang recorded in `knowledge/backend.md`.
