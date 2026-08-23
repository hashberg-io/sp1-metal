# What to try next, in order

Written 2026-08-23, against the state after the three ports of `cpu-plan.md`: **17.0 s against a 63.6 s CPU arm, 3.74x**, 76 user and 50 system CPU-seconds, the device busy 27.5% of the proof.
Every figure below is measured on the acceptance workload — fibonacci n = 2,000,000, `SHARD_SIZE` 4 MiB, four shards — with the same binary switched.

## The correction this list turns on

`cpu-plan.md` ruled the Reed-Solomon encode out on a probe reading **1.02x**, and the check made of that number — whether the encoder that ships is packed — came back yes, which seemed to settle it. It did not. That ratio is for **one butterfly stage** against a hand-written packed stage. `results/encode.json` holds three figures, and only the first two were ever quoted:

| | ns per input cell | against what ships |
|---|---:|---:|
| slop's encoder, `p3` `Radix2DitParallel`, rayon on all cores | 6.78 | 1.0x |
| a hand-written packed CPU encode, rayon on all cores | 0.89 | 7.6x |
| **a full Metal encode, written and parity-checked** | **0.517** | **13.1x** |

Being packed and being fast are different things: `Radix2DitParallel` *is* packed and is still 7.6x slower than a blocked packed implementation, because it makes sixteen passes over the codeword with bit reversal and transposes between them. The lesson is the project's usual one in a new costume — **check what a ratio's denominator is before letting it close a question** — and it cost the encode being set aside twice.

## The list

### 1. The Reed-Solomon encode — ~25 thread-seconds, 23% of what is left

The largest portable block and the best-evidenced payoff on this list, since the kernel already exists in the probe suite and is parity-checked against slop element for element, at a shape with and without the tiled tail.

It is also on the per-shard critical path: `commit traces` is 3.99 s per shard, the Merkle half is already on the GPU, and measurements put the phase at about 53% encode by cost — so nearly all of what is left there is this.

Two estimates disagree, and that is worth knowing before starting. The exchange rate observed so far is about 22 CPU-seconds per 0.4-1.0 s of wall, which predicts **17.0 → 16.0-16.5 s**. Its position on the critical path predicts up to 3 s off the slowest shard, which predicts **14-15 s**. Resolve which by measuring the slowest shard's `commit traces` span, not the total.

**Estimated: 17.0 → 14-16.5 s, 4.1x - 4.5x.** *Stop if* `commit traces` does not fall below about 1.5 s per shard.

The obstacle is that the encoder is a concrete `Arc<Radix2DitParallel>` inside the basefold prover rather than a type parameter, so unlike the commitment this cannot be reached forklessly; it needs `slop-basefold-prover` vendored, or the encode intercepted where the trace is handed over.

### 2. Folding MLEs — 28.5 thread-seconds, the largest single block, owner unknown

Bigger than the encode, and unrankable until it is attributed. Rayon steals the stack for 22 of the 28.5 seconds, so sampling cannot say who is asking; and a threshold sweep (16384 → 512 entries, no change in wall or CPU) rules out the small-chip fallback, which was the obvious guess.

**The first action is a counter at each call site, not another profile.** Sampling has now failed at this twice, and the call sites are few: `logup_gkr::metal::stage_fold`, `generate_gkr_circuit`, `zerocheck_fix_last_variable`, `LongMle::fix_last_variable`, and whatever basefold does.

One gap is already known: `LongMle::fix_last_variable` only reaches the GPU when a component has a single polynomial (`guts().sizes()[1] == 1`), and falls to the CPU otherwise.

**Estimated: unknown, potentially the largest on this list.** Cheap to find out.

### 3. System time — 50 of the 126 CPU-seconds, about 40%

Never targeted, and nearly as large as all the user CPU work that is left. Earlier measurement had it tracking the process's memory footprint at roughly 0.7 GB per system-second on both arms, which makes it first-touch page faults rather than anything Metal does; and batching the GKR chips cut it 14 seconds as a side effect, purely by not allocating on the CPU path.

The attack is allocation churn, not a kernel: reuse the buffers a layer folds into instead of allocating each round.

**Estimated: 0.5-1.5 s of wall**, on the precedent of that 14-second side effect.

### 4. Dispatch blocking — 7.4 thread-seconds over about 3,000 command buffers

A layer's round reduction and its fold are separate command buffers with a host wait between them, though the second only needs the first's data, not its result. Chaining them, and not waiting until the round message is actually read, would remove most of it.

**Estimated: 0.3-0.6 s.**

### 5. The basefold Dense PCS proof — about 11 thread-seconds

The remaining occupant of `prove evaluation claims`, and the reason that phase is still 3.83 s per shard against `cpu-plan.md`'s target of 2.5. Needs `slop-basefold-prover` vendored, which item 1 may need anyway — worth doing together if item 1 goes that way.

**Estimated: 0.3-0.6 s.**

### 6. The tail and the startup — about 9 of the 17 seconds

Roughly 4 s of startup at 1-3 of 16 cores, and about 5 s of tail as the last shards finish. Neither is CPU-throughput-bound, so no port touches them: they need shards to overlap better, and the cap is `ProverSemaphore::new(4)` in `sp1-prover`, which is not vendored.

**This is the floor.** It bounds everything above at roughly 7x however much else moves, and it is the only item here that is scheduling rather than arithmetic.

## What is not on the list, and why

- **Trace generation and guest execution, about 20 thread-seconds.** Not field arithmetic and not GPU-shaped.
- **Rayon and allocator overhead, about 10 thread-seconds.** Real, but the fix is fewer and larger parallel tasks, not a kernel.
- **Making the existing kernels faster.** The device is busy 27.5% of the proof; a kernel that is twice as fast buys nothing while the CPU is the constraint. Coverage, then scheduling, then tuning — in that order.

## The arithmetic of the whole list

Items 1 to 5 are about 72 thread-seconds. At the exchange rate the last three ports actually achieved — 46 user-CPU-seconds for 1.5 s of wall — that is about 2 s and **4.2x**. Only if item 1's position on the critical path beats that rate does this reach **4.5x - 5x**. Item 6 says nothing here goes past about 7x.
