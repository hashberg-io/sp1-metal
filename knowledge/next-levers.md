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

---

# What happened

Written 2026-08-23, after working the list end to end.
**17.01 s to 13.68 s against a 61.46 s CPU arm, 3.74x to 4.49x**, peak resident memory 47.1 GB to 32.2, user CPU 76 s to 45 and system 50 s to 17.

Three of the six moved the number. One was already solved by another. Two were wrong about what they were, and finding that out is most of what this section is for.

| | what it was worth | what it turned out to be |
|---|---|---|
| 1. The encode | **17.01 → 15.57 s** | a better seam than the one predicted, and a kernel measured at the wrong blowup |
| 2. Folding MLEs | **15.57 → 14.62 s** | not 28.5 thread-seconds and not arithmetic: rayon overhead on tiny calls |
| 3. System time | already spent | 50 → 17 seconds as a side effect of item 1; the rest is not churn |
| 4. Dispatch blocking | nothing | the two buffers cannot be chained, and the layer fold was already batched |
| 5. The dense PCS proof | **14.62 → 13.68 s** | not FRI: the single-threaded batching before it |
| 6. The tail and startup | the floor | see below |

## 1. The encode — done, and the seam was in the other direction

The item said this needed `slop-basefold-prover` vendored, because `CpuDftEncoder` holds a concrete `Arc<Radix2DitParallel>` and there is nothing above it to substitute. That is true and it pointed the wrong way.

`Radix2DitParallel`'s only relevant property is that it implements `Dft`, and that impl is forty lines of `slop-dft` — four dependencies, one consumer in the whole prover. Replacing the body of one trait method reaches every encode a proof performs. The rule this is an instance of: **when a concrete type blocks substitution from above, look at what the concrete type itself is standing on.** A fork is cheapest at the narrowest point of the graph, not at the point where the problem was noticed.

The kernel then did not fire at all, because `eligible` refused every call. The probe suite had been written against `LOG_BLOWUP = 1` — "the blowup slop's default FRI config uses" — and the core prover's `CORE_LOG_BLOWUP` is 2. The codeword is four times the trace, not twice. So the encode's 13.1x had been measured at a shape the prover never asks for, on top of having been dismissed twice for a ratio whose denominator was one butterfly stage. Three separate mislabellings of the same probe.

The spread kernel now takes the blowup and does its top stages together, which reads the trace once and never touches the padding.

**What it was worth: 17.01 → 15.57 s, and `commit traces` 3.99 → 1.47 seconds a shard**, which was the stop condition. The larger result was not the time: **peak resident memory fell from 47.1 GB to 27.8**, because upstream builds each codeword in a fresh allocation and copies the trace into it, and this writes into the tensor that will hold it through a buffer that borrows its pages.

One trap worth writing down. A tensor from `Tensor::with_sizes_in` has the capacity but a **length of zero** — upstream fills it by extending, so it never has to say so. A kernel writing into one has to set that length itself, and forgetting is silent in the worst way: the codeword is correct in memory and every reader sees an empty slice.

## 2. Folding MLEs — the premise was wrong, and the counter said so in one run

The item insisted on a counter rather than a third profile, and was right to. It was wrong about everything the counter would find.

Every fold in the prover funnels through `mle_fix_last_variable`, so the counter went there rather than at the five call sites. Over the acceptance workload:

| width | calls | cells | thread-seconds | ns/cell |
|---|---:|---:|---:|---:|
| 17+ polynomials | 3,148 | 1,001,884,844 | 1.73 | 1.72 |
| 1 polynomial | 624 | 18,616,528 | 0.12 | 6.47 |
| 2-16 | 500 | 5,385,168 | 0.11 | 20.26 |
| **under 1024 cells** | **98,610** | **3,814,200** | **4.48** | **1175.66** |

Folding cost **6.4 thread-seconds, not 28.5**, and seventy percent of it was 98,610 calls averaging thirty-nine cells, running at seven hundred times the unit cost of the calls that carry the work. Sumcheck halves its polynomial every round, so its last rounds are a handful of rows and there are a great many of them — and every one was being handed to a work-stealing pool.

This is why two sampling profiles blamed rayon and could not say who was calling it. **Rayon really was where the time went.** The samples were not being stolen from the caller; there was no caller-side work to attribute. Small folds now run inline and the large ones use `par_chunks` instead of `chunks(..).par_bridge()`, which pulls a sequential iterator through a mutex and cannot split.

**What it was worth: 15.57 → 14.62 s**, folding 6.44 → 2.16 thread-seconds, the tiny calls 4.48 → 0.096.

And it closes the item in the other direction too: there is no fold left worth a kernel. A billion cells at 1.63 ns each is 1.6 thread-seconds of a 13.7 second proof, and basefold's own fold and the even-index restriction are 0.11 between them, so nothing is hiding behind the funnel either.

## 3. System time — spent by item 1, and what is left is not churn

The item was written when system time was 50 of 126 CPU-seconds. After the encode moved it is **17 of 62**, and user CPU fell from 76 to 45 in the same step. The item's own diagnosis — that this tracks the process's memory footprint — is what made it fall: peak resident memory dropped by 19 GB and system time dropped with it.

What remains is 3.39 million page reclaims, about 54 GB of first-touch faulting against 175 GB the kernels write, so two thirds of the writes already land on pages that have been faulted once. That is the allocator reusing memory, not churning it.

Tested rather than argued: building the host against mimalloc instead of the system allocator. It is **decisively worse** — 19.72 s against 13.33, 14.3 million page reclaims against 3.39, peak resident 83 GB against 32. The feature is left in place, off by default, so this does not get re-litigated. The system allocator is doing the right thing for this workload and the remaining faults are on memory that has to exist.

## 4. Dispatch blocking — the mechanism does not exist

The item proposed chaining a layer's round reduction and its fold into one command buffer, "though the second only needs the first's data, not its result."

It needs the result. The round reduction produces the round message; the challenger absorbs it and samples `alpha`; the fold is by `alpha`. There is a verifier between the two buffers and no amount of restructuring removes it. And the other half of the proposal was already built: `fix_layer` folds a whole layer in one dispatch, so the round trips are already two per round and cannot be fewer.

The 5.3 thread-seconds of blocking are real, but they are not host round trips. The device is now busy **44% of the proof and 100% of several seconds of it**, with four proving threads submitting concurrently: most of that wait is queueing behind the other three. It is a symptom of the GPU becoming a constraint, which is a different problem with a different fix — see below.

## 5. The dense PCS proof — done, and it was not FRI

`Dense PCS evaluation proof` was the third largest span. Almost none of it was the FRI rounds: those commit through the Merkle kernels already, and `fold_even_odd` was parallel upstream.

It was the step before them. `FriCpuProver::batch` combines every MLE of a shard into one polynomial with a random linear combination, and did it on one thread — `iter_mut().zip_eq(..)`, once per polynomial batch. Every row of the accumulator is independent of every other; the loop was sequential because it was written as a fold. It is now one parallel pass over the rows with the batches iterated inside, which also touches the accumulator once rather than once per batch.

**What it was worth: 5.4 → 3.1 thread-seconds and 14.62 → 13.68 s.** No kernel, and the item's guess that it needed the same vendoring as item 1 was right about the need and wrong about the crate.

## 6. The tail and startup — still the floor, and now measured rather than estimated

Not reachable: the shard concurrency cap is `ProverSemaphore::new(4)` at `sp1-prover/src/worker/builder.rs:332`, a literal with no environment variable behind it, in a crate that would drag the whole recursion stack in with it. And for this workload it is not the binding constraint anyway — four shards, four permits.

But the floor itself can be measured directly, which the item only estimated. Proving a guest that runs a thousand steps — one shard, and nothing in it — takes **3.09 seconds**:

| | n = 1,000 | n = 250,000 | n = 2,000,000 |
|---|---:|---:|---:|
| wall | **3.09 s** | 5.26 s | 13.68 s |
| zerocheck | 2.2 | 2.5 | 8.5 |
| setup | 1.0 | 0.9 | 1.0 |
| pcs | 1.1 | 1.7 | 8.5 |

The startup is the smaller half of it. `setup` is a second and barely moves, but **zerocheck costs 2.2 thread-seconds on a shard with nothing in it** — because SP1 pads every shard to a fixed shape, so the smallest possible proof still runs a full-shape shard. The floor is not "startup plus tail", it is **one padded shard**, and it is 3.09 seconds.

That sharpens what everything above is bounded by. Of the 13.68 seconds at n = 2,000,000, about 3.1 are fixed; the part that scales with work is 10.6 against roughly 58 on the CPU, or **5.5x on the part a backend can touch**. The ratio the whole project reports is that number diluted by a constant, and the dilution is worth more than any remaining item on this list.

---

# The next list, from 4.49x

Written 2026-08-23 against **13.56 s paired with 61.46 s**, device busy 44.3%.

## What changed, and it changes the ranking

The previous list ruled out tuning the existing kernels, and gave a reason that has stopped being true: *"The device is busy 27.5% of the proof; a kernel that is twice as fast buys nothing while the CPU is the constraint."*

**The GPU is now oversubscribed.** Four proving threads want 10.09 GPU-seconds inside a 13.6 second proof and get a 6.78 second union, so **8.4 thread-seconds are spent waiting on the device** — more than folding was ever thought to cost.

| | dispatches | GPU s | wall s | blocked |
|---|---:|---:|---:|---:|
| commit | 137 | **4.91** | 6.27 | 1.36 |
| zerocheck | 1258 | 2.59 | 3.92 | 1.33 |
| gkr | 2304 | 1.75 | 5.39 | **3.64** |
| pcs | 408 | 0.84 | 2.92 | 2.08 |

Coverage, then scheduling, then tuning was the right order. Tuning is where the order arrives.

## 1. The encode kernel — 3.05 of the 10.09 GPU-seconds, 30% of all GPU time

Measured by differencing rather than attributed: `commit` GPU time is **4.91 s with the encode on and 1.86 s off**, and device busy goes 30.9% to 44.3%. One port made itself the largest GPU consumer in the proof.

It makes about **eighteen passes over a codeword four times the size of the trace** — the spread, one butterfly stage per global pass, and the tiled tail. A radix-4 stage does two stages per pass, taking sixteen global passes to eight: eighteen passes to ten. The transform is memory-bound, so passes are the cost.

**Estimated: 13.56 → 12.6-13.0 s, 4.7x-4.9x.** *Stop if* `commit` GPU seconds do not fall below about 3.5.

Cheaper alternative if radix-4 proves awkward: reshape the tile. 64 rows by 64 columns holds six stages in 16 KB; 256 by 16 holds eight; 32 KB of threadgroup memory holds nine.

## 2. Zerocheck's unattributed 4.6 thread-seconds — the largest block nobody has measured

`zerocheck` is the largest span at 8.5 thread-seconds and its Metal region accounts for 3.9. Nearly five thread-seconds are elsewhere in that span and nothing measures them.

**The first action is a counter, not a profile.** That instruction paid off twice on the last list and cost nothing both times. Candidates worth their own counters: building `ZeroCheckPoly`, the `partial_lagrange` and eq construction, and chips rejected by `MAX_NODES = 16_384` in the transpiler — but the point is to stop guessing between them.

**Estimated: unknown, potentially the largest here. Cheap to find out.**

## 3. Dispatch latency, framed correctly this time — 8.4 thread-seconds

The previous list's mechanism was wrong; the waiting is real. GKR blocks **1.58 ms per dispatch for 0.76 ms of GPU work**, across 2,304 dispatches. That is not host overhead, it is queueing behind three other proving threads on a device at 44%.

Which makes most of it downstream of item 1: shrink the GPU work and the queue drains. What is left after that is `pcs` — 408 dispatches for 0.84 GPU-seconds, 2 ms of wall each for 0.2 ms of work, the worst ratio in the table and the one place fewer and larger dispatches are still available.

**Estimated: 0.3-0.8 s, most of it as a side effect of item 1.**

## 4. `generate GKR circuit` — 4.4 thread-seconds

The first layer and the layer transitions are both on the GPU and 4.4 thread-seconds are still here. Same treatment as item 2: count before porting.

## 5. The jagged sumcheck's construction — 2.6 + 2.1 thread-seconds

`create jagged sumcheck poly` (2.6) and `prove evaluation claims` (2.1). The guts and the assist sumcheck are already ported, so what is left is building the polynomial rather than proving it.

## 6. The floor — 3.09 s, now 23% of the proof

Unchanged and still unreachable, but grown from a footnote to nearly a quarter of the wall clock. **It caps the absolute ratio at 19.9x and the realistic one at about 7x.** The difference from last time is that items 1 to 5 total roughly 2 s of wall, so the floor is now the largest single line in the budget.

## Not on the list, and why

- **Raising `SP1_METAL_MIN_LAYER`.** Tested rather than assumed: 64 entries is 13.56 s, 1,024 is 13.93, 16,384 is 14.19. The gate is at its optimum, and raising it hands work back to the CPU faster than it saves round trips.
- **Chaining a round to its fold.** Impossible; the challenger stands between them.
- **Folding MLEs.** Done. 2.2 thread-seconds, of which 1.6 is a billion cells of real arithmetic.
- **Allocation churn.** Measured and answered: mimalloc was 45% slower.
- **Tracegen and execution.** 1.8 thread-seconds, and not GPU-shaped.

## The arithmetic

Items 1, 3 and 5 are worth roughly 1.5-2 s between them, landing at **4.9x-5.5x**. Item 2 is the only one that could beat that, and the only honest thing to say about it is that nobody has measured it yet. Item 6 says none of it goes past about 7x.

---

# What happened, the second time

Written 2026-08-23, after working the second list end to end.
**13.56 s to 11.16 s against a 60.03 s CPU arm, 4.49x to 5.38x.**

| | what it was worth | what it turned out to be |
|---|---|---|
| 1. The encode kernel | **13.56 → 13.13 s** | as predicted: eighteen passes became ten |
| 2. Zerocheck | **13.13 → 12.18 s** | not "unattributed": one table built a hundred times a round |
| 3. Dispatch latency | nothing directly | measured to be contention, and the sites are now visible |
| 4. `generate GKR circuit` | **12.18 → 12.00 s** | not the CPU fallbacks, which are 0.16% of the cells |
| 5. The jagged construction | **12.00 → 11.16 s** | three passes where one would do |
| 6. The floor | **3.09 → 1.86 s** | it was never structural |

## 1. The encode kernel — as predicted, which is worth noting

The estimate was 12.6-13.0 s and the stop condition was `commit` GPU seconds below about 3.5. They landed at 3.52 and the wall at 13.13.

A radix-4 butterfly does two decimation-in-frequency stages for one pass at the same multiply count, because the second stage's twiddle on the odd half collapses to a fourth root of unity — the stride times the half-block is exactly a quarter of the codeword, whatever the stage. That root is already in the twiddle table, so it costs nothing to find. Eighteen passes became ten, the encode's GPU time 3.05 seconds became 1.66, and the device went from 44.3% busy to 40.8%.

The parity check gained three shapes, because the two it had only reached one of the paths: an odd number of stages leaves a radix-2 pass at the top, and a column count that is not a whole number of tiles skips the tiled tail altogether.

## 2. Zerocheck — the item was wrong about its own premise, and the counters said so in one run

The item said "nearly five thread-seconds are somewhere else under that span". They were not somewhere else. Four buckets that partition the span put the setup loop, the sumcheck driver and the opening assembly at **a millisecond between them**, and 92% of zerocheck in the round reduction — which is supposed to be on the GPU.

A tally of every reason the round reduction refuses the device found the rest: **one third of it was building the eq weights**, 1,606 tables against 1,258 dispatches. Upstream leaves a TODO on the line saying it could be computed once per sumcheck, and it is right for a reason worth stating — every chip of a shard folds the same point, so the hundred-odd chips of a round each build the same table, use it once and drop it. The driver walks them in order on one thread, so remembering the last one is enough. Tables built fell from 1,606 to 88, which is twenty-two rounds times four shards.

The same counters showed 348 round reductions with no kernel at all. Seven chips: Byte, Program and Range have no constraints in either round shape, and `Global` is over the node bound at 22,794 against 16,384. **Raising the bound to let it through costs 1,814 seconds of kernel compilation** and takes the proof from 12.0 seconds to 463.6, while the other sixty-one kernels compile in 0.04 seconds between them. The bound is not conservative; it is the edge of a cliff. What `Global` needs is the treatment the GKR first layer got — its constraints as a table one kernel walks — not a larger number.

## 3. Dispatch latency — measured rather than guessed at, and it is contention

With one shard proving, a GKR dispatch blocks **0.34 ms**; with four, **1.60**. The fixed part of a round trip is a third of a millisecond and the rest is three other threads asking for the same device. So the item is real but not separately actionable: it shrinks when GPU seconds shrink, which is items 1 and 2.

Each of the five GKR dispatch sites now counts its own round trips, which is how the two worst ratios became visible — and they are not where the GPU time is. The zerocheck column fold is 348 buffers blocking 3.93 ms apiece for 1.58 ms of work; the layer transition is 201 blocking 4.37 ms for half a millisecond. Both are loops the driver walks one chip at a time.

Sweeping the zerocheck fold's gate says the fold does belong on the device — turning it off costs 4% — and that moving the threshold within a sixteen-fold range moves nothing outside the run-to-run spread.

## 4. `generate GKR circuit` — the guess was wrong and cost nothing to disprove

Both loops that build the circuit walk a layer one chip at a time and decline the chips too small for a round trip, so the obvious suspicion was the declines. Counting says they handle **0.16% of the cells** — 75 first-layer chips and 1,959 transitions, 0.14 thread-seconds between them. Batching them would have bought nothing.

The time is around the dispatch, not in it, and two things were doing it.

Splitting a layer into its even and odd halves is written as four folds at alpha zero and alpha one, and **a fold at either end of the interpolation is a choice rather than a combination**: `alpha * (y - x) + x` is `x` at zero and `y` at one. Taking them as written costs a multiply and a subtraction per cell to compute a copy. Folds of the widest kind fell from 1.69 to 0.76 nanoseconds a cell.

And the first layer's two output arrays were filled before they were written, when every cell of both is written on both paths. The denominators are filled with ones rather than zeros, so it was a real pass over the largest arrays in the phase.

The ratio moved the *wrong* way here, 5.09x to 5.04x, because both changes are CPU-side and the same binary runs both arms. That is the honest direction for it to move.

## 5. The jagged construction — three passes where one would do

`create jagged sumcheck poly` splits cleanly in two, and it is not the jagged multilinear (0.5 thread-seconds, already on the GPU). It is laying the base polynomials end to end, which was **2.19** and made three passes over the whole polynomial: a transpose of each component into a fresh allocation, an append into a running buffer, and a transpose of the result — that last one of a one-row matrix, so it permutes nothing and copies everything. On one thread.

It now makes one pass. Each component transposes straight into its own range of the output, the ranges are carved out in advance so their disjointness is checked rather than asserted, and they run in parallel; what comes out is sliced rather than transposed. **2.19 to 0.42 thread-seconds.**

## 6. The floor — it was never structural

The previous list measured the floor at **3.09 seconds** and concluded it was one padded shard: a guest running a thousand steps still paid 2.2 thread-seconds of zerocheck, because SP1 pads every shard to a fixed shape.

That conclusion was wrong, and the way it was wrong is worth keeping. **The floor is now 1.86 seconds**, and the trivial proof's zerocheck is 0.8 thread-seconds rather than 2.2 — because the eq table it was rebuilding a hundred times a round is rebuilt once, and that is as true of an empty shard as of a full one. What looked like an irreducible per-shard cost was the same inefficiency, measured where nothing else was happening.

The lesson is not about zerocheck. **A floor measured on a small workload is only a floor if the work inside it has been looked at**, and this one had not been. The cap that is actually structural — `ProverSemaphore::new(4)` at `sp1-prover/src/worker/builder.rs:332`, a literal in an un-vendored crate — is untouched, and was never the binding constraint for four shards anyway.

Of the 11.16 seconds at n = 2,000,000, about 1.9 are floor and 9.3 scale, against roughly 58 on the CPU: **about 6.2x on the part a backend can touch**, and an absolute ceiling of 32x rather than 20.

### Acceptance

| axis | point | CPU | Metal | | was |
|---|---|---:|---:|---:|---:|
| shard size | 1 MiB | 59.64 s | 10.95 s | **5.449x** | 4.507x |
| | 2 MiB | 61.12 s | 11.46 s | 5.333x | 4.489x |
| | 4 MiB | 60.34 s | 11.11 s | 5.431x | 4.637x |
| | 8 MiB | 60.22 s | 11.28 s | 5.339x | 4.558x |
| | 16 MiB | 60.06 s | 11.20 s | 5.363x | 4.621x |
| work | fib n=250k | 14.02 s | 3.72 s | 3.766x | 2.933x |
| | fib n=500k | 20.77 s | 5.54 s | 3.747x | 3.106x |
| | fib n=1M | 35.45 s | 8.41 s | 4.216x | 3.382x |
| | fib n=2M | 60.27 s | 11.14 s | 5.411x | 4.535x |
| guest | mem n=400k | 31.67 s | 8.19 s | 3.868x | 3.231x |
| | bits n=1M | 49.94 s | 9.75 s | 5.122x | 4.175x |

Still flat in shard size across a sixteen-fold range, still climbing with total work, and the envelope moves from 2.93 - 4.64x to **3.75 - 5.45x**. The floor rose most again, 2.93x to 3.77x, which is the same thing the floor measurement says: what looked like an irreducible per-shard cost was work nobody had looked at.

---

# The third list, from 5.38x

Written 2026-08-23 against **10.95 s paired with 60.03 s**.

## The shape of the problem has changed

| | |
|---|---:|
| CPU used | 34.3 user + 19.1 system = **4.9 of 16 cores** |
| device busy, union of intervals | **5.96 s, 54.4% of the proof** |
| time spent waiting on the device | **8.4 thread-seconds** |
| total accounted span time | 28.8 (43.8 at 3.74x) |

**Neither resource is saturated.** The proof is dependency-bound: four shards, each a chain of sumchecks whose rounds cannot overlap because a round's challenge comes from the previous round's message. That makes latency the target rather than throughput, which is the opposite of where this work started.

It also sets the ceiling. The device union is 5.96 seconds, so **even with no CPU cost at all and perfect packing, the proof cannot go below about six seconds — roughly 10x** — until the kernels themselves get cheaper.

## 1. `prove evaluation claims` and `Dense PCS evaluation proof` — 5.3 thread-seconds, never counted

The largest block nobody has instrumented: 2.4 and 2.9 thread-seconds.

Every previous time a block on one of these lists was counted rather than guessed at, the premise turned out to be wrong — four for four, one run each. **The first action is counters, not a port.**

**Estimated: unknown, and the cheapest thing on this list to find out.**

## 2. Stop waiting for what only the GPU reads — 8.4 thread-seconds

Two seconds a shard spent blocked, on a machine that is half idle.

A `MTLCommandQueue` runs the buffers committed to it in commit order, so a fold whose output is read only by the *next* dispatch needs no host wait at all: the ordering is already guaranteed and the wait buys nothing.

This is the sound form of the lever that was twice proposed and twice wrong. Not a round and its own fold — the challenger stands between those — but **a fold and the next round's reduction**, which are both determined once alpha is known.

**Estimated: 0.5-1.0 s.** *Stop if* blocking does not fall below about 5 thread-seconds.

The risk is worth naming: any host read of memory the GPU has not finished writing is silent corruption rather than a crash. It needs an explicit drain at every CPU fallback and every staged copy, and the parity modes are what would catch a missed one.

## 3. `Global`'s zerocheck as a table-driven kernel — 15% of zerocheck's cells

Zerocheck's round reduction is 4.38 thread-seconds, still the largest single block, and `Global` carries 4.04 of its 27.6 G cells on the CPU because transpiling it costs 1,814 seconds of compilation.

The GKR first layer solved exactly this shape: the constraints go up as **a table one kernel walks** rather than as a DAG unrolled into source, so every chip shares one pipeline and nothing is compiled at run time. This is the largest port left.

**Estimated: 0.3-0.6 s.**

## 4. Tuning where the device union actually goes — 5.96 s

`commit` is 3.07 GPU-seconds, `zerocheck` 2.85, `gkr` 2.04. The encode has had its pass count halved already; the Merkle tree is Poseidon2-bound; **the zerocheck kernel is generated per chip and has never been looked at**. This also drains the queue that item 2 is waiting in.

**Estimated: 0.3-0.6 s**, and it is the only item here that moves the ~10x ceiling.

## 5. The two worst dispatch ratios

`layer transition` blocks 4.08 ms per call for 0.56 ms of work; `first layer` blocks **21 ms** per call. Both are loops the driver walks one chip at a time, which is the shape the small-chip batch already solves for the GKR round.

**Estimated: 0.2-0.4 s**, much of it subsumed by item 2.

## 6. The floor — 1.86 s, now 17% of the proof

`setup` is a thread-second of it, including `build compress program` — recursion setup running during a *core* proof. Not reachable without vendoring `sp1-prover`. It caps the absolute ratio at about 32x.

## Not on the list, and why

- **Tracegen and execution**, 1.9 thread-seconds. Not GPU-shaped.
- **Allocation churn.** Answered by measurement: mimalloc was 45% worse. System time is 19.1 seconds but the CPU is not the constraint.
- **More shard concurrency.** The `ProverSemaphore::new(4)` cap is not binding with four shards; the constraint is the chain inside a shard.
- **Raising the layer gate or the zerocheck node bound.** Both measured worse, one catastrophically.

## The arithmetic

Items 1 to 5 are worth perhaps 1.5-2.5 s between them, landing at **6.4x-7.0x**. Past that the device union is the wall: nothing gets below about six seconds until the kernels get cheaper, which makes item 4 the one that decides whether there is a fourth list.
