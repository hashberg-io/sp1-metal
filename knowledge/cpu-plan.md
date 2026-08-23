# What the CPU still does, and what is worth moving

A profile of the Metal prover's residual CPU work, and a plan built on it.
Everything here is measured on the acceptance workload — fibonacci n = 2,000,000, about 18M cycles, `SHARD_SIZE` 4 MiB, four shards — against the same binary with every Metal switch off.

## How it was measured

Three instruments, because no one of them answers the question.

**Total CPU, from the operating system.** `/usr/bin/time` on both arms: the Metal path spends **121 user-seconds and 63 system-seconds** against the CPU path's **884 and 85**. So the backend has removed 7.3x of the CPU work while removing only 3.4x of the wall clock, which is the fact the rest of this document is about.

**Where that CPU time goes, from a sampling profiler.** `sample` needs no sudo on one's own processes, and `tools/sample_profile.py` turns its call graph into a flat profile. Two things make it usable. It attributes each sample's *self* time to the deepest frame naming a recognisable operation, because the largest leaf in this prover is `BinomialExtensionField::mul` at a fifth of all CPU time, which tells you nothing — every phase multiplies. And it reads the sampling interval from the report's own header, having once reported a 10 ms profile as a tenth of the work it was.

**Utilisation over time, by differencing the kernel's own counter.** `ps -o cputime` differenced across half-second samples gives instantaneous cores used. `ps -o %cpu` does not: it is a decaying average, and it smears exactly the structure worth seeing.

Three traps met on the way, all of which produced a confident wrong answer first:

- `sample` follows the PID it is given, and `/usr/bin/time cmd &` gives the PID of `time`. A profile of the wrapper looks like a profile with no work in it.
- Two `nice`d proving processes are not a throughput experiment. macOS maps `nice` onto a background QoS class, and the two runs came back at 18.7 s and 125.2 s having done byte-for-byte identical work. Whatever that measures, it is not the machine's capacity.
- Rayon loses the caller. Work stolen by another thread has no project frame on its stack, and 17 of the 26 thread-seconds spent folding MLEs cannot be attributed to a phase this way. Where it mattered, the call sites were read from the source instead.

## Where the CPU time goes

Thread-seconds over a 17 s window, 5 ms sampling, summed across phases because rayon makes the phase column unreliable. The window's 139 thread-seconds of work reconcile with the 184 CPU-seconds the operating system reports for the whole run.

| operation | thread-s | share |
|---|---:|---:|
| folding an MLE's last variable | 24.3 | 17.4% |
| the Reed-Solomon encode (DFT, bit reversal, transpose) | 22.0 | 15.8% |
| **GKR interaction values** (`generate_interaction_vals`) | 18.2 | 13.1% |
| trace generation | 13.8 | 9.9% |
| **the jagged evaluation sumcheck** | 12.8 | 9.2% |
| GKR round evaluations on the CPU | 9.6 | 6.9% |
| rayon scheduling and the allocator | 8.9 | 6.4% |
| blocked inside a dispatch | 6.4 | 4.6% |
| executing the guest | 6.1 | 4.4% |
| `partial_lagrange` | 4.5 | 3.3% |
| dot products (`dot_along_dim`) | 4.4 | 3.1% |
| GKR circuit generation, the rest | 3.9 | 2.8% |
| Poseidon2 left on the CPU | 2.2 | 1.5% |
| zerocheck on the CPU | 0.4 | 0.3% |

## The shape of the proof in time

Instantaneous cores used, of sixteen:

| | duration | cores | what it is |
|---|---:|---:|---|
| startup | ~4.3 s | 1–3 | guest execution and the first traces |
| the middle | ~9 s | **12–15.6** | four shards proving at once, the machine saturated |
| the tail | ~5 s | 3–11 | the last shards finishing |

The device, meanwhile, is busy 23% of the proof, and unevenly: a histogram of the command buffers' own execution intervals shows it saturated for about a second in the middle and under 20% for most of the rest. There is a great deal of GPU left.

Two consequences, and they pull in opposite directions.

**In the middle the machine is the constraint**, so removing CPU work there does shorten the proof. **In the tail it is not**, so only work on the last shard's critical path counts. That path, for the slowest shard, is `commit traces` 3.9 s, `generate GKR circuit` 2.4 s, `prove GKR circuit` 4.2 s, `zerocheck` 3.2 s, `prove evaluation claims` 4.2 s — sequential within the shard, and summing to the 18.3 s the shard took.

A caution on how much any of this is worth: dropping the per-chip dispatch threshold from 16384 entries to 64 moved **7.2 thread-seconds** of GKR work to the GPU and changed the wall clock by nothing measurable. That is 5% of the CPU work, worth at most half a second in the saturated middle, which is inside this workload's run-to-run spread of about a second. It bounds the value of small savings; it does not show that CPU work is free.

## What is worth moving, in order

**1. The GKR first layer — `generate_interaction_vals`, 18.2 thread-seconds.**
The largest block of residual CPU work that is not already ruled out, 13% of all of it, and 2.4 s of the critical path in a span that still measures only 1.9x.
It is also the best-shaped port on the board. Per row and per interaction it evaluates `VirtualPairCol::apply` — a sparse linear form over the row's columns, statically known per chip — and accumulates `alpha + sum(beta_j * value_j)`. That is the same shape as the zerocheck's constraint reduction, which this project already transpiles per chip into Metal from the symbolic DAG. Its inputs are the main and preprocessed traces, which are page-aligned host arrays the commitment already reads, so nothing is copied; its output is the first GKR layer, which kernels we already have then consume.
*Stop if*: `generate GKR circuit` does not fall below about 1 s per shard.

**2. The jagged evaluation sumcheck, 12.8 thread-seconds.**
9% of the CPU work, 4.2 s of the critical path, and it runs at roughly 1.5 of sixteen cores — the least parallel phase in the prover.
Upstream has already built the seam: `JaggedAssistSumAsPoly<F, EF, A, ..>` is a trait over the backend with a `JaggedAssistSumAsPolyCPUImpl` beside it, in a crate this project has already vendored. A Metal implementation needs no type surgery.
*Stop if*: `prove evaluation claims` does not fall below about 2.5 s per shard.

**3. GKR's small chips, batched — 9.6 thread-seconds of round evaluations and about 4 of folding.**
The threshold sweep above is the evidence for *how* rather than *whether*: dispatching these chips one at a time is a wash, because each is a few hundred elements and the dispatch costs what the arithmetic saves. The move is one kernel over all of a layer's small chips at once, with an offset table, rather than a lower threshold.
*Stop if*: `prove GKR circuit` does not fall below about 3 s per shard.

**4. The 6.4 thread-seconds spent blocked inside dispatches**, over 3163 command buffers. A layer's round and fold could share a command buffer without the host waiting between them.

## What is not worth moving

**The Reed-Solomon encode, 22.0 thread-seconds** — the second-largest block, and a trap. It was previously dismissed on a probe measuring 1.02x against a *packed* CPU baseline, which raised the question of whether the shipped encode is packed. It is: `p3-dft`'s butterflies use `F::Packing::pack_slice_with_suffix_mut`, so the baseline is what actually runs and the 1.02x stands. Leave it.

**Trace generation and guest execution, 19.9 thread-seconds together.** Not field arithmetic, and not GPU-shaped.

**Rayon and allocator overhead, 8.9 thread-seconds.** Real, but the fix is fewer and larger parallel tasks, not a kernel.

## What this can be worth

The portable work is items 1 to 3: about 51 of the 139 thread-seconds, 37%. If all of it moved and the GPU absorbed it — it is at 23%, so it has the room — the saturated middle shrinks from about 9 s to about 5.7 s and the proof goes from 18.5 s to something near 15 s. Against the 62.9 s CPU arm that is **about 4.1x**, from 3.4x.

Startup and tail together are 9.3 s of the 18.5, and neither is CPU-throughput-bound. They are the floor: no amount of porting takes this workload below roughly 9 s, which is 7x, and only the tail is even addressable — by overlapping shards better, which is scheduling rather than a kernel.

## Open

A dead zone appeared once at about t = 8 s, a second in which neither the CPU nor the GPU did anything at all. It did not reproduce, `vm_stat` showed no compressor or swap activity across the run, and run-time kernel compilation is only 44 ms once macOS's shader cache is warm. Worth watching for, not worth chasing yet.
