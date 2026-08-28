# What a slow kernel costs, and what warming it buys

> Measured 2026-08-27 on an M4 Max, machine idle, `SP1_METAL_COMPILE_LOG=1`.

`results/2026-08-24-m4-max-kernel-compile-pathology.tsv` recorded that some chips take about ten minutes to compile and that node count does not predict which. This is what was done about it and what each option is worth.

## The compile times, measured per chip

`SP1_METAL_COMPILE_LOG=1` prints every kernel's compile time and how much of it was spent queueing behind another compile, which is the only way to tell a chip that compiles slowly from one that waited for a chip that does.

| chip | nodes | compiling |
|---|---:|---:|
| `KeccakPermuteControl:true` | 8,234 | 11.41 s |
| `KeccakPermuteControl:false` | 12,351 | **565.94 s** |

Fifty times the compile for one and a half times the nodes. Every other chip in a keccak proof compiles in 0.00–0.01 s.

## Optimising for size does not help — measured, negative

The shape of the numbers suggested the Metal optimiser going superlinear on a large straight-line function, in which case asking for less optimisation would be a fix rather than a workaround. `SP1_METAL_COMPILE_OPT=size` sets `MTLCompileOptions.optimizationLevel` to `Size`.

`KeccakPermuteControl:false` compiles in **552.22 s** that way, against 565.94 s at the default. It is not the optimiser's aggressiveness.

The switch is kept because the question looks obvious and the answer is not.

## Warming does help, and lasts

macOS caches a compiled shader against its source **and its compile options**, and the cache survives the process. So the cost is payable once:

```sh
SP1_METAL_COMPILE_BUDGET=1800 <your prover, once, on a representative input>
```

That run blocks until every kernel it asks for is built — 571.48 s for keccak — and every run after it gets those kernels at the shipped one-second budget:

| | keccak, n=2000 | divrem, n=1,199,680 |
|---|---:|---:|
| shipped budget, shader cache cold | 6.30 s | 18.70 s, 19.25 s |
| shipped budget, shader cache warm | **5.55 s**, 5.59 s | **14.27 s** |
| the warming run itself | 571.48 s | already warm when asked |

Both then match their best figures from before any of this work — divrem was 14.13 s on 2026-08-24, when that cache had been warm for days.

Whether it is worth 571 seconds of somebody's time is theirs to decide, which is why it is a switch and not a default.

## And it is a caution about two rows of the standard tier

Those two guests are measured against whatever the machine's shader cache happened to hold. Re-measured warm, same binary, same machine, two repeats an arm:

| guest | size | cpu_med | gpu_med | warm | tier (cold) |
|---|---|---:|---:|---:|---:|
| divrem | small | 16.39 | 5.12 | **3.20x** | 2.47x |
| divrem | standard | 72.78 | 14.27 | **5.10x** | 3.75x |
| divrem | large | 142.35 | 28.23 | **5.04x** | 5.01x |
| keccak | small | 9.05 | 2.87 | **3.15x** | 2.41x |
| keccak | standard | 13.77 | 5.64 | **2.44x** | 2.07x |
| keccak | large | 19.91 | 9.29 | **2.14x** | 1.94x |

So divrem/standard is 5.10x rather than 3.75x, and the tier's worst point, keccak/large, is 2.14x rather than 1.94x. **No other guest has a chip slow enough for this to matter** — every other kernel in the machine compiles in hundredths of a second, warm or cold.

## Why the default does not wait

A round waits one second for its chip's kernel and takes the CPU if it is not ready; the kernel is installed whenever it arrives, for every round after that. Two earlier designs were worse:

- **Declining a chip that missed a deadline, and remembering it.** The first compile of a source on a machine is cold, so this condemned `ShaCompress` and `ShaExtend` — chips that had never been a problem — and cost sha256 more than half its advantage, permanently. A cold measurement is not a property of the chip.
- **Waiting longer.** Sixty seconds simply added sixty seconds to the first proof of any program with a slow chip, for a kernel that was going to arrive anyway.

And only one compile runs at a time, with nobody waiting on the one in progress: queueing is what made the first slow chip cost every chip behind it a full budget each, which turned a keccak proof into two minutes.
