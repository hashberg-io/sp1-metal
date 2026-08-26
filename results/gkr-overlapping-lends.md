# The secp256k1 cliff was a wrong proof, not a slow one

> **Fixed 2026-08-26.** One line: the GKR round's per-chip bindings are copied rather than lent.

`results/precompile-scaling.tsv` recorded secp256k1 falling off a cliff between n=1000 and n=1500 — the CPU arm flat at 7.7 s, the Metal arm going from 2.15 s to not finishing in 900 — and called it a performance defect, "more than 115x *slower* than the CPU arm". It is not slow. It never finishes because it never produces a valid proof:

```
shard verification failed: invalid shard proof:
  GKR verification failed: inconsistent evaluation at round 16
... exhausted its retry budget
```

That is the same disguise the zerocheck defect wore: SP1 retries a failed shard silently and leaves every thread parked, so a correctness defect reads as a hang from outside. Only `RUST_LOG=error` says otherwise. **Twice now.**

## It is not a cliff

| n | 1000 | 1200 | 1400 | 1500 |
|---|---|---|---|---|
| result | verified | **never finishes** | verified | **never finishes** |

Not monotone in the size, so not a threshold. Nor in the GPU threshold: at n=1400 it fails with `SP1_METAL_MIN_ENTRIES` at 0 and at 8192, and verifies at 1024 and at 16384.

## Finding it

1. **Gate bisection.** With all ten switches off, enabling only the GKR round reduction reproduces it. Each of the ten verifies on its own.
2. **A check mode for that reduction** — the last phase without one, and the second phase without one to be found wrong. It reported **no disagreement**: the round's value is right.
3. **And the proof still failed while the CPU's value was the one being returned.** That is only possible if *running* the reduction damages something. The check compares the CPU round before the GPU round as well as after, so it also rules out the reduction damaging the layer it reads.
4. Not a race: it fails identically at `SP1_METAL_PERMITS=1`. Not the upload cache: it fails with `SP1_METAL_UPLOAD_CACHE=0`.
5. **`SP1_METAL_UPLOAD_FRESH=1`** — every upload a private copy — verifies. So the damage is in how data reaches the device.
6. **`SP1_METAL_UPLOAD_LEND=0`** — never lend host pages, but keep the staging arena — verifies. So it is the lending.

## What was wrong

`bind_small` lent sub-slices of arrays a layer binds **once per chip**. `eq_row[..2 * n_pairs]` starts at the same address for every chip, so a layer with several wide chips created several `newBufferWithBytesNoCopy` buffers over the same pages, live at the same time. Metal does not allow that.

It is shape-dependent because it needs more than one chip in a layer over the 4 KB inline limit and past the dispatch threshold — which is why the widest precompiles in the machine found it and nothing else did.

Those slices are small by construction: anything under four kilobytes is already bound inline. Copying them costs nothing measurable.

## After

Every size verifies and the discontinuity is gone.

| guest | n | cpu_s | gpu_s | speedup |
|---|---|---|---|---|
| secp256k1 | 1200 | 7.93 | 2.28 | 3.48x |
| secp256k1 | 1400 | 8.00 | 2.28 | 3.50x |
| secp256k1 | 1500 | 8.09 | 2.33 | 3.47x |
| secp256k1 | 2000 | 8.29 | 2.32 | 3.57x |
| secp256k1 | 4000 | 8.66 | 2.41 | 3.59x |
| bn254 | 1000 | 7.90 | 2.32 | 3.41x |
| bn254 | 1500 | 8.53 | 2.41 | 3.55x |

## The general rule this establishes

`crates/sp1-metal-core/src/nocopy.rs` documented one requirement for lending — a page-aligned pointer — and said so emphatically, because believing in a second, invented one had cost 155 GB of memcpy a proof. There *is* a second requirement, and this is it: **no two live no-copy buffers may cover the same page.** A whole allocation lent once is always safe. A sub-slice is only safe if nothing else lends a slice sharing its pages at the same time.
