# An intermittent wrong proof, and the run that never returns

Found 2026-08-28 by a consumer — the ProofofSeed SNARK lifter — proving leaf after leaf through this backend at `v0.1.0` against SP1 6.4.0.
It is not reproduced from inside this repository, and the evidence below is that consumer's.

## What happens

A proof fails SP1's own shard verification with

```
ERROR task failed with retryable error: shard verification failed: invalid shard proof:
  invalid pcs opening proof: dense pcs verification failed: PCS error:
  Query failed to verify: root mismatch
```

and the run then **stops making progress and never returns**.
SP1 treats the failure as retryable, retries, and parks; nothing is printed after that line and no process exits.
Only `RUST_LOG=error` shows the cause at all.

This is the third time this project has met that disguise. `defect-plan.md` records the previous two — the secp256k1 "performance cliff" that was a wrong proof, and the transpiler defect that presented as a hang — and says so in as many words: a timeout here is SP1 retrying a proof that will not verify.

**Nothing invalid is accepted.** The failure is a proof being refused before anything downstream sees it, so what is at risk is a run finishing, not a statement being proved. For an unattended run that is the whole cost: it produces nothing and says nothing.

## Rate

Three occurrences over the 84 single-response proofs that three runs completed, each proof about 23 s — roughly **one in 28**, or one per eight minutes of proving.

| run | configuration | failed at | phase |
|---|---|---:|---|
| 1 | defaults | proof 8 | core shard verification |
| 2 | defaults | proof ~36 | `lift/deferred verify` (recursion) |
| 3 | `SP1_METAL_UPLOAD_FRESH=1` | proof 40 | `lift/deferred verify` (recursion) |

**It is not deterministic in the input**: the proof that failed in run 1 succeeded in runs 2 and 3, on identical input.
It appears in both core proving and the recursion layer.

## What has been excluded, and how

- **Not the upload path.** `SP1_METAL_UPLOAD_FRESH=1` gives every upload a private buffer, sharing nothing and outliving nothing. Its own documentation says a corruption surviving it is not in how the data reaches the device. It survived, at the fortieth proof.
- **Not an ordinary wrong kernel.** A generated kernel that reads the wrong column or applies the wrong constant is wrong on every row and every run; this is wrong on about one run in twenty-eight. The per-chip prefix comparison never declined a chip, and a full run with `ZEROCHECK`, `GKR_ROUND`, `ENCODE`, `FIRST_LAYER`, `BATCH`, `JAGGED_VALUES` and `JAGGED_ASSIST` all at `=check` passed with no disagreement.
- **Reproduces only in a process that proves many times.** Every failure came from a driver making dozens of sequential `prove` calls; five single-proof runs in the same session were clean. That points at state carried across proofs rather than at a kernel.

## What is suspected

**`MERKLE`** — the trace commitment's Merkle tree.

The error is a Merkle root mismatch: a query opened a path that did not hash to the committed root. Of the phases that could produce that, the encode and both jagged phases have comparison modes and passed them, and **the Merkle tree is the only accelerated phase with no comparison mode at all** — so it is both the best fit for the message and the one place the checks already run could not have looked.

That is an inference from exclusion, not a localisation. Nothing here identifies a fault inside the phase.

## The evidence for it, and the workaround

`SP1_METAL_MERKLE=0` leaves the tree on the CPU and accelerates everything else.

**115 consecutive proofs with no failure** — a run of 51 stopped by its own wall-clock cap, and a complete run of 64 — followed by a **full-scale proof of a far larger body of shards**, which also completed clean: 219 ZKBoo responses verified in one zkVM program, 2 h 51 m 36 s, a Groth16 wrap, and an on-chain verification at 270,441 gas.

Against a rate of one in 28, those 115 proofs predict 4.1 failures and the chance of seeing none is **1.6%**. Three events is a thin basis for a rate and the interval around it is wide, so this is evidence at about the two-percent level rather than a demonstration.

**It costs most of the speedup.** On that consumer's workload a proof goes from 23.26 s to 57.00 s, so the backend drops from **4.72× over the same machine's CPU prover to 1.92×**. The Merkle tree is where most of the advantage was.

| | CPU | Merkle off | all on |
|---|---:|---:|---:|
| one-response proof | 109.72 s | 57.00 s | 23.26 s |
| speedup | 1.00× | 1.92× | 4.72× |
| survives a long run | yes | 115 proofs, no failure | **no** |

## What a fix would need

- **Give the Merkle phase a `=check` mode.** Seven phases have one and this does not, which is why the fault survived a full seven-phase checked run. On the evidence here that is the single most valuable thing to add: it would both catch this and localise it.
- **Look for state carried across `prove` calls**, since a single-proof process does not reproduce it.
- **Fail loudly.** Even without a fix, this parking the run is most of the cost. A retry that never returns turns a five-minute loss into an all-night one, and nothing but `RUST_LOG=error` distinguishes it from slow progress.

## Provenance

Measured on an M4 Max, 16 cores / 137 GB, `RAYON_NUM_THREADS=12 GOMAXPROCS=12`, one run at a time on an otherwise idle machine, against `sp1-metal` v0.1.0 and SP1 6.4.0.
The consumer-side record, including the logs, is in `ProofofSeed/snark-lifter/knowledge/gpu-corruption.md` and `fixtures/gpu-failure-*.log`.

⚠️ Separately, that full-scale run is the measurement [verdict.md](verdict.md) names as the one that would change it — wall clock on the ProofofSeed classical-128 lift, the workload D7's bar is written against and which had never been run on. It is **2.27×** end to end at the safe setting, against 6 h 28 m for the same pass on the CPU. That belongs in the verdict rather than in a defect note, and is left for whoever picks this up.
