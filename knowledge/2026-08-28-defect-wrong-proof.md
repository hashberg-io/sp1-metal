# An intermittent wrong proof, and the run that never returns

Found 2026-08-28 by a consumer running a SNARK lifter, proving leaf after leaf through this backend at `v0.1.0` against SP1 6.4.0, and reproduced here the same day.

The first half of this note is the consumer's evidence and the reasoning done from it, kept as it was written so the guesses can be read against the answer.
[Reproduced, and localised to the leaves](#reproduced-and-localised-to-the-leaves) is where it turns into a measurement.

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

This is the third time this project has met that disguise. `2026-08-26-defect-plan.md` records the previous two — the secp256k1 "performance cliff" that was a wrong proof, and the transpiler defect that presented as a hang — and says so in as many words: a timeout here is SP1 retrying a proof that will not verify.

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

- **Not the GKR upload path.** `SP1_METAL_UPLOAD_FRESH=1` gives every upload a private buffer, sharing nothing and outliving nothing. It survived, at the fortieth proof. ⚠️ **That exclusion was read wider than it goes.** The switch is read in `logup_gkr/metal.rs` and nowhere else, so what it makes private are the GKR round's uploads. The route the *codeword* takes to the Merkle kernels — `slop-dft`'s encode leaving its buffer in `sp1_metal_core::handoff`, and `sp1_metal_probes::commit::upload` taking it — never consults it. See [What the check mode was built to look at](#what-the-check-mode-was-built-to-look-at) below: there was a defect on exactly that route.
- **Not an ordinary wrong kernel.** A generated kernel that reads the wrong column or applies the wrong constant is wrong on every row and every run; this is wrong on about one run in twenty-eight. The per-chip prefix comparison never declined a chip, and a full run with `ZEROCHECK`, `GKR_ROUND`, `ENCODE`, `FIRST_LAYER`, `BATCH`, `JAGGED_VALUES` and `JAGGED_ASSIST` all at `=check` passed with no disagreement.
- **Reproduces only in a process that proves many times.** Every failure came from a driver making dozens of sequential `prove` calls; five single-proof runs in the same session were clean. That points at state carried across proofs rather than at a kernel.

## What is suspected

> Written before the reproduction, and right. The tree was where it was, though not for the reason that made it the best guess.

**`MERKLE`** — the trace commitment's Merkle tree.

The error is a Merkle root mismatch: a query opened a path that did not hash to the committed root. Of the phases that could produce that, the encode and both jagged phases have comparison modes and passed them, and **the Merkle tree is the only accelerated phase with no comparison mode at all** — so it is both the best fit for the message and the one place the checks already run could not have looked.

That is an inference from exclusion, not a localisation. Nothing here identifies a fault inside the phase.

## The evidence for it, and the workaround

`SP1_METAL_MERKLE=0` leaves the tree on the CPU and accelerates everything else.

**115 consecutive proofs with no failure** — a run of 51 stopped by its own wall-clock cap, and a complete run of 64 — followed by a **full-scale proof of a far larger body of shards**, which also completed clean: 219 responses verified in one zkVM program, 2 h 51 m 36 s, a Groth16 wrap, and an on-chain verification at 270,441 gas.

Against a rate of one in 28, those 115 proofs predict 4.1 failures and the chance of seeing none is **1.6%**. Three events is a thin basis for a rate and the interval around it is wide, so this is evidence at about the two-percent level rather than a demonstration.

**It costs most of the speedup.** On that consumer's workload a proof goes from 23.26 s to 57.00 s, so the backend drops from **4.72× over the same machine's CPU prover to 1.92×**. The Merkle tree is where most of the advantage was.

| | CPU | Merkle off | all on |
|---|---:|---:|---:|
| one-response proof | 109.72 s | 57.00 s | 23.26 s |
| speedup | 1.00× | 1.92× | 4.72× |
| survives a long run | yes | 115 proofs, no failure | **no** |

## Reproduced, and localised to the leaves

> **2026-08-28, from inside this repository.** The check mode below was built first; this is what it caught on its first long run.

```
sp1-profile-host --elf fib --stdin-u32 250000 --mode compressed --repeat 100 --timeout 900
SP1_METAL_MERKLE=check, hand-off table as it shipped in v0.1.0
```

**93 proofs verified. The 94th disagreed:**

```
metal merkle disagrees with the CPU over 4194304 rows of total width 8:
  layer 0 digest 0 of 4194304:
    GPU [584766356, 1225150442, 586329964, 219219076, 496528698, 703454750, 759079440, 1209945701]
    CPU [2050579817, 1806036363, 1739980070, 82759147, 1306557486, 603359774, 1237711924, 136306970]
```

Three things about that line, in order of how much they narrow it.

**Layer zero is the leaves.** Every layer above it is Poseidon2 over the layer below, so a disagreement that starts at layer zero is not about the tree: the compression is fine and the bytes that went into it were not. Had the kernel been wrong the disagreement would start there too — but the same kernel had just built ninety-three trees over the same shapes and agreed on every digest of every one of them. A kernel that computes the wrong hash does not do it once in ninety-four.

**Digest zero.** The first row of the matrix, not a row somewhere in the middle where a boundary or a tail might be mishandled.

**Total width eight, which means one tensor.** The leaf kernel reads a gathered buffer when a commitment has several tensors and the uploaded tensor itself when it has one, so at width eight it read exactly what `upload` handed it. That function has three routes: take the buffer the encode left in the hand-off table, wrap the words where they lie, or copy them. The last two are built fresh on every call and cannot be intermittent across proofs. The first is the only state in the path that outlives a call.

**It parked, exactly as reported.** The check's panic reaches SP1 as a task failure, SP1 calls it retryable, and the process then sat for fifteen minutes with nothing printed and nothing exiting, until the watchdog said so and left:

```
PROOF 94 STUCK: no result after 900s. A shard that fails verification is
retried silently and then parks; re-run with RUST_LOG=error to see the cause.
```

That is the same disguise the consumer described and the same one two earlier defects here wore.

What this does and does not establish: the GPU tree is demonstrably built over the wrong bytes, intermittently, about once in ninety-four compressed proofs of one process on this workload. The consumer's failure is that same tree one step later — published rather than compared, so the openings do not hash to it. The reproduction is of the cause; the root mismatch is its consequence.

Worth noting where in the run it fell: the nine proofs before it were 21 to 28 seconds against a steady 15.9, because another job had come onto the machine. More contention means more tokio worker migration, which is one of the two ways a hand-off is orphaned. One event is not a correlation, but it is the direction the mechanism below predicts.

## What the check mode was built to look at

`SP1_METAL_MERKLE=check` exists because this was the only accelerated phase without a way to check itself — it builds both trees and compares every digest of every layer, reporting the lowest layer that disagrees rather than only the root. Building it meant reading the codeword's whole route to the tree, and the route has a hazard on it.

### A pointer-keyed table whose stated invariant was false

The encode writes a codeword and the Merkle kernels hash it a moment later, with nothing on the host touching it in between. It is the largest allocation in the proof, so rather than wire it into a Metal buffer twice, the encode leaves its buffer in a thread-local table keyed on `(address, length)` and the commit takes it. `sp1_metal_core::handoff` is that table.

**Two live allocations cannot share an address.** So an entry that matches an allocation it was not made for can only mean the original was freed and its address handed out again — and the buffer wired over that address is then not the caller's memory in any sense the language will defend. The device would read whatever the pages held when the buffer was made, which is exactly a tree built over the wrong bytes and exactly a root mismatch.

The module asserted this could not happen, on the grounds that an entry left by a producer whose consumer never ran "is evicted by the next producer". It is not. The table holds sixteen entries and evicts the oldest only when full, so an orphan sits there for up to sixteen hand-offs; a lookup returned the *first* match rather than the newest, which is the stale one whenever there are two; and nothing dropped an orphan when the commitment it belonged to had gone.

### Orphans happen, and this is how many

The counters are now part of what a run reports: hand-offs kept, taken, dropped unclaimed, and left on an address a dead entry still held. Over twelve compressed `fib` proofs in one process, with the table left uncleared so that orphans accumulate the way they did:

| | kept | taken | never taken | on a dead entry's address |
|---|---:|---:|---:|---:|
| codewords, encode to Merkle | 265 | 261 | 4 | 0 |
| arrays, jagged reduce to fold | 2,556 | 852 | 0 | 0 |

So the codeword hand-off is orphaned about one time in seventy, roughly once every three proofs — a rate that held over four hundred proofs later on, at 108 in 8,413. No orphan had its address handed out again at the same length inside those twelve proofs, which is the same statement as "the fault did not fire"; it took ninety-four proofs to fire above. The jagged cache, which has the same shape, never leaked at all, so its fix is a precaution rather than a repair.

**Why `SP1_METAL_MERKLE=0` is a complete workaround for it.** With the tree on the CPU nothing ever *takes*: the encode still leaves buffers, `digest_layers` is never called, and no stale entry can be matched. That is consistent with 115 consecutive proofs and a full-scale run passing on the consumer's machine, and it is consistent with the failure surviving `SP1_METAL_UPLOAD_FRESH=1`, which does not reach this route.

### The audit that found it

There are three caches in this backend keyed on `(address, length)`, and the difference between them is the whole lesson.

| cache | what it holds | keeps the allocation alive? |
|---|---|---|
| `logup_gkr::CACHE` | an MLE already uploaded this round | **yes** — the entry carries an `Arc` to it |
| `sp1_metal_core::handoff` | a codeword's buffer, encode to Merkle | no |
| `slop_jagged::ROUND_BUFFERS` | a round's arrays, reduce to fold | no |

The first cannot go wrong: while the entry lives the allocation lives, so nothing else can be handed its address. The other two argued instead that the window is short enough — and an argument about a window is only as good as its narrowest claim, which in both cases was about a code path that turns out not always to be taken. A pointer-keyed cache should hold what it is keyed on.

### What was changed

- A producer replaces any entry with its own key rather than queueing behind it, and a lookup takes the newest match. Both are counted, so an address handed out twice is now a number rather than a coincidence.
- A commitment ends the hand-off window: whatever is still in the table when it returns is dropped, because past that point nothing can want one and every key is a pointer about to be freed.
- The jagged round cache, which is the same design one crate over, gets the same treatment: its reduction drops what the previous round left, since the fold that was supposed to clear it is skipped for a round with too few pairs.
- `sp1-profile-host` takes `--repeat N`, which proves through one client the way a consumer's driver does, and `--timeout SECS`, which is a watchdog thread that says the run is parked and leaves with status 2. The parking is most of what this defect costs, and nothing in SP1 says it has happened.

### The same soak with the fix in

Same binary but for the hand-off, same workload, same checking switch, four times as long:

```
400 of 400 proofs verified, 0 failed, in 3 h 03 m
handoff codewords:    8,413 kept, 8,305 taken,   108 dropped unclaimed, 0 on a dead entry's address
handoff round arrays: 85,200 kept, 28,400 taken,   0 dropped unclaimed, 0 on a dead entry's address
```

**One hundred and eight orphans.** Each is an entry that under the previous arrangement would have sat on a pointer to freed memory for up to sixteen hand-offs, waiting for the allocator to hand that address out again; each was dropped at the end of the commitment it belonged to instead. That is the fault's supply, counted, and it does not stop being produced — it stops being reachable.

The unfixed build failed at the ninety-fourth proof of this soak. Four hundred clean proofs at that rate is a **one-in-seventy** coincidence, so this is evidence rather than proof, and it is the kind of evidence a rare fault admits of. The jagged cache produced no orphans at all in 85,200 hand-offs, which settles that its fix was a precaution.

What is still not settled is the consumer's own workload, which has many more shards per proof than this one and measured the fault four times more often. A hand-off is orphaned per commitment, so the exposure scales with commitments rather than with proofs, and their run is the one that would show it soonest.

### What is left

- **Run it on the consumer's statement.** That is the workload the rate was measured on, and it is the only one that can retire the workaround for them.
- **Decide whether the hand-off is worth keeping at all.** It exists so that the largest allocation in a proof is not wired into a Metal buffer twice, and the module's own note says that effect is "small enough that it needs an A/B on one binary to be seen at all". If that holds, `SP1_METAL_REUSE` defaulting to off would remove this hazard class rather than bound it, which is a better answer than the one above. The A/B needs an idle machine and has not been taken.

## What a fix would need

The three below were written before any of the work above. The first and the third are done and the second turned something up; they are kept as written so what was expected can be read against what happened.

- ~~**Give the Merkle phase a `=check` mode.**~~ Done: `SP1_METAL_MERKLE=check`. It compares every digest of every layer and names the lowest one that disagrees, which separates a tree built wrongly from a tree built correctly over the wrong bytes.
- ~~**Look for state carried across `prove` calls**~~, since a single-proof process does not reproduce it. This is where the hand-off table was found. It is thread-local state keyed on a raw pointer, and pointers are precisely what a long-running process hands out again.
- ~~**Fail loudly.**~~ Done, on the harness side: `--timeout SECS` is a watchdog on its own OS thread, because the runtime is what has stopped turning. Nothing was added inside the backend, which never learns that a proof it contributed to was refused.

What is still open:

- **Reproduce it here.** 165 compressed proofs at the defaults did not, so the workload has to change rather than the count: more shards, more concurrent threads, or the consumer's own statement.
- **Run a long soak under `SP1_METAL_MERKLE=check`.** That is the instrument the first item built, and it is the only thing that can say whether the tree is ever wrong rather than whether a proof is ever refused.

## Provenance

Measured on an M4 Max, 16 cores / 137 GB, `RAYON_NUM_THREADS=12 GOMAXPROCS=12`, one run at a time on an otherwise idle machine, against `sp1-metal` v0.1.0 and SP1 6.4.0.
The consumer-side record, including the logs, is held by the consumer and is not part of this repository.

Separately, that full-scale run is the measurement [2026-08-21-verdict.md](2026-08-21-verdict.md) named as the one that would change it — wall clock on the canonical classical-128 lift, the workload D7's bar is written against and which had never been run on. It is **2.27×** end to end at the safe setting, against 6 h 28 m for the same pass on the CPU. It has since been written into the verdict, where it closes that item without moving the answer: the bar is 5x.
