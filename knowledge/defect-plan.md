# Closing the four open defects

> **Status: planned, not started.** Written 2026-08-26 as a handover, so it can be picked up with no memory of the session that found these.

Four defects remain in the Metal backend. None blocks a release — each is documented in [USING.md](../USING.md)'s known-issues section with a workaround — but each costs something real. This is how to close them, in the order that makes the later ones cheaper.

The state they are being closed from is in [benchmark-status.md](benchmark-status.md), and the evidence for each is named below.

## The four

| # | Defect | Cost today | Root cause known? |
|---|---|---|---|
| 1 | Chips of ~7k–13k constraint nodes take about ten minutes to compile | A Keccak or division program's first proof looks like a hang | Inside Apple's compiler; ours is the admission control |
| 2 | The node bound cannot be raised | Over-bound chips can never be accelerated | Same as 1 |
| 3 | secp256k1 falls off a cliff past n≈1000 | >115x *slower* than the CPU arm | No |
| 4 | The transpiler mis-transpiles eight recursion chip variants | They run on the CPU during recursion | No |

Take **4 first**: it is mechanical now in a way it was not before, it buys back real performance, and it may well explain 3.

---

## 4. Find the transpiler bug

**What is known.** `BaseAlu`, `ExtAlu`, `PrefixSumChecks` and `PublicValues`, in both rounds each, generate kernels whose round polynomial disagrees with the CPU. They are detected and declined on first use, so this is now a performance cost. `results/zerocheck-invalid-proofs.md` has the bisection that found them.

**What was already ruled out**, so do not spend time here again: it is not the body interleaving (`SP1_METAL_ZC_ILP=0` fails identically); not a main-versus-preprocessed row-count mismatch (all four counts are 592,384 for `BaseAlu`); not a next-row read (`reads_next_row` is clear, and is now checked anyway); not row selectors or public values (the generated kernel contains none); and not an asymmetry between interpolation points (the point-2 and point-4 bodies are structurally identical). Reading the generated MSL and looking for something wrong did not converge.

**Why it is tractable now.** `SP1_METAL_ZEROCHECK=check` makes a disagreement fatal and names the chip, and `SP1_METAL_ZC_DUMP=<chip key>` prints the generated kernel. That turns this from inspection into bisection.

**The plan.**

1. **Bisect the constraint list.** `BaseAlu` has five constraints; the kernel accumulates `sum_k ap[k] * constraint_k`. Add a switch that emits only constraint `k` (and have the CPU comparison do the same), then walk `k` until one disagrees. This narrows a whole chip to one constraint in five runs.
2. **Bisect that constraint's DAG.** With one constraint failing, emit its sub-expressions as separate outputs and compare each. The `Codegen` already memoises by `Rc` pointer, so a sub-expression is easy to name.
3. **Read the one node that differs.** By this point the suspect is a single `SymbolicExpression` variant in a specific context — degree handling in `Mul`, a `Ty::Base`/`Ty::Ext` promotion, or the `ap[k]` batching index.
4. **Fix, then confirm the eight variants stop declining themselves.** The self-check is the acceptance test: run compressed and look for `disagreed with the CPU` in the chip table. Zero occurrences is the target.

**Do not remove the self-check when this is fixed.** It is what makes the next generator bug a performance cost instead of an unverifiable proof.

**Verify:** compressed proves and verifies with no declines; `results/compressed-mode.tsv` re-measured (expect better than 2.85x–3.72x, since eight chips move back to the GPU); core unchanged at ~5.6x.

---

## 3. Diagnose the secp256k1 cliff

**What is known.** Between n=1000 and n=1500 the CPU arm is flat (7.73 → 7.79 s) and the shard count does not change, while the Metal arm goes from 2.15 s to not finishing in 900 s. `Secp256k1DoubleAssign` is 167,376 and 251,064 nodes — far over the bound, so it never gets a kernel and its zerocheck falls back to the CPU. It is also the widest chip in the machine. `results/precompile-scaling.tsv`.

**The plan.**

1. **One run at n=1500 with the per-phase report** (the host prints it at exit; give it a long timeout so it reaches the print, or lower n until it does). The phase table names which of `gkr`, `zerocheck`, `commit`, `pcs` explodes. Because the CPU arm is flat, whatever explodes is in the Metal path specifically.
2. **If it is a phase with an `=check` mode, use it.** If it is zerocheck, note that this chip has no kernel at all, so the cost is the *fallback*, not a wrong kernel.
3. **Suspect the widest-chip paths**: an upload of a very wide trace, or a per-chip dispatch that scales with width. `sp1_metal_core::stats` reports bytes by phase and residency, which will show a copy that should not be happening.
4. Fix or, failing that, decline the chip by width and document the threshold.

**Verify:** a size ladder from n=500 to n=4000 with no discontinuity, added to `results/precompile-scaling.tsv`.

---

## 1 and 2. Replace the node bound with a compile budget

These are one fix. The bound exists to keep chips that are too expensive to compile off the GPU, and it does not work: node count does not predict compile cost. `KeccakPermuteControl` at 12,351 nodes takes ten minutes; `Global:true` at **15,196** compiles instantly. `results/kernel-compile-pathology.tsv`.

**The plan.**

1. **Compile on a worker thread with a deadline.** `new_library_with_source` is synchronous and cannot be cancelled, so hand it to a thread, wait with a timeout (a few seconds), and on expiry decline the chip and let the thread finish and drop its result. The abandoned thread costs nothing but its own completion.
2. **Cache the verdict on disk**, keyed by the same source hash macOS keys its shader cache on, so the deadline is paid once per source per machine rather than once per process.
3. **Then raise the default bound**, since a budget makes it safe: a chip that compiles fast is accepted whatever its node count, and one that does not is declined whatever its node count. That closes defect 2 by construction.
4. Keep `SP1_METAL_ZEROCHECK_MAX_NODES` as an override.

**Watch for:** the compile lock in `sp1_metal_core::compile` serialises pipeline creation process-wide, so a deadline must not be held while another thread waits on that lock — take the deadline around the whole guarded region, not inside it.

**Verify:** first proof of `guests/keccak` completes in seconds rather than ten minutes, with the offending chips declined; `results/kernel-compile-pathology.tsv` re-measured.

---

## One thing to do first, whichever you start with

**Re-run the standard tier** (`tools/bench_suite.py --tier standard`, about two hours) against the fixed backend. Every number in `results/bench-standard.jsonl` predates the zerocheck fix, and the tier now includes a compressed point. Without that baseline there is nothing to compare a fix against.
