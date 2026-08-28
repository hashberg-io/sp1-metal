# Knowledge

Curated notes on what this project measured, built and concluded.
They are working documents kept as a record, not a tidied-up narrative, and several of them are wrong in interesting ways that later documents correct — which is the point of keeping them.

**Read [2026-08-21-verdict.md](2026-08-21-verdict.md) first.** It is the only document that states the current answer, and it says which of the project's earlier claims survived and which did not.

## Current

| document | what it is |
|---|---|
| [2026-08-21-verdict.md](2026-08-21-verdict.md) | The answer: what was asked, what was measured, what is still not met. |
| [2026-08-22-backend.md](2026-08-22-backend.md) | How the backend works and how it got there, in seventeen sections numbered 5 to 21 — it continues the plan's numbering, and the first four steps are in the spikes. |
| [2026-08-23-next-levers.md](2026-08-23-next-levers.md) | Five rounds of optimisation from 3.74x to 5.56x, every ranked list, and every negative result. |
| [2026-08-21-measurements.md](2026-08-21-measurements.md) | What the machine can do: the two silicon ratios, per-kernel throughput, the phase profile of a real proof. Measured 2026-08-21. |
| [2026-08-19-probes.md](2026-08-19-probes.md) | The probe suite — what each of the thirteen probes measures, how to run it, and its limits. |
| [2026-08-19-desiderata.md](2026-08-19-desiderata.md) | The requirements this work was to meet. D7 is the bar the verdict is judged against, and it is still not met. |
| [2026-08-28-defect-wrong-proof.md](2026-08-28-defect-wrong-proof.md) | An intermittent wrong proof that read as a hang: how it was reproduced, localised to the leaves of the tree, and traced to a buffer hand-off keyed on a pointer that outlived its allocation. Fixed before `v1.0.0`, with 400 clean proofs against a fault that fired at 94. |

## Executed plans

Written before the work, kept because the gap between a plan and its outcome is usually the most useful thing in it.
Each records its own result at the end; do not read the opening as a live statement.

| document | outcome |
|---|---|
| [2026-08-21-backend-plan.md](2026-08-21-backend-plan.md) | Executed. It opens by quoting a verdict that has since been overturned, and its projection of 10.7x proved optimistic by roughly a factor of two. |
| [2026-08-23-cpu-plan.md](2026-08-23-cpu-plan.md) | Executed. Three ports took the backend from 3.40x to 3.74x, which is close to what it predicted, for the reason it gave. |
| [2026-08-22-resident-plan.md](2026-08-22-resident-plan.md) | Stage 1 executed, 2.60x to 3.40x, and it was one false page-alignment rule. Stages 2 to 4 deliberately not built; the document says why. |
| [2026-08-24-release-plan.md](2026-08-24-release-plan.md) | Executed, and pushed: items 1 to 6, the `v0.1.0` tag, and the repository public. |

## Benchmarking

| document | what it is |
|---|---|
| [2026-08-26-defect-plan.md](2026-08-26-defect-plan.md) | The four defects and how each was closed — two were not what the plan thought they were, and three more were found while closing them. |
| [2026-08-24-benchmark-status.md](2026-08-24-benchmark-status.md) | Where the benchmark work stands, what it found, how to check on the running suite and how to resume it. **Read this first if picking the work back up.** |
| [2026-08-24-benchmark-plan.md](2026-08-24-benchmark-plan.md) | Why the suite measures the regimes it does, and how its sizes were calibrated. The design note the built suite came from. |
| [2026-08-27-pairing-guests.md](2026-08-27-pairing-guests.md) | Whether a pairing can be measured through SP1's precompiles, the one test that decided it, and the rule that a number nobody would deploy is worse than no number. |
| [2026-08-28-thermal.md](2026-08-28-thermal.md) | Keeping a measurement honest on hardware that throttles: which heat signals the OS actually offers, what the suite does about it, and the pass that panicked a machine by not doing it. |

## Historical record

| document | what it is |
|---|---|
| [2026-08-22-spikes.md](2026-08-22-spikes.md) | The three unknowns that could each have sunk the backend, and what answering them found. The refit here moved the projection from 6.5x to 8.2x. |

## A note on the numbers

Ratios in these documents are not all against the same baseline, and the difference matters more than any single figure.

- **Against the prover as it ships** — scalar, which is what SP1 6.4.0 actually runs. This is the D7 reading and the source of the headline figure, now 6.0x at its best.
- **Against a packed CPU baseline** — the same algorithms using this machine's vector units. The fair silicon-to-silicon comparison, and about 3x smaller.

The CPU prover is still scalar, so both readings remain true of different questions. Any figure quoted without its baseline is a figure to distrust, including in these documents.
