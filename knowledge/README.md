# Knowledge

Curated notes on what this project measured, built and concluded.
They are working documents kept as a record, not a tidied-up narrative, and several of them are wrong in interesting ways that later documents correct — which is the point of keeping them.

**Read [verdict.md](verdict.md) first.** It is the only document that states the current answer, and it says which of the project's earlier claims survived and which did not.

## Current

| document | what it is |
|---|---|
| [verdict.md](verdict.md) | The answer: what was asked, what was measured, what is still not met. |
| [backend.md](backend.md) | How the backend works and how it got there, in twenty-one sections, in the order they happened. |
| [next-levers.md](next-levers.md) | Five rounds of optimisation from 3.74x to 5.56x, every ranked list, and every negative result. |
| [measurements.md](measurements.md) | What the machine can do: the two silicon ratios, per-kernel throughput, the phase profile of a real proof. Measured 2026-08-21. |
| [probes.md](probes.md) | The probe suite — what each of the fourteen probes measures, how to run it, and its limits. |
| [desiderata.md](desiderata.md) | The requirements this work was to meet. D7 is the bar the verdict is judged against, and it is still not met. |

## Executed plans

Written before the work, kept because the gap between a plan and its outcome is usually the most useful thing in it.
Each records its own result at the end; do not read the opening as a live statement.

| document | outcome |
|---|---|
| [backend-plan.md](backend-plan.md) | Executed. It opens by quoting a verdict that has since been overturned, and its projection of 10.7x proved optimistic by roughly a factor of two. |
| [cpu-plan.md](cpu-plan.md) | Executed. Three ports took the backend from 3.40x to 3.74x, which is close to what it predicted, for the reason it gave. |
| [resident-plan.md](resident-plan.md) | Stage 1 executed, 2.60x to 3.40x, and it was one false page-alignment rule. Stages 2 to 4 deliberately not built; the document says why. |
| [release-plan.md](release-plan.md) | Executed. Items 1 to 6 and the v0.1.0 tag; the push is the only part left. |

## Benchmarking

| document | what it is |
|---|---|
| [benchmark-status.md](benchmark-status.md) | Where the benchmark work stands, what it found, how to check on the running suite and how to resume it. **Read this first if picking the work back up.** |
| [benchmark-plan.md](benchmark-plan.md) | A suite that measures the backend across the regimes it is sensitive to and checks correctness from the same runs. Not built; blocked on the SP1 guest toolchain. |

## Historical record

| document | what it is |
|---|---|
| [spikes.md](spikes.md) | The three unknowns that could each have sunk the backend, and what answering them found. The refit here moved the projection from 6.5x to 8.2x. |

## A note on the numbers

Ratios in these documents are not all against the same baseline, and the difference matters more than any single figure.

- **Against the prover as it ships** — scalar, which is what SP1 6.4.0 actually runs. This is the D7 reading and the source of the headline 5.56x.
- **Against a packed CPU baseline** — the same algorithms using this machine's vector units. The fair silicon-to-silicon comparison, and about 3x smaller.

The CPU prover is still scalar, so both readings remain true of different questions. Any figure quoted without its baseline is a figure to distrust, including in these documents.
