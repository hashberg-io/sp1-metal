# Results

Every measurement the documents cite, kept as it was taken.

## How these are named

`yyyy-mm-dd-<machine>-<slug>.<ext>`

- **The date is when the measurement was taken**, not when the file was committed. Two runs of the same tier on different days are two files, and neither overwrites the other.
- **The machine is `m4-max` or `m5-max`.** Every figure in this project comes from one of two machines, and both report the same `Darwin 25.6.0 arm64` in the file's own header, so the name is the only thing that tells them apart.
- **The slug says what was measured**, and carries any qualifier that makes the run what it is — `-budget-5s`, `-before-defect-fixes`, `-thermally-contaminated`.

`tools/check_consistency.py` enforces this, so a file that does not follow it fails a check rather than sitting unnoticed.

A run in progress writes to an undated working path — `tools/bench_suite.py` defaults to `results/bench-<tier>.jsonl` — so file it under the convention when it finishes.

## What is here

- **`*-bench-{smoke,standard,full,apps}.{jsonl,md}`** — the benchmark suite's tiers. The JSON Lines file is one object per run and is what everything else is derived from; the `.md` beside it is `tools/bench_report.py` output.
- **`*-comparison.md`** — the two machines against each other.
- **`*-spotcheck-*.jsonl`** — a handful of points re-taken to check a question about a tier, rather than a tier of its own.
- **`*.tsv`, `*.json`** — the probe suite, the scaling ladders, and the phase model's inputs, from the work that preceded the backend.
- **`*.md` with a defect's name** — a reproduction and what it turned out to be.
