#!/usr/bin/env python3
"""Turn the acceptance scaling table into the relationship it implies.

Reads the TSV written by the acceptance study (axis, guest, n, shard_size,
cpu_s, gpu_s, speedup) and reports, per axis, how the speedup of the Metal path
moves with the variable that axis varies. The point of separating the axes is
that SP1 shards at a fixed cycle budget, so "more work" and "bigger shards" are
different experiments: one adds shards, the other makes each one's GKR layers
bigger, and only the second is what the kernels are sensitive to.
"""

import sys
from math import log2


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header) or "FAIL" in parts:
                continue
            row = dict(zip(header, parts))
            try:
                row["cpu_s"] = float(row["cpu_s"])
                row["gpu_s"] = float(row["gpu_s"])
                row["speedup"] = row["cpu_s"] / row["gpu_s"]
                row["n"] = int(row["n"])
                row["shard_size"] = int(row["shard_size"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    return rows


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept of y against x."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    return slope, my - slope * mx


def report(rows: list[dict], axis: str, key: str, label: str) -> None:
    subset = [r for r in rows if r["axis"] == axis]
    if len(subset) < 2:
        return
    subset.sort(key=lambda r: r[key])
    print(f"\n{label}")
    for r in subset:
        print(
            f"  {r['guest']:5s} {key}={r[key]:<10} cpu {r['cpu_s']:7.2f}s"
            f"  metal {r['gpu_s']:7.2f}s  {r['speedup']:5.3f}x"
        )
    xs = [log2(r[key]) for r in subset]
    ys = [r["speedup"] for r in subset]
    slope, _ = fit(xs, ys)
    lo, hi = subset[0], subset[-1]
    print(
        f"  -> {slope:+.3f}x per doubling of {key}"
        f"  ({lo['speedup']:.3f}x at {lo[key]} to {hi['speedup']:.3f}x at {hi[key]})"
    )


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/scaling.tsv"
    rows = load(path)
    if not rows:
        print(f"no usable rows in {path}")
        return 1
    report(rows, "shard_size", "shard_size", "Shard size (fixed work): bigger shards mean bigger GKR layers")
    report(rows, "work", "n", "Total work (fixed shard size): more shards of the same size")
    others = [r for r in rows if r["axis"] == "guest"]
    if others:
        print("\nDifferent computations at one setting")
        for r in others:
            print(
                f"  {r['guest']:5s} n={r['n']:<10} cpu {r['cpu_s']:7.2f}s"
                f"  metal {r['gpu_s']:7.2f}s  {r['speedup']:5.3f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
