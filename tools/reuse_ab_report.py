#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Read a `reuse_ab.py` run and say whether the hand-off is worth its hazard.

The comparison is paired: within one repeat the two arms ran back to back on
the same machine in the same thermal state, so the difference between them is
a far better measurement than either arm's average. What is reported is the
distribution of those per-pair differences, and beside it the noise floor --
how much the *same* arm varies from repeat to repeat. A difference smaller than
the noise floor is not a small effect that needs more samples; it is an effect
this workload cannot distinguish from zero, which for a decision about whether
to keep a hazard is the same answer.

Every pair is checked before it is counted: the on arm must show buffers kept
and the off arm must show none. An A/B whose switch silently did nothing looks
exactly like a null result, and this is the only thing that tells them apart.

Usage: tools/reuse_ab_report.py <run.jsonl> [--md]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics


def bootstrap_ci(values: list[float], f=statistics.median, n: int = 20000, conf: float = 0.95) -> tuple:
    """A percentile interval for `f` over `values`, resampled with replacement.

    Eight pairs is too few to lean on a normal approximation and enough to
    resample. Seeded, so the reported interval is the same every time the
    report is rendered from the same data.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(20260829)
    stats = sorted(f(rng.choices(values, k=len(values))) for _ in range(n))
    lo = stats[int((1 - conf) / 2 * n)]
    hi = stats[int((1 + conf) / 2 * n) - 1]
    return (lo, hi)


def noise_floor(values: list[float]) -> float:
    """Median absolute difference between successive runs of one arm.

    The scale on which this workload's timings wander when nothing has been
    changed, which is what any claimed effect has to clear.
    """
    if len(values) < 2:
        return float("nan")
    return statistics.median(abs(b - a) for a, b in zip(values, values[1:]))


def load(path: str):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    meta = next((r for r in rows if r.get("kind") == "meta"), {})
    runs = [r for r in rows if r.get("kind") != "meta"]
    return meta, runs


def analyse(runs: list[dict]) -> tuple[list[dict], list[str]]:
    by_point = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in runs:
        by_point[(r["guest"], r["size"])][r["repeat"]][r["arm"]] = r

    out, problems = [], []
    for point, reps in sorted(by_point.items()):
        pairs = []
        for rep, arms in sorted(reps.items()):
            on, off = arms.get("on"), arms.get("off")
            if not on or not off:
                problems.append(f"{point[0]}/{point[1]} r{rep}: only one arm ran")
                continue
            if not (on.get("verified") and off.get("verified")):
                problems.append(f"{point[0]}/{point[1]} r{rep}: an arm did not verify its proof")
                continue
            # The switch's own evidence. Without this a pair where the
            # environment never reached the prover reads as a clean null.
            if on.get("kept_total", 0) <= 0:
                problems.append(f"{point[0]}/{point[1]} r{rep}: the on arm kept no buffers")
                continue
            if off.get("kept_total", 0) != 0:
                problems.append(f"{point[0]}/{point[1]} r{rep}: the off arm kept buffers")
                continue
            pairs.append((on, off))
        if not pairs:
            continue
        row = {"guest": point[0], "size": point[1], "n_pairs": len(pairs)}
        for field in ("wall_s", "blocked_s", "gpu_s"):
            on_v = [p[0].get(field) for p in pairs]
            off_v = [p[1].get(field) for p in pairs]
            if any(v is None for v in on_v + off_v):
                continue
            deltas = [o - n for n, o in zip(on_v, off_v)]  # positive: reuse is faster
            lo, hi = bootstrap_ci(deltas)
            row[field] = {
                "on_median": statistics.median(on_v),
                "off_median": statistics.median(off_v),
                "delta_median": statistics.median(deltas),
                "delta_ci": (lo, hi),
                "pct": statistics.median(deltas) / statistics.median(off_v) * 100.0,
                "favour_on": sum(1 for d in deltas if d > 0),
                "noise_on": noise_floor(on_v),
                "noise_off": noise_floor(off_v),
            }
        row["kept"] = statistics.median([p[0]["kept_total"] for p in pairs])
        row["collided"] = sum(p[0].get("collided_total", 0) for p in pairs)
        out.append(row)
    return out, problems


def render(meta: dict, rows: list[dict], problems: list[str], md: bool) -> str:
    L = []
    add = L.append
    if md:
        add(f"# Is the buffer hand-off worth keeping?\n")
        add(f"`SP1_METAL_REUSE` on against off, both arms the GPU path through one binary, "
            f"git `{meta.get('git')}`, {meta.get('host')}, {meta.get('repeats')} pairs per point, "
            f"measured {meta.get('ts', '')[:10]}.\n")
    else:
        add(f"SP1_METAL_REUSE A/B -- git {meta.get('git')} -- {meta.get('host')}")
    for r in rows:
        w, b = r.get("wall_s"), r.get("blocked_s")
        if not w:
            continue
        if md:
            add(f"\n## {r['guest']}/{r['size']}\n")
            add(f"| metric | reuse on | reuse off | median difference | 95% interval | noise floor | pairs favouring on |")
            add(f"|---|---|---|---|---|---|---|")
            for label, f in (("wall clock", w), ("blocked (thread-s)", b), ("gpu (thread-s)", r.get("gpu_s"))):
                if not f:
                    continue
                add(f"| {label} | {f['on_median']:.2f} s | {f['off_median']:.2f} s | "
                    f"{f['delta_median']:+.2f} s ({f['pct']:+.1f}%) | "
                    f"{f['delta_ci'][0]:+.2f} to {f['delta_ci'][1]:+.2f} s | "
                    f"±{max(f['noise_on'], f['noise_off']):.2f} s | "
                    f"{f['favour_on']}/{r['n_pairs']} |")
            add(f"\n{int(r['kept'])} buffers kept per proof with the hand-off on; "
                f"{r['collided']} of them landed on an address a dead entry still held.")
        else:
            add(f"\n{r['guest']}/{r['size']}  ({r['n_pairs']} pairs, {int(r['kept'])} hand-offs a proof)")
            for label, f in (("wall   ", w), ("blocked", b), ("gpu    ", r.get("gpu_s"))):
                if not f:
                    continue
                add(f"  {label} on {f['on_median']:7.2f}s  off {f['off_median']:7.2f}s  "
                    f"delta {f['delta_median']:+6.2f}s ({f['pct']:+5.1f}%)  "
                    f"ci [{f['delta_ci'][0]:+.2f},{f['delta_ci'][1]:+.2f}]  "
                    f"noise +-{max(f['noise_on'], f['noise_off']):.2f}s  "
                    f"{f['favour_on']}/{r['n_pairs']} favour on")
    if problems:
        add("\nexcluded pairs:" if not md else "\n## Excluded pairs\n")
        for p in problems:
            add(f"  {p}" if not md else f"- {p}")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    meta, runs = load(a.path)
    rows, problems = analyse(runs)
    print(render(meta, rows, problems, a.md), end="")


if __name__ == "__main__":
    main()
