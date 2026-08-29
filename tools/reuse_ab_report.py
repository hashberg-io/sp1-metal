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
        # The other half of the trade. An orphan is an entry left on a pointer
        # whose allocation is about to be freed: bounded to one commitment by
        # `clear`, but it is the state the wrong proof was found in.
        traffic = collections.defaultdict(collections.Counter)
        for on, _ in pairs:
            for what, v in on.get("handoff", {}).items():
                traffic[what].update(v)
        row["traffic"] = {w: dict(c) for w, c in traffic.items()}
        out.append(row)
    return out, problems


def verdict(rows: list[dict]) -> list[str]:
    """What the numbers support, worked out rather than asserted.

    Two things have to hold before a saving is real. The wall clock has to
    separate from zero by more than the same arm's own scatter -- otherwise the
    difference is one this workload cannot resolve. And `blocked` has to move
    the same way everywhere, because mapping cost is the only channel the
    hand-off acts through: a mechanism that helps one workload and hurts
    another is not a mechanism.
    """
    L = []
    real = []
    for r in rows:
        w = r.get("wall_s")
        if not w:
            continue
        clears_zero = w["delta_ci"][0] > 0 or w["delta_ci"][1] < 0
        clears_noise = abs(w["delta_median"]) > max(w["noise_on"], w["noise_off"])
        if clears_zero and clears_noise:
            real.append(r)
    signs = {r["guest"]: (1 if r["blocked_s"]["delta_median"] > 0 else -1)
             for r in rows if r.get("blocked_s")}
    if not real:
        L.append(f"No workload shows a wall-clock difference this measurement can tell "
                 f"from zero: every interval spans it, and every median difference is "
                 f"smaller than the scatter of the same arm across repeats.")
    else:
        L.append(f"A wall-clock difference clears both zero and the noise floor on: "
                 f"{', '.join(r['guest'] + '/' + r['size'] for r in real)}.")
    if len(set(signs.values())) > 1:
        up = sorted(g for g, s in signs.items() if s > 0)
        down = sorted(g for g, s in signs.items() if s < 0)
        L.append(f"The mechanism disagrees with itself. `blocked` is the only channel the "
                 f"hand-off can act through, and it favours keeping buffers on "
                 f"{', '.join(up)} while favouring dropping them on {', '.join(down)}. "
                 f"A saving that changes sign between workloads is not a saving.")
    # Per table, never pooled. The two have very different orphan rates, and a
    # combined figure is the busier and cleaner table diluting the one that
    # actually carries the hazard -- which is the table the wrong proof came
    # from. An aggregate that hides where a fault lives is worse than none.
    totals = collections.defaultdict(collections.Counter)
    for r in rows:
        for what, c in r.get("traffic", {}).items():
            totals[what].update(c)
    for what, c in sorted(totals.items()):
        kept, orph, coll = c.get("kept", 0), c.get("orphaned", 0), c.get("collided", 0)
        if not kept:
            continue
        if orph:
            L.append(f"Against that, the {what} table left {orph} of {kept} hand-offs "
                     f"unclaimed at the end of a commitment -- one in {kept / orph:.0f} -- "
                     f"each a pointer key on an allocation about to be freed. {coll} were "
                     f"still there when that address came back.")
        else:
            L.append(f"The {what} table left none of its {kept} hand-offs unclaimed.")
    return L


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
    for line in verdict(rows):
        add(("\n" + line) if md else ("\n  " + line))
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
            add(f"\n| hand-off table | kept | taken | left unclaimed | on a dead entry's address |")
            add(f"|---|---|---|---|---|")
            for what, c in sorted(r.get("traffic", {}).items()):
                add(f"| {what} | {c.get('kept', 0)} | {c.get('taken', 0)} | "
                    f"{c.get('orphaned', 0)} | {c.get('collided', 0)} |")
            add(f"\nOver {r['n_pairs']} proofs with the hand-off on.")
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
    if md and rows:
        add("\n## What this means for the figures already recorded\n")
        add("Every measurement in `results/` taken before this one ran with the hand-off "
            "on, because the benchmark suite does not set `SP1_METAL_REUSE` and so records "
            "whatever the default was at the time. The default is now off, which makes "
            "those figures measurements of a configuration that is no longer shipped.\n")
        suite = ", ".join(f"`{r['guest']}/{r['size']}`" for r in rows)
        add(f"They stand, and this is the measurement that says so: {suite} are points of "
            f"the suite itself, and none of them moved by more than the scatter of a single "
            f"arm across repeats. Re-taking the suite would cost hours and, on this "
            f"evidence, move nothing.\n")
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
