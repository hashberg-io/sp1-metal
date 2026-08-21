#!/usr/bin/env python3
"""Aggregate tracing span timings from an SP1 CPU prover log into phase buckets.

Feed it the output of an SP1 host built with `sp1_sdk::utils::setup_logger()` and run with `RUST_LOGGER=flat RUST_LOG=debug`.
The flat logger writes to stdout and colours its output even when that is a file, so capture with `> prove.log 2>&1`; this script strips the escape codes itself.
The flat logger emits one `close time.busy=.. time.idle=..` line per completed span; busy time includes time spent in child spans entered on the same thread, so this script reports exclusive time per span path (total minus direct children) before bucketing.
Phase fractions from this profile are the main input to tools/fit_model.py.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LINE_RE = re.compile(
    r"^\S+\s+(?:INFO|DEBUG|TRACE|WARN|ERROR)\s+(?P<path>.+?):\s+close\s+"
    r"time\.busy=(?P<busy>\S+)\s+time\.idle=(?P<idle>\S+)\s*$"
)
FIELDS_RE = re.compile(r"\{[^{}]*\}")
DUR_RE = re.compile(r"^(\d+(?:\.\d+)?)(ns|µs|us|ms|s|m)$")
UNIT_SECONDS: dict[str, float] = {
    "ns": 1e-9,
    "µs": 1e-6,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
}
EXACT_BUCKETS: dict[str, str] = {
    "execute_program": "execute",
    "runtime.run(...)": "execute",
    "generate main traces": "tracegen",
    "generate full traces": "tracegen",
    "generate dependencies": "tracegen",
    "splicing": "tracegen",
    "splicing trace chunk": "tracegen",
    "emit globals": "tracegen",
    "zerocheck": "zerocheck",
    "generate GKR circuit": "gkr",
    "prove GKR circuit": "gkr",
    "logup gkr proof": "gkr",
    "commit traces": "commit",
    "prove evaluation claims": "pcs",
    "jagged evaluation proof": "pcs",
    "compute jagged values": "pcs",
    "create jagged sumcheck poly": "pcs",
    "Dense PCS evaluation proof": "pcs",
    "setup": "setup",
    "setup_from_preprocessed_data_and_traces": "setup",
    "SetupVkey": "setup",
    "initial_global_cumulative_sum": "tracegen",
}
SUBSTRING_BUCKETS: list[tuple[str, str]] = [
    ("recursion", "recursion"),
    ("reduce", "recursion"),
    ("shrink", "recursion"),
    ("wrap", "recursion"),
    ("deferred", "recursion"),
    ("merkle", "commit"),
    ("poseidon", "commit"),
    ("commit", "commit"),
    ("sumcheck", "pcs"),
    ("basefold", "pcs"),
    ("jagged", "pcs"),
    ("gkr", "gkr"),
    ("trace", "tracegen"),
    ("execute", "execute"),
    ("setup", "setup"),
]


def parse_duration(text: str) -> float:
    match = DUR_RE.match(text)
    if match is None:
        raise ValueError(f"unrecognised duration: {text!r}")
    return float(match.group(1)) * UNIT_SECONDS[match.group(2)]


def span_path(raw: str) -> tuple[str, ...]:
    stripped = FIELDS_RE.sub("", raw)
    return tuple(seg.strip() for seg in stripped.split(":") if seg.strip())


def parse_log(lines: list[str]) -> dict[tuple[str, ...], float]:
    totals: dict[tuple[str, ...], float] = defaultdict(float)
    for line in lines:
        match = LINE_RE.match(ANSI_RE.sub("", line).rstrip())
        if match is None:
            continue
        path = span_path(match.group("path"))
        if path:
            totals[path] += parse_duration(match.group("busy"))
    return dict(totals)


def exclusive_times(totals: dict[tuple[str, ...], float]) -> dict[tuple[str, ...], float]:
    child_sums: dict[tuple[str, ...], float] = defaultdict(float)
    for path, busy in totals.items():
        if len(path) > 1:
            child_sums[path[:-1]] += busy
    return {path: max(0.0, busy - child_sums.get(path, 0.0)) for path, busy in totals.items()}


def bucket_of(name: str) -> str:
    if name in EXACT_BUCKETS:
        return EXACT_BUCKETS[name]
    lowered = name.lower()
    for needle, bucket in SUBSTRING_BUCKETS:
        if needle in lowered:
            return bucket
    return "other"


def aggregate(exclusive: dict[tuple[str, ...], float]) -> tuple[dict[str, float], dict[str, float]]:
    phases: dict[str, float] = defaultdict(float)
    by_span: dict[str, float] = defaultdict(float)
    for path, seconds in exclusive.items():
        leaf = path[-1]
        phases[bucket_of(leaf)] += seconds
        by_span[leaf] += seconds
    return dict(phases), dict(by_span)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="path to the prover log, or - for stdin")
    parser.add_argument("--json", dest="json_out", help="write phase totals to this JSON file")
    parser.add_argument("--top", type=int, default=20, help="how many spans to list (default 20)")
    args = parser.parse_args()
    lines = sys.stdin.readlines() if args.log == "-" else open(args.log, encoding="utf-8").readlines()
    totals = parse_log(lines)
    if not totals:
        print("no `close time.busy=` span lines found; was the host run with RUST_LOGGER=flat RUST_LOG=debug?", file=sys.stderr)
        return 1
    exclusive = exclusive_times(totals)
    phases, by_span = aggregate(exclusive)
    grand_total = sum(phases.values())
    print(f"total accounted span time: {grand_total:.1f}s over {len(totals)} span paths")
    print()
    print(f"{'phase':<12} {'seconds':>12} {'share':>8}")
    for phase, seconds in sorted(phases.items(), key=lambda kv: -kv[1]):
        print(f"{phase:<12} {seconds:>12.1f} {seconds / grand_total:>7.1%}")
    print()
    print(f"top {args.top} spans by exclusive time:")
    for name, seconds in sorted(by_span.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"  {seconds:>10.1f}s  [{bucket_of(name):<9}] {name}")
    if args.json_out:
        payload = {"phases": phases, "spans": by_span, "total_seconds": grand_total}
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
