#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Predict end-to-end Metal speedup from a span profile and per-phase GPU speedups.

Inputs: the phase totals JSON from tools/span_profile.py, and a JSON object mapping phase names to expected GPU speedup factors (phases absent from the map, or mapped to 1, stay CPU-resident).
Two predictions are reported: serial (GPU phases compress, CPU phases unchanged, everything sequential) and pipelined (CPU-resident work overlaps GPU work across shards, so wall time is the maximum of the two, plus any phases marked serial-only).
The pipelined figure is the design target; the serial figure is the floor a naive integration would hit.
"""

import argparse
import json


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def predict(phases: dict[str, float], speedups: dict[str, float], serial_only: set[str]) -> dict[str, float]:
    total = sum(phases.values())
    cpu_resident = 0.0
    gpu_time = 0.0
    serial_time = 0.0
    for phase, seconds in phases.items():
        factor = float(speedups.get(phase, 1.0))
        if phase in serial_only:
            serial_time += seconds / factor
        elif factor <= 1.0:
            cpu_resident += seconds
        else:
            gpu_time += seconds / factor
    serial_total = cpu_resident + gpu_time + serial_time
    pipelined_total = max(cpu_resident, gpu_time) + serial_time
    return {
        "baseline_seconds": total,
        "cpu_resident_seconds": cpu_resident,
        "gpu_seconds": gpu_time,
        "serial_only_seconds": serial_time,
        "serial_prediction_seconds": serial_total,
        "serial_speedup": total / serial_total if serial_total > 0 else float("inf"),
        "pipelined_prediction_seconds": pipelined_total,
        "pipelined_speedup": total / pipelined_total if pipelined_total > 0 else float("inf"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", help="phase totals JSON from span_profile.py --json")
    parser.add_argument("speedups", help="JSON mapping phase name to expected GPU speedup factor")
    parser.add_argument(
        "--serial-only",
        default="setup,recursion",
        help="comma-separated phases that neither pipeline nor stay CPU-resident (default: setup,recursion)",
    )
    args = parser.parse_args()
    phases: dict[str, float] = load(args.profile)["phases"]
    speedups: dict[str, float] = load(args.speedups)
    serial_only = {p.strip() for p in args.serial_only.split(",") if p.strip()}
    result = predict(phases, speedups, serial_only)
    print(f"baseline (profile total):   {result['baseline_seconds']:>10.1f}s")
    print(f"CPU-resident (pipelines):   {result['cpu_resident_seconds']:>10.1f}s")
    print(f"GPU work after speedups:    {result['gpu_seconds']:>10.1f}s")
    print(f"serial-only after speedups: {result['serial_only_seconds']:>10.1f}s")
    print()
    print(f"serial integration:    {result['serial_prediction_seconds']:>10.1f}s  ({result['serial_speedup']:.2f}x)")
    print(f"pipelined integration: {result['pipelined_prediction_seconds']:>10.1f}s  ({result['pipelined_speedup']:.2f}x)")
    print()
    print("D7 shipping bar is 5.00x end to end; the pipelined figure is the relevant one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
