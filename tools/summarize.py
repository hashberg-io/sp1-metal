#!/usr/bin/env python3
"""Turn a probe results file into the two things a decision needs: a table of
GPU-versus-CPU ratios, and a per-phase speedup vector to feed to fit_model.py.

The ratios are the conservative ones. Where a probe reports several CPU
baselines, the packed (NEON) one wins, because that is what the SP1 CPU prover
actually runs; where a phase maps onto several kernels, the slowest ratio wins,
because a phase is not faster than its slowest part.

Usage:
    python3 tools/summarize.py results.json [--factors speedups.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def probe(results: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for r in results:
        if r.get("probe") == name:
            return r
    return None


def ratio(node: dict[str, Any]) -> float | None:
    """The ratio to quote for one kernel: against the packed CPU baseline if
    there is one, otherwise against whatever baseline was measured."""
    if "gpu_over_cpu_packed" in node:
        return float(node["gpu_over_cpu_packed"])
    if "gpu_over_cpu" in node:
        return float(node["gpu_over_cpu"])
    return None


def table(results: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    """Every measured ratio, as (probe, kernel, gpu/cpu)."""
    rows: list[tuple[str, str, float]] = []
    for r in results:
        name = r["probe"]
        top = ratio(r)
        if top is not None:
            rows.append((name, name, top))
        for key in ("base_mul", "ext_mul"):
            if key in r and (v := ratio(r[key])) is not None:
                rows.append((name, key, v))
        for kernel, node in r.get("kernels", {}).items():
            if (v := ratio(node)) is not None:
                rows.append((name, kernel, v))
    return rows


def factors(results: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, str]]:
    """Per-phase speedup factors, with the measurement each one came from.

    execute and tracegen stay at 1.0: they are CPU-resident work that no probe
    in this suite prices, and claiming a speedup for them would be inventing
    one. The pipelined model is where their cost gets recovered, by overlapping
    them with GPU phases, not by pretending they run on the GPU.
    """
    rows = {(p, k): v for p, k, v in table(results)}
    out: dict[str, float] = {"execute": 1.0, "tracegen": 1.0, "setup": 1.0, "other": 1.0}
    why: dict[str, str] = {
        "execute": "CPU-resident (RISC-V emulation), unpriced",
        "tracegen": "CPU-resident witness generation, unpriced",
        "setup": "one-off, unpriced",
        "other": "unattributed span time, unpriced",
    }
    if (v := rows.get(("zerocheck", "zerocheck"))) is not None:
        out["zerocheck"] = v
        why["zerocheck"] = "zerocheck probe vs packed cache-tiled CPU"
    if (v := rows.get(("poseidon2", "poseidon2"))) is not None:
        out["commit"] = v
        why["commit"] = "poseidon2 probe vs packed CPU permutation"
    fold_keys = [("fold", k) for k in ("fold_ext", "flv_ext", "fold_base", "flv_base")]
    fold = [rows[k] for k in fold_keys if k in rows]
    if fold:
        out["pcs"] = min(fold)
        why["pcs"] = f"slowest of the fold kernels vs packed CPU ({min(fold):.2f} of {len(fold)})"
    gkr = [rows[k] for k in (("fieldops", "ext_mul"), ("fold", "fold_ext")) if k in rows]
    if gkr:
        out["gkr"] = min(gkr)
        why["gkr"] = "slowest of extension multiply and extension fold vs packed CPU"
    if "zerocheck" in out:
        out["recursion"] = out["zerocheck"]
        why["recursion"] = "same shapes as a shard proof; borrows the zerocheck factor"
    return out, why


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="results.json written by `probes all --cpu`")
    ap.add_argument("--factors", help="write the per-phase speedup vector here")
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        doc = json.load(f)
    results = doc["result"] if isinstance(doc.get("result"), list) else [doc["result"]]

    print(f"device: {doc.get('host', {}).get('device', 'unknown')}\n")
    print(f"{'probe':<12}{'kernel':<12}{'gpu/cpu':>9}")
    print("-" * 33)
    for p, k, v in table(results):
        print(f"{p:<12}{k:<12}{v:>8.2f}x")

    launch = probe(results, "launch")
    if launch:
        print(
            f"\nfixed costs: {launch['round_trip_us']:.0f} us per command-buffer round trip, "
            f"{launch['marginal_dispatch_us']:.2f} us per extra dispatch in one buffer"
        )
        tail, batched = launch["fold_tail"], launch["fold_tail_batched"]
        print(
            f"fold tail ({launch['params']['rounds']} rounds): {tail['total_ms']:.2f} ms with a "
            f"round trip per round, {batched['total_ms']:.2f} ms batched "
            f"({batched['speedup_over_round_trips']:.1f}x)"
        )

    f, why = factors(results)
    print("\nper-phase factors:")
    for phase in sorted(f):
        print(f"  {phase:<12}{f[phase]:>6.2f}x   {why[phase]}")
    if args.factors:
        with open(args.factors, "w", encoding="utf-8") as out:
            json.dump(f, out, indent=2, sort_keys=True)
            out.write("\n")
        print(f"\nwrote {args.factors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
