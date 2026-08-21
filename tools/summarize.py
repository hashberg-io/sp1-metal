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
        if "butterfly_stage" in r and (v := ratio(r["butterfly_stage"])) is not None:
            rows.append((name, "ntt_stage", v))
    return rows


def factors(results: list[dict[str, Any]]) -> dict[str, tuple[float, float, str]]:
    """Per-phase speedup factors as (conservative, optimistic, provenance).

    A phase gets a single number only where a probe measures the work that
    phase actually does. Where it does not, the phase gets a range whose ends
    are both measured quantities, and the provenance says what is missing:

    - `commit` is a Reed-Solomon encode (a radix-2 DFT) followed by Merkle
      hashing. The suite prices the hashing and not the encode, so the range
      runs from the streaming-bandwidth ratio to the Poseidon2 ratio.
    - `gkr` is sumcheck over the extension field. Its rounds have the shape the
      zerocheck probe measures but over extension values, so the range runs
      from the streamed extension multiply to the zerocheck round.
    - `execute` and `tracegen` stay at 1.0: CPU-resident work no probe prices,
      and claiming a speedup for them would be inventing one. The pipelined
      model recovers their cost by overlapping them, not by moving them.
    """
    rows = {(p, k): v for p, k, v in table(results)}
    stream = rows.get(("triad", "triad"), 1.0)
    out: dict[str, tuple[float, float, str]] = {
        "execute": (1.0, 1.0, "CPU-resident (RISC-V emulation), unpriced"),
        "tracegen": (1.0, 1.0, "CPU-resident witness generation, unpriced"),
        "setup": (1.0, 1.0, "one-off, unpriced"),
        "other": (1.0, 1.0, "unattributed span time, unpriced"),
        "verify": (1.0, 1.0, "proof checking, unpriced"),
    }
    zc = rows.get(("zerocheck", "zerocheck"))
    if zc is not None:
        out["zerocheck"] = (zc, zc, "zerocheck probe vs packed cache-tiled CPU")
    p2 = rows.get(("poseidon2", "poseidon2"))
    encode = probe(results, "encode")
    if p2 is not None and encode is not None:
        out["commit"] = commit_factor(results, encode, p2, rows)
    elif p2 is not None:
        out["commit"] = (stream, p2, "ENCODE UNPRICED: streaming ratio to Poseidon2 ratio")
    fold = [rows[k] for k in [("fold", n) for n in ("fold_ext", "flv_ext", "fold_base", "flv_base")] if k in rows]
    if fold:
        out["pcs"] = (min(fold), max(fold), "slowest to fastest fold kernel vs packed CPU")
    ext = rows.get(("fieldops", "ext_mul"))
    if ext is not None and zc is not None:
        out["gkr"] = (min(ext, zc), max(ext, zc), "SHAPE UNPRICED: streamed extension multiply to zerocheck round")
    if zc is not None:
        out["recursion"] = (min(zc, stream), zc, "same shapes as a shard proof, at a fraction of the size")
    return out


def commit_factor(
    results: list[dict[str, Any]],
    encode: dict[str, Any],
    p2_ratio: float,
    rows: dict[tuple[str, str], float],
) -> tuple[float, float, str]:
    """Committing a trace is a Reed-Solomon encode followed by a Merkle tree, so
    its factor is the two factors mixed in the proportion the CPU spends on
    them. The encode's own factor is bounded rather than measured: the butterfly
    stage is the floor (no block kept resident across stages) and the register
    multiply roofline is the ceiling (perfectly blocked).
    """
    enc_ns = float(encode["cpu"]["nanoseconds_per_input_cell"])
    p2 = probe(results, "poseidon2")
    perms_per_s = float(p2["cpu_packed"]["mega_perms_per_s_best"]) * 1e6
    # The tree is built over the codeword, which the blowup makes twice the
    # trace; a padding-free sponge of rate 8 hashes eight cells per permutation.
    blowup = 1 << int(encode["params"]["log_blowup"])
    hash_ns = blowup / 8.0 / perms_per_s * 1e9
    w_enc = enc_ns / (enc_ns + hash_ns)
    lo_enc = float(encode["butterfly_stage"]["gpu_over_cpu_packed"])
    hi_enc = rows.get(("imul", "imul"), lo_enc)
    lo = 1.0 / (w_enc / lo_enc + (1.0 - w_enc) / p2_ratio)
    hi = 1.0 / (w_enc / hi_enc + (1.0 - w_enc) / p2_ratio)
    return (
        lo,
        hi,
        f"encode {w_enc:.0%} of the phase at {lo_enc:.2f}x..{hi_enc:.2f}x, hashing at {p2_ratio:.1f}x",
    )


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

    f = factors(results)
    print("\nper-phase factors (conservative .. optimistic):")
    for phase in sorted(f):
        lo, hi, why = f[phase]
        span = f"{lo:.2f}x" if lo == hi else f"{lo:.2f}x .. {hi:.2f}x"
        print(f"  {phase:<12}{span:<18}{why}")
    if args.factors:
        stem = args.factors.removesuffix(".json")
        for label, index in (("low", 0), ("high", 1)):
            path = f"{stem}-{label}.json"
            with open(path, "w", encoding="utf-8") as out:
                json.dump({k: v[index] for k, v in f.items()}, out, indent=2, sort_keys=True)
                out.write("\n")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
