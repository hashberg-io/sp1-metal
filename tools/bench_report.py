#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Turn a bench_suite.py JSON Lines file into a report.

Medians, not means, and the range beside every median: the Metal arm's
run-to-run spread reaches 25% at six shards and about a third on zerocheck's
own GPU time, so a single number from a single pair has misled this project
more than once.

Usage: tools/bench_report.py results/bench-standard.jsonl [--md]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict


def load(path: str):
    meta, rows = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") == "meta":
                meta = r
            elif not r.get("warmup"):
                rows.append(r)
    return meta, rows


def key(r):
    return (r["guest"], r["size"], r["config"], r["mode"], r["shard_size"])


def summarise(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[key(r)][r["arm"]].append(r)
    out = []
    for k, arms in sorted(by.items()):
        guest, size, config, mode, shard = k
        rec = {"guest": guest, "size": size, "config": config, "mode": mode,
               "shard_mib": shard >> 20}
        for arm in ("cpu", "gpu"):
            got = [x for x in arms.get(arm, []) if x.get("verified")]
            failed = [x for x in arms.get(arm, []) if not x.get("verified")]
            rec[f"{arm}_fail"] = len(failed)
            if got:
                w = [x["wall_s"] for x in got]
                rec[f"{arm}_med"] = st.median(w)
                rec[f"{arm}_min"] = min(w)
                rec[f"{arm}_max"] = max(w)
                rec[f"{arm}_n"] = len(w)
                rss = [x["peak_rss_gb"] for x in got if x.get("peak_rss_gb")]
                if rss:
                    rec[f"{arm}_rss"] = st.median(rss)
        if rec.get("cpu_med") and rec.get("gpu_med"):
            rec["speedup"] = rec["cpu_med"] / rec["gpu_med"]
        one = arms.get("gpu") or arms.get("cpu") or []
        if one:
            rec["cycles"] = one[0].get("cycles")
            rec["shards"] = one[0].get("shards")
            rec["tier"] = one[0].get("tier")
        out.append(rec)
    return out


def fmt(v, spec=".2f", dash="-"):
    return dash if v is None else format(v, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--md", action="store_true", help="markdown table")
    args = ap.parse_args()

    meta, rows = load(args.path)
    recs = summarise(rows)
    fails = [r for r in rows if not r.get("verified")]

    if meta:
        print(f"# {meta.get('tier')} tier — {meta.get('host')} — git {meta.get('git')} — {meta.get('ts')}")
        print(f"{len(rows)} timed runs, {len(recs)} measured points, {len(fails)} failures\n")

    hdr = ["guest", "size", "config", "cyc", "shd", "cpu_med", "gpu_med", "speedup", "cpu_GB", "gpu_GB"]
    if args.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
    else:
        print(f"{'guest':<11}{'size':<9}{'config':<20}{'cyc':>10}{'shd':>4}"
              f"{'cpu_med':>9}{'gpu_med':>9}{'speedup':>9}{'cpu_GB':>8}{'gpu_GB':>8}")
    for r in recs:
        cells = [r["guest"], r["size"], r["config"],
                 f"{r['cycles']:,}" if r.get("cycles") else "-",
                 str(r.get("shards") or "-"),
                 fmt(r.get("cpu_med")), fmt(r.get("gpu_med")),
                 fmt(r.get("speedup"), ".2f") + ("x" if r.get("speedup") else ""),
                 fmt(r.get("cpu_rss")), fmt(r.get("gpu_rss"))]
        if args.md:
            print("| " + " | ".join(cells) + " |")
        else:
            print(f"{cells[0]:<11}{cells[1]:<9}{cells[2]:<20}{cells[3]:>10}{cells[4]:>4}"
                  f"{cells[5]:>9}{cells[6]:>9}{cells[7]:>9}{cells[8]:>8}{cells[9]:>8}")

    ok = [r for r in recs if r.get("speedup")]
    if ok:
        sp = [r["speedup"] for r in ok]
        print(f"\nspeedup across {len(sp)} points: median {st.median(sp):.2f}x, "
              f"range {min(sp):.2f}x - {max(sp):.2f}x")
        best = max(ok, key=lambda r: r["speedup"])
        worst = min(ok, key=lambda r: r["speedup"])
        print(f"  best  {best['guest']}/{best['size']} {best['speedup']:.2f}x")
        print(f"  worst {worst['guest']}/{worst['size']} {worst['speedup']:.2f}x")

    if fails:
        print(f"\n{len(fails)} FAILED runs (correctness or timeout):")
        seen = set()
        for f in fails:
            k = (f["guest"], f["size"], f["arm"], f.get("error", "unverified"))
            if k in seen:
                continue
            seen.add(k)
            print(f"  {f['guest']}/{f['size']} {f['arm']}: {f.get('error', 'did not verify')}"
                  f" after {f.get('elapsed_s')}s")
    else:
        print("\nEvery run verified its proof.")


if __name__ == "__main__":
    main()
