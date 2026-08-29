#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Does keeping a buffer between two kernels pay for the hazard it carries?

`sp1_metal_core::handoff` keeps a Metal buffer alive between the kernel that
wrote an array and the kernel that reads it, so the largest allocation in a
proof is not wired into a buffer twice. The table is keyed on a pointer, and a
pointer-keyed cache that outlives its allocation is a use-after-free waiting
for the allocator to repeat itself -- which is what
`knowledge/2026-08-28-defect-wrong-proof.md` traced an intermittent wrong proof
to. That defect is fixed and the exposure is bounded, but the module's own note
says the benefit "is small enough that it needs an A/B on one binary to be seen
at all". If the benefit is nil, `SP1_METAL_REUSE` defaulting to off removes the
hazard class rather than bounding it, which is the better answer.

This measures it. Both arms are the GPU arm through one binary, differing only
in `SP1_METAL_REUSE`, and the workloads are the three the hand-off should
flatter most: the ones whose `blocked` time -- the wait between committing a
command buffer and the kernel starting, which is what mapping an allocation
costs and the only thing the hand-off can save -- is the largest share of the
proof. An effect absent where it should be largest is absent.

Two guards against measuring something else:

- **The arms alternate.** Repeat `r` runs on-then-off for even `r` and
  off-then-on for odd `r`, so a machine that drifts one way over the run
  contributes to both arms equally instead of to whichever went second.
- **Each arm proves it took effect.** The host reports the hand-off counters,
  so the on arm must show buffers kept and the off arm must show none. A pair
  where that does not hold is recorded and excluded rather than averaged in;
  an A/B whose switch did nothing is the failure mode that looks most like a
  null result.

Usage: tools/reuse_ab.py [--repeats N] [--out results/....jsonl] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_suite as bs  # noqa: E402

ROOT = bs.ROOT

# guest, size label, mode. Chosen by blocked-time share of the GPU proof in
# `results/2026-08-27-m4-max-bench-standard.jsonl` and `-apps.jsonl`: divrem
# and fib large are the most mapping-bound workloads in the suite, and
# btc-headers is the most mapping-bound of the real statements.
POINTS = [
    ("divrem", "large", "core"),
    ("fib", "large", "core"),
    ("btc-headers", "standard", "core"),
]

HANDOFF = re.compile(
    r"handoff (codewords|round arrays): (\d+) kept, (\d+) taken, (\d+) dropped unclaimed, (\d+) on an address"
)


def measure(name: str, n: int, reuse: bool, shard_size: int, mode: str, timeout: int) -> dict:
    """One proof with every GPU gate on and the hand-off switched as asked."""
    env = dict(os.environ)
    env["SHARD_SIZE"] = str(shard_size)
    env["SP1_METAL_MIN_ENTRIES"] = "16384"
    for g in bs.GATES:
        env[g] = "1"
    env["SP1_METAL_REUSE"] = "1" if reuse else "0"

    cmd = ["/usr/bin/time", "-l", "nice", "-n", "5", str(bs.HOST),
           "--elf", str(bs.elf_path(name)), "--stdin-u32", str(n), "--mode", mode]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, start_new_session=True)
    try:
        out = proc.communicate(timeout=timeout)[0]
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), bs.signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        return {"verified": False, "error": "timeout", "elapsed_s": round(time.time() - t0, 1)}

    row = bs.parse(out)
    row["returncode"] = rc
    row["elapsed_s"] = round(time.time() - t0, 3)
    m = re.search(r"([0-9]+)\s+maximum resident set size", out)
    if m:
        row["peak_rss_gb"] = round(int(m.group(1)) / 1e9, 2)
    # The switch's own evidence, not its setting.
    row["handoff"] = {
        what: {"kept": int(k), "taken": int(t), "orphaned": int(o), "collided": int(c)}
        for what, k, t, o, c in HANDOFF.findall(out)
    }
    row["kept_total"] = sum(v["kept"] for v in row["handoff"].values())
    row["collided_total"] = sum(v["collided"] for v in row["handoff"].values())
    if not row.get("verified"):
        row["tail"] = out[-1500:]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not bs.HOST.exists():
        raise SystemExit(f"host not built: {bs.HOST}")

    day = time.strftime("%Y-%m-%d")
    out_path = pathlib.Path(args.out) if args.out else ROOT / f"results/{day}-m4-max-reuse-ab.jsonl"
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    header = {"kind": "meta", "what": "SP1_METAL_REUSE A/B", "git": sha,
              "dirty": bool(dirty), "repeats": args.repeats,
              "host": subprocess.run(["uname", "-mrs"], capture_output=True, text=True).stdout.strip(),
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(header), flush=True)

    shard_size = 4 << 20
    with out_path.open("a") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.flush()
        for name, label, mode in POINTS:
            n = bs.GUESTS[name]["sizes"][label]
            timeout = bs.GUESTS[name].get("timeout", 1800)
            # Kernels compile on first use in a fresh process; discard one run
            # so that cost lands outside the measured pairs.
            bs.wait_until_quiet(args.force)
            w = measure(name, n, True, shard_size, mode, timeout)
            print(f"  warm {name:<12}{label:<9} discarded {w.get('wall_s')}", flush=True)
            bs.cool_down(w.get("elapsed_s") or 0.0, args.force)
            for rep in range(args.repeats):
                # Alternate which arm goes first, so drift over the run lands
                # on both arms rather than on whichever always went second.
                order = [True, False] if rep % 2 == 0 else [False, True]
                for reuse in order:
                    waited = bs.wait_until_quiet(args.force)
                    thermal_before = bs.thermal_state()
                    idle_before = bs.cpu_idle_pct()
                    r = measure(name, n, reuse, shard_size, mode, timeout)
                    r.update({
                        "guest": name, "size": label, "n": n, "mode": mode,
                        "shard_size": shard_size, "arm": "on" if reuse else "off",
                        "repeat": rep, "first": order[0] == reuse,
                        "idle_before_pct": round(idle_before, 1),
                        "thermal_before": thermal_before,
                        "thermal_after": bs.thermal_state(),
                        "settle_wait_s": waited,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    r["cooled_s"] = bs.cool_down(r.get("elapsed_s") or 0.0, args.force)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    status = "ok " if r.get("verified") else "FAIL"
                    print(f"  {status} {name:<12}{label:<9} reuse={'on ' if reuse else 'off'} "
                          f"r{rep} wall={r.get('wall_s') or r.get('error')} "
                          f"blocked={r.get('blocked_s')} kept={r.get('kept_total')}", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
