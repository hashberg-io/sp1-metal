#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The benchmark and correctness suite.

Every row is a pair of proofs through the *same* binary, switched by the
environment variables that gate the GPU, so nothing but the path under test
differs between the arms. Both arms verify their proof rather than merely
producing it, so every benchmark row is also a correctness row: a `verified`
field that is not true is a failure however fast the run was.

Results are written as JSON Lines, one object per run, flushed as they happen,
so a suite that is interrupted keeps everything it had measured.

Usage:
    tools/bench_suite.py --tier smoke|standard|full [--out results/bench.jsonl]
                         [--repeats N] [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOST = ROOT / "crates/sp1-profile-host/target/release/sp1-profile-host"

# Every switch that moves work to the GPU, set together so the CPU arm really
# is one: leaving any of them on would understate the pair.
GATES = [
    "SP1_METAL_MERKLE",
    "SP1_METAL_TRANSITION",
    "SP1_METAL_ZEROCHECK",
    "SP1_METAL_UPLOAD_CACHE",
    "SP1_METAL_JAGGED",
    "SP1_METAL_FIRST_LAYER",
    "SP1_METAL_JAGGED_VALUES",
    "SP1_METAL_JAGGED_ASSIST",
    "SP1_METAL_BATCH",
    "SP1_METAL_ENCODE",
]

# name -> (elf crate name, {size label: n})
# Sizes are calibrated so that a guest's standard size lands near the four
# shards of the existing acceptance point, NOT to a common cycle count: a
# precompile syscall is one guest instruction but many trace rows, so cycles
# understate proving work for the Tier B guests by more than an order of
# magnitude.
GUESTS: dict[str, dict] = {}


def guest(name, crate, small, standard, large, tier, stresses, env=None, timeout=1800):
    """Register a guest.

    `env` is applied to every run of this guest, for a workload that cannot be
    proved at the shipped defaults; it is recorded in the config label so no
    row silently claims to be a default-configuration measurement.
    """
    GUESTS[name] = {
        "crate": crate,
        "sizes": {"small": small, "standard": standard, "large": large},
        "tier": tier,
        "stresses": stresses,
        "env": env or {},
        "timeout": timeout,
    }


def elf_path(name: str) -> pathlib.Path:
    crate = GUESTS[name]["crate"]
    built = ROOT / f"guests/{name}/target/elf-compilation/riscv64im-succinct-zkvm-elf/release/{crate}"
    shipped = ROOT / f"guests/{name}/elf/{crate}"
    return built if built.exists() else shipped


def load_average() -> float:
    return os.getloadavg()[0]


def cpu_idle_pct() -> float:
    """Instantaneous CPU idle, which is what "quiet" actually means here.

    The load average on this machine is a phantom: it reads 7 while `top`
    reports 96% idle and nothing of ours is running. Waiting on it cost up to
    100 seconds a run and protected nothing. `top -l 2` discards its first
    sample, which is a since-boot average, and reports the second.
    """
    try:
        out = subprocess.run(["top", "-l", "2", "-n", "0", "-s", "1"],
                             capture_output=True, text=True, timeout=30).stdout
        hits = re.findall(r"([0-9.]+)%\s+idle", out)
        return float(hits[-1]) if hits else 0.0
    except Exception:
        return 0.0


def wait_until_quiet(force: bool, threshold: float = 70.0) -> float:
    """Block until the machine is genuinely idle. Returns seconds waited.

    Deliberately patient rather than fatal: this machine runs its owner's own
    long jobs, and a suite that waits is worth more than one that finishes on
    a busy machine. Every wrong answer in this project's history came from a
    contaminated pair.
    """
    if force:
        return 0.0
    t0 = time.time()
    announced = False
    while True:
        idle = cpu_idle_pct()
        if idle >= threshold:
            return round(time.time() - t0, 1)
        if not announced:
            print(f"  ... waiting for the machine ({idle:.0f}% idle)", flush=True)
            announced = True
        time.sleep(5)
        if time.time() - t0 > 900 and int(time.time() - t0) % 900 < 6:
            print(f"  ... still waiting ({(time.time()-t0)/60:.0f} min, {idle:.0f}% idle)", flush=True)


NUM = r"([0-9]+\.?[0-9]*)"


def parse(out: str) -> dict:
    """Pull every field out of the host's report, loudly."""
    got: dict = {}
    m = re.search(r"proved and verified in " + NUM, out)
    got["verified"] = m is not None
    got["wall_s"] = float(m.group(1)) if m else None

    m = re.search(r"gpu is " + NUM + r"% of the " + NUM + r"s proof", out)
    if m:
        got["device_busy_pct"] = float(m.group(1))
    m = re.search(r"dispatches ([0-9]+) on ([0-9]+) threads \| gpu " + NUM + r"s \| blocked " + NUM + r"s", out)
    if m:
        got["dispatches"] = int(m.group(1))
        got["prover_threads"] = int(m.group(2))
        got["gpu_s"] = float(m.group(3))
        got["blocked_s"] = float(m.group(4))

    # The per-phase table.
    phases = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) > 4 and f[0] in ("gkr", "zerocheck", "commit", "pcs") and re.fullmatch(r"[0-9]+", f[1] or ""):
            try:
                phases[f[0]] = {"dispatch": int(f[1]), "gpu_s": float(f[2]), "wall_s": float(f[3])}
            except ValueError:
                pass
    if phases:
        got["phases"] = phases

    m = re.search(r"^setup\s+([0-9]+)\s", out, re.M)
    if m:
        got["shards"] = int(m.group(1))
    return got


_CYCLES: dict[tuple, int] = {}


def cycles(name: str, n: int) -> int | None:
    """Guest cycles for this input, from a cheap execute-only run."""
    key = (name, n)
    if key not in _CYCLES:
        p = subprocess.run([str(HOST), "--elf", str(elf_path(name)),
                            "--stdin-u32", str(n), "--execute"],
                           capture_output=True, text=True, timeout=1800)
        m = re.search(r"cycles ([0-9]+)", p.stdout + p.stderr)
        _CYCLES[key] = int(m.group(1)) if m else None
    return _CYCLES[key]


def run_once(name: str, n: int, arm: str, shard_size: int, mode: str,
             extra_env: dict | None = None, timeout: int = 1800) -> dict:
    env = dict(os.environ)
    env["SHARD_SIZE"] = str(shard_size)
    if arm == "cpu":
        env["SP1_METAL_MIN_ENTRIES"] = "999999999"
        for g in GATES:
            env[g] = "0"
    else:
        env["SP1_METAL_MIN_ENTRIES"] = "16384"
        for g in GATES:
            env[g] = "1"
    if extra_env:
        env.update(extra_env)

    cmd = ["/usr/bin/time", "-l", "nice", "-n", "5", str(HOST),
           "--elf", str(elf_path(name)), "--stdin-u32", str(n), "--mode", mode]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        out = p.stdout + p.stderr
        rc = p.returncode
    except subprocess.TimeoutExpired:
        return {"verified": False, "error": "timeout", "elapsed_s": time.time() - t0}

    row = parse(out)
    row["returncode"] = rc
    row["elapsed_s"] = round(time.time() - t0, 3)
    m = re.search(r"([0-9]+)\s+maximum resident set size", out)
    if m:
        row["peak_rss_gb"] = round(int(m.group(1)) / 1e9, 2)
    if not row.get("verified"):
        row["tail"] = out[-1500:]
    return row


_WARMED: set = set()


def pair(name: str, size_label: str, n: int, shard_size: int, mode: str, repeats: int,
         extra_env: dict | None, config: str, out_fh, force: bool,
         arms: tuple = ("cpu", "gpu")) -> None:
    """One measured point: `repeats` runs of each arm, written as they finish."""
    cyc = cycles(name, n)
    # A guest-level override is part of the configuration, not a hidden default.
    g_env = GUESTS[name].get("env") or {}
    if g_env:
        extra_env = {**g_env, **(extra_env or {})}
        config = f"{config}+{','.join(k.replace('SP1_METAL_', '') + '=' + v for k, v in g_env.items())}"
    # Any configuration that changes generated shader source pays a full
    # recompile on its first run -- one such run took 108 s against 11 -- so a
    # warm-up is run and discarded before the GPU arm is timed.
    warm_key = (name, config, mode)
    if "gpu" in arms and warm_key not in _WARMED:
        wait_until_quiet(force)
        w = run_once(name, n, "gpu", shard_size, mode, extra_env,
                     timeout=GUESTS[name].get("timeout", 1800))
        w.update({"guest": name, "size": size_label, "n": n, "config": config,
                  "mode": mode, "arm": "gpu", "warmup": True, "cycles": cyc,
                  "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        out_fh.write(json.dumps(w) + "\n")
        out_fh.flush()
        _WARMED.add(warm_key)
        print(f"  warm {name:<11} {size_label:<9} {config:<22} discarded "
              f"{w.get('wall_s')}", flush=True)
    for rep in range(repeats):
        for arm in arms:
            waited = wait_until_quiet(force)
            idle_before = cpu_idle_pct()
            r = run_once(name, n, arm, shard_size, mode, extra_env,
                         timeout=GUESTS[name].get("timeout", 1800))
            r["idle_before_pct"] = round(idle_before, 1)
            r["load_before"] = round(load_average(), 2)
            r.update({
                "guest": name, "tier": GUESTS[name]["tier"], "stresses": GUESTS[name]["stresses"],
                "size": size_label, "n": n, "shard_size": shard_size, "mode": mode,
                "arm": arm, "repeat": rep, "config": config, "cycles": cyc,
                "settle_wait_s": waited,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            out_fh.write(json.dumps(r) + "\n")
            out_fh.flush()
            status = "ok " if r.get("verified") else "FAIL"
            print(f"  {status} {name:<11} {size_label:<9} {config:<22} {arm} r{rep} "
                  f"{r.get('wall_s') or r.get('error')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=["smoke", "standard", "full"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not HOST.exists():
        raise SystemExit(f"host not built: {HOST}")

    out_path = pathlib.Path(args.out) if args.out else ROOT / f"results/bench-{args.tier}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repeats = args.repeats if args.repeats is not None else (1 if args.tier == "smoke" else 3)

    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    header = {"kind": "meta", "tier": args.tier, "git": sha, "repeats": repeats,
              "host": subprocess.run(["uname", "-mrs"], capture_output=True, text=True).stdout.strip(),
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    print(json.dumps(header), flush=True)

    with out_path.open("a") as fh:
        fh.write(json.dumps(header) + "\n")
        run_tier(args.tier, repeats, fh, args.force)
    print(f"\nwrote {out_path}", flush=True)


def run_tier(tier: str, repeats: int, fh, force: bool) -> None:
    if tier == "smoke":
        names = ["fib", "sha256", "secp256k1"]
        for nm in names:
            g = GUESTS[nm]
            pair(nm, "standard", g["sizes"]["standard"], 4 << 20, "core", repeats, None, "baseline", fh, force)
        return

    # standard: every Tier A and B guest, three sizes, core mode.
    names = [n for n, g in GUESTS.items() if g["tier"] in ("A", "B")]
    for nm in names:
        for label in ("small", "standard", "large"):
            g = GUESTS[nm]
            pair(nm, label, g["sizes"][label], 4 << 20, "core", repeats, None, "baseline", fh, force)
    if tier == "standard":
        return

    # full: adds Tier C, the sweeps, the ablation and compressed mode.
    for nm in [n for n, g in GUESTS.items() if g["tier"] == "C"]:
        for label in ("small", "standard"):
            pair(nm, label, GUESTS[nm]["sizes"][label], 4 << 20, "core", repeats, None, "baseline", fh, force)

    for ss in (1 << 20, 2 << 20, 8 << 20, 16 << 20):
        for nm in ("fib", "sha256"):
            pair(nm, "standard", GUESTS[nm]["sizes"]["standard"], ss, "core", repeats, None,
                 f"shard={ss >> 20}MiB", fh, force)

    for permits in (1, 2, 8):
        for nm in ("fib", "sha256"):
            pair(nm, "standard", GUESTS[nm]["sizes"]["standard"], 4 << 20, "core", repeats,
                 {"SP1_METAL_PERMITS": str(permits)}, f"permits={permits}", fh, force)

    # The node bound raised, so the chips that fall back compile instead.
    for nm in ("secp256k1", "bn254"):
        pair(nm, "standard", GUESTS[nm]["sizes"]["standard"], 4 << 20, "core", repeats,
             {"SP1_METAL_ZEROCHECK_MAX_NODES": "1000000"}, "nodebound=1e6", fh, force)

    # Phase ablation: everything on, one phase off at a time.
    for gate in ("SP1_METAL_MERKLE", "SP1_METAL_ZEROCHECK", "SP1_METAL_TRANSITION",
                 "SP1_METAL_ENCODE", "SP1_METAL_JAGGED", "SP1_METAL_FIRST_LAYER"):
        pair("fib", "standard", GUESTS["fib"]["sizes"]["standard"], 4 << 20, "core", repeats,
             {gate: "0"}, f"ablate={gate.replace('SP1_METAL_', '')}", fh, force, arms=("gpu",))

    for nm in ("fib", "sha256", "secp256k1"):
        pair(nm, "small", GUESTS[nm]["sizes"]["small"], 4 << 20, "compressed", repeats, None,
             "compressed", fh, force)


# The registry, at module scope so it can be inspected without running anything.
# Sizes come from the calibration recorded in knowledge/benchmark-plan.md: a
# guest's standard size lands near the four shards of the acceptance point.
guest("fib", "acceptance-guest", 250_000, 2_000_000, 4_000_000, "A", "ALU add/mul")
guest("mem", "acceptance-mem", 50_000, 400_000, 800_000, "A", "memory chips")
guest("bits", "acceptance-bits", 125_000, 1_000_000, 2_000_000, "A", "bitwise and shift")
guest("branch", "bench-branch", 170_000, 1_358_569, 2_700_000, "A", "control flow")
guest("divrem", "bench-divrem", 150_000, 1_199_680, 2_400_000, "A", "division")
guest("mulh", "bench-mulh", 375_000, 2_999_198, 6_000_000, "A", "widening multiply")
guest("sha256", "bench-sha256", 5_000, 20_000, 40_000, "B", "ShaExtend/ShaCompress")
guest("keccak", "bench-keccak", 500, 2_000, 4_000, "B", "KeccakPermute",
      env={"SP1_METAL_ZEROCHECK_MAX_NODES": "8000"}, timeout=900)
guest("secp256k1", "bench-secp256k1", 250, 500, 1_000, "B", "Weierstrass/Fp", timeout=900)
guest("bn254", "bench-bn254", 250, 500, 1_000, "B", "Bn254 add/double", timeout=900)
guest("uint256", "bench-uint256", 250, 500, 1_000, "B", "Uint256Mul", timeout=900)
guest("sort", "bench-sort", 15_000, 123_941, 250_000, "C", "memory and compare")
guest("matmul", "bench-matmul", 57, 114, 160, "C", "dense regular arithmetic")
guest("merkle", "bench-merkle", 1_024, 4_096, 8_192, "C", "sha256 + addressing")


if __name__ == "__main__":
    main()
