#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# One pair of proofs, to see whether the claim holds on this machine.
#
# Builds the profiling host, proves the same workload twice through the same
# binary -- once with the Metal path on and once with every switch forced off
# -- and prints both times and the ratio. Both arms through one binary is the
# point: it is the only comparison in which nothing but the path under test
# differs.
#
# Usage: tools/quickstart.sh [n] [--force]
# where n is the guest's iteration count, default 2000000, which is the size
# the recorded acceptance figures use.
set -u

N=2000000
FORCE=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    ''|*[!0-9]*) echo "usage: tools/quickstart.sh [n] [--force]" >&2; exit 2 ;;
    *) N=$a ;;
  esac
done

cd "$(dirname "$0")/.."
HOST=crates/sp1-profile-host/target/release/sp1-profile-host
ELF=guests/fib/elf/acceptance-guest
SHARD=4194304

if [ "$(uname -m)" != arm64 ] || [ "$(uname -s)" != Darwin ]; then
  echo "This needs an Apple Silicon Mac; found $(uname -s)/$(uname -m)." >&2
  exit 1
fi

# A timed pair on a loaded machine measures the load. This project has one
# recorded measurement that was corrupted exactly that way, so the check is
# here rather than in a comment.
cores=$(sysctl -n hw.ncpu)
load=$(uptime | sed 's/.*load averages*: //' | awk '{print $1}' | tr -d ',')
busy=$(python3 -c "print(1 if $load > $cores * 0.25 else 0)")
if [ "$busy" = 1 ] && [ "$FORCE" = 0 ]; then
  cat >&2 <<MSG
The machine is busy: load average $load across $cores cores.

A pair timed now would measure whatever else is running as much as it measures
the prover. Wait until it is idle, or pass --force if you know better.
MSG
  exit 1
fi

if [ ! -x "$HOST" ]; then
  echo "Building the host. First time is about eight minutes; after that it is cached."
  ( cd crates/sp1-profile-host && nice -n 5 cargo build --release --features metal -j4 ) || exit 1
fi

prove() { # on|off
  local min on
  if [ "$1" = on ]; then min=16384; on=1; else min=999999999; on=0; fi
  SHARD_SIZE=$SHARD SP1_METAL_MIN_ENTRIES=$min SP1_METAL_MERKLE=$on \
    SP1_METAL_TRANSITION=$on SP1_METAL_ZEROCHECK=$on SP1_METAL_UPLOAD_CACHE=$on \
    SP1_METAL_JAGGED=$on SP1_METAL_FIRST_LAYER=$on SP1_METAL_JAGGED_VALUES=$on \
    SP1_METAL_JAGGED_ASSIST=$on SP1_METAL_BATCH=$on SP1_METAL_ENCODE=$on \
    nice -n 5 "$HOST" --elf "$ELF" --stdin-u32 "$N" --mode core 2>&1 |
    grep -oE "proved and verified in [0-9.]+" | grep -oE "[0-9.]+$"
}

echo "Proving fib($N) at a $SHARD-cycle shard size, twice. A couple of minutes."
echo "  CPU arm ..."
cpu=$(prove off)
echo "  GPU arm ..."
gpu=$(prove on)

if [ -z "$cpu" ] || [ -z "$gpu" ]; then
  echo "One of the arms did not report a verified proof. Run the host directly to see why:" >&2
  echo "  $HOST --elf $ELF --stdin-u32 $N --mode core" >&2
  exit 1
fi

python3 - "$cpu" "$gpu" <<'PY'
import sys
cpu, gpu = float(sys.argv[1]), float(sys.argv[2])
print()
print(f"  CPU      {cpu:8.2f} s")
print(f"  Metal    {gpu:8.2f} s")
print(f"  speedup  {cpu/gpu:8.2f}x")
print()
lo, hi = 3.71, 5.63
r = cpu / gpu
if r < lo:
    print(f"  Below the 3.71x-5.63x envelope recorded on an M4 Max. Either this is a")
    print(f"  smaller GPU, or something else was running. results/scaling-quiet.tsv has")
    print(f"  the reference ladder.")
elif r > hi:
    print(f"  Above the recorded envelope, which is good news and worth checking twice.")
else:
    print(f"  Inside the {lo}x-{hi}x envelope recorded on an M4 Max.")
PY
echo "Both proofs were verified, not merely produced."
