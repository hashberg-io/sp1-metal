#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# One paired measurement of the acceptance workload: the same binary twice,
# with every Metal switch off and then on, and peak resident memory sampled
# from ps rather than taken from `/usr/bin/time -l`, which overstates it.
#
# Usage: tools/paired_run.sh <elf> <n> <shard_size> <out-cpu> <out-gpu>
#
# Both output files are required: the two arms are written separately so that
# a failed arm leaves the other's report intact.
set -u
ELF=${1:?elf}
N=${2:?n}
SHARD=${3:?shard_size}
OUT_CPU=${4:?out-cpu}
OUT_GPU=${5:?out-gpu}
HOST=crates/sp1-profile-host/target/release/sp1-profile-host

run() { # on|off outfile
  local min on
  if [ "$1" = on ]; then min=16384; on=1; else min=999999999; on=0; fi
  SHARD_SIZE=$SHARD SP1_METAL_MIN_ENTRIES=$min SP1_METAL_MERKLE=$on \
    SP1_METAL_TRANSITION=$on SP1_METAL_ZEROCHECK=$on SP1_METAL_UPLOAD_CACHE=$on \
    SP1_METAL_JAGGED=$on SP1_METAL_FIRST_LAYER=$on SP1_METAL_JAGGED_VALUES=$on \
    SP1_METAL_JAGGED_ASSIST=$on SP1_METAL_BATCH=$on SP1_METAL_ENCODE=$on \
    nice -n 5 "$HOST" --elf "$ELF" --stdin-u32 "$N" --mode core > "$2" 2>&1 &
  local pid=$!
  local peak=0 rss
  while kill -0 $pid 2>/dev/null; do
    rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
    [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    sleep 0.2
  done
  wait $pid
  echo "peak_rss_gb $(python3 -c "print(f'{$peak/1048576:.1f}')")" >> "$2"
}

run off "$OUT_CPU"
run on "$OUT_GPU"
