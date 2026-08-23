#!/bin/bash
# The acceptance study: how the Metal path's speedup scales.
#
# Every point is two proofs through the *same* binary, switched by the two
# environment variables that gate the GPU, so nothing but the path under test
# differs between the pair. Three axes, because "a bigger workload" conflates
# two experiments: a bigger shard means bigger GKR layers and fewer of them,
# while more work at a fixed shard size means more shards of the same size.
#
# Usage: tools/scaling_study.sh <guest-root> [out.tsv]
# where <guest-root> holds the built guests (guest/, mem/, bits/), each an
# `cargo prove build` tree.
set -u
GUESTS=${1:?guest root}
OUT=${2:-results/scaling.tsv}
HOST=crates/sp1-profile-host/target/release/sp1-profile-host
elf() { echo "$GUESTS/$1/target/elf-compilation/riscv64im-succinct-zkvm-elf/release/acceptance-$2"; }

# $5 = on|off for the whole Metal path
run() { # guest_dir guest_name n shard_size on|off
  # Every switch together, so the CPU arm really is one: leaving any of them on
  # would understate the pair.
  local min on
  if [ "$5" = on ]; then min=16384; on=1; else min=999999999; on=0; fi
  SHARD_SIZE=$4 SP1_METAL_MIN_ENTRIES=$min SP1_METAL_MERKLE=$on \
    SP1_METAL_TRANSITION=$on SP1_METAL_ZEROCHECK=$on SP1_METAL_UPLOAD_CACHE=$on \
    SP1_METAL_JAGGED=$on SP1_METAL_FIRST_LAYER=$on SP1_METAL_JAGGED_VALUES=$on \
    SP1_METAL_JAGGED_ASSIST=$on SP1_METAL_BATCH=$on SP1_METAL_ENCODE=$on \
    timeout 900 nice -n 5 "$HOST" --elf "$(elf "$1" "$2")" --stdin-u32 "$3" --mode core 2>&1 |
    grep -oE "proved and verified in [0-9.]+" | grep -oE "[0-9.]+$"
}

row() { # axis guest_dir elf_name label n shard_size
  local c g
  c=$(run "$2" "$3" "$5" "$6" off)
  g=$(run "$2" "$3" "$5" "$6" on)
  if [ -n "$c" ] && [ -n "$g" ]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$4" "$5" "$6" "$c" "$g" \
      "$(python3 -c "print(f'{$c/$g:.3f}')")"
  else
    printf '%s\t%s\t%s\t%s\tFAIL\tFAIL\t-\n' "$1" "$4" "$5" "$6"
  fi
}

mkdir -p "$(dirname "$OUT")"
{
  printf 'axis\tguest\tn\tshard_size\tcpu_s\tgpu_s\tspeedup\n'
  for ss in 1048576 2097152 4194304 8388608 16777216; do
    row shard_size guest guest fib 2000000 $ss
  done
  for n in 250000 500000 1000000 2000000; do
    row work guest guest fib $n 4194304
  done
  row guest mem mem mem 400000 4194304
  row guest bits bits bits 1000000 4194304
} | tee "$OUT"
echo STUDYDONE
