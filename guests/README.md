# Benchmark guests

The guest programs every measurement in this repository is taken against.
Each is a small SP1 guest with one dependency, and each reads its size from stdin as a `u32`, so a workload is made larger without changing its shape.

Three tiers, chosen to span what the Metal backend is sensitive to rather than what zkVMs are usually benchmarked on — see [benchmark-plan.md](../knowledge/benchmark-plan.md).

| directory | tier | crate and ELF | what it stresses |
|---|---|---|---|
| `fib/` | A | `acceptance-guest` | modular Fibonacci — ALU add/mul |
| `mem/` | A | `acceptance-mem` | pseudorandom scatter/gather — memory chips |
| `bits/` | A | `acceptance-bits` | xorshift and rotates — bitwise and shift chips |
| `branch/` | A | `bench-branch` | unpredictable branches — the branch chip |
| `divrem/` | A | `bench-divrem` | division and remainder — a DAG unlike add or shift |
| `mulh/` | A | `bench-mulh` | widening 64-bit multiply — MUL's high half |
| `sha256/` | B | `bench-sha256` | ShaExtend + ShaCompress, driven by syscall |
| `keccak/` | B | `bench-keccak` | KeccakPermute — see the warning below |
| `secp256k1/` | B | `bench-secp256k1` | Weierstrass double from the generator |
| `bn254/` | B | `bench-bn254` | Bn254 G1 double from (1, 2) |
| `uint256/` | B | `bench-uint256` | Uint256 mulmod — big integers, no curve |
| `sort/` | C | `bench-sort` | sort n u32s — memory and compare |
| `matmul/` | C | `bench-matmul` | dense d x d u32 multiply — n is the dimension |
| `merkle/` | C | `bench-merkle` | Merkle tree over n leaves, sha256 precompile |

**Tier A** are microbenchmarks with one dominant chip family each.
**Tier B** drive SP1's precompiles directly through `sp1_zkvm::syscalls`, rather than through patched crates, so the chip under test is unambiguous.
**Tier C** are realistic composites.

The elliptic-curve guests use repeated *doubling* rather than addition. Doubling is total on a prime-order curve: every intermediate stays a valid affine point and none reaches infinity, so the guest cannot wander into a precompile precondition it fails to meet.

## A warning about `keccak`

`KeccakPermuteControl` has 8,234 and 12,351 constraint nodes, which is under the default `SP1_METAL_ZEROCHECK_MAX_NODES` of 16,384, so the backend generates and compiles a Metal kernel for it. That compile does not finish in any reasonable time.

Set `SP1_METAL_ZEROCHECK_MAX_NODES=8000` to prove keccak workloads, which takes the wall time from "does not complete" to a couple of seconds. `Global:true` at 15,196 nodes compiles without trouble, so node count is not what predicts this.

## The binaries ship with the sources

`<guest>/elf/<crate>` is the built RISC-V ELF, committed so that a clone can reproduce every measurement without installing the SP1 guest toolchain.

Built with `cargo prove build` from `~/.sp1` (cargo-prove f66b4bf, 2026-08-12) against `sp1-zkvm =6.4.0`.

**The build is reproducible.** Rebuilding `fib` from source on the machine that produced it gives a byte-identical ELF, so these hashes are a real check rather than a formality:

```
0424b02596b9e3f5a373853d10ec6cb848bd493db7bdc4e82b094dd0063f9f4b  fib/elf/acceptance-guest
2e84bb2c401365c6cecfc1a891854c85171253745ff9bfec36c349b929ea9f4f  mem/elf/acceptance-mem
dd680d27fd66fe476ed1825f6f49e2a0ee38dd1235e37d51d1515f88d2a8d590  bits/elf/acceptance-bits
8aeb2e1582e015c42735d2bf15da7b817de5cf68d54f6e79f633337d56605037  branch/elf/bench-branch
9311ad84ed42eeaf5b2f0c0a6ee77f4b64524fbb763da20c5036095c9fb72060  divrem/elf/bench-divrem
8d01976729498508c3538fb328ddec79323b48e430e6e35c5be2c36a01e4fafe  mulh/elf/bench-mulh
e39916583f87b4af17686db34aa7e6a25bf8f4baa1aee18108c67f67033367b2  sha256/elf/bench-sha256
9d3f49a8cfcb1a2d0622c84143c3c4c43944f3240cb6dc2d2b05cdf71fad15ef  keccak/elf/bench-keccak
abd5ed1e2e95687a2f1fd088518e5b88ee40ede6a7b819c6178219a558313ef2  secp256k1/elf/bench-secp256k1
9643eeb9c5966c544a7c33765f2f121de099bbb11692c66eb40bad747660efba  bn254/elf/bench-bn254
dcbd8991057e0c3866453b16fa86979ac2f3ba0514ed7b1458ba370c9bb15d99  uint256/elf/bench-uint256
dc736c66657b5b8b9d3297fd2ad1742639cb9360df59caca3ae0c4eac6b6770b  sort/elf/bench-sort
22234bfb1ca7027800ed1b4ad72a27651782097018c186eb0ca011f0c0acd274  matmul/elf/bench-matmul
804aeff959e521593a41d935bf0825fd8817ae4484a1d22b519120566d1cb46e  merkle/elf/bench-merkle
```

With [`sp1up`](https://docs.succinct.xyz) installed:

```sh
cd guests/sha256 && cargo prove build
```

`tools/scaling_study.sh` and `tools/bench_suite.py` both prefer a freshly built ELF under `target/` over the one in `elf/`, so rebuilding is how you check the shipped one rather than a separate procedure.

## Running them

```sh
tools/quickstart.sh                       # one pair against fib
tools/scaling_study.sh                    # the acceptance ladder
tools/bench_suite.py --tier standard      # the full benchmark and correctness suite
```

Or one proof directly:

```sh
crates/sp1-profile-host/target/release/sp1-profile-host \
  --elf guests/sha256/elf/bench-sha256 --stdin-u32 20000 --mode core
```

`--execute` runs the guest without proving and reports its cycle count, which is the cheap way to size a new workload — though note that for Tier B a precompile syscall is one guest instruction and many trace rows, so cycles understate proving cost by more than an order of magnitude.

## The shipped binary reproduces the published number

`results/scaling-quiet.tsv` records 60.25 s on the CPU against 10.85 s on the GPU at a 4 MiB shard size, a ratio of 5.556x, measured against the fib guest before it was committed here.
Run against the committed ELF on a quiet machine, `tools/quickstart.sh` gives 59.84 s against 10.94 s, a ratio of 5.47x — inside 1.5% on every figure.
