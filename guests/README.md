# Benchmark guests

The guest programs every measurement in this repository is taken against.
Each reads its size from stdin as a `u32`, so a workload is made larger without changing its shape, and twenty of the twenty-two depend on no crate outside SP1's own — sixteen on `sp1-zkvm` alone and four on `sp1-lib` beside it.

Four tiers. The first three span what the Metal backend is sensitive to rather than what zkVMs are usually benchmarked on — see [benchmark-plan.md](../knowledge/benchmark-plan.md). The fourth measures whole protocol operations that the Bitcoin and Ethereum communities actually pay for.

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
| `btc-headers/` | D | `bench-btc-headers` | a Bitcoin header chain, mined and verified |
| `eth-mpt/` | D | `bench-eth-mpt` | Merkle-Patricia proofs — keccak over trie nodes |
| `eth-ecrecover/` | D | `bench-eth-ecrecover` | ECDSA sign and recover, and the address |
| `btc-schnorr/` | D | `bench-btc-schnorr` | BIP-340 sign and verify |
| `eth-groth16-inputs/` | D | `bench-eth-groth16-inputs` | the public-input half of a Groth16 verification |
| `eth-pairing-check/` | D | `bench-eth-pairing-check` | the pairing half — EIP-197, four pairs |
| `eth-bls-aggregate/` | D | `bench-eth-bls-aggregate` | summing a sync committee's public keys |
| `eth-bls-verify/` | D | `bench-eth-bls-verify` | a whole sync-committee signature verification |

**Tier A** are microbenchmarks with one dominant chip family each.
**Tier B** drive SP1's precompiles directly through `sp1_zkvm::syscalls`, rather than through patched crates, so the chip under test is unambiguous.
**Tier C** are realistic composites.
**Tier D** are whole statements from Bitcoin and Ethereum, and each one self-tests before it measures: published vectors where they exist — `sha256("abc")`, the real mainnet genesis block hash, `keccak256("")` — and algebraic identities where they do not, always including something that must *fail*, because a check that accepts everything passes every check that only accepts.

Tier D's sizes are calibrated against each other's wall clock rather than against the acceptance point's shard count. Four of the eight sit at five or six shards for every input that finishes in a sane time, because their widest chip fixes the count and `n` cannot move it.

The elliptic-curve guests use repeated *doubling* rather than addition. Doubling is total on a prime-order curve: every intermediate stays a valid affine point and none reaches infinity, so the guest cannot wander into a precompile precondition it fails to meet.

## The two guests that use patched crates, and why

`eth-pairing-check` and `eth-bls-verify` are the only guests here that depend on anything outside SP1's own crates, because they are the only ones that need a pairing and `sp1-lib` does not have one. They take it from the patched `substrate-bn` and `bls12_381`, and `eth-bls-verify` takes SHA-256 from the patched `sha2` as well, since Ethereum's hash-to-curve expands the message with it.

The patches are what make the numbers worth quoting rather than a detail of packaging. Measured against an unpatched control, the same pairing probe costs:

| curve | patched | unpatched |
|---|---|---|
| BN254 | 3.59 M cycles | 64.6 M |
| BLS12-381 | 9.46 M cycles | 92.4 M |

An unpatched pairing is a real number and the wrong one: nobody deploying a verifier would ship it, so quoting it at the communities that care about these statements would misrepresent what they would get.

All three patch tags name SP1 **6.2.0** while this project is on **6.4.0**. They resolve anyway — cargo pulls `sp1-lib` 6.5.0 for them and the syscall ABI has not moved — and each guest's self-test checks bilinearity and non-degeneracy before it measures, so a version drift that broke the ABI would fail loudly rather than produce a fast proof of nothing. `knowledge/pairing-guests.md` records the test that established this.

## Why `keccak` and `divrem` are slower the first time

`KeccakPermuteControl` and `DivRem` each take about ten minutes in the Metal compiler, while `Global:true` at nearly twice their node count compiles instantly — so node count does not predict compile cost and the bound drawn on it never could have caught them. The backend no longer tries: a round waits a second for its kernel, takes the CPU if it is not ready, and installs the kernel whenever it arrives.

The consequence is that these two guests are slower on a machine that has never proved them, and settle once macOS has cached the kernels. The two benchmark matrices in `results/` happen to measure exactly this, having been taken ten hours apart on a cold and a warm cache: `divrem/standard` moves from 3.75x to 5.14x and `keccak/large` from 1.94x to 2.16x, and nothing else in the suite moves at all. `USING.md` explains how to pay that cost once, up front.

The `SP1_METAL_ZEROCHECK_MAX_NODES=8000` workaround these guests used to need is gone, and so is the override the benchmark harness applied to keccak.

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
6ce0cc3569d1b148f34c7479e0ee19dbdbf43af0fbfbacff31ca64fb4e3717f9  btc-headers/elf/bench-btc-headers
a26eafec928ba8862342236ba7c5992f0d69555ef6344f83f43c034703515ea7  eth-mpt/elf/bench-eth-mpt
78be39c54de62d2c2786d2439aeafec93bd788d60313e83a086cf190ea90fdc3  eth-ecrecover/elf/bench-eth-ecrecover
4beada9934edcbb5166086bdf048c8b6fefcb9324e1fcf020c58e296f23a96ec  btc-schnorr/elf/bench-btc-schnorr
18d3590857218b000de833c9a6a9fadd0e72c6df79d392860ff752331244202b  eth-groth16-inputs/elf/bench-eth-groth16-inputs
a59feb7d37feeb877685f219386a930acfea1de7b324eb45ab40339cd2c1b81d  eth-pairing-check/elf/bench-eth-pairing-check
16a067dbdfcf570cb16962d3986ed46c1d75bdcb2f8cc572aabfd27bd7056258  eth-bls-aggregate/elf/bench-eth-bls-aggregate
06c009d258181f6c3067831f4d43cd4b4339af9c5acb688bc52191993621a611  eth-bls-verify/elf/bench-eth-bls-verify
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
tools/bench_suite.py --tier apps          # the Bitcoin and Ethereum statements
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
