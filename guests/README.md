# Acceptance guests

The three guest programs every measurement in this repository is taken against.
Each is a few lines of SP1 guest code with one dependency, chosen so that the three between them load different parts of the machine: a shard's speedup depends on its chip mix, and a single workload would not show that.

| directory | crate and ELF | shape of the work | cycles per iteration |
|---|---|---|---|
| `fib/` | `acceptance-guest` | modular Fibonacci — arithmetic chips | 9 |
| `mem/` | `acceptance-mem` | pseudorandom scatter and gather over a resident array — memory chips | 21.7 |
| `bits/` | `acceptance-bits` | xorshift and rotates — bitwise and shift chips | 14 |

The directory names are the workloads; the crate names are not, because renaming a crate means rebuilding its ELF and the binaries below could not be regenerated when they were committed.

## The binaries ship with the sources

`<guest>/elf/acceptance-*` is the built RISC-V ELF, committed so that a clone can reproduce the acceptance ladder without installing the SP1 guest toolchain.

Built with `cargo prove build` against `sp1-zkvm =6.4.0`, the same version the host side pins throughout.

```
0424b02596b9e3f5a373853d10ec6cb848bd493db7bdc4e82b094dd0063f9f4b  fib/elf/acceptance-guest
2e84bb2c401365c6cecfc1a891854c85171253745ff9bfec36c349b929ea9f4f  mem/elf/acceptance-mem
dd680d27fd66fe476ed1825f6f49e2a0ee38dd1235e37d51d1515f88d2a8d590  bits/elf/acceptance-bits
```

A committed binary is a thing to be suspicious of, so the sources are here to be read and rebuilt against.
With [`sp1up`](https://docs.succinct.xyz) installed:

```sh
cd guests/fib && cargo prove build
```

`tools/scaling_study.sh` prefers a freshly built ELF under `target/` over the one in `elf/`, so rebuilding is how you check the shipped one rather than a separate procedure.
`cargo prove build` is not reproducible byte for byte across toolchain versions, so expect the hashes to differ; what should not differ is the measurement.

## Running them

The ladder runs all three:

```sh
tools/scaling_study.sh                 # defaults to this directory
```

A single pair, which is what `tools/quickstart.sh` does:

```sh
crates/sp1-profile-host/target/release/sp1-profile-host \
  --elf guests/fib/elf/acceptance-guest --stdin-u32 2000000 --mode core
```

The argument each guest reads from stdin is its iteration count, which is how a workload is made larger without changing its shape.
