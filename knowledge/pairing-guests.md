# Whether to build pairing-based guests, and the one test that decides it

> **Queued, 2026-08-27.** Not started: the machine is running the full tier and this needs a build. Everything needed to carry it out is here.

Two statements the Ethereum community would care about most — verifying a Groth16 proof, and verifying a sync committee's BLS signature — both end in a pairing. `guests/eth-groth16-inputs` and `guests/eth-bls-aggregate` measure the parts that *scale* and deliberately stop short of it.

## The decision rule, which is the important part

**If the pairing cannot be measured through SP1's precompiles, it is not measured at all.** A pairing computed in plain RISC-V is a real number and the wrong one: nobody deploying this would run unpatched crates, so quoting it at those communities would misrepresent what they would get. Unrealistic numbers are worse than no numbers.

So there is no fallback. The test below either passes and we build the guests, or it fails and we do not.

## The test

Patched crates that route pairing field arithmetic through the precompiles exist, but the newest tags are pinned to **SP1 6.2.0** and this project is on **6.4.0**:

| crate | tag |
|---|---|
| `substrate-bn` (BN254) | `patch-0.6.0-sp1-6.2.0-substrate-bn`, from `github.com/sp1-patches/bn` |
| `bls12_381` | `patch-0.8.0-sp1-6.2.0`, from `github.com/sp1-patches/bls12_381` |

In a scratch guest, not in the suite:

```toml
[dependencies]
sp1-zkvm = "=6.4.0"
substrate-bn = { version = "0.6.0", default-features = false }

[patch.crates-io]
substrate-bn = { git = "https://github.com/sp1-patches/bn", tag = "patch-0.6.0-sp1-6.2.0-substrate-bn" }
```

Then `cargo prove build`. Three outcomes:

1. **Does not resolve** — most likely a conflicting `sp1-lib` requirement. Route unavailable; stop.
2. **Builds, self-test fails** — the syscall ABI moved between 6.2.0 and 6.4.0. Route unavailable; stop.
3. **Builds and the self-test passes** — build both guests properly.

## The self-test, which is what makes outcome 2 loud rather than silent

A wrong precompile result cannot survive these:

- **Bilinearity**: `e(aP, bQ) == e(abP, Q)`, for a couple of small scalars.
- **Non-degeneracy**: `e(G1, G2) != 1`. Bilinearity alone is satisfied by the map that is always 1, so both are needed.
- **A real vector**: a genuine proof and verifying key that must verify, and a tampered proof that must not. `ark-groth16` can generate the pair host-side; embed the bytes.

The first two are cheap and catch an ABI mismatch immediately. The third is what proves the implementation is Groth16 and not merely something bilinear.

## If it passes

The two existing guests keep their names and their scope — what grows with the committee, and what grows with the statement — and two new ones measure a whole verification. Both numbers are worth having: one says what a bigger committee costs, the other what a verification costs.

## Why the existing two are named as they are

`eth-groth16-inputs` and `eth-bls-aggregate`, not `eth-groth16` and `eth-bls`. A results table prints the guest's name and not its documentation, and a row reading `eth-groth16` would be taken for the cost of verifying a proof.
