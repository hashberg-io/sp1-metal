#![no_main]
//! Ethereum's alt_bn128 pairing check — precompile `0x08`, EIP-197 — in the
//! four-pair shape a Groth16 verifier calls it with.
//!
//! Every Groth16 proof verified on Ethereum ends in one call to this
//! precompile, and that call is the expensive half of the verification. The
//! other half, the multiplication of the verifying key's input points by the
//! public inputs, is `eth-groth16-inputs`; between them the two guests cover a
//! whole on-chain Groth16 verification, and they are kept apart because their
//! costs scale with different things — the pairing with nothing, the input
//! multiplication with the number of public inputs.
//!
//! Each round checks `e(-A, B) · e(α, β) · e(vk_x, γ) · e(C, δ) = 1`, which is
//! Groth16's equation written the way EIP-197 wants it. The instance is genuine
//! rather than a mock: the scalars behind the eight points are chosen freely
//! except for `C`, which is solved for so that the exponents cancel. Every
//! point really is on the curve and the check really does hold, so a precompile
//! that computed the wrong thing would be caught by the guest's own assertion
//! rather than producing a fast proof of nothing.
//!
//! **The pairing is on SP1's precompiles, through the patched `substrate-bn`.**
//! That is the only way this number is worth quoting: the same probe costs 3.59
//! million cycles patched and 64.6 million unpatched, and nobody deploying a
//! verifier would ship the unpatched arithmetic. The patch's newest tag is
//! pinned to SP1 6.2.0 while this project is on 6.4.0, so the self-test below
//! checks bilinearity and non-degeneracy before anything is measured; if the
//! syscall ABI had moved between those versions, it would fail here.

sp1_zkvm::entrypoint!(main);
use substrate_bn::{pairing, pairing_batch, Fr, Group, Gt, G1, G2};

/// A field element from a sixty-four bit seed, by doubling and adding. The
/// crate offers no conversion from an integer that works without `std`.
fn fr_from(seed: u64) -> Fr {
    let mut acc = Fr::zero();
    let mut base = Fr::one();
    for i in 0..64 {
        if (seed >> i) & 1 == 1 {
            acc = acc + base;
        }
        base = base + base;
    }
    acc
}

/// The eight points of a Groth16 verification, and the scalars behind them.
///
/// `c` is not free: the check holds exactly when the exponents cancel, so
/// `c = (a·b − α·β − x·γ) / δ`. Everything else is chosen.
struct Instance {
    neg_a: G1,
    b: G2,
    alpha: G1,
    beta: G2,
    vk_x: G1,
    gamma: G2,
    c: G1,
    delta: G2,
}

struct Setup {
    a: Fr,
    b: Fr,
    alpha_beta: Fr,
    x_gamma: Fr,
    delta_inv: Fr,
    beta_g2: G2,
    gamma_g2: G2,
    delta_g2: G2,
    alpha_g1: G1,
    vk_x_g1: G1,
    b_g2: G2,
}

fn setup(seed: u64) -> Setup {
    let a = fr_from(seed | 1);
    let b = fr_from(seed.wrapping_mul(3) | 1);
    let alpha = fr_from(seed.wrapping_mul(5) | 1);
    let beta = fr_from(seed.wrapping_mul(7) | 1);
    let x = fr_from(seed.wrapping_mul(11) | 1);
    let gamma = fr_from(seed.wrapping_mul(13) | 1);
    let delta = fr_from(seed.wrapping_mul(17) | 1);
    Setup {
        a,
        b,
        alpha_beta: alpha * beta,
        x_gamma: x * gamma,
        delta_inv: delta.inverse().expect("delta is invertible"),
        beta_g2: G2::one() * beta,
        gamma_g2: G2::one() * gamma,
        delta_g2: G2::one() * delta,
        alpha_g1: G1::one() * alpha,
        vk_x_g1: G1::one() * x,
        b_g2: G2::one() * b,
    }
}

/// The instance for round `round`, which differs from the one before it in `A`
/// and therefore in `C`.
///
/// Varying the instance costs one group addition and one multiplication against
/// the four Miller loops that follow, and it is what stops the compiler lifting
/// a loop-invariant pairing out of the measurement.
fn instance(s: &Setup, round: u64) -> Instance {
    let a = s.a + fr_from(round);
    let c = (a * s.b - s.alpha_beta - s.x_gamma) * s.delta_inv;
    Instance {
        neg_a: -(G1::one() * a),
        b: s.b_g2,
        alpha: s.alpha_g1,
        beta: s.beta_g2,
        vk_x: s.vk_x_g1,
        gamma: s.gamma_g2,
        c: G1::one() * c,
        delta: s.delta_g2,
    }
}

/// The precompile's own question: does the product of these pairings equal one?
fn pairing_check(i: &Instance) -> bool {
    pairing_batch(&[
        (i.neg_a, i.b),
        (i.alpha, i.beta),
        (i.vk_x, i.gamma),
        (i.c, i.delta),
    ]) == Gt::one()
}

/// What makes an ABI drift between SP1 6.2.0 and 6.4.0 loud.
///
/// Bilinearity alone would be satisfied by the map that is always one, so
/// non-degeneracy is checked too; and a four-pair instance is verified both
/// intact and tampered with, because a check that accepts everything would pass
/// the first two and still be useless.
fn self_test() {
    let a = fr_from(7);
    let b = fr_from(11);
    assert!(
        pairing(G1::one() * a, G2::one() * b) == pairing(G1::one() * (a * b), G2::one()),
        "bilinearity"
    );
    assert!(pairing(G1::one(), G2::one()) != Gt::one(), "non-degeneracy");
    let s = setup(0x5150_1234_9ABC_DEF1);
    let good = instance(&s, 0);
    assert!(pairing_check(&good), "a sound instance must verify");
    let tampered = Instance {
        c: good.c + G1::one(),
        ..good
    };
    assert!(!pairing_check(&tampered), "a tampered instance must not verify");
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    self_test();

    let s = setup(0x0123_4567_89AB_CDEF);
    let mut verified = 0u32;
    for round in 0..n {
        assert!(pairing_check(&instance(&s, u64::from(round))), "round {round}");
        verified += 1;
    }
    sp1_zkvm::io::commit(&verified);
}
