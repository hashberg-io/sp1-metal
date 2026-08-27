#![no_main]
//! **Named for what it measures, not for signature verification.** A results
//! table shows a guest's name and not its documentation, and `eth-bls` beside a
//! number would be read as the cost of verifying a committee's signature.
//!
//! Aggregating BLS12-381 public keys over a participation bitfield — the part
//! of Ethereum sync-committee verification that grows with the committee, and
//! the reason a light-client proof costs what it does.
//!
//! A sync-committee proof says: these validators of the five hundred and twelve
//! signed, their aggregate key is this, and this signature verifies against it.
//! The last step is a single pairing, a fixed cost whatever the committee size.
//! Everything before it is point arithmetic over G1 that scales with the number
//! of participants, and that is what this measures: `n` committees, each
//! aggregating the keys named by a bitfield and checking the result.
//!
//! **The pairing is not here, and the guest does not pretend otherwise.**
//! SP1 6.4.0 gives a guest BLS12-381 point operations and Fp/Fp2 arithmetic —
//! `syscall_bls12381_add`, `_double`, `_fp2_mulmod` and the rest — but no G2
//! type and no pairing, so the Miller loop and the final exponentiation would
//! have to be written on top of the Fp2 precompiles. That is a much larger
//! piece of work and a separate question; this guest measures the part that
//! scales, and the fixed part is named rather than folded in.
//!
//! Keys are derived from the index, which stands in for a committee's keys
//! being read from a beacon state. Aggregation is what it costs to add them up
//! whatever they are.

sp1_zkvm::entrypoint!(main);
use sp1_lib::bls12381::Bls12381Point;
use sp1_lib::utils::{AffinePoint, WeierstrassAffinePoint};
use sp1_zkvm::syscalls::syscall_keccak_permute;

/// Ethereum's sync committee is this many validators.
const COMMITTEE: usize = 512;

/// Keccak-256, used only to derive scalars deterministically. Ethereum's own
/// key derivation is not what is being measured, and using a hash the suite
/// already exercises keeps this guest about the curve.
fn keccak256(msg: &[u8]) -> [u8; 32] {
    let mut state = [0u64; 25];
    let mut block = [0u8; 136];
    let mut i = 0;
    while msg.len() - i >= 136 {
        block.copy_from_slice(&msg[i..i + 136]);
        absorb(&mut state, &block);
        i += 136;
    }
    let rest = msg.len() - i;
    block = [0u8; 136];
    block[..rest].copy_from_slice(&msg[i..]);
    block[rest] |= 0x01;
    block[135] |= 0x80;
    absorb(&mut state, &block);
    let mut out = [0u8; 32];
    for (j, lane) in state.iter().take(4).enumerate() {
        out[8 * j..8 * j + 8].copy_from_slice(&lane.to_le_bytes());
    }
    out
}

fn absorb(state: &mut [u64; 25], block: &[u8; 136]) {
    for (j, lane) in state.iter_mut().take(17).enumerate() {
        let mut w = [0u8; 8];
        w.copy_from_slice(&block[8 * j..8 * j + 8]);
        *lane ^= u64::from_le_bytes(w);
    }
    unsafe {
        syscall_keccak_permute(state);
    }
}

fn generator() -> Bls12381Point {
    Bls12381Point::new(<Bls12381Point as AffinePoint<12>>::GENERATOR)
}

/// A small scalar derived from a seed. Small on purpose: a validator's key is
/// `sk * G` for a full-width secret, but what this guest needs is a spread of
/// distinct points, and paying a full-width scalar multiplication per key would
/// measure key *generation* rather than aggregation.
fn small_scalar(seed: &[u8]) -> [u64; 4] {
    let h = keccak256(seed);
    let mut out = [0u64; 4];
    let mut w = [0u8; 8];
    w.copy_from_slice(&h[0..8]);
    // Sixteen bits keeps the derivation cheap against the aggregation it feeds.
    out[0] = (u64::from_le_bytes(w) & 0xFFFF) | 1;
    out
}

/// One committee's five hundred and twelve keys, each a distinct point of G1.
///
/// Every key after the first is the one before it plus the generator, rather
/// than its own scalar multiplication. That is deliberate: multiplying five
/// hundred and twelve times costs more than all the aggregation this guest then
/// measures, so the figure would be a key-derivation figure wearing an
/// aggregation label. A beacon client does not derive these at all — it reads
/// them out of the beacon state — so the cheapest construction that yields
/// distinct valid points is the honest one.
fn committee() -> [Bls12381Point; COMMITTEE] {
    let mut current = generator();
    current.mul_assign(&small_scalar(b"committee"));
    let step = generator();
    core::array::from_fn(|_| {
        let out = current;
        current.complete_add_assign(&step);
        out
    })
}

/// Aggregates the keys the bitfield names, as a light client does before it
/// checks the committee's signature.
fn aggregate(keys: &[Bls12381Point], bits: &[u8]) -> (Bls12381Point, u32) {
    let mut agg: Option<Bls12381Point> = None;
    let mut participants = 0u32;
    for (i, k) in keys.iter().enumerate() {
        if bits[i / 8] >> (i % 8) & 1 == 0 {
            continue;
        }
        participants += 1;
        match agg.as_mut() {
            // `complete_add_assign` and not `add_assign`: two equal keys, or a
            // key and its negation, are the cases a bitfield can produce and a
            // plain addition cannot handle.
            Some(a) => a.complete_add_assign(k),
            None => agg = Some(*k),
        }
    }
    (agg.unwrap_or_else(Bls12381Point::infinity), participants)
}

fn self_test() {
    assert_eq!(
        keccak256(b""),
        [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2,
            0xdc, 0xc7, 0x03, 0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
            0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
        ],
        "keccak256(\"\")"
    );
    // Aggregation is a group homomorphism, and this is the identity that says
    // so: adding a*G to b*G gives the same point as (a+b)*G. If the curve
    // operations are wrong, this fails before anything is measured.
    let mut left = generator();
    left.mul_assign(&[3, 0, 0, 0]);
    let mut right = generator();
    right.mul_assign(&[5, 0, 0, 0]);
    left.complete_add_assign(&right);
    let mut eight = generator();
    eight.mul_assign(&[8, 0, 0, 0]);
    assert_eq!(left.limbs_ref(), eight.limbs_ref(), "3G + 5G == 8G");
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    self_test();

    // One committee's keys, built once as a beacon client would hold them.
    let keys = committee();

    let mut total = 0u32;
    let mut acc = 0u64;
    for round in 0..n {
        // A participation bitfield, as it arrives in a light-client update.
        let mut bits = [0u8; COMMITTEE / 8];
        for (j, b) in bits.iter_mut().enumerate() {
            *b = keccak256(&[round as u8, (round >> 8) as u8, j as u8])[0];
        }
        let (agg, participants) = aggregate(&keys, &bits);
        assert!(participants > 0, "a committee with no participants");
        assert!(!agg.is_infinity(), "the aggregate key is the identity");
        total += participants;
        acc ^= agg.limbs_ref()[0];
    }
    sp1_zkvm::io::commit(&total);
    sp1_zkvm::io::commit(&(acc as u32));
}
