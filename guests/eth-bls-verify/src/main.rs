#![no_main]
//! Verifying an Ethereum sync committee's aggregate BLS signature — the whole
//! statement a light client proves, pairing included.
//!
//! A sync-committee update says: these validators of the five hundred and
//! twelve signed this beacon block root, and here is one signature for all of
//! them. Checking it means summing the participating public keys, hashing the
//! message to a point of G2, and asking whether `e(σ, g₁) = e(H(m), pk)`. The
//! first part grows with the committee; the pairing does not. `eth-bls-aggregate`
//! measures the part that grows on its own, because knowing which half of the
//! cost moves with committee size is worth more than one number that hides it.
//!
//! The scheme is Ethereum's: BLS12-381, signatures in G2, keys in G1, and the
//! `BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_` domain separation tag that the
//! beacon chain signs under. The keys are synthetic, which changes nothing
//! about what verifying them costs.
//!
//! Producing the aggregate signature is not part of what is measured, and it is
//! not what a light client does — the signature arrives over the wire. Each
//! round builds it from the sum of the participating secret keys, which yields
//! exactly the point the individual signatures would have summed to, at the
//! price of one multiplication rather than two hundred and fifty-six.
//!
//! **Both the pairing and the hashing are on SP1's precompiles**, through the
//! patched `bls12_381` and `sha2`. That is the only version of this number
//! worth quoting: the same pairing costs 9.46 million cycles patched and 92.4
//! million unpatched. Both patches are pinned to SP1 6.2.0 while this project
//! is on 6.4.0, so the self-test checks bilinearity, non-degeneracy, and a
//! signature that must verify beside one that must not, before anything is
//! measured.

sp1_zkvm::entrypoint!(main);
extern crate alloc;

use alloc::vec::Vec;
use bls12_381::hash_to_curve::{ExpandMsgXmd, HashToCurve};
use bls12_381::{
    multi_miller_loop, pairing, G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Gt,
    Scalar,
};
use sha2::{Digest, Sha256};

/// Ethereum's sync committee is this many validators.
const COMMITTEE: usize = 512;

/// The domain the beacon chain signs sync-committee messages under.
const DST: &[u8] = b"BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_POP_";

/// The committee's secret keys and the public keys they imply.
///
/// The secret keys are consecutive, so each public key is the one before it
/// plus the generator. Five hundred and twelve scalar multiplications would
/// cost more than every verification this guest then measures, and a beacon
/// client derives no keys at all — it reads them out of state — so the cheapest
/// construction that yields distinct valid keys is the honest one.
fn committee() -> (Vec<Scalar>, Vec<G1Affine>) {
    let base = Scalar::from(0x5150_1234_9ABC_DEF1u64);
    let mut secrets = Vec::with_capacity(COMMITTEE);
    let mut keys = Vec::with_capacity(COMMITTEE);
    let mut point = G1Projective::generator() * base;
    let step = G1Projective::generator();
    for i in 0..COMMITTEE {
        secrets.push(base + Scalar::from(i as u64));
        keys.push(G1Affine::from(point));
        point += step;
    }
    (secrets, keys)
}

/// The message a round signs, standing in for a beacon block root.
fn block_root(round: u32) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"beacon-block-root");
    h.update(round.to_le_bytes());
    h.finalize().into()
}

/// A participation bitfield, as it arrives in a light-client update.
fn participation(round: u32) -> [u8; COMMITTEE / 8] {
    let mut bits = [0u8; COMMITTEE / 8];
    for (chunk, slot) in bits.chunks_mut(32).enumerate() {
        let mut h = Sha256::new();
        h.update(b"participation");
        h.update(round.to_le_bytes());
        h.update((chunk as u32).to_le_bytes());
        let digest: [u8; 32] = h.finalize().into();
        slot.copy_from_slice(&digest[..slot.len()]);
    }
    bits
}

fn participates(bits: &[u8], i: usize) -> bool {
    bits[i / 8] >> (i % 8) & 1 == 1
}

/// The aggregate public key of the validators the bitfield names.
fn aggregate_key(keys: &[G1Affine], bits: &[u8]) -> (G1Affine, u32) {
    let mut acc = G1Projective::identity();
    let mut participants = 0u32;
    for (i, k) in keys.iter().enumerate() {
        if participates(bits, i) {
            acc += k;
            participants += 1;
        }
    }
    (G1Affine::from(acc), participants)
}

/// The signature the participating validators would have produced between them.
/// This is the signer's side of the protocol, not the verifier's.
fn aggregate_signature(secrets: &[Scalar], bits: &[u8], message: &G2Projective) -> G2Affine {
    let mut total = Scalar::zero();
    for (i, s) in secrets.iter().enumerate() {
        if participates(bits, i) {
            total += s;
        }
    }
    G2Affine::from(message * total)
}

fn hash_to_g2(message: &[u8]) -> G2Projective {
    <G2Projective as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve([message], DST)
}

/// The verifier's question: is `e(σ, g₁) = e(H(m), pk)`?
///
/// Written as a single two-term Miller loop with one final exponentiation,
/// which is how it is actually computed and about a third cheaper than two
/// separate pairings.
fn verify(aggregate: &G1Affine, signature: &G2Affine, message: &G2Affine) -> bool {
    let terms = [
        (&G1Affine::generator(), &G2Prepared::from(*signature)),
        (&-*aggregate, &G2Prepared::from(*message)),
    ];
    multi_miller_loop(&terms).final_exponentiation() == Gt::identity()
}

/// What makes an ABI drift between SP1 6.2.0 and 6.4.0 loud.
fn self_test(secrets: &[Scalar], keys: &[G1Affine]) {
    let a = Scalar::from(7u64);
    let b = Scalar::from(11u64);
    let ap = G1Affine::from(G1Projective::generator() * a);
    let bq = G2Affine::from(G2Projective::generator() * b);
    let abp = G1Affine::from(G1Projective::generator() * (a * b));
    assert!(
        pairing(&ap, &bq) == pairing(&abp, &G2Affine::generator()),
        "bilinearity"
    );
    assert!(
        pairing(&G1Affine::generator(), &G2Affine::generator()) != Gt::identity(),
        "non-degeneracy"
    );
    // A genuine signature must verify, and the same signature checked against a
    // committee with one member swapped must not. A check that accepted
    // everything would pass the two identities above and still be worthless.
    let bits = participation(0);
    let point = hash_to_g2(&block_root(0));
    let message = G2Affine::from(point);
    let signature = aggregate_signature(secrets, &bits, &point);
    let (aggregate, participants) = aggregate_key(keys, &bits);
    assert!(participants > 0, "a committee with no participants");
    assert!(
        verify(&aggregate, &signature, &message),
        "an honest signature must verify"
    );
    let wrong = G1Affine::from(G1Projective::from(aggregate) + G1Projective::generator());
    assert!(
        !verify(&wrong, &signature, &message),
        "a signature must not verify against the wrong committee"
    );
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    let (secrets, keys) = committee();
    self_test(&secrets, &keys);

    let mut total = 0u32;
    for round in 0..n {
        let bits = participation(round);
        let point = hash_to_g2(&block_root(round));
        let message = G2Affine::from(point);
        // The signer's side, which the verifier receives rather than computes.
        let signature = aggregate_signature(&secrets, &bits, &point);
        // The verifier's side, which is what is measured.
        let (aggregate, participants) = aggregate_key(&keys, &bits);
        assert!(participants > 0, "a committee with no participants");
        assert!(
            verify(&aggregate, &signature, &message),
            "round {round} did not verify"
        );
        total += participants;
    }
    sp1_zkvm::io::commit(&total);
}
