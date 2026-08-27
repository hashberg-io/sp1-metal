#![no_main]
//! The public-input half of Groth16 verification on BN254 — the work a
//! verifier does that grows with the statement, and the reason a rollup with
//! many public inputs costs more to verify than one with few.
//!
//! Groth16 verification is two things. A multi-scalar multiplication in G1 that
//! folds the public inputs into a single point, `vk_x = IC[0] + sum(a_i *
//! IC[i])`, whose cost is linear in the number of inputs; and then three
//! pairings, whose cost is fixed. This measures the first, over `n` proofs of
//! `INPUTS` public inputs each, on the same curve Ethereum's `ecPairing`
//! precompile uses.
//!
//! **The three pairings are not here, and the guest does not pretend
//! otherwise.** SP1 6.4.0 gives a guest BN254 point operations and Fp/Fp2
//! arithmetic but no G2 type and no pairing, so a Miller loop and a final
//! exponentiation over the Fp12 tower would have to be built on the Fp2
//! precompiles — several hundred lines resting on Frobenius constants that
//! have to be right to the bit. That is worth doing and worth doing carefully;
//! it is not worth guessing at. Until then this guest measures the part that
//! scales and names the part it leaves out, rather than quietly reporting a
//! fraction of Groth16 as though it were all of it.
//!
//! The verifying key and the inputs are derived from the index. A verifier's
//! cost does not depend on which statement it is checking, only on how many
//! field elements the statement has.

sp1_zkvm::entrypoint!(main);
use sp1_lib::bn254::Bn254Point;
use sp1_lib::utils::{AffinePoint, WeierstrassAffinePoint};
use sp1_zkvm::syscalls::syscall_keccak_permute;

/// Public inputs per proof. A rollup's block commitment is of this order:
/// a few roots, a few counters.
const INPUTS: usize = 8;

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

fn generator() -> Bn254Point {
    Bn254Point::new(<Bn254Point as AffinePoint<8>>::GENERATOR)
}

/// A full-width scalar, which is what a public input is: an element of the
/// scalar field, arbitrary.
fn scalar(seed: &[u8]) -> [u64; 4] {
    let h = keccak256(seed);
    let mut out = [0u64; 4];
    for i in 0..4 {
        let mut w = [0u8; 8];
        w.copy_from_slice(&h[8 * i..8 * i + 8]);
        out[i] = u64::from_le_bytes(w);
    }
    // Below the scalar field's order by a comfortable margin, and never zero.
    out[3] &= 0x0FFF_FFFF_FFFF_FFFF;
    if out.iter().all(|l| *l == 0) {
        out[0] = 1;
    }
    out
}

/// A point of the verifying key's input basis. In a real key these are fixed by
/// the trusted setup; what matters to a verifier is that they are arbitrary
/// points it did not choose.
fn ic_point(index: usize) -> Bn254Point {
    let mut p = generator();
    let mut s = scalar(&[b'i', index as u8]);
    // A small multiple keeps building the key cheap against the folding that
    // is being measured.
    s = [s[0] & 0xFFFF | 1, 0, 0, 0];
    p.mul_assign(&s);
    p
}

/// `vk_x = IC[0] + sum(input_i * IC[i + 1])`, the linear combination a Groth16
/// verifier folds its public inputs into.
fn fold_inputs(ic: &[Bn254Point], inputs: &[[u64; 4]; INPUTS]) -> Bn254Point {
    let mut acc = ic[0];
    for (i, input) in inputs.iter().enumerate() {
        let mut term = ic[i + 1];
        term.mul_assign(input);
        acc.complete_add_assign(&term);
    }
    acc
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
    // The folding is linear, and this says so: folding `[1, 0, 0, ...]` against
    // a basis gives `IC[0] + IC[1]`. If the curve operations are wrong it fails
    // before anything is measured.
    let ic: [Bn254Point; INPUTS + 1] = core::array::from_fn(ic_point);
    let mut inputs = [[0u64; 4]; INPUTS];
    inputs[0] = [1, 0, 0, 0];
    let folded = fold_inputs(&ic, &inputs);
    let mut expected = ic[0];
    expected.complete_add_assign(&ic[1]);
    assert_eq!(folded.limbs_ref(), expected.limbs_ref(), "the fold is linear");
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    self_test();

    // The verifying key's input basis, built once as a verifier holds it.
    let ic: [Bn254Point; INPUTS + 1] = core::array::from_fn(ic_point);

    let mut acc = 0u64;
    for proof in 0..n {
        let inputs: [[u64; 4]; INPUTS] =
            core::array::from_fn(|i| scalar(&[proof as u8, (proof >> 8) as u8, i as u8]));
        let vk_x = fold_inputs(&ic, &inputs);
        assert!(!vk_x.is_infinity(), "the folded input point is the identity");
        acc ^= vk_x.limbs_ref()[0];
    }
    sp1_zkvm::io::commit(&(acc as u32));
}
