#![no_main]
//! Recovering the signer of secp256k1 signatures — `ecrecover`, which every
//! Ethereum transaction costs once and which a rollup verifying a batch pays
//! for thousands of times.
//!
//! Each iteration signs a message and recovers the public key from the
//! signature, then checks that the key it recovered is the one it signed with
//! and derives the twenty-byte address the way Ethereum does: the low twenty
//! bytes of the Keccak-256 of the uncompressed public key. Signing as well as
//! recovering makes every iteration self-checking, which is worth more here
//! than a pure cost figure: recovery alone is about sixty per cent of the work
//! measured, since it does two scalar multiplications to signing's one.
//!
//! The curve operations are SP1's precompiles, reached through `sp1-lib`'s
//! `AffinePoint`. The scalar field is arithmetic mod the group order, done in
//! software as every zkVM ECDSA implementation does it, on top of the
//! `sys_bigint` modular multiply.
//!
//! Nothing here is secret and nothing needs to be: the keys are derived from
//! the loop index, and the guest is a cost model with a correctness check
//! attached, not a wallet.

sp1_zkvm::entrypoint!(main);
use sp1_lib::secp256k1::Secp256k1Point;
use sp1_lib::utils::{AffinePoint, WeierstrassAffinePoint};
use sp1_lib::sys_bigint;
use sp1_zkvm::syscalls::syscall_keccak_permute;

/// The order of the secp256k1 group, little-endian by 64-bit limb.
const ORDER: [u64; 4] = [
    0xBFD2_5E8C_D036_4141,
    0xBAAE_DCE6_AF48_A03B,
    0xFFFF_FFFF_FFFF_FFFE,
    0xFFFF_FFFF_FFFF_FFFF,
];

/// `a * b` modulo the group order.
///
/// `sys_bigint`'s operation selector is not documented in the crate, so the
/// self-test below multiplies `order - 1` by itself and insists on 1. If the
/// selector is wrong the guest stops there rather than producing a fast and
/// meaningless proof.
fn mulmod(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    unsafe {
        sys_bigint(&mut out, 0, a, b, &ORDER);
    }
    out
}

fn is_zero(a: &[u64; 4]) -> bool {
    a.iter().all(|l| *l == 0)
}

/// Whether `a >= b`, comparing from the most significant limb down.
fn ge(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn sub_in_place(a: &mut [u64; 4], b: &[u64; 4]) {
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        a[i] = d;
        borrow = u64::from(b1 || b2);
    }
}

/// `a + b` modulo the group order.
fn addmod(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(carry);
        out[i] = s;
        carry = u64::from(c1 || c2);
    }
    if carry == 1 || ge(&out, &ORDER) {
        sub_in_place(&mut out, &ORDER);
    }
    out
}

/// The additive inverse modulo the group order.
fn negmod(a: &[u64; 4]) -> [u64; 4] {
    if is_zero(a) {
        return [0u64; 4];
    }
    let mut out = ORDER;
    sub_in_place(&mut out, a);
    out
}

/// The multiplicative inverse modulo the group order, by Fermat: the order is
/// prime, so `a^(order - 2)` inverts. Square and multiply over 256 bits, which
/// is about 380 modular multiplies and each of those is one precompile call.
fn invmod(a: &[u64; 4]) -> [u64; 4] {
    let mut exponent = ORDER;
    sub_in_place(&mut exponent, &[2, 0, 0, 0]);
    let mut result = [1u64, 0, 0, 0];
    let mut base = *a;
    for limb in 0..4 {
        for bit in 0..64 {
            if (exponent[limb] >> bit) & 1 == 1 {
                result = mulmod(&result, &base);
            }
            base = mulmod(&base, &base);
        }
    }
    result
}

/// Reduces a field element -- an x coordinate, which lives modulo the prime --
/// into the scalar field. The prime is less than twice the order, so at most
/// one subtraction is needed.
fn to_scalar(x: &[u64; 4]) -> [u64; 4] {
    let mut out = *x;
    if ge(&out, &ORDER) {
        sub_in_place(&mut out, &ORDER);
    }
    out
}

/// The secp256k1 field prime, little-endian by 64-bit limb. Coordinates live
/// modulo this one; scalars live modulo `ORDER`.
const PRIME: [u64; 4] = [
    0xFFFF_FFFE_FFFF_FC2F,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

/// `(prime + 1) / 4`. The prime is 3 modulo 4, so raising a square to this
/// power is a square root of it.
const SQRT_EXPONENT: [u64; 4] = [
    0xFFFF_FFFF_BFFF_FF0C,
    0xFFFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
    0x3FFF_FFFF_FFFF_FFFF,
];

/// `a * b` modulo the field prime.
fn mulmod_p(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    unsafe {
        sys_bigint(&mut out, 0, a, b, &PRIME);
    }
    out
}

/// `a + b` modulo the field prime.
fn addmod_p(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        let (s, c1) = a[i].overflowing_add(b[i]);
        let (s, c2) = s.overflowing_add(carry);
        out[i] = s;
        carry = u64::from(c1 || c2);
    }
    if carry == 1 || ge(&out, &PRIME) {
        sub_in_place(&mut out, &PRIME);
    }
    out
}

/// `base` to the `exponent` modulo the field prime, square and multiply from
/// the least significant bit up.
fn powmod_p(base: &[u64; 4], exponent: &[u64; 4]) -> [u64; 4] {
    let mut result = [1u64, 0, 0, 0];
    let mut b = *base;
    for limb in 0..4 {
        for bit in 0..64 {
            if (exponent[limb] >> bit) & 1 == 1 {
                result = mulmod_p(&result, &b);
            }
            b = mulmod_p(&b, &b);
        }
    }
    result
}

/// The point with this x coordinate and a y of the parity asked for.
///
/// SP1 6.4.0 has no Weierstrass decompression precompile. The only
/// implementation left in its executor panics on sight, and nothing else
/// implements the syscall, so this does the work in guest code:
/// `y = (x^3 + 7)^((prime + 1) / 4)`, which is a square root because the prime
/// is 3 modulo 4, negated when its parity is not the one wanted. Squaring the
/// root back and comparing rejects an x that is not on the curve, which is the
/// check the precompile also owed its caller.
///
/// The cost is about 380 modular multiplies, each one a `sys_bigint` call. That
/// is real work rather than a stub, and it is what any guest on this version of
/// SP1 must pay to turn an x coordinate into a point.
fn decompress(x: &[u64; 4], y_is_odd: bool) -> Secp256k1Point {
    assert!(!ge(x, &PRIME), "x is not a field element");
    let x3 = mulmod_p(&mulmod_p(x, x), x);
    let square = addmod_p(&x3, &[7, 0, 0, 0]);
    let mut y = powmod_p(&square, &SQRT_EXPONENT);
    assert!(mulmod_p(&y, &y) == square, "x is not on the curve");
    if !is_zero(&y) && (y[0] & 1 == 1) != y_is_odd {
        let mut negated = PRIME;
        sub_in_place(&mut negated, &y);
        y = negated;
    }
    let mut limbs = [0u64; 8];
    limbs[0..4].copy_from_slice(x);
    limbs[4..8].copy_from_slice(&y);
    Secp256k1Point::new(limbs)
}

fn generator() -> Secp256k1Point {
    Secp256k1Point::new(<Secp256k1Point as AffinePoint<8>>::GENERATOR)
}

/// `scalar * point`, or the identity when the scalar is zero.
fn mul(point: &Secp256k1Point, scalar: &[u64; 4]) -> Secp256k1Point {
    let mut p = *point;
    p.mul_assign(scalar);
    p
}

fn coords(p: &Secp256k1Point) -> ([u64; 4], [u64; 4]) {
    let limbs = p.limbs_ref();
    let mut x = [0u64; 4];
    let mut y = [0u64; 4];
    x.copy_from_slice(&limbs[0..4]);
    y.copy_from_slice(&limbs[4..8]);
    (x, y)
}

/// Recovers the public key that signed `hash`, given `(r, s)` and the parity of
/// the y coordinate of the nonce point -- Ethereum's `ecrecover`.
///
/// `R` is rebuilt from `r` by decompression; then
/// `Q = (-hash / r) * G + (s / r) * R`.
fn recover(hash: &[u64; 4], r: &[u64; 4], s: &[u64; 4], y_is_odd: bool) -> Secp256k1Point {
    assert!(!is_zero(r) && !is_zero(s), "signature scalars must be non-zero");
    let big_r = decompress(r, y_is_odd);
    let r_inv = invmod(r);
    let u1 = mulmod(&negmod(hash), &r_inv);
    let u2 = mulmod(s, &r_inv);
    let mut q = mul(&generator(), &u1);
    let other = mul(&big_r, &u2);
    q.complete_add_assign(&other);
    q
}

/// Keccak-256, on the permutation precompile, with Ethereum's padding.
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

/// The twenty-byte Ethereum address of a public key: the low twenty bytes of
/// the Keccak-256 of its sixty-four uncompressed coordinate bytes, big-endian.
fn address(q: &Secp256k1Point) -> [u8; 20] {
    let (x, y) = coords(q);
    let mut bytes = [0u8; 64];
    for i in 0..4 {
        bytes[24 - 8 * i..32 - 8 * i].copy_from_slice(&x[i].to_be_bytes());
        bytes[56 - 8 * i..64 - 8 * i].copy_from_slice(&y[i].to_be_bytes());
    }
    let h = keccak256(&bytes);
    let mut out = [0u8; 20];
    out.copy_from_slice(&h[12..32]);
    out
}

/// A scalar in `[1, order)` derived from a seed, so that neither the key nor
/// the nonce is ever zero.
fn scalar_from(seed: &[u8]) -> [u64; 4] {
    let h = keccak256(seed);
    let mut out = [0u64; 4];
    for i in 0..4 {
        let mut w = [0u8; 8];
        w.copy_from_slice(&h[8 * i..8 * i + 8]);
        out[i] = u64::from_le_bytes(w);
    }
    // Clear the top bits so the value is comfortably below the order, then
    // rule out zero.
    out[3] &= 0x7FFF_FFFF_FFFF_FFFF;
    if is_zero(&out) {
        out[0] = 1;
    }
    out
}

/// Signs `hash` with private key `d` and nonce `k`, returning `(r, s)` and the
/// parity of the nonce point's y coordinate.
fn sign(hash: &[u64; 4], d: &[u64; 4], k: &[u64; 4]) -> ([u64; 4], [u64; 4], bool) {
    let big_r = mul(&generator(), k);
    let (rx, ry) = coords(&big_r);
    let r = to_scalar(&rx);
    let s = mulmod(&invmod(k), &addmod(hash, &mulmod(&r, d)));
    (r, s, ry[0] & 1 == 1)
}

/// The modular multiply, the curve arithmetic and the whole sign-and-recover
/// round trip, each checked before anything is measured.
fn self_test() {
    // `(order - 1)^2 = 1` modulo the order, which pins both the constant above
    // and `sys_bigint`'s operation selector.
    let mut minus_one = ORDER;
    sub_in_place(&mut minus_one, &[1, 0, 0, 0]);
    assert_eq!(
        mulmod(&minus_one, &minus_one),
        [1, 0, 0, 0],
        "modular multiply, or the group order, or the sys_bigint selector"
    );
    // And inversion undoes multiplication.
    let a = scalar_from(b"inverse");
    assert_eq!(mulmod(&a, &invmod(&a)), [1, 0, 0, 0], "modular inverse");
    // Keccak-256 of the empty string.
    assert_eq!(
        keccak256(b""),
        [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2,
            0xdc, 0xc7, 0x03, 0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
            0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
        ],
        "keccak256(\"\")"
    );
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    self_test();

    let mut acc = 0u32;
    for i in 0..n {
        let d = scalar_from(&[b'k', i as u8, (i >> 8) as u8, (i >> 16) as u8]);
        let k = scalar_from(&[b'n', i as u8, (i >> 8) as u8, (i >> 16) as u8]);
        let hash_bytes = keccak256(&i.to_le_bytes());
        let mut hash = [0u64; 4];
        for j in 0..4 {
            let mut w = [0u8; 8];
            w.copy_from_slice(&hash_bytes[8 * j..8 * j + 8]);
            hash[j] = u64::from_le_bytes(w);
        }
        let hash = to_scalar(&hash);

        let public = mul(&generator(), &d);
        let (r, s, y_is_odd) = sign(&hash, &d, &k);
        let recovered = recover(&hash, &r, &s, y_is_odd);
        assert!(!recovered.is_infinity(), "recovered the identity");
        assert_eq!(
            coords(&recovered),
            coords(&public),
            "recovered the wrong key for signature {i}"
        );
        acc ^= u32::from_le_bytes(address(&recovered)[0..4].try_into().unwrap());
    }
    sp1_zkvm::io::commit(&acc);
}
