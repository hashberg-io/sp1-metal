#![no_main]
//! Verifying BIP-340 Schnorr signatures — what every Taproot spend costs, and
//! what a Bitcoin rollup or a covenant verifier would pay per input.
//!
//! Each iteration signs a message and then verifies the signature, so every
//! iteration checks itself. Verification alone is roughly half the measured
//! work: it does one double-scalar multiplication where signing does one
//! single, and both pay the same tagged hashes.
//!
//! The whole of BIP-340 is here and in its own terms: x-only public keys, the
//! even-y convention that `lift_x` imposes, the tagged hashes with their
//! doubled tag prefix, and the `R.x == r` check that makes the signature
//! sixty-four bytes rather than sixty-five. The curve operations are SP1's
//! secp256k1 precompiles through `sp1-lib`; the scalar field is arithmetic
//! modulo the group order on `sys_bigint`.
//!
//! Keys and messages are derived from the loop index. Nothing here is secret.

sp1_zkvm::entrypoint!(main);
use sp1_lib::secp256k1::Secp256k1Point;
use sp1_lib::utils::{AffinePoint, WeierstrassAffinePoint};
use sp1_lib::sys_bigint;
use sp1_zkvm::syscalls::{syscall_sha256_compress, syscall_sha256_extend};

/// The order of the secp256k1 group, little-endian by 64-bit limb.
const ORDER: [u64; 4] = [
    0xBFD2_5E8C_D036_4141,
    0xBAAE_DCE6_AF48_A03B,
    0xFFFF_FFFF_FFFF_FFFE,
    0xFFFF_FFFF_FFFF_FFFF,
];

const INIT: [u64; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

fn compress(state: &mut [u64; 8], block: &[u8]) {
    let mut w = [0u64; 64];
    for (i, slot) in w.iter_mut().enumerate().take(16) {
        *slot = u32::from_be_bytes([
            block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3],
        ]) as u64;
    }
    unsafe {
        syscall_sha256_extend(&mut w);
        syscall_sha256_compress(&mut w, state);
    }
}

fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut state = INIT;
    let mut i = 0;
    while msg.len() - i >= 64 {
        compress(&mut state, &msg[i..i + 64]);
        i += 64;
    }
    let mut tail = [0u8; 128];
    let rest = msg.len() - i;
    tail[..rest].copy_from_slice(&msg[i..]);
    tail[rest] = 0x80;
    let blocks = if rest + 9 > 64 { 2 } else { 1 };
    let bits = (msg.len() as u64) * 8;
    tail[blocks * 64 - 8..blocks * 64].copy_from_slice(&bits.to_be_bytes());
    for b in 0..blocks {
        compress(&mut state, &tail[b * 64..(b + 1) * 64]);
    }
    let mut out = [0u8; 32];
    for (i, word) in state.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&(*word as u32).to_be_bytes());
    }
    out
}

/// BIP-340's tagged hash: `SHA256(SHA256(tag) || SHA256(tag) || msg)`.
///
/// The tag hash appears twice on purpose — it fills a whole SHA-256 block, so
/// the prefix costs one compression and cannot collide with any other tag's.
fn tagged_hash(tag: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let th = sha256(tag);
    let mut msg = [0u8; 64 + 160];
    msg[0..32].copy_from_slice(&th);
    msg[32..64].copy_from_slice(&th);
    let mut len = 64;
    for part in parts {
        msg[len..len + part.len()].copy_from_slice(part);
        len += part.len();
    }
    sha256(&msg[..len])
}

fn mulmod(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    unsafe {
        sys_bigint(&mut out, 0, a, b, &ORDER);
    }
    out
}

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

fn is_zero(a: &[u64; 4]) -> bool {
    a.iter().all(|l| *l == 0)
}

fn negmod(a: &[u64; 4]) -> [u64; 4] {
    if is_zero(a) {
        return [0u64; 4];
    }
    let mut out = ORDER;
    sub_in_place(&mut out, a);
    out
}

fn reduce(a: &[u64; 4]) -> [u64; 4] {
    let mut out = *a;
    if ge(&out, &ORDER) {
        sub_in_place(&mut out, &ORDER);
    }
    out
}

/// BIP-340 encodes every integer as thirty-two big-endian bytes.
fn to_be(a: &[u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..4 {
        out[24 - 8 * i..32 - 8 * i].copy_from_slice(&a[i].to_be_bytes());
    }
    out
}

fn from_be(bytes: &[u8; 32]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for i in 0..4 {
        let mut w = [0u8; 8];
        w.copy_from_slice(&bytes[24 - 8 * i..32 - 8 * i]);
        out[i] = u64::from_be_bytes(w);
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

/// `lift_x`: the point with this x coordinate and an even y, which is the only
/// one BIP-340 will accept. A thirty-two byte public key names a point this way
/// rather than carrying a parity bit.
fn lift_x(x: &[u64; 4]) -> Secp256k1Point {
    decompress(x, false)
}

fn scalar_from(seed: &[u8]) -> [u64; 4] {
    let h = sha256(seed);
    let mut out = from_be(&h);
    out[3] &= 0x7FFF_FFFF_FFFF_FFFF;
    if is_zero(&out) {
        out[0] = 1;
    }
    out
}

/// Signs `msg` with secret scalar `d0`, returning the sixty-four byte signature
/// and the x-only public key.
///
/// The two negations are BIP-340's: the secret is negated when the public
/// point has an odd y, and the nonce when the nonce point does, so that both
/// x-only encodings mean what the verifier assumes.
fn sign(msg: &[u8; 32], d0: &[u64; 4], aux: &[u8; 32]) -> ([u8; 64], [u8; 32]) {
    let public = mul(&generator(), d0);
    let (px, py) = coords(&public);
    let d = if py[0] & 1 == 1 { negmod(d0) } else { *d0 };
    let px_bytes = to_be(&px);

    // The nonce, bound to the secret, the key and the message.
    let t = {
        let mut t = to_be(&d);
        let a = tagged_hash(b"BIP0340/aux", &[aux]);
        for i in 0..32 {
            t[i] ^= a[i];
        }
        t
    };
    let rand = tagged_hash(b"BIP0340/nonce", &[&t, &px_bytes, msg]);
    let k0 = reduce(&from_be(&rand));
    assert!(!is_zero(&k0), "nonce is zero");

    let big_r = mul(&generator(), &k0);
    let (rx, ry) = coords(&big_r);
    let k = if ry[0] & 1 == 1 { negmod(&k0) } else { k0 };
    let rx_bytes = to_be(&rx);

    let e = reduce(&from_be(&tagged_hash(
        b"BIP0340/challenge",
        &[&rx_bytes, &px_bytes, msg],
    )));
    let s = addmod(&k, &mulmod(&e, &d));

    let mut sig = [0u8; 64];
    sig[0..32].copy_from_slice(&rx_bytes);
    sig[32..64].copy_from_slice(&to_be(&s));
    (sig, px_bytes)
}

/// BIP-340 verification, in full.
fn verify(msg: &[u8; 32], public_key: &[u8; 32], sig: &[u8; 64]) -> bool {
    let px = from_be(public_key);
    let point = lift_x(&px);
    let mut r_bytes = [0u8; 32];
    let mut s_bytes = [0u8; 32];
    r_bytes.copy_from_slice(&sig[0..32]);
    s_bytes.copy_from_slice(&sig[32..64]);
    let r = from_be(&r_bytes);
    let s = from_be(&s_bytes);
    if ge(&s, &ORDER) {
        return false;
    }
    let e = reduce(&from_be(&tagged_hash(
        b"BIP0340/challenge",
        &[&r_bytes, public_key, msg],
    )));
    // R = s*G - e*P, computed as s*G + (order - e)*P.
    let mut big_r = mul(&generator(), &s);
    let minus_e_p = mul(&point, &negmod(&e));
    big_r.complete_add_assign(&minus_e_p);
    if big_r.is_infinity() {
        return false;
    }
    let (rx, ry) = coords(&big_r);
    ry[0] & 1 == 0 && rx == r
}

fn self_test() {
    assert_eq!(
        sha256(b"abc"),
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde,
            0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
            0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
        ],
        "sha256(\"abc\")"
    );
    let mut minus_one = ORDER;
    sub_in_place(&mut minus_one, &[1, 0, 0, 0]);
    assert_eq!(
        mulmod(&minus_one, &minus_one),
        [1, 0, 0, 0],
        "modular multiply, the group order, or the sys_bigint selector"
    );
    // A signature verifies, and the same signature against a different message
    // does not — which is the check that a verifier returning `true` blindly
    // would pass and this one must not.
    let d = scalar_from(b"self test");
    let msg = sha256(b"a message");
    let (sig, pk) = sign(&msg, &d, &[0u8; 32]);
    assert!(verify(&msg, &pk, &sig), "a valid signature must verify");
    let other = sha256(b"another message");
    assert!(!verify(&other, &pk, &sig), "a signature must not verify for another message");
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    self_test();

    let mut accepted = 0u32;
    for i in 0..n {
        let d = scalar_from(&[b'd', i as u8, (i >> 8) as u8, (i >> 16) as u8]);
        let msg = sha256(&i.to_le_bytes());
        let (sig, pk) = sign(&msg, &d, &[0u8; 32]);
        assert!(verify(&msg, &pk, &sig), "signature {i} did not verify");
        accepted += 1;
    }
    sp1_zkvm::io::commit(&accepted);
}
