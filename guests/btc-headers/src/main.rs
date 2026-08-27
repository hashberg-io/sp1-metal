#![no_main]
//! Verifying a chain of Bitcoin block headers — what every Bitcoin light
//! client proves, and the statement ZeroSync and its descendants are built on.
//!
//! For each of `n` headers: hash the 80 bytes twice with SHA-256, check the
//! result against the target its `bits` field encodes, and check that it names
//! the previous header. That is the whole of header validation, and it is
//! dominated by the double hash: three SHA-256 compressions per header.
//!
//! **The chain is synthetic and its difficulty is regtest's.** A verifier's
//! cost does not depend on difficulty — difficulty is what the *miner* pays, and
//! the verifier compares one 256-bit number either way — so a chain at
//! `0x207fffff`, which is the real parameter Bitcoin's own regtest network uses,
//! costs exactly what a mainnet chain costs to check. What it saves is having to
//! mine, which would otherwise dwarf the thing being measured.
//!
//! The SHA-256 built on the precompiles is checked against two published
//! vectors before any of that: `sha256("abc")`, and the double hash of the real
//! mainnet genesis header, whose block hash every Bitcoin user has seen.

sp1_zkvm::entrypoint!(main);
use sp1_zkvm::syscalls::{syscall_sha256_compress, syscall_sha256_extend};


/// One SHA-256 compression, over a 64-byte block.
fn compress(state: &mut [u64; 8], block: &[u8]) {
    let mut w = [0u64; 64];
    for (i, slot) in w.iter_mut().enumerate().take(16) {
        *slot = u32::from_be_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]) as u64;
    }
    unsafe {
        syscall_sha256_extend(&mut w);
        syscall_sha256_compress(&mut w, state);
    }
}

const INIT: [u64; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// SHA-256 of a message shorter than 56 bytes over its final block, which is
/// every message this guest hashes: an 80-byte header and a 32-byte digest.
fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut state = INIT;
    let mut i = 0;
    while msg.len() - i >= 64 {
        compress(&mut state, &msg[i..i + 64]);
        i += 64;
    }
    // The tail, its 0x80 terminator and the bit length. A second block is
    // needed when the tail leaves no room for the eight length bytes.
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

fn double_sha256(msg: &[u8]) -> [u8; 32] {
    sha256(&sha256(msg))
}

/// The 256-bit target that a compact `bits` field encodes, big-endian.
fn target_from_bits(bits: u32) -> [u8; 32] {
    let exponent = (bits >> 24) as usize;
    let mantissa = bits & 0x007f_ffff;
    let mut target = [0u8; 32];
    // The mantissa sits so that its last byte is at position `exponent` counted
    // from the low end, which is how Bitcoin defines the encoding.
    if exponent <= 3 {
        let shifted = mantissa >> (8 * (3 - exponent));
        target[29..32].copy_from_slice(&shifted.to_be_bytes()[1..4]);
    } else if exponent <= 32 {
        let start = 32 - exponent;
        target[start..start + 3].copy_from_slice(&mantissa.to_be_bytes()[1..4]);
    }
    target
}

/// Whether `hash` is at or below `target`. The hash is compared in the order
/// Bitcoin displays it, most significant byte first, which is the reverse of
/// the order the digest comes out in.
fn meets_target(hash: &[u8; 32], target: &[u8; 32]) -> bool {
    for i in 0..32 {
        let h = hash[31 - i];
        let t = target[i];
        if h != t {
            return h < t;
        }
    }
    true
}

/// A synthetic header at regtest difficulty, naming `prev`.
///
/// Every field is where Bitcoin puts it and is the width Bitcoin gives it, so
/// the bytes a verifier hashes are laid out exactly as a real header's are.
fn header(index: u32, prev: &[u8; 32], bits: u32) -> [u8; 80] {
    let mut h = [0u8; 80];
    h[0..4].copy_from_slice(&1u32.to_le_bytes());
    h[4..36].copy_from_slice(prev);
    // Stands in for the Merkle root of the block's transactions, which a
    // header-chain proof does not open.
    let root = sha256(&index.to_le_bytes());
    h[36..68].copy_from_slice(&root);
    h[68..72].copy_from_slice(&(1_231_006_505u32 + index * 600).to_le_bytes());
    h[72..76].copy_from_slice(&bits.to_le_bytes());
    h[76..80].copy_from_slice(&index.to_le_bytes());
    h
}

/// The real mainnet genesis header, as 80 bytes.
const GENESIS: [u8; 80] = [
    0x01, 0x00, 0x00, 0x00,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0x3b, 0xa3, 0xed, 0xfd, 0x7a, 0x7b, 0x12, 0xb2, 0x7a, 0xc7, 0x2c, 0x3e,
    0x67, 0x76, 0x8f, 0x61, 0x7f, 0xc8, 0x1b, 0xc3, 0x88, 0x8a, 0x51, 0x32,
    0x3a, 0x9f, 0xb8, 0xaa, 0x4b, 0x1e, 0x5e, 0x4a,
    0x29, 0xab, 0x5f, 0x49,
    0xff, 0xff, 0x00, 0x1d,
    0x1d, 0xac, 0x2b, 0x7c,
];

/// Two published vectors, checked before the measurement so that a wrong hash
/// fails inside the guest rather than becoming a fast and meaningless proof.
fn self_test() {
    let abc = sha256(b"abc");
    assert_eq!(
        abc,
        [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde,
            0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
            0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
        ],
        "sha256(\"abc\")"
    );
    // The genesis block hash, in the internal little-endian order; reversed for
    // display it is 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f.
    let genesis = double_sha256(&GENESIS);
    assert_eq!(
        genesis,
        [
            0x6f, 0xe2, 0x8c, 0x0a, 0xb6, 0xf1, 0xb3, 0x72, 0xc1, 0xa6, 0xa2, 0x46,
            0xae, 0x63, 0xf7, 0x4f, 0x93, 0x1e, 0x83, 0x65, 0xe1, 0x5a, 0x08, 0x9c,
            0x68, 0xd6, 0x19, 0x00, 0x00, 0x00, 0x00, 0x00,
        ],
        "the mainnet genesis block hash"
    );
    // And the genesis header meets the mainnet target it declares.
    assert!(
        meets_target(&genesis, &target_from_bits(0x1d00_ffff)),
        "genesis proof of work"
    );
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    self_test();

    // Regtest's own difficulty parameter: a target so high that any header
    // meets it, which is what makes a synthetic chain possible without mining.
    const BITS: u32 = 0x207f_ffff;
    let target = target_from_bits(BITS);

    let mut prev = [0u8; 32];
    let mut work: u64 = 0;
    for i in 0..n {
        let h = header(i, &prev, BITS);
        let hash = double_sha256(&h);
        assert!(meets_target(&hash, &target), "header {i} proof of work");
        // What a light client actually checks: that this header names the one
        // before it. Reading the field back out rather than trusting the
        // builder is the point of the exercise.
        assert_eq!(&h[4..36], &prev[..], "header {i} does not extend the chain");
        work += 1;
        prev = hash;
    }
    sp1_zkvm::io::commit(&(u32::from_le_bytes([prev[0], prev[1], prev[2], prev[3]])));
    sp1_zkvm::io::commit(&(work as u32));
}
