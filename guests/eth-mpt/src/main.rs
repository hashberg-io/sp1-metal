#![no_main]
//! Verifying Merkle-Patricia proofs against an Ethereum state root — what a
//! light client, a bridge or a storage-proof oracle spends its time on.
//!
//! Each proof is a path of RLP-encoded nodes from the root to a leaf. Verifying
//! one means, at every level: Keccak-256 the node, check that the hash is the
//! one the parent pointed at, decode the RLP to find the next pointer, and
//! consume the nibbles of the key that the node accounts for. That mix —
//! Keccak over short buffers, interleaved with branchy byte-level decoding — is
//! what makes this different from anything else in the suite, where the
//! precompiles are driven in tight loops with nothing between them.
//!
//! **The tries are synthetic.** The node layout, the RLP encoding, the
//! sixteen-way branching and the hash-linking are Ethereum's; the keys and
//! values are generated from a seed. A verifier's cost depends on the shape of
//! the trie and the length of the paths, not on whose state it holds.
//!
//! Keccak-256 is checked against its published empty-string vector first, so a
//! wrong sponge fails here rather than becoming a fast, meaningless proof.

sp1_zkvm::entrypoint!(main);
use sp1_zkvm::syscalls::syscall_keccak_permute;

/// Keccak-256 over one buffer, built on the permutation precompile.
///
/// The rate is 136 bytes and the padding is Keccak's original `0x01`/`0x80`
/// rule, which is what Ethereum uses — not SHA-3's `0x06`.
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

/// RLP for a byte string, which is every payload this builds.
fn rlp_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    if bytes.len() == 1 && bytes[0] < 0x80 {
        out.push(bytes[0]);
    } else if bytes.len() <= 55 {
        out.push(0x80 + bytes.len() as u8);
        out.extend_from_slice(bytes);
    } else {
        let len = bytes.len().to_be_bytes();
        let start = len.iter().position(|b| *b != 0).unwrap_or(len.len() - 1);
        out.push(0xb7 + (len.len() - start) as u8);
        out.extend_from_slice(&len[start..]);
        out.extend_from_slice(bytes);
    }
}

/// RLP for a list whose payload is already encoded.
fn rlp_list(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 4);
    if payload.len() <= 55 {
        out.push(0xc0 + payload.len() as u8);
    } else {
        let len = payload.len().to_be_bytes();
        let start = len.iter().position(|b| *b != 0).unwrap_or(len.len() - 1);
        out.push(0xf7 + (len.len() - start) as u8);
        out.extend_from_slice(&len[start..]);
    }
    out.extend_from_slice(payload);
    out
}

/// Reads one RLP header, returning whether it is a list, the payload offset and
/// the payload length. This is the decoding a verifier really does: it is given
/// bytes it did not build and has to find its way through them.
fn rlp_header(node: &[u8]) -> (bool, usize, usize) {
    let prefix = node[0];
    if prefix < 0x80 {
        (false, 0, 1)
    } else if prefix <= 0xb7 {
        (false, 1, (prefix - 0x80) as usize)
    } else if prefix < 0xc0 {
        let n = (prefix - 0xb7) as usize;
        let mut len = 0usize;
        for b in &node[1..1 + n] {
            len = (len << 8) | *b as usize;
        }
        (false, 1 + n, len)
    } else if prefix <= 0xf7 {
        (true, 1, (prefix - 0xc0) as usize)
    } else {
        let n = (prefix - 0xf7) as usize;
        let mut len = 0usize;
        for b in &node[1..1 + n] {
            len = (len << 8) | *b as usize;
        }
        (true, 1 + n, len)
    }
}

/// A branch node: sixteen child pointers and an empty value slot.
fn branch_node(children: &[[u8; 32]; 16]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(17 * 33);
    for child in children.iter() {
        rlp_bytes(&mut payload, child);
    }
    rlp_bytes(&mut payload, &[]);
    rlp_list(&payload)
}

/// A leaf node: the remaining nibbles of the key, hex-prefix encoded, and the
/// value.
fn leaf_node(path: &[u8], value: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(path.len() / 2 + 1);
    // Hex-prefix, the encoding that distinguishes a leaf from an extension and
    // records whether the nibble count is odd.
    if path.len() % 2 == 0 {
        encoded.push(0x20);
        for pair in path.chunks(2) {
            encoded.push((pair[0] << 4) | pair[1]);
        }
    } else {
        encoded.push(0x30 | path[0]);
        for pair in path[1..].chunks(2) {
            encoded.push((pair[0] << 4) | pair[1]);
        }
    }
    let mut payload = Vec::with_capacity(encoded.len() + value.len() + 4);
    rlp_bytes(&mut payload, &encoded);
    rlp_bytes(&mut payload, value);
    rlp_list(&payload)
}

/// Builds one proof: a path of `depth` branch nodes ending in a leaf, hashed
/// from the bottom up so that each parent names its child.
///
/// Returned bottom-up, with the root last, which is the order the verifier
/// walks it in reverse.
fn build_proof(seed: u32, depth: usize) -> (Vec<Vec<u8>>, Vec<u8>, [u8; 32]) {
    let value = keccak256(&seed.to_le_bytes()).to_vec();
    // The nibbles of the key this proof opens. The first `depth` of them choose
    // a child at each branch; the rest sit in the leaf.
    let key: Vec<u8> = (0..64u8)
        .map(|i| (keccak256(&[seed as u8, i])[0]) & 0x0f)
        .collect();
    let mut nodes = Vec::with_capacity(depth + 1);
    let leaf = leaf_node(&key[depth..], &value);
    let mut hash = keccak256(&leaf);
    nodes.push(leaf);
    for level in (0..depth).rev() {
        let mut children = [[0u8; 32]; 16];
        // The siblings are arbitrary but fixed, as they are in a real trie:
        // the verifier hashes the whole node regardless.
        for (j, slot) in children.iter_mut().enumerate() {
            *slot = keccak256(&[seed as u8, level as u8, j as u8]);
        }
        children[key[level] as usize] = hash;
        let node = branch_node(&children);
        hash = keccak256(&node);
        nodes.push(node);
    }
    (nodes, key, hash)
}

/// Verifies one proof against `root`, the way a light client does: hash each
/// node, check it is the one the parent named, decode to find the next pointer.
fn verify_proof(nodes: &[Vec<u8>], key: &[u8], root: &[u8; 32]) -> usize {
    // The proof arrives bottom-up; walk it root first.
    let mut expected = *root;
    let mut level = 0usize;
    for node in nodes.iter().rev() {
        assert_eq!(keccak256(node), expected, "node hash does not match its parent");
        let (is_list, off, len) = rlp_header(node);
        assert!(is_list, "a trie node is an RLP list");
        let payload = &node[off..off + len];
        if level == nodes.len() - 1 {
            // The leaf: two items, the hex-prefix path and the value.
            let (_, o, l) = rlp_header(payload);
            assert!(o + l <= payload.len(), "leaf path runs past the node");
            return level + 1;
        }
        // A branch: walk to the child the next nibble of the key selects.
        let mut cursor = 0usize;
        let mut child: Option<[u8; 32]> = None;
        for slot in 0..16usize {
            let (_, o, l) = rlp_header(&payload[cursor..]);
            if slot == key[level] as usize {
                assert_eq!(l, 32, "a branch child is a 32-byte hash");
                let mut h = [0u8; 32];
                h.copy_from_slice(&payload[cursor + o..cursor + o + l]);
                child = Some(h);
            }
            cursor += o + l;
        }
        expected = child.expect("the branch names the child the key selects");
        level += 1;
    }
    level
}

fn main() {
    let n: u32 = sp1_zkvm::io::read();
    // Keccak-256 of the empty string, the vector every implementation is
    // checked against.
    assert_eq!(
        keccak256(b""),
        [
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2,
            0xdc, 0xc7, 0x03, 0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
            0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
        ],
        "keccak256(\"\")"
    );

    // Six levels is about what a populated Ethereum state trie needs to
    // separate a few million accounts.
    const DEPTH: usize = 6;
    let mut levels = 0usize;
    for i in 0..n {
        let (nodes, key, root) = build_proof(i, DEPTH);
        levels += verify_proof(&nodes, &key, &root);
    }
    sp1_zkvm::io::commit(&(levels as u32));
}
