// SPDX-License-Identifier: MIT OR Apache-2.0
//! slop's Merkle tree commitment, built on the GPU.
//!
//! `commit_tensors` is the expensive half of committing a shard's traces — 47.9
//! thread-seconds of a 52.8 second proof, more than any other single thing the
//! prover does. The tree itself is simple: hash each row of the concatenated
//! tensors into a digest, then compress pairs upwards. The kernels for both are
//! the ones the probe suite already checks against slop's own committer.
//!
//! What this reproduces exactly, because the commitment has to match:
//!
//! - the leaf layer is padded to a power of two with **zero** digests, which is
//!   what `first_digest_layer` does for rows past the matrix height;
//! - a leaf absorbs every tensor's row in order, so tensors are gathered into
//!   one interleaved buffer first;
//! - every layer is kept, since they are the prover data the opening proofs
//!   walk.
//!
//! The final `compress([root, hash([log_height, total_width])])` is left to the
//! caller on the CPU: it is two permutations, and doing it with slop's own
//! hasher removes any chance of the commitment differing.

use std::sync::OnceLock;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

use crate::gpu::shader_src;
use crate::probes::poseidon2::rc_monty;

const THREADS: u64 = 256;

struct Context {
    device: Device,
    gather: ComputePipelineState,
    leaves: ComputePipelineState,
    compress: ComputePipelineState,
    round_constants: Buffer,
}

/// One device, one queue, built once. `MTLCommandQueue` is thread-safe, which
/// matters because shards commit concurrently.
impl Context {
    /// This thread's command queue; see [`crate::nocopy::thread_queue`].
    fn queue(&self) -> CommandQueue {
        crate::nocopy::thread_queue(&self.device)
    }
}

fn context() -> &'static Context {
    static CTX: OnceLock<Context> = OnceLock::new();
    CTX.get_or_init(|| {
        let device = Device::system_default().expect("no Metal device found");
        let library = device
            .new_library_with_source(&shader_src(), &metal::CompileOptions::new())
            .expect("shader compilation failed");
        let pso = |name: &str| {
            let f = library.get_function(name, None).expect("kernel not found");
            device.new_compute_pipeline_state_with_function(&f).expect("pipeline")
        };
        let rc = rc_monty();
        let round_constants = device.new_buffer_with_data(
            rc.as_ptr().cast(),
            std::mem::size_of_val(rc.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Context {
            gather: pso("merkle_gather"),
            leaves: pso("merkle_leaves"),
            compress: pso("merkle_compress"),
            round_constants,
            device,
        }
    })
}

/// Whether this build can reach a Metal device at all.
#[must_use]
pub fn available() -> bool {
    Device::system_default().is_some()
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, v: u32) {
    enc.set_bytes(index, 4, (&v as *const u32).cast());
}

fn dispatch(enc: &metal::ComputeCommandEncoderRef, n: u64) {
    enc.dispatch_thread_groups(
        MTLSize::new(n.div_ceil(THREADS), 1, 1),
        MTLSize::new(THREADS, 1, 1),
    );
}

/// Takes a slice of Montgomery words without copying when it can.
fn upload(device: &Device, words: &[u32]) -> Buffer {
    let bytes = std::mem::size_of_val(words);
    let start = std::time::Instant::now();
    // The encode wrote this codeword through a buffer a moment ago; taking it
    // rather than making another saves mapping the largest allocation in the
    // proof a second time. See `sp1_metal_core::handoff`.
    if let Some(buffer) = crate::handoff::take(words.as_ptr().cast(), bytes) {
        crate::stats::record_input(crate::stats::Input::Resident, bytes, 0);
        return buffer;
    }
    if bytes > 0 && crate::nocopy::takeable(words.as_ptr().cast()) {
        crate::stats::record_input(crate::stats::Input::Wrapped, bytes, 0);
        // SAFETY: the buffer borrows `words` for the dispatch, which is awaited
        // before this returns.
        return unsafe {
            device.new_buffer_with_bytes_no_copy(
                words.as_ptr().cast(),
                bytes as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        };
    }
    // A copy is the host reading these words; see `nocopy::read_only`.
    crate::queue::drain();
    let buffer = device.new_buffer_with_data(
        words.as_ptr().cast(),
        bytes.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    crate::stats::record_input(
        crate::stats::Input::Copied,
        bytes,
        start.elapsed().as_nanos() as u64,
    );
    buffer
}

/// Builds every digest layer of the tree slop would build, bottom up.
///
/// `tensors` gives each tensor's Montgomery words and its width; all of them
/// have `rows` rows, which is what upstream asserts before committing. Layer
/// zero has `rows.next_power_of_two()` entries; the last has one.
///
/// Returns `None` when there is nothing to commit to, leaving the caller to
/// fall back to the CPU committer.
#[must_use]
pub fn digest_layers(tensors: &[(&[u32], usize)], rows: usize) -> Option<Vec<Vec<[u32; 8]>>> {
    if rows == 0 || tensors.is_empty() {
        return None;
    }
    let total_width: usize = tensors.iter().map(|(_, w)| *w).sum();
    if total_width == 0 {
        return None;
    }
    let ctx = context();
    let padded = rows.next_power_of_two();
    // The caller must own its layers, so the kernels write straight into the
    // vectors that will be handed back rather than into device buffers whose
    // contents are copied out afterwards. The bottom layer is by far the
    // largest, and it is the one big enough to be page-aligned and so taken
    // where it lies; the small ones near the root fall back to a copy.
    let mut layers: Vec<Vec<[u32; 8]>> = (0..=padded.trailing_zeros())
        .map(|l| {
            let n = padded >> l;
            let mut v: Vec<[u32; 8]> = Vec::with_capacity(n);
            // SAFETY: the kernels write every digest before anything reads
            // them, except the padded tail of layer zero, zeroed below.
            unsafe { v.set_len(n) };
            v
        })
        .collect();
    let mut direct = Vec::with_capacity(layers.len());
    let levels: Vec<Buffer> = layers
        .iter_mut()
        .map(|v| {
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    v.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(v.as_slice()),
                )
            };
            // SAFETY: the buffer is read and written only by the dispatches
            // below, which are awaited before the vectors are read.
            let (buffer, aliased) = unsafe { crate::nocopy::write_into(&ctx.device, bytes) };
            direct.push(aliased);
            buffer
        })
        .collect();

    // Rows beyond the matrix height hash to the default digest, which is zero.
    // Written before the buffer is submitted, so the compression layers see it.
    if padded > rows {
        // SAFETY: shared storage is host-writable, and this is inside the buffer.
        unsafe {
            let base = levels[0].contents().cast::<u32>().add(rows * 8);
            std::ptr::write_bytes(base, 0, (padded - rows) * 8);
        }
    }

    let uploaded: Vec<Buffer> =
        tensors.iter().map(|(words, _)| upload(&ctx.device, words)).collect();
    // One tensor is already in the layout the leaf hash wants.
    let gathered = if tensors.len() == 1 {
        None
    } else {
        Some(ctx.device.new_buffer(
            (rows * total_width * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        ))
    };

    let queue = ctx.queue();
    let cb = queue.new_command_buffer();
    if let Some(dst) = &gathered {
        let mut col = 0usize;
        for ((_, width), src) in tensors.iter().zip(uploaded.iter()) {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.gather);
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            set_u32(enc, 2, *width as u32);
            set_u32(enc, 3, total_width as u32);
            set_u32(enc, 4, col as u32);
            set_u32(enc, 5, (rows * width) as u32);
            dispatch(enc, (rows * width) as u64);
            enc.end_encoding();
            col += width;
        }
    }
    let rows_buf = gathered.as_ref().unwrap_or(&uploaded[0]);
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&ctx.leaves);
    enc.set_buffer(0, Some(rows_buf), 0);
    enc.set_buffer(1, Some(&levels[0]), 0);
    set_u32(enc, 2, total_width as u32);
    set_u32(enc, 3, rows as u32);
    enc.set_buffer(4, Some(&ctx.round_constants), 0);
    dispatch(enc, rows as u64);
    enc.end_encoding();
    for l in 1..levels.len() {
        let parents = padded >> l;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.compress);
        enc.set_buffer(0, Some(&levels[l - 1]), 0);
        enc.set_buffer(1, Some(&levels[l]), 0);
        set_u32(enc, 2, parents as u32);
        enc.set_buffer(3, Some(&ctx.round_constants), 0);
        dispatch(enc, parents as u64);
        enc.end_encoding();
    }
    // Waits for the encode as well, which is committed to this queue just
    // before the tree and deliberately not waited for there.
    crate::queue::submit(cb);

    let start = std::time::Instant::now();
    let mut copied = 0usize;
    for ((layer, buf), aliased) in layers.iter_mut().zip(levels.iter()).zip(direct) {
        if aliased {
            continue;
        }
        let bytes = std::mem::size_of_val(layer.as_slice());
        copied += bytes;
        // SAFETY: the kernels wrote eight words per digest into a buffer of
        // exactly this size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buf.contents().cast::<u8>(),
                layer.as_mut_ptr().cast::<u8>(),
                bytes,
            );
        }
    }
    if copied > 0 {
        crate::stats::record_output(
            crate::stats::Output::Staged,
            copied,
            start.elapsed().as_nanos() as u64,
        );
    }
    Some(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use slop_algebra::AbstractField;
    use slop_challenger::IopCtx;
    use slop_commit::Message;
    use slop_koala_bear::KoalaBearDegree4Duplex;
    use slop_merkle_tree::{MerkleTreeTcs, Poseidon2KoalaBear16Prover, TensorCsProver};
    use slop_symmetric::{CryptographicHasher, PseudoCompressionFunction};
    use slop_tensor::Tensor;

    use crate::kb::F;

    /// KoalaBear is `repr(transparent)` over its Montgomery word.
    fn words(xs: &[F]) -> &[u32] {
        unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<u32>(), xs.len()) }
    }

    fn digest_words(d: &[F; 8]) -> [u32; 8] {
        let w = words(d.as_slice());
        std::array::from_fn(|i| w[i])
    }

    /// Builds the same tree both ways and insists every digest matches, then
    /// that the commitment slop would publish comes out of ours.
    fn check(shape: &[(usize, usize)], rows: usize) {
        let (hasher, compressor) = KoalaBearDegree4Duplex::default_hasher_and_compressor();
        let prover = Poseidon2KoalaBear16Prover::new(MerkleTreeTcs {
            hasher: hasher.clone(),
            compressor: compressor.clone(),
        });
        let mut seed = 1u32;
        let tensors: Vec<Tensor<F>> = shape
            .iter()
            .map(|&(_, width)| {
                let vals: Vec<F> = (0..rows * width)
                    .map(|_| {
                        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        F::from_canonical_u32(seed % 0x7f00_0001)
                    })
                    .collect();
                Tensor::from(vals).reshape([rows, width])
            })
            .collect();
        let message: Message<Tensor<F>> = tensors.clone().into();
        let (expected_commitment, data) = prover.commit_tensors(message).expect("slop commit");

        let borrowed: Vec<(&[u32], usize)> =
            tensors.iter().map(|t| (words(t.as_slice()), t.sizes()[1])).collect();
        let layers = digest_layers(&borrowed, rows).expect("metal commit");

        let expected = &data.0.digest_layers;
        assert_eq!(layers.len(), expected.len(), "layer count for {shape:?} rows {rows}");
        for (l, (got, want)) in layers.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got.len(), want.len(), "layer {l} length");
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(*g, digest_words(w), "layer {l} digest {i}");
            }
        }

        // The published commitment binds the shape, so check that too.
        let total_width: usize = shape.iter().map(|&(_, w)| w).sum();
        let log_height = rows.next_power_of_two().ilog2() as usize;
        let root: [F; 8] = *expected.last().unwrap().first().unwrap();
        let shape_hash: [F; 8] = hasher.hash_iter([
            F::from_canonical_usize(log_height),
            F::from_canonical_usize(total_width),
        ]);
        let ours = compressor.compress([root, shape_hash]);
        assert_eq!(ours, expected_commitment, "commitment for {shape:?} rows {rows}");
    }

    #[test]
    fn one_tensor_matches_slop() {
        check(&[(0, 13)], 1024);
    }

    #[test]
    fn several_tensors_are_hashed_in_order() {
        check(&[(0, 5), (0, 8), (0, 3)], 512);
    }

    #[test]
    fn a_height_that_is_not_a_power_of_two_pads_with_zero_digests() {
        check(&[(0, 7)], 1000);
    }

    #[test]
    fn a_single_row_still_builds_a_tree() {
        check(&[(0, 16)], 1);
    }

    #[test]
    fn a_wide_row_absorbs_in_several_chunks() {
        check(&[(0, 40)], 256);
    }
}
