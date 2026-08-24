// SPDX-License-Identifier: MIT OR Apache-2.0
//! What slop's device seam costs on unified memory.
//!
//! This drives a real `slop_alloc::Buffer<T, MetalBackend>` through a kernel:
//! the buffer is allocated by slop, filled through its host pointer, handed to
//! a Metal kernel by looking up the `MTLBuffer` behind that pointer, and read
//! back through the pointer again. Nothing is copied at any point, and the
//! parity check is against slop's own `fold_mle` so the result is known to be
//! the same data the CPU path would have produced.
//!
//! The benchmark prices the copy that did not happen: what a backend with
//! separate device memory would have to move before the same kernel could run.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};
use slop_alloc::mem::{CopyDirection, DeviceMemory};
use slop_alloc::Buffer as SlopBuffer;
use slop_multilinear::fold_mle;
use slop_tensor::Tensor;

use crate::backend::MetalBackend;
use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::{f_of_monty, monty_of_f, rand_f_vec, F};
use crate::probes::{cpu_bench, times_json};

/// Allocates through slop, fills through the host pointer, and returns the
/// buffer. On shared storage these are the same bytes the GPU will read.
fn slop_buffer_of(backend: &MetalBackend, vals: &[u32]) -> SlopBuffer<u32, MetalBackend> {
    let mut buf = SlopBuffer::<u32, MetalBackend>::with_capacity_in(vals.len(), backend.clone());
    unsafe {
        std::ptr::copy_nonoverlapping(vals.as_ptr(), buf.as_mut_ptr(), vals.len());
        buf.set_len(vals.len());
    }
    buf
}

/// Binds a slop buffer to a kernel argument without copying it.
fn bind(
    backend: &MetalBackend,
    enc: &metal::ComputeCommandEncoderRef,
    index: u64,
    buf: &SlopBuffer<u32, MetalBackend>,
) -> Result<(), String> {
    let (mtl, offset) = backend
        .buffer_for(buf.as_ptr().cast())
        .ok_or("slop buffer is not backed by a Metal allocation")?;
    enc.set_buffer(index, Some(&mtl), offset);
    Ok(())
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let backend = MetalBackend::new()?;
    let before = backend.live_allocations();
    let mut rng = StdRng::seed_from_u64(81);
    let n = 1 << 12;
    let vals = rand_f_vec(&mut rng, n);
    let beta: F = {
        use rand::Rng;
        rng.gen()
    };

    let input = slop_buffer_of(&backend, &vals.iter().copied().map(monty_of_f).collect::<Vec<_>>());
    let mut output = SlopBuffer::<u32, MetalBackend>::with_capacity_in(n / 2, backend.clone());
    unsafe { output.set_len(n / 2) };
    if backend.live_allocations() != before + 2 {
        return Err("slop allocations did not reach the Metal backend".to_string());
    }

    let pso = gpu.pipeline("fold_base")?;
    let cb = gpu.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    bind(&backend, enc, 0, &input)?;
    bind(&backend, enc, 1, &output)?;
    set_bytes(enc, 2, &monty_of_f(beta));
    dispatch_1d(enc, (n / 2) as u64);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    // Read the result straight out of the slop buffer: the kernel wrote into
    // the same pages this pointer addresses.
    let got: Vec<F> = (0..n / 2)
        .map(|i| f_of_monty(unsafe { *output.as_ptr().add(i) }))
        .collect();
    let want = fold_mle(&Tensor::from(vals).reshape([n, 1]), beta);
    if got != want.as_slice() {
        return Err("kernel over a slop MetalBackend buffer disagrees with fold_mle".to_string());
    }
    drop(input);
    drop(output);
    if backend.live_allocations() != before {
        return Err("MetalBackend leaked allocations".to_string());
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n: usize, reps: usize) -> Result<Value, String> {
    let backend = MetalBackend::new()?;
    let mut rng = StdRng::seed_from_u64(82);
    let vals: Vec<u32> = rand_f_vec(&mut rng, n).into_iter().map(monty_of_f).collect();
    let input = slop_buffer_of(&backend, &vals);
    let mut output = SlopBuffer::<u32, MetalBackend>::with_capacity_in(n / 2, backend.clone());
    unsafe { output.set_len(n / 2) };
    let bytes = (n * 4) as f64;

    // Making the data visible to a kernel: a pointer lookup.
    let lookup = cpu_bench(reps, || {
        std::hint::black_box(backend.buffer_for(input.as_ptr().cast()).is_some());
    });
    // What a backend with its own device memory would have to move first.
    let mut staged = vec![0u32; n];
    let staging = cpu_bench(reps, || unsafe {
        backend
            .copy_nonoverlapping(
                vals.as_ptr().cast(),
                staged.as_mut_ptr().cast(),
                n * 4,
                CopyDirection::HostToDevice,
            )
            .expect("copy");
    });
    // The kernel itself, over the slop buffer, for scale.
    let pso = gpu.pipeline("fold_base")?;
    let beta = monty_of_f(F::from(rand::Rng::gen::<F>(&mut rng)));
    let kernel = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        bind(&backend, enc, 0, &input).expect("bind input");
        bind(&backend, enc, 1, &output).expect("bind output");
        set_bytes(enc, 2, &beta);
        dispatch_1d(enc, (n / 2) as u64);
    });

    Ok(json!({
        "probe": "backend",
        "params": { "n": n, "bytes": bytes, "reps": reps },
        "make_visible_to_kernel": {
            "times": times_json(&lookup),
            "nanoseconds": lookup.best * 1e9,
            "note": "resolving the MTLBuffer behind a slop pointer; the whole of the upload",
        },
        "staging_copy_avoided": {
            "times": times_json(&staging),
            "milliseconds": staging.best * 1e3,
            "gbytes_per_s": bytes / staging.best / 1e9,
            "note": "the host-to-device copy a backend with separate device memory would pay first",
        },
        "kernel": {
            "times": times_json(&kernel),
            "milliseconds": kernel.best * 1e3,
        },
        "copy_as_fraction_of_kernel": staging.best / kernel.best,
    }))
}
