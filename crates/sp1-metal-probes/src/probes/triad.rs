// SPDX-License-Identifier: MIT OR Apache-2.0
//! STREAM-style triad on integer lanes: pins the streaming-bandwidth ceiling,
//! which is the hard limit for fold-shaped (bandwidth-bound) prover kernels.

use rayon::prelude::*;
use serde_json::{json, Value};

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::probes::{cpu_bench, times_json};

const SCALE: u32 = 3;

fn fill(n4: usize, salt: u32) -> Vec<[u32; 4]> {
    (0..n4)
        .map(|i| {
            let x = (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(salt);
            [x, x ^ 0x5555_5555, x.wrapping_add(1), x.wrapping_mul(3)]
        })
        .collect()
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let n4 = 4096usize;
    let a = fill(n4, 1);
    let b = fill(n4, 2);
    let buf_a = gpu.buffer_from(&a);
    let buf_b = gpu.buffer_from(&b);
    let buf_c = gpu.buffer_empty::<[u32; 4]>(n4);
    let pso = gpu.pipeline("triad")?;
    gpu.time_encoded(1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        set_bytes(enc, 3, &SCALE);
        dispatch_1d(enc, n4 as u64);
    });
    let c: Vec<[u32; 4]> = gpu.read_vec(&buf_c, n4);
    for i in 0..n4 {
        for lane in 0..4 {
            let expected = a[i][lane].wrapping_add(SCALE.wrapping_mul(b[i][lane]));
            if c[i][lane] != expected {
                return Err(format!("triad mismatch at [{i}][{lane}]"));
            }
        }
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n4: usize, reps: usize, with_cpu: bool) -> Result<Value, String> {
    let a = fill(n4, 1);
    let b = fill(n4, 2);
    let buf_a = gpu.buffer_from(&a);
    let buf_b = gpu.buffer_from(&b);
    let buf_c = gpu.buffer_empty::<[u32; 4]>(n4);
    let pso = gpu.pipeline("triad")?;
    let times = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        set_bytes(enc, 3, &SCALE);
        dispatch_1d(enc, n4 as u64);
    });
    let bytes_per_pass = (3 * n4 * 16) as f64;
    let mut result = json!({
        "probe": "triad",
        "params": { "n_uint4": n4, "reps": reps },
        "times": times_json(&times),
        "gbytes_per_s_best": bytes_per_pass / times.best / 1e9,
        "gbytes_per_s_mean": bytes_per_pass / times.mean / 1e9,
    });
    if with_cpu {
        let mut c = vec![[0u32; 4]; n4];
        let t = cpu_bench(reps, || {
            c.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(|((c, a), b)| {
                for lane in 0..4 {
                    c[lane] = a[lane].wrapping_add(SCALE.wrapping_mul(b[lane]));
                }
            });
        });
        result["cpu"] = json!({
            "times": times_json(&t),
            "gbytes_per_s_best": bytes_per_pass / t.best / 1e9,
            "note": "same triad over rayon on all cores",
        });
        result["gpu_over_cpu"] = json!(t.best / times.best);
    }
    Ok(result)
}
