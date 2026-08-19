//! Register-resident integer op throughput: add, 32x32 low multiply, mulhi,
//! and full KoalaBear Montgomery multiply. The mulhi rate is the single
//! biggest unknown in the compute model — Apple does not publish it — and it
//! decides the field-multiply ceiling.

use serde_json::{json, Value};

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::probes::times_json;

const KERNELS: [(&str, u64); 4] =
    [("tp_add", 8), ("tp_mul_lo", 8), ("tp_mul_hi", 8), ("tp_kb_mul", 8)];

fn run_kernel(gpu: &Gpu, name: &str, grid: usize, iters: u32, reps: usize) -> Result<(Vec<u32>, f64), String> {
    let pso = gpu.pipeline(name)?;
    let out = gpu.buffer_empty::<u32>(grid);
    let times = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&out), 0);
        set_bytes(enc, 1, &iters);
        dispatch_1d(enc, grid as u64);
    });
    Ok((gpu.read_vec(&out, grid), times.best))
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    for (name, _) in KERNELS {
        let (first, _) = run_kernel(gpu, name, 256, 16, 1)?;
        let (second, _) = run_kernel(gpu, name, 256, 16, 1)?;
        if first != second {
            return Err(format!("{name} is not deterministic across runs"));
        }
        if first.iter().all(|&x| x == 0) {
            return Err(format!("{name} produced all-zero output (dead-code eliminated?)"));
        }
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, grid: usize, iters: u32, reps: usize) -> Result<Value, String> {
    let mut results = serde_json::Map::new();
    for (name, chains) in KERNELS {
        let pso = gpu.pipeline(name)?;
        let out = gpu.buffer_empty::<u32>(grid);
        let times = gpu.bench_encoded(reps, 1, |enc| {
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&out), 0);
            set_bytes(enc, 1, &iters);
            dispatch_1d(enc, grid as u64);
        });
        let ops = (grid as f64) * (iters as f64) * (chains as f64);
        results.insert(
            name.to_string(),
            json!({
                "times": times_json(&times),
                "giga_ops_per_s_best": ops / times.best / 1e9,
            }),
        );
    }
    Ok(json!({
        "probe": "imul",
        "params": { "grid": grid, "iters": iters, "reps": reps, "note": "ops-of-interest per thread-iter-chain is 1; each carries one extra xor/or" },
        "kernels": results,
    }))
}
