//! Register-resident integer op throughput: add, 32x32 low multiply, mulhi,
//! and full KoalaBear Montgomery multiply. The mulhi rate is the single
//! biggest unknown in the compute model — Apple does not publish it — and it
//! decides the field-multiply ceiling.

use rayon::prelude::*;
use serde_json::{json, Value};
use slop_algebra::AbstractField;

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::F;
use crate::packed::{LANES, PF};
use crate::probes::{cpu_bench, times_json};

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

/// Independent multiply chains per CPU thread: enough to fill the vector
/// pipelines, since one chain is latency-bound rather than throughput-bound.
const CPU_CHAINS: usize = 8;

/// The CPU's own register-resident multiply ceiling, in packed lanes. Without
/// it there is no way to tell whether a kernel sitting at the GPU's field
/// multiply roofline is beating the CPU's hardware or only beating one
/// particular loop.
fn cpu_kb_mul(tasks: usize, iters: u32) -> f64 {
    let total: usize = (0..tasks)
        .into_par_iter()
        .map(|t| {
            let mut acc: [PF; CPU_CHAINS] =
                std::array::from_fn(|c| PF::from(F::from_canonical_u32((t * 31 + c * 7 + 3) as u32)));
            let k: [PF; CPU_CHAINS] =
                std::array::from_fn(|c| PF::from(F::from_canonical_u32((c * 11 + 5) as u32)));
            for _ in 0..iters {
                for c in 0..CPU_CHAINS {
                    acc[c] *= k[c];
                }
            }
            std::hint::black_box(&acc);
            CPU_CHAINS * iters as usize * LANES
        })
        .sum();
    total as f64
}

pub fn bench(gpu: &Gpu, grid: usize, iters: u32, reps: usize, with_cpu: bool) -> Result<Value, String> {
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
    let mut result = json!({
        "probe": "imul",
        "params": { "grid": grid, "iters": iters, "reps": reps, "note": "ops-of-interest per thread-iter-chain is 1; each carries one extra xor/or" },
        "kernels": results,
    });
    if with_cpu {
        let tasks = 1024usize;
        let cpu_iters = 4096u32;
        let total = (tasks * CPU_CHAINS * cpu_iters as usize * LANES) as f64;
        let t = cpu_bench(reps, || {
            std::hint::black_box(cpu_kb_mul(tasks, cpu_iters));
        });
        let gpu_best = result["kernels"]["tp_kb_mul"]["giga_ops_per_s_best"]
            .as_f64()
            .expect("gpu kb_mul rate recorded");
        let cpu_rate = total / t.best / 1e9;
        result["cpu_packed"] = json!({
            "times": times_json(&t),
            "lanes": LANES,
            "chains": CPU_CHAINS,
            "tasks": tasks,
            "iters": cpu_iters,
            "giga_kb_muls_per_s_best": cpu_rate,
            "note": "packed (NEON) KoalaBear multiplies in registers, rayon on all cores",
        });
        result["gpu_over_cpu_packed"] = json!(gpu_best / cpu_rate);
    }
    Ok(result)
}
