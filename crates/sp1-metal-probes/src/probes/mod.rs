pub mod fieldops;
pub mod fold;
pub mod imul;
pub mod launch;
pub mod poseidon2;
pub mod triad;
pub mod zerocheck;

use serde_json::{json, Value};

use crate::gpu::BenchTimes;

pub fn times_json(t: &BenchTimes) -> Value {
    json!({ "best_s": t.best, "mean_s": t.mean, "all_s": t.times })
}

/// Times a CPU closure the same way `Gpu::bench_encoded` times a kernel: one
/// untimed warmup pass, then `reps` timed passes, reported best and mean. The
/// warmup matters for the rayon baselines, whose first call also pays thread
/// pool startup.
pub fn cpu_bench<F: FnMut()>(reps: usize, mut f: F) -> BenchTimes {
    f();
    let times: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = std::time::Instant::now();
            f();
            t0.elapsed().as_secs_f64()
        })
        .collect();
    let best = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
    BenchTimes { best, mean, times }
}
