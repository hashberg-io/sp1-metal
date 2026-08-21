//! BaseFold fold and sumcheck fix-last-variable, parity-checked against
//! slop-multilinear's `fold_mle` and `mle_fix_last_variable_constant_padding`.
//! These are the canonical bandwidth-bound prover primitives.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use rayon::prelude::*;
use slop_algebra::{AbstractExtensionField, AbstractField};
use slop_multilinear::{fold_mle, mle_fix_last_variable_constant_padding};
use slop_tensor::Tensor;

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::{
    ef_of_monty4, f_of_monty, monty4_of_ef, monty4_vec_of_ef, monty_of_f, monty_vec_of_f,
    rand_ef_vec, rand_f_vec, EF, F,
};
use crate::packed::{
    broadcast_ext, split_pack_base, split_pack_ext, unpack_base, unpack_ext, LANES, PEF, PF,
};
use crate::probes::{cpu_bench, times_json};

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(21);
    let n_in = 4096usize;
    let n_out = n_in / 2;
    let vals = rand_f_vec(&mut rng, n_in);
    let beta: F = rng.gen();
    let tensor = Tensor::from(vals.clone()).reshape([n_in, 1]);
    let expected = fold_mle(&tensor, beta);
    let buf_in = gpu.buffer_from(&monty_vec_of_f(&vals));
    let buf_out = gpu.buffer_empty::<u32>(n_out);
    let pso = gpu.pipeline("fold_base")?;
    let beta_m = monty_of_f(beta);
    gpu.time_encoded(1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_in), 0);
        enc.set_buffer(1, Some(&buf_out), 0);
        set_bytes(enc, 2, &beta_m);
        dispatch_1d(enc, n_out as u64);
    });
    let out: Vec<u32> = gpu.read_vec(&buf_out, n_out);
    for i in 0..n_out {
        if f_of_monty(out[i]) != expected.as_slice()[i] {
            return Err(format!("fold_base mismatch at {i}"));
        }
    }
    let evals = rand_ef_vec(&mut rng, n_in);
    let beta_e: EF = rng.gen();
    let tensor_e = Tensor::from(evals.clone()).reshape([n_in, 1]);
    let expected_e = fold_mle(&tensor_e, beta_e);
    let buf_in_e = gpu.buffer_from(&monty4_vec_of_ef(&evals));
    let buf_out_e = gpu.buffer_empty::<[u32; 4]>(n_out);
    let pso_e = gpu.pipeline("fold_ext")?;
    let beta_em = monty4_of_ef(beta_e);
    gpu.time_encoded(1, |enc| {
        enc.set_compute_pipeline_state(&pso_e);
        enc.set_buffer(0, Some(&buf_in_e), 0);
        enc.set_buffer(1, Some(&buf_out_e), 0);
        set_bytes(enc, 2, &beta_em);
        dispatch_1d(enc, n_out as u64);
    });
    let out_e: Vec<[u32; 4]> = gpu.read_vec(&buf_out_e, n_out);
    for i in 0..n_out {
        if ef_of_monty4(out_e[i]) != expected_e.as_slice()[i] {
            return Err(format!("fold_ext mismatch at {i}"));
        }
    }
    check_flv(gpu, &mut rng)?;
    check_cpu_packed(&mut rng)
}

/// The packed CPU baselines are competitors, not oracles, but a baseline that
/// computes the wrong thing would silently flatter the kernel — so they are
/// held to the same parity standard as the Metal code.
fn check_cpu_packed(rng: &mut StdRng) -> Result<(), String> {
    let n_in = 4096usize;
    let n_out = n_in / 2;
    let groups = n_out / LANES;
    let vals = rand_f_vec(rng, n_in);
    let evals = rand_ef_vec(rng, n_in);
    let beta: F = rng.gen();
    let alpha: EF = rng.gen();
    let tensor = Tensor::from(vals.clone()).reshape([n_in, 1]);
    let tensor_e = Tensor::from(evals.clone()).reshape([n_in, 1]);
    let (base_evens, base_odds) = split_pack_base(&vals);
    let (ext_evens, ext_odds) = split_pack_ext(&evals);
    let beta_p = PF::from(beta);
    let alpha_p = broadcast_ext(alpha);
    let mut out_base = vec![PF::zero(); groups];
    let mut out_ext = vec![PEF::zero(); groups];
    cpu_fold_base(&base_evens, &base_odds, &mut out_base, beta_p);
    if unpack_base(&out_base) != fold_mle(&tensor, beta).as_slice() {
        return Err("packed CPU fold_base disagrees with slop".to_string());
    }
    cpu_fold_ext(&ext_evens, &ext_odds, &mut out_ext, alpha_p);
    if unpack_ext(&out_ext) != fold_mle(&tensor_e, alpha).as_slice() {
        return Err("packed CPU fold_ext disagrees with slop".to_string());
    }
    cpu_flv_base(&base_evens, &base_odds, &mut out_ext, alpha_p);
    if unpack_ext(&out_ext)
        != mle_fix_last_variable_constant_padding(&tensor, alpha, F::zero()).as_slice()
    {
        return Err("packed CPU flv_base disagrees with slop".to_string());
    }
    cpu_flv_ext(&ext_evens, &ext_odds, &mut out_ext, alpha_p);
    if unpack_ext(&out_ext)
        != mle_fix_last_variable_constant_padding(&tensor_e, alpha, EF::zero()).as_slice()
    {
        return Err("packed CPU flv_ext disagrees with slop".to_string());
    }
    Ok(())
}

fn check_flv(gpu: &Gpu, rng: &mut StdRng) -> Result<(), String> {
    let n_in = 4096usize;
    let n_out = n_in / 2;
    let vals = rand_f_vec(rng, n_in);
    let alpha: EF = rng.gen();
    let tensor = Tensor::from(vals.clone()).reshape([n_in, 1]);
    let expected = mle_fix_last_variable_constant_padding(&tensor, alpha, F::zero());
    let buf_in = gpu.buffer_from(&monty_vec_of_f(&vals));
    let buf_out = gpu.buffer_empty::<[u32; 4]>(n_out);
    let pso = gpu.pipeline("flv_base")?;
    let alpha_m = monty4_of_ef(alpha);
    gpu.time_encoded(1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_in), 0);
        enc.set_buffer(1, Some(&buf_out), 0);
        set_bytes(enc, 2, &alpha_m);
        dispatch_1d(enc, n_out as u64);
    });
    let out: Vec<[u32; 4]> = gpu.read_vec(&buf_out, n_out);
    for i in 0..n_out {
        if ef_of_monty4(out[i]) != expected.as_slice()[i] {
            return Err(format!("flv_base mismatch at {i}"));
        }
    }
    let evals = rand_ef_vec(rng, n_in);
    let tensor_e = Tensor::from(evals.clone()).reshape([n_in, 1]);
    let expected_e = mle_fix_last_variable_constant_padding(&tensor_e, alpha, EF::zero());
    let buf_in_e = gpu.buffer_from(&monty4_vec_of_ef(&evals));
    let buf_out_e = gpu.buffer_empty::<[u32; 4]>(n_out);
    let pso_e = gpu.pipeline("flv_ext")?;
    gpu.time_encoded(1, |enc| {
        enc.set_compute_pipeline_state(&pso_e);
        enc.set_buffer(0, Some(&buf_in_e), 0);
        enc.set_buffer(1, Some(&buf_out_e), 0);
        set_bytes(enc, 2, &alpha_m);
        dispatch_1d(enc, n_out as u64);
    });
    let out_e: Vec<[u32; 4]> = gpu.read_vec(&buf_out_e, n_out);
    for i in 0..n_out {
        if ef_of_monty4(out_e[i]) != expected_e.as_slice()[i] {
            return Err(format!("flv_ext mismatch at {i}"));
        }
    }
    Ok(())
}

/// Chunk size for the packed CPU passes: large enough that rayon's per-task
/// overhead vanishes, small enough that many tasks keep every core fed.
const CPU_CHUNK: usize = 1 << 12;

/// What a CPU backend built to compete would run: packed lanes, preallocated
/// output, contiguous rayon chunks, and the interleaved pairs deinterleaved
/// once up front rather than strided over on every pass. slop's shipped
/// `fold_mle` is the other baseline; the gap between the two is CPU headroom,
/// not GPU advantage, and the probe reports both so the model cannot confuse
/// the one for the other.
fn cpu_fold_base(evens: &[PF], odds: &[PF], out: &mut [PF], beta: PF) {
    out.par_chunks_mut(CPU_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let base = ci * CPU_CHUNK;
        for (k, o) in chunk.iter_mut().enumerate() {
            *o = evens[base + k] + beta * odds[base + k];
        }
    });
}

fn cpu_fold_ext(evens: &[PEF], odds: &[PEF], out: &mut [PEF], beta: PEF) {
    out.par_chunks_mut(CPU_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let base = ci * CPU_CHUNK;
        for (k, o) in chunk.iter_mut().enumerate() {
            *o = evens[base + k] + beta * odds[base + k];
        }
    });
}

/// `alpha * (y - x) + x` with `x`, `y` in the base field: multiplying the
/// packed extension by a packed *base* value costs four packed multiplies, not
/// the sixteen a full extension multiply would.
fn cpu_flv_base(evens: &[PF], odds: &[PF], out: &mut [PEF], alpha: PEF) {
    out.par_chunks_mut(CPU_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let base = ci * CPU_CHUNK;
        for (k, o) in chunk.iter_mut().enumerate() {
            let x = evens[base + k];
            *o = alpha * (odds[base + k] - x) + PEF::from_base(x);
        }
    });
}

fn cpu_flv_ext(evens: &[PEF], odds: &[PEF], out: &mut [PEF], alpha: PEF) {
    out.par_chunks_mut(CPU_CHUNK).enumerate().for_each(|(ci, chunk)| {
        let base = ci * CPU_CHUNK;
        for (k, o) in chunk.iter_mut().enumerate() {
            let x = evens[base + k];
            *o = alpha * (odds[base + k] - x) + x;
        }
    });
}

pub fn bench(gpu: &Gpu, n_out: usize, reps: usize, with_cpu: bool) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(22);
    let n_in = 2 * n_out;
    let mut results = serde_json::Map::new();
    let base_vals = rand_f_vec(&mut rng, n_in);
    let base_in = monty_vec_of_f(&base_vals);
    let beta: F = rng.gen();
    let alpha: EF = rng.gen();
    let beta_m = monty_of_f(beta);
    let alpha_m = monty4_of_ef(alpha);
    let buf_base_in = gpu.buffer_from(&base_in);
    let buf_base_out = gpu.buffer_empty::<u32>(n_out);
    let buf_ext4_out = gpu.buffer_empty::<[u32; 4]>(n_out);
    let pso_fold = gpu.pipeline("fold_base")?;
    let t = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso_fold);
        enc.set_buffer(0, Some(&buf_base_in), 0);
        enc.set_buffer(1, Some(&buf_base_out), 0);
        set_bytes(enc, 2, &beta_m);
        dispatch_1d(enc, n_out as u64);
    });
    results.insert(
        "fold_base".into(),
        json!({ "times": times_json(&t), "gbytes_per_s_best": (n_in * 4 + n_out * 4) as f64 / t.best / 1e9 }),
    );
    let pso_flv = gpu.pipeline("flv_base")?;
    let t = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso_flv);
        enc.set_buffer(0, Some(&buf_base_in), 0);
        enc.set_buffer(1, Some(&buf_ext4_out), 0);
        set_bytes(enc, 2, &alpha_m);
        dispatch_1d(enc, n_out as u64);
    });
    results.insert(
        "flv_base".into(),
        json!({ "times": times_json(&t), "gbytes_per_s_best": (n_in * 4 + n_out * 16) as f64 / t.best / 1e9 }),
    );
    let ext_vals = rand_ef_vec(&mut rng, n_in);
    let ext_in = monty4_vec_of_ef(&ext_vals);
    let buf_ext_in = gpu.buffer_from(&ext_in);
    let pso_folde = gpu.pipeline("fold_ext")?;
    let t = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso_folde);
        enc.set_buffer(0, Some(&buf_ext_in), 0);
        enc.set_buffer(1, Some(&buf_ext4_out), 0);
        set_bytes(enc, 2, &alpha_m);
        dispatch_1d(enc, n_out as u64);
    });
    results.insert(
        "fold_ext".into(),
        json!({ "times": times_json(&t), "gbytes_per_s_best": (n_in * 16 + n_out * 16) as f64 / t.best / 1e9 }),
    );
    let pso_flve = gpu.pipeline("flv_ext")?;
    let t = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso_flve);
        enc.set_buffer(0, Some(&buf_ext_in), 0);
        enc.set_buffer(1, Some(&buf_ext4_out), 0);
        set_bytes(enc, 2, &alpha_m);
        dispatch_1d(enc, n_out as u64);
    });
    results.insert(
        "flv_ext".into(),
        json!({ "times": times_json(&t), "gbytes_per_s_best": (n_in * 16 + n_out * 16) as f64 / t.best / 1e9 }),
    );
    if with_cpu {
        let base_tensor = Tensor::from(base_vals).reshape([n_in, 1]);
        let ext_tensor = Tensor::from(ext_vals).reshape([n_in, 1]);
        let mut cpu = serde_json::Map::new();
        let mut record = |results: &mut serde_json::Map<String, Value>,
                          name: &str,
                          t: &crate::gpu::BenchTimes,
                          bytes: usize| {
            let gpu_best = results[name]["times"]["best_s"].as_f64().expect("gpu time recorded");
            let ratio = t.best / gpu_best;
            results[name]["gpu_over_cpu"] = json!(ratio);
            cpu.insert(
                name.into(),
                json!({ "times": times_json(t), "gbytes_per_s_best": bytes as f64 / t.best / 1e9 }),
            );
        };
        let t = cpu_bench(reps, || {
            std::hint::black_box(fold_mle(&base_tensor, beta));
        });
        record(&mut results, "fold_base", &t, n_in * 4 + n_out * 4);
        let t = cpu_bench(reps, || {
            std::hint::black_box(mle_fix_last_variable_constant_padding(
                &base_tensor,
                alpha,
                F::zero(),
            ));
        });
        record(&mut results, "flv_base", &t, n_in * 4 + n_out * 16);
        let t = cpu_bench(reps, || {
            std::hint::black_box(fold_mle(&ext_tensor, alpha));
        });
        record(&mut results, "fold_ext", &t, n_in * 16 + n_out * 16);
        let t = cpu_bench(reps, || {
            std::hint::black_box(mle_fix_last_variable_constant_padding(
                &ext_tensor,
                alpha,
                EF::zero(),
            ));
        });
        record(&mut results, "flv_ext", &t, n_in * 16 + n_out * 16);
        drop(record);
        let (base_evens, base_odds) = split_pack_base(base_tensor.as_slice());
        let (ext_evens, ext_odds) = split_pack_ext(ext_tensor.as_slice());
        let beta_p = PF::from(beta);
        let alpha_p = broadcast_ext(alpha);
        let groups = n_out / LANES;
        let mut out_base = vec![PF::zero(); groups];
        let mut out_ext = vec![PEF::zero(); groups];
        let mut tuned = serde_json::Map::new();
        let mut record_packed = |results: &mut serde_json::Map<String, Value>,
                                 name: &str,
                                 t: &crate::gpu::BenchTimes,
                                 bytes: usize| {
            let gpu_best = results[name]["times"]["best_s"].as_f64().expect("gpu time recorded");
            results[name]["gpu_over_cpu_packed"] = json!(t.best / gpu_best);
            tuned.insert(
                name.into(),
                json!({ "times": times_json(t), "gbytes_per_s_best": bytes as f64 / t.best / 1e9 }),
            );
        };
        let t = cpu_bench(reps, || cpu_fold_base(&base_evens, &base_odds, &mut out_base, beta_p));
        record_packed(&mut results, "fold_base", &t, n_in * 4 + n_out * 4);
        let t = cpu_bench(reps, || cpu_flv_base(&base_evens, &base_odds, &mut out_ext, alpha_p));
        record_packed(&mut results, "flv_base", &t, n_in * 4 + n_out * 16);
        let t = cpu_bench(reps, || cpu_fold_ext(&ext_evens, &ext_odds, &mut out_ext, alpha_p));
        record_packed(&mut results, "fold_ext", &t, n_in * 16 + n_out * 16);
        let t = cpu_bench(reps, || cpu_flv_ext(&ext_evens, &ext_odds, &mut out_ext, alpha_p));
        record_packed(&mut results, "flv_ext", &t, n_in * 16 + n_out * 16);
        drop(record_packed);
        return Ok(json!({
            "probe": "fold",
            "params": { "n_out": n_out, "n_in": n_in, "reps": reps },
            "kernels": results,
            "cpu": {
                "kernels": cpu,
                "note": "slop's own CpuBackend fold_mle / mle_fix_last_variable_constant_padding, rayon on all cores",
            },
            "cpu_packed": {
                "kernels": tuned,
                "note": "packed (NEON) lanes, deinterleaved once up front, contiguous rayon chunks",
            },
        }));
    }
    Ok(json!({
        "probe": "fold",
        "params": { "n_out": n_out, "n_in": n_in, "reps": reps },
        "kernels": results,
    }))
}
