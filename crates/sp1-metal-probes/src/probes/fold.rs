//! BaseFold fold and sumcheck fix-last-variable, parity-checked against
//! slop-multilinear's `fold_mle` and `mle_fix_last_variable_constant_padding`.
//! These are the canonical bandwidth-bound prover primitives.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use slop_algebra::AbstractField;
use slop_multilinear::{fold_mle, mle_fix_last_variable_constant_padding};
use slop_tensor::Tensor;

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::{
    ef_of_monty4, f_of_monty, monty4_of_ef, monty4_vec_of_ef, monty_of_f, monty_vec_of_f,
    rand_ef_vec, rand_f_vec, EF, F,
};
use crate::probes::times_json;

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
    check_flv(gpu, &mut rng)
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

pub fn bench(gpu: &Gpu, n_out: usize, reps: usize) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(22);
    let n_in = 2 * n_out;
    let mut results = serde_json::Map::new();
    let base_in = monty_vec_of_f(&rand_f_vec(&mut rng, n_in));
    let beta_m = monty_of_f(rng.gen::<F>());
    let alpha_m = monty4_of_ef(rng.gen::<EF>());
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
    let ext_in = monty4_vec_of_ef(&rand_ef_vec(&mut rng, n_in));
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
    Ok(json!({
        "probe": "fold",
        "params": { "n_out": n_out, "n_in": n_in, "reps": reps },
        "kernels": results,
    }))
}
