//! Streamed elementwise field multiplication, base and degree-4 extension,
//! parity-checked against slop's field types.

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};

use crate::gpu::{dispatch_1d, Gpu};
use crate::kb::{ef_of_monty4, f_of_monty, monty4_vec_of_ef, monty_vec_of_f, rand_ef_vec, rand_f_vec};
use crate::probes::times_json;

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(11);
    let n = 4096usize;
    let a = rand_f_vec(&mut rng, n);
    let b = rand_f_vec(&mut rng, n);
    let buf_a = gpu.buffer_from(&monty_vec_of_f(&a));
    let buf_b = gpu.buffer_from(&monty_vec_of_f(&b));
    let buf_c = gpu.buffer_empty::<u32>(n);
    let pso = gpu.pipeline("kb_mul_elems")?;
    gpu.time_encoded(1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        dispatch_1d(enc, n as u64);
    });
    let c: Vec<u32> = gpu.read_vec(&buf_c, n);
    for i in 0..n {
        if f_of_monty(c[i]) != a[i] * b[i] {
            return Err(format!("kb_mul_elems mismatch at {i}"));
        }
    }
    let ne = 1024usize;
    let ea = rand_ef_vec(&mut rng, ne);
    let eb = rand_ef_vec(&mut rng, ne);
    let buf_ea = gpu.buffer_from(&monty4_vec_of_ef(&ea));
    let buf_eb = gpu.buffer_from(&monty4_vec_of_ef(&eb));
    let buf_ec = gpu.buffer_empty::<[u32; 4]>(ne);
    let pso_e = gpu.pipeline("ext_mul_elems")?;
    gpu.time_encoded(1, |enc| {
        enc.set_compute_pipeline_state(&pso_e);
        enc.set_buffer(0, Some(&buf_ea), 0);
        enc.set_buffer(1, Some(&buf_eb), 0);
        enc.set_buffer(2, Some(&buf_ec), 0);
        dispatch_1d(enc, ne as u64);
    });
    let ec: Vec<[u32; 4]> = gpu.read_vec(&buf_ec, ne);
    for i in 0..ne {
        if ef_of_monty4(ec[i]) != ea[i] * eb[i] {
            return Err(format!("ext_mul_elems mismatch at {i}"));
        }
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n: usize, reps: usize) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(12);
    let a = monty_vec_of_f(&rand_f_vec(&mut rng, n));
    let b = monty_vec_of_f(&rand_f_vec(&mut rng, n));
    let buf_a = gpu.buffer_from(&a);
    let buf_b = gpu.buffer_from(&b);
    let buf_c = gpu.buffer_empty::<u32>(n);
    let pso = gpu.pipeline("kb_mul_elems")?;
    let times_base = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        dispatch_1d(enc, n as u64);
    });
    let ne = n / 4;
    let ea = monty4_vec_of_ef(&rand_ef_vec(&mut rng, ne));
    let eb = monty4_vec_of_ef(&rand_ef_vec(&mut rng, ne));
    let buf_ea = gpu.buffer_from(&ea);
    let buf_eb = gpu.buffer_from(&eb);
    let buf_ec = gpu.buffer_empty::<[u32; 4]>(ne);
    let pso_e = gpu.pipeline("ext_mul_elems")?;
    let times_ext = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso_e);
        enc.set_buffer(0, Some(&buf_ea), 0);
        enc.set_buffer(1, Some(&buf_eb), 0);
        enc.set_buffer(2, Some(&buf_ec), 0);
        dispatch_1d(enc, ne as u64);
    });
    Ok(json!({
        "probe": "fieldops",
        "params": { "n_base": n, "n_ext": ne, "reps": reps },
        "base_mul": {
            "times": times_json(&times_base),
            "giga_muls_per_s_best": n as f64 / times_base.best / 1e9,
            "gbytes_per_s_best": (3 * n * 4) as f64 / times_base.best / 1e9,
        },
        "ext_mul": {
            "times": times_json(&times_ext),
            "giga_ext_muls_per_s_best": ne as f64 / times_ext.best / 1e9,
            "gbytes_per_s_best": (3 * ne * 16) as f64 / times_ext.best / 1e9,
        },
    }))
}
