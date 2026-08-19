//! Poseidon2 width-16 over KoalaBear, parity-checked against slop's
//! `my_kb_16_perm` permutation (the one the Merkle/hashing layers use).
//! This is the canonical compute-bound prover primitive.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::{json, Value};
use slop_koala_bear::{
    my_kb_16_perm, KoalaBear_BEGIN_EXT_CONSTS, KoalaBear_END_EXT_CONSTS, KoalaBear_PARTIAL_CONSTS,
};
use slop_symmetric::Permutation;

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::{f_of_monty, monty_of_f, F};
use crate::probes::times_json;

/// Round constants in kernel layout: 4 beginning full rounds (16 each), then
/// 4 ending full rounds (16 each), then 20 partial-round constants; all in
/// Montgomery form.
fn rc_monty() -> Vec<u32> {
    let mut rc = Vec::with_capacity(8 * 16 + 20);
    for round in KoalaBear_BEGIN_EXT_CONSTS.iter() {
        rc.extend(round.iter().map(|&c| monty_of_f(c)));
    }
    for round in KoalaBear_END_EXT_CONSTS.iter() {
        rc.extend(round.iter().map(|&c| monty_of_f(c)));
    }
    rc.extend(KoalaBear_PARTIAL_CONSTS.iter().map(|&c| monty_of_f(c)));
    rc
}

fn rand_states(rng: &mut StdRng, n: usize) -> Vec<[F; 16]> {
    (0..n).map(|_| std::array::from_fn(|_| rng.gen())).collect()
}

fn states_to_monty(states: &[[F; 16]]) -> Vec<u32> {
    states.iter().flat_map(|s| s.iter().map(|&x| monty_of_f(x))).collect()
}

fn run_gpu(gpu: &Gpu, input_monty: &[u32], n: usize, reps: usize) -> Result<(Vec<u32>, crate::gpu::BenchTimes), String> {
    let buf_in = gpu.buffer_from(input_monty);
    let buf_out = gpu.buffer_empty::<u32>(n * 16);
    let buf_rc = gpu.buffer_from(&rc_monty());
    let pso = gpu.pipeline("poseidon2_16")?;
    let n32 = n as u32;
    let times = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_in), 0);
        enc.set_buffer(1, Some(&buf_out), 0);
        enc.set_buffer(2, Some(&buf_rc), 0);
        set_bytes(enc, 3, &n32);
        dispatch_1d(enc, n as u64);
    });
    Ok((gpu.read_vec(&buf_out, n * 16), times))
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(31);
    let n = 256usize;
    let states = rand_states(&mut rng, n);
    let (out, _) = run_gpu(gpu, &states_to_monty(&states), n, 1)?;
    let perm = my_kb_16_perm();
    for (i, s) in states.iter().enumerate() {
        let expected = perm.permute(*s);
        for j in 0..16 {
            if f_of_monty(out[i * 16 + j]) != expected[j] {
                return Err(format!("poseidon2_16 mismatch at state {i} lane {j}"));
            }
        }
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n: usize, reps: usize, with_cpu: bool) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(32);
    let states = rand_states(&mut rng, n);
    let input = states_to_monty(&states);
    let (_, times) = run_gpu(gpu, &input, n, reps)?;
    let mut result = json!({
        "probe": "poseidon2",
        "params": { "n_perms": n, "reps": reps },
        "times": times_json(&times),
        "mega_perms_per_s_best": n as f64 / times.best / 1e6,
    });
    if with_cpu {
        let perm = my_kb_16_perm();
        let mut cpu_states = states;
        let t0 = std::time::Instant::now();
        cpu_states.par_iter_mut().for_each(|s| *s = perm.permute(*s));
        let cpu_s = t0.elapsed().as_secs_f64();
        result["cpu"] = json!({
            "seconds": cpu_s,
            "mega_perms_per_s": n as f64 / cpu_s / 1e6,
            "note": "rayon over all cores via slop my_kb_16_perm",
        });
    }
    Ok(result)
}
