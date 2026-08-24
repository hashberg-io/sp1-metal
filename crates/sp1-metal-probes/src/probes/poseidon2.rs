// SPDX-License-Identifier: MIT OR Apache-2.0
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
use slop_algebra::{Field, PackedValue};
use slop_symmetric::Permutation;

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::{f_of_monty, monty_of_f, F};
use crate::probes::{cpu_bench, times_json};

/// Round constants in kernel layout: 4 beginning full rounds (16 each), then
/// 4 ending full rounds (16 each), then 20 partial-round constants; all in
/// Montgomery form.
pub fn rc_monty() -> Vec<u32> {
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
        let t = cpu_bench(reps, || {
            cpu_states.par_iter_mut().for_each(|s| *s = perm.permute(*s));
        });
        // A single-thread pass over a slice large enough to be steady-state but
        // small enough to stay cheap: this separates "how fast is one core" from
        // "how well does rayon scale", which the all-core figure conflates.
        let n_serial = n.min(1 << 17);
        let mut serial_states = cpu_states[..n_serial].to_vec();
        let ts = cpu_bench(reps, || {
            for s in serial_states.iter_mut() {
                *s = perm.permute(*s);
            }
        });
        result["cpu"] = json!({
            "times": times_json(&t),
            "mega_perms_per_s_best": n as f64 / t.best / 1e6,
            "single_thread": {
                "n_perms": n_serial,
                "times": times_json(&ts),
                "mega_perms_per_s_best": n_serial as f64 / ts.best / 1e6,
            },
            "note": "slop my_kb_16_perm, rayon over all cores; single_thread is one core",
        });
        result["gpu_over_cpu"] = json!(t.best / times.best);
        // p3's Poseidon2 permutes `[AF; 16]` for any abstract field, so the packed
        // (NEON) field permutes WIDTH states at once — this is how the CPU prover
        // actually hashes, and the only fair opponent for the kernel.
        type P = <F as Field>::Packing;
        let lanes = P::WIDTH;
        let n_groups = n / lanes;
        let mut packed: Vec<[P; 16]> = (0..n_groups)
            .map(|g| std::array::from_fn(|i| P::from_fn(|lane| cpu_states[g * lanes + lane][i])))
            .collect();
        let tp = cpu_bench(reps, || {
            packed.par_iter_mut().for_each(|s| perm.permute_mut(s));
        });
        result["cpu_packed"] = json!({
            "times": times_json(&tp),
            "lanes": lanes,
            "n_perms": n_groups * lanes,
            "mega_perms_per_s_best": (n_groups * lanes) as f64 / tp.best / 1e6,
            "note": "same slop permutation over KoalaBear::Packing (NEON), rayon on all cores",
        });
        result["gpu_over_cpu_packed"] = json!(tp.best / times.best);
    }
    Ok(result)
}
