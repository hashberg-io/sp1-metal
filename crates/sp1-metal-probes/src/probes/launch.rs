//! What a kernel costs before it computes anything.
//!
//! The Hypercube prover's sumcheck and BaseFold rounds halve their working set
//! every round, and each round's challenge is derived from the previous round's
//! output — so the rounds cannot be batched into one submission, and the tail of
//! a fold is a sequence of ever-smaller kernels each paying a full host round
//! trip. This probe prices that fixed cost three ways: the marginal cost of one
//! more dispatch inside a command buffer, the cost of a command buffer of one's
//! own, and the end-to-end cost of folding a vector all the way down to a single
//! element one round trip at a time.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use slop_multilinear::fold_mle;
use slop_tensor::Tensor;

use crate::gpu::{dispatch_1d, set_bytes, Gpu};
use crate::kb::{f_of_monty, monty_of_f, monty_vec_of_f, rand_f_vec, F};
use crate::probes::times_json;

/// Dispatches encoded into one command buffer when measuring marginal cost.
const MANY: usize = 512;

/// Command buffers submitted one at a time when measuring round-trip cost.
const TRIPS: usize = 200;

/// Encodes round `round` of a fold that halves every round. Buffers alternate:
/// round `r` reads the buffer it did not write, so the final value lands in
/// `bufs[rounds % 2]`.
fn encode_round(
    enc: &metal::ComputeCommandEncoderRef,
    pso: &metal::ComputePipelineState,
    bufs: &[metal::Buffer; 2],
    n: usize,
    betas: &[u32],
    round: usize,
) {
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(&bufs[round % 2]), 0);
    enc.set_buffer(1, Some(&bufs[(round + 1) % 2]), 0);
    set_bytes(enc, 2, &betas[round]);
    dispatch_1d(enc, (n >> (round + 1)) as u64);
}

/// Folds all the way down to a single element, one command buffer per round —
/// the shape the prover is forced into when each round's challenge comes from
/// the previous round's output. Runs exactly once, so the answer is the fold of
/// the data actually uploaded.
fn fold_to_scalar(
    gpu: &Gpu,
    bufs: &[metal::Buffer; 2],
    n: usize,
    betas: &[u32],
) -> Result<u32, String> {
    let pso = gpu.pipeline("fold_base")?;
    let rounds = n.trailing_zeros() as usize;
    gpu.time_round_trips(rounds, |enc, round| encode_round(enc, &pso, bufs, n, betas, round));
    Ok(gpu.read_vec::<u32>(&bufs[rounds % 2], 1)[0])
}

fn setup(gpu: &Gpu, vals: &[F]) -> [metal::Buffer; 2] {
    [gpu.buffer_from(&monty_vec_of_f(vals)), gpu.buffer_empty::<u32>(vals.len())]
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(51);
    let n = 1024usize;
    let vals = rand_f_vec(&mut rng, n);
    let betas: Vec<F> = (0..n.trailing_zeros()).map(|_| rng.gen()).collect();
    let betas_m: Vec<u32> = betas.iter().copied().map(monty_of_f).collect();
    let bufs = setup(gpu, &vals);
    let got = fold_to_scalar(gpu, &bufs, n, &betas_m)?;
    let mut tensor = Tensor::from(vals).reshape([n, 1]);
    for &beta in &betas {
        tensor = fold_mle(&tensor, beta);
    }
    let expected = tensor.as_slice()[0];
    if f_of_monty(got) != expected {
        return Err(format!("fold tail mismatch: got {:?}, expected {expected:?}", f_of_monty(got)));
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n: usize, reps: usize) -> Result<Value, String> {
    let pso = gpu.pipeline("noop")?;
    let counter = gpu.buffer_empty::<u32>(1);
    let one = gpu.bench_encoded(reps, 1, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&counter), 0);
        dispatch_1d(enc, 1);
    });
    let many = gpu.bench_encoded(reps, MANY, |enc| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&counter), 0);
        dispatch_1d(enc, 1);
    });
    let marginal_us = (many.best - one.best) / (MANY - 1) as f64 * 1e6;
    let trips = gpu.bench_round_trips(reps, TRIPS, |enc, _| {
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&counter), 0);
        dispatch_1d(enc, 1);
    });
    let round_trip_us = trips.best / TRIPS as f64 * 1e6;

    let mut rng = StdRng::seed_from_u64(52);
    let vals = rand_f_vec(&mut rng, n);
    let betas: Vec<u32> = (0..n.trailing_zeros()).map(|_| monty_of_f(rng.gen::<F>())).collect();
    let bufs = setup(gpu, &vals);
    let rounds = n.trailing_zeros() as usize;
    let fold_pso = gpu.pipeline("fold_base")?;
    let tail = gpu.bench_round_trips(reps, rounds, |enc, round| {
        encode_round(enc, &fold_pso, &bufs, n, &betas, round)
    });
    // The same rounds with no host in the loop: what the tail would cost if the
    // challenges were derived on the GPU, so every round could be encoded into
    // one submission. The difference between the two is what a GPU-side
    // challenger would be worth.
    let round = std::cell::Cell::new(0usize);
    let batched = gpu.bench_encoded(reps, rounds, |enc| {
        let r = round.get();
        encode_round(enc, &fold_pso, &bufs, n, &betas, r % rounds);
        round.set(r + 1);
    });
    // What the same rounds would cost if only the data mattered: every round
    // reads 3/2 of its output in bytes, at the streaming rate the fold probe
    // measures. Anything above this is fixed cost.
    let bytes: f64 = (0..rounds).map(|r| (3 * (n >> (r + 1)) * 4) as f64).sum();

    Ok(json!({
        "probe": "launch",
        "params": { "n": n, "rounds": rounds, "reps": reps, "dispatches": MANY, "round_trips": TRIPS },
        "single_dispatch_us": one.best * 1e6,
        "marginal_dispatch_us": marginal_us,
        "round_trip_us": round_trip_us,
        "fold_tail": {
            "times": times_json(&tail),
            "total_ms": tail.best * 1e3,
            "per_round_us": tail.best / rounds as f64 * 1e6,
            "bytes_moved": bytes,
            "effective_gbytes_per_s": bytes / tail.best / 1e9,
        },
        "fold_tail_batched": {
            "times": times_json(&batched),
            "total_ms": batched.best * 1e3,
            "per_round_us": batched.best / rounds as f64 * 1e6,
            "effective_gbytes_per_s": bytes / batched.best / 1e9,
            "speedup_over_round_trips": tail.best / batched.best,
            "note": "all rounds in one command buffer: the ceiling a GPU-side challenger would unlock",
        },
    }))
}
