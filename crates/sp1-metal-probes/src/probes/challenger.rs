//! The GPU-side challenger, and what removing the host from the round loop is
//! worth.
//!
//! Every Fiat-Shamir challenge in SP1 comes from a duplex sponge over the
//! width-16 Poseidon2 permutation: `SP1GlobalContext` is
//! `KoalaBearDegree4Duplex`, whose challenger is
//! `DuplexChallenger<KoalaBear, KoalaPerm, 16, 8>`. Because each sumcheck round
//! observes the round message and samples the next challenge, a challenger that
//! lives on the host forces a command-buffer round trip per round, and the
//! launch probe measured that at 87 us against a 1.6 us marginal dispatch.
//!
//! This probe runs the sponge on the device, parity-checked against slop's own
//! challenger, and then times a complete BaseFold-shaped fold both ways: with
//! the host deriving each challenge, and with every round encoded into one
//! submission because the challenge never leaves the GPU.

use std::cell::Cell;

use metal::MTLSize;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};
use slop_algebra::AbstractField;
use slop_challenger::{CanObserve, FieldChallenger, IopCtx};
use slop_koala_bear::KoalaBearDegree4Duplex;

use crate::gpu::{dispatch_1d, dispatch_groups, set_bytes, BenchTimes, Gpu, THREADGROUP_SIZE};
use crate::kb::{ef_of_monty4, monty4_of_ef, monty4_vec_of_ef, monty_vec_of_f, rand_ef_vec, rand_f_vec, EF};
use crate::probes::poseidon2::rc_monty;
use crate::probes::times_json;

/// Independent folds kept in flight when measuring what concurrency alone buys.
const CONCURRENT: usize = 4;

/// Words in the device-side `Challenger` struct: 16 sponge, 8 input, 8 output,
/// and the two buffer lengths.
const CHALLENGER_WORDS: usize = 34;

/// Buffers a fold with a device-side challenger needs, beyond the data itself.
struct FoldCtx {
    data: [metal::Buffer; 2],
    partials: metal::Buffer,
    msg: metal::Buffer,
    beta: metal::Buffer,
    chal: metal::Buffer,
    rc: metal::Buffer,
}

impl FoldCtx {
    fn new(gpu: &Gpu, vals: &[EF], rounds: usize) -> Self {
        let n = vals.len();
        let groups = n.div_ceil(2 * THREADGROUP_SIZE as usize);
        Self {
            data: [gpu.buffer_from(&monty4_vec_of_ef(vals)), gpu.buffer_empty::<[u32; 4]>(n / 2)],
            partials: gpu.buffer_empty::<[u32; 4]>(2 * groups.max(1)),
            msg: gpu.buffer_empty::<[u32; 4]>(2 * rounds),
            beta: gpu.buffer_empty::<[u32; 4]>(rounds),
            chal: gpu.buffer_empty::<u32>(CHALLENGER_WORDS),
            rc: gpu.buffer_from(&rc_monty()),
        }
    }

    fn betas(&self, gpu: &Gpu, rounds: usize) -> Vec<EF> {
        gpu.read_vec::<[u32; 4]>(&self.beta, rounds).into_iter().map(ef_of_monty4).collect()
    }

    fn result(&self, gpu: &Gpu, rounds: usize) -> EF {
        ef_of_monty4(gpu.read_vec::<[u32; 4]>(&self.data[rounds % 2], 1)[0])
    }
}

/// The pipelines one fold round needs.
struct FoldPsos {
    partials: metal::ComputePipelineState,
    reduce: metal::ComputePipelineState,
    challenge: metal::ComputePipelineState,
    fold: metal::ComputePipelineState,
}

impl FoldPsos {
    fn new(gpu: &Gpu) -> Result<Self, String> {
        Ok(Self {
            partials: gpu.pipeline("sc_partials")?,
            reduce: gpu.pipeline("sc_reduce")?,
            challenge: gpu.pipeline("sc_challenge")?,
            fold: gpu.pipeline("fold_ext_dev")?,
        })
    }
}

/// Dispatches one threadgroup: the reduction of the per-group partials and the
/// sponge itself are sequential, so they run in a single group.
fn dispatch_one_group(enc: &metal::ComputeCommandEncoderRef, threads: u64) {
    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(threads, 1, 1));
}

/// Encodes the reduction that forms round `round`'s message from the current
/// working set. Shared by both paths, so they differ only in where the
/// challenge comes from.
fn encode_message(
    enc: &metal::ComputeCommandEncoderRef,
    psos: &FoldPsos,
    ctx: &FoldCtx,
    n: usize,
    round: usize,
) {
    let half = (n >> round) / 2;
    let groups = half.div_ceil(THREADGROUP_SIZE as usize);
    enc.set_compute_pipeline_state(&psos.partials);
    enc.set_buffer(0, Some(&ctx.data[round % 2]), 0);
    enc.set_buffer(1, Some(&ctx.partials), 0);
    set_bytes(enc, 2, &(half as u32));
    dispatch_groups(enc, groups as u64);
    enc.set_compute_pipeline_state(&psos.reduce);
    enc.set_buffer(0, Some(&ctx.partials), 0);
    enc.set_buffer(1, Some(&ctx.msg), (round * 32) as u64);
    set_bytes(enc, 2, &(groups as u32));
    dispatch_one_group(enc, THREADGROUP_SIZE);
}

/// Encodes the fold of round `round`, reading the challenge from device memory.
fn encode_fold(
    enc: &metal::ComputeCommandEncoderRef,
    psos: &FoldPsos,
    ctx: &FoldCtx,
    n: usize,
    round: usize,
) {
    let half = (n >> round) / 2;
    enc.set_compute_pipeline_state(&psos.fold);
    enc.set_buffer(0, Some(&ctx.data[round % 2]), 0);
    enc.set_buffer(1, Some(&ctx.data[(round + 1) % 2]), 0);
    enc.set_buffer(2, Some(&ctx.beta), (round * 16) as u64);
    dispatch_1d(enc, half as u64);
}

/// Encodes one complete round with the challenge derived on the device: reduce,
/// observe and sample in the sponge, then fold. No host involvement, so every
/// round of a fold can be encoded before any of them has run.
fn encode_round_gpu(
    enc: &metal::ComputeCommandEncoderRef,
    psos: &FoldPsos,
    ctx: &FoldCtx,
    n: usize,
    round: usize,
) {
    encode_message(enc, psos, ctx, n, round);
    enc.set_compute_pipeline_state(&psos.challenge);
    enc.set_buffer(0, Some(&ctx.msg), (round * 32) as u64);
    enc.set_buffer(1, Some(&ctx.chal), 0);
    enc.set_buffer(2, Some(&ctx.beta), (round * 16) as u64);
    enc.set_buffer(3, Some(&ctx.rc), 0);
    dispatch_one_group(enc, 1);
    encode_fold(enc, psos, ctx, n, round);
}

fn encode_init(enc: &metal::ComputeCommandEncoderRef, pso: &metal::ComputePipelineState, ctx: &FoldCtx) {
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(&ctx.chal), 0);
    dispatch_one_group(enc, 1);
}

/// Folds all the way down to a scalar with the challenger on the device: one
/// command buffer for the whole fold.
fn fold_on_gpu(gpu: &Gpu, ctx: &FoldCtx, n: usize, rounds: usize) -> Result<f64, String> {
    let psos = FoldPsos::new(gpu)?;
    let init = gpu.pipeline("chal_init")?;
    Ok(gpu.time_encoded(rounds + 1, {
        let round = Cell::new(0usize);
        move |enc| {
            let r = round.get();
            if r == 0 {
                encode_init(enc, &init, ctx);
            } else {
                encode_round_gpu(enc, &psos, ctx, n, r - 1);
            }
            round.set(r + 1);
        }
    }))
}

/// Folds all the way down to a scalar with the challenger on the host: the
/// shape a port without a device-side sponge is forced into. Each round's fold
/// is encoded together with the next round's reduction, so this is the best a
/// host-driven implementation can do — one round trip per round, not two.
fn fold_on_host(gpu: &Gpu, ctx: &FoldCtx, n: usize, rounds: usize) -> Result<(f64, Vec<EF>), String> {
    let psos = FoldPsos::new(gpu)?;
    let mut challenger = KoalaBearDegree4Duplex::default_challenger();
    let mut betas = Vec::with_capacity(rounds);
    let t0 = std::time::Instant::now();
    for round in 0..rounds {
        let cb = gpu.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        if round > 0 {
            encode_fold(enc, &psos, ctx, n, round - 1);
        }
        encode_message(enc, &psos, ctx, n, round);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let msg = gpu.read_vec::<[u32; 4]>(&ctx.msg, 2 * rounds);
        let g0 = ef_of_monty4(msg[2 * round]);
        let g1 = ef_of_monty4(msg[2 * round + 1]);
        challenger.observe_ext_element(g0);
        challenger.observe_ext_element(g1);
        let beta: EF = challenger.sample_ext_element();
        betas.push(beta);
        let words = monty4_of_ef(beta);
        let ptr = ctx.beta.contents().cast::<u32>();
        assert!(!ptr.is_null(), "beta buffer has no CPU-visible contents");
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), ptr.add(4 * round), 4) };
    }
    let cb = gpu.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    encode_fold(enc, &psos, ctx, n, rounds - 1);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok((t0.elapsed().as_secs_f64(), betas))
}

/// The device-challenger fold with several folds in flight: each fold is
/// already a single submission, so this simply commits them all before waiting.
fn fold_on_gpu_concurrent(
    gpu: &Gpu,
    ctxs: &[FoldCtx],
    n: usize,
    rounds: usize,
) -> Result<f64, String> {
    let psos = FoldPsos::new(gpu)?;
    let init = gpu.pipeline("chal_init")?;
    let t0 = std::time::Instant::now();
    let buffers: Vec<_> = ctxs
        .iter()
        .map(|ctx| {
            let cb = gpu.queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder();
            encode_init(enc, &init, ctx);
            enc.end_encoding();
            for round in 0..rounds {
                let enc = cb.new_compute_command_encoder();
                encode_round_gpu(enc, &psos, ctx, n, round);
                enc.end_encoding();
            }
            cb.commit();
            cb
        })
        .collect();
    for cb in &buffers {
        cb.wait_until_completed();
    }
    Ok(t0.elapsed().as_secs_f64())
}

/// Folds `k` independent working sets with the challenger on the host, but with
/// all `k` command buffers of a round in flight before any of them is waited
/// on. The prover always has many folds available — 79 shard proofs went
/// through a queue that never drained, and each shard opens many polynomials —
/// so the round-trip latency is a per-round cost for the whole batch rather
/// than per fold. This measures whether concurrency alone buys what a
/// device-side sponge was supposed to buy.
fn fold_on_host_concurrent(
    gpu: &Gpu,
    ctxs: &[FoldCtx],
    n: usize,
    rounds: usize,
) -> Result<f64, String> {
    let psos = FoldPsos::new(gpu)?;
    let mut challengers: Vec<_> =
        (0..ctxs.len()).map(|_| KoalaBearDegree4Duplex::default_challenger()).collect();
    let t0 = std::time::Instant::now();
    for round in 0..rounds {
        let buffers: Vec<_> = ctxs
            .iter()
            .map(|ctx| {
                let cb = gpu.queue.new_command_buffer().to_owned();
                let enc = cb.new_compute_command_encoder();
                if round > 0 {
                    encode_fold(enc, &psos, ctx, n, round - 1);
                }
                encode_message(enc, &psos, ctx, n, round);
                enc.end_encoding();
                cb.commit();
                cb
            })
            .collect();
        for cb in &buffers {
            cb.wait_until_completed();
        }
        for (ctx, challenger) in ctxs.iter().zip(challengers.iter_mut()) {
            let msg = gpu.read_vec::<[u32; 4]>(&ctx.msg, 2 * rounds);
            challenger.observe_ext_element(ef_of_monty4(msg[2 * round]));
            challenger.observe_ext_element(ef_of_monty4(msg[2 * round + 1]));
            let beta: EF = challenger.sample_ext_element();
            let words = monty4_of_ef(beta);
            let ptr = ctx.beta.contents().cast::<u32>();
            assert!(!ptr.is_null(), "beta buffer has no CPU-visible contents");
            unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), ptr.add(4 * round), 4) };
        }
    }
    let buffers: Vec<_> = ctxs
        .iter()
        .map(|ctx| {
            let cb = gpu.queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder();
            encode_fold(enc, &psos, ctx, n, rounds - 1);
            enc.end_encoding();
            cb.commit();
            cb
        })
        .collect();
    for cb in &buffers {
        cb.wait_until_completed();
    }
    Ok(t0.elapsed().as_secs_f64())
}

/// The fold the two paths above compute, as slop would: sum the even and odd
/// entries for the round message, observe both, sample, then `fold_mle`.
fn cpu_reference(vals: &[EF], rounds: usize) -> (Vec<EF>, EF) {
    let mut challenger = KoalaBearDegree4Duplex::default_challenger();
    let mut cur = vals.to_vec();
    let mut betas = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let half = cur.len() / 2;
        let mut g0 = EF::zero();
        let mut g1 = EF::zero();
        for i in 0..half {
            g0 += cur[2 * i];
            g1 += cur[2 * i + 1];
        }
        challenger.observe_ext_element(g0);
        challenger.observe_ext_element(g1);
        let beta: EF = challenger.sample_ext_element();
        betas.push(beta);
        cur = (0..half).map(|i| cur[2 * i] + beta * cur[2 * i + 1]).collect();
    }
    (betas, cur[0])
}

/// Runs a scripted sequence of observes and samples on the device sponge,
/// keeping the state across dispatches, and returns everything sampled.
fn run_sponge(gpu: &Gpu, vals: &[u32], script: &[(usize, usize)]) -> Result<Vec<EF>, String> {
    let total_samples: usize = script.iter().map(|&(_, s)| s).sum();
    let buf_vals = gpu.buffer_from(vals);
    let buf_out = gpu.buffer_empty::<[u32; 4]>(total_samples.max(1));
    let buf_chal = gpu.buffer_empty::<u32>(CHALLENGER_WORDS);
    let buf_rc = gpu.buffer_from(&rc_monty());
    let init = gpu.pipeline("chal_init")?;
    let pso = gpu.pipeline("chal_observe_sample")?;
    let cb = gpu.queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&init);
    enc.set_buffer(0, Some(&buf_chal), 0);
    dispatch_one_group(enc, 1);
    enc.end_encoding();
    let (mut consumed, mut produced) = (0usize, 0usize);
    for &(n_obs, n_samp) in script {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&buf_chal), 0);
        enc.set_buffer(1, Some(&buf_vals), (consumed * 4) as u64);
        enc.set_buffer(2, Some(&buf_out), (produced * 16) as u64);
        set_bytes(enc, 3, &(n_obs as u32));
        set_bytes(enc, 4, &(n_samp as u32));
        enc.set_buffer(5, Some(&buf_rc), 0);
        dispatch_one_group(enc, 1);
        enc.end_encoding();
        consumed += n_obs;
        produced += n_samp;
    }
    cb.commit();
    cb.wait_until_completed();
    Ok(gpu.read_vec::<[u32; 4]>(&buf_out, total_samples).into_iter().map(ef_of_monty4).collect())
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    // The sponge itself, against slop's challenger. The script crosses the rate
    // boundary in both directions: partial input buffers that must duplex on
    // sample, a full buffer that must duplex on observe, and more samples than
    // one permutation's output can supply.
    let mut rng = StdRng::seed_from_u64(61);
    let script = [(3usize, 1usize), (8, 2), (1, 3), (16, 1), (0, 4), (5, 2)];
    let n_vals: usize = script.iter().map(|&(o, _)| o).sum();
    let vals = rand_f_vec(&mut rng, n_vals);
    let got = run_sponge(gpu, &monty_vec_of_f(&vals), &script)?;
    let mut challenger = KoalaBearDegree4Duplex::default_challenger();
    let mut expected = Vec::new();
    let mut consumed = 0usize;
    for &(n_obs, n_samp) in &script {
        for &v in &vals[consumed..consumed + n_obs] {
            challenger.observe(v);
        }
        consumed += n_obs;
        for _ in 0..n_samp {
            expected.push(challenger.sample_ext_element::<EF>());
        }
    }
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        if g != e {
            return Err(format!("duplex sponge mismatch at sample {i}: got {g:?}, expected {e:?}"));
        }
    }

    // The whole fold with the challenger in the loop, both paths, against the
    // same fold computed on the CPU with slop's challenger.
    let n = 1024usize;
    let rounds = n.trailing_zeros() as usize;
    let vals = rand_ef_vec(&mut rng, n);
    let (want_betas, want) = cpu_reference(&vals, rounds);

    let ctx = FoldCtx::new(gpu, &vals, rounds);
    fold_on_gpu(gpu, &ctx, n, rounds)?;
    let got_betas = ctx.betas(gpu, rounds);
    for (i, (g, e)) in got_betas.iter().zip(want_betas.iter()).enumerate() {
        if g != e {
            return Err(format!("device challenge mismatch at round {i}: got {g:?}, expected {e:?}"));
        }
    }
    let got = ctx.result(gpu, rounds);
    if got != want {
        return Err(format!("device-challenger fold mismatch: got {got:?}, expected {want:?}"));
    }

    let ctx = FoldCtx::new(gpu, &vals, rounds);
    let (_, host_betas) = fold_on_host(gpu, &ctx, n, rounds)?;
    if host_betas != want_betas {
        return Err("host-challenger fold sampled different challenges".to_string());
    }
    let got = ctx.result(gpu, rounds);
    if got != want {
        return Err(format!("host-challenger fold mismatch: got {got:?}, expected {want:?}"));
    }
    Ok(())
}

fn best_of<F: FnMut() -> Result<f64, String>>(reps: usize, mut f: F) -> Result<BenchTimes, String> {
    f()?;
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        times.push(f()?);
    }
    let best = times.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
    Ok(BenchTimes { best, mean, times })
}

pub fn bench(gpu: &Gpu, n: usize, reps: usize) -> Result<Value, String> {
    let mut rng = StdRng::seed_from_u64(62);
    let vals = rand_ef_vec(&mut rng, n);
    let rounds = n.trailing_zeros() as usize;
    let ctx = FoldCtx::new(gpu, &vals, rounds);

    let on_gpu = best_of(reps, || fold_on_gpu(gpu, &ctx, n, rounds))?;
    let on_host = best_of(reps, || fold_on_host(gpu, &ctx, n, rounds).map(|(t, _)| t))?;
    // The same host-driven rounds with several folds in flight at once. The data
    // is shared because only the timing matters; each fold has its own
    // challenger and its own buffers.
    let ctxs: Vec<FoldCtx> = (0..CONCURRENT).map(|_| FoldCtx::new(gpu, &vals, rounds)).collect();
    let on_host_conc = best_of(reps, || fold_on_host_concurrent(gpu, &ctxs, n, rounds))?;
    let per_fold_conc = on_host_conc.best / CONCURRENT as f64;
    let on_gpu_conc = best_of(reps, || fold_on_gpu_concurrent(gpu, &ctxs, n, rounds))?;
    let per_fold_gpu_conc = on_gpu_conc.best / CONCURRENT as f64;

    // The cost of the sponge alone, so the fold numbers can be read as
    // "reduction plus fold plus this".
    let sponge = {
        let psos = FoldPsos::new(gpu)?;
        let init = gpu.pipeline("chal_init")?;
        let t = gpu.bench_encoded(reps, rounds + 1, {
            let round = Cell::new(0usize);
            move |enc| {
                let r = round.get();
                if r == 0 {
                    encode_init(enc, &init, &ctx);
                } else {
                    enc.set_compute_pipeline_state(&psos.challenge);
                    enc.set_buffer(0, Some(&ctx.msg), ((r - 1) * 32) as u64);
                    enc.set_buffer(1, Some(&ctx.chal), 0);
                    enc.set_buffer(2, Some(&ctx.beta), ((r - 1) * 16) as u64);
                    enc.set_buffer(3, Some(&ctx.rc), 0);
                    dispatch_one_group(enc, 1);
                }
                round.set(r + 1);
            }
        });
        t
    };

    Ok(json!({
        "probe": "challenger",
        "params": { "n": n, "rounds": rounds, "reps": reps },
        "fold_host_challenger": {
            "times": times_json(&on_host),
            "total_ms": on_host.best * 1e3,
            "per_round_us": on_host.best / rounds as f64 * 1e6,
            "note": "one command buffer per round, the best a host-driven port can do",
        },
        "fold_device_challenger": {
            "times": times_json(&on_gpu),
            "total_ms": on_gpu.best * 1e3,
            "per_round_us": on_gpu.best / rounds as f64 * 1e6,
            "note": "every round in one submission; the sponge runs on the GPU",
        },
        "fold_host_challenger_concurrent": {
            "times": times_json(&on_host_conc),
            "folds": CONCURRENT,
            "total_ms": on_host_conc.best * 1e3,
            "per_fold_ms": per_fold_conc * 1e3,
            "per_fold_per_round_us": per_fold_conc / rounds as f64 * 1e6,
            "note": "one round-trip latency per round for the whole batch, not per fold",
        },
        "fold_device_challenger_concurrent": {
            "times": times_json(&on_gpu_conc),
            "folds": CONCURRENT,
            "total_ms": on_gpu_conc.best * 1e3,
            "per_fold_ms": per_fold_gpu_conc * 1e3,
            "per_fold_per_round_us": per_fold_gpu_conc / rounds as f64 * 1e6,
            "note": "device sponge and concurrency together",
        },
        "speedup": on_host.best / on_gpu.best,
        "speedup_from_concurrency": on_host.best / per_fold_conc,
        "speedup_from_both": on_host.best / per_fold_gpu_conc,
        "device_sponge_gain_given_concurrency": per_fold_conc / per_fold_gpu_conc,
        "sponge_only": {
            "times": times_json(&sponge),
            "per_challenge_us": sponge.best / rounds as f64 * 1e6,
            "note": "observe two extension elements and sample one, on the device",
        },
    }))
}
