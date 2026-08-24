// SPDX-License-Identifier: MIT OR Apache-2.0
//! slop's sumcheck driver, running unmodified with GPU kernels underneath.
//!
//! `reduce_sumcheck_to_evaluation` is generic over `SumcheckPolyFirstRound`,
//! which slop provides by blanket impl to any polynomial whose backend
//! implements three traits in `slop_sumcheck::backend`. That is the designed
//! extension point for a device backend, and the zerocheck path is wired
//! through it on the CPU side, so it is the seam a Metal zerocheck would use.
//!
//! This implements those three traits for `MetalBackend` over a multilinear
//! polynomial whose values live in a `slop_alloc::Buffer<_, MetalBackend>`, and
//! checks that the driver produces the identical proof to the one it produces
//! over slop's own `Mle` on the CPU: the same univariate messages, the same
//! point, the same claims.
//!
//! Note what the shape of this says. The driver observes each round's message
//! into a host challenger and samples the next alpha there, so every round is a
//! host round trip no matter how fast the kernels are — which is the constraint
//! the challenger spike measured, showing up here in upstream's own control
//! flow rather than in a probe.

use std::sync::Mutex;

use metal::{CommandQueue, ComputePipelineState};
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde_json::{json, Value};
use slop_algebra::{interpolate_univariate_polynomial, AbstractField, UnivariatePolynomial};
use slop_alloc::{Buffer as SlopBuffer, HasBackend};
use slop_challenger::IopCtx;
use slop_koala_bear::KoalaBearDegree4Duplex;
use slop_multilinear::Mle;
use slop_sumcheck::{
    reduce_sumcheck_to_evaluation, ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend,
    SumcheckPolyBackend, SumcheckPolyBase,
};

use crate::backend::MetalBackend;
use crate::gpu::{dispatch_1d, dispatch_groups, set_bytes, Gpu, THREADGROUP_SIZE};
use crate::kb::{ef_of_monty4, monty4_of_ef, monty4_vec_of_ef, rand_ef_vec, EF};
use crate::probes::times_json;

/// The queue, pipelines and scratch a device polynomial needs to reduce and
/// fold itself. The scratch is allocated once: a sumcheck runs one reduction
/// per round over a working set that halves, so the same two small buffers
/// serve every round.
#[derive(Clone)]
pub struct MetalOps {
    queue: CommandQueue,
    partials_pso: ComputePipelineState,
    reduce_pso: ComputePipelineState,
    flv_pso: ComputePipelineState,
    partials: metal::Buffer,
    msg: metal::Buffer,
}

/// Threadgroups the shared partials buffer can hold, which caps the working set
/// a single reduction can cover.
const MAX_GROUPS: usize = 1 << 18;

impl MetalOps {
    pub fn new(gpu: &Gpu) -> Result<Self, String> {
        Ok(Self {
            queue: gpu.device.new_command_queue(),
            partials_pso: gpu.pipeline("sc_partials")?,
            reduce_pso: gpu.pipeline("sc_reduce")?,
            flv_pso: gpu.pipeline("flv_ext")?,
            partials: gpu
                .device
                .new_buffer((MAX_GROUPS * 2 * 16) as u64, metal::MTLResourceOptions::StorageModeShared),
            msg: gpu.device.new_buffer(32, metal::MTLResourceOptions::StorageModeShared),
        })
    }
}

/// A multilinear polynomial over the degree-4 extension whose values live in a
/// slop buffer allocated by `MetalBackend`, and are therefore already a Metal
/// buffer.
pub struct MetalMle {
    ops: MetalOps,
    backend: MetalBackend,
    values: SlopBuffer<[u32; 4], MetalBackend>,
    num_variables: u32,
    /// The fold that produced these values, if the host has not yet waited on
    /// it. Reductions synchronise through the queue, but a host read of the
    /// values themselves has to wait for it explicitly.
    pending: Mutex<Option<metal::CommandBuffer>>,
}

impl MetalMle {
    pub fn new(ops: MetalOps, backend: MetalBackend, vals: &[EF]) -> Self {
        assert!(vals.len().is_power_of_two(), "an MLE needs a power-of-two number of values");
        let monty = monty4_vec_of_ef(vals);
        let mut values =
            SlopBuffer::<[u32; 4], MetalBackend>::with_capacity_in(monty.len(), backend.clone());
        unsafe {
            std::ptr::copy_nonoverlapping(monty.as_ptr(), values.as_mut_ptr(), monty.len());
            values.set_len(monty.len());
        }
        let num_variables = vals.len().trailing_zeros();
        Self { ops, backend, values, num_variables, pending: Mutex::new(None) }
    }

    fn len(&self) -> usize {
        1usize << self.num_variables
    }

    fn mtl(&self) -> (metal::Buffer, u64) {
        self.backend
            .buffer_for(self.values.as_ptr().cast())
            .expect("MetalMle values are not backed by a Metal allocation")
    }

    fn value_at(&self, i: usize) -> EF {
        if let Some(cb) = self.pending.lock().expect("pending fold poisoned").take() {
            cb.wait_until_completed();
        }
        ef_of_monty4(unsafe { *self.values.as_ptr().add(i) })
    }

    /// Sums the entries at even indices: the polynomial's value at 0 in its
    /// last variable, summed over every assignment of the others.
    fn sum_at_zero(&self) -> EF {
        let n_pairs = self.len() / 2;
        let groups = n_pairs.div_ceil(THREADGROUP_SIZE as usize);
        assert!(groups <= MAX_GROUPS, "working set exceeds the shared partials buffer");
        let partials = &self.ops.partials;
        let msg = &self.ops.msg;
        let (values, offset) = self.mtl();
        let cb = self.ops.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.ops.partials_pso);
        enc.set_buffer(0, Some(&values), offset);
        enc.set_buffer(1, Some(partials), 0);
        set_bytes(enc, 2, &(n_pairs as u32));
        dispatch_groups(enc, groups as u64);
        enc.end_encoding();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.ops.reduce_pso);
        enc.set_buffer(0, Some(partials), 0);
        enc.set_buffer(1, Some(msg), 0);
        set_bytes(enc, 2, &(groups as u32));
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let raw = msg.contents().cast::<[u32; 4]>();
        ef_of_monty4(unsafe { *raw })
    }

    /// Fixes the last variable to `alpha`, halving the polynomial.
    fn fix_last(self, alpha: EF) -> Self {
        let half = self.len() / 2;
        let mut out =
            SlopBuffer::<[u32; 4], MetalBackend>::with_capacity_in(half, self.backend.clone());
        unsafe { out.set_len(half) };
        let (src, src_off) = self.mtl();
        let (dst, dst_off) = self
            .backend
            .buffer_for(out.as_ptr().cast())
            .expect("output is not backed by a Metal allocation");
        let cb = self.ops.queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.ops.flv_pso);
        enc.set_buffer(0, Some(&src), src_off);
        enc.set_buffer(1, Some(&dst), dst_off);
        set_bytes(enc, 2, &monty4_of_ef(alpha));
        dispatch_1d(enc, half as u64);
        enc.end_encoding();
        // Committed but not waited on: command buffers on one queue run in
        // order, so the next round's reduction is already sequenced behind this
        // fold and there is nothing for the host to learn by blocking here.
        cb.commit();
        Self {
            ops: self.ops,
            backend: self.backend,
            values: out,
            num_variables: self.num_variables - 1,
            pending: Mutex::new(Some(cb)),
        }
    }
}

impl SumcheckPolyBase for MetalMle {
    fn num_variables(&self) -> u32 {
        self.num_variables
    }
}

impl HasBackend for MetalMle {
    type Backend = MetalBackend;

    fn backend(&self) -> &Self::Backend {
        &self.backend
    }
}

impl ComponentPolyEvalBackend<MetalMle, EF> for MetalBackend {
    fn get_component_poly_evals(poly: &MetalMle) -> Vec<EF> {
        vec![poly.value_at(0)]
    }
}

impl SumcheckPolyBackend<MetalMle, EF> for MetalBackend {
    fn fix_last_variable(poly: MetalMle, alpha: EF) -> MetalMle {
        poly.fix_last(alpha)
    }

    fn sum_as_poly_in_last_variable(
        poly: &MetalMle,
        claim: Option<EF>,
    ) -> UnivariatePolynomial<EF> {
        if poly.num_variables == 0 {
            return UnivariatePolynomial::new(vec![poly.value_at(0), EF::zero()]);
        }
        let claim = claim.expect("expected a claim for a non-zero-variate polynomial");
        let eval_zero = poly.sum_at_zero();
        let eval_one = claim - eval_zero;
        interpolate_univariate_polynomial(&[EF::zero(), EF::one()], &[eval_zero, eval_one])
    }
}

impl SumCheckPolyFirstRoundBackend<MetalMle, EF> for MetalBackend {
    type NextRoundPoly = MetalMle;

    fn fix_t_variables(poly: MetalMle, alpha: EF, t: usize) -> Self::NextRoundPoly {
        assert_eq!(t, 1, "this backend folds one variable per round");
        poly.fix_last(alpha)
    }

    fn sum_as_poly_in_last_t_variables(
        poly: &MetalMle,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert_eq!(t, 1, "this backend folds one variable per round");
        <MetalBackend as SumcheckPolyBackend<MetalMle, EF>>::sum_as_poly_in_last_variable(
            poly, claim,
        )
    }
}

/// Runs slop's driver over slop's own CPU `Mle`.
fn prove_on_cpu(mle: Mle<EF>, claim: EF, lambda: EF) -> (Vec<Vec<EF>>, Vec<EF>, EF) {
    let mut challenger = KoalaBearDegree4Duplex::default_challenger();
    let (proof, evals) =
        reduce_sumcheck_to_evaluation(vec![mle], &mut challenger, vec![claim], 1, lambda);
    (
        proof.univariate_polys.iter().map(|p| p.coefficients.clone()).collect(),
        proof.point_and_eval.0.to_vec(),
        evals[0][0],
    )
}

/// Runs the same driver over `MetalMle`, changing nothing but the backend.
fn prove_on_gpu(mle: MetalMle, claim: EF, lambda: EF) -> (Vec<Vec<EF>>, Vec<EF>, EF) {
    let mut challenger = KoalaBearDegree4Duplex::default_challenger();
    let (proof, evals) =
        reduce_sumcheck_to_evaluation(vec![mle], &mut challenger, vec![claim], 1, lambda);
    (
        proof.univariate_polys.iter().map(|p| p.coefficients.clone()).collect(),
        proof.point_and_eval.0.to_vec(),
        evals[0][0],
    )
}

pub fn check(gpu: &Gpu) -> Result<(), String> {
    let ops = MetalOps::new(gpu)?;
    let backend = MetalBackend::new()?;
    let mut rng = StdRng::seed_from_u64(101);
    for log_n in [1usize, 4, 10] {
        let vals = rand_ef_vec(&mut rng, 1 << log_n);
        let claim: EF = vals.iter().copied().sum();
        let lambda: EF = {
            use rand::Rng;
            rng.gen()
        };
        let want = prove_on_cpu(Mle::from(vals.clone()), claim, lambda);
        let got =
            prove_on_gpu(MetalMle::new(ops.clone(), backend.clone(), &vals), claim, lambda);
        if got.0 != want.0 {
            return Err(format!("sumcheck round messages differ at 2^{log_n} values"));
        }
        if got.1 != want.1 {
            return Err(format!("sumcheck evaluation point differs at 2^{log_n} values"));
        }
        if got.2 != want.2 {
            return Err(format!("sumcheck final evaluation differs at 2^{log_n} values"));
        }
    }
    Ok(())
}

pub fn bench(gpu: &Gpu, n: usize, reps: usize) -> Result<Value, String> {
    let ops = MetalOps::new(gpu)?;
    let backend = MetalBackend::new()?;
    let mut rng = StdRng::seed_from_u64(102);
    let vals = rand_ef_vec(&mut rng, n);
    let claim: EF = vals.iter().copied().sum();
    let lambda: EF = {
        use rand::Rng;
        rng.gen()
    };
    // The driver consumes the polynomial, so each pass needs a fresh one. Both
    // are built outside the timed region: the Metal side would otherwise be
    // charged for converting four million extension elements into Montgomery
    // form, which is setup rather than proving.
    let time_each = |mut make: Box<dyn FnMut() -> Box<dyn FnOnce()>>| {
        let mut times = Vec::with_capacity(reps);
        for i in 0..=reps {
            let run = make();
            let t0 = std::time::Instant::now();
            run();
            let elapsed = t0.elapsed().as_secs_f64();
            if i > 0 {
                times.push(elapsed);
            }
        }
        let best = times.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
        crate::gpu::BenchTimes { best, mean, times }
    };
    let g = {
        let ops = ops.clone();
        let backend = backend.clone();
        let vals = vals.clone();
        time_each(Box::new(move || {
            let mle = MetalMle::new(ops.clone(), backend.clone(), &vals);
            Box::new(move || {
                std::hint::black_box(prove_on_gpu(mle, claim, lambda));
            })
        }))
    };
    let c = {
        let vals = vals.clone();
        time_each(Box::new(move || {
            let mle = Mle::from(vals.clone());
            Box::new(move || {
                std::hint::black_box(prove_on_cpu(mle, claim, lambda));
            })
        }))
    };
    let rounds = n.trailing_zeros() as usize;
    Ok(json!({
        "probe": "sumcheck",
        "params": { "n": n, "rounds": rounds, "reps": reps },
        "driver_on_metal": {
            "times": times_json(&g),
            "milliseconds": g.best * 1e3,
            "per_round_us": g.best / rounds as f64 * 1e6,
        },
        "driver_on_cpu": {
            "times": times_json(&c),
            "milliseconds": c.best * 1e3,
            "per_round_us": c.best / rounds as f64 * 1e6,
            "note": "slop's own Mle on CpuBackend, scalar as the prover runs it",
        },
        "metal_over_cpu": c.best / g.best,
        "note": "the driver keeps the challenger on the host, so every round is a round trip",
    }))
}
