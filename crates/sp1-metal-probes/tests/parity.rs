//! GPU-vs-CPU parity at small sizes. Every Metal primitive is compared
//! against the slop CPU implementation on shared random vectors; these run in
//! milliseconds and are the CI gate for kernel correctness.

use sp1_metal_probes::gpu::Gpu;
use sp1_metal_probes::probes;

fn gpu() -> Gpu {
    Gpu::new().expect("Metal device required for parity tests")
}

#[test]
fn triad_parity() {
    probes::triad::check(&gpu()).unwrap();
}

#[test]
fn imul_determinism() {
    probes::imul::check(&gpu()).unwrap();
}

#[test]
fn field_mul_parity() {
    probes::fieldops::check(&gpu()).unwrap();
}

#[test]
fn fold_and_fix_last_variable_parity() {
    probes::fold::check(&gpu()).unwrap();
}

#[test]
fn poseidon2_parity() {
    probes::poseidon2::check(&gpu()).unwrap();
}

#[test]
fn zerocheck_parity() {
    probes::zerocheck::check(&gpu()).unwrap();
}

#[test]
fn fold_tail_round_trip_parity() {
    probes::launch::check(&gpu()).unwrap();
}

#[test]
fn ntt_butterfly_stage_parity() {
    probes::encode::check(&gpu()).unwrap();
}
