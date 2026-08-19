pub mod fieldops;
pub mod fold;
pub mod imul;
pub mod poseidon2;
pub mod triad;
pub mod zerocheck;

use serde_json::{json, Value};

use crate::gpu::BenchTimes;

pub fn times_json(t: &BenchTimes) -> Value {
    json!({ "best_s": t.best, "mean_s": t.mean, "all_s": t.times })
}
