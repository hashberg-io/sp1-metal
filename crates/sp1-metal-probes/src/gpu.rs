// SPDX-License-Identifier: MIT OR Apache-2.0
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLResourceOptions, MTLSize,
};

/// The probe suite's own kernels.
const PROBE_SRC: &str = include_str!("shaders.metal");

/// The encode kernels, included from the vendored crate that ships them by path
/// rather than by depending on it: that crate sits below this one, and the
/// point of reading its file is that the parity check tests the source a proof
/// runs, not a copy.
const DFT_SRC: &str = include_str!("../../vendor/slop-dft/src/shaders.metal");

/// The whole library: everything the probes compile against.
pub fn shader_src() -> String {
    format!("{PROBE_SRC}\n{DFT_SRC}")
}
/// Threads per group the probes dispatch with unless a probe says otherwise.
/// Large enough to fill a threadgroup on every M-series part measured here.
pub const THREADGROUP_SIZE: u64 = 256;

/// A Metal device with the probe library compiled against it.
///
/// Pipelines are compiled on first use and cached, so a probe that dispatches
/// the same kernel a hundred times pays for it once.
pub struct Gpu {
    /// The system default device.
    pub device: Device,
    /// One queue, shared by every dispatch this harness makes.
    pub queue: CommandQueue,
    library: Library,
    pipelines: RefCell<HashMap<String, ComputePipelineState>>,
}

/// The timings of a repeated measurement.
///
/// `best` is what the probes report: on a machine with other work on it the
/// minimum is the reading least contaminated by that work.
pub struct BenchTimes {
    /// The fastest run, in seconds.
    pub best: f64,
    /// The arithmetic mean of every run, in seconds.
    pub mean: f64,
    /// Every run, in order, in seconds.
    pub times: Vec<f64>,
}

impl Gpu {
    /// Opens the default device and compiles the probe library against it.
    ///
    /// The compile is the expensive part and macOS caches it against the
    /// source, so the first run after any shader edit is far slower than the
    /// rest -- a fact that has cost this project a wrong conclusion before.
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device found")?;
        let queue = device.new_command_queue();
        let library = device
            .new_library_with_source(&shader_src(), &CompileOptions::new())
            .map_err(|e| format!("shader compilation failed: {e}"))?;
        Ok(Self { device, queue, library, pipelines: RefCell::new(HashMap::new()) })
    }

    /// The compute pipeline for kernel `name`, compiled once and cached.
    pub fn pipeline(&self, name: &str) -> Result<ComputePipelineState, String> {
        if let Some(pso) = self.pipelines.borrow().get(name) {
            return Ok(pso.clone());
        }
        let function = self
            .library
            .get_function(name, None)
            .map_err(|e| format!("kernel `{name}` not found: {e}"))?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("pipeline for `{name}` failed: {e}"))?;
        self.pipelines.borrow_mut().insert(name.to_string(), pso.clone());
        Ok(pso)
    }

    pub fn buffer_from<T: Copy>(&self, data: &[T]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            (data.len() * size_of::<T>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn buffer_empty<T>(&self, len: usize) -> Buffer {
        self.device
            .new_buffer((len * size_of::<T>()) as u64, MTLResourceOptions::StorageModeShared)
    }

    pub fn read_vec<T: Copy>(&self, buffer: &Buffer, len: usize) -> Vec<T> {
        let ptr = buffer.contents().cast::<T>();
        assert!(!ptr.is_null(), "buffer has no CPU-visible contents");
        assert!(len * size_of::<T>() <= buffer.length() as usize, "read past end of buffer");
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    /// Encode `encodes` compute passes via `encode` into one command buffer,
    /// then time commit-to-completion on the host clock.
    pub fn time_encoded<F: Fn(&ComputeCommandEncoderRef)>(&self, encodes: usize, encode: F) -> f64 {
        let cb = self.queue.new_command_buffer();
        for _ in 0..encodes {
            let enc = cb.new_compute_command_encoder();
            encode(enc);
            enc.end_encoding();
        }
        let t0 = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        t0.elapsed().as_secs_f64()
    }

    /// Encode and submit `count` command buffers one at a time, waiting for each
    /// before building the next. This is the shape a sumcheck round tail is
    /// forced into: every round's challenge depends on the previous round's
    /// output, so the rounds cannot be batched into one submission.
    pub fn time_round_trips<F: Fn(&ComputeCommandEncoderRef, usize)>(
        &self,
        count: usize,
        encode: F,
    ) -> f64 {
        let t0 = Instant::now();
        for round in 0..count {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode(enc, round);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        t0.elapsed().as_secs_f64()
    }

    /// One warmup pass, then `reps` timed passes of `count` round trips each.
    pub fn bench_round_trips<F: Fn(&ComputeCommandEncoderRef, usize)>(
        &self,
        reps: usize,
        count: usize,
        encode: F,
    ) -> BenchTimes {
        let _ = self.time_round_trips(count, &encode);
        let times: Vec<f64> = (0..reps).map(|_| self.time_round_trips(count, &encode)).collect();
        let best = times.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
        BenchTimes { best, mean, times }
    }

    /// One warmup pass, then `reps` timed passes of `encodes` encodings each.
    pub fn bench_encoded<F: Fn(&ComputeCommandEncoderRef)>(
        &self,
        reps: usize,
        encodes: usize,
        encode: F,
    ) -> BenchTimes {
        let _ = self.time_encoded(encodes, &encode);
        let times: Vec<f64> = (0..reps).map(|_| self.time_encoded(encodes, &encode)).collect();
        let best = times.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = times.iter().sum::<f64>() / times.len().max(1) as f64;
        BenchTimes { best, mean, times }
    }
}

pub fn set_bytes<T: Copy>(enc: &ComputeCommandEncoderRef, index: u64, value: &T) {
    enc.set_bytes(index, size_of::<T>() as u64, (value as *const T).cast());
}

pub fn dispatch_1d(enc: &ComputeCommandEncoderRef, n: u64) {
    let tg = THREADGROUP_SIZE.min(n.max(1));
    enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
}

/// Dispatches a grid with the column index fastest, so consecutive threads read
/// consecutive columns of the same row.
pub fn dispatch_2d(enc: &ComputeCommandEncoderRef, cols: u64, rows: u64) {
    let tw = THREADGROUP_SIZE.min(cols.max(1));
    let th = (THREADGROUP_SIZE / tw).max(1).min(rows.max(1));
    enc.dispatch_threads(MTLSize::new(cols, rows, 1), MTLSize::new(tw, th, 1));
}

/// Dispatches `x` by `y` threadgroups of `THREADGROUP_SIZE` threads each.
pub fn dispatch_groups_2d(enc: &ComputeCommandEncoderRef, x: u64, y: u64) {
    enc.dispatch_thread_groups(MTLSize::new(x, y, 1), MTLSize::new(THREADGROUP_SIZE, 1, 1));
}

pub fn dispatch_groups(enc: &ComputeCommandEncoderRef, groups: u64) {
    enc.dispatch_thread_groups(
        MTLSize::new(groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
}
