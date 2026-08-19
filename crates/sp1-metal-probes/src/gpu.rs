use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLResourceOptions, MTLSize,
};

pub const SHADER_SRC: &str = include_str!("shaders.metal");
pub const THREADGROUP_SIZE: u64 = 256;

pub struct Gpu {
    pub device: Device,
    pub queue: CommandQueue,
    library: Library,
    pipelines: RefCell<HashMap<String, ComputePipelineState>>,
}

pub struct BenchTimes {
    pub best: f64,
    pub mean: f64,
    pub times: Vec<f64>,
}

impl Gpu {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device found")?;
        let queue = device.new_command_queue();
        let library = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .map_err(|e| format!("shader compilation failed: {e}"))?;
        Ok(Self { device, queue, library, pipelines: RefCell::new(HashMap::new()) })
    }

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

pub fn dispatch_groups(enc: &ComputeCommandEncoderRef, groups: u64) {
    enc.dispatch_thread_groups(
        MTLSize::new(groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
}
