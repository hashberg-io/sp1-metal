// SPDX-License-Identifier: MIT OR Apache-2.0
//! Scores the fork's logup GKR round kernel against ceilings measured here.
//!
//! Times are the GPU's own, from `GPUStartTime` and `GPUEndTime` on the command
//! buffer, so they contain execution and nothing else. The shaders are read
//! from the fork at run time, so this always measures what actually ships.

use metal::objc::{msg_send, sel, sel_impl};
use metal::{ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::mem::size_of;
use std::path::PathBuf;

/// The KoalaBear prime.
const P: u32 = 0x7f00_0001;
const THREADS: u64 = 256;
/// Matches `MAX_GROUPS` in the fork: past this the kernel strides.
const MAX_GROUPS: u64 = 2048;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 32) as u32) % P
    }

    fn vec(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn add_p(a: u32, b: u32) -> u32 {
    let s = a as u64 + b as u64;
    (if s >= P as u64 { s - P as u64 } else { s }) as u32
}

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("repo root")
}

fn read(p: PathBuf) -> String {
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The fork's own shader text: shared field arithmetic, then the round kernels.
fn shader_source() -> String {
    let root = repo();
    let probes = read(root.join("crates/sp1-metal-probes/src/shaders.metal"));
    let header = probes
        .split("// ---- end of shared field arithmetic ----")
        .next()
        .expect("field arithmetic sentinel")
        .to_string();
    let one = ((1u64 << 32) % P as u64) as u32;
    let shipped = read(root.join("crates/vendor/sp1-hypercube/src/logup_gkr/shaders.metal"));
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let baseline = read(here.join("shaders/baseline.metal"));
    let reference = read(here.join("shaders/reference.metal"));
    format!("{header}\nconstant uint KB_ONE = {one}u;\n{shipped}\n{baseline}\n{reference}")
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n_pairs: u32,
    n_full_pairs: u32,
    width: u32,
}

/// Best GPU-only time over `reps` runs, the first discarded as warm-up.
fn time_gpu<F>(queue: &metal::CommandQueue, reps: usize, mut encode: F) -> f64
where
    F: FnMut(&metal::ComputeCommandEncoderRef),
{
    let mut best = f64::INFINITY;
    for r in 0..=reps {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        encode(enc);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if r > 0 {
            // Not bound by the metal crate, but ordinary properties on MTLCommandBuffer.
            let (t0, t1): (f64, f64) =
                unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
            best = best.min(t1 - t0);
        }
    }
    best
}

fn sum_partials(buf: &metal::Buffer, groups: usize) -> [[u32; 4]; 3] {
    let raw = buf.contents().cast::<u32>();
    let mut out = [[0u32; 4]; 3];
    for g in 0..groups {
        for k in 0..3 {
            for c in 0..4 {
                // SAFETY: the kernel wrote three extension elements per threadgroup.
                let v = unsafe { *raw.add((3 * g + k) * 4 + c) };
                out[k][c] = add_p(out[k][c], v);
            }
        }
    }
    out
}

fn main() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(&shader_source(), &metal::CompileOptions::new())
        .unwrap_or_else(|e| panic!("shader compilation failed:\n{e}"));
    let pso = |name: &str| -> ComputePipelineState {
        let f = lib.get_function(name, None).unwrap_or_else(|e| panic!("{name}: {e}"));
        device.new_compute_pipeline_state_with_function(&f).expect("pipeline")
    };
    let shipped = pso("logup_round_ext");
    let baseline = pso("logup_round_ext_baseline");
    let stream1 = pso("stream1");
    let stream4 = pso("stream4");
    let alu = [
        ("schoolbook", pso("alu_peak_school")),
        ("lazy reduction", pso("alu_peak_lazy")),
        ("karatsuba", pso("alu_peak_kara")),
    ];

    println!("device: {}", device.name());
    println!(
        "  shipped threadgroup memory {} B, baseline {} B",
        shipped.static_threadgroup_memory_length(),
        baseline.static_threadgroup_memory_length()
    );

    let reps = 20;
    let mut rng = Rng(0x1234_5678_9abc_def0);
    let mk = |v: &[u32]| {
        device.new_buffer_with_data(
            v.as_ptr().cast(),
            (v.len() * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };

    // What streaming reaches here, so a kernel's rate means something.
    let n_ref = 8 << 20;
    let a = rng.vec(n_ref * 4);
    let (ba, bb, bc, bd) = (mk(&a), mk(&a), mk(&a), mk(&a));
    let g_ref = (n_ref as u64).div_ceil(THREADS);
    let out = device.new_buffer(g_ref * 16, MTLResourceOptions::StorageModeShared);
    let n = n_ref as u32;
    let t1 = time_gpu(&queue, reps, |enc| {
        enc.set_compute_pipeline_state(&stream1);
        enc.set_buffer(0, Some(&ba), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.set_bytes(2, 4, (&n as *const u32).cast());
        enc.dispatch_thread_groups(MTLSize::new(g_ref, 1, 1), MTLSize::new(THREADS, 1, 1));
    });
    let t4 = time_gpu(&queue, reps, |enc| {
        enc.set_compute_pipeline_state(&stream4);
        for (i, b) in [&ba, &bb, &bc, &bd].into_iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_buffer(4, Some(&out), 0);
        enc.set_bytes(5, 4, (&n as *const u32).cast());
        enc.dispatch_thread_groups(MTLSize::new(g_ref, 1, 1), MTLSize::new(THREADS, 1, 1));
    });
    let ceiling = ((n_ref * 16) as f64 / t1).max((n_ref * 64) as f64 / t4) / 1e9;
    println!("\nmemory ceiling, measured");
    println!("  one array    {:>8.3} ms  {:>7.1} GB/s", t1 * 1e3, (n_ref * 16) as f64 / t1 / 1e9);
    println!("  four arrays  {:>8.3} ms  {:>7.1} GB/s", t4 * 1e3, (n_ref * 64) as f64 / t4 / 1e9);

    println!("\narithmetic ceiling, no memory traffic");
    let (threads_alu, iters) = (1u64 << 20, 512u32);
    let bout = device.new_buffer(threads_alu * 16, MTLResourceOptions::StorageModeShared);
    for (name, p) in &alu {
        let t = time_gpu(&queue, 5, |enc| {
            enc.set_compute_pipeline_state(p);
            enc.set_buffer(0, Some(&ba), 0);
            enc.set_buffer(1, Some(&bout), 0);
            enc.set_bytes(2, 4, (&iters as *const u32).cast());
            enc.dispatch_thread_groups(
                MTLSize::new(threads_alu / THREADS, 1, 1),
                MTLSize::new(THREADS, 1, 1),
            );
        });
        let rate = threads_alu as f64 * iters as f64 * 4.0 / t / 1e9;
        println!("  {name:<15} {:>7.2} G ext_mul/s", rate);
    }

    let rows: usize =
        std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(65536);
    println!("\nlogup round reduction, {rows} rows");
    println!(
        "  {:<6} {:>11} {:>10} {:>9} {:>9} {:>7} {:>6}",
        "width", "baseline ms", "shipped ms", "speedup", "GB/s", "% ceil", "match"
    );
    for &width in &[1usize, 4, 8, 13, 32, 64] {
        let cells = rows * width;
        let n_pairs = rows / 2;
        let params =
            Params { n_pairs: n_pairs as u32, n_full_pairs: n_pairs as u32, width: width as u32 };
        let mut rng = Rng(0xdead_beef_0000_0001 ^ width as u64);
        let (n0, n1, d0, d1) =
            (rng.vec(cells * 4), rng.vec(cells * 4), rng.vec(cells * 4), rng.vec(cells * 4));
        let (eq_int, eq_row) = (rng.vec(width * 4), rng.vec(rows * 4));
        let lambda = [rng.next(), rng.next(), rng.next(), rng.next()];
        let bufs = [mk(&n0), mk(&n1), mk(&d0), mk(&d1), mk(&eq_int), mk(&eq_row)];
        let g_base = (n_pairs as u64).div_ceil(THREADS);
        let g_ship = (cells as u64).div_ceil(THREADS).min(MAX_GROUPS);
        let part = device
            .new_buffer(3 * g_base.max(g_ship) * 16, MTLResourceOptions::StorageModeShared);
        let bind = |enc: &metal::ComputeCommandEncoderRef| {
            for (i, b) in bufs.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.set_buffer(6, Some(&part), 0);
            enc.set_bytes(7, size_of::<Params>() as u64, (&params as *const Params).cast());
            enc.set_bytes(8, 16, lambda.as_ptr().cast());
        };
        let t_base = time_gpu(&queue, reps, |enc| {
            enc.set_compute_pipeline_state(&baseline);
            bind(enc);
            enc.dispatch_thread_groups(MTLSize::new(g_base, 1, 1), MTLSize::new(THREADS, 1, 1));
        });
        let s_base = sum_partials(&part, g_base as usize);
        let t_ship = time_gpu(&queue, reps, |enc| {
            enc.set_compute_pipeline_state(&shipped);
            bind(enc);
            enc.dispatch_thread_groups(MTLSize::new(g_ship, 1, 1), MTLSize::new(THREADS, 1, 1));
        });
        let s_ship = sum_partials(&part, g_ship as usize);
        let gbs = (cells * 64) as f64 / t_ship / 1e9;
        println!(
            "  {:<6} {:>11.3} {:>10.3} {:>9.2} {:>9.1} {:>6.1}% {:>6}",
            width,
            t_base * 1e3,
            t_ship * 1e3,
            t_base / t_ship,
            gbs,
            gbs / ceiling * 100.0,
            if s_base == s_ship { "yes" } else { "NO" }
        );
        assert_eq!(s_base, s_ship, "shipped kernel disagrees with the baseline at width {width}");
    }
}
