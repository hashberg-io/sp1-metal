// SPDX-License-Identifier: MIT OR Apache-2.0
//! Where the time in an upload actually goes: allocating the buffer, or copying.
use metal::{Device, MTLResourceOptions};
use std::time::Instant;

fn main() {
    let device = Device::system_default().expect("no Metal device");
    // The real prover's average upload was 179 GB over 15415 uploads.
    let bytes = 11_600_000usize;
    let reps = 200;
    let src = vec![0u8; bytes];
    let total_gb = (bytes * reps) as f64 / 1e9;

    let t = Instant::now();
    for _ in 0..reps {
        let b = device.new_buffer_with_data(
            src.as_ptr().cast(),
            bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        std::hint::black_box(&b);
    }
    let alloc_and_copy = t.elapsed().as_secs_f64();

    // Same copy, into a buffer that already exists.
    let pooled = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    let t = Instant::now();
    for _ in 0..reps {
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), pooled.contents().cast::<u8>(), bytes);
        }
        std::hint::black_box(&pooled);
    }
    let copy_only = t.elapsed().as_secs_f64();

    // Allocation alone, nothing copied.
    let t = Instant::now();
    for _ in 0..reps {
        let b = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
        std::hint::black_box(&b);
    }
    let alloc_only = t.elapsed().as_secs_f64();

    println!("{reps} uploads of {:.1} MB ({total_gb:.1} GB total)", bytes as f64 / 1e6);
    println!("  new_buffer_with_data  {alloc_and_copy:>7.3} s  {:>6.1} GB/s", total_gb / alloc_and_copy);
    println!("  copy into pooled      {copy_only:>7.3} s  {:>6.1} GB/s", total_gb / copy_only);
    println!("  allocate only         {alloc_only:>7.3} s", );
}
