// Reference kernels: what this machine can actually do, so a real kernel is
// measured against a ceiling that was observed rather than quoted.

// Pure streaming read of one array.
kernel void stream1(device const uint4* a [[buffer(0)]],
                    device uint4* out [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    ushort slane [[thread_index_in_simdgroup]],
                    ushort sgid [[simdgroup_index_in_threadgroup]]) {
    uint4 acc = uint4(0u);
    if (gid < n) acc = a[gid];
    for (ushort off = 16u; off > 0u; off >>= 1u) acc = ext_add(acc, simd_shuffle_down(acc, off));
    threadgroup uint4 s[8];
    if (tid < 8u) s[tid] = uint4(0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slane == 0u) s[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint4 t = s[0];
        for (uint k = 1u; k < 8u; k++) t = ext_add(t, s[k]);
        out[tgid] = t;
    }
}

// Streaming read of the same four arrays the round reads, same element count.
kernel void stream4(device const uint4* a [[buffer(0)]],
                    device const uint4* b [[buffer(1)]],
                    device const uint4* c [[buffer(2)]],
                    device const uint4* d [[buffer(3)]],
                    device uint4* out [[buffer(4)]],
                    constant uint& n [[buffer(5)]],
                    uint gid [[thread_position_in_grid]],
                    uint tgid [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]],
                    ushort slane [[thread_index_in_simdgroup]],
                    ushort sgid [[simdgroup_index_in_threadgroup]]) {
    uint4 acc = uint4(0u);
    if (gid < n) {
        acc = ext_add(ext_add(a[gid], b[gid]), ext_add(c[gid], d[gid]));
    }
    for (ushort off = 16u; off > 0u; off >>= 1u) acc = ext_add(acc, simd_shuffle_down(acc, off));
    threadgroup uint4 s[8];
    if (tid < 8u) s[tid] = uint4(0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slane == 0u) s[sgid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint4 t = s[0];
        for (uint k = 1u; k < 8u; k++) t = ext_add(t, s[k]);
        out[tgid] = t;
    }
}


// The arithmetic ceiling: extension multiplies with no memory traffic, four
// independent chains so the measurement is throughput and not latency.
kernel void alu_peak_school(device const uint4* a [[buffer(0)]],
                     device uint4* out [[buffer(1)]],
                     constant uint& iters [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint4 x0 = a[gid & 255u];
    uint4 x1 = a[(gid + 1u) & 255u];
    uint4 x2 = a[(gid + 2u) & 255u];
    uint4 x3 = a[(gid + 3u) & 255u];
    uint4 k0 = a[(gid + 5u) & 255u];
    for (uint i = 0u; i < iters; i++) {
        x0 = ext_mul(x0, k0);
        x1 = ext_mul(x1, k0);
        x2 = ext_mul(x2, k0);
        x3 = ext_mul(x3, k0);
    }
    out[gid] = ext_add(ext_add(x0, x1), ext_add(x2, x3));
}

// The arithmetic ceiling: extension multiplies with no memory traffic, four
// independent chains so the measurement is throughput and not latency.
kernel void alu_peak_lazy(device const uint4* a [[buffer(0)]],
                     device uint4* out [[buffer(1)]],
                     constant uint& iters [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint4 x0 = a[gid & 255u];
    uint4 x1 = a[(gid + 1u) & 255u];
    uint4 x2 = a[(gid + 2u) & 255u];
    uint4 x3 = a[(gid + 3u) & 255u];
    uint4 k0 = a[(gid + 5u) & 255u];
    for (uint i = 0u; i < iters; i++) {
        x0 = ext_mul_fast(x0, k0);
        x1 = ext_mul_fast(x1, k0);
        x2 = ext_mul_fast(x2, k0);
        x3 = ext_mul_fast(x3, k0);
    }
    out[gid] = ext_add(ext_add(x0, x1), ext_add(x2, x3));
}

// The arithmetic ceiling: extension multiplies with no memory traffic, four
// independent chains so the measurement is throughput and not latency.
kernel void alu_peak_kara(device const uint4* a [[buffer(0)]],
                     device uint4* out [[buffer(1)]],
                     constant uint& iters [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint4 x0 = a[gid & 255u];
    uint4 x1 = a[(gid + 1u) & 255u];
    uint4 x2 = a[(gid + 2u) & 255u];
    uint4 x3 = a[(gid + 3u) & 255u];
    uint4 k0 = a[(gid + 5u) & 255u];
    for (uint i = 0u; i < iters; i++) {
        x0 = ext_mul_kara(x0, k0);
        x1 = ext_mul_kara(x1, k0);
        x2 = ext_mul_kara(x2, k0);
        x3 = ext_mul_kara(x3, k0);
    }
    out[gid] = ext_add(ext_add(x0, x1), ext_add(x2, x3));
}
