
// ---------------------------------------------------------------------------
// The logup GKR round reduction.
//
// One thread per row pair. Each thread sums over the chip's interactions, then
// weights by that row pair's eq values, and the threadgroup reduces the three
// accumulators the round message is built from.
//
// A "full" pair has both of its rows present; the last pair of a chip may not,
// in which case the missing row is the padding, whose numerators are zero and
// whose denominators are one.
// ---------------------------------------------------------------------------

struct LogupRoundParams {
    uint n_pairs;       // row pairs in this chip
    uint n_full_pairs;  // of those, how many have both rows
    uint width;         // interactions in this chip
};

// Later layers: numerators and denominators are both extension elements.
kernel void logup_round_ext(device const uint4* n0 [[buffer(0)]],
                            device const uint4* n1 [[buffer(1)]],
                            device const uint4* d0 [[buffer(2)]],
                            device const uint4* d1 [[buffer(3)]],
                            device const uint4* eq_int [[buffer(4)]],
                            device const uint4* eq_row [[buffer(5)]],
                            device uint4* partials [[buffer(6)]],
                            constant LogupRoundParams& p [[buffer(7)]],
                            constant uint4& lambda [[buffer(8)]],
                            uint gid [[thread_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 sh0[256];
    threadgroup uint4 shh[256];
    threadgroup uint4 she[256];
    uint4 acc0 = uint4(0u, 0u, 0u, 0u);
    uint4 acch = uint4(0u, 0u, 0u, 0u);
    uint4 acce = uint4(0u, 0u, 0u, 0u);
    if (gid < p.n_pairs) {
        uint base = gid * 2u * p.width;
        uint4 eq0 = eq_row[2u * gid];
        uint4 eqs = ext_add(eq0, eq_row[2u * gid + 1u]);
        uint4 s_num0 = uint4(0u, 0u, 0u, 0u);
        uint4 s_den0 = uint4(0u, 0u, 0u, 0u);
        uint4 s_numh = uint4(0u, 0u, 0u, 0u);
        uint4 s_denh = uint4(0u, 0u, 0u, 0u);
        uint4 s_eq = uint4(0u, 0u, 0u, 0u);
        bool full = gid < p.n_full_pairs;
        for (uint j = 0; j < p.width; j++) {
            uint4 e = eq_int[j];
            uint4 a0 = n0[base + j];
            uint4 a1 = n1[base + j];
            uint4 b0 = d0[base + j];
            uint4 b1 = d1[base + j];
            s_num0 = ext_add(s_num0, ext_mul(e, ext_add(ext_mul(b0, a1), ext_mul(b1, a0))));
            s_den0 = ext_add(s_den0, ext_mul(e, ext_mul(b0, b1)));
            uint4 a0s, a1s, b0s, b1s;
            if (full) {
                uint k = base + p.width + j;
                a0s = ext_add(a0, n0[k]);
                a1s = ext_add(a1, n1[k]);
                b0s = ext_add(b0, d0[k]);
                b1s = ext_add(b1, d1[k]);
            } else {
                a0s = a0;
                a1s = a1;
                b0s = b0;
                b1s = b1;
                b0s.x = kb_add(b0s.x, KB_ONE);
                b1s.x = kb_add(b1s.x, KB_ONE);
            }
            s_numh = ext_add(s_numh, ext_mul(e, ext_add(ext_mul(b0s, a1s), ext_mul(b1s, a0s))));
            s_denh = ext_add(s_denh, ext_mul(e, ext_mul(b0s, b1s)));
            s_eq = ext_add(s_eq, ext_mul(e, eqs));
        }
        acc0 = ext_mul(ext_add(ext_mul(lambda, s_num0), s_den0), eq0);
        acch = ext_mul(ext_add(ext_mul(lambda, s_numh), s_denh), eqs);
        acce = s_eq;
    }
    sh0[tid] = acc0;
    shh[tid] = acch;
    she[tid] = acce;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh0[tid] = ext_add(sh0[tid], sh0[tid + s]);
            shh[tid] = ext_add(shh[tid], shh[tid + s]);
            she[tid] = ext_add(she[tid], she[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partials[3u * tgid] = sh0[0];
        partials[3u * tgid + 1u] = shh[0];
        partials[3u * tgid + 2u] = she[0];
    }
}

// The first layer: numerators are base-field elements, denominators extension.
kernel void logup_round_base(device const uint* n0 [[buffer(0)]],
                             device const uint* n1 [[buffer(1)]],
                             device const uint4* d0 [[buffer(2)]],
                             device const uint4* d1 [[buffer(3)]],
                             device const uint4* eq_int [[buffer(4)]],
                             device const uint4* eq_row [[buffer(5)]],
                             device uint4* partials [[buffer(6)]],
                             constant LogupRoundParams& p [[buffer(7)]],
                             constant uint4& lambda [[buffer(8)]],
                             uint gid [[thread_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 sh0[256];
    threadgroup uint4 shh[256];
    threadgroup uint4 she[256];
    uint4 acc0 = uint4(0u, 0u, 0u, 0u);
    uint4 acch = uint4(0u, 0u, 0u, 0u);
    uint4 acce = uint4(0u, 0u, 0u, 0u);
    if (gid < p.n_pairs) {
        uint base = gid * 2u * p.width;
        uint4 eq0 = eq_row[2u * gid];
        uint4 eqs = ext_add(eq0, eq_row[2u * gid + 1u]);
        uint4 s_num0 = uint4(0u, 0u, 0u, 0u);
        uint4 s_den0 = uint4(0u, 0u, 0u, 0u);
        uint4 s_numh = uint4(0u, 0u, 0u, 0u);
        uint4 s_denh = uint4(0u, 0u, 0u, 0u);
        uint4 s_eq = uint4(0u, 0u, 0u, 0u);
        bool full = gid < p.n_full_pairs;
        for (uint j = 0; j < p.width; j++) {
            uint4 e = eq_int[j];
            uint a0 = n0[base + j];
            uint a1 = n1[base + j];
            uint4 b0 = d0[base + j];
            uint4 b1 = d1[base + j];
            s_num0 = ext_add(
                s_num0,
                ext_mul(e, ext_add(ext_mul_base(b0, a1), ext_mul_base(b1, a0))));
            s_den0 = ext_add(s_den0, ext_mul(e, ext_mul(b0, b1)));
            uint a0s, a1s;
            uint4 b0s, b1s;
            if (full) {
                uint k = base + p.width + j;
                a0s = kb_add(a0, n0[k]);
                a1s = kb_add(a1, n1[k]);
                b0s = ext_add(b0, d0[k]);
                b1s = ext_add(b1, d1[k]);
            } else {
                a0s = a0;
                a1s = a1;
                b0s = b0;
                b1s = b1;
                b0s.x = kb_add(b0s.x, KB_ONE);
                b1s.x = kb_add(b1s.x, KB_ONE);
            }
            s_numh = ext_add(
                s_numh,
                ext_mul(e, ext_add(ext_mul_base(b0s, a1s), ext_mul_base(b1s, a0s))));
            s_denh = ext_add(s_denh, ext_mul(e, ext_mul(b0s, b1s)));
            s_eq = ext_add(s_eq, ext_mul(e, eqs));
        }
        acc0 = ext_mul(ext_add(ext_mul(lambda, s_num0), s_den0), eq0);
        acch = ext_mul(ext_add(ext_mul(lambda, s_numh), s_denh), eqs);
        acce = s_eq;
    }
    sh0[tid] = acc0;
    shh[tid] = acch;
    she[tid] = acce;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh0[tid] = ext_add(sh0[tid], sh0[tid + s]);
            shh[tid] = ext_add(shh[tid], shh[tid + s]);
            she[tid] = ext_add(she[tid], she[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partials[3u * tgid] = sh0[0];
        partials[3u * tgid + 1u] = shh[0];
        partials[3u * tgid + 2u] = she[0];
    }
}
