
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
// One thread per (row pair, interaction) cell, strided over the grid.
//
// Giving a thread a whole row pair and looping it over the interactions puts
// the lanes of a SIMD group `2 * width` elements apart, so each one pulls its
// own cache line. Taking a cell at a time makes neighbouring lanes read
// neighbouring words: the eq weights factor out of the interaction sum
// linearly, so folding them into the cell leaves the round one flat sum for
// one extra multiply.
//
// The grid is capped by the caller and each thread strides over it, which
// keeps the number of partials the host has to add up bounded, and gives the
// scheduler several cells in flight per thread.
//
// The threadgroup reduction is a SIMD shuffle rather than three 256-wide
// threadgroup arrays, freeing 12 KB per threadgroup for occupancy.

inline uint4 simd_ext_sum(uint4 v) {
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        v = ext_add(v, simd_shuffle_down(v, off));
    }
    return v;
}

#define TG_SIMDS 8u

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
                               uint gsize [[threads_per_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint tgid [[threadgroup_position_in_grid]],
                               ushort sgid [[simdgroup_index_in_threadgroup]],
                               ushort slane [[thread_index_in_simdgroup]]) {
    threadgroup uint4 s0[TG_SIMDS];
    threadgroup uint4 sh[TG_SIMDS];
    threadgroup uint4 se[TG_SIMDS];
    if (tid < TG_SIMDS) {
        s0[tid] = uint4(0u);
        sh[tid] = uint4(0u);
        se[tid] = uint4(0u);
    }
    uint4 acc0 = uint4(0u);
    uint4 acch = uint4(0u);
    uint4 acce = uint4(0u);
    uint total = p.n_pairs * p.width;
    for (uint cell = gid; cell < total; cell += gsize) {
        uint q = cell / p.width;
        uint j = cell - q * p.width;
        uint lo = 2u * q * p.width + j;
        uint4 e = eq_int[j];
        uint4 eq0 = eq_row[2u * q];
        uint4 eqs = ext_add(eq0, eq_row[2u * q + 1u]);
        uint4 a0 = n0[lo];
        uint4 a1 = n1[lo];
        uint4 b0 = d0[lo];
        uint4 b1 = d1[lo];
        uint4 a0s, a1s, b0s, b1s;
        if (q < p.n_full_pairs) {
            uint hi = lo + p.width;
            a0s = ext_add(a0, n0[hi]);
            a1s = ext_add(a1, n1[hi]);
            b0s = ext_add(b0, d0[hi]);
            b1s = ext_add(b1, d1[hi]);
        } else {
            a0s = a0;
            a1s = a1;
            b0s = b0;
            b1s = b1;
            b0s.x = kb_add(b0s.x, KB_ONE);
            b1s.x = kb_add(b1s.x, KB_ONE);
        }
        uint4 w0 = ext_mul_kara(e, eq0);
        uint4 wh = ext_mul_kara(e, eqs);
        uint4 num0 = ext_add(ext_mul_kara(b0, a1), ext_mul_kara(b1, a0));
        uint4 den0 = ext_mul_kara(b0, b1);
        uint4 numh = ext_add(ext_mul_kara(b0s, a1s), ext_mul_kara(b1s, a0s));
        uint4 denh = ext_mul_kara(b0s, b1s);
        acc0 = ext_add(acc0, ext_mul_kara(w0, ext_add(ext_mul_kara(lambda, num0), den0)));
        acch = ext_add(acch, ext_mul_kara(wh, ext_add(ext_mul_kara(lambda, numh), denh)));
        acce = ext_add(acce, wh);
    }
    acc0 = simd_ext_sum(acc0);
    acch = simd_ext_sum(acch);
    acce = simd_ext_sum(acce);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slane == 0u) {
        s0[sgid] = acc0;
        sh[sgid] = acch;
        se[sgid] = acce;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint4 t0 = s0[0], th = sh[0], te = se[0];
        for (uint k = 1u; k < TG_SIMDS; k++) {
            t0 = ext_add(t0, s0[k]);
            th = ext_add(th, sh[k]);
            te = ext_add(te, se[k]);
        }
        partials[3u * tgid] = t0;
        partials[3u * tgid + 1u] = th;
        partials[3u * tgid + 2u] = te;
    }
}

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
                                uint gsize [[threads_per_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tgid [[threadgroup_position_in_grid]],
                                ushort sgid [[simdgroup_index_in_threadgroup]],
                                ushort slane [[thread_index_in_simdgroup]]) {
    threadgroup uint4 s0[TG_SIMDS];
    threadgroup uint4 sh[TG_SIMDS];
    threadgroup uint4 se[TG_SIMDS];
    if (tid < TG_SIMDS) {
        s0[tid] = uint4(0u);
        sh[tid] = uint4(0u);
        se[tid] = uint4(0u);
    }
    uint4 acc0 = uint4(0u);
    uint4 acch = uint4(0u);
    uint4 acce = uint4(0u);
    uint total = p.n_pairs * p.width;
    for (uint cell = gid; cell < total; cell += gsize) {
        uint q = cell / p.width;
        uint j = cell - q * p.width;
        uint lo = 2u * q * p.width + j;
        uint4 e = eq_int[j];
        uint4 eq0 = eq_row[2u * q];
        uint4 eqs = ext_add(eq0, eq_row[2u * q + 1u]);
        uint a0 = n0[lo];
        uint a1 = n1[lo];
        uint4 b0 = d0[lo];
        uint4 b1 = d1[lo];
        uint a0s, a1s;
        uint4 b0s, b1s;
        if (q < p.n_full_pairs) {
            uint hi = lo + p.width;
            a0s = kb_add(a0, n0[hi]);
            a1s = kb_add(a1, n1[hi]);
            b0s = ext_add(b0, d0[hi]);
            b1s = ext_add(b1, d1[hi]);
        } else {
            a0s = a0;
            a1s = a1;
            b0s = b0;
            b1s = b1;
            b0s.x = kb_add(b0s.x, KB_ONE);
            b1s.x = kb_add(b1s.x, KB_ONE);
        }
        uint4 w0 = ext_mul_kara(e, eq0);
        uint4 wh = ext_mul_kara(e, eqs);
        uint4 num0 = ext_add(ext_mul_base(b0, a1), ext_mul_base(b1, a0));
        uint4 den0 = ext_mul_kara(b0, b1);
        uint4 numh = ext_add(ext_mul_base(b0s, a1s), ext_mul_base(b1s, a0s));
        uint4 denh = ext_mul_kara(b0s, b1s);
        acc0 = ext_add(acc0, ext_mul_kara(w0, ext_add(ext_mul_kara(lambda, num0), den0)));
        acch = ext_add(acch, ext_mul_kara(wh, ext_add(ext_mul_kara(lambda, numh), denh)));
        acce = ext_add(acce, wh);
    }
    acc0 = simd_ext_sum(acc0);
    acch = simd_ext_sum(acch);
    acce = simd_ext_sum(acce);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slane == 0u) {
        s0[sgid] = acc0;
        sh[sgid] = acch;
        se[sgid] = acce;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint4 t0 = s0[0], th = sh[0], te = se[0];
        for (uint k = 1u; k < TG_SIMDS; k++) {
            t0 = ext_add(t0, s0[k]);
            th = ext_add(th, sh[k]);
            te = ext_add(te, se[k]);
        }
        partials[3u * tgid] = t0;
        partials[3u * tgid + 1u] = th;
        partials[3u * tgid + 2u] = te;
    }
}

// ---------------------------------------------------------------------------
// Fixing a padded MLE's last variable: `out = alpha * (y - x) + x`, where the
// partner row is the padding when the row count is odd. This reads the same
// data the reduction does, so leaving it on the CPU caps a ported round at
// about four times.
// ---------------------------------------------------------------------------

struct FoldParams {
    uint out_rows;
    uint rows;
    uint width;
};

kernel void fold_padded_base(device const uint* in [[buffer(0)]],
                             device const uint* padding [[buffer(1)]],
                             device uint4* out [[buffer(2)]],
                             constant FoldParams& p [[buffer(3)]],
                             constant uint4& alpha [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    uint total = p.out_rows * p.width;
    if (gid >= total) {
        return;
    }
    uint i = gid / p.width;
    uint j = gid - i * p.width;
    uint x = in[(2u * i) * p.width + j];
    uint y = (2u * i + 1u < p.rows) ? in[(2u * i + 1u) * p.width + j] : padding[j];
    uint4 r = ext_mul_base(alpha, kb_sub(y, x));
    r.x = kb_add(r.x, x);
    out[gid] = r;
}

kernel void fold_padded_ext(device const uint4* in [[buffer(0)]],
                            device const uint4* padding [[buffer(1)]],
                            device uint4* out [[buffer(2)]],
                            constant FoldParams& p [[buffer(3)]],
                            constant uint4& alpha [[buffer(4)]],
                            uint gid [[thread_position_in_grid]]) {
    uint total = p.out_rows * p.width;
    if (gid >= total) {
        return;
    }
    uint i = gid / p.width;
    uint j = gid - i * p.width;
    uint4 x = in[(2u * i) * p.width + j];
    uint4 y = (2u * i + 1u < p.rows) ? in[(2u * i + 1u) * p.width + j] : padding[j];
    out[gid] = ext_add(ext_mul_kara(alpha, ext_sub(y, x)), x);
}

// ---------------------------------------------------------------------------
// Building the next circuit layer.
//
// Each layer of the logup circuit adds fractions pairwise, so a row pair of
// (numerator, denominator) becomes one row of
//
//   n' = d_b * n_a + d_a * n_b,   d' = d_a * d_b
//
// once for the even row of the pair and once for the odd one. There is no AIR
// and no reduction here, just one output cell per thread. When the row count is
// odd the last pair has no odd row, and that half of the output keeps the
// identity fraction 0/1 the CPU fills it with.
// ---------------------------------------------------------------------------

struct TransitionParams {
    uint out_rows;
    uint rows;
    uint width;
};

kernel void gkr_transition_ext(device const uint4* n0 [[buffer(0)]],
                               device const uint4* d0 [[buffer(1)]],
                               device const uint4* n1 [[buffer(2)]],
                               device const uint4* d1 [[buffer(3)]],
                               device uint4* out_n0 [[buffer(4)]],
                               device uint4* out_d0 [[buffer(5)]],
                               device uint4* out_n1 [[buffer(6)]],
                               device uint4* out_d1 [[buffer(7)]],
                               constant TransitionParams& p [[buffer(8)]],
                               uint gid [[thread_position_in_grid]]) {
    uint total = p.out_rows * p.width;
    if (gid >= total) {
        return;
    }
    uint q = gid / p.width;
    uint j = gid - q * p.width;
    uint lo = (2u * q) * p.width + j;
    uint4 a0 = n0[lo], b0 = d0[lo], a1 = n1[lo], b1 = d1[lo];
    out_n0[gid] = ext_add(ext_mul_kara(b1, a0), ext_mul_kara(b0, a1));
    out_d0[gid] = ext_mul_kara(b0, b1);
    if (2u * q + 1u < p.rows) {
        uint hi = lo + p.width;
        uint4 c0 = n0[hi], e0 = d0[hi], c1 = n1[hi], e1 = d1[hi];
        out_n1[gid] = ext_add(ext_mul_kara(e1, c0), ext_mul_kara(e0, c1));
        out_d1[gid] = ext_mul_kara(e0, e1);
    } else {
        out_n1[gid] = uint4(0u);
        out_d1[gid] = uint4(KB_ONE, 0u, 0u, 0u);
    }
}

// The first transition: numerators are still base-field elements.
kernel void gkr_transition_base(device const uint* n0 [[buffer(0)]],
                                device const uint4* d0 [[buffer(1)]],
                                device const uint* n1 [[buffer(2)]],
                                device const uint4* d1 [[buffer(3)]],
                                device uint4* out_n0 [[buffer(4)]],
                                device uint4* out_d0 [[buffer(5)]],
                                device uint4* out_n1 [[buffer(6)]],
                                device uint4* out_d1 [[buffer(7)]],
                                constant TransitionParams& p [[buffer(8)]],
                                uint gid [[thread_position_in_grid]]) {
    uint total = p.out_rows * p.width;
    if (gid >= total) {
        return;
    }
    uint q = gid / p.width;
    uint j = gid - q * p.width;
    uint lo = (2u * q) * p.width + j;
    uint a0 = n0[lo], a1 = n1[lo];
    uint4 b0 = d0[lo], b1 = d1[lo];
    out_n0[gid] = ext_add(ext_mul_base(b1, a0), ext_mul_base(b0, a1));
    out_d0[gid] = ext_mul_kara(b0, b1);
    if (2u * q + 1u < p.rows) {
        uint hi = lo + p.width;
        uint c0 = n0[hi], c1 = n1[hi];
        uint4 e0 = d0[hi], e1 = d1[hi];
        out_n1[gid] = ext_add(ext_mul_base(e1, c0), ext_mul_base(e0, c1));
        out_d1[gid] = ext_mul_kara(e0, e1);
    } else {
        out_n1[gid] = uint4(0u);
        out_d1[gid] = uint4(KB_ONE, 0u, 0u, 0u);
    }
}
