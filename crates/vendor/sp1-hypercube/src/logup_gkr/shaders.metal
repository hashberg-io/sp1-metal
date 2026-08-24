// SPDX-License-Identifier: MIT OR Apache-2.0
// Added to sp1-hypercube 6.4.0 by the sp1-metal project; see CHANGES.md.

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

// ---------------------------------------------------------------------------
// The GKR circuit's first layer: one numerator and one denominator per row per
// interaction.
//
// An interaction's denominator is `alpha + beta_0 * kind + sum_j beta_j * v_j`,
// where each `v_j` is an affine form over the row's preprocessed and main
// columns, and its numerator is one more such form, negated for a receive.
// Those forms are fixed per chip, so rather than generating a kernel for each
// -- which is what the zerocheck does, and what costs it a compile per chip --
// they are uploaded as a table this one kernel walks. Every chip in the proof
// then shares a single pipeline.
//
// The tables are tiny beside the trace: tens of interactions of tens of terms,
// against a height in the millions.
struct FirstLayerParams {
    uint rows;
    uint num_interactions;
    uint main_width;
    uint prep_width;
};

// One affine form over a row, evaluated in the base field. A column index with
// the top bit set names the preprocessed row rather than the main one.
inline uint fl_apply(device const uint* main_row,
                     device const uint* prep_row,
                     device const uint* term_col,
                     device const uint* term_w,
                     uint from,
                     uint to,
                     uint constant_term) {
    uint acc = constant_term;
    for (uint t = from; t < to; ++t) {
        uint c = term_col[t];
        uint x = (c & 0x80000000u) ? prep_row[c & 0x7fffffffu] : main_row[c];
        acc = kb_add(acc, kb_mul(term_w[t], x));
    }
    return acc;
}

kernel void first_layer(device const uint* main [[buffer(0)]],
                        device const uint* prep [[buffer(1)]],
                        device const uint4* betas [[buffer(2)]],
                        device const uint* val_off [[buffer(3)]],
                        device const uint* mult_val [[buffer(4)]],
                        device const uint* term_off [[buffer(5)]],
                        device const uint* val_const [[buffer(6)]],
                        device const uint* term_col [[buffer(7)]],
                        device const uint* term_w [[buffer(8)]],
                        device const uint* arg_index [[buffer(9)]],
                        device const uint* is_send [[buffer(10)]],
                        device uint* numer_out [[buffer(11)]],
                        device uint4* denom_out [[buffer(12)]],
                        constant FirstLayerParams& p [[buffer(13)]],
                        constant uint4& alpha [[buffer(14)]],
                        uint gid [[thread_position_in_grid]],
                        uint grid [[threads_per_grid]]) {
    uint total = p.rows * p.num_interactions;
    for (uint idx = gid; idx < total; idx += grid) {
        uint row = idx / p.num_interactions;
        uint i = idx - row * p.num_interactions;
        device const uint* main_row = main + row * p.main_width;
        device const uint* prep_row = prep + row * p.prep_width;

        uint4 denom = ext_add(alpha, ext_mul_base(betas[0], arg_index[i]));
        // The multiplicity is stored just past this interaction's values, so
        // it -- not `val_off[i + 1]` -- is where the values stop.
        uint vfrom = val_off[i];
        uint vto = mult_val[i];
        for (uint v = vfrom; v < vto; ++v) {
            uint a = fl_apply(main_row, prep_row, term_col, term_w,
                              term_off[v], term_off[v + 1u], val_const[v]);
            denom = ext_add(denom, ext_mul_base(betas[v - vfrom + 1u], a));
        }

        uint mv = mult_val[i];
        uint mult = fl_apply(main_row, prep_row, term_col, term_w,
                             term_off[mv], term_off[mv + 1u], val_const[mv]);
        numer_out[idx] = is_send[i] != 0u ? mult : kb_neg(mult);
        denom_out[idx] = denom;
    }
}

// ---------------------------------------------------------------------------
// The same round reduction, but over every small chip of a layer at once.
//
// A shard's layer holds a long tail of chips too small to be worth a dispatch
// each: the arithmetic is a few hundred cells and the launch costs more than it
// saves, which a sweep of the per-chip threshold confirmed -- lowering it moved
// work to the GPU and bought no time at all. Batching is the way to reach them,
// and it is available because a layer's chips all add into the same three sums,
// so nothing has to be kept separate.
//
// Their four arrays are copied into one staging buffer beside each other, which
// is the one place this backend copies on purpose: a chip below the threshold
// is at most a few hundred kilobytes, and the alternative is a dispatch for
// each.
struct ChipDesc {
    uint cells_before;
    uint data_off;
    uint eqi_off;
    uint width;
    uint n_pairs;
    uint n_full;
};

struct BatchParams {
    uint n_chips;
    uint total_cells;
};

#define LOGUP_ROUND_BATCH(NAME, NUM_T, NUM_TO_EXT)                                        \
kernel void NAME(device const NUM_T* n0 [[buffer(0)]],                                    \
                 device const NUM_T* n1 [[buffer(1)]],                                    \
                 device const uint4* d0 [[buffer(2)]],                                    \
                 device const uint4* d1 [[buffer(3)]],                                    \
                 device const uint4* eq_int [[buffer(4)]],                                \
                 device const uint4* eq_row [[buffer(5)]],                                \
                 device uint4* partials [[buffer(6)]],                                    \
                 constant BatchParams& p [[buffer(7)]],                                   \
                 constant uint4& lambda [[buffer(8)]],                                    \
                 device const ChipDesc* desc [[buffer(9)]],                               \
                 uint gid [[thread_position_in_grid]],                                    \
                 uint gsize [[threads_per_grid]],                                         \
                 uint tid [[thread_position_in_threadgroup]],                             \
                 uint tgid [[threadgroup_position_in_grid]],                              \
                 ushort sgid [[simdgroup_index_in_threadgroup]],                          \
                 ushort slane [[thread_index_in_simdgroup]]) {                            \
    threadgroup uint4 s0[TG_SIMDS];                                                       \
    threadgroup uint4 sh[TG_SIMDS];                                                       \
    threadgroup uint4 se[TG_SIMDS];                                                       \
    if (tid < TG_SIMDS) { s0[tid] = uint4(0u); sh[tid] = uint4(0u); se[tid] = uint4(0u); } \
    uint4 acc0 = uint4(0u);                                                               \
    uint4 acch = uint4(0u);                                                               \
    uint4 acce = uint4(0u);                                                               \
    for (uint cell = gid; cell < p.total_cells; cell += gsize) {                          \
        uint lo_c = 0u;                                                                   \
        uint hi_c = p.n_chips - 1u;                                                       \
        while (lo_c < hi_c) {                                                             \
            uint mid = (lo_c + hi_c + 1u) >> 1;                                           \
            if (desc[mid].cells_before <= cell) { lo_c = mid; } else { hi_c = mid - 1u; } \
        }                                                                                 \
        ChipDesc c = desc[lo_c];                                                          \
        uint local = cell - c.cells_before;                                               \
        uint q = local / c.width;                                                         \
        uint j = local - q * c.width;                                                     \
        uint lo = c.data_off + 2u * q * c.width + j;                                      \
        uint4 e = eq_int[c.eqi_off + j];                                                  \
        uint4 eq0 = eq_row[2u * q];                                                       \
        uint4 eqs = ext_add(eq0, eq_row[2u * q + 1u]);                                    \
        uint4 a0 = NUM_TO_EXT(n0[lo]);                                                    \
        uint4 a1 = NUM_TO_EXT(n1[lo]);                                                    \
        uint4 b0 = d0[lo];                                                                \
        uint4 b1 = d1[lo];                                                                \
        uint4 a0s, a1s, b0s, b1s;                                                         \
        if (q < c.n_full) {                                                               \
            uint hi = lo + c.width;                                                       \
            a0s = ext_add(a0, NUM_TO_EXT(n0[hi]));                                        \
            a1s = ext_add(a1, NUM_TO_EXT(n1[hi]));                                        \
            b0s = ext_add(b0, d0[hi]);                                                    \
            b1s = ext_add(b1, d1[hi]);                                                    \
        } else {                                                                          \
            a0s = a0; a1s = a1; b0s = b0; b1s = b1;                                       \
            b0s.x = kb_add(b0s.x, KB_ONE);                                                \
            b1s.x = kb_add(b1s.x, KB_ONE);                                                \
        }                                                                                 \
        uint4 w0 = ext_mul_kara(e, eq0);                                                  \
        uint4 wh = ext_mul_kara(e, eqs);                                                  \
        uint4 num0 = ext_add(ext_mul_kara(b0, a1), ext_mul_kara(b1, a0));                 \
        uint4 den0 = ext_mul_kara(b0, b1);                                                \
        uint4 numh = ext_add(ext_mul_kara(b0s, a1s), ext_mul_kara(b1s, a0s));             \
        uint4 denh = ext_mul_kara(b0s, b1s);                                              \
        acc0 = ext_add(acc0, ext_mul_kara(w0, ext_add(ext_mul_kara(lambda, num0), den0))); \
        acch = ext_add(acch, ext_mul_kara(wh, ext_add(ext_mul_kara(lambda, numh), denh))); \
        acce = ext_add(acce, wh);                                                         \
    }                                                                                     \
    acc0 = simd_ext_sum(acc0);                                                            \
    acch = simd_ext_sum(acch);                                                            \
    acce = simd_ext_sum(acce);                                                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
    if (slane == 0u) { s0[sgid] = acc0; sh[sgid] = acch; se[sgid] = acce; }               \
    threadgroup_barrier(mem_flags::mem_threadgroup);                                      \
    if (tid == 0u) {                                                                      \
        uint4 t0 = s0[0], th = sh[0], te = se[0];                                         \
        for (uint k = 1u; k < TG_SIMDS; k++) {                                            \
            t0 = ext_add(t0, s0[k]);                                                      \
            th = ext_add(th, sh[k]);                                                      \
            te = ext_add(te, se[k]);                                                      \
        }                                                                                 \
        partials[3u * tgid] = t0;                                                         \
        partials[3u * tgid + 1u] = th;                                                    \
        partials[3u * tgid + 2u] = te;                                                    \
    }                                                                                     \
}

inline uint4 batch_ext_id(uint4 x) { return x; }
inline uint4 batch_base_lift(uint x) { return uint4(x, 0u, 0u, 0u); }

LOGUP_ROUND_BATCH(logup_round_batch_ext, uint4, batch_ext_id)
LOGUP_ROUND_BATCH(logup_round_batch_base, uint, batch_base_lift)
