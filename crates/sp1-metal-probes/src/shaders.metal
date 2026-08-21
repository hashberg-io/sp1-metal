#include <metal_stdlib>
using namespace metal;

// KoalaBear base field, Montgomery form with R = 2^32, mirroring p3-koala-bear
// (p3-koala-bear-0.4.3-succinct/src/koala_bear.rs). All arithmetic stays in
// 32-bit integer lanes: Apple GPUs emulate 64-bit integer ops.
constant uint KB_P = 0x7f000001u;
// P^-1 mod 2^32 (p3's borrow-free convention: MU = P^-1, not -P^-1).
constant uint KB_MU = 0x81000001u;
// monty(3): the binomial-extension constant W for x^4 - 3.
constant uint KB_W_MONTY = 0x05fffffau;

// Montgomery reduction of the 64-bit value (hi, lo), valid for hi < P.
// Since t = lo * P^-1 mod 2^32, the low limb of t*P equals lo exactly, so the
// subtraction only involves the high limbs.
inline uint kb_redc(uint hi, uint lo) {
    uint t = lo * KB_MU;
    uint u = mulhi(t, KB_P);
    uint r = hi - u;
    return (hi < u) ? (r + KB_P) : r;
}

inline uint kb_mul(uint a, uint b) {
    return kb_redc(mulhi(a, b), a * b);
}

inline uint kb_add(uint a, uint b) {
    uint s = a + b;
    return (s >= KB_P) ? (s - KB_P) : s;
}

inline uint kb_sub(uint a, uint b) {
    return (a >= b) ? (a - b) : (a + (KB_P - b));
}

inline uint kb_neg(uint a) {
    return (a == 0u) ? 0u : (KB_P - a);
}

inline uint kb_double(uint a) {
    return kb_add(a, a);
}

inline uint kb_cube(uint x) {
    return kb_mul(kb_mul(x, x), x);
}

// Degree-4 binomial extension over x^4 = 3; uint4 lanes are the coefficients
// (c0, c1, c2, c3) in Montgomery form, matching p3's BinomialExtensionField.
inline uint4 ext_add(uint4 a, uint4 b) {
    return uint4(kb_add(a.x, b.x), kb_add(a.y, b.y), kb_add(a.z, b.z), kb_add(a.w, b.w));
}

inline uint4 ext_sub(uint4 a, uint4 b) {
    return uint4(kb_sub(a.x, b.x), kb_sub(a.y, b.y), kb_sub(a.z, b.z), kb_sub(a.w, b.w));
}

inline uint4 ext_mul_base(uint4 a, uint b) {
    return uint4(kb_mul(a.x, b), kb_mul(a.y, b), kb_mul(a.z, b), kb_mul(a.w, b));
}

inline uint4 ext_mul(uint4 a, uint4 b) {
    uint c0 = kb_add(
        kb_mul(a.x, b.x),
        kb_mul(KB_W_MONTY, kb_add(kb_add(kb_mul(a.y, b.w), kb_mul(a.z, b.z)), kb_mul(a.w, b.y))));
    uint c1 = kb_add(
        kb_add(kb_mul(a.x, b.y), kb_mul(a.y, b.x)),
        kb_mul(KB_W_MONTY, kb_add(kb_mul(a.z, b.w), kb_mul(a.w, b.z))));
    uint c2 = kb_add(
        kb_add(kb_mul(a.x, b.z), kb_mul(a.y, b.y)),
        kb_add(kb_mul(a.z, b.x), kb_mul(KB_W_MONTY, kb_mul(a.w, b.w))));
    uint c3 = kb_add(
        kb_add(kb_mul(a.x, b.w), kb_mul(a.y, b.z)),
        kb_add(kb_mul(a.z, b.y), kb_mul(a.w, b.x)));
    return uint4(c0, c1, c2, c3);
}

// ---------------------------------------------------------------------------
// Probe: streaming bandwidth (STREAM triad, integer lanes).
// ---------------------------------------------------------------------------

kernel void triad(device const uint4* a [[buffer(0)]],
                  device const uint4* b [[buffer(1)]],
                  device uint4* c [[buffer(2)]],
                  constant uint& scale [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    c[gid] = a[gid] + scale * b[gid];
}

// ---------------------------------------------------------------------------
// Probes: register-resident integer op throughput. Eight independent
// dependency chains per thread hide instruction latency; the xor keeps every
// chain data-dependent so nothing folds at compile time. Each iteration does
// one op-of-interest plus one xor per chain.
// ---------------------------------------------------------------------------

constant uint TP_MULC[8] = {
    0x9e3779b9u, 0x85ebca6bu, 0xc2b2ae35u, 0x27d4eb2fu,
    0x165667b1u, 0xd3a2646du, 0xfd7046c5u, 0xb55a4f09u,
};

kernel void tp_add(device uint* out [[buffer(0)]],
                   constant uint& iters [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    uint a[4];
    uint b[4];
    for (uint j = 0; j < 4; j++) {
        a[j] = gid * TP_MULC[j] + 1u;
        b[j] = gid ^ TP_MULC[j + 4];
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 4; j++) {
            a[j] += b[j];
            b[j] ^= a[j];
        }
    }
    out[gid] = a[0] ^ a[1] ^ a[2] ^ a[3] ^ b[0] ^ b[1] ^ b[2] ^ b[3];
}

kernel void tp_mul_lo(device uint* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x[8];
    for (uint j = 0; j < 8; j++) {
        x[j] = gid * TP_MULC[j] + 1u;
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 8; j++) {
            x[j] = (x[j] * TP_MULC[j]) ^ gid;
        }
    }
    uint acc = 0u;
    for (uint j = 0; j < 8; j++) {
        acc ^= x[j];
    }
    out[gid] = acc;
}

kernel void tp_mul_hi(device uint* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x[8];
    for (uint j = 0; j < 8; j++) {
        x[j] = gid * TP_MULC[j] + 1u;
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 8; j++) {
            x[j] = mulhi(x[j], TP_MULC[j]) ^ gid;
        }
    }
    uint acc = 0u;
    for (uint j = 0; j < 8; j++) {
        acc ^= x[j];
    }
    out[gid] = acc;
}

kernel void tp_kb_mul(device uint* out [[buffer(0)]],
                      constant uint& iters [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint x[8];
    for (uint j = 0; j < 8; j++) {
        x[j] = (gid * TP_MULC[j]) % (KB_P - 1u) + 1u;
    }
    for (uint i = 0; i < iters; i++) {
        for (uint j = 0; j < 8; j++) {
            x[j] = kb_mul(x[j], x[j] | 1u);
        }
    }
    uint acc = 0u;
    for (uint j = 0; j < 8; j++) {
        acc ^= x[j];
    }
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// Probes: streamed elementwise field multiplication (bandwidth-bound shape).
// ---------------------------------------------------------------------------

kernel void kb_mul_elems(device const uint* a [[buffer(0)]],
                         device const uint* b [[buffer(1)]],
                         device uint* c [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    c[gid] = kb_mul(a[gid], b[gid]);
}

kernel void ext_mul_elems(device const uint4* a [[buffer(0)]],
                          device const uint4* b [[buffer(1)]],
                          device uint4* c [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    c[gid] = ext_mul(a[gid], b[gid]);
}

// ---------------------------------------------------------------------------
// Probes: BaseFold fold and sumcheck fix-last-variable, mirroring
// slop-multilinear's fold_mle (out[i] = in[2i] + beta * in[2i+1]) and
// mle_fix_last_variable (out[i] = x + alpha * (y - x) on interleaved pairs).
// ---------------------------------------------------------------------------

kernel void fold_base(device const uint* in [[buffer(0)]],
                      device uint* out [[buffer(1)]],
                      constant uint& beta [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    out[gid] = kb_add(in[2u * gid], kb_mul(beta, in[2u * gid + 1u]));
}

kernel void fold_ext(device const uint4* in [[buffer(0)]],
                     device uint4* out [[buffer(1)]],
                     constant uint4& beta [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = ext_add(in[2u * gid], ext_mul(beta, in[2u * gid + 1u]));
}

kernel void flv_base(device const uint* in [[buffer(0)]],
                     device uint4* out [[buffer(1)]],
                     constant uint4& alpha [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint x = in[2u * gid];
    uint y = in[2u * gid + 1u];
    uint4 r = ext_mul_base(alpha, kb_sub(y, x));
    r.x = kb_add(r.x, x);
    out[gid] = r;
}

kernel void flv_ext(device const uint4* in [[buffer(0)]],
                    device uint4* out [[buffer(1)]],
                    constant uint4& alpha [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    uint4 x = in[2u * gid];
    uint4 y = in[2u * gid + 1u];
    out[gid] = ext_add(ext_mul(alpha, ext_sub(y, x)), x);
}

// ---------------------------------------------------------------------------
// Probe: Poseidon2 width-16 over KoalaBear, D=3, 8 external + 20 internal
// rounds, mirroring p3-poseidon2 (external M4 light MDS + block sums) and
// p3-koala-bear's DiffusionMatrixKoalaBear (shift-based internal diagonal).
// Round constants arrive in Montgomery form via buffer(2): 8*16 external
// (first 4 rounds then last 4), then 20 internal.
// ---------------------------------------------------------------------------

constant uint P2_DIAG_SHIFTS[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15};

inline void p2_mat4(thread uint* x) {
    uint t01 = kb_add(x[0], x[1]);
    uint t23 = kb_add(x[2], x[3]);
    uint t0123 = kb_add(t01, t23);
    uint t01123 = kb_add(t0123, x[1]);
    uint t01233 = kb_add(t0123, x[3]);
    x[3] = kb_add(t01233, kb_double(x[0]));
    x[1] = kb_add(t01123, kb_double(x[2]));
    x[0] = kb_add(t01123, t01);
    x[2] = kb_add(t01233, t23);
}

inline void p2_external(thread uint* s) {
    for (uint i = 0; i < 16; i += 4) {
        p2_mat4(s + i);
    }
    uint sums[4];
    for (uint j = 0; j < 4; j++) {
        sums[j] = kb_add(kb_add(s[j], s[4 + j]), kb_add(s[8 + j], s[12 + j]));
    }
    for (uint i = 0; i < 16; i++) {
        s[i] = kb_add(s[i], sums[i & 3u]);
    }
}

inline void p2_full_round(thread uint* s, constant uint* rc) {
    for (uint i = 0; i < 16; i++) {
        s[i] = kb_cube(kb_add(s[i], rc[i]));
    }
    p2_external(s);
}

// The internal diffusion mirrors DiffusionMatrixKoalaBear::permute_mut, which
// accumulates raw Montgomery values into a 64-bit sum before one reduction;
// here the 64-bit accumulator is two explicit 32-bit limbs (hi stays tiny, so
// kb_redc's hi < P precondition holds).
inline void p2_internal(thread uint* s, uint rc) {
    s[0] = kb_cube(kb_add(s[0], rc));
    uint plo = 0u;
    uint phi = 0u;
    for (uint i = 1; i < 16; i++) {
        plo += s[i];
        phi += (plo < s[i]) ? 1u : 0u;
    }
    uint flo = plo + s[0];
    uint fhi = phi + ((flo < s[0]) ? 1u : 0u);
    uint n0 = kb_neg(s[0]);
    uint slo = plo + n0;
    uint shi = phi + ((slo < n0) ? 1u : 0u);
    uint new0 = kb_redc(shi, slo);
    for (uint i = 1; i < 16; i++) {
        uint v = s[i];
        uint sh = P2_DIAG_SHIFTS[i - 1];
        uint addlo = v << sh;
        uint addhi = (sh == 0u) ? 0u : (v >> (32u - sh));
        uint l = flo + addlo;
        uint h = fhi + addhi + ((l < addlo) ? 1u : 0u);
        s[i] = kb_redc(h, l);
    }
    s[0] = new0;
}

kernel void poseidon2_16(device const uint* in [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant uint* rc [[buffer(2)]],
                         constant uint& n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) {
        return;
    }
    uint s[16];
    for (uint i = 0; i < 16; i++) {
        s[i] = in[gid * 16u + i];
    }
    p2_external(s);
    for (uint r = 0; r < 4; r++) {
        p2_full_round(s, rc + r * 16u);
    }
    for (uint r = 0; r < 20; r++) {
        p2_internal(s, rc[128u + r]);
    }
    for (uint r = 4; r < 8; r++) {
        p2_full_round(s, rc + r * 16u);
    }
    for (uint i = 0; i < 16; i++) {
        out[gid * 16u + i] = s[i];
    }
}

// ---------------------------------------------------------------------------
// Probe: synthetic zerocheck round. One thread per row of a column-major
// base-field trace evaluates T degree-3 monomials, combines them with powers
// of alpha (extension field), multiplies by eq(row), and threadgroup-reduces
// to one partial extension-field sum per group. This prices the
// constraint-evaluation shape without a real chip's constraint transpiler.
// ---------------------------------------------------------------------------

struct ZcParams {
    uint n;
    uint num_cols;
    uint num_terms;
};

kernel void zc_round(device const uint* trace [[buffer(0)]],
                     device const uint4* eq [[buffer(1)]],
                     device const uint4* alpha_pows [[buffer(2)]],
                     device const uint4* term_cols [[buffer(3)]],
                     constant ZcParams& p [[buffer(4)]],
                     device uint4* partials [[buffer(5)]],
                     uint gid [[thread_position_in_grid]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 shared[256];
    uint4 acc = uint4(0u);
    if (gid < p.n) {
        for (uint t = 0; t < p.num_terms; t++) {
            uint4 cols = term_cols[t];
            uint m = kb_mul(
                kb_mul(trace[cols.x * p.n + gid], trace[cols.y * p.n + gid]),
                trace[cols.z * p.n + gid]);
            acc = ext_add(acc, ext_mul_base(alpha_pows[t], m));
        }
        acc = ext_mul(acc, eq[gid]);
    }
    shared[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared[tid] = ext_add(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        partials[tgid] = shared[0];
    }
}

// A kernel that does almost nothing: used to price the fixed cost of a
// dispatch, separately from the fixed cost of a command-buffer round trip.
kernel void noop(device uint* out [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid == 0) {
        out[0] = out[0] + 1u;
    }
}

// One radix-2 decimation-in-time butterfly stage, the inner loop of an NTT and
// so of the Reed-Solomon encode inside a BaseFold commitment. Each thread owns
// one butterfly: the pair it touches is `m` apart, and `tw` holds the `m`
// twiddles for this stage.
kernel void ntt_stage(device uint* a [[buffer(0)]],
                      device const uint* tw [[buffer(1)]],
                      constant uint& log_m [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    uint m = 1u << log_m;
    uint j = gid & (m - 1u);
    uint block = gid >> log_m;
    uint i = (block << (log_m + 1u)) + j;
    uint u = a[i];
    uint t = kb_mul(a[i + m], tw[j]);
    a[i] = kb_add(u, t);
    a[i + m] = kb_sub(u, t);
}

// One sumcheck round of a logup GKR layer, in the shape sp1-hypercube's
// `sum_as_poly_in_last_t_variables` computes it: over each pair of consecutive
// entries, the round polynomial's evaluation at the first point uses the even
// entry, and its evaluation at the second uses the pair's sum. Five extension
// accumulators come out — numerator and denominator at each of the two points,
// plus the eq sum — and the caller combines them with lambda.
kernel void gkr_round(device const uint4* num_0 [[buffer(0)]],
                      device const uint4* num_1 [[buffer(1)]],
                      device const uint4* den_0 [[buffer(2)]],
                      device const uint4* den_1 [[buffer(3)]],
                      device const uint4* eq [[buffer(4)]],
                      constant uint& pairs [[buffer(5)]],
                      device uint4* partials [[buffer(6)]],
                      uint gid [[thread_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]]) {
    threadgroup uint4 shared[256];
    uint4 acc[5];
    for (uint k = 0u; k < 5u; k++) {
        acc[k] = uint4(0u);
    }
    if (gid < pairs) {
        uint i = 2u * gid;
        uint4 n0a = num_0[i], n0b = num_0[i + 1u];
        uint4 n1a = num_1[i], n1b = num_1[i + 1u];
        uint4 d0a = den_0[i], d0b = den_0[i + 1u];
        uint4 d1a = den_1[i], d1b = den_1[i + 1u];
        uint4 qa = eq[i], qb = eq[i + 1u];
        uint4 numer = ext_add(ext_mul(d0a, n1a), ext_mul(d1a, n0a));
        acc[0] = ext_mul(qa, numer);
        acc[1] = ext_mul(qa, ext_mul(d0a, d1a));
        uint4 n0h = ext_add(n0a, n0b);
        uint4 n1h = ext_add(n1a, n1b);
        uint4 d0h = ext_add(d0a, d0b);
        uint4 d1h = ext_add(d1a, d1b);
        uint4 qh = ext_add(qa, qb);
        uint4 numer_h = ext_add(ext_mul(d0h, n1h), ext_mul(d1h, n0h));
        acc[2] = ext_mul(qh, numer_h);
        acc[3] = ext_mul(qh, ext_mul(d0h, d1h));
        acc[4] = qh;
    }
    for (uint k = 0u; k < 5u; k++) {
        shared[tid] = acc[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                shared[tid] = ext_add(shared[tid], shared[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) {
            partials[tgid * 5u + k] = shared[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
